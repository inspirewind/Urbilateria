use std::path::PathBuf;
use urbilateria::models::deepseek_v4::{
    prompt::{self, Message, PromptOptions, Role, ThinkingMode},
    runtime::DeepseekRuntimeModel,
    schema,
};
use urbilateria::runtime::RuntimeLoadOptions;
use urbilateria::storage::{inspect_weight_matrix, load_weight_matrix, TensorIndex};
use urbilateria::tokenizer::ByteBpeTokenizer;
use urbilateria::{ModelConfig, ModelFamily};

fn checkpoint_dir() -> PathBuf {
    std::env::var_os("URB_DEEPSEEK_V4_DIR")
        .map(PathBuf::from)
        .expect("set URB_DEEPSEEK_V4_DIR to run the real-checkpoint tests")
}

#[test]
#[ignore = "requires the 156 GiB DeepSeek-V4-Flash-0731 checkpoint"]
fn headers_match_the_complete_release_schema() {
    let model_dir = checkpoint_dir();
    let config = ModelConfig::load(&model_dir).unwrap();
    assert_eq!(config.family(), ModelFamily::DeepseekV4);
    let ModelConfig::DeepseekV4(config) = config else {
        unreachable!();
    };
    let index = TensorIndex::open(&model_dir).unwrap();
    let requirements = schema::inspect_requirements(&config, &index, 1, 0).unwrap();
    assert_eq!(requirements.required_tensor_count, 72_317);
    assert_eq!(requirements.checkpoint_tensor_count, 72_317);
    assert_eq!(requirements.unexpected_tensor_count, 0);
    assert_eq!(requirements.base_expert_count, 43 * 256);
    assert_eq!(requirements.dspark_stage_count, 3);
}

#[test]
#[ignore = "requires bounded payload reads from DeepSeek-V4-Flash-0731"]
fn real_mxfp8_and_mxfp4_prefixes_decode_exactly() {
    let index = TensorIndex::open(&checkpoint_dir()).unwrap();

    let fp8_name = "layers.0.attn.wq_a.weight";
    let fp8_layout = inspect_weight_matrix(&index, fp8_name, 1024, 4096).unwrap();
    let fp8 = load_weight_matrix(&index, fp8_name, 1024, 4096, fp8_layout.resident_bytes).unwrap();
    assert_eq!(fp8.row(0).unwrap()[0], -0.0078125);

    let fp4_name = "layers.0.ffn.experts.0.w1.weight";
    let fp4_layout = inspect_weight_matrix(&index, fp4_name, 2048, 4096).unwrap();
    let fp4 = load_weight_matrix(&index, fp4_name, 2048, 4096, fp4_layout.resident_bytes).unwrap();
    assert_eq!(&fp4.row(0).unwrap()[..2], &[-0.015625, 0.0]);
}

#[test]
#[ignore = "validates all headers and loads the bounded DeepSeek-V4 runtime root"]
fn real_checkpoint_constructs_the_base_runtime() {
    let model_dir = checkpoint_dir();
    let requirements = DeepseekRuntimeModel::inspect_requirements(&model_dir, 1, 0).unwrap();
    let model = DeepseekRuntimeModel::load(
        &model_dir,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.resident_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.kv_cache_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.maximum_expert_bytes,
            context_limit: 1,
        },
    )
    .unwrap();
    assert_eq!(model.config().num_hidden_layers, 43);
    assert_eq!(model.new_state().unwrap().position(), 0);
}

#[test]
#[ignore = "executes one complete 43-layer base-model token on the release checkpoint"]
fn real_checkpoint_executes_one_complete_base_token() {
    let model_dir = checkpoint_dir();
    let requirements = DeepseekRuntimeModel::inspect_requirements(&model_dir, 1, 0).unwrap();
    let model = DeepseekRuntimeModel::load(
        &model_dir,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.resident_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.kv_cache_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.maximum_expert_bytes,
            context_limit: 1,
        },
    )
    .unwrap();
    let mut state = model.new_state().unwrap();
    let step = model.forward_token(0, &mut state).unwrap();

    assert_eq!(step.logits.len(), model.config().vocab_size);
    assert!(step.logits.iter().all(|value| value.is_finite()));
    assert_eq!(step.routes_by_layer.len(), model.config().num_hidden_layers);
    assert!(step
        .routes_by_layer
        .iter()
        .all(|routes| routes.len() == model.config().num_experts_per_tok));
    assert_eq!(state.position(), 1);
}

#[test]
#[ignore = "executes the official chat-mode Hello prompt through the real 43-layer base model"]
fn real_checkpoint_chat_prompt_predicts_hello() {
    let model_dir = checkpoint_dir();
    let tokenizer = ByteBpeTokenizer::load(&model_dir).unwrap();
    let rendered = prompt::render_chat(
        &[Message::new(Role::User, "Hello")],
        PromptOptions {
            thinking_mode: ThinkingMode::Chat,
            ..PromptOptions::default()
        },
    );
    let prompt_tokens = tokenizer.encode(&rendered).unwrap();
    assert_eq!(prompt_tokens, [0, 128803, 19923, 128804, 128822]);

    let expert_slots = 6;
    let requirements =
        DeepseekRuntimeModel::inspect_requirements(&model_dir, prompt_tokens.len(), expert_slots)
            .unwrap();
    let model = DeepseekRuntimeModel::load(
        &model_dir,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.resident_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.kv_cache_bytes,
            expert_slots_per_layer: expert_slots,
            maximum_expert_bytes: requirements.maximum_expert_bytes,
            context_limit: prompt_tokens.len(),
        },
    )
    .unwrap();
    let mut state = model.new_state().unwrap();
    let mut logits = Vec::new();
    for token in prompt_tokens {
        logits = model.forward_token(token, &mut state).unwrap().logits;
    }
    let next = logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0 as u32;
    assert_eq!(next, 19923);
    assert_eq!(tokenizer.decode(&[next], false).unwrap(), "Hello");
}

#[test]
#[ignore = "requires the release tokenizer.json"]
fn release_tokenizer_matches_the_official_chat_golden() {
    let tokenizer = ByteBpeTokenizer::load(checkpoint_dir()).unwrap();
    let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜><think>";
    assert_eq!(
        tokenizer.encode(prompt).unwrap(),
        vec![0, 128803, 3085, 344, 223, 20, 13, 20, 33, 128804, 128821]
    );
    assert_eq!(tokenizer.encode("Hello").unwrap(), vec![19923]);
    assert_eq!(tokenizer.encode("你好").unwrap(), vec![30594]);
}
