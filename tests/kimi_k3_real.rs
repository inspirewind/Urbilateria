use std::path::PathBuf;
use std::process::Command;

use urbilateria::models::kimi_k3::{
    expert::{KimiK3Expert, KIMI_K3_EXPERT_RESIDENT_BYTES},
    runtime::KimiK3RuntimeModel,
    schema,
    tokenizer::KimiK3Tokenizer,
    weights::{KimiK3LayerWeightError, KimiK3LayerWeights},
};
use urbilateria::runtime::RuntimeLoadOptions;
use urbilateria::storage::{load_weight_matrix, TensorIndex};
use urbilateria::{ModelConfig, ModelFamily};

fn model_dir() -> PathBuf {
    std::env::var_os("KIMI_K3_MODEL_DIR")
        .map(PathBuf::from)
        .expect("set KIMI_K3_MODEL_DIR to the official Kimi-K3 checkpoint")
}

#[test]
#[ignore = "validates all decoder layers currently present in a large Kimi-K3 transfer"]
fn completed_decoder_layers_match_the_release_schema() {
    let model_dir = model_dir();
    let config = ModelConfig::load(&model_dir).unwrap();
    assert_eq!(config.family(), ModelFamily::KimiK3);
    let config = config.as_kimi_k3().unwrap();
    let index = TensorIndex::open(&model_dir).unwrap();
    let partial = schema::inspect_available_layers(config, &index).unwrap();
    assert!(!partial.validated_layers.is_empty());
    assert_eq!(partial.validated_layers[0], 0);
    assert!(partial
        .validated_layers
        .windows(2)
        .all(|pair| pair[1] == pair[0] + 1));
}

#[test]
#[ignore = "requires all 96 official Kimi-K3 checkpoint shards"]
fn headers_match_the_complete_multimodal_release_schema() {
    let model_dir = model_dir();
    let config = ModelConfig::load(&model_dir).unwrap();
    let index = TensorIndex::open(&model_dir).unwrap();
    let requirements = schema::inspect_requirements(config.as_kimi_k3().unwrap(), &index).unwrap();
    assert_eq!(requirements.required_tensor_count, 497_220);
    assert_eq!(requirements.checkpoint_tensor_count, 497_220);
    assert_eq!(requirements.checkpoint_shard_count, 96);
}

#[test]
#[ignore = "requires tiktoken.model and tokenizer_config.json from the official release"]
fn release_tokenizer_matches_text_goldens_and_stop_identity() {
    let tokenizer = KimiK3Tokenizer::load(model_dir()).unwrap();
    assert_eq!(tokenizer.encode("Hello world").unwrap(), [19_180, 2_695]);
    assert_eq!(tokenizer.encode("你好世界").unwrap(), [33_845, 2_243]);
    assert_eq!(tokenizer.stop_token_id(), 163_586);
    assert_eq!(tokenizer.token_id("[EOS]"), Some(163_585));
    assert_eq!(tokenizer.token_id("<|end_of_msg|>"), Some(163_586));
}

#[test]
#[ignore = "reads one bounded MXFP4 expert matrix from shard 2 of the official release"]
fn real_mxfp4_expert_prefix_decodes_in_low_nibble_order() {
    let model_dir = model_dir();
    let index = TensorIndex::open(&model_dir).unwrap();
    let matrix = load_weight_matrix(
        &index,
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
        3_072,
        3_584,
        6 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(matrix.resident_bytes(), 5_849_088);
    assert_eq!(
        &matrix.row(0).unwrap()[..16],
        &[
            0.0625, 0.0078125, -0.0078125, -0.0234375, -0.0625, -0.03125, 0.0234375, 0.015625,
            0.046875, -0.0, -0.0234375, -0.015625, -0.0234375, -0.0078125, 0.015625, -0.015625,
        ]
    );
}

#[test]
#[ignore = "loads all three native MXFP4 matrices for one official routed expert"]
fn real_expert_loader_uses_the_official_block_sparse_moe_namespace() {
    let model_dir = model_dir();
    let index = TensorIndex::open(&model_dir).unwrap();
    let expert =
        KimiK3Expert::load(&index, 1, 0, 3_584, 3_072, KIMI_K3_EXPERT_RESIDENT_BYTES).unwrap();
    assert_eq!(expert.resident_bytes(), KIMI_K3_EXPERT_RESIDENT_BYTES);
    assert!(expert
        .tensor_names()
        .w1_packed
        .contains(".block_sparse_moe.experts.0."));
}

#[test]
#[ignore = "metadata-preflights one complete official decoder layer without reading its payload"]
fn real_layer_loader_rejects_budget_after_complete_metadata_preflight() {
    let model_dir = model_dir();
    let config = ModelConfig::load(&model_dir).unwrap();
    let index = TensorIndex::open(&model_dir).unwrap();
    let error = KimiK3LayerWeights::load(config.as_kimi_k3().unwrap(), &index, 1, 1).unwrap_err();
    assert!(matches!(
        error,
        KimiK3LayerWeightError::Budget {
            layer: 1,
            required,
            maximum: 1
        } if required > 1
    ));
}

#[test]
#[ignore = "requires all 96 shards; loads only root vectors after a strict full header gate"]
fn complete_text_runtime_loads_with_the_audited_memory_contract() {
    let model_dir = model_dir();
    let requirements = KimiK3RuntimeModel::inspect_requirements(&model_dir, 1, 0).unwrap();
    assert_eq!(requirements.schema.required_tensor_count, 497_220);
    assert_eq!(requirements.streamed_layer_bytes, 2_341_299_200);
    assert_eq!(requirements.kda_state_bytes, 464_633_856);
    assert_eq!(requirements.attn_res_working_bytes, 516_096);
    assert_eq!(requirements.resident_core_bytes, 2_806_449_152);
    assert_eq!(requirements.prefill_scratch_bytes_per_token, 258_432);
    assert_eq!(requirements.mla_cache_bytes, 55_296);
    assert_eq!(requirements.routed_expert_bytes, 17_547_264);

    let model = KimiK3RuntimeModel::load(
        &model_dir,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.resident_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.mla_cache_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.routed_expert_bytes,
            context_limit: 1,
        },
    )
    .unwrap();
    assert_eq!(
        model.requirements().resident_bytes,
        requirements.resident_bytes
    );
}

#[test]
#[ignore = "executes a complete 93-layer five-token prefill and compares every logit with the independent PyTorch oracle"]
fn real_checkpoint_logits_match_the_independent_pytorch_golden() {
    let model_dir = model_dir();
    let golden_path = std::env::var_os("KIMI_K3_REFERENCE_LOGITS")
        .map(PathBuf::from)
        .expect("set KIMI_K3_REFERENCE_LOGITS to the independent ref_logits.json fixture");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&golden_path).unwrap()).unwrap();
    let prompt_tokens = golden["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    assert_eq!(prompt_tokens, [3, 4, 5, 6, 7]);

    // Zero slots keeps routed experts transient: it changes I/O reuse, never the arithmetic.
    let requirements =
        KimiK3RuntimeModel::inspect_requirements(&model_dir, prompt_tokens.len(), 0).unwrap();
    let model = KimiK3RuntimeModel::load(
        &model_dir,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.resident_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.mla_cache_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.routed_expert_bytes,
            context_limit: prompt_tokens.len(),
        },
    )
    .unwrap();
    let mut state = model.new_state().unwrap();
    let step = model.prefill_tokens(&prompt_tokens, &mut state).unwrap();

    assert_eq!(step.logits.len(), model.config().text_config.vocab_size);
    assert!(step.logits.iter().all(|value| value.is_finite()));
    assert_eq!(state.position(), prompt_tokens.len());
    assert_eq!(step.routes_by_layer.len(), 93);
    assert!(step.routes_by_layer[0].is_empty());
    assert!(step.routes_by_layer[1..]
        .iter()
        .all(|routes| routes.len() == 16));

    let reference = golden["logits_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| f32::from_bits(value.as_u64().unwrap() as u32))
        .collect::<Vec<_>>();
    assert_eq!(reference.len(), step.logits.len());
    let predicted = argmax(&step.logits);
    let expected = argmax(&reference);
    assert_eq!(expected, golden["argmax"].as_u64().unwrap() as usize);

    let mut maximum_difference = 0.0f64;
    let mut total_difference = 0.0f64;
    let mut reference_scale = 0.0f64;
    for (&actual, &expected_value) in step.logits.iter().zip(&reference) {
        let difference = (f64::from(actual) - f64::from(expected_value)).abs();
        maximum_difference = maximum_difference.max(difference);
        total_difference += difference;
        reference_scale = reference_scale.max(f64::from(expected_value.abs()));
    }
    let relative_error = maximum_difference / reference_scale.max(1e-30);
    let relative_budget = 1.2e-7 * (7_168.0f64).sqrt() * 50.0;
    let reference_top_ten = top_indices(&reference, 10);
    let top_ten_overlap = top_indices(&step.logits, 10)
        .into_iter()
        .filter(|index| reference_top_ten.contains(index))
        .count();
    eprintln!(
        "Kimi-K3 full logits: argmax={predicted} reference={expected}, top-10={top_ten_overlap}/10, max_diff={maximum_difference:.6e}, relative={relative_error:.6e} (budget {relative_budget:.6e}), mean_diff={:.6e}",
        total_difference / step.logits.len() as f64,
    );

    assert_eq!(predicted, expected);
    assert_eq!(top_ten_overlap, 10);
    assert!(
        relative_error <= relative_budget,
        "full-stack relative logit error {relative_error:.6e} exceeds {relative_budget:.6e}"
    );
}

#[test]
#[ignore = "runs a natural-language prompt through all 93 layers and checks the published first-token continuation"]
fn real_checkpoint_completes_france_with_paris() {
    let model_dir = model_dir();
    let tokenizer = KimiK3Tokenizer::load(&model_dir).unwrap();
    let prompt_tokens = tokenizer.encode("The capital of France is").unwrap();
    assert_eq!(prompt_tokens, [1_008, 10_484, 318, 15_383, 387]);

    let requirements =
        KimiK3RuntimeModel::inspect_requirements(&model_dir, prompt_tokens.len(), 0).unwrap();
    let model = KimiK3RuntimeModel::load(
        &model_dir,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.resident_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.mla_cache_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.routed_expert_bytes,
            context_limit: prompt_tokens.len(),
        },
    )
    .unwrap();
    let mut state = model.new_state().unwrap();
    let step = model.prefill_tokens(&prompt_tokens, &mut state).unwrap();
    let next = argmax(&step.logits) as u32;
    let decoded = tokenizer.decode(&[next], false).unwrap();
    eprintln!("Kimi-K3 completion: token={next}, decoded={decoded:?}");

    assert_eq!(next, 17_374);
    assert_eq!(decoded, " Paris");
}

#[test]
#[ignore = "runs the public urb generate command through Kimi's native no-thinking chat path"]
fn cli_generate_runs_the_native_kimi_chat_path() {
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("generate")
        .arg(model_dir())
        .args([
            "--prompt",
            "Hello",
            "--ram-gib",
            "8",
            "--max-new-tokens",
            "1",
            "--threads",
            "20",
            "--no-thinking",
            "--allow-large-model",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "urb generate failed:\n{stderr}");
    assert!(stderr.contains("preflight: Kimi-K3 prompt=22 tokens"));
    assert!(stderr.contains("done: new_tokens=1, stop=max_new_tokens"));
    assert_eq!(output.stdout, b"Hello\n");
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0
}

fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    indices.truncate(count);
    indices
}
