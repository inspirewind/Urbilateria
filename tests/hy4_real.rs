//! Opt-in correctness gates for Tencent's official Hy4-preview-FP8 release.

use std::path::PathBuf;
use std::process::Command;
use urbilateria::models::hy4::{
    expert::Hy4Expert,
    prompt::{render_chat, Message, PromptOptions, Role},
    runtime::Hy4RuntimeModel,
    schema::{self, RELEASE_SHARD_COUNT, RELEASE_TENSOR_COUNT},
};
use urbilateria::runtime::RuntimeLoadOptions;
use urbilateria::storage::{inspect_weight_matrix, load_weight_matrix, TensorIndex};
use urbilateria::tokenizer::ByteBpeTokenizer;
use urbilateria::{ModelConfig, ModelFamily};

fn checkpoint_dir() -> PathBuf {
    std::env::var_os("HY4_MODEL_DIR")
        .map(PathBuf::from)
        .expect("set HY4_MODEL_DIR to the official Hy4-preview-FP8 directory")
}

#[test]
#[ignore = "requires all 130 official Hy4-preview-FP8 shards; reads headers only"]
fn complete_headers_match_release_schema_and_shard_assignments() {
    let directory = checkpoint_dir();
    let detected = ModelConfig::load(&directory).unwrap();
    assert_eq!(detected.family(), ModelFamily::Hy4);
    let ModelConfig::Hy4(config) = detected else {
        unreachable!();
    };
    let index = TensorIndex::open(&directory).unwrap();
    let report = schema::inspect_requirements(&config, &index, 2_048, 1).unwrap();
    assert_eq!(report.required_tensor_count, RELEASE_TENSOR_COUNT);
    assert_eq!(report.checkpoint_tensor_count, RELEASE_TENSOR_COUNT);
    assert_eq!(report.checkpoint_shard_count, RELEASE_SHARD_COUNT);
    assert_eq!(report.base_tensor_count, 2_799);
    assert_eq!(report.mtp_tensor_count, 39);
    assert_eq!(report.logical_parameter_count, 779_960_992_733);
}

#[test]
#[ignore = "reads one bounded ModelOpt-MXFP8 matrix from the Hy4 release"]
fn real_modelopt_mxfp8_matrix_decodes_with_one_by_32_scales() {
    let directory = checkpoint_dir();
    let index = TensorIndex::open(&directory).unwrap();
    let name = "model.layers.0.self_attn.q_a_proj.weight";
    let layout = inspect_weight_matrix(&index, name, 2_048, 6_144).unwrap();
    let matrix = load_weight_matrix(&index, name, 2_048, 6_144, layout.resident_bytes).unwrap();
    assert_eq!(matrix.resident_bytes() as u64, layout.resident_bytes);
    assert!(matrix.row(0).unwrap().iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "reads one 37.1 MiB expert slice from Hy4's consolidated expert tensors"]
fn consolidated_expert_loader_reads_only_one_native_expert() {
    let directory = checkpoint_dir();
    let config = ModelConfig::load(&directory).unwrap();
    let ModelConfig::Hy4(config) = config else {
        unreachable!();
    };
    let index = TensorIndex::open(&directory).unwrap();
    let expert = Hy4Expert::load(&index, &config, 1, 0, 38_928_384).unwrap();
    assert_eq!(expert.resident_bytes(), 38_928_384);
    assert!(expert
        .gate_row(0)
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
    assert!(expert
        .up_row(0)
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
    assert!(expert
        .down_row(0)
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires the official Hy4 tokenizer.json"]
fn release_tokenizer_and_chat_prompt_match_goldens() {
    let directory = checkpoint_dir();
    let tokenizer = ByteBpeTokenizer::load(&directory).unwrap();
    assert_eq!(tokenizer.encode("Hello").unwrap(), [12_433]);
    assert_eq!(
        tokenizer.token_id("<｜hy_start:opensource｜>"),
        Some(120_000)
    );
    assert_eq!(
        tokenizer.token_id("<｜hy_middle:opensource｜>"),
        Some(120_001)
    );
    assert_eq!(tokenizer.token_id("<｜hy_end:opensource｜>"), Some(120_025));
    assert_eq!(
        tokenizer.token_id("<｜reasoning_mode:opensource｜>"),
        Some(120_039)
    );

    let prompt = render_chat(
        &[Message::new(Role::User, "Hello")],
        PromptOptions::default(),
    );
    let ids = tokenizer.encode(&prompt).unwrap();
    assert_eq!(ids.len(), 28);
    assert_eq!(&ids[..4], [120_000, 13_251, 120_001, 120_039]);
    assert_eq!(ids[15], 12_433);
}

#[test]
#[ignore = "validates all headers and loads the bounded Hy4 runtime root"]
fn real_checkpoint_constructs_the_base_runtime() {
    let directory = checkpoint_dir();
    let requirements = Hy4RuntimeModel::inspect_requirements(&directory, 1, 0).unwrap();
    let model = Hy4RuntimeModel::load(
        &directory,
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
    assert_eq!(model.config().num_hidden_layers, 78);
    assert_eq!(model.new_state().unwrap().position(), 0);
}

#[test]
#[ignore = "executes one complete 78-layer token on the 770B Hy4 release"]
fn real_checkpoint_executes_one_complete_base_token() {
    let directory = checkpoint_dir();
    let requirements = Hy4RuntimeModel::inspect_requirements(&directory, 1, 0).unwrap();
    let model = Hy4RuntimeModel::load(
        &directory,
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
    let step = model
        .forward_token(model.config().bos_token_id, &mut state)
        .unwrap();
    assert_eq!(step.logits.len(), model.config().vocab_size);
    assert!(step.logits.iter().all(|value| value.is_finite()));
    assert_eq!(step.routes_by_layer.len(), 78);
    assert!(step.routes_by_layer[0].is_empty());
    assert!(step.routes_by_layer[1..]
        .iter()
        .all(|routes| routes.len() == model.config().num_experts_per_tok));
    assert_eq!(state.position(), 1);
}

#[test]
#[ignore = "runs public Hy4 generation through all 78 layers on the release checkpoint"]
fn cli_generate_runs_the_native_hy4_path() {
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("generate")
        .arg(checkpoint_dir())
        .args([
            "--prompt",
            "<｜hy_start:opensource｜>",
            "--ram-gib",
            "2",
            "--max-new-tokens",
            "1",
            "--threads",
            "20",
            "--raw-prompt",
            "--allow-large-model",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "urb generate failed:\n{stderr}");
    assert!(stderr.contains("preflight: Hy4-preview-FP8 prompt=1 tokens"));
    assert!(stderr.contains("done: new_tokens=1, stop=max_new_tokens"));
    assert!(stderr.contains("misses=616"));
    assert_eq!(output.stdout, b"_\n");
}
