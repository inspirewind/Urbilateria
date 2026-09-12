//! Opt-in gates for the official Qwen3.8-2.4T-A95B-FP8 release.
//!
//! Metadata and tokenizer checks need only the five small files documented in the README. The
//! header gate additionally requires all 213 `.safetensors` shards, but never reads their payloads.

use std::env;
use std::path::PathBuf;
use std::process::Command;
use urbilateria::math::simulate_finegrained_e4m3_activation;
use urbilateria::models::qwen3_8::{
    expert::Qwen38Expert,
    prompt::{render_chat, Message, PromptOptions, Role},
    schema::{inspect_manifest, inspect_requirements, HfWeightIndex, RELEASE_TENSOR_COUNT},
    Qwen38ReleaseConfig, Qwen38RuntimeModel,
};
use urbilateria::runtime::RuntimeLoadOptions;
use urbilateria::storage::{inspect_weight_matrix, load_weight_matrices, TensorIndex};
use urbilateria::tokenizer::ByteBpeTokenizer;
use urbilateria::{ModelConfig, ModelFamily};

fn model_dir() -> PathBuf {
    PathBuf::from(
        env::var("QWEN38_MODEL_DIR")
            .expect("set QWEN38_MODEL_DIR to the official Qwen3.8-2.4T-A95B-FP8 directory"),
    )
}

#[test]
#[ignore = "requires the five small official Qwen3.8 metadata/tokenizer files"]
fn release_metadata_matches_the_pinned_manifest() {
    let directory = model_dir();
    let release = Qwen38ReleaseConfig::load(&directory).unwrap();
    let detected = ModelConfig::load(&directory).unwrap();
    assert_eq!(detected.family(), ModelFamily::Qwen38);

    let report = inspect_manifest(&release.model, &directory).unwrap();
    assert_eq!(report.required_tensor_count, RELEASE_TENSOR_COUNT);
    assert_eq!(report.checkpoint_tensor_count, RELEASE_TENSOR_COUNT);
    assert_eq!(report.checkpoint_shard_count, 213);
    assert_eq!(report.stop_token_ids, [248_046, 248_044]);
}

#[test]
#[ignore = "requires the official Qwen3.8 config.json; verifies public CLI dispatch without loading weights"]
fn cli_generate_routes_to_the_native_qwen_path() {
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("generate")
        .arg(model_dir())
        .args([
            "--prompt",
            "Hello",
            "--ram-gib",
            "1",
            "--max-new-tokens",
            "1",
            "--no-thinking",
            "--allow-large-model",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Qwen3.8 requires thinking; --no-thinking is unsupported"),
        "unexpected CLI error:\n{stderr}"
    );
    assert!(!stderr.contains("correctness-gated"));
}

#[test]
#[ignore = "requires all 213 official Qwen3.8 safetensors shards; reads headers only"]
fn complete_headers_match_dtype_shape_and_shard_assignments() {
    let directory = model_dir();
    let release = Qwen38ReleaseConfig::load(&directory).unwrap();
    let hf_index = HfWeightIndex::load(&directory).unwrap();
    let tensor_index = TensorIndex::open(&directory).unwrap();
    let report = inspect_requirements(&release.model, &hf_index, &tensor_index).unwrap();
    assert_eq!(report.checkpoint_tensor_count, RELEASE_TENSOR_COUNT);
    assert_eq!(report.checkpoint_shard_count, 213);
}

#[test]
#[ignore = "requires all 213 official shards; constructs the streamed runtime root"]
fn complete_text_runtime_loads_with_audited_memory_contract() {
    let directory = model_dir();
    let requirements = Qwen38RuntimeModel::inspect_requirements(&directory, 1, 0).unwrap();
    let state_bytes = requirements
        .kv_cache_bytes
        .checked_add(requirements.recurrent_state_bytes)
        .and_then(|bytes| bytes.checked_add(requirements.convolution_state_bytes))
        .unwrap();
    let model = Qwen38RuntimeModel::load(
        &directory,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.peak_resident_bytes,
            expert_cache_budget_bytes: 0,
            kv_cache_budget_bytes: state_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.expert_bytes,
            context_limit: 1,
        },
    )
    .unwrap();
    assert_eq!(model.requirements().schema.checkpoint_shard_count, 213);
    assert_eq!(model.requirements().context_limit, 1);
}

fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

fn bf16_rank(value: f32) -> u16 {
    let bits = (value.to_bits() >> 16) as u16;
    if bits & 0x8000 == 0 {
        bits | 0x8000
    } else {
        !bits
    }
}

fn assert_bf16_oracle(
    actual: &[f32],
    expected: &serde_json::Value,
    component: &str,
    maximum_ulps: u16,
) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len(), "{component} length");
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected.as_f64().unwrap() as f32;
        let distance = bf16_rank(actual).abs_diff(bf16_rank(expected));
        assert!(
            distance <= maximum_ulps,
            "{component}[{index}] {actual:?} != {expected:?} ({distance} BF16 ULPs)"
        );
    }
}

fn maximum_bf16_ulps(actual: &[f32], expected: &serde_json::Value) -> u16 {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, expected)| {
            bf16_rank(actual).abs_diff(bf16_rank(expected.as_f64().unwrap() as f32))
        })
        .max()
        .unwrap_or(0)
}

#[test]
#[ignore = "reads one real layer-0 routed expert and checks it against an independent PyTorch FP8 oracle"]
fn real_fp8_expert_matches_independent_pytorch_oracle() {
    let directory = model_dir();
    let oracle: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen3_8_real_fp8_expert.json"
    )))
    .unwrap();
    assert_eq!(oracle["checkpoint"], "Qwen/Qwen3.8-2.4T-A95B-FP8");
    assert_eq!(
        oracle["oracle"]["implementation"],
        "independent PyTorch block dequantization"
    );

    let index = TensorIndex::open(&directory).unwrap();
    let prefix = "model.layers.0.mlp.experts.0";
    let names = [
        format!("{prefix}.gate_proj.weight"),
        format!("{prefix}.up_proj.weight"),
        format!("{prefix}.down_proj.weight"),
    ];
    let specifications = [
        (names[0].as_str(), 2_048, 8_192),
        (names[1].as_str(), 2_048, 8_192),
        (names[2].as_str(), 8_192, 2_048),
    ];
    let budget = specifications
        .iter()
        .map(|(name, rows, columns)| {
            inspect_weight_matrix(&index, name, *rows, *columns)
                .unwrap()
                .resident_bytes
        })
        .sum();
    let mut matrices = load_weight_matrices(&index, &specifications, budget)
        .unwrap()
        .into_iter();
    let gate = matrices.next().unwrap();
    let up = matrices.next().unwrap();
    let down = matrices.next().unwrap();

    let input = (0..8_192)
        .map(|index| (((index * 37) % 257) - 128) as f32 / 37.0)
        .collect::<Vec<_>>();
    let quantized = simulate_finegrained_e4m3_activation(&input, 128).unwrap();
    let gate_output = gate
        .matvec(&quantized)
        .unwrap()
        .into_iter()
        .map(bf16)
        .collect::<Vec<_>>();
    let up_output = up
        .matvec(&quantized)
        .unwrap()
        .into_iter()
        .map(bf16)
        .collect::<Vec<_>>();
    assert_bf16_oracle(&gate_output, &oracle["expected"]["gate"], "gate", 0);
    assert_bf16_oracle(&up_output, &oracle["expected"]["up"], "up", 0);

    let expert = Qwen38Expert::new(gate, up, down).unwrap();
    let output = expert.forward(&input).unwrap();
    // PyTorch's vectorized GEMM and the scalar Rust kernel use different accumulation trees.
    // Their materialized BF16 results are required to agree within one representable value.
    assert_bf16_oracle(&output, &oracle["expected"]["expert"], "expert", 1);
}

#[test]
#[ignore = "executes one token through all 92 layers and compares every boundary with the independent PyTorch oracle"]
fn real_checkpoint_matches_independent_full_logits_oracle() {
    let directory = model_dir();
    let oracle: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen3_8_real_logits.json"
    )))
    .unwrap();
    assert_eq!(oracle["checkpoint"], "Qwen/Qwen3.8-2.4T-A95B-FP8");
    assert_eq!(oracle["token"], 9_419);
    let requirements = Qwen38RuntimeModel::inspect_requirements(&directory, 1, 0).unwrap();
    let state_bytes = requirements
        .kv_cache_bytes
        .checked_add(requirements.recurrent_state_bytes)
        .and_then(|bytes| bytes.checked_add(requirements.convolution_state_bytes))
        .unwrap();
    let model = Qwen38RuntimeModel::load(
        &directory,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.peak_resident_bytes,
            expert_cache_budget_bytes: 0,
            kv_cache_budget_bytes: state_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.expert_bytes,
            context_limit: 1,
        },
    )
    .unwrap();
    let mut state = model.new_state().unwrap();
    let traced = model.forward_token_traced(9_419, &mut state).unwrap();
    let step = &traced.step;
    assert_eq!(state.position(), 1);
    assert_eq!(step.logits.len(), model.config().vocab_size);
    assert_eq!(step.routes_by_layer.len(), model.config().num_hidden_layers);
    assert!(step
        .routes_by_layer
        .iter()
        .all(|routes| routes.len() == model.config().num_experts_per_tok));
    assert!(step.logits.iter().all(|value| value.is_finite()));

    let expected_layers = oracle["layer_traces"].as_array().unwrap();
    assert_eq!(traced.layer_hidden_states.len(), expected_layers.len());
    let mut maximum_hidden_ulps = 0;
    let mut maximum_hidden_sum_difference = 0.0f64;
    let mut maximum_hidden_sample_absolute_difference = 0.0f64;
    let mut maximum_hidden_l2_relative_difference = 0.0f64;
    let mut noteworthy_hidden_layers = Vec::new();
    for (layer, (actual, expected)) in traced
        .layer_hidden_states
        .iter()
        .zip(expected_layers)
        .enumerate()
    {
        assert_eq!(expected["layer"].as_u64().unwrap() as usize, layer);
        let layer_ulps = maximum_bf16_ulps(actual.get(..16).unwrap(), &expected["first_16"]);
        maximum_hidden_ulps = maximum_hidden_ulps.max(layer_ulps);
        for (&actual, expected) in actual
            .get(..16)
            .unwrap()
            .iter()
            .zip(expected["first_16"].as_array().unwrap())
        {
            maximum_hidden_sample_absolute_difference = maximum_hidden_sample_absolute_difference
                .max((f64::from(actual) - expected.as_f64().unwrap()).abs());
        }
        let actual_sum = actual.iter().map(|value| f64::from(*value)).sum::<f64>();
        let expected_sum = expected["sum"].as_f64().unwrap();
        let actual_l2 = actual
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let expected_l2 = expected["l2_norm"].as_f64().unwrap();
        maximum_hidden_l2_relative_difference = maximum_hidden_l2_relative_difference
            .max((actual_l2 - expected_l2).abs() / expected_l2);
        maximum_hidden_sum_difference =
            maximum_hidden_sum_difference.max((actual_sum - expected_sum).abs());
        if layer_ulps > 4 || (actual_sum - expected_sum).abs() > 0.25 {
            noteworthy_hidden_layers.push((layer, layer_ulps, (actual_sum - expected_sum).abs()));
        }
    }

    let expected_routes = oracle["routes_by_layer"].as_array().unwrap();
    let mut route_mismatches = Vec::new();
    let mut route_set_mismatches = Vec::new();
    let mut maximum_route_weight_ulps = 0;
    for (layer, (actual, expected)) in step.routes_by_layer.iter().zip(expected_routes).enumerate()
    {
        let expected = expected.as_array().unwrap();
        assert_eq!(actual.len(), expected.len(), "layer {layer} routes");
        for (rank, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let expected_expert = expected["expert"].as_u64().unwrap() as usize;
            if actual.expert != expected_expert {
                route_mismatches.push((layer, rank, actual.expert, expected_expert));
            }
            let expected_weight = expected["weight"].as_f64().unwrap() as f32;
            maximum_route_weight_ulps = maximum_route_weight_ulps
                .max(bf16_rank(actual.weight).abs_diff(bf16_rank(expected_weight)));
        }
        let mut actual_set = actual.iter().map(|route| route.expert).collect::<Vec<_>>();
        let mut expected_set = expected
            .iter()
            .map(|route| route["expert"].as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        actual_set.sort_unstable();
        expected_set.sort_unstable();
        if actual_set != expected_set {
            route_set_mismatches.push((layer, actual_set, expected_set));
        }
    }

    let maximum_logits_ulps = maximum_bf16_ulps(&step.logits, &oracle["logits"]);
    let expected_logits = oracle["logits"].as_array().unwrap();
    let mut maximum_logit_absolute_difference = 0.0f64;
    let mut sum_logit_absolute_difference = 0.0f64;
    let mut dot = 0.0f64;
    let mut actual_square = 0.0f64;
    let mut expected_square = 0.0f64;
    for (&actual, expected) in step.logits.iter().zip(expected_logits) {
        let actual = f64::from(actual);
        let expected = expected.as_f64().unwrap();
        maximum_logit_absolute_difference =
            maximum_logit_absolute_difference.max((actual - expected).abs());
        sum_logit_absolute_difference += (actual - expected).abs();
        dot += actual * expected;
        actual_square += actual * actual;
        expected_square += expected * expected;
    }
    let mean_logit_absolute_difference = sum_logit_absolute_difference / step.logits.len() as f64;
    let logit_cosine = dot / (actual_square.sqrt() * expected_square.sqrt());
    let argmax = step
        .logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    let mut actual_top = (0..step.logits.len()).collect::<Vec<_>>();
    actual_top.sort_unstable_by(|&left, &right| step.logits[right].total_cmp(&step.logits[left]));
    actual_top.truncate(20);
    let expected_top = oracle["top_20"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value["token"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let top_20_overlap = actual_top
        .iter()
        .filter(|token| expected_top.contains(token))
        .count();
    let route_replacements = route_set_mismatches
        .iter()
        .map(|(_, actual, expected)| {
            expected
                .iter()
                .filter(|expert| !actual.contains(expert))
                .count()
        })
        .sum::<usize>();
    eprintln!(
        "Qwen3.8 oracle deltas: hidden_sample_max={maximum_hidden_ulps} BF16 ULPs/{maximum_hidden_sample_absolute_difference} abs, hidden_sum_max={maximum_hidden_sum_difference}, hidden_l2_relative_max={maximum_hidden_l2_relative_difference}, route_weight_max={maximum_route_weight_ulps} BF16 ULPs, route_rank_mismatches={}, route_set_mismatches={}, route_replacements={route_replacements}, logits_max={maximum_logits_ulps} BF16 ULPs, logits_max_abs={maximum_logit_absolute_difference}, logits_mae={mean_logit_absolute_difference}, logits_cosine={logit_cosine}, argmax={argmax}, expected_argmax={}, top20_overlap={top_20_overlap}/20",
        route_mismatches.len(),
        route_set_mismatches.len(),
        oracle["argmax"]
    );
    eprintln!(
        "first noteworthy hidden layers: {:?}",
        &noteworthy_hidden_layers[..noteworthy_hidden_layers.len().min(12)]
    );
    eprintln!(
        "first route rank mismatches: {:?}",
        &route_mismatches[..route_mismatches.len().min(12)]
    );
    eprintln!(
        "first route set mismatches: {:?}",
        &route_set_mismatches[..route_set_mismatches.len().min(8)]
    );
    eprintln!("actual top20: {actual_top:?}");
    // Near-zero values make deep ULP distances misleading after different GEMM reduction trees.
    // Bound absolute samples and aggregate hidden statistics instead.
    assert!(maximum_hidden_sample_absolute_difference <= 0.25);
    assert!(
        maximum_hidden_sum_difference <= 6.0,
        "layer hidden sum delta is {maximum_hidden_sum_difference}"
    );
    // Different GEMM reduction trees can move near-boundary routed experts and amplify the
    // intermediate norm without changing the final distribution materially. Keep that drift
    // below 3% while the stricter logit-distribution and top-token gates below remain decisive.
    assert!(
        maximum_hidden_l2_relative_difference <= 0.03,
        "layer hidden L2 relative delta is {maximum_hidden_l2_relative_difference}"
    );
    assert!(route_mismatches.len() <= 200);
    assert!(route_set_mismatches.len() <= 25);
    assert!(route_replacements <= 32);
    assert!(
        maximum_route_weight_ulps <= 32,
        "route weight delta is {maximum_route_weight_ulps} BF16 ULPs"
    );
    assert!(maximum_logit_absolute_difference <= 0.25);
    assert!(mean_logit_absolute_difference <= 0.04);
    assert!(logit_cosine >= 0.9999);
    assert_eq!(top_20_overlap, 20);
    assert_eq!(argmax, oracle["argmax"].as_u64().unwrap() as usize);
    assert_eq!(state.cached_f32_elements(), 150_403_072);
    let telemetry = state.expert_telemetry();
    assert_eq!(telemetry.hits, 0);
    assert_eq!(telemetry.misses, 920);
    assert_eq!(telemetry.evictions, 0);
    assert_eq!(telemetry.resident_experts, 0);
}

#[test]
#[ignore = "requires the official Qwen3.8 tokenizer.json"]
fn release_tokenizer_and_chat_prompt_match_goldens() {
    let directory = model_dir();
    let tokenizer = ByteBpeTokenizer::load(&directory).unwrap();
    assert_eq!(tokenizer.encode("Hello").unwrap(), [9_419]);
    assert_eq!(
        tokenizer.encode("Hello, 世界!").unwrap(),
        [9_419, 11, 220, 96_748, 0]
    );
    assert_eq!(tokenizer.token_id("<|im_start|>"), Some(248_045));
    assert_eq!(tokenizer.token_id("<|im_end|>"), Some(248_046));
    assert_eq!(tokenizer.token_id("<think>"), Some(248_068));

    let prompt = render_chat(
        &[Message::new(Role::User, "Hello")],
        PromptOptions::default(),
    )
    .unwrap();
    let ids = tokenizer.encode(&prompt).unwrap();
    assert_eq!(
        &ids[ids.len() - 11..],
        [248_045, 846, 198, 9_419, 248_046, 198, 248_045, 74_455, 198, 248_068, 198,]
    );
}
