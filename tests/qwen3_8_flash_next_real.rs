//! Explicit opt-in: the 131-shard BF16 checkpoint is not part of the test suite.
use std::{env, path::PathBuf};
use urbilateria::generation::CausalDecoder;
use urbilateria::models::qwen3_8_flash_next::{schema, Qwen38FlashNextRuntimeModel};
use urbilateria::runtime::session::SessionState;
fn directory() -> PathBuf {
    env::var("QWEN38_FLASH_NEXT_MODEL_DIR")
        .expect("set QWEN38_FLASH_NEXT_MODEL_DIR")
        .into()
}

#[test]
#[ignore = "requires complete Flash-Next checkpoint; reads only headers"]
fn complete_checkpoint_and_budget_contract() {
    let req = Qwen38FlashNextRuntimeModel::inspect_requirements(&directory(), 2048, 2).unwrap();
    assert_eq!(req.schema.checkpoint_shard_count, 131);
    assert_eq!(
        req.schema.logical_parameter_count,
        schema::RELEASE_LOGICAL_PARAMETER_COUNT
    );
    assert!(req.indexer_cache_bytes > 0 && req.ple_state_bytes > 0);
    assert!(req.plan_load_options(1024, true).is_err());
    let options = req
        .plan_load_options(16 * 1024 * 1024 * 1024, true)
        .unwrap();
    assert!(
        options.resident_budget_bytes + options.kv_cache_budget_bytes + req.checkpoint_bytes()
            <= 16 * 1024 * 1024 * 1024
    );
    assert!(Qwen38FlashNextRuntimeModel::inspect_requirements(&directory(), 262145, 0).is_err());
    assert!(Qwen38FlashNextRuntimeModel::inspect_requirements(&directory(), 1, 513).is_err());
}

#[test]
#[ignore = "streams real weights, checks independent PyTorch logits and session/prefill parity"]
fn real_logits_match_independent_upstream_reference() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3_8_flash_next_real.json")).unwrap();
    let req = Qwen38FlashNextRuntimeModel::inspect_requirements(&directory(), 8, 0).unwrap();
    let model = Qwen38FlashNextRuntimeModel::load(
        &directory(),
        req.plan_load_options(12 * 1024 * 1024 * 1024, true)
            .unwrap(),
    )
    .unwrap();
    let mut state = model.new_state().unwrap();
    let mut expected_final = Vec::new();
    let mut checkpoint = None;
    for (position, step) in oracle["steps"].as_array().unwrap().iter().enumerate() {
        if position == 1 {
            checkpoint = Some(state.checkpoint());
        }
        let (logits, layers) = model
            .forward_token_traced(step["token"].as_u64().unwrap() as u32, &mut state)
            .unwrap();
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let samples = step["logit_samples"].as_array().unwrap();
        let errors = samples
            .iter()
            .map(|s| {
                (logits[s[0].as_u64().unwrap() as usize] - s[1].as_f64().unwrap() as f32).abs()
            })
            .collect::<Vec<_>>();
        let mae = errors.iter().sum::<f32>() / errors.len() as f32;
        let max = errors.iter().copied().fold(0.0, f32::max);
        eprintln!(
            "position={position} argmax={argmax} expected={} MAE={mae} max={max}",
            step["argmax"]
        );
        for (i, layer) in layers.iter().enumerate() {
            let expected = &step["layers"][i]["first_16"];
            let max = layer
                .iter()
                .take(16)
                .zip(expected.as_array().unwrap())
                .map(|(&a, b)| (a - b.as_f64().unwrap() as f32).abs())
                .fold(0.0, f32::max);
            let norm = layer
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            let expected_norm = step["layers"][i]["l2_norm"].as_f64().unwrap();
            assert!(
                (norm - expected_norm).abs() / expected_norm < 0.04,
                "layer {i} norm drift"
            );
            if max > 0.05 {
                eprintln!("layer {i} first16 max delta {max}");
            }
        }
        assert_eq!(argmax, step["argmax"].as_u64().unwrap() as usize);
        assert!(mae < 0.1 && max < 0.5, "logit error MAE={mae}, max={max}");
        expected_final = logits;
    }
    let tokens = oracle["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["token"].as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    state.restore(checkpoint.unwrap()).unwrap();
    let replay = model.forward_token(tokens[1], &mut state).unwrap();
    assert_same_logits(&replay, &expected_final);
    let mut fresh = model.new_state().unwrap();
    assert_same_logits(
        &model.prefill(&tokens, &mut fresh).unwrap(),
        &expected_final,
    );
    let streamed = Qwen38FlashNextRuntimeModel::load(
        &directory(),
        req.plan_load_options(1024 * 1024 * 1024, true).unwrap(),
    )
    .unwrap();
    let mut streamed_state = streamed.new_state().unwrap();
    assert_same_logits(
        &streamed.prefill(&tokens, &mut streamed_state).unwrap(),
        &expected_final,
    );
    assert_eq!(fresh.position(), tokens.len());
    assert!(fresh.restore(state.checkpoint()).is_err());
    assert!(model.forward_token(248320, &mut fresh).is_err());
    assert_eq!(fresh.position(), tokens.len());
}

#[test]
#[ignore = "requires the official tokenizer files"]
fn tokenizer_matches_shipped_jinja_fixture_and_cli_dispatch() {
    let tokenizer = urbilateria::tokenizer::ByteBpeTokenizer::load(directory()).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3_8_flash_next_tiny.json")).unwrap();
    for case in fixture["prompts"].as_array().unwrap() {
        let actual = tokenizer.encode(case["prompt"].as_str().unwrap()).unwrap();
        let expected = case["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("generate")
        .arg(directory())
        .args([
            "--prompt",
            "Hello",
            "--ram-gib",
            "0.01",
            "--no-thinking",
            "--allow-large-model",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Qwen3.8-Flash-Next") && stderr.contains("RAM budget"),
        "{stderr}"
    );
}

fn assert_same_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let first = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, b))| a != b);
    assert!(first.is_none(), "first mismatched logit: {first:?}");
}
