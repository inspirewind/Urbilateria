//! Opt-in native-checkpoint gates. QWEN36_MODEL_DIR points to the unmodified release.
use std::{env, path::PathBuf};
use urbilateria::models::qwen3_6::Qwen36RuntimeModel;
use urbilateria::runtime::RuntimeLoadOptions;

fn directory() -> PathBuf {
    env::var("QWEN36_MODEL_DIR")
        .expect("set QWEN36_MODEL_DIR")
        .into()
}
fn model(context: usize) -> Qwen36RuntimeModel {
    let directory = directory();
    let r = Qwen36RuntimeModel::inspect_requirements(&directory, context, 0).unwrap();
    assert_eq!(r.schema.checkpoint_tensor_count, 1045);
    assert_eq!(r.schema.checkpoint_shard_count, 26);
    assert_eq!(r.expert_bytes, 6 * 1024 * 1024);
    Qwen36RuntimeModel::load(
        directory,
        RuntimeLoadOptions {
            resident_budget_bytes: r.peak_resident_bytes,
            expert_cache_budget_bytes: 0,
            kv_cache_budget_bytes: r.kv_cache_bytes
                + r.recurrent_state_bytes
                + r.convolution_state_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: r.expert_bytes,
            context_limit: context,
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires all 26 native Qwen3.6 BF16 shards and the generated Transformers oracle"]
fn real_decode_matches_transformers() {
    let path = env::var("QWEN36_ORACLE")
        .unwrap_or_else(|_| "tests/fixtures/qwen3_6_real_oracle.json".into());
    let oracle: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let expected_steps = oracle["steps"].as_array().unwrap();
    let model = model(expected_steps.len());
    let mut state = model.new_state().unwrap();
    for (position, expected) in expected_steps.iter().enumerate() {
        let traced = model
            .forward_token_traced(expected["token"].as_u64().unwrap() as u32, &mut state)
            .unwrap();
        assert_eq!(state.position(), position + 1);
        assert!(traced.step.logits.iter().all(|x| x.is_finite()));
        let argmax = traced
            .step
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let mut max_error = 0.0f64;
        let mut squared = 0.0f64;
        for sample in expected["logit_samples"].as_array().unwrap() {
            let actual = traced.step.logits[sample[0].as_u64().unwrap() as usize] as f64;
            let delta = (actual - sample[1].as_f64().unwrap()).abs();
            max_error = max_error.max(delta);
            squared += delta * delta;
        }
        let rmse = (squared / expected["logit_samples"].as_array().unwrap().len() as f64).sqrt();
        let mut max_layer_relative = 0.0f64;
        let mut route_overlap = 0usize;
        for (layer, (actual, reference)) in traced
            .layer_hidden_states
            .iter()
            .zip(expected["layers"].as_array().unwrap())
            .enumerate()
        {
            let norm = actual
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            max_layer_relative =
                max_layer_relative.max((norm / reference["l2_norm"].as_f64().unwrap() - 1.0).abs());
            let expected_experts = reference["experts"].as_array().unwrap();
            route_overlap += traced.step.routes_by_layer[layer]
                .iter()
                .filter(|r| {
                    expected_experts
                        .iter()
                        .any(|e| e.as_u64() == Some(r.expert as u64))
                })
                .count();
        }
        eprintln!("position={position} argmax={argmax} expected={} logit max={max_error} rmse={rmse} max layer relative={max_layer_relative}", expected["argmax"]);
        assert_eq!(argmax as u64, expected["argmax"].as_u64().unwrap());
        // CPU BF16 GEMMs use different accumulation trees; gate distribution and layer drift.
        // Native BF16 matmul vs Rust FP32/FMA and recurrent vs chunked DeltaNet differ
        // at rounding boundaries. Near-tied top-k experts amplify the first-layer ~1 ULP
        // discrepancy. These are bounded distribution gates, not bitwise parity claims.
        assert!(
            max_error <= 0.5 && rmse <= 0.125,
            "logits differ from upstream"
        );
        assert!(
            route_overlap >= 280,
            "less than 87.5% of expert selections agree"
        );
        assert!(max_layer_relative <= 0.04, "hidden-state norm drift");
    }
}

#[test]
#[ignore = "reads native checkpoint; verifies hybrid recurrent/GQA prefill and decode agree"]
fn real_prefill_matches_incremental_decode() {
    let model = model(3);
    let tokens = [9419, 11, 0];
    let mut incremental = model.new_state().unwrap();
    let mut last = Vec::new();
    for token in tokens {
        last = model.forward_token(token, &mut incremental).unwrap().logits;
    }
    let mut batched = model.new_state().unwrap();
    let actual = model.prefill_tokens(&tokens, &mut batched).unwrap();
    assert_eq!(actual.logits, last);
    assert_eq!(batched.position(), 3);
}

#[test]
#[ignore = "requires the native tokenizer; matches independent Transformers token IDs"]
fn real_tokenizer_matches_official_prompt_fixtures() {
    let tokenizer = urbilateria::tokenizer::ByteBpeTokenizer::load(directory()).unwrap();
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3_6_prompts.json")).unwrap();
    for fixture in fixtures.as_array().unwrap() {
        let expected: Vec<u32> = serde_json::from_value(fixture["tokens"].clone()).unwrap();
        assert_eq!(
            tokenizer
                .encode(fixture["rendered"].as_str().unwrap())
                .unwrap(),
            expected
        );
    }
}
