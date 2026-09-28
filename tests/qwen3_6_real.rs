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
    assert_eq!(model.default_prefill_batch_size(), 32);
    let tokens = [9419, 11, 0];
    let mut incremental = model.new_state().unwrap();
    let mut last = Vec::new();
    for token in tokens {
        last = model.forward_token(token, &mut incremental).unwrap().logits;
    }
    for batch in [1, 2, 32, Qwen36RuntimeModel::MAX_PREFILL_BATCH] {
        let mut batched = model.new_state().unwrap();
        for invalid in [0, Qwen36RuntimeModel::MAX_PREFILL_BATCH + 1] {
            assert!(model
                .prefill_tokens_with_batch_size(&tokens, &mut batched, invalid)
                .is_err());
            assert_eq!(batched.position(), 0);
        }
        let actual = model
            .prefill_tokens_with_batch_size(&tokens, &mut batched, batch)
            .unwrap();
        assert_eq!(actual.logits, last);
        assert_eq!(batched.position(), 3);
    }
}

#[test]
#[ignore = "requires native weights; verifies all prefill chunk boundaries, continuation and rewind"]
fn batched_prefill_crosses_boundary_and_preserves_continuation() {
    use urbilateria::runtime::session::SessionState;
    let maximum = Qwen36RuntimeModel::MAX_PREFILL_BATCH;
    let r = Qwen36RuntimeModel::inspect_requirements(directory(), maximum + 8, 0).unwrap();
    let options = r.plan_load_options(90 << 30, true).unwrap();
    let model = Qwen36RuntimeModel::load(directory(), options).unwrap();
    assert_eq!(model.default_prefill_batch_size(), maximum);
    let mut state = model.new_state().unwrap();
    model.forward_token(9419, &mut state).unwrap();
    let checkpoint = state.checkpoint();
    let tokens: Vec<u32> = [11, 9419, 0, 271, 32]
        .into_iter()
        .cycle()
        .take(maximum + 1)
        .collect();
    let mut expected = None;
    for &token in &tokens {
        expected = Some(model.forward_token(token, &mut state).unwrap());
    }
    let continuation = [32, 2972].map(|token| model.forward_token(token, &mut state).unwrap());
    let misses = state.expert_telemetry().misses;
    state.restore(checkpoint).unwrap();
    let expected = expected.unwrap();
    for batch in [32, 64, maximum] {
        let checkpoint = state.checkpoint();
        let hits = state.expert_telemetry().hits;
        let actual = if batch == maximum {
            model.prefill_tokens(&tokens, &mut state).unwrap()
        } else {
            model
                .prefill_tokens_with_batch_size(&tokens, &mut state, batch)
                .unwrap()
        };
        assert_eq!(actual, expected, "batch={batch}");
        assert_eq!(state.position(), maximum + 2);
        for (token, expected) in [32, 2972].into_iter().zip(&continuation) {
            assert_eq!(&model.forward_token(token, &mut state).unwrap(), expected);
        }
        assert_eq!(state.expert_telemetry().misses, misses);
        assert_eq!(
            state.expert_telemetry().hits - hits,
            (tokens.len() as u64 + 2) * 40 * 8
        );
        state.restore(checkpoint).unwrap();
    }
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

#[test]
#[ignore = "requires native weights; verifies resident layers/LM head/expert cache preserve every logit and route"]
fn resident_weights_match_streamed_and_survive_rewind() {
    use urbilateria::runtime::session::SessionState;
    let streamed = model(4);
    let r = streamed.requirements();
    let mut options = r.plan_load_options(8 * 1024 * 1024 * 1024, true).unwrap();
    // Retain at least all experts touched by these three tokens without allocating all 35B weights.
    options.expert_slots_per_layer = 24;
    options.expert_cache_budget_bytes = r.expert_bytes * 40 * 24;
    options.resident_budget_bytes = r.peak_resident_bytes
        + options.expert_cache_budget_bytes
        + r.lm_head_resident_bytes
        + r.layer_resident_bytes.iter().sum::<u64>();
    let resident = Qwen36RuntimeModel::load(directory(), options).unwrap();
    assert_eq!(resident.default_prefill_batch_size(), 32);
    let mut reference = streamed.new_state().unwrap();
    let mut cached = resident.new_state().unwrap();
    let start = cached.checkpoint();
    let mut expected = Vec::new();
    for token in [9419, 11, 271] {
        let a = streamed
            .forward_token_traced(token, &mut reference)
            .unwrap();
        let b = resident.forward_token_traced(token, &mut cached).unwrap();
        assert_eq!(a, b);
        expected.push(a.step);
    }
    let before = cached.expert_telemetry().clone();
    cached.restore(start).unwrap();
    for (token, expected) in [9419, 11, 271].into_iter().zip(expected) {
        assert_eq!(
            resident.forward_token(token, &mut cached).unwrap(),
            expected
        );
    }
    assert_eq!(cached.expert_telemetry().misses, before.misses);
    assert_eq!(cached.expert_telemetry().bytes_read, before.bytes_read);
    assert_eq!(cached.expert_telemetry().hits - before.hits, 3 * 40 * 8);
}

#[test]
#[ignore = "requires checkpoint metadata; checks automatic RAM planning boundaries"]
fn memory_plan_reserves_state_and_bounds_all_caches() {
    let r = Qwen36RuntimeModel::inspect_requirements(directory(), 128, 0).unwrap();
    let state = r.kv_cache_bytes + r.recurrent_state_bytes + r.convolution_state_bytes;
    let snapshot = r.recurrent_state_bytes + r.convolution_state_bytes;
    let minimum = r.peak_resident_bytes + state + snapshot;
    assert!(r.plan_load_options(minimum - 1, true).is_err());
    let tiny = r.plan_load_options(minimum, true).unwrap();
    assert_eq!(tiny.expert_slots_per_layer, 0);
    assert_eq!(r.backbone_residency(tiny.resident_budget_bytes), (0, false));
    for ram in [minimum, 2 << 30, 4 << 30, 8 << 30, 32 << 30, 90 << 30] {
        let p = r.plan_load_options(ram, true).unwrap();
        assert!(p.resident_budget_bytes + state + snapshot <= ram);
        assert_eq!(
            p.expert_cache_budget_bytes,
            p.expert_slots_per_layer as u64 * 40 * r.expert_bytes
        );
        assert!(p.expert_slots_per_layer <= 256);
    }
    let full = r.plan_load_options(90 << 30, true).unwrap();
    assert_eq!(full.expert_slots_per_layer, 256);
    assert_eq!(
        r.backbone_residency(full.resident_budget_bytes - full.expert_cache_budget_bytes),
        (40, true)
    );
}

#[test]
#[ignore = "requires native weights; checks a two-layer resident prefix and synchronous streaming tail"]
fn partial_backbone_cache_preserves_logits_and_load_counts() {
    use urbilateria::profiling::ProfileSession;
    let streamed = model(2);
    let r = streamed.requirements();
    let causal_bytes = r.kv_cache_bytes + r.recurrent_state_bytes + r.convolution_state_bytes;
    let ram = r.peak_resident_bytes
        + causal_bytes
        + r.lm_head_resident_bytes
        + r.layer_resident_bytes[..2].iter().sum::<u64>();
    let options = r.plan_load_options(ram, false).unwrap();
    assert_eq!(options.expert_slots_per_layer, 0);
    assert_eq!(
        r.backbone_residency(options.resident_budget_bytes),
        (2, true)
    );
    let partial = Qwen36RuntimeModel::load(directory(), options).unwrap();
    let mut reference = streamed.new_state().unwrap();
    let mut state = partial.new_state().unwrap();
    for (step, token) in [9419, 11].into_iter().enumerate() {
        let expected = streamed.forward_token(token, &mut reference).unwrap();
        let profile = ProfileSession::start();
        let actual = partial.forward_token(token, &mut state).unwrap();
        let report = profile.finish();
        assert_eq!(actual, expected);
        let loads = report
            .stages
            .iter()
            .find(|s| s.stage == "qwen36.layer.load")
            .unwrap();
        assert_eq!(loads.calls, if step == 0 { 40 } else { 38 });
    }
}
