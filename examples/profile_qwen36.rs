//! Fixed-length benchmark: MODEL_DIR RAM_GIB THREADS TOKENS [PASSES] [BATCH_SIZES] [PROMPT_FILE].
//! BATCH_SIZES is a comma-separated cycle in 1..=128; defaults to the model's cache-aware policy.
//! RAM_GIB=0 forces the minimum-memory reference path. Warm pass replays the same
//! prompt and continuation using retained weights, while clearing all causal state.
use std::{error::Error, path::Path, time::Instant};
use urbilateria::{
    execution::configure_threads,
    models::qwen3_6::{
        prompt::{render_chat, Message, PromptOptions, Role},
        Qwen36RuntimeModel,
    },
    profiling::ProfileSession,
    runtime::{session::SessionState, RuntimeLoadOptions},
    tokenizer::ByteBpeTokenizer,
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let dir = Path::new(args.get(1).ok_or("MODEL_DIR RAM_GIB THREADS TOKENS")?);
    let ram: u64 = args.get(2).map(String::as_str).unwrap_or("90").parse()?;
    let threads: usize = args.get(3).map(String::as_str).unwrap_or("20").parse()?;
    let count: usize = args.get(4).map(String::as_str).unwrap_or("8").parse()?;
    if count < 2 {
        return Err("TOKENS must be >= 2".into());
    }
    let pass_count: usize = args.get(5).map(String::as_str).unwrap_or("2").parse()?;
    let batches = args
        .get(6)
        .map(|value| {
            value
                .split(',')
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    if pass_count < 2
        || batches.as_ref().is_some_and(|sizes| {
            sizes
                .iter()
                .any(|size| !(1..=Qwen36RuntimeModel::MAX_PREFILL_BATCH).contains(size))
        })
    {
        return Err(format!(
            "PASSES must be >= 2 and each BATCH_SIZE must be in 1..={}",
            Qwen36RuntimeModel::MAX_PREFILL_BATCH
        )
        .into());
    }
    configure_threads(threads)?;
    let tokenizer = ByteBpeTokenizer::load(dir)?;
    let user_prompt = match args.get(7) {
        Some(path) => std::fs::read_to_string(path)?,
        None => "Explain what a compiler does.".to_owned(),
    };
    let prompt = render_chat(
        &[Message::new(Role::User, user_prompt)],
        PromptOptions {
            enable_thinking: false,
            ..Default::default()
        },
    )?;
    let tokens = tokenizer.encode(&prompt)?;
    let r = Qwen36RuntimeModel::inspect_requirements(dir, tokens.len() + count, 0)?;
    let options = if ram == 0 {
        RuntimeLoadOptions {
            resident_budget_bytes: r.peak_resident_bytes,
            expert_cache_budget_bytes: 0,
            kv_cache_budget_bytes: r.kv_cache_bytes
                + r.recurrent_state_bytes
                + r.convolution_state_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: r.expert_bytes,
            context_limit: r.context_limit,
        }
    } else {
        r.plan_load_options(ram * 1024 * 1024 * 1024, true)?
    };
    let (cached_layers, cached_head) =
        r.backbone_residency(options.resident_budget_bytes - options.expert_cache_budget_bytes);
    let start = Instant::now();
    let model = Qwen36RuntimeModel::load(dir, options)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let batches = batches.unwrap_or_else(|| vec![model.default_prefill_batch_size()]);
    let mut state = model.new_state()?;
    let mut passes = Vec::new();
    let mask = tokenizer.decodable_token_mask(model.config().vocab_size);
    for pass in 0..pass_count {
        let batch_size = batches[pass % batches.len()];
        let checkpoint = state.checkpoint();
        let previous = state.expert_telemetry().clone();
        let profile = ProfileSession::start_with_threads(Some(threads));
        let start = Instant::now();
        let mut step = model.prefill_tokens_with_batch_size(&tokens, &mut state, batch_size)?;
        let prefill_seconds = start.elapsed().as_secs_f64();
        let prefill_stages = profile.stage_snapshot();
        let mut generated = Vec::new();
        let mut samples = Vec::new();
        let mut hashes = Vec::new();
        for i in 0..count {
            hashes.push(
                step.logits
                    .iter()
                    .fold(0u64, |acc, x| acc.rotate_left(7) ^ u64::from(x.to_bits())),
            );
            let next = step
                .logits
                .iter()
                .enumerate()
                .filter(|(i, _)| mask[*i])
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap()
                .0 as u32;
            generated.push(next);
            if i + 1 < count {
                let start = Instant::now();
                step = model.forward_token(next, &mut state)?;
                samples.push(start.elapsed().as_secs_f64());
            }
        }
        let p = profile.finish();
        let decode_stages = p.stages.iter().filter_map(|entry| {
            let before = prefill_stages.iter().find(|before| before.stage == entry.stage);
            let calls = entry.calls - before.map_or(0, |entry| entry.calls);
            (calls > 0).then(|| serde_json::json!({
                "stage": entry.stage, "calls": calls,
                "total_ns": entry.total_ns - before.map_or(0, |entry| entry.total_ns),
                "work_items": entry.work_items - before.map_or(0, |entry| entry.work_items),
                "logical_bytes": entry.logical_bytes - before.map_or(0, |entry| entry.logical_bytes),
            }))
        }).collect::<Vec<_>>();
        let telemetry = state.expert_telemetry();
        let tps = samples.len() as f64 / samples.iter().sum::<f64>();
        eprintln!("pass={pass} batch={batch_size} prefill={prefill_seconds:.3}s decode={tps:.3} tok/s misses={} tokens={generated:?}", telemetry.misses-previous.misses);
        passes.push(serde_json::json!({"pass":pass, "prefill_batch_size":batch_size, "prefill_stages":prefill_stages, "decode_stage_totals":decode_stages, "prefill_seconds":prefill_seconds, "decode_tokens_per_second":tps, "decode_seconds":samples, "tokens":generated, "logit_hashes":hashes, "expert_hits":telemetry.hits-previous.hits, "expert_misses":telemetry.misses-previous.misses, "expert_read_bytes":telemetry.bytes_read-previous.bytes_read, "resident_expert_bytes":telemetry.resident_bytes, "profile":p}));
        state.restore(checkpoint)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"schema_version":1, "prefill_batch_cycle":batches, "prompt_file":args.get(7), "model":"qwen3.6-35b-a3b", "ram_gib":ram,"threads":threads,"prompt_tokens":tokens,"load_seconds":load_seconds,"cached_layers":cached_layers,"resident_lm_head":cached_head,"expert_slots_per_layer":options.expert_slots_per_layer,"passes":passes})
        )?
    );
    Ok(())
}
