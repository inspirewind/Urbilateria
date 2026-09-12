#[path = "support/inference_performance.rs"]
mod performance;

use performance::{
    derive_timing, expert_report, measure_inference, timed, write_report, MeasurementMetadata,
    MemoryPlanReport, SetupTimingReport, StateMetrics, REPORT_SCHEMA_VERSION,
};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use urbilateria::execution::configure_threads;
use urbilateria::generation::{CausalDecoder, GenerationConfig};
use urbilateria::models::deepseek_v4::{
    prompt as deepseek_prompt, runtime::DeepseekRuntimeModel, DeepseekV4Config,
};
use urbilateria::models::glm::runtime::GlmRuntimeModel;
use urbilateria::models::kimi_k3::{
    prompt as kimi_prompt, runtime::KimiK3RuntimeModel, tokenizer::KimiK3Tokenizer, KimiK3Config,
};
use urbilateria::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use urbilateria::tokenizer::{
    render_chat, ByteBpeTokenizer, ChatMessage, ChatRole, ChatTemplateOptions,
};
use urbilateria::GlmConfig;

static REAL_PERFORMANCE_TEST: Mutex<()> = Mutex::new(());

#[derive(Debug)]
struct ScriptedState {
    tokens: Vec<u32>,
    experts: ExpertTelemetry,
}

impl StateMetrics for ScriptedState {
    fn position(&self) -> usize {
        self.tokens.len()
    }

    fn cached_f32_elements(&self) -> usize {
        self.tokens.len() * 2
    }

    fn expert_telemetry(&self) -> &ExpertTelemetry {
        &self.experts
    }
}

#[derive(Debug)]
struct ScriptedModel;

impl CausalDecoder for ScriptedModel {
    type State = ScriptedState;
    type Error = std::convert::Infallible;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        Ok(ScriptedState {
            tokens: Vec::new(),
            experts: ExpertTelemetry::default(),
        })
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        state.tokens.push(token);
        state.experts.misses += 1;
        Ok(vec![1.0, 0.0])
    }
}

#[test]
fn timing_rates_use_only_post_ttft_decode_tokens() {
    let timing = derive_timing(
        1_000_000_000,
        4_000_000_000,
        4_000_000_000,
        4,
        Some(900_000_000),
        Some(1_500_000_000),
    );
    assert_eq!(timing.observed_decode_ns, Some(3_000_000_000));
    assert_eq!(timing.observed_decode_tokens_per_second, Some(1.0));
    assert_eq!(timing.model_decode_tokens_per_second, Some(2.0));
    assert_eq!(timing.end_to_end_tokens_per_second, Some(1.0));

    let single = derive_timing(10, 10, 20, 1, Some(9), None);
    assert_eq!(single.observed_decode_ns, None);
    assert_eq!(single.observed_decode_tokens_per_second, None);
    assert_eq!(single.model_decode_tokens_per_second, None);
}

#[test]
fn expert_hit_rate_preserves_the_no_accesses_case() {
    let empty = expert_report(&ExpertTelemetry::default());
    assert_eq!(empty.accesses, 0);
    assert_eq!(empty.hit_rate, None);

    let measured = expert_report(&ExpertTelemetry {
        hits: 7,
        misses: 3,
        ..ExpertTelemetry::default()
    });
    assert_eq!(measured.accesses, 10);
    assert_eq!(measured.hit_rate, Some(0.7));
}

#[test]
fn model_free_measurement_emits_stable_cross_model_metrics() {
    let model = ScriptedModel;
    let mut state = model.new_state().unwrap();
    let generation = GenerationConfig::greedy(3, Vec::new());
    let report = measure_inference(
        &model,
        &mut state,
        &[1, 1],
        &generation,
        MeasurementMetadata {
            model_family: "scripted",
            scenario: "metric-contract".to_owned(),
            requested_threads: 1,
            expert_slots_per_layer: 0,
            setup: SetupTimingReport::default(),
            memory_plan: MemoryPlanReport {
                resident_bytes: 0,
                expert_cache_bytes: 0,
                attention_state_bytes: 64,
                maximum_expert_bytes: 0,
            },
        },
    )
    .unwrap();

    assert_eq!(report.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(report.workload.prompt_tokens, 2);
    assert_eq!(report.workload.generated_tokens, 3);
    assert_eq!(report.workload.committed_tokens, 4);
    assert!(report.workload.fixed_length_greedy);
    assert_eq!(report.expert_cache.misses, 4);
    assert_eq!(report.attention_state.position_after, 4);
    assert_eq!(report.attention_state.logical_f32_elements_after, 8);
    assert_eq!(report.attention_state.logical_bytes_after, 32);
    assert_eq!(
        report
            .profile
            .stage(urbilateria::profiling::ProfileStage::Decode)
            .unwrap()
            .work_items,
        2
    );
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["schema_version"], REPORT_SCHEMA_VERSION);
    assert!(json["timing"]["engine_ttft_ns"].as_u64().unwrap() > 0);
}

#[test]
#[ignore = "requires the converted GLM-5.2 checkpoint and performs release inference"]
fn glm_52_inference_performance() -> Result<(), Box<dyn Error>> {
    let _serial = real_performance_guard()?;
    let options = PerformanceOptions::load()?;
    configure_threads(options.threads)?;
    let model_dir = required_model_dir("URB_GLM52_DIR")?;
    let context_limit_hint = options.max_new_tokens;

    let ((config, prompt_tokens, allowed_mask), tokenizer_prompt_ns) = timed(|| {
        let config = GlmConfig::load(&model_dir)?;
        let tokenizer = ByteBpeTokenizer::load(&model_dir)?;
        tokenizer.validate_chat_content(&options.prompt)?;
        let rendered = render_chat(
            &[ChatMessage::new(ChatRole::User, &options.prompt)],
            ChatTemplateOptions {
                enable_thinking: false,
                ..ChatTemplateOptions::default()
            },
        );
        let prompt_tokens = tokenizer.encode(&rendered)?;
        let allowed_mask = tokenizer.decodable_token_mask(config.vocab_size);
        Ok::<_, Box<dyn Error>>((config, prompt_tokens, allowed_mask))
    })?;
    let context_limit = prompt_tokens
        .len()
        .checked_add(context_limit_hint)
        .ok_or("GLM performance context overflows usize")?;
    let (requirements, checkpoint_preflight_ns) = timed(|| {
        GlmRuntimeModel::inspect_requirements(&model_dir, context_limit, options.expert_slots)
    })?;
    let (model, model_load_ns) = timed(|| {
        GlmRuntimeModel::load(
            &model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: requirements.resident_bytes,
                expert_cache_budget_bytes: requirements.expert_cache_bytes,
                kv_cache_budget_bytes: requirements.kv_cache_bytes,
                expert_slots_per_layer: options.expert_slots,
                maximum_expert_bytes: requirements.maximum_expert_bytes,
                context_limit,
            },
        )
    })?;
    assert_eq!(model.config().vocab_size, config.vocab_size);
    let (mut state, state_init_ns) = timed(|| model.new_state())?;
    let generation = fixed_length_generation(options.max_new_tokens, allowed_mask)?;
    let report = measure_inference(
        &model,
        &mut state,
        &prompt_tokens,
        &generation,
        options.metadata(
            "glm-5.2",
            SetupTimingReport {
                tokenizer_prompt_ns,
                checkpoint_preflight_ns,
                model_load_ns,
                state_init_ns,
            },
            MemoryPlanReport {
                resident_bytes: requirements.resident_bytes,
                expert_cache_bytes: requirements.expert_cache_bytes,
                attention_state_bytes: requirements.kv_cache_bytes,
                maximum_expert_bytes: requirements.maximum_expert_bytes,
            },
        ),
    )?;
    finish_real_report(&report, &options.output_dir)
}

#[test]
#[ignore = "requires DeepSeek-V4-Flash-0731 and performs release inference"]
fn deepseek_v4_inference_performance() -> Result<(), Box<dyn Error>> {
    let _serial = real_performance_guard()?;
    let options = PerformanceOptions::load()?;
    configure_threads(options.threads)?;
    let model_dir = required_model_dir("URB_DEEPSEEK_V4_DIR")?;

    let ((config, prompt_tokens, allowed_mask), tokenizer_prompt_ns) = timed(|| {
        let config = DeepseekV4Config::load(&model_dir)?;
        let tokenizer = ByteBpeTokenizer::load(&model_dir)?;
        tokenizer.validate_chat_content(&options.prompt)?;
        let rendered = deepseek_prompt::render_chat(
            &[deepseek_prompt::Message::new(
                deepseek_prompt::Role::User,
                &options.prompt,
            )],
            deepseek_prompt::PromptOptions {
                thinking_mode: deepseek_prompt::ThinkingMode::Chat,
                ..deepseek_prompt::PromptOptions::default()
            },
        );
        let prompt_tokens = tokenizer.encode(&rendered)?;
        let allowed_mask = tokenizer.decodable_token_mask(config.vocab_size);
        Ok::<_, Box<dyn Error>>((config, prompt_tokens, allowed_mask))
    })?;
    let context_limit = prompt_tokens
        .len()
        .checked_add(options.max_new_tokens)
        .ok_or("DeepSeek performance context overflows usize")?;
    let (requirements, checkpoint_preflight_ns) = timed(|| {
        DeepseekRuntimeModel::inspect_requirements(&model_dir, context_limit, options.expert_slots)
    })?;
    let (model, model_load_ns) = timed(|| {
        DeepseekRuntimeModel::load(
            &model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: requirements.resident_bytes,
                expert_cache_budget_bytes: requirements.expert_cache_bytes,
                kv_cache_budget_bytes: requirements.kv_cache_bytes,
                expert_slots_per_layer: options.expert_slots,
                maximum_expert_bytes: requirements.maximum_expert_bytes,
                context_limit,
            },
        )
    })?;
    assert_eq!(model.config().vocab_size, config.vocab_size);
    let (mut state, state_init_ns) = timed(|| model.new_state())?;
    let generation = fixed_length_generation(options.max_new_tokens, allowed_mask)?;
    let report = measure_inference(
        &model,
        &mut state,
        &prompt_tokens,
        &generation,
        options.metadata(
            "deepseek-v4-flash-0731",
            SetupTimingReport {
                tokenizer_prompt_ns,
                checkpoint_preflight_ns,
                model_load_ns,
                state_init_ns,
            },
            MemoryPlanReport {
                resident_bytes: requirements.resident_bytes,
                expert_cache_bytes: requirements.expert_cache_bytes,
                attention_state_bytes: requirements.kv_cache_bytes,
                maximum_expert_bytes: requirements.maximum_expert_bytes,
            },
        ),
    )?;
    finish_real_report(&report, &options.output_dir)
}

#[test]
#[ignore = "requires all 96 official Kimi-K3 shards and performs release inference"]
fn kimi_k3_inference_performance() -> Result<(), Box<dyn Error>> {
    let _serial = real_performance_guard()?;
    let options = PerformanceOptions::load()?;
    configure_threads(options.threads)?;
    let model_dir = required_model_dir("KIMI_K3_MODEL_DIR")?;

    let ((config, prompt_tokens, allowed_mask), tokenizer_prompt_ns) = timed(|| {
        let config = KimiK3Config::load(&model_dir)?;
        let tokenizer = KimiK3Tokenizer::load(&model_dir)?;
        let prompt_tokens = tokenizer.encode_chat(
            &[kimi_prompt::Message::new(
                kimi_prompt::Role::User,
                &options.prompt,
            )],
            kimi_prompt::PromptOptions {
                thinking: false,
                thinking_effort: None,
                ..kimi_prompt::PromptOptions::default()
            },
        )?;
        let allowed_mask = tokenizer.decodable_token_mask(config.text_config.vocab_size);
        Ok::<_, Box<dyn Error>>((config, prompt_tokens, allowed_mask))
    })?;
    let context_limit = prompt_tokens
        .len()
        .checked_add(options.max_new_tokens)
        .ok_or("Kimi-K3 performance context overflows usize")?;
    let (requirements, checkpoint_preflight_ns) = timed(|| {
        KimiK3RuntimeModel::inspect_requirements(&model_dir, context_limit, options.expert_slots)
    })?;
    let (model, model_load_ns) = timed(|| {
        KimiK3RuntimeModel::load(
            &model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: requirements.resident_bytes,
                expert_cache_budget_bytes: requirements.expert_cache_bytes,
                kv_cache_budget_bytes: requirements.mla_cache_bytes,
                expert_slots_per_layer: options.expert_slots,
                maximum_expert_bytes: requirements.routed_expert_bytes,
                context_limit,
            },
        )
    })?;
    assert_eq!(
        model.config().text_config.vocab_size,
        config.text_config.vocab_size
    );
    let (mut state, state_init_ns) = timed(|| model.new_state())?;
    let generation = fixed_length_generation(options.max_new_tokens, allowed_mask)?;
    let attention_state_bytes = requirements
        .kda_state_bytes
        .checked_add(requirements.mla_cache_bytes)
        .ok_or("Kimi-K3 planned attention state bytes overflow u64")?;
    let report = measure_inference(
        &model,
        &mut state,
        &prompt_tokens,
        &generation,
        options.metadata(
            "kimi-k3",
            SetupTimingReport {
                tokenizer_prompt_ns,
                checkpoint_preflight_ns,
                model_load_ns,
                state_init_ns,
            },
            MemoryPlanReport {
                resident_bytes: requirements.resident_bytes,
                expert_cache_bytes: requirements.expert_cache_bytes,
                attention_state_bytes,
                maximum_expert_bytes: requirements.routed_expert_bytes,
            },
        ),
    )?;
    finish_real_report(&report, &options.output_dir)
}

#[derive(Debug)]
struct PerformanceOptions {
    prompt: String,
    scenario: String,
    threads: usize,
    expert_slots: usize,
    max_new_tokens: usize,
    output_dir: PathBuf,
}

impl PerformanceOptions {
    fn load() -> Result<Self, Box<dyn Error>> {
        if cfg!(debug_assertions) {
            return Err("real inference performance tests must be run with --release".into());
        }
        let prompt = std::env::var("URB_PERF_PROMPT").unwrap_or_else(|_| "Hello".to_owned());
        if prompt.is_empty() {
            return Err("URB_PERF_PROMPT must not be empty".into());
        }
        let default_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let output_dir = std::env::var_os("URB_PERF_OUTPUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/perf"));
        Ok(Self {
            prompt,
            scenario: std::env::var("URB_PERF_SCENARIO")
                .unwrap_or_else(|_| "hello-no-thinking".to_owned()),
            threads: env_usize("URB_PERF_THREADS", default_threads)?,
            expert_slots: env_usize("URB_PERF_EXPERT_SLOTS", 1)?,
            max_new_tokens: env_usize("URB_PERF_NEW_TOKENS", 4)?,
            output_dir,
        })
    }

    fn metadata(
        &self,
        model_family: &'static str,
        setup: SetupTimingReport,
        memory_plan: MemoryPlanReport,
    ) -> MeasurementMetadata {
        MeasurementMetadata {
            model_family,
            scenario: self.scenario.clone(),
            requested_threads: self.threads,
            expert_slots_per_layer: self.expert_slots,
            setup,
            memory_plan,
        }
    }
}

fn fixed_length_generation(
    max_new_tokens: usize,
    allowed_mask: Vec<bool>,
) -> Result<GenerationConfig, Box<dyn Error>> {
    let mut generation = GenerationConfig::greedy(max_new_tokens, Vec::new());
    generation.allowed_token_mask = Some(allowed_mask);
    generation.validate()?;
    Ok(generation)
}

fn finish_real_report(
    report: &performance::InferencePerformanceReport,
    output_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let path = write_report(report, output_dir)?;
    eprintln!("{}", serde_json::to_string_pretty(report)?);
    eprintln!("performance report: {}", path.display());
    Ok(())
}

fn real_performance_guard() -> Result<std::sync::MutexGuard<'static, ()>, Box<dyn Error>> {
    REAL_PERFORMANCE_TEST
        .lock()
        .map_err(|_| "real performance test serialization lock is poisoned".into())
}

fn required_model_dir(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("set {name} to run this real performance test").into())
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    let value = value
        .to_str()
        .ok_or_else(|| format!("{name} is not valid UTF-8"))?;
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}={value:?}: {error}"))?;
    if parsed == 0 && name != "URB_PERF_EXPERT_SLOTS" {
        return Err(format!("{name} must be greater than zero").into());
    }
    Ok(parsed)
}
