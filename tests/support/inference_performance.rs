use serde::Serialize;
use std::convert::Infallible;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use urbilateria::generation::{
    try_generate_with_state, CausalDecoder, GenerationConfig, GenerationOutput,
};
use urbilateria::models::deepseek_v4::runtime::DeepseekRuntimeState;
use urbilateria::models::glm::runtime::GlmRuntimeState;
use urbilateria::models::kimi_k3::runtime::KimiK3RuntimeState;
use urbilateria::profiling::{ProfileReport, ProfileSession, ProfileStage};
use urbilateria::runtime::ExpertTelemetry;

pub const REPORT_SCHEMA_VERSION: u32 = 1;

pub trait StateMetrics {
    fn position(&self) -> usize;
    fn cached_f32_elements(&self) -> usize;
    fn expert_telemetry(&self) -> &ExpertTelemetry;
}

impl StateMetrics for GlmRuntimeState {
    fn position(&self) -> usize {
        self.position()
    }

    fn cached_f32_elements(&self) -> usize {
        self.cached_f32_elements()
    }

    fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.expert_telemetry()
    }
}

impl StateMetrics for DeepseekRuntimeState {
    fn position(&self) -> usize {
        self.position()
    }

    fn cached_f32_elements(&self) -> usize {
        self.cached_f32_elements()
    }

    fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.expert_telemetry()
    }
}

impl StateMetrics for KimiK3RuntimeState {
    fn position(&self) -> usize {
        self.position()
    }

    fn cached_f32_elements(&self) -> usize {
        self.cached_f32_elements()
    }

    fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.expert_telemetry()
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SetupTimingReport {
    pub tokenizer_prompt_ns: u64,
    pub checkpoint_preflight_ns: u64,
    pub model_load_ns: u64,
    pub state_init_ns: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MemoryPlanReport {
    pub resident_bytes: u64,
    pub expert_cache_bytes: u64,
    pub attention_state_bytes: u64,
    pub maximum_expert_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct MeasurementMetadata {
    pub model_family: &'static str,
    pub scenario: String,
    pub requested_threads: usize,
    pub expert_slots_per_layer: usize,
    pub setup: SetupTimingReport,
    pub memory_plan: MemoryPlanReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkloadReport {
    pub scenario: String,
    pub prompt_tokens: usize,
    pub requested_new_tokens: usize,
    pub generated_tokens: usize,
    pub committed_tokens: usize,
    pub requested_threads: usize,
    pub expert_slots_per_layer: usize,
    pub fixed_length_greedy: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TimingReport {
    pub engine_ttft_ns: u64,
    pub profiled_ttft_ns: Option<u64>,
    pub generation_total_ns: u64,
    pub observed_decode_ns: Option<u64>,
    pub model_decode_forward_ns: Option<u64>,
    pub observed_decode_tokens_per_second: Option<f64>,
    pub model_decode_tokens_per_second: Option<f64>,
    pub end_to_end_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ExpertCacheReport {
    pub hits: u64,
    pub misses: u64,
    pub accesses: u64,
    pub hit_rate: Option<f64>,
    pub evictions: u64,
    pub bytes_read: u64,
    pub resident_experts: usize,
    pub resident_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct AttentionStateReport {
    pub position_before: usize,
    pub position_after: usize,
    pub logical_f32_elements_before: usize,
    pub logical_f32_elements_after: usize,
    pub logical_f32_elements_delta: usize,
    pub logical_bytes_before: u64,
    pub logical_bytes_after: u64,
    pub logical_bytes_delta: u64,
    /// Runtime planning bound. For Kimi-K3 this includes fixed KDA state and dynamic MLA cache.
    pub planned_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PlatformReport {
    pub os: &'static str,
    pub architecture: &'static str,
    pub release_build: bool,
    pub available_parallelism: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferencePerformanceReport {
    pub schema_version: u32,
    pub model_family: &'static str,
    pub workload: WorkloadReport,
    pub setup: SetupTimingReport,
    pub timing: TimingReport,
    pub expert_cache: ExpertCacheReport,
    pub attention_state: AttentionStateReport,
    pub memory_plan: MemoryPlanReport,
    pub platform: PlatformReport,
    pub profile: ProfileReport,
}

#[derive(Debug, Clone, Copy)]
struct StateSnapshot {
    position: usize,
    cached_f32_elements: usize,
    expert: ExpertTelemetrySnapshot,
}

impl StateSnapshot {
    fn capture(state: &impl StateMetrics) -> Self {
        Self {
            position: state.position(),
            cached_f32_elements: state.cached_f32_elements(),
            expert: ExpertTelemetrySnapshot::from(state.expert_telemetry()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpertTelemetrySnapshot {
    hits: u64,
    misses: u64,
    evictions: u64,
    bytes_read: u64,
    resident_experts: usize,
    resident_bytes: u64,
}

impl From<&ExpertTelemetry> for ExpertTelemetrySnapshot {
    fn from(telemetry: &ExpertTelemetry) -> Self {
        Self {
            hits: telemetry.hits,
            misses: telemetry.misses,
            evictions: telemetry.evictions,
            bytes_read: telemetry.bytes_read,
            resident_experts: telemetry.resident_experts,
            resident_bytes: telemetry.resident_bytes,
        }
    }
}

pub fn measure_inference<M>(
    model: &M,
    state: &mut M::State,
    prompt: &[u32],
    generation: &GenerationConfig,
    metadata: MeasurementMetadata,
) -> Result<InferencePerformanceReport, Box<dyn Error>>
where
    M: CausalDecoder,
    M::State: StateMetrics,
    M::Error: 'static,
{
    if prompt.is_empty() {
        return Err("performance prompt must not be empty".into());
    }
    generation.validate()?;

    let before = StateSnapshot::capture(state);
    let profile = ProfileSession::start_with_threads(Some(metadata.requested_threads));
    let started = Instant::now();
    let mut first_token_at = None;
    let mut last_token_at = None;
    let output = try_generate_with_state(
        model,
        prompt,
        generation,
        state,
        |_| -> Result<(), Infallible> {
            let elapsed = started.elapsed();
            first_token_at.get_or_insert(elapsed);
            last_token_at = Some(elapsed);
            Ok(())
        },
    )?;
    let generation_total = started.elapsed();
    let profile = profile.finish();
    let after = StateSnapshot::capture(state);

    build_report(
        prompt.len(),
        generation,
        output,
        metadata,
        before,
        after,
        first_token_at.ok_or("generation returned without producing a first token")?,
        last_token_at.ok_or("generation returned without producing a last token")?,
        generation_total,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    prompt_tokens: usize,
    generation: &GenerationConfig,
    output: GenerationOutput,
    metadata: MeasurementMetadata,
    before: StateSnapshot,
    after: StateSnapshot,
    first_token_at: Duration,
    last_token_at: Duration,
    generation_total: Duration,
    profile: ProfileReport,
) -> Result<InferencePerformanceReport, Box<dyn Error>> {
    let generated_tokens = output.generated_tokens.len();
    let expected_committed = prompt_tokens
        .checked_add(generated_tokens.saturating_sub(1))
        .ok_or("committed token count overflows usize")?;
    let committed_tokens = after.position.saturating_sub(before.position);
    if committed_tokens != expected_committed {
        return Err(format!(
            "runtime committed {committed_tokens} tokens, expected {expected_committed} from prompt={prompt_tokens} and generated={generated_tokens}"
        )
        .into());
    }
    if after.cached_f32_elements < before.cached_f32_elements {
        return Err(format!(
            "logical attention state shrank from {} to {} F32 elements during generation",
            before.cached_f32_elements, after.cached_f32_elements
        )
        .into());
    }
    let logical_attention_bytes = f32_bytes(after.cached_f32_elements);
    if logical_attention_bytes > metadata.memory_plan.attention_state_bytes {
        return Err(format!(
            "logical attention state uses {logical_attention_bytes} bytes, exceeding the planned {} bytes",
            metadata.memory_plan.attention_state_bytes
        )
        .into());
    }

    let expected_decode_forwards = generated_tokens.saturating_sub(1) as u64;
    let decode_entry = profile.stage(ProfileStage::Decode);
    let decode_work_items = decode_entry.map_or(0, |entry| entry.work_items);
    if decode_work_items != expected_decode_forwards {
        return Err(format!(
            "profile recorded {decode_work_items} decode work items, expected {expected_decode_forwards}"
        )
        .into());
    }
    let profiled_ttft_ns = profile
        .stage(ProfileStage::TimeToFirstToken)
        .map(|entry| entry.total_ns);
    if profiled_ttft_ns.is_none() {
        return Err("profile did not record generation.time_to_first_token".into());
    }

    let timing = derive_timing(
        duration_ns(first_token_at),
        duration_ns(last_token_at),
        duration_ns(generation_total),
        generated_tokens,
        profiled_ttft_ns,
        decode_entry.map(|entry| entry.total_ns),
    );
    let expert_cache = expert_delta(before.expert, after.expert);
    let attention_state = attention_delta(
        before.position,
        after.position,
        before.cached_f32_elements,
        after.cached_f32_elements,
        metadata.memory_plan.attention_state_bytes,
    );

    Ok(InferencePerformanceReport {
        schema_version: REPORT_SCHEMA_VERSION,
        model_family: metadata.model_family,
        workload: WorkloadReport {
            scenario: metadata.scenario,
            prompt_tokens,
            requested_new_tokens: generation.max_new_tokens,
            generated_tokens,
            committed_tokens,
            requested_threads: metadata.requested_threads,
            expert_slots_per_layer: metadata.expert_slots_per_layer,
            fixed_length_greedy: generation.eos_token_ids.is_empty(),
        },
        setup: metadata.setup,
        timing,
        expert_cache,
        attention_state,
        memory_plan: metadata.memory_plan,
        platform: PlatformReport {
            os: std::env::consts::OS,
            architecture: std::env::consts::ARCH,
            release_build: !cfg!(debug_assertions),
            available_parallelism: std::thread::available_parallelism().ok().map(usize::from),
        },
        profile,
    })
}

pub fn derive_timing(
    first_token_ns: u64,
    last_token_ns: u64,
    generation_total_ns: u64,
    generated_tokens: usize,
    profiled_ttft_ns: Option<u64>,
    model_decode_forward_ns: Option<u64>,
) -> TimingReport {
    let decode_tokens = generated_tokens.saturating_sub(1) as u64;
    let observed_decode_ns =
        (decode_tokens != 0).then(|| last_token_ns.saturating_sub(first_token_ns));
    TimingReport {
        engine_ttft_ns: first_token_ns,
        profiled_ttft_ns,
        generation_total_ns,
        observed_decode_ns,
        model_decode_forward_ns: (decode_tokens != 0)
            .then_some(model_decode_forward_ns)
            .flatten(),
        observed_decode_tokens_per_second: observed_decode_ns
            .and_then(|nanoseconds| rate_per_second(decode_tokens, nanoseconds)),
        model_decode_tokens_per_second: model_decode_forward_ns
            .filter(|_| decode_tokens != 0)
            .and_then(|nanoseconds| rate_per_second(decode_tokens, nanoseconds)),
        end_to_end_tokens_per_second: rate_per_second(generated_tokens as u64, generation_total_ns),
    }
}

pub fn expert_report(telemetry: &ExpertTelemetry) -> ExpertCacheReport {
    ExpertCacheReport {
        hits: telemetry.hits,
        misses: telemetry.misses,
        accesses: telemetry.accesses(),
        hit_rate: telemetry.hit_rate(),
        evictions: telemetry.evictions,
        bytes_read: telemetry.bytes_read,
        resident_experts: telemetry.resident_experts,
        resident_bytes: telemetry.resident_bytes,
    }
}

fn expert_delta(
    before: ExpertTelemetrySnapshot,
    after: ExpertTelemetrySnapshot,
) -> ExpertCacheReport {
    expert_report(&ExpertTelemetry {
        hits: after.hits.saturating_sub(before.hits),
        misses: after.misses.saturating_sub(before.misses),
        evictions: after.evictions.saturating_sub(before.evictions),
        bytes_read: after.bytes_read.saturating_sub(before.bytes_read),
        resident_experts: after.resident_experts,
        resident_bytes: after.resident_bytes,
    })
}

fn attention_delta(
    position_before: usize,
    position_after: usize,
    elements_before: usize,
    elements_after: usize,
    planned_bytes: u64,
) -> AttentionStateReport {
    AttentionStateReport {
        position_before,
        position_after,
        logical_f32_elements_before: elements_before,
        logical_f32_elements_after: elements_after,
        logical_f32_elements_delta: elements_after.saturating_sub(elements_before),
        logical_bytes_before: f32_bytes(elements_before),
        logical_bytes_after: f32_bytes(elements_after),
        logical_bytes_delta: f32_bytes(elements_after.saturating_sub(elements_before)),
        planned_bytes,
    }
}

pub fn timed<T, E>(operation: impl FnOnce() -> Result<T, E>) -> Result<(T, u64), E> {
    let started = Instant::now();
    operation().map(|value| (value, duration_ns(started.elapsed())))
}

pub fn write_report(
    report: &InferencePerformanceReport,
    output_dir: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    let stem = format!(
        "{}-{}-t{}-slots{}",
        sanitize(report.model_family),
        sanitize(&report.workload.scenario),
        report.workload.requested_threads,
        report.workload.expert_slots_per_layer
    );
    let path = output_dir.join(format!("{stem}.json"));
    let temporary = output_dir.join(format!(".{stem}.json.tmp"));
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&temporary, &path)?;
    Ok(path)
}

fn sanitize(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('-').to_owned()
}

fn rate_per_second(items: u64, nanoseconds: u64) -> Option<f64> {
    (items != 0 && nanoseconds != 0).then(|| items as f64 * 1_000_000_000.0 / nanoseconds as f64)
}

fn f32_bytes(elements: usize) -> u64 {
    u64::try_from(elements)
        .unwrap_or(u64::MAX)
        .saturating_mul(4)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
