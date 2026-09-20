//! Low-overhead, opt-in wall-clock profiling for inference.
//!
//! A [`ProfileSession`] activates spans on the current inference thread. Disabled spans only do a
//! thread-local lookup; enabled spans aggregate into a stable, serializable report. Stage times are
//! inclusive, so nested rows intentionally do not add up to the session wall time. Sessions may be
//! nested and finished in any order; a span dropped after its owning session ends is ignored.
//! Reports never create the CPU pool merely to inspect it, so `worker_threads` is `null` until the
//! pool has actually been initialized.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod resources;
mod trace;
pub use resources::{ResourceSample, ResourceSummary, SystemResourceReport};

thread_local! {
    static ACTIVE: RefCell<Vec<Arc<Collector>>> = const { RefCell::new(Vec::new()) };
}

/// Captured profiling activation that can be installed temporarily on a worker thread.
///
/// Rayon workers have independent thread-local state. Explicit propagation keeps nested kernel
/// spans attached to the caller's session when higher-level work is pipelined across workers.
#[derive(Clone, Default)]
pub(crate) struct ProfileContext {
    collector: Option<Arc<Collector>>,
}

pub(crate) fn capture_context() -> ProfileContext {
    ProfileContext {
        collector: ACTIVE.with(|active| active.borrow().last().cloned()),
    }
}

impl ProfileContext {
    pub(crate) fn enter<R>(&self, operation: impl FnOnce() -> R) -> R {
        let Some(collector) = self.collector.as_ref() else {
            return operation();
        };
        ACTIVE.with(|active| active.borrow_mut().push(Arc::clone(collector)));
        let _activation = ProfileActivation {
            collector: Arc::clone(collector),
        };
        operation()
    }
}

struct ProfileActivation {
    collector: Arc<Collector>,
}

impl Drop for ProfileActivation {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if let Some(position) = active
                .iter()
                .rposition(|collector| Arc::ptr_eq(collector, &self.collector))
            {
                active.remove(position);
            }
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfileStage {
    GenerateTotal,
    ConfigLoad,
    TokenizerPrompt,
    CheckpointPreflight,
    ModelLoad,
    StateInit,
    TimeToFirstToken,
    Prefill,
    Decode,
    GlmToken,
    GlmEmbedding,
    GlmLayer,
    GlmAttention,
    GlmFeedForward,
    GlmExpertLoad,
    GlmExpertCompute,
    GlmFinalNorm,
    GlmLmHead,
    DeepseekToken,
    DeepseekActivationQuantization,
    DeepseekStateCheckpoint,
    DeepseekEmbeddingRead,
    DeepseekLayerLoad,
    DeepseekLayer,
    DeepseekAttention,
    DeepseekMoe,
    DeepseekExpertLoad,
    DeepseekExpertCompute,
    DeepseekFinalization,
    DeepseekLmHead,
    Hy4LayerLoad,
    Hy4Layer,
    Hy4Attention,
    Hy4ActivationQuantization,
    Hy4Moe,
    Hy4ExpertLoad,
    Hy4ExpertCompute,
    Hy4LmHead,
    StreamedMatrixReadDecode,
    StreamedMatrixCompute,
    MatvecF32,
    MatvecBf16,
    MatvecInt8,
    MatvecInt4,
    MatvecMxFp8,
    MatvecMxFp4,
    TransposeMatvecF32,
    TransposeMatvecInt8,
    TransposeMatvecInt4,
    TransposeMatvecMxFp8,
    TransposeMatvecMxFp4,
}

const ALL_STAGES: &[ProfileStage] = &[
    ProfileStage::GenerateTotal,
    ProfileStage::ConfigLoad,
    ProfileStage::TokenizerPrompt,
    ProfileStage::CheckpointPreflight,
    ProfileStage::ModelLoad,
    ProfileStage::StateInit,
    ProfileStage::TimeToFirstToken,
    ProfileStage::Prefill,
    ProfileStage::Decode,
    ProfileStage::GlmToken,
    ProfileStage::GlmEmbedding,
    ProfileStage::GlmLayer,
    ProfileStage::GlmAttention,
    ProfileStage::GlmFeedForward,
    ProfileStage::GlmExpertLoad,
    ProfileStage::GlmExpertCompute,
    ProfileStage::GlmFinalNorm,
    ProfileStage::GlmLmHead,
    ProfileStage::DeepseekToken,
    ProfileStage::DeepseekActivationQuantization,
    ProfileStage::DeepseekStateCheckpoint,
    ProfileStage::DeepseekEmbeddingRead,
    ProfileStage::DeepseekLayerLoad,
    ProfileStage::DeepseekLayer,
    ProfileStage::DeepseekAttention,
    ProfileStage::DeepseekMoe,
    ProfileStage::DeepseekExpertLoad,
    ProfileStage::DeepseekExpertCompute,
    ProfileStage::DeepseekFinalization,
    ProfileStage::DeepseekLmHead,
    ProfileStage::Hy4LayerLoad,
    ProfileStage::Hy4Layer,
    ProfileStage::Hy4Attention,
    ProfileStage::Hy4ActivationQuantization,
    ProfileStage::Hy4Moe,
    ProfileStage::Hy4ExpertLoad,
    ProfileStage::Hy4ExpertCompute,
    ProfileStage::Hy4LmHead,
    ProfileStage::StreamedMatrixReadDecode,
    ProfileStage::StreamedMatrixCompute,
    ProfileStage::MatvecF32,
    ProfileStage::MatvecBf16,
    ProfileStage::MatvecInt8,
    ProfileStage::MatvecInt4,
    ProfileStage::MatvecMxFp8,
    ProfileStage::MatvecMxFp4,
    ProfileStage::TransposeMatvecF32,
    ProfileStage::TransposeMatvecInt8,
    ProfileStage::TransposeMatvecInt4,
    ProfileStage::TransposeMatvecMxFp8,
    ProfileStage::TransposeMatvecMxFp4,
];

impl ProfileStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GenerateTotal => "generate.total",
            Self::ConfigLoad => "setup.config_load",
            Self::TokenizerPrompt => "setup.tokenizer_prompt",
            Self::CheckpointPreflight => "setup.checkpoint_preflight",
            Self::ModelLoad => "setup.model_load",
            Self::StateInit => "setup.state_init",
            Self::TimeToFirstToken => "generation.time_to_first_token",
            Self::Prefill => "generation.prefill_forward",
            Self::Decode => "generation.decode_forward",
            Self::GlmToken => "glm.token",
            Self::GlmEmbedding => "glm.embedding",
            Self::GlmLayer => "glm.layer",
            Self::GlmAttention => "glm.attention",
            Self::GlmFeedForward => "glm.feed_forward",
            Self::GlmExpertLoad => "glm.expert.load",
            Self::GlmExpertCompute => "glm.expert.compute",
            Self::GlmFinalNorm => "glm.final_norm",
            Self::GlmLmHead => "glm.lm_head",
            Self::DeepseekToken => "deepseek.token",
            Self::DeepseekActivationQuantization => "deepseek.activation_quantization",
            Self::DeepseekStateCheckpoint => "deepseek.state_checkpoint",
            Self::DeepseekEmbeddingRead => "deepseek.embedding.read",
            Self::DeepseekLayerLoad => "deepseek.layer.load",
            Self::DeepseekLayer => "deepseek.layer",
            Self::DeepseekAttention => "deepseek.attention",
            Self::DeepseekMoe => "deepseek.moe",
            Self::DeepseekExpertLoad => "deepseek.expert.load",
            Self::DeepseekExpertCompute => "deepseek.expert.compute",
            Self::DeepseekFinalization => "deepseek.finalization",
            Self::DeepseekLmHead => "deepseek.lm_head",
            Self::Hy4LayerLoad => "hy4.layer.load",
            Self::Hy4Layer => "hy4.layer",
            Self::Hy4Attention => "hy4.attention",
            Self::Hy4ActivationQuantization => "hy4.activation_quantization",
            Self::Hy4Moe => "hy4.moe",
            Self::Hy4ExpertLoad => "hy4.expert.load",
            Self::Hy4ExpertCompute => "hy4.expert.compute",
            Self::Hy4LmHead => "hy4.lm_head",
            Self::StreamedMatrixReadDecode => "streamed_matrix.read_decode",
            Self::StreamedMatrixCompute => "streamed_matrix.compute",
            Self::MatvecF32 => "kernel.matvec.f32",
            Self::MatvecBf16 => "kernel.matvec.bf16",
            Self::MatvecInt8 => "kernel.matvec.int8",
            Self::MatvecInt4 => "kernel.matvec.int4",
            Self::MatvecMxFp8 => "kernel.matvec.mxfp8",
            Self::MatvecMxFp4 => "kernel.matvec.mxfp4",
            Self::TransposeMatvecF32 => "kernel.transpose_matvec.f32",
            Self::TransposeMatvecInt8 => "kernel.transpose_matvec.int8",
            Self::TransposeMatvecInt4 => "kernel.transpose_matvec.int4",
            Self::TransposeMatvecMxFp8 => "kernel.transpose_matvec.mxfp8",
            Self::TransposeMatvecMxFp4 => "kernel.transpose_matvec.mxfp4",
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfileEntry {
    pub stage: &'static str,
    pub calls: u64,
    pub total_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    /// Scalar multiply-accumulates for matrix stages; zero for non-kernel stages.
    pub work_items: u64,
    /// Logical payload bytes associated with the stage; this is not physical block-device I/O.
    pub logical_bytes: u64,
}

impl ProfileEntry {
    pub fn average_ns(&self) -> u64 {
        self.total_ns / self.calls.max(1)
    }

    pub fn work_items_per_second(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.work_items as f64 * 1_000_000_000.0 / self.total_ns as f64
        }
    }

    pub fn logical_bytes_per_second(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.logical_bytes as f64 * 1_000_000_000.0 / self.total_ns as f64
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfileReport {
    pub schema_version: u32,
    pub wall_time_ns: u64,
    pub requested_threads: Option<usize>,
    /// Effective workers when the CPU pool was initialized during the session.
    pub worker_threads: Option<usize>,
    /// Session-aligned Linux process resource samples. `None` on unsupported platforms or when
    /// procfs is unavailable.
    pub resources: Option<SystemResourceReport>,
    /// Inclusive stage timings in a stable schema order. Nested totals may overlap.
    pub stages: Vec<ProfileEntry>,
    #[serde(skip)]
    trace: Option<trace::TraceSnapshot>,
}

impl ProfileReport {
    pub fn stage(&self, stage: ProfileStage) -> Option<&ProfileEntry> {
        self.stages
            .iter()
            .find(|entry| entry.stage == stage.as_str())
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "profile: wall={:.3} ms, CPU workers={}{} (inclusive stages; nested totals overlap)",
            self.wall_time_ns as f64 / 1_000_000.0,
            self.worker_threads
                .map(|threads| threads.to_string())
                .unwrap_or_else(|| "not initialized".to_owned()),
            self.requested_threads
                .map(|threads| format!(", requested={threads}"))
                .unwrap_or_default()
        );
        if let Some(resources) = &self.resources {
            let summary = &resources.summary;
            let _ = writeln!(
                output,
                "resources: CPU avg={:.2} cores ({}) peak={:.2} cores; RSS start/end/peak={:.1}/{:.1}/{:.1} MiB; storage read/write={:.1}/{:.1} MiB ({:.1}/{:.1} MiB/s avg)",
                summary.average_cpu_cores,
                summary
                    .average_machine_cpu_percent
                    .map(|percent| format!("{percent:.1}% machine"))
                    .unwrap_or_else(|| "machine share unavailable".to_owned()),
                summary.peak_interval_cpu_cores,
                bytes_to_mib(summary.start_rss_bytes),
                bytes_to_mib(summary.end_rss_bytes),
                bytes_to_mib(summary.peak_rss_bytes),
                bytes_to_mib(summary.storage_read_bytes),
                bytes_to_mib(summary.storage_write_bytes),
                bytes_to_mib(summary.average_storage_read_bytes_per_second as u64),
                bytes_to_mib(summary.average_storage_write_bytes_per_second as u64),
            );
        }
        let _ = writeln!(
            output,
            "  {:<32} {:>8} {:>12} {:>12} {:>10} {:>10}",
            "stage", "calls", "total ms", "avg ms", "GMAC/s", "MiB/s"
        );
        let mut entries = self.stages.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            right
                .total_ns
                .cmp(&left.total_ns)
                .then_with(|| left.stage.cmp(right.stage))
        });
        for entry in entries {
            let rate = if entry.work_items == 0 {
                "-".to_owned()
            } else {
                format!("{:.3}", entry.work_items_per_second() / 1_000_000_000.0)
            };
            let byte_rate = if entry.logical_bytes == 0 {
                "-".to_owned()
            } else {
                format!(
                    "{:.1}",
                    entry.logical_bytes_per_second() / (1024.0 * 1024.0)
                )
            };
            let _ = writeln!(
                output,
                "  {:<32} {:>8} {:>12.3} {:>12.3} {:>10} {:>10}",
                entry.stage,
                entry.calls,
                entry.total_ns as f64 / 1_000_000.0,
                entry.average_ns() as f64 / 1_000_000.0,
                rate,
                byte_rate
            );
        }
        output
    }

    /// Serializes the optional per-span timeline as Chrome Trace Event JSON for Perfetto.
    pub fn chrome_trace_json_pretty(&self) -> serde_json::Result<Option<Vec<u8>>> {
        self.trace
            .as_ref()
            .map(|trace| trace.to_json_pretty(self.resources.as_ref()))
            .transpose()
    }

    pub fn trace_event_count(&self) -> usize {
        self.trace
            .as_ref()
            .map_or(0, trace::TraceSnapshot::event_count)
    }

    pub fn trace_dropped_events(&self) -> u64 {
        self.trace
            .as_ref()
            .map_or(0, trace::TraceSnapshot::dropped_events)
    }
}

/// Activates profiling until [`finish`](Self::finish) or drop on the creating thread.
pub struct ProfileSession {
    collector: Arc<Collector>,
    started: Instant,
    requested_threads: Option<usize>,
    active: bool,
    resources: Option<resources::ResourceMonitor>,
    _not_send: PhantomData<Rc<()>>,
}

impl ProfileSession {
    pub fn start() -> Self {
        Self::start_with_threads(None)
    }

    pub fn start_with_threads(requested_threads: Option<usize>) -> Self {
        Self::start_with_threads_and_trace(requested_threads, false)
    }

    pub fn start_with_threads_and_trace(
        requested_threads: Option<usize>,
        trace_enabled: bool,
    ) -> Self {
        let started = Instant::now();
        let collector = Arc::new(Collector::new(started, trace_enabled));
        ACTIVE.with(|active| active.borrow_mut().push(Arc::clone(&collector)));
        Self {
            collector,
            started,
            requested_threads,
            active: true,
            resources: resources::ResourceMonitor::start(),
            _not_send: PhantomData,
        }
    }

    pub fn finish(mut self) -> ProfileReport {
        let elapsed = self.started.elapsed();
        self.collector.close();
        self.deactivate();
        let resources = self.resources.take().map(|monitor| monitor.finish(elapsed));
        self.collector
            .snapshot(elapsed, self.requested_threads, resources)
    }

    fn deactivate(&mut self) {
        if self.active {
            ACTIVE.with(|active| {
                let mut active = active.borrow_mut();
                if let Some(position) = active
                    .iter()
                    .rposition(|collector| Arc::ptr_eq(collector, &self.collector))
                {
                    active.remove(position);
                }
            });
            self.active = false;
        }
    }
}

impl Drop for ProfileSession {
    fn drop(&mut self) {
        self.collector.close();
        self.deactivate();
    }
}

pub struct ProfileSpan {
    collector: Option<Arc<Collector>>,
    stage: ProfileStage,
    started: Option<Instant>,
    work_items: u64,
    logical_bytes: u64,
    trace_thread_id: Option<u64>,
    trace_metadata: TraceMetadata,
}

impl ProfileSpan {
    /// Adds logical payload bytes discovered only after an operation completes.
    pub fn add_logical_bytes(&mut self, bytes: u64) {
        self.logical_bytes = self.logical_bytes.saturating_add(bytes);
    }

    pub fn set_token(&mut self, position: usize, token_id: usize) {
        self.set_token_position(position);
        self.set_token_id(token_id);
    }

    pub fn set_token_position(&mut self, position: usize) {
        if self.trace_thread_id.is_some() {
            self.trace_metadata.token_position = Some(saturating_u64(position));
        }
    }

    pub fn set_token_id(&mut self, token_id: usize) {
        if self.trace_thread_id.is_some() {
            self.trace_metadata.token_id = Some(saturating_u64(token_id));
        }
    }

    pub fn set_layer_id(&mut self, layer_id: usize) {
        if self.trace_thread_id.is_some() {
            self.trace_metadata.layer_id = Some(saturating_u64(layer_id));
        }
    }

    pub fn set_expert_id(&mut self, expert_id: usize) {
        if self.trace_thread_id.is_some() {
            self.trace_metadata.expert_id = Some(saturating_u64(expert_id));
        }
    }

    pub fn set_cache_hit(&mut self, cache_hit: bool) {
        if self.trace_thread_id.is_some() {
            self.trace_metadata.cache_hit = Some(cache_hit);
        }
    }

    pub fn set_flow_id(&mut self, flow_id: u64) {
        if self.trace_thread_id.is_some() {
            self.trace_metadata.flow_id = Some(flow_id);
        }
    }

    pub fn set_batch_tokens(&mut self, batch_tokens: usize) {
        if self.trace_thread_id.is_some() {
            self.trace_metadata.batch_tokens = Some(saturating_u64(batch_tokens));
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TraceMetadata {
    token_position: Option<u64>,
    token_id: Option<u64>,
    layer_id: Option<u64>,
    expert_id: Option<u64>,
    cache_hit: Option<bool>,
    flow_id: Option<u64>,
    batch_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct CompletedSpan {
    stage: ProfileStage,
    started: Instant,
    elapsed: Duration,
    trace_thread_id: Option<u64>,
    trace_metadata: TraceMetadata,
    work_items: u64,
    logical_bytes: u64,
}

/// Starts an inclusive stage span when a profile session is active.
pub fn span(stage: ProfileStage) -> ProfileSpan {
    span_with_work(stage, 0)
}

/// Starts a stage span and attaches a deterministic amount of scalar work.
pub fn span_with_work(stage: ProfileStage, work_items: usize) -> ProfileSpan {
    span_with_metrics(stage, work_items, 0)
}

/// Starts a stage span with scalar work and logical payload byte metrics.
pub fn span_with_metrics(
    stage: ProfileStage,
    work_items: usize,
    logical_bytes: usize,
) -> ProfileSpan {
    let collector = ACTIVE.with(|active| active.borrow().last().cloned());
    let started = collector.as_ref().map(|_| Instant::now());
    let trace_thread_id = collector
        .as_ref()
        .and_then(|collector| collector.trace.as_ref())
        .map(trace::TraceRecorder::thread_id);
    ProfileSpan {
        collector,
        stage,
        started,
        work_items: u64::try_from(work_items).unwrap_or(u64::MAX),
        logical_bytes: u64::try_from(logical_bytes).unwrap_or(u64::MAX),
        trace_thread_id,
        trace_metadata: TraceMetadata::default(),
    }
}

impl Drop for ProfileSpan {
    fn drop(&mut self) {
        if let (Some(collector), Some(started)) = (&self.collector, self.started) {
            collector.record(CompletedSpan {
                stage: self.stage,
                started,
                elapsed: started.elapsed(),
                trace_thread_id: self.trace_thread_id,
                trace_metadata: self.trace_metadata,
                work_items: self.work_items,
                logical_bytes: self.logical_bytes,
            });
        }
    }
}

struct Collector {
    stages: Mutex<HashMap<ProfileStage, Aggregate>>,
    closed: AtomicBool,
    trace: Option<trace::TraceRecorder>,
}

impl Collector {
    fn new(origin: Instant, trace_enabled: bool) -> Self {
        Self {
            stages: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
            trace: trace_enabled.then(|| trace::TraceRecorder::new(origin)),
        }
    }

    fn record(&self, completed: CompletedSpan) {
        let nanoseconds = duration_ns(completed.elapsed);
        let mut stages = self
            .stages
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        let aggregate = stages.entry(completed.stage).or_insert(Aggregate {
            calls: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            work_items: 0,
            logical_bytes: 0,
        });
        aggregate.calls = aggregate.calls.saturating_add(1);
        aggregate.total_ns = aggregate.total_ns.saturating_add(nanoseconds);
        aggregate.min_ns = aggregate.min_ns.min(nanoseconds);
        aggregate.max_ns = aggregate.max_ns.max(nanoseconds);
        aggregate.work_items = aggregate.work_items.saturating_add(completed.work_items);
        aggregate.logical_bytes = aggregate
            .logical_bytes
            .saturating_add(completed.logical_bytes);
        if let (Some(trace), Some(thread_id)) = (&self.trace, completed.trace_thread_id) {
            trace.record(completed, thread_id);
        }
    }

    fn snapshot(
        &self,
        wall_time: Duration,
        requested_threads: Option<usize>,
        resources: Option<SystemResourceReport>,
    ) -> ProfileReport {
        self.close();
        let stages = self
            .stages
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let entries = ALL_STAGES
            .iter()
            .filter_map(|&stage| {
                stages.get(&stage).map(|aggregate| ProfileEntry {
                    stage: stage.as_str(),
                    calls: aggregate.calls,
                    total_ns: aggregate.total_ns,
                    min_ns: aggregate.min_ns,
                    max_ns: aggregate.max_ns,
                    work_items: aggregate.work_items,
                    logical_bytes: aggregate.logical_bytes,
                })
            })
            .collect();
        ProfileReport {
            schema_version: 2,
            wall_time_ns: duration_ns(wall_time),
            requested_threads,
            worker_threads: crate::execution::initialized_worker_threads(),
            resources,
            stages: entries,
            trace: self.trace.as_ref().map(trace::TraceRecorder::snapshot),
        }
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
struct Aggregate {
    calls: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    work_items: u64,
    logical_bytes: u64,
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_calls_work_and_nested_stages() {
        let session = ProfileSession::start();
        {
            let _outer = span(ProfileStage::GlmToken);
            for _ in 0..2 {
                let mut kernel = span_with_metrics(ProfileStage::MatvecF32, 128, 32);
                kernel.add_logical_bytes(16);
                std::hint::black_box((0..100).sum::<usize>());
            }
        }
        let report = session.finish();
        assert_eq!(report.schema_version, 2);
        assert_eq!(report.stage(ProfileStage::GlmToken).unwrap().calls, 1);
        let kernel = report.stage(ProfileStage::MatvecF32).unwrap();
        assert_eq!(kernel.calls, 2);
        assert_eq!(kernel.work_items, 256);
        assert_eq!(kernel.logical_bytes, 96);
        assert!(report.render_text().contains("kernel.matvec.f32"));
    }

    #[test]
    fn disabled_spans_do_not_create_a_report_or_panic() {
        let _span = span(ProfileStage::MatvecF32);
    }

    #[test]
    fn captured_context_records_nested_spans_on_another_thread() {
        let session = ProfileSession::start();
        let context = capture_context();
        std::thread::spawn(move || {
            context.enter(|| {
                let _span = span_with_work(ProfileStage::MatvecMxFp4, 256);
            });
        })
        .join()
        .unwrap();
        let report = session.finish();
        let entry = report.stage(ProfileStage::MatvecMxFp4).unwrap();
        assert_eq!(entry.calls, 1);
        assert_eq!(entry.work_items, 256);
    }

    #[test]
    fn nested_sessions_survive_non_lifo_finish() {
        let outer = ProfileSession::start();
        let inner = ProfileSession::start();
        let outer_report = outer.finish();
        assert!(outer_report.stages.is_empty());
        {
            let _span = span(ProfileStage::GlmToken);
        }
        let inner_report = inner.finish();
        assert_eq!(inner_report.stage(ProfileStage::GlmToken).unwrap().calls, 1);
    }

    #[test]
    fn spans_outliving_their_session_are_ignored() {
        let session = ProfileSession::start();
        let late_span = span(ProfileStage::MatvecF32);
        let report = session.finish();
        drop(late_span);
        assert!(report.stage(ProfileStage::MatvecF32).is_none());
    }

    #[test]
    fn optional_trace_exports_complete_spans_and_resource_counters() {
        let session = ProfileSession::start_with_threads_and_trace(Some(2), true);
        {
            let _outer = span(ProfileStage::GenerateTotal);
            let mut inner = span_with_metrics(ProfileStage::MatvecF32, 128, 64);
            inner.set_token(7, 42);
            inner.set_layer_id(3);
            inner.set_expert_id(11);
            inner.set_cache_hit(true);
            inner.set_flow_id(99);
            inner.set_batch_tokens(1);
        }
        let report = session.finish();
        assert_eq!(report.trace_event_count(), 2);
        assert_eq!(report.trace_dropped_events(), 0);
        let trace: serde_json::Value = serde_json::from_slice(
            &report
                .chrome_trace_json_pretty()
                .unwrap()
                .expect("trace collection was enabled"),
        )
        .unwrap();
        let events = trace["traceEvents"].as_array().unwrap();
        assert!(events
            .iter()
            .any(|event| event["ph"] == "X" && event["name"] == "kernel.matvec.f32"));
        let kernel = events
            .iter()
            .find(|event| event["name"] == "kernel.matvec.f32")
            .unwrap();
        assert_eq!(kernel["args"]["token_position"], 7);
        assert_eq!(kernel["args"]["token_id"], 42);
        assert_eq!(kernel["args"]["layer_id"], 3);
        assert_eq!(kernel["args"]["expert_id"], 11);
        assert_eq!(kernel["args"]["cache_hit"], true);
        assert_eq!(kernel["args"]["flow_id"], 99);
        assert_eq!(kernel["args"]["batch_tokens"], 1);
        // Span timing is portable; process-resource counters currently require Linux procfs.
        #[cfg(target_os = "linux")]
        {
            assert!(report.resources.is_some());
            assert!(events
                .iter()
                .any(|event| event["ph"] == "C" && event["name"] == "CPU cores"));
        }
        #[cfg(not(target_os = "linux"))]
        {
            assert!(report.resources.is_none());
            assert!(events.iter().all(|event| event["ph"] != "C"));
        }
    }
}
