//! Optional per-span trace recording and Chrome Trace Event export.

use super::{duration_ns, CompletedSpan, SystemResourceReport, TraceMetadata};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use std::thread::ThreadId;
use std::time::Instant;

const MAX_TRACE_EVENTS: usize = 1_000_000;

pub(super) struct TraceRecorder {
    origin: Instant,
    state: Mutex<TraceState>,
}

#[derive(Default)]
struct TraceState {
    spans: Vec<TraceSpan>,
    threads: HashMap<ThreadId, u64>,
    thread_names: HashMap<u64, String>,
    next_thread_id: u64,
    dropped_events: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct TraceSpan {
    name: &'static str,
    start_ns: u64,
    duration_ns: u64,
    thread_id: u64,
    work_items: u64,
    logical_bytes: u64,
    metadata: TraceMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct TraceSnapshot {
    spans: Vec<TraceSpan>,
    thread_names: Vec<(u64, String)>,
    dropped_events: u64,
}

impl TraceRecorder {
    pub(super) fn new(origin: Instant) -> Self {
        Self {
            origin,
            state: Mutex::new(TraceState::default()),
        }
    }

    pub(super) fn thread_id(&self) -> u64 {
        let thread = std::thread::current();
        let native_id = thread.id();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(&id) = state.threads.get(&native_id) {
            return id;
        }
        state.next_thread_id = state.next_thread_id.saturating_add(1);
        let id = state.next_thread_id;
        state.threads.insert(native_id, id);
        state.thread_names.insert(
            id,
            thread
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("thread-{id}")),
        );
        id
    }

    pub(super) fn record(&self, completed: CompletedSpan, thread_id: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.spans.len() >= MAX_TRACE_EVENTS {
            state.dropped_events = state.dropped_events.saturating_add(1);
            return;
        }
        state.spans.push(TraceSpan {
            name: completed.stage.as_str(),
            start_ns: duration_ns(completed.started.saturating_duration_since(self.origin)),
            duration_ns: duration_ns(completed.elapsed),
            thread_id,
            work_items: completed.work_items,
            logical_bytes: completed.logical_bytes,
            metadata: completed.trace_metadata,
        });
    }

    pub(super) fn snapshot(&self) -> TraceSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut spans = state.spans.clone();
        spans.sort_by(|left, right| {
            left.start_ns
                .cmp(&right.start_ns)
                .then_with(|| right.duration_ns.cmp(&left.duration_ns))
        });
        let mut thread_names = state
            .thread_names
            .iter()
            .map(|(&id, name)| (id, name.clone()))
            .collect::<Vec<_>>();
        thread_names.sort_by_key(|(id, _)| *id);
        TraceSnapshot {
            spans,
            thread_names,
            dropped_events: state.dropped_events,
        }
    }
}

impl TraceSnapshot {
    pub(super) fn event_count(&self) -> usize {
        self.spans.len()
    }

    pub(super) fn dropped_events(&self) -> u64 {
        self.dropped_events
    }

    pub(super) fn to_json_pretty(
        &self,
        resources: Option<&SystemResourceReport>,
    ) -> serde_json::Result<Vec<u8>> {
        let mut events = Vec::with_capacity(
            self.spans.len()
                + self.thread_names.len()
                + resources.map_or(0, |report| report.samples.len().saturating_mul(5))
                + 2,
        );
        events.push(ChromeEvent::metadata(
            "process_name",
            0,
            json!({ "name": "urb inference" }),
        ));
        for (thread_id, name) in &self.thread_names {
            events.push(ChromeEvent::metadata(
                "thread_name",
                *thread_id,
                json!({ "name": name }),
            ));
        }
        events.extend(self.spans.iter().map(|span| {
            let mut args = serde_json::Map::new();
            args.insert("work_items".to_owned(), span.work_items.into());
            args.insert("logical_bytes".to_owned(), span.logical_bytes.into());
            insert_optional(&mut args, "token_position", span.metadata.token_position);
            insert_optional(&mut args, "token_id", span.metadata.token_id);
            insert_optional(&mut args, "layer_id", span.metadata.layer_id);
            insert_optional(&mut args, "expert_id", span.metadata.expert_id);
            insert_optional(&mut args, "flow_id", span.metadata.flow_id);
            insert_optional(&mut args, "batch_tokens", span.metadata.batch_tokens);
            if let Some(cache_hit) = span.metadata.cache_hit {
                args.insert("cache_hit".to_owned(), cache_hit.into());
            }
            ChromeEvent {
                name: span.name,
                category: stage_category(span.name),
                phase: "X",
                timestamp_us: ns_to_us(span.start_ns),
                duration_us: Some(ns_to_us(span.duration_ns)),
                process_id: 1,
                thread_id: span.thread_id,
                args: Value::Object(args),
            }
        }));
        if let Some(resources) = resources {
            append_resource_counters(&mut events, resources);
        }
        if self.dropped_events != 0 {
            events.push(ChromeEvent {
                name: "trace.events_dropped",
                category: "profiling",
                phase: "I",
                timestamp_us: 0.0,
                duration_us: None,
                process_id: 1,
                thread_id: 0,
                args: json!({ "count": self.dropped_events }),
            });
        }
        serde_json::to_vec_pretty(&ChromeTraceDocument {
            trace_events: events,
            display_time_unit: "ms",
            metadata: ChromeTraceMetadata {
                source: "urbilateria",
                format_version: 1,
                recorded_span_events: self.spans.len(),
                dropped_span_events: self.dropped_events,
            },
        })
    }
}

#[derive(Serialize)]
struct ChromeTraceDocument<'a> {
    #[serde(rename = "traceEvents")]
    trace_events: Vec<ChromeEvent<'a>>,
    #[serde(rename = "displayTimeUnit")]
    display_time_unit: &'static str,
    metadata: ChromeTraceMetadata,
}

#[derive(Serialize)]
struct ChromeTraceMetadata {
    source: &'static str,
    format_version: u32,
    recorded_span_events: usize,
    dropped_span_events: u64,
}

#[derive(Serialize)]
struct ChromeEvent<'a> {
    name: &'a str,
    #[serde(rename = "cat")]
    category: &'a str,
    #[serde(rename = "ph")]
    phase: &'static str,
    #[serde(rename = "ts")]
    timestamp_us: f64,
    #[serde(rename = "dur", skip_serializing_if = "Option::is_none")]
    duration_us: Option<f64>,
    #[serde(rename = "pid")]
    process_id: u32,
    #[serde(rename = "tid")]
    thread_id: u64,
    args: Value,
}

impl<'a> ChromeEvent<'a> {
    fn metadata(name: &'a str, thread_id: u64, args: Value) -> Self {
        Self {
            name,
            category: "__metadata",
            phase: "M",
            timestamp_us: 0.0,
            duration_us: None,
            process_id: 1,
            thread_id,
            args,
        }
    }

    fn counter(name: &'a str, timestamp_us: f64, value: f64) -> Self {
        Self {
            name,
            category: "resources",
            phase: "C",
            timestamp_us,
            duration_us: None,
            process_id: 1,
            thread_id: 0,
            args: json!({ "value": value }),
        }
    }
}

fn append_resource_counters(events: &mut Vec<ChromeEvent<'_>>, resources: &SystemResourceReport) {
    let samples = &resources.samples;
    for (index, sample) in samples.iter().enumerate() {
        let timestamp_us = ns_to_us(sample.elapsed_ns);
        let previous = index.checked_sub(1).and_then(|index| samples.get(index));
        let (cpu_cores, read_rate, storage_rate) = previous.map_or((0.0, 0.0, 0.0), |previous| {
            let elapsed = sample.elapsed_ns.saturating_sub(previous.elapsed_ns).max(1);
            (
                sample.cpu_time_ns.saturating_sub(previous.cpu_time_ns) as f64 / elapsed as f64,
                bytes_per_second(
                    sample
                        .read_char_bytes
                        .saturating_sub(previous.read_char_bytes),
                    elapsed,
                ),
                bytes_per_second(
                    sample
                        .storage_read_bytes
                        .saturating_sub(previous.storage_read_bytes),
                    elapsed,
                ),
            )
        });
        events.push(ChromeEvent::counter("CPU cores", timestamp_us, cpu_cores));
        events.push(ChromeEvent::counter(
            "RSS MiB",
            timestamp_us,
            bytes_to_mib(sample.rss_bytes),
        ));
        events.push(ChromeEvent::counter(
            "syscall read MiB/s",
            timestamp_us,
            bytes_to_mib(read_rate as u64),
        ));
        events.push(ChromeEvent::counter(
            "storage read MiB/s",
            timestamp_us,
            bytes_to_mib(storage_rate as u64),
        ));
        events.push(ChromeEvent::counter(
            "minor faults",
            timestamp_us,
            sample.minor_faults as f64,
        ));
    }
}

fn stage_category(stage: &str) -> &str {
    stage
        .split_once('.')
        .map_or(stage, |(category, _)| category)
}

fn ns_to_us(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000.0
}

fn bytes_per_second(bytes: u64, elapsed_ns: u64) -> f64 {
    bytes as f64 * 1_000_000_000.0 / elapsed_ns.max(1) as f64
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn insert_optional(args: &mut serde_json::Map<String, Value>, name: &str, value: Option<u64>) {
    if let Some(value) = value {
        args.insert(name.to_owned(), value.into());
    }
}
