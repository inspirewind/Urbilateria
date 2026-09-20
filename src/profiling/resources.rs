//! Linux process-resource sampling for profiling sessions.
//!
//! `/proc/self/io` distinguishes bytes submitted to the storage layer (`read_bytes` and
//! `write_bytes`) from bytes returned by read/write syscalls (`rchar` and `wchar`). The latter may
//! be served by the page cache, so the two groups must not be interpreted as the same thing.

use serde::Serialize;
use std::fs;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResourceSample {
    pub elapsed_ns: u64,
    /// Process CPU time consumed since the profile started. One busy core advances by one ns/ns.
    pub cpu_time_ns: u64,
    pub rss_bytes: u64,
    pub swap_bytes: u64,
    pub thread_count: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
    /// Bytes read through syscalls, including data served from the page cache.
    pub read_char_bytes: u64,
    /// Bytes written through syscalls, including data retained in the page cache.
    pub write_char_bytes: u64,
    /// Bytes this process caused Linux to fetch from the storage layer.
    pub storage_read_bytes: u64,
    /// Bytes this process caused Linux to submit to the storage layer.
    pub storage_write_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResourceSummary {
    /// Average number of logical CPU cores occupied during the session.
    pub average_cpu_cores: f64,
    pub average_machine_cpu_percent: Option<f64>,
    pub peak_interval_cpu_cores: f64,
    pub start_rss_bytes: u64,
    pub end_rss_bytes: u64,
    pub peak_rss_bytes: u64,
    /// Kernel high-water RSS for the process lifetime, which may predate this profile session.
    pub process_lifetime_peak_rss_bytes: u64,
    pub peak_rss_percent_system_memory: Option<f64>,
    pub peak_swap_bytes: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
    pub read_char_bytes: u64,
    pub write_char_bytes: u64,
    pub storage_read_bytes: u64,
    pub storage_write_bytes: u64,
    pub average_storage_read_bytes_per_second: f64,
    pub average_storage_write_bytes_per_second: f64,
    pub peak_interval_storage_read_bytes_per_second: f64,
    pub peak_interval_storage_write_bytes_per_second: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SystemResourceReport {
    pub source: &'static str,
    pub sample_interval_ns: u64,
    pub clock_ticks_per_second: u64,
    pub logical_cpu_count: Option<usize>,
    pub system_memory_total_bytes: Option<u64>,
    pub system_memory_available_start_bytes: Option<u64>,
    pub system_memory_available_end_bytes: Option<u64>,
    pub summary: ResourceSummary,
    /// Cumulative session-relative counters suitable for plotting against `elapsed_ns`.
    pub samples: Vec<ResourceSample>,
}

pub(super) struct ResourceMonitor {
    baseline: RawSnapshot,
    started: Instant,
    stop: Sender<()>,
    worker: Option<JoinHandle<Vec<TimedSnapshot>>>,
    memory_total_bytes: Option<u64>,
    memory_available_start_bytes: Option<u64>,
    clock_ticks_per_second: u64,
}

impl ResourceMonitor {
    pub(super) fn start() -> Option<Self> {
        let baseline = RawSnapshot::capture()?;
        let (stop, receiver) = mpsc::channel();
        let started = Instant::now();
        let worker_started = started;
        let worker = thread::Builder::new()
            .name("urb-profile-resources".to_owned())
            .spawn(move || {
                let mut samples = Vec::new();
                while receiver.recv_timeout(SAMPLE_INTERVAL).is_err() {
                    if let Some(snapshot) = RawSnapshot::capture() {
                        samples.push(TimedSnapshot {
                            elapsed: worker_started.elapsed(),
                            snapshot,
                        });
                    }
                }
                samples
            })
            .ok()?;
        let memory = read_meminfo();
        let clock_ticks_per_second = clock_ticks_per_second();
        Some(Self {
            baseline,
            started,
            stop,
            worker: Some(worker),
            memory_total_bytes: memory.total_bytes,
            memory_available_start_bytes: memory.available_bytes,
            clock_ticks_per_second,
        })
    }

    pub(super) fn finish(mut self, wall_time: Duration) -> SystemResourceReport {
        // Capture the end boundary before stopping and joining the sampling worker so profiler
        // shutdown time is not charged to the measured workload.
        let final_sample = RawSnapshot::capture().map(|snapshot| TimedSnapshot {
            elapsed: self.started.elapsed(),
            snapshot,
        });
        let _ = self.stop.send(());
        let mut captured = self
            .worker
            .take()
            .and_then(|worker| worker.join().ok())
            .unwrap_or_default();
        if let Some(sample) = final_sample {
            captured.push(sample);
        }
        build_report(
            self.baseline,
            captured,
            wall_time,
            self.memory_total_bytes,
            self.memory_available_start_bytes,
            read_meminfo().available_bytes,
            self.clock_ticks_per_second,
        )
    }

    fn stop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Debug, Clone, Copy)]
struct TimedSnapshot {
    elapsed: Duration,
    snapshot: RawSnapshot,
}

#[derive(Debug, Clone, Copy)]
struct RawSnapshot {
    cpu_ticks: u64,
    rss_bytes: u64,
    lifetime_peak_rss_bytes: u64,
    swap_bytes: u64,
    thread_count: u64,
    minor_faults: u64,
    major_faults: u64,
    read_char_bytes: u64,
    write_char_bytes: u64,
    storage_read_bytes: u64,
    storage_write_bytes: u64,
}

impl RawSnapshot {
    fn capture() -> Option<Self> {
        let stat = fs::read_to_string("/proc/self/stat").ok()?;
        let status_text = fs::read_to_string("/proc/self/status").ok()?;
        let io_text = fs::read_to_string("/proc/self/io").ok()?;
        let status = parse_key_values(&status_text);
        let io = parse_key_values(&io_text);
        let stat = parse_stat(&stat)?;
        Some(Self {
            cpu_ticks: stat.cpu_ticks,
            rss_bytes: kib_value(&status, "VmRSS"),
            lifetime_peak_rss_bytes: kib_value(&status, "VmHWM"),
            swap_bytes: kib_value(&status, "VmSwap"),
            thread_count: plain_value(&status, "Threads"),
            minor_faults: stat.minor_faults,
            major_faults: stat.major_faults,
            read_char_bytes: plain_value(&io, "rchar"),
            write_char_bytes: plain_value(&io, "wchar"),
            storage_read_bytes: plain_value(&io, "read_bytes"),
            storage_write_bytes: plain_value(&io, "write_bytes"),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct StatValues {
    cpu_ticks: u64,
    minor_faults: u64,
    major_faults: u64,
}

fn parse_stat(stat: &str) -> Option<StatValues> {
    // The command name is parenthesized and may itself contain spaces or `)`, so split at the last
    // closing parenthesis. `fields[0]` is then field 3 (`state`) from proc_pid_stat(5).
    let fields = stat
        .get(stat.rfind(')')? + 1..)?
        .split_whitespace()
        .collect::<Vec<_>>();
    Some(StatValues {
        minor_faults: fields.get(7)?.parse().ok()?,
        major_faults: fields.get(9)?.parse().ok()?,
        cpu_ticks: fields
            .get(11)?
            .parse::<u64>()
            .ok()?
            .saturating_add(fields.get(12)?.parse::<u64>().ok()?),
    })
}

fn parse_key_values(input: &str) -> std::collections::HashMap<&str, u64> {
    input
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            Some((key, value.split_whitespace().next()?.parse().ok()?))
        })
        .collect()
}

fn plain_value(values: &std::collections::HashMap<&str, u64>, key: &str) -> u64 {
    values.get(key).copied().unwrap_or(0)
}

fn kib_value(values: &std::collections::HashMap<&str, u64>, key: &str) -> u64 {
    plain_value(values, key).saturating_mul(1024)
}

#[derive(Default)]
struct MemoryInfo {
    total_bytes: Option<u64>,
    available_bytes: Option<u64>,
}

fn read_meminfo() -> MemoryInfo {
    let Ok(input) = fs::read_to_string("/proc/meminfo") else {
        return MemoryInfo::default();
    };
    let values = parse_key_values(&input);
    MemoryInfo {
        total_bytes: values
            .get("MemTotal")
            .map(|value| value.saturating_mul(1024)),
        available_bytes: values
            .get("MemAvailable")
            .map(|value| value.saturating_mul(1024)),
    }
}

fn build_report(
    baseline: RawSnapshot,
    captured: Vec<TimedSnapshot>,
    wall_time: Duration,
    memory_total_bytes: Option<u64>,
    memory_available_start_bytes: Option<u64>,
    memory_available_end_bytes: Option<u64>,
    clock_ticks_per_second: u64,
) -> SystemResourceReport {
    let process_lifetime_peak_rss_bytes = captured
        .iter()
        .map(|sample| sample.snapshot.lifetime_peak_rss_bytes)
        .fold(baseline.lifetime_peak_rss_bytes, u64::max);
    let mut samples = Vec::with_capacity(captured.len() + 1);
    samples.push(to_sample(
        Duration::ZERO,
        baseline,
        baseline,
        clock_ticks_per_second,
    ));
    samples.extend(captured.into_iter().map(|sample| {
        to_sample(
            sample.elapsed,
            sample.snapshot,
            baseline,
            clock_ticks_per_second,
        )
    }));
    samples.sort_by_key(|sample| sample.elapsed_ns);
    let end = samples
        .last()
        .expect("the baseline resource sample always exists");
    let wall_ns = duration_ns(wall_time).max(1);
    let logical_cpu_count = thread::available_parallelism().ok().map(usize::from);
    let average_cpu_cores = end.cpu_time_ns as f64 / wall_ns as f64;
    let (peak_cpu, peak_read, peak_write) = interval_peaks(&samples);
    let peak_rss_bytes = samples
        .iter()
        .map(|sample| sample.rss_bytes)
        .max()
        .unwrap_or(0);
    SystemResourceReport {
        source: "linux_procfs",
        sample_interval_ns: duration_ns(SAMPLE_INTERVAL),
        clock_ticks_per_second,
        logical_cpu_count,
        system_memory_total_bytes: memory_total_bytes,
        system_memory_available_start_bytes: memory_available_start_bytes,
        system_memory_available_end_bytes: memory_available_end_bytes,
        summary: ResourceSummary {
            average_cpu_cores,
            average_machine_cpu_percent: logical_cpu_count
                .map(|count| average_cpu_cores * 100.0 / count as f64),
            peak_interval_cpu_cores: peak_cpu,
            start_rss_bytes: baseline.rss_bytes,
            end_rss_bytes: end.rss_bytes,
            peak_rss_bytes,
            process_lifetime_peak_rss_bytes,
            peak_rss_percent_system_memory: memory_total_bytes
                .filter(|total| *total != 0)
                .map(|total| peak_rss_bytes as f64 * 100.0 / total as f64),
            peak_swap_bytes: samples
                .iter()
                .map(|sample| sample.swap_bytes)
                .max()
                .unwrap_or(0),
            minor_faults: end.minor_faults,
            major_faults: end.major_faults,
            read_char_bytes: end.read_char_bytes,
            write_char_bytes: end.write_char_bytes,
            storage_read_bytes: end.storage_read_bytes,
            storage_write_bytes: end.storage_write_bytes,
            average_storage_read_bytes_per_second: rate(end.storage_read_bytes, wall_ns),
            average_storage_write_bytes_per_second: rate(end.storage_write_bytes, wall_ns),
            peak_interval_storage_read_bytes_per_second: peak_read,
            peak_interval_storage_write_bytes_per_second: peak_write,
        },
        samples,
    }
}

fn to_sample(
    elapsed: Duration,
    current: RawSnapshot,
    baseline: RawSnapshot,
    clock_ticks_per_second: u64,
) -> ResourceSample {
    ResourceSample {
        elapsed_ns: duration_ns(elapsed),
        cpu_time_ns: ticks_to_ns(
            current.cpu_ticks.saturating_sub(baseline.cpu_ticks),
            clock_ticks_per_second,
        ),
        rss_bytes: current.rss_bytes,
        swap_bytes: current.swap_bytes,
        thread_count: current.thread_count,
        minor_faults: current.minor_faults.saturating_sub(baseline.minor_faults),
        major_faults: current.major_faults.saturating_sub(baseline.major_faults),
        read_char_bytes: current
            .read_char_bytes
            .saturating_sub(baseline.read_char_bytes),
        write_char_bytes: current
            .write_char_bytes
            .saturating_sub(baseline.write_char_bytes),
        storage_read_bytes: current
            .storage_read_bytes
            .saturating_sub(baseline.storage_read_bytes),
        storage_write_bytes: current
            .storage_write_bytes
            .saturating_sub(baseline.storage_write_bytes),
    }
}

fn interval_peaks(samples: &[ResourceSample]) -> (f64, f64, f64) {
    samples
        .windows(2)
        .fold((0.0_f64, 0.0_f64, 0.0_f64), |peaks, pair| {
            let elapsed = pair[1].elapsed_ns.saturating_sub(pair[0].elapsed_ns);
            if elapsed == 0 {
                return peaks;
            }
            (
                peaks.0.max(
                    pair[1].cpu_time_ns.saturating_sub(pair[0].cpu_time_ns) as f64 / elapsed as f64,
                ),
                peaks.1.max(rate(
                    pair[1]
                        .storage_read_bytes
                        .saturating_sub(pair[0].storage_read_bytes),
                    elapsed,
                )),
                peaks.2.max(rate(
                    pair[1]
                        .storage_write_bytes
                        .saturating_sub(pair[0].storage_write_bytes),
                    elapsed,
                )),
            )
        })
}

fn rate(value: u64, elapsed_ns: u64) -> f64 {
    value as f64 * 1_000_000_000.0 / elapsed_ns.max(1) as f64
}

fn clock_ticks_per_second() -> u64 {
    const AT_CLKTCK: u64 = 17;
    let Ok(auxv) = fs::read("/proc/self/auxv") else {
        return 100;
    };
    let word_bytes = std::mem::size_of::<usize>();
    for entry in auxv.chunks_exact(word_bytes * 2) {
        let key = native_word(&entry[..word_bytes]);
        let value = native_word(&entry[word_bytes..]);
        if key == AT_CLKTCK {
            return value.max(1);
        }
    }
    100
}

fn native_word(bytes: &[u8]) -> u64 {
    let mut word = [0_u8; 8];
    word[..bytes.len()].copy_from_slice(bytes);
    u64::from_ne_bytes(word)
}

fn ticks_to_ns(ticks: u64, ticks_per_second: u64) -> u64 {
    ticks
        .saturating_mul(1_000_000_000)
        .saturating_div(ticks_per_second.max(1))
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proc_stat_after_a_command_name_with_spaces() {
        let mut fields = vec!["S"; 22];
        fields[7] = "11";
        fields[9] = "3";
        fields[11] = "17";
        fields[12] = "5";
        let input = format!("123 (worker pool) {}", fields.join(" "));
        let parsed = parse_stat(&input).unwrap();
        assert_eq!(parsed.minor_faults, 11);
        assert_eq!(parsed.major_faults, 3);
        assert_eq!(parsed.cpu_ticks, 22);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn samples_current_linux_process() {
        let snapshot = RawSnapshot::capture().unwrap();
        assert!(snapshot.rss_bytes > 0);
        assert!(snapshot.thread_count > 0);
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn resource_sampling_is_unavailable_on_unsupported_platforms() {
        assert!(RawSnapshot::capture().is_none());
        assert!(ResourceMonitor::start().is_none());
    }

    #[test]
    fn reads_a_plausible_clock_tick_rate() {
        assert!((10..=10_000).contains(&clock_ticks_per_second()));
    }
}
