//! Interval resource usage of the inference child, independent of token output and profiling.

use std::time::{Duration, Instant};

const INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Usage {
    /// One fully occupied logical CPU is 100%; multithreaded inference can exceed 100%.
    pub cpu_percent: Option<f64>,
    /// Storage-layer reads attributed to the process, excluding page-cache hits.
    pub read_bytes_per_second: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    cpu_time: Option<Duration>,
    read_bytes: Option<u64>,
}

impl Counters {
    fn usage_since(self, previous: Self, elapsed: Duration) -> Usage {
        let seconds = elapsed.as_secs_f64();
        if seconds == 0.0 {
            return Usage::default();
        }
        Usage {
            cpu_percent: self.cpu_time.zip(previous.cpu_time).and_then(|(now, old)| {
                now.checked_sub(old)
                    .map(|delta| delta.as_secs_f64() / seconds * 100.0)
            }),
            read_bytes_per_second: self
                .read_bytes
                .zip(previous.read_bytes)
                .and_then(|(now, old)| now.checked_sub(old).map(|delta| delta as f64 / seconds)),
        }
    }
}

pub struct Monitor {
    pid: u32,
    previous: Counters,
    sampled_at: Instant,
}

impl Monitor {
    pub fn new(pid: u32) -> Self {
        let previous = capture(pid);
        Self {
            pid,
            previous,
            sampled_at: Instant::now(),
        }
    }

    pub fn is_due(&self) -> bool {
        self.sampled_at.elapsed() >= INTERVAL
    }

    pub fn sample(&mut self) -> Usage {
        let current = capture(self.pid);
        let now = Instant::now();
        let usage = current.usage_since(self.previous, now.duration_since(self.sampled_at));
        self.previous = current;
        self.sampled_at = now;
        usage
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", test))]
fn cpu_duration(ticks: u64, numerator: u64, denominator: u64) -> Option<Duration> {
    let nanos = (u128::from(ticks) * u128::from(numerator)).checked_div(denominator.into())?;
    Some(Duration::new(
        u64::try_from(nanos / 1_000_000_000).ok()?,
        (nanos % 1_000_000_000) as u32,
    ))
}

#[cfg(target_os = "linux")]
fn capture(pid: u32) -> Counters {
    use std::fs;
    use std::sync::OnceLock;
    static TICKS: OnceLock<Option<u64>> = OnceLock::new();
    let ticks_per_second = *TICKS.get_or_init(|| {
        // SAFETY: sysconf only queries the clock tick rate; no pointers are involved.
        let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        u64::try_from(ticks).ok().filter(|ticks| *ticks != 0)
    });
    let cpu_time = fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|text| cpu_ticks(&text))
        .zip(ticks_per_second)
        .and_then(|(ticks, hz)| cpu_duration(ticks, 1_000_000_000, hz));
    let read_bytes = fs::read_to_string(format!("/proc/{pid}/io"))
        .ok()
        .and_then(|text| storage_read_bytes(&text));
    Counters {
        cpu_time,
        read_bytes,
    }
}

#[cfg(target_os = "linux")]
fn cpu_ticks(stat: &str) -> Option<u64> {
    // comm can contain spaces and ')'. After its last ')', field 3 is the first token.
    let mut fields = stat.get(stat.rfind(')')? + 1..)?.split_whitespace();
    let user = fields.nth(11)?.parse::<u64>().ok()?;
    let system = fields.next()?.parse::<u64>().ok()?;
    user.checked_add(system)
}

#[cfg(target_os = "linux")]
fn storage_read_bytes(io: &str) -> Option<u64> {
    io.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == "read_bytes")
            .then(|| value.trim().parse().ok())
            .flatten()
    })
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc still exposes the stable mach_timebase_info ABI.
fn capture(pid: u32) -> Counters {
    use std::sync::OnceLock;
    static TIMEBASE: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    let timebase = *TIMEBASE.get_or_init(|| {
        let mut timebase = libc::mach_timebase_info { numer: 0, denom: 0 };
        // SAFETY: the pointer addresses a valid writable mach_timebase_info buffer.
        let result = unsafe { libc::mach_timebase_info(&mut timebase) };
        (result == 0 && timebase.numer != 0 && timebase.denom != 0)
            .then_some((timebase.numer, timebase.denom))
    });
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return Counters::default();
    };
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    // SAFETY: RUSAGE_INFO_V2 matches the writable buffer's layout and size. libproc's
    // historical rusage_info_t* signature expects the address of the struct itself.
    let result =
        unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V2, info.as_mut_ptr().cast()) };
    if result != 0 {
        return Counters::default();
    }
    // SAFETY: proc_pid_rusage succeeded and initialized this zeroed integer-only struct.
    let info = unsafe { info.assume_init() };
    Counters {
        cpu_time: info
            .ri_user_time
            .checked_add(info.ri_system_time)
            // rusage CPU times are Mach ticks, not nanoseconds (notably on Apple Silicon).
            .zip(timebase)
            .and_then(|(ticks, (numer, denom))| cpu_duration(ticks, numer.into(), denom.into())),
        read_bytes: Some(info.ri_diskio_bytesread),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn capture(_pid: u32) -> Counters {
    Counters::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_clock_conversion_handles_linux_ticks_and_mach_timebases() {
        assert_eq!(
            cpu_duration(125, 1_000_000_000, 100),
            Some(Duration::from_millis(1250))
        );
        assert_eq!(
            cpu_duration(24_000_000, 125, 3),
            Some(Duration::from_secs(1))
        );
        assert_eq!(
            cpu_duration(u64::MAX, 1_000_000_000, 1),
            Some(Duration::from_secs(u64::MAX))
        );
        assert_eq!(cpu_duration(1, 1, 0), None);
    }

    #[test]
    fn interval_rates_allow_multiple_cores_and_distinguish_missing_from_zero() {
        let previous = Counters {
            cpu_time: Some(Duration::from_secs(10)),
            read_bytes: Some(1_048_576),
        };
        let current = Counters {
            cpu_time: Some(Duration::from_secs(13)),
            read_bytes: Some(3_145_728),
        };
        assert_eq!(
            current.usage_since(previous, Duration::from_secs(2)),
            Usage {
                cpu_percent: Some(150.0),
                read_bytes_per_second: Some(1_048_576.0),
            }
        );
        assert_eq!(
            previous.usage_since(previous, Duration::from_secs(1)),
            Usage {
                cpu_percent: Some(0.0),
                read_bytes_per_second: Some(0.0),
            }
        );
        assert_eq!(
            previous.usage_since(current, Duration::from_secs(1)),
            Usage::default()
        );
        assert_eq!(
            current.usage_since(previous, Duration::ZERO),
            Usage::default()
        );
        let missing_io = Counters {
            read_bytes: None,
            ..current
        };
        assert_eq!(
            missing_io.usage_since(previous, Duration::from_secs(2)),
            Usage {
                cpu_percent: Some(150.0),
                read_bytes_per_second: None,
            }
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn proc_parsing_handles_command_names_and_ignores_cached_read_counters() {
        let mut fields = vec!["0"; 22];
        fields[0] = "R";
        fields[11] = "125";
        fields[12] = "75";
        fields[13] = "900"; // Child CPU time is not charged to this process.
        assert_eq!(
            cpu_ticks(&format!("123 (worker ) pool) {}", fields.join(" "))),
            Some(200)
        );
        assert_eq!(cpu_ticks("123 (truncated) R 0"), None);
        assert_eq!(
            storage_read_bytes("rchar: 999999\nread_bytes: 4096\n"),
            Some(4096)
        );
        assert_eq!(storage_read_bytes("rchar: 999999\n"), None);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn captures_process_counters_and_starts_with_a_fresh_baseline() {
        let counters = capture(std::process::id());
        assert!(counters.cpu_time.is_some());
        assert!(counters.read_bytes.is_some());
        let monitor = Monitor::new(std::process::id());
        assert!(!monitor.is_due());
    }
}
