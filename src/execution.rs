//! Process-wide CPU execution policy.
//!
//! Inference uses one persistent Rayon pool. Large matrix operations split independent output
//! rows across that pool, while each row keeps its original scalar reduction order.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::fs;
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

/// Minimum number of scalar multiply-accumulates before a kernel enters the worker pool.
///
/// This avoids paying scheduling costs for the many tiny matrices in synthetic/reference tests.
pub const PARALLEL_MIN_WORK: usize = 64 * 1024;
/// Defensive upper bound for an explicitly requested worker pool.
pub const MAX_WORKER_THREADS: usize = 1_024;

static CPU_POOL: OnceLock<ThreadPool> = OnceLock::new();
static IO_POOL: OnceLock<IoPool> = OnceLock::new();

type IoOperation = Box<dyn FnOnce() + Send + 'static>;

struct IoPool {
    queue: Arc<(Mutex<VecDeque<IoOperation>>, Condvar)>,
}

pub(crate) struct IoTask<R> {
    receiver: mpsc::Receiver<R>,
}

impl<R> IoTask<R> {
    pub(crate) fn join(self) -> R {
        self.receiver
            .recv()
            .expect("an inference I/O worker stopped before returning its result")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadPoolError(String);

impl fmt::Display for ThreadPoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ThreadPoolError {}

/// Configures the persistent inference pool.
///
/// Call this before the first inference kernel. Repeating the same configuration is harmless;
/// changing it after initialization is rejected because Rayon pools cannot be resized safely.
pub fn configure_threads(threads: usize) -> Result<(), ThreadPoolError> {
    if threads == 0 {
        return Err(ThreadPoolError(
            "CPU worker count must be greater than zero".to_owned(),
        ));
    }
    if threads > MAX_WORKER_THREADS {
        return Err(ThreadPoolError(format!(
            "CPU worker count {threads} exceeds the safety limit {MAX_WORKER_THREADS}"
        )));
    }
    limit_glibc_arenas();
    if let Some(pool) = CPU_POOL.get() {
        return validate_worker_count(pool, threads);
    }
    let affinity = Arc::new(physical_core_first_cpu_order());
    let worker_affinity = Arc::clone(&affinity);
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("urb-cpu-{index}"))
        .start_handler(move |index| {
            if let Some(&cpu) = worker_affinity.get(index) {
                pin_current_thread(cpu);
            }
        })
        .build()
        .map_err(|error| ThreadPoolError(format!("could not create CPU worker pool: {error}")))?;
    match CPU_POOL.set(pool) {
        Ok(()) => Ok(()),
        Err(_) => validate_worker_count(
            CPU_POOL
                .get()
                .expect("a failed OnceLock set has a winning value"),
            threads,
        ),
    }
}

/// Large routed-expert buffers are allocated and evicted from many Rayon workers. Glibc's default
/// per-thread arena growth can retain several GiB after those buffers are freed, reducing the RAM
/// available to the expert cache. Two arenas retain parallel allocation while bounding that high
/// water mark. This is a best-effort GNU/Linux optimization; other allocators are unchanged.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn limit_glibc_arenas() {
    static CONFIGURED: OnceLock<()> = OnceLock::new();
    CONFIGURED.get_or_init(|| {
        const M_ARENA_MAX: i32 = -8;
        unsafe extern "C" {
            fn mallopt(parameter: i32, value: i32) -> i32;
        }
        // SAFETY: `mallopt` takes two integers and mutates only glibc's process allocator policy.
        // Failure is harmless and intentionally ignored so non-glibc-compatible runtimes fail
        // open to their existing allocation behavior.
        let _ = unsafe { mallopt(M_ARENA_MAX, 2) };
    });
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn limit_glibc_arenas() {}

/// Keeps streamed weight allocations in glibc arenas so same-shaped decoder layers can reuse
/// already-faulted pages instead of cycling anonymous mmap regions.
///
/// Callers opt in only when their RAM plan streams decoder layers. Fully resident plans gain no
/// reuse but can retain several GiB of otherwise free expert/prefill scratch, so they deliberately
/// keep glibc's default large-allocation policy. The 64 MiB ceiling covers V4's 32 MiB projection
/// payloads and remains below the planner's existing 512 MiB allocator-fragmentation reserve.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn enable_streamed_weight_allocation_reuse() {
    static CONFIGURED: OnceLock<()> = OnceLock::new();
    CONFIGURED.get_or_init(|| {
        const M_MMAP_THRESHOLD: i32 = -3;
        const REUSABLE_WEIGHT_ALLOCATION_MAX: i32 = 64 * 1024 * 1024;
        unsafe extern "C" {
            fn mallopt(parameter: i32, value: i32) -> i32;
        }
        // SAFETY: `mallopt` takes two integers and mutates only glibc's process allocator policy.
        // Failure is harmless and intentionally ignored so compatible runtimes fail open.
        let _ = unsafe { mallopt(M_MMAP_THRESHOLD, REUSABLE_WEIGHT_ALLOCATION_MAX) };
    });
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub fn enable_streamed_weight_allocation_reuse() {}

/// Returns the effective worker count, lazily creating the default pool when needed.
pub fn worker_threads() -> usize {
    pool().current_num_threads()
}

/// Returns the effective worker count without lazily creating the pool.
pub fn initialized_worker_threads() -> Option<usize> {
    CPU_POOL.get().map(ThreadPool::current_num_threads)
}

pub(crate) fn should_parallelize(tasks: usize, work: usize) -> bool {
    tasks > 1 && work >= PARALLEL_MIN_WORK && worker_threads() > 1
}

pub(crate) fn install<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    pool().install(operation)
}

/// Runs blocking direct-I/O work outside the Rayon compute pool. At most eight workers are
/// needed because routed layers select eight experts; on hybrid CPUs they prefer otherwise-unused
/// SMT siblings so synchronous storage waits cannot occupy physical compute workers.
pub(crate) fn spawn_io<R: Send + 'static>(
    operation: impl FnOnce() -> R + Send + 'static,
) -> IoTask<R> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let task = Box::new(move || {
        let result = operation();
        let _ = sender.send(result);
    });
    let io_pool = IO_POOL.get_or_init(|| IoPool::new(worker_threads().clamp(1, 8)));
    let (queue, ready) = &*io_pool.queue;
    queue
        .lock()
        .expect("inference I/O queue lock is not poisoned")
        .push_back(task);
    ready.notify_one();
    IoTask { receiver }
}

pub(crate) fn spawn_compute<R: Send + 'static>(
    operation: impl FnOnce() -> R + Send + 'static,
) -> IoTask<R> {
    let (sender, receiver) = mpsc::sync_channel(1);
    pool().spawn(move || {
        let result = operation();
        let _ = sender.send(result);
    });
    IoTask { receiver }
}

impl IoPool {
    fn new(workers: usize) -> Self {
        let queue = Arc::new((Mutex::new(VecDeque::<IoOperation>::new()), Condvar::new()));
        let affinity = physical_core_first_cpu_order();
        let physical_cores = physical_core_count();
        let siblings = affinity
            .into_iter()
            .skip(physical_cores)
            .collect::<Vec<_>>();
        for worker in 0..workers {
            let queue = Arc::clone(&queue);
            let cpu = siblings.get(worker).copied();
            std::thread::Builder::new()
                .name(format!("urb-io-{worker}"))
                .spawn(move || {
                    if let Some(cpu) = cpu {
                        pin_current_thread(cpu);
                    }
                    loop {
                        let task = {
                            let (tasks, ready) = &*queue;
                            let mut tasks = tasks
                                .lock()
                                .expect("inference I/O queue lock is not poisoned");
                            while tasks.is_empty() {
                                tasks = ready
                                    .wait(tasks)
                                    .expect("inference I/O queue lock is not poisoned");
                            }
                            tasks
                                .pop_front()
                                .expect("a non-empty inference I/O queue has one task")
                        };
                        task();
                    }
                })
                .expect("inference I/O worker creation succeeds");
        }
        Self { queue }
    }
}

fn pool() -> &'static ThreadPool {
    CPU_POOL.get_or_init(|| {
        let affinity = Arc::new(physical_core_first_cpu_order());
        let worker_affinity = Arc::clone(&affinity);
        ThreadPoolBuilder::new()
            .num_threads(recommended_worker_threads())
            .thread_name(|index| format!("urb-cpu-{index}"))
            .start_handler(move |index| {
                if let Some(&cpu) = worker_affinity.get(index) {
                    pin_current_thread(cpu);
                }
            })
            .build()
            .expect("default Rayon CPU pool configuration is valid")
    })
}

/// Default compute width: one worker per available physical core, falling back to the platform's
/// logical parallelism when topology is unavailable. Explicit `--threads` still overrides this.
pub fn recommended_worker_threads() -> usize {
    let physical = physical_core_count();
    if physical != 0 {
        physical
    } else {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }
}

/// Orders available logical CPUs so every physical core receives one worker before SMT siblings.
/// Hybrid Intel parts expose P-core siblings first in CPU-number order and E-cores as singleton
/// cores, which gives `--threads 20` one worker on each physical core of a 14700K.
#[cfg(target_os = "linux")]
fn physical_core_first_cpu_order() -> Vec<usize> {
    physical_core_first(&available_cpu_topology())
}

#[cfg(target_os = "linux")]
fn available_cpu_topology() -> Vec<(usize, usize, usize)> {
    let allowed = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("Cpus_allowed_list:")
                    .and_then(|value| parse_cpu_list(value.trim()))
            })
        })
        .map(|cpus| cpus.into_iter().collect::<HashSet<_>>());
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return Vec::new();
    };
    let mut topology = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let cpu = name.to_str()?.strip_prefix("cpu")?.parse::<usize>().ok()?;
            if allowed
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&cpu))
            {
                return None;
            }
            let path = entry.path().join("topology");
            let package = fs::read_to_string(path.join("physical_package_id"))
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()?;
            let core = fs::read_to_string(path.join("core_id"))
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()?;
            Some((cpu, package, core))
        })
        .collect::<Vec<_>>();
    topology.sort_unstable();
    topology
}

#[cfg(target_os = "linux")]
fn physical_core_count() -> usize {
    available_cpu_topology()
        .into_iter()
        .map(|(_, package, core)| (package, core))
        .collect::<HashSet<_>>()
        .len()
}

#[cfg(not(target_os = "linux"))]
fn physical_core_count() -> usize {
    0
}

#[cfg(not(target_os = "linux"))]
fn physical_core_first_cpu_order() -> Vec<usize> {
    Vec::new()
}

fn physical_core_first(topology: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut primary = Vec::new();
    let mut siblings = Vec::new();
    for &(cpu, package, core) in topology {
        if seen.insert((package, core)) {
            primary.push(cpu);
        } else {
            siblings.push(cpu);
        }
    }
    primary.extend(siblings);
    primary
}

fn parse_cpu_list(value: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.split(',') {
        let (start, end): (usize, usize) = match part.split_once('-') {
            Some((start, end)) => (start.parse().ok()?, end.parse().ok()?),
            None => {
                let cpu = part.parse().ok()?;
                (cpu, cpu)
            }
        };
        if end < start {
            return None;
        }
        cpus.extend(start..=end);
    }
    Some(cpus)
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) {
    const CPU_SET_BITS: usize = 1024;
    const WORDS: usize = CPU_SET_BITS / usize::BITS as usize;
    #[repr(C)]
    struct CpuSet {
        words: [usize; WORDS],
    }
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
    }
    if cpu >= CPU_SET_BITS {
        return;
    }
    let mut set = CpuSet { words: [0; WORDS] };
    set.words[cpu / usize::BITS as usize] |= 1usize << (cpu % usize::BITS as usize);
    // SAFETY: `set` matches Linux's fixed-size cpu_set_t bitmask and remains alive for the call;
    // pid 0 applies affinity only to the calling Rayon worker. Failure is an intentional
    // best-effort fallback to the scheduler's existing policy.
    unsafe {
        sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) {}

#[cfg(all(target_os = "linux", test))]
fn current_cpu() -> Option<usize> {
    unsafe extern "C" {
        fn sched_getcpu() -> i32;
    }
    // SAFETY: `sched_getcpu` has no arguments or memory side effects.
    let cpu = unsafe { sched_getcpu() };
    (cpu >= 0).then_some(cpu as usize)
}

fn validate_worker_count(pool: &ThreadPool, requested: usize) -> Result<(), ThreadPoolError> {
    if pool.current_num_threads() == requested {
        Ok(())
    } else {
        Err(ThreadPoolError(format!(
            "CPU pool already has {} workers and cannot be resized to {requested}",
            pool.current_num_threads()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_work_stays_inline() {
        assert!(!should_parallelize(128, PARALLEL_MIN_WORK - 1));
        assert!(!should_parallelize(1, PARALLEL_MIN_WORK));
        assert!(configure_threads(MAX_WORKER_THREADS + 1).is_err());
    }

    #[test]
    fn cpu_lists_and_physical_core_order_are_deterministic() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(
            physical_core_first(&[(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 2)]),
            vec![0, 2, 4, 1, 3]
        );
    }

    #[test]
    fn detached_io_and_compute_work_return_results() {
        let io = spawn_io(|| 6usize * 7);
        let compute = spawn_compute(|| 9usize * 5);
        assert_eq!(io.join(), 42);
        assert_eq!(compute.join(), 45);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rayon_workers_are_pinned_in_physical_core_first_order() {
        let expected = physical_core_first_cpu_order();
        let workers = worker_threads();
        if expected.len() < workers {
            return;
        }
        let mut actual = pool().broadcast(|context| (context.index(), current_cpu()));
        actual.sort_by_key(|&(index, _)| index);
        assert_eq!(
            actual
                .into_iter()
                .map(|(_, cpu)| cpu.expect("Linux reports each worker CPU"))
                .collect::<Vec<_>>(),
            expected[..workers]
        );
    }
}
