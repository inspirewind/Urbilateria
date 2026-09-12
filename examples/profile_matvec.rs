use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use urbilateria::execution::{configure_threads, worker_threads};
use urbilateria::model::DenseMatrix;
use urbilateria::profiling::{ProfileSession, ProfileStage};

fn main() -> Result<(), Box<dyn Error>> {
    let requested_threads = environment_usize("URB_BENCH_THREADS")?
        .unwrap_or(std::thread::available_parallelism()?.get());
    let default_dimension = if cfg!(debug_assertions) { 256 } else { 4_096 };
    let default_iterations = if cfg!(debug_assertions) { 5 } else { 25 };
    let rows = environment_usize("URB_BENCH_ROWS")?.unwrap_or(default_dimension);
    let columns = environment_usize("URB_BENCH_COLS")?.unwrap_or(default_dimension);
    let iterations = environment_usize("URB_BENCH_ITERATIONS")?.unwrap_or(default_iterations);
    if rows == 0 || columns == 0 || iterations == 0 {
        return Err("benchmark dimensions and iterations must be greater than zero".into());
    }
    configure_threads(requested_threads)?;

    let values = (0..rows.checked_mul(columns).ok_or("matrix shape overflows")?)
        .map(|index| ((index.wrapping_mul(17) % 257) as f32 - 128.0) / 256.0)
        .collect::<Vec<_>>();
    let input = (0..columns)
        .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
        .collect::<Vec<_>>();
    let matrix = DenseMatrix::new(rows, columns, values)?;

    // Initialize workers and warm caches outside the measured samples.
    black_box(matrix.matvec(black_box(&input))?);

    let profile = ProfileSession::start_with_threads(Some(requested_threads));
    let mut samples_ns = Vec::with_capacity(iterations);
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let started = Instant::now();
        let output = black_box(matrix.matvec(black_box(&input))?);
        samples_ns.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        checksum = output.iter().fold(0u64, |state, value| {
            state.rotate_left(7) ^ u64::from(value.to_bits())
        });
        black_box(output);
    }
    let report = profile.finish();
    samples_ns.sort_unstable();
    let kernel = report
        .stage(ProfileStage::MatvecF32)
        .ok_or("profiler did not record the F32 matvec kernel")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "case": "dense_f32_matvec",
            "rows": rows,
            "columns": columns,
            "iterations": iterations,
            "requested_threads": requested_threads,
            "worker_threads": worker_threads(),
            "parallel_min_work": urbilateria::execution::PARALLEL_MIN_WORK,
            "min_ns": samples_ns[0],
            "p50_ns": percentile(&samples_ns, 1, 2),
            "p95_ns": percentile(&samples_ns, 19, 20),
            "max_ns": samples_ns[samples_ns.len() - 1],
            "aggregate_gmac_per_second": kernel.work_items_per_second() / 1_000_000_000.0,
            "checksum": checksum,
            "profile": report,
        }))?
    );
    Ok(())
}

fn percentile(samples: &[u64], numerator: usize, denominator: usize) -> u64 {
    let rank = samples
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator);
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

fn environment_usize(name: &str) -> Result<Option<usize>, Box<dyn Error>> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid {name}={value:?}: {error}").into())
        })
        .transpose()
}
