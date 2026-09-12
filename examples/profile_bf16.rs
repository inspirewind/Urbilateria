use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use urbilateria::execution::{configure_threads, worker_threads};
use urbilateria::model::Bf16Matrix;

fn main() -> Result<(), Box<dyn Error>> {
    let rows = environment_usize("URB_BENCH_ROWS")?.unwrap_or(6_144);
    let columns = environment_usize("URB_BENCH_COLS")?.unwrap_or(6_144);
    let iterations = environment_usize("URB_BENCH_ITERATIONS")?.unwrap_or(20);
    let threads = environment_usize("URB_BENCH_THREADS")?.unwrap_or(20);
    configure_threads(threads)?;
    let values = (0..rows.saturating_mul(columns))
        .flat_map(|index| {
            let value = ((index.wrapping_mul(29) % 257) as f32 - 128.0) / 64.0;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect::<Vec<_>>();
    let input = (0..columns)
        .map(|index| ((index.wrapping_mul(17) % 127) as f32 - 63.0) / 32.0)
        .collect::<Vec<_>>();
    let matrix = Bf16Matrix::from_le_bytes(rows, columns, values)?;
    black_box(matrix.matvec(black_box(&input))?);
    let started = Instant::now();
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let output = black_box(matrix.matvec(black_box(&input))?);
        checksum = output.iter().fold(0u64, |state, value| {
            state.rotate_left(7) ^ u64::from(value.to_bits())
        });
        black_box(output);
    }
    let elapsed = started.elapsed();
    let work = rows as f64 * columns as f64 * iterations as f64;
    println!(
        "threads={} rows={rows} columns={columns} iterations={iterations} elapsed_ms={:.3} gmac_s={:.3} checksum={checksum}",
        worker_threads(),
        elapsed.as_secs_f64() * 1_000.0,
        work / elapsed.as_secs_f64() / 1.0e9,
    );
    Ok(())
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
