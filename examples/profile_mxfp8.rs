use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use urbilateria::execution::{configure_threads, worker_threads};
use urbilateria::math::MxFp8Matrix;

fn main() -> Result<(), Box<dyn Error>> {
    let threads = environment_usize("URB_BENCH_THREADS")?.unwrap_or(20);
    let rows = environment_usize("URB_BENCH_ROWS")?.unwrap_or(2_048);
    let columns = environment_usize("URB_BENCH_COLS")?.unwrap_or(6_144);
    let block_rows = environment_usize("URB_BENCH_BLOCK_ROWS")?.unwrap_or(1);
    let block_columns = environment_usize("URB_BENCH_BLOCK_COLS")?.unwrap_or(32);
    let iterations = environment_usize("URB_BENCH_ITERATIONS")?.unwrap_or(20);
    let batch = environment_usize("URB_BENCH_BATCH")?.unwrap_or(1);
    let batch_kernel = environment_usize("URB_BENCH_BATCH_KERNEL")?.unwrap_or(0) != 0;
    let exponent_zero_period = environment_usize("URB_BENCH_EXPONENT_ZERO_PERIOD")?;
    if rows == 0
        || columns == 0
        || block_rows == 0
        || block_columns == 0
        || iterations == 0
        || batch == 0
    {
        return Err("benchmark dimensions and iterations must be non-zero".into());
    }
    configure_threads(threads)?;

    let values = if let Some(period) = exponent_zero_period {
        let codes = [0x18, 0x20, 0x30, 0x38, 0x40, 0x48, 0xb8];
        (0..rows * columns)
            .map(|index| {
                if period != 0 && index % period == 0 {
                    0x00
                } else {
                    codes[index.wrapping_mul(17) % codes.len()]
                }
            })
            .collect::<Vec<_>>()
    } else {
        let codes = [0x00, 0x18, 0x20, 0x30, 0x38, 0x40, 0x48, 0xb8];
        (0..rows * columns)
            .map(|index| codes[index.wrapping_mul(17) % codes.len()])
            .collect::<Vec<_>>()
    };
    let scale_columns = columns.div_ceil(block_columns);
    let scales = (0..rows.div_ceil(block_rows) * scale_columns)
        .map(|index| 119 + (index.wrapping_mul(13) % 17) as u8)
        .collect::<Vec<_>>();
    let input = (0..batch * columns)
        .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
        .collect::<Vec<_>>();
    let matrix =
        MxFp8Matrix::from_packed(rows, columns, block_rows, block_columns, values, scales)?;

    let run = || -> Result<Vec<f32>, Box<dyn Error>> {
        if batch_kernel {
            Ok(matrix.matmul_rows(black_box(&input), batch)?)
        } else {
            Ok(input
                .chunks_exact(columns)
                .map(|input| matrix.matvec(black_box(input)))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect())
        }
    };
    black_box(run()?);
    let started = Instant::now();
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let output = black_box(run()?);
        checksum = output.iter().fold(0u64, |state, value| {
            state.rotate_left(7) ^ u64::from(value.to_bits())
        });
        black_box(output);
    }
    let elapsed = started.elapsed();
    let work = rows as f64 * columns as f64 * batch as f64 * iterations as f64;
    println!(
        "threads={} rows={} columns={} block={}x{} batch={} batch_kernel={} iterations={} elapsed_ms={:.3} gmac_s={:.3} checksum={checksum}",
        worker_threads(),
        rows,
        columns,
        block_rows,
        block_columns,
        batch,
        batch_kernel,
        iterations,
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
