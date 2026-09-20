use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use urbilateria::execution::{configure_threads, worker_threads};
use urbilateria::math::MxFp4Matrix;

fn main() -> Result<(), Box<dyn Error>> {
    let threads = environment_usize("URB_BENCH_THREADS")?.unwrap_or(20);
    let rows = environment_usize("URB_BENCH_ROWS")?.unwrap_or(2_048);
    let columns = environment_usize("URB_BENCH_COLS")?.unwrap_or(4_096);
    let group_size = environment_usize("URB_BENCH_GROUP_SIZE")?.unwrap_or(128);
    let batch = environment_usize("URB_BENCH_BATCH")?.unwrap_or(1);
    let iterations = environment_usize("URB_BENCH_ITERATIONS")?.unwrap_or(200);
    if rows == 0 || columns == 0 || group_size == 0 || batch == 0 || iterations == 0 {
        return Err(
            "benchmark dimensions, group size, batch, and iterations must be non-zero".into(),
        );
    }
    configure_threads(threads)?;

    let row_bytes = columns.div_ceil(2);
    let groups = columns.div_ceil(group_size);
    let packed = (0..rows * row_bytes)
        .map(|index| {
            let low = (index.wrapping_mul(7) % 16) as u8;
            let high = (index.wrapping_mul(11).wrapping_add(3) % 16) as u8;
            low | (high << 4)
        })
        .collect::<Vec<_>>();
    let scales = (0..rows * groups)
        .map(|index| 119 + (index.wrapping_mul(13) % 17) as u8)
        .collect::<Vec<_>>();
    let input = (0..batch * columns)
        .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
        .collect::<Vec<_>>();
    let matrix = MxFp4Matrix::from_packed(rows, columns, group_size, packed, scales)?;

    if batch > 1 {
        let expected = input
            .chunks_exact(columns)
            .map(|input| matrix.matvec(input))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let actual = matrix.matmul_rows(&input, batch)?;
        if actual
            .iter()
            .map(|value| value.to_bits())
            .ne(expected.iter().map(|value| value.to_bits()))
        {
            return Err("batched MXFP4 output differs from independent matvecs".into());
        }
        let started = Instant::now();
        for _ in 0..iterations {
            for input in black_box(&input).chunks_exact(columns) {
                black_box(matrix.matvec(black_box(input))?);
            }
        }
        let independent = started.elapsed();
        black_box(matrix.matmul_rows(black_box(&input), batch)?);
        let started = Instant::now();
        let mut checksum = 0u64;
        for _ in 0..iterations {
            let output = black_box(matrix.matmul_rows(black_box(&input), batch)?);
            checksum = output.iter().fold(0u64, |state, value| {
                state.rotate_left(7) ^ u64::from(value.to_bits())
            });
            black_box(output);
        }
        let batched = started.elapsed();
        let work = rows as f64 * columns as f64 * batch as f64 * iterations as f64;
        println!(
            "threads={} rows={} columns={} group_size={} batch={} iterations={} independent_ms={:.3} batched_ms={:.3} speedup={:.3} independent_gmac_s={:.3} batched_gmac_s={:.3} checksum={checksum}",
            worker_threads(),
            rows,
            columns,
            group_size,
            batch,
            iterations,
            independent.as_secs_f64() * 1_000.0,
            batched.as_secs_f64() * 1_000.0,
            independent.as_secs_f64() / batched.as_secs_f64(),
            work / independent.as_secs_f64() / 1.0e9,
            work / batched.as_secs_f64() / 1.0e9,
        );
        return Ok(());
    }

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
        "threads={} rows={} columns={} group_size={} batch={} iterations={} elapsed_ms={:.3} gmac_s={:.3} checksum={checksum}",
        worker_threads(),
        rows,
        columns,
        group_size,
        batch,
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
