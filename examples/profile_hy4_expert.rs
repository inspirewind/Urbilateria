use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use urbilateria::execution::configure_threads;
use urbilateria::models::hy4::expert::Hy4Expert;
use urbilateria::storage::TensorIndex;
use urbilateria::ModelConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: profile_hy4_expert MODEL_DIR [ITERATIONS]")?;
    let iterations = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    configure_threads(20)?;
    let ModelConfig::Hy4(config) = ModelConfig::load(&directory)? else {
        return Err("checkpoint is not Hy4".into());
    };
    let index = TensorIndex::open(&directory)?;
    if std::env::var_os("URB_BENCH_SCAN_CODES").is_some() {
        let name = "model.layers.1.mlp.experts.gate_up_proj";
        let down_name = "model.layers.1.mlp.experts.down_proj";
        let rows = config.moe_intermediate_size * 2;
        let bytes = rows * config.hidden_size;
        let values = index.read_range(name, 0, bytes)?;
        let exponent_zero = values.iter().filter(|&&code| code & 0x7f < 8).count();
        let all_normal_blocks = values
            .chunks_exact(32)
            .filter(|block| block.iter().all(|code| code & 0x7f >= 8))
            .count();
        println!(
            "gate_up_values={} exponent_zero={} ({:.3}%) all_normal_blocks={}/{} ({:.3}%)",
            values.len(),
            exponent_zero,
            exponent_zero as f64 * 100.0 / values.len() as f64,
            all_normal_blocks,
            values.len() / 32,
            all_normal_blocks as f64 * 3200.0 / values.len() as f64,
        );
        for tensor_name in [
            name,
            "model.layers.1.mlp.experts.gate_up_proj_scale",
            down_name,
            "model.layers.1.mlp.experts.down_proj_scale",
        ] {
            let tensor = index.require(tensor_name)?;
            println!(
                "tensor={tensor_name} offset={} mod512={} mod4096={} len={} len_mod4096={}",
                tensor.data_offset,
                tensor.data_offset % 512,
                tensor.data_offset % 4096,
                tensor.data_len,
                tensor.data_len % 4096,
            );
        }
    }
    let expert = Hy4Expert::load(&index, &config, 1, 0, 38_928_384)?;
    let input = (0..config.hidden_size)
        .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
        .collect::<Vec<_>>();
    black_box(expert.forward_dequantized_reference(&input, config.swiglu_limit as f32)?);
    let started = Instant::now();
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let output =
            black_box(expert.forward_dequantized_reference(&input, config.swiglu_limit as f32)?);
        checksum = output.iter().fold(0u64, |state, value| {
            state.rotate_left(7) ^ u64::from(value.to_bits())
        });
        black_box(output);
    }
    println!(
        "iterations={iterations} elapsed_ms={:.3} expert_s={:.3} checksum={checksum}",
        started.elapsed().as_secs_f64() * 1_000.0,
        iterations as f64 / started.elapsed().as_secs_f64(),
    );
    Ok(())
}
