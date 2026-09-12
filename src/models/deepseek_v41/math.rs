//! DeepSeek-V4.1 scalar dtype boundaries.

use crate::execution::{install, should_parallelize, PARALLEL_MIN_WORK};
use crate::math::{rms_norm, simulate_e4m3_activation, MathError, RouteChoice};
use crate::model::{WeightError, WeightMatrix};
use crate::models::deepseek_v4::math::{
    bounded_swiglu, paired_rope, round_to_bf16_in_place, route_sqrt_softplus,
    sparse_attention_with_sink, DeepseekMathError, HyperConnectionMix,
};
use rayon::prelude::*;

pub use crate::models::deepseek_v4::math::{
    bounded_swiglu as bounded_swiglu_reference, paired_rope as paired_rope_reference,
    route_sqrt_softplus as route_sqrt_softplus_reference,
    sparse_attention_with_sink as sparse_attention_with_sink_reference,
};

/// V4.1 uses 32-wide dynamic activation blocks for every native MX matrix.
pub fn linear(weight: &WeightMatrix, input: &[f32]) -> Result<Vec<f32>, DeepseekMathError> {
    let quantized_input;
    let uses_mx_quantization = weight.uses_mx_activation_quantization();
    let input = if uses_mx_quantization {
        quantized_input = simulate_e4m3_activation(input, 32).map_err(|error| {
            DeepseekMathError::Invalid(format!("V4.1 activation quantization failed: {error}"))
        })?;
        &quantized_input
    } else {
        input
    };
    let mut output = weight.matvec_fp32_accum(input)?;
    if uses_mx_quantization {
        round_to_bf16_in_place(&mut output)?;
    }
    Ok(output)
}

pub fn bf16_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, V41MathError> {
    let mut output = rms_norm(input, weight, eps)?;
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

pub fn post_bf16(
    branch: &[f32],
    residual: &[f32],
    hidden_size: usize,
    mix: &HyperConnectionMix,
) -> Result<Vec<f32>, DeepseekMathError> {
    if branch.len() != hidden_size
        || mix.post.len() != mix.pre.len()
        || mix.combination.len() != mix.pre.len() * mix.pre.len()
        || residual.len() != mix.pre.len() * hidden_size
    {
        return Err(DeepseekMathError::Shape(
            "V4.1 branch/residual/mix do not match HC geometry".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; residual.len()];
    for target in 0..mix.pre.len() {
        for feature in 0..hidden_size {
            let mut value = mix.post[target] * branch[feature];
            for source in 0..mix.pre.len() {
                value += mix.combination[target * mix.pre.len() + source]
                    * residual[source * hidden_size + feature];
            }
            output[target * hidden_size + feature] = value;
        }
    }
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

pub fn collapse_bf16(
    hidden: &[f32],
    pre_mix: &[f32],
    hidden_size: usize,
) -> Result<Vec<f32>, DeepseekMathError> {
    if pre_mix.is_empty() || hidden.len() != pre_mix.len().saturating_mul(hidden_size) {
        return Err(DeepseekMathError::Shape(
            "V4.1 final mHC collapse geometry is invalid".to_owned(),
        ));
    }
    let mut output = vec![0.0; hidden_size];
    for (copy, &weight) in pre_mix.iter().enumerate() {
        for feature in 0..hidden_size {
            output[feature] += weight * hidden[copy * hidden_size + feature];
        }
    }
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

/// V4.1 sparse kernel with independent heads assigned to the inference worker pool.
/// Each head retains the scalar FP32 reduction order; its exponentials are cast to BF16 before
/// the value GEMM exactly as in the published TileLang kernel.
pub fn sparse_attention(
    queries: &[f32],
    head_count: usize,
    head_dim: usize,
    keys_values: &[Vec<f32>],
    selected: &[usize],
    sinks: &[f32],
    softmax_scale: f32,
) -> Result<Vec<f32>, DeepseekMathError> {
    if head_count == 0
        || head_dim == 0
        || queries.len() != head_count * head_dim
        || sinks.len() != head_count
        || selected.is_empty()
        || selected.iter().any(|&index| index >= keys_values.len())
        || keys_values.iter().any(|row| row.len() != head_dim)
    {
        return Err(DeepseekMathError::Shape(
            "V4.1 sparse attention geometry is invalid".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; queries.len()];
    let compute_head = |head: usize, output: &mut [f32]| -> Result<(), DeepseekMathError> {
        let query = &queries[head * head_dim..(head + 1) * head_dim];
        let mut scores = selected
            .iter()
            .map(|&index| {
                query
                    .iter()
                    .zip(&keys_values[index])
                    .fold(0.0f32, |sum, (&q, &k)| sum + q * k)
                    * softmax_scale
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(sinks[head], f32::max);
        for score in &mut scores {
            *score = (*score - maximum).exp();
        }
        let denominator = scores.iter().sum::<f32>() + (sinks[head] - maximum).exp();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(DeepseekMathError::NonFinite);
        }
        // The denominator must use the original FP32 exponentials, before this bulk BF16 cast.
        round_to_bf16_in_place(&mut scores)?;
        for (&index, &exponential) in selected.iter().zip(&scores) {
            for (value, &key_value) in output.iter_mut().zip(&keys_values[index]) {
                *value += exponential * key_value;
            }
        }
        for value in output {
            *value /= denominator;
        }
        Ok(())
    };
    let work = queries
        .len()
        .saturating_mul(selected.len())
        .saturating_mul(2);
    // Per-head scratch and softmax make tiny dispatches less efficient than dense row kernels.
    // Keep the first few decoding positions serial, even when the model has many query heads.
    if work >= 4 * PARALLEL_MIN_WORK && should_parallelize(head_count, work) {
        install(|| {
            output
                .par_chunks_mut(head_dim)
                .enumerate()
                .try_for_each(|(head, output)| compute_head(head, output))
        })?;
    } else {
        for (head, output) in output.chunks_mut(head_dim).enumerate() {
            compute_head(head, output)?;
        }
    }
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

pub fn engram_inject(
    hidden: &[f32],
    key_value: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    hidden_size: usize,
    hc_mult: usize,
    norm_eps: f32,
) -> Result<Vec<f32>, DeepseekMathError> {
    if hidden.len() != hidden_size.saturating_mul(hc_mult)
        || key_value.len() != hidden_size.saturating_mul(hc_mult + 1)
        || q_weight.len() != hidden.len()
        || k_weight.len() != hidden.len()
        || norm_eps <= 0.0
        || !norm_eps.is_finite()
    {
        return Err(DeepseekMathError::Shape(
            "Engram hidden/key/value/gate geometry is invalid".to_owned(),
        ));
    }
    let (key, value) = key_value.split_at(hidden_size * hc_mult);
    let mut output = hidden.to_vec();
    for copy in 0..hc_mult {
        let start = copy * hidden_size;
        let h = &hidden[start..start + hidden_size];
        let k = &key[start..start + hidden_size];
        let h_ms = h.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32;
        let k_ms = k.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32;
        let reciprocal = (h_ms + norm_eps).sqrt().recip() * (k_ms + norm_eps).sqrt().recip();
        let dot = (0..hidden_size)
            .map(|feature| {
                h[feature] * q_weight[start + feature] * k_weight[start + feature] * k[feature]
            })
            .sum::<f32>()
            * reciprocal
            * (hidden_size as f32).sqrt().recip();
        let transformed = dot.abs().max(1e-6).sqrt().copysign(dot);
        let gate = 1.0 / (1.0 + (-transformed).exp());
        for feature in 0..hidden_size {
            output[start + feature] += gate * value[feature];
        }
    }
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

#[derive(Debug)]
pub enum V41MathError {
    Math(MathError),
    Deepseek(DeepseekMathError),
    Weight(WeightError),
}

impl std::fmt::Display for V41MathError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Math(error) => error.fmt(formatter),
            Self::Deepseek(error) => error.fmt(formatter),
            Self::Weight(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for V41MathError {}

impl From<MathError> for V41MathError {
    fn from(value: MathError) -> Self {
        Self::Math(value)
    }
}

impl From<DeepseekMathError> for V41MathError {
    fn from(value: DeepseekMathError) -> Self {
        Self::Deepseek(value)
    }
}

impl From<WeightError> for V41MathError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

// Keep the imported reference primitives visibly tied to this model boundary.
#[allow(dead_code)]
fn _reference_symbols(mix: &HyperConnectionMix, routes: &[RouteChoice]) -> (usize, usize) {
    let _ = (
        bounded_swiglu as fn(&[f32], &[f32], f32) -> _,
        paired_rope as fn(&mut [f32], usize, f32, Option<usize>, f32, usize, usize, bool) -> _,
        route_sqrt_softplus as fn(&[f32], Option<&[f32]>, Option<&[usize]>, usize, f32) -> _,
        sparse_attention_with_sink
            as fn(&[f32], usize, usize, &[Vec<f32>], &[usize], &[f32], f32) -> _,
    );
    (mix.pre.len(), routes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keep the original scalar attention and an independent BF16 cast as a reduction-order
    // oracle. In particular, summing BF16 exponentials changes the denominator and must fail.
    fn scalar_bf16(value: f32) -> Result<f32, DeepseekMathError> {
        let bits = value.to_bits();
        let rounded = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000);
        if value.is_finite() && rounded.is_finite() {
            Ok(rounded)
        } else {
            Err(DeepseekMathError::NonFinite)
        }
    }

    fn scalar_attention(
        queries: &[f32],
        head_count: usize,
        head_dim: usize,
        keys_values: &[Vec<f32>],
        selected: &[usize],
        sinks: &[f32],
        softmax_scale: f32,
    ) -> Result<Vec<f32>, DeepseekMathError> {
        let mut output = vec![0.0f32; queries.len()];
        for head in 0..head_count {
            let query = &queries[head * head_dim..(head + 1) * head_dim];
            let scores = selected
                .iter()
                .map(|&index| {
                    query
                        .iter()
                        .zip(&keys_values[index])
                        .fold(0.0f32, |sum, (&q, &k)| sum + q * k)
                        * softmax_scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(sinks[head], f32::max);
            let exponentials = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = exponentials.iter().sum::<f32>() + (sinks[head] - maximum).exp();
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(DeepseekMathError::NonFinite);
            }
            for (&index, &exponential) in selected.iter().zip(&exponentials) {
                let exponential = scalar_bf16(exponential)?;
                for feature in 0..head_dim {
                    output[head * head_dim + feature] += exponential * keys_values[index][feature];
                }
            }
            for feature in 0..head_dim {
                output[head * head_dim + feature] /= denominator;
            }
        }
        output.into_iter().map(scalar_bf16).collect()
    }

    fn values(length: usize, salt: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                ((index
                    .wrapping_mul(7919)
                    .wrapping_add(salt.wrapping_mul(6271))
                    % 8191) as f32
                    - 4095.0)
                    / 2048.0
            })
            .collect()
    }

    #[test]
    fn sparse_attention_matches_scalar_bits_for_head_tails_and_duplicate_keys() {
        // Cover both the serial and parallel work thresholds, odd feature/selected counts and
        // the real model's head geometry. Selected IDs deliberately repeat and are unsorted.
        for (heads, dim, rows, selections) in [
            (1, 7, 19, 5),
            (3, 17, 11, 1),
            (17, 129, 31, 129),
            (64, 512, 1, 1),
            (64, 512, 2, 2),
            (64, 512, 128, 128),
        ] {
            let queries = values(heads * dim, 3);
            let keys = (0..rows).map(|row| values(dim, row)).collect::<Vec<_>>();
            let selected = (0..selections)
                .map(|index| (index * 7 + index / 3) % rows)
                .collect::<Vec<_>>();
            let sinks = values(heads, 19);
            let scale = (dim as f32).sqrt().recip();
            let expected =
                scalar_attention(&queries, heads, dim, &keys, &selected, &sinks, scale).unwrap();
            let actual =
                sparse_attention(&queries, heads, dim, &keys, &selected, &sinks, scale).unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "heads={heads}, dim={dim}, selected={selections}"
            );
        }
    }

    #[test]
    fn sparse_attention_preserves_nonfinite_errors_and_negative_infinite_sink() {
        for (heads, dim, rows) in [(1, 7, 3), (17, 129, 67)] {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for source in 0..3 {
                    let mut queries = values(heads * dim, 3);
                    let mut keys = (0..rows).map(|row| values(dim, row)).collect::<Vec<_>>();
                    let mut sinks = values(heads, 19);
                    let selected = (0..rows).rev().collect::<Vec<_>>();
                    match source {
                        0 => queries[heads * dim - 1] = invalid,
                        1 => keys[rows - 1][dim - 1] = invalid,
                        _ => sinks[heads - 1] = invalid,
                    }
                    let expected =
                        scalar_attention(&queries, heads, dim, &keys, &selected, &sinks, 0.125);
                    let actual =
                        sparse_attention(&queries, heads, dim, &keys, &selected, &sinks, 0.125);
                    match expected {
                        Ok(expected) => assert_eq!(
                            actual
                                .unwrap()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        ),
                        Err(DeepseekMathError::NonFinite) => {
                            assert!(matches!(actual, Err(DeepseekMathError::NonFinite)));
                        }
                        Err(error) => panic!("unexpected scalar error: {error}"),
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_attention_rejects_invalid_geometry() {
        let row = vec![1.0; 7];
        for result in [
            sparse_attention(&row, 0, 7, &[row.clone()], &[0], &[0.0], 1.0),
            sparse_attention(&row, 1, 0, &[row.clone()], &[0], &[0.0], 1.0),
            sparse_attention(&row, 2, 7, &[row.clone()], &[0], &[0.0], 1.0),
            sparse_attention(&row, 1, 7, &[row.clone()], &[0], &[], 1.0),
            sparse_attention(&row, 1, 7, &[row.clone()], &[], &[0.0], 1.0),
            sparse_attention(&row, 1, 7, &[row.clone()], &[1], &[0.0], 1.0),
            sparse_attention(&row, 1, 7, &[vec![1.0; 6]], &[0], &[0.0], 1.0),
        ] {
            assert!(matches!(result, Err(DeepseekMathError::Shape(_))));
        }
    }
}
