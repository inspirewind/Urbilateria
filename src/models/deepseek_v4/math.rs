//! DeepSeek-V4-specific scalar reference math.

use crate::math::{simulate_e4m3_activation, RouteChoice};
use crate::model::{WeightError, WeightMatrix};
use crate::profiling::{span, ProfileStage};
use std::fmt;

#[derive(Debug)]
pub enum DeepseekMathError {
    Shape(String),
    Invalid(String),
    NonFinite,
    Weight(WeightError),
}

impl fmt::Display for DeepseekMathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(f, "DeepSeek-V4 shape error: {reason}"),
            Self::Invalid(reason) => write!(f, "invalid DeepSeek-V4 math: {reason}"),
            Self::NonFinite => f.write_str("DeepSeek-V4 math produced NaN or infinity"),
            Self::Weight(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for DeepseekMathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WeightError> for DeepseekMathError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

pub fn round_to_bf16_in_place(values: &mut [f32]) -> Result<(), DeepseekMathError> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection proves AVX2 support; every vector load/store is bounded to a
        // complete eight-value chunk, with invalid chunks and the tail replayed scalarly.
        return unsafe { round_to_bf16_in_place_avx2(values) };
    }
    for value in values.iter_mut() {
        *value = round_to_bf16(*value)?;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn round_to_bf16_in_place_avx2(values: &mut [f32]) -> Result<(), DeepseekMathError> {
    use std::arch::x86_64::*;

    let exponent_mask = _mm256_set1_epi32(0x7f80_0000);
    let retained_mask = _mm256_set1_epi32(0xffff_0000u32 as i32);
    let rounding_bias = _mm256_set1_epi32(0x7fff);
    let one = _mm256_set1_epi32(1);
    let mut offset = 0usize;
    while offset + 8 <= values.len() {
        let bits = _mm256_loadu_si256(values.as_ptr().add(offset).cast());
        let input_invalid =
            _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent_mask), exponent_mask);
        let retained_lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), one);
        let rounded = _mm256_and_si256(
            _mm256_add_epi32(bits, _mm256_add_epi32(rounding_bias, retained_lsb)),
            retained_mask,
        );
        let output_invalid =
            _mm256_cmpeq_epi32(_mm256_and_si256(rounded, exponent_mask), exponent_mask);
        if _mm256_movemask_epi8(_mm256_or_si256(input_invalid, output_invalid)) != 0 {
            for value in &mut values[offset..] {
                *value = round_to_bf16(*value)?;
            }
            return Ok(());
        }
        _mm256_storeu_si256(values.as_mut_ptr().add(offset).cast(), rounded);
        offset += 8;
    }
    for value in &mut values[offset..] {
        *value = round_to_bf16(*value)?;
    }
    Ok(())
}

fn round_to_bf16(value: f32) -> Result<f32, DeepseekMathError> {
    if !value.is_finite() {
        return Err(DeepseekMathError::NonFinite);
    }
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    let rounded = f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000);
    rounded
        .is_finite()
        .then_some(rounded)
        .ok_or(DeepseekMathError::NonFinite)
}

/// Executes one native linear layer, including the checkpoint's required activation-side MXFP8
/// quantize/dequantize simulation for MXFP8 and MXFP4 weights.
pub fn linear(weight: &WeightMatrix, input: &[f32]) -> Result<Vec<f32>, DeepseekMathError> {
    let input = if weight.uses_mx_activation_quantization() {
        let _profile = span(ProfileStage::DeepseekActivationQuantization);
        simulate_e4m3_activation(input, 128).map_err(|error| {
            DeepseekMathError::Invalid(format!("activation quantization failed: {error}"))
        })?
    } else {
        input.to_vec()
    };
    let mut output = weight.matvec(&input)?;
    if weight.uses_mx_activation_quantization() {
        // The release kernels accumulate in FP32 but materialize MX GEMM outputs as BF16.
        round_to_bf16_in_place(&mut output)?;
    }
    Ok(output)
}

pub fn linear_rows(
    weight: &WeightMatrix,
    start: usize,
    count: usize,
    input: &[f32],
) -> Result<Vec<f32>, DeepseekMathError> {
    let input = if weight.uses_mx_activation_quantization() {
        let _profile = span(ProfileStage::DeepseekActivationQuantization);
        simulate_e4m3_activation(input, 128).map_err(|error| {
            DeepseekMathError::Invalid(format!("activation quantization failed: {error}"))
        })?
    } else {
        input.to_vec()
    };
    let mut output = weight.matvec_rows(start, count, &input)?;
    if weight.uses_mx_activation_quantization() {
        round_to_bf16_in_place(&mut output)?;
    }
    Ok(output)
}

pub fn unit_rms_norm_in_place(
    values: &mut [f32],
    chunk_size: usize,
    eps: f32,
) -> Result<(), DeepseekMathError> {
    if values.is_empty() || chunk_size == 0 || values.len() % chunk_size != 0 {
        return Err(DeepseekMathError::Shape(
            "unit RMSNorm chunks do not cover the input".to_owned(),
        ));
    }
    validate_positive("RMSNorm epsilon", eps)?;
    for chunk in values.chunks_mut(chunk_size) {
        // The standalone model keeps this query normalization in BF16: square, reduction
        // result, epsilon addition, reciprocal square root, and the in-place multiply all
        // materialize in that dtype. The reduction itself uses an FP32 accumulator.
        let mut square_sum = 0.0f32;
        for &value in chunk.iter() {
            square_sum += round_to_bf16(value * value)?;
        }
        let mean_square = round_to_bf16(square_sum / chunk_size as f32)?;
        let shifted = round_to_bf16(mean_square + eps)?;
        let scale = round_to_bf16(shifted.sqrt().recip())?;
        for value in chunk {
            *value = round_to_bf16(*value * scale)?;
        }
    }
    Ok(())
}

/// Scalar form of the official sparse-attention kernel. The learned sink contributes only a
/// zero-valued softmax slot, hence it changes the denominator but never the value numerator.
pub fn sparse_attention_with_sink(
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
        || keys_values.iter().any(|values| values.len() != head_dim)
        || !softmax_scale.is_finite()
        || softmax_scale <= 0.0
    {
        return Err(DeepseekMathError::Shape(
            "sparse attention geometry or selected indices are invalid".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; queries.len()];
    for head in 0..head_count {
        let query = &queries[head * head_dim..(head + 1) * head_dim];
        let scores = selected
            .iter()
            .map(|&index| {
                query
                    .iter()
                    .zip(&keys_values[index])
                    .map(|(&left, &right)| left * right)
                    .sum::<f32>()
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
        for (&index, probability) in selected.iter().zip(
            exponentials
                .iter()
                .map(|exponential| exponential / denominator),
        ) {
            for feature in 0..head_dim {
                output[head * head_dim + feature] += probability * keys_values[index][feature];
            }
        }
    }
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionMix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    /// Row-major `[hc, hc]` near-doubly-stochastic residual mixing matrix.
    pub combination: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn hyper_connection_pre(
    hidden: &[f32],
    hidden_size: usize,
    hc_mult: usize,
    function: &WeightMatrix,
    scale: &[f32],
    base: &[f32],
    sinkhorn_iters: usize,
    norm_eps: f32,
    sinkhorn_eps: f32,
) -> Result<(Vec<f32>, HyperConnectionMix), DeepseekMathError> {
    let width = hidden_size
        .checked_mul(hc_mult)
        .ok_or_else(|| DeepseekMathError::Shape("HC width overflows".to_owned()))?;
    let mix_count = (2 + hc_mult)
        .checked_mul(hc_mult)
        .ok_or_else(|| DeepseekMathError::Shape("HC mix count overflows".to_owned()))?;
    if hidden.len() != width
        || function.rows() != mix_count
        || function.cols() != width
        || scale.len() != 3
        || base.len() != mix_count
        || hc_mult == 0
        || hidden_size == 0
        || sinkhorn_iters == 0
    {
        return Err(DeepseekMathError::Shape(
            "hidden/function/base/scale do not match HC geometry".to_owned(),
        ));
    }
    validate_positive("norm_eps", norm_eps)?;
    validate_positive("sinkhorn_eps", sinkhorn_eps)?;
    if hidden
        .iter()
        .chain(scale)
        .chain(base)
        .any(|value| !value.is_finite())
    {
        return Err(DeepseekMathError::NonFinite);
    }
    let mean_square = hidden
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / width as f64;
    let reciprocal_rms = (mean_square + f64::from(norm_eps)).sqrt().recip() as f32;
    let mut mixes = function.matvec(hidden)?;
    for value in &mut mixes {
        *value *= reciprocal_rms;
    }
    let mut pre = Vec::with_capacity(hc_mult);
    let mut post = Vec::with_capacity(hc_mult);
    for index in 0..hc_mult {
        pre.push(sigmoid(mixes[index] * scale[0] + base[index]) + sinkhorn_eps);
        post.push(2.0 * sigmoid(mixes[hc_mult + index] * scale[1] + base[hc_mult + index]));
    }
    let combination_offset = 2 * hc_mult;
    let mut combination = Vec::with_capacity(hc_mult * hc_mult);
    for index in 0..hc_mult * hc_mult {
        combination
            .push(mixes[combination_offset + index] * scale[2] + base[combination_offset + index]);
    }
    row_softmax_in_place(&mut combination, hc_mult)?;
    for value in &mut combination {
        *value += sinkhorn_eps;
    }
    normalize_columns(&mut combination, hc_mult, sinkhorn_eps);
    for _ in 1..sinkhorn_iters {
        normalize_rows(&mut combination, hc_mult, sinkhorn_eps);
        normalize_columns(&mut combination, hc_mult, sinkhorn_eps);
    }

    let mut collapsed = vec![0.0f64; hidden_size];
    for copy in 0..hc_mult {
        for feature in 0..hidden_size {
            collapsed[feature] +=
                f64::from(pre[copy]) * f64::from(hidden[copy * hidden_size + feature]);
        }
    }
    let mut collapsed = collapsed
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    round_to_bf16_in_place(&mut collapsed)?;
    Ok((
        collapsed,
        HyperConnectionMix {
            pre,
            post,
            combination,
        },
    ))
}

pub fn hyper_connection_post(
    branch: &[f32],
    residual: &[f32],
    hidden_size: usize,
    mix: &HyperConnectionMix,
) -> Result<Vec<f32>, DeepseekMathError> {
    let hc_mult = mix.pre.len();
    if branch.len() != hidden_size
        || mix.post.len() != hc_mult
        || mix.combination.len() != hc_mult * hc_mult
        || residual.len() != hc_mult * hidden_size
    {
        return Err(DeepseekMathError::Shape(
            "branch/residual/mix do not match HC geometry".to_owned(),
        ));
    }
    let mut output = vec![0.0f64; residual.len()];
    for target in 0..hc_mult {
        for feature in 0..hidden_size {
            let mut value = f64::from(mix.post[target]) * f64::from(branch[feature]);
            for source in 0..hc_mult {
                value += f64::from(mix.combination[target * hc_mult + source])
                    * f64::from(residual[source * hidden_size + feature]);
            }
            output[target * hidden_size + feature] = value;
        }
    }
    finite_vec(output)
}

#[allow(clippy::too_many_arguments)]
pub fn hyper_connection_head(
    hidden: &[f32],
    hidden_size: usize,
    hc_mult: usize,
    function: &WeightMatrix,
    scale: f32,
    base: &[f32],
    norm_eps: f32,
    hc_eps: f32,
) -> Result<Vec<f32>, DeepseekMathError> {
    let width = hidden_size * hc_mult;
    if hidden.len() != width
        || function.rows() != hc_mult
        || function.cols() != width
        || base.len() != hc_mult
    {
        return Err(DeepseekMathError::Shape(
            "head reduction does not match HC geometry".to_owned(),
        ));
    }
    validate_positive("norm_eps", norm_eps)?;
    validate_positive("hc_eps", hc_eps)?;
    let mean_square = hidden
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / width as f64;
    let reciprocal_rms = (mean_square + f64::from(norm_eps)).sqrt().recip() as f32;
    let mixes = function.matvec(hidden)?;
    let weights = mixes
        .iter()
        .zip(base)
        .map(|(&mix, &base)| sigmoid(mix * reciprocal_rms * scale + base) + hc_eps)
        .collect::<Vec<_>>();
    let mut output = vec![0.0f64; hidden_size];
    for copy in 0..hc_mult {
        for feature in 0..hidden_size {
            output[feature] +=
                f64::from(weights[copy]) * f64::from(hidden[copy * hidden_size + feature]);
        }
    }
    finite_vec(output)
}

pub fn route_sqrt_softplus(
    logits: &[f32],
    correction_bias: Option<&[f32]>,
    hash_experts: Option<&[usize]>,
    top_k: usize,
    routed_scaling_factor: f32,
) -> Result<Vec<RouteChoice>, DeepseekMathError> {
    if logits.is_empty()
        || top_k == 0
        || top_k > logits.len()
        || correction_bias.is_some_and(|bias| bias.len() != logits.len())
        || hash_experts.is_some_and(|experts| experts.len() != top_k)
    {
        return Err(DeepseekMathError::Shape(
            "router logits/bias/hash/top-k geometry is inconsistent".to_owned(),
        ));
    }
    validate_positive("routed_scaling_factor", routed_scaling_factor)?;
    if logits
        .iter()
        .chain(correction_bias.into_iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(DeepseekMathError::NonFinite);
    }
    let scores = logits
        .iter()
        .map(|&value| softplus(value).sqrt())
        .collect::<Vec<_>>();
    let selection = scores
        .iter()
        .enumerate()
        .map(|(index, &score)| score + correction_bias.map_or(0.0, |bias| bias[index]))
        .collect::<Vec<_>>();
    let experts = if let Some(experts) = hash_experts {
        for &expert in experts {
            if expert >= logits.len() {
                return Err(DeepseekMathError::Invalid(
                    "hash expert IDs must be in range".to_owned(),
                ));
            }
        }
        experts.to_vec()
    } else {
        let mut ranking = (0..logits.len()).collect::<Vec<_>>();
        ranking.sort_by(|&left, &right| {
            selection[right]
                .total_cmp(&selection[left])
                .then_with(|| left.cmp(&right))
        });
        ranking.truncate(top_k);
        ranking
    };
    let denominator = experts.iter().map(|&expert| scores[expert]).sum::<f32>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(DeepseekMathError::NonFinite);
    }
    Ok(experts
        .into_iter()
        .map(|expert| RouteChoice {
            expert,
            weight: scores[expert] / denominator * routed_scaling_factor,
            selection_score: selection[expert],
        })
        .collect())
}

pub fn bounded_swiglu(gate: &[f32], up: &[f32], limit: f32) -> Result<Vec<f32>, DeepseekMathError> {
    let mut output = gate.to_vec();
    bounded_swiglu_in_place(&mut output, up, limit)?;
    Ok(output)
}

/// In-place bounded SwiGLU for runtimes that can reuse the gate allocation as their output.
/// Shape, limit, and all values are validated before the first write.
pub fn bounded_swiglu_in_place(
    gate: &mut [f32],
    up: &[f32],
    limit: f32,
) -> Result<(), DeepseekMathError> {
    if gate.len() != up.len() {
        return Err(DeepseekMathError::Shape(
            "gate and up activations have different lengths".to_owned(),
        ));
    }
    validate_positive("SwiGLU limit", limit)?;
    if gate.iter().chain(up).any(|value| !value.is_finite()) {
        return Err(DeepseekMathError::NonFinite);
    }
    for (gate, &up) in gate.iter_mut().zip(up) {
        *gate = {
            let gate = (*gate).min(limit);
            let up = up.clamp(-limit, limit);
            gate / (1.0 + (-gate).exp()) * up
        };
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn paired_rope(
    values: &mut [f32],
    position: usize,
    base: f32,
    original_sequence_length: Option<usize>,
    factor: f32,
    beta_fast: usize,
    beta_slow: usize,
    inverse: bool,
) -> Result<(), DeepseekMathError> {
    if values.is_empty() || values.len() % 2 != 0 {
        return Err(DeepseekMathError::Shape(
            "paired RoPE needs a non-empty even-width vector".to_owned(),
        ));
    }
    validate_positive("RoPE base", base)?;
    validate_positive("RoPE factor", factor)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DeepseekMathError::NonFinite);
    }
    let dim = values.len();
    let correction_range = original_sequence_length.map(|length| {
        let correction = |rotations: usize| {
            dim as f32 * (length as f32 / (rotations as f32 * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = correction(beta_fast).floor().max(0.0) as usize;
        let high = correction(beta_slow).ceil().min((dim - 1) as f32) as usize;
        (low, high)
    });
    for pair in 0..dim / 2 {
        let mut frequency = 1.0 / base.powf((2 * pair) as f32 / dim as f32);
        if let Some((low, high)) = correction_range {
            let denominator = if low == high {
                high as f32 + 0.001 - low as f32
            } else {
                (high - low) as f32
            };
            let ramp = ((pair as f32 - low as f32) / denominator).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            frequency = frequency / factor * (1.0 - smooth) + frequency * smooth;
        }
        let angle = position as f32 * frequency * if inverse { -1.0 } else { 1.0 };
        let (sin, cos) = angle.sin_cos();
        let real = values[2 * pair];
        let imaginary = values[2 * pair + 1];
        values[2 * pair] = real * cos - imaginary * sin;
        values[2 * pair + 1] = real * sin + imaginary * cos;
    }
    // `apply_rotary_emb` computes in FP32 and copies back into the original BF16 tensor.
    round_to_bf16_in_place(values)
}

fn softplus(value: f32) -> f32 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn row_softmax_in_place(values: &mut [f32], width: usize) -> Result<(), DeepseekMathError> {
    for row in values.chunks_exact_mut(width) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - maximum).exp();
            denominator += *value;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(DeepseekMathError::NonFinite);
        }
        for value in row {
            *value /= denominator;
        }
    }
    Ok(())
}

fn normalize_rows(values: &mut [f32], width: usize, epsilon: f32) {
    for row in values.chunks_exact_mut(width) {
        let denominator = row.iter().sum::<f32>() + epsilon;
        for value in row {
            *value /= denominator;
        }
    }
}

fn normalize_columns(values: &mut [f32], width: usize, epsilon: f32) {
    for column in 0..width {
        let denominator = (0..width)
            .map(|row| values[row * width + column])
            .sum::<f32>()
            + epsilon;
        for row in 0..width {
            values[row * width + column] /= denominator;
        }
    }
}

fn finite_vec(values: Vec<f64>) -> Result<Vec<f32>, DeepseekMathError> {
    let mut values = values
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    round_to_bf16_in_place(&mut values)?;
    Ok(values)
}

fn validate_positive(name: &str, value: f32) -> Result<(), DeepseekMathError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(DeepseekMathError::Invalid(format!(
            "{name} must be finite and positive"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DenseMatrix;

    #[test]
    fn bf16_rounding_is_ties_to_even_and_rejects_overflow() {
        let mut values = [f32::from_bits(0x3f80_8000), f32::from_bits(0x3f81_8000)];
        round_to_bf16_in_place(&mut values).unwrap();
        assert_eq!(values, [1.0, 1.015625]);

        let mut overflow = [f32::MAX];
        assert!(matches!(
            round_to_bf16_in_place(&mut overflow),
            Err(DeepseekMathError::NonFinite)
        ));
    }

    #[test]
    fn avx2_bf16_rounding_matches_scalar_values_and_error_position() {
        let mut state = 0x1234_5678u32;
        for length in 0usize..40 {
            let input = (0..length)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let sign = state & 0x8000_0000;
                    let exponent = ((state >> 23) % 254) << 23;
                    f32::from_bits(sign | exponent | (state & 0x007f_ffff))
                })
                .collect::<Vec<_>>();
            let mut expected = input.clone();
            for value in &mut expected {
                *value = round_to_bf16(*value).unwrap();
            }
            let mut actual = input;
            round_to_bf16_in_place(&mut actual).unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        for invalid in [f32::INFINITY, f32::NAN, f32::MAX] {
            for position in 0usize..17 {
                let mut expected = vec![1.234_375f32; 17];
                expected[position] = invalid;
                let mut actual = expected.clone();
                let expected_error = (|| {
                    for value in &mut expected {
                        *value = round_to_bf16(*value)?;
                    }
                    Ok::<_, DeepseekMathError>(())
                })();
                let actual_error = round_to_bf16_in_place(&mut actual);
                assert!(expected_error.is_err() && actual_error.is_err());
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn unit_rms_norm_materializes_official_bf16_intermediates() {
        let mut values = [0.68359375, 0.671875, 2.796875, -2.515625, 0.34375];
        unit_rms_norm_in_place(&mut values, 5, 1e-6).unwrap();
        assert_eq!(
            values,
            [0.392_578_13, 0.38671875, 1.609375, -1.4453125, 0.197_265_63]
        );
    }

    #[test]
    fn hash_router_uses_fixed_ids_but_learned_weights() {
        let choices =
            route_sqrt_softplus(&[0.0, 1.0, 2.0, 3.0], None, Some(&[3, 0]), 2, 1.5).unwrap();
        assert_eq!(
            choices
                .iter()
                .map(|choice| choice.expert)
                .collect::<Vec<_>>(),
            [3, 0]
        );
        assert!((choices.iter().map(|choice| choice.weight).sum::<f32>() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn correction_bias_selects_but_does_not_weight() {
        let plain = route_sqrt_softplus(&[3.0, 2.0, 1.0], None, None, 1, 1.0).unwrap();
        let biased =
            route_sqrt_softplus(&[3.0, 2.0, 1.0], Some(&[0.0, 0.0, 10.0]), None, 1, 1.0).unwrap();
        assert_eq!(plain[0].expert, 0);
        assert_eq!(biased[0].expert, 2);
        assert_eq!(biased[0].weight, 1.0);
    }

    #[test]
    fn bounded_swiglu_clamps_the_two_branches_asymmetrically() {
        let output = bounded_swiglu(&[20.0, -20.0], &[20.0, -20.0], 10.0).unwrap();
        assert!((output[0] - (10.0 / (1.0 + (-10.0f32).exp()) * 10.0)).abs() < 1e-5);
        assert!(output[1] > 0.0 && output[1] < 1e-6);

        let mut in_place = [20.0, -20.0];
        bounded_swiglu_in_place(&mut in_place, &[20.0, -20.0], 10.0).unwrap();
        assert_eq!(
            in_place.map(f32::to_bits),
            output.iter().copied().map(f32::to_bits).collect::<Vec<_>>()[..]
        );

        let mut invalid = [1.0, 2.0];
        let before = invalid;
        assert!(matches!(
            bounded_swiglu_in_place(&mut invalid, &[3.0, f32::NAN], 10.0),
            Err(DeepseekMathError::NonFinite)
        ));
        assert_eq!(invalid, before);
    }

    #[test]
    fn paired_rope_position_zero_is_identity_and_inverse_roundtrips() {
        let original = [1.0, 2.0, 3.0, 4.0];
        let mut zero = original;
        paired_rope(&mut zero, 0, 10_000.0, Some(64), 16.0, 32, 1, false).unwrap();
        assert_eq!(zero, original);
        let mut rotated = original;
        paired_rope(&mut rotated, 7, 10_000.0, Some(64), 16.0, 32, 1, false).unwrap();
        paired_rope(&mut rotated, 7, 10_000.0, Some(64), 16.0, 32, 1, true).unwrap();
        for (&left, &right) in rotated.iter().zip(&original) {
            assert!((left - right).abs() < 0.02);
        }
    }

    #[test]
    fn zero_function_hc_has_explicit_stable_mixing() {
        let function = WeightMatrix::F32(DenseMatrix::zeros(8, 4).unwrap());
        let (collapsed, mix) = hyper_connection_pre(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &function,
            &[1.0; 3],
            &[0.0; 8],
            4,
            1e-6,
            1e-6,
        )
        .unwrap();
        assert!((collapsed[0] - 2.000004).abs() < 1e-4);
        assert!((collapsed[1] - 3.000006).abs() < 1e-4);
        for column in 0..2 {
            let sum = mix.combination[column] + mix.combination[2 + column];
            assert!((sum - 1.0).abs() < 1e-5);
        }
        let output = hyper_connection_post(&[10.0, 20.0], &[1.0, 2.0, 3.0, 4.0], 2, &mix).unwrap();
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn attention_sink_dilutes_without_adding_a_value() {
        let output =
            sparse_attention_with_sink(&[1.0, 0.0], 1, 2, &[vec![2.0, 4.0]], &[0], &[2.0], 1.0)
                .unwrap();
        assert_eq!(output, vec![1.0, 2.0]);
    }
}
