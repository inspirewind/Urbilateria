//! Single-Pass mHC scheduling for DeepSeek-V4.1.

use crate::model::WeightMatrix;
use crate::models::deepseek_v4::math::{
    round_to_bf16_in_place, DeepseekMathError, HyperConnectionMix,
};

/// Predicts the current sublayer's post/residual mix while collapsing its input with the `pre`
/// coefficients produced by the preceding sublayer. This one-step shift is the architectural
/// difference between V4.1 Single-Pass mHC and the original V4 schedule.
#[allow(clippy::too_many_arguments)]
pub fn single_pass_pre(
    hidden: &[f32],
    incoming_pre: &[f32],
    hidden_size: usize,
    hc_mult: usize,
    function: &WeightMatrix,
    scale: &[f32],
    base: &[f32],
    sinkhorn_iters: usize,
    norm_eps: f32,
    sinkhorn_eps: f32,
) -> Result<(Vec<f32>, HyperConnectionMix), DeepseekMathError> {
    if incoming_pre.len() != hc_mult
        || incoming_pre.iter().any(|value| !value.is_finite())
        || hidden.len() != hidden_size.saturating_mul(hc_mult)
    {
        return Err(DeepseekMathError::Shape(
            "incoming pre-mix or hidden state does not match Single-Pass mHC geometry".to_owned(),
        ));
    }
    let mix_count = (2 + hc_mult) * hc_mult;
    if function.rows() != mix_count
        || function.cols() != hidden.len()
        || scale.len() != 3
        || base.len() != mix_count
        || sinkhorn_iters == 0
        || !(norm_eps.is_finite() && norm_eps > 0.0)
        || !(sinkhorn_eps.is_finite() && sinkhorn_eps > 0.0)
    {
        return Err(DeepseekMathError::Shape(
            "Single-Pass mHC predictor geometry is invalid".to_owned(),
        ));
    }
    // The release Mega-mHC kernel uses FP32 for the normalized projection and every Sinkhorn
    // reduction. This cannot reuse V4's intentionally higher-precision scalar reference without
    // changing MoE top-k decisions after only one layer.
    let mean_square =
        hidden.iter().fold(0.0f32, |sum, value| sum + value * value) / hidden.len() as f32;
    let reciprocal_rms = (mean_square + norm_eps).sqrt().recip();
    let mut mixes = function.matvec_fp32_accum(hidden)?;
    for value in &mut mixes {
        *value *= reciprocal_rms;
    }
    let mut pre = Vec::with_capacity(hc_mult);
    let mut post = Vec::with_capacity(hc_mult);
    for index in 0..hc_mult {
        pre.push(sigmoid(mixes[index] * scale[0] + base[index]) + sinkhorn_eps);
        post.push(2.0 * sigmoid(mixes[hc_mult + index] * scale[1] + base[hc_mult + index]));
    }
    let offset = 2 * hc_mult;
    let mut combination = mixes[offset..]
        .iter()
        .zip(&base[offset..])
        .map(|(&value, &base)| value * scale[2] + base)
        .collect::<Vec<_>>();
    for row in combination.chunks_mut(hc_mult) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        for value in row {
            *value = *value / sum + sinkhorn_eps;
        }
    }
    normalize_columns(&mut combination, hc_mult, sinkhorn_eps);
    for _ in 1..sinkhorn_iters {
        normalize_rows(&mut combination, hc_mult, sinkhorn_eps);
        normalize_columns(&mut combination, hc_mult, sinkhorn_eps);
    }
    let predicted = HyperConnectionMix {
        pre,
        post,
        combination,
    };
    let mut collapsed = vec![0.0f32; hidden_size];
    for copy in 0..hc_mult {
        for feature in 0..hidden_size {
            collapsed[feature] += incoming_pre[copy] * hidden[copy * hidden_size + feature];
        }
    }
    round_to_bf16_in_place(&mut collapsed)?;
    Ok((collapsed, predicted))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn normalize_rows(values: &mut [f32], width: usize, eps: f32) {
    for row in values.chunks_mut(width) {
        let sum = row.iter().sum::<f32>();
        for value in row {
            *value /= sum + eps;
        }
    }
}

fn normalize_columns(values: &mut [f32], width: usize, eps: f32) {
    for column in 0..width {
        let sum = (0..width)
            .map(|row| values[row * width + column])
            .sum::<f32>();
        for row in 0..width {
            values[row * width + column] /= sum + eps;
        }
    }
}

pub fn identity_pre_mix(hc_mult: usize) -> Result<Vec<f32>, DeepseekMathError> {
    if hc_mult == 0 {
        return Err(DeepseekMathError::Shape(
            "Single-Pass mHC needs at least one residual stream".to_owned(),
        ));
    }
    let mut mix = vec![0.0; hc_mult];
    mix[0] = 1.0;
    Ok(mix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DenseMatrix;

    #[test]
    fn first_attention_consumes_identity_instead_of_its_predicted_pre_mix() {
        let hidden = [1.0, 2.0, 9.0, 10.0];
        let hc_mult = 2;
        let hidden_size = 2;
        let mix_count = (2 + hc_mult) * hc_mult;
        let function = WeightMatrix::from(
            DenseMatrix::new(
                mix_count,
                hidden.len(),
                vec![0.125; mix_count * hidden.len()],
            )
            .unwrap(),
        );
        let (collapsed, predicted) = single_pass_pre(
            &hidden,
            &identity_pre_mix(hc_mult).unwrap(),
            hidden_size,
            hc_mult,
            &function,
            &[1.0; 3],
            &[0.0; 8],
            3,
            1e-6,
            1e-6,
        )
        .unwrap();
        assert_eq!(collapsed, vec![1.0, 2.0]);
        assert_ne!(predicted.pre, vec![1.0, 0.0]);
    }

    #[test]
    fn next_sublayer_consumes_the_pre_mix_predicted_by_the_previous_one() {
        let hidden = [1.0, 2.0, 9.0, 10.0];
        let hc_mult = 2;
        let hidden_size = 2;
        let mix_count = (2 + hc_mult) * hc_mult;
        let function = WeightMatrix::from(
            DenseMatrix::new(mix_count, hidden.len(), vec![0.0; mix_count * hidden.len()]).unwrap(),
        );
        let incoming = [0.25, 0.75];
        let (collapsed, _) = single_pass_pre(
            &hidden,
            &incoming,
            hidden_size,
            hc_mult,
            &function,
            &[1.0; 3],
            &[0.0; 8],
            3,
            1e-6,
            1e-6,
        )
        .unwrap();
        assert_eq!(collapsed, vec![7.0, 8.0]);
    }
}
