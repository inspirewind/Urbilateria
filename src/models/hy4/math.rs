//! Hy4-specific identity Hyper-Connection reference math.

use crate::model::{WeightError, WeightMatrix};
use crate::models::deepseek_v4::math::round_to_bf16_in_place;
use std::fmt;

#[derive(Debug)]
pub enum Hy4MathError {
    Shape(String),
    Invalid(String),
    NonFinite,
    Weight(WeightError),
}

impl fmt::Display for Hy4MathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(formatter, "Hy4 shape error: {reason}"),
            Self::Invalid(reason) => write!(formatter, "invalid Hy4 math: {reason}"),
            Self::NonFinite => formatter.write_str("Hy4 math produced NaN or infinity"),
            Self::Weight(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Hy4MathError {}

impl From<WeightError> for Hy4MathError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdentityHyperConnectionMix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
}

/// Computes the branch input and branch-output gates for identity Hyper-Connections.
///
/// Unlike DeepSeek-V4 HC, iHC has no learned stream-to-stream Sinkhorn matrix. The residual
/// transport is the identity; only the branch read (`pre`) and branch write (`post`) are gated.
#[allow(clippy::too_many_arguments)]
pub fn identity_hyper_connection_pre(
    hidden: &[f32],
    hidden_size: usize,
    hc_mult: usize,
    function: &WeightMatrix,
    scale: &[f32],
    base: &[f32],
    norm_eps: f32,
    hc_eps: f32,
    magnitude: f32,
) -> Result<(Vec<f32>, IdentityHyperConnectionMix), Hy4MathError> {
    let width = hidden_size
        .checked_mul(hc_mult)
        .ok_or_else(|| Hy4MathError::Shape("iHC width overflows".to_owned()))?;
    let mix_count = hc_mult
        .checked_mul(2)
        .ok_or_else(|| Hy4MathError::Shape("iHC gate count overflows".to_owned()))?;
    if hidden_size == 0
        || hc_mult == 0
        || hidden.len() != width
        || function.rows() != mix_count
        || function.cols() != width
        || scale.len() != 2
        || base.len() != mix_count
    {
        return Err(Hy4MathError::Shape(
            "hidden/function/base/scale do not match identity-HC geometry".to_owned(),
        ));
    }
    if !norm_eps.is_finite()
        || norm_eps <= 0.0
        || !hc_eps.is_finite()
        || hc_eps <= 0.0
        || !magnitude.is_finite()
        || magnitude <= 0.0
        || hidden
            .iter()
            .chain(scale)
            .chain(base)
            .any(|value| !value.is_finite())
    {
        return Err(Hy4MathError::Invalid(
            "iHC epsilons/magnitude must be positive and inputs finite".to_owned(),
        ));
    }
    let mean_square = hidden
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / width as f64;
    let reciprocal_rms = (mean_square + f64::from(norm_eps)).sqrt().recip() as f32;
    let mixes = function.matvec(hidden)?;
    let mut pre = Vec::with_capacity(hc_mult);
    let mut post = Vec::with_capacity(hc_mult);
    for stream in 0..hc_mult {
        pre.push(sigmoid(mixes[stream] * reciprocal_rms * scale[0] + base[stream]) + hc_eps);
        post.push(
            magnitude
                * sigmoid(
                    mixes[hc_mult + stream] * reciprocal_rms * scale[1] + base[hc_mult + stream],
                )
                + hc_eps,
        );
    }
    let mut collapsed = vec![0.0f64; hidden_size];
    for stream in 0..hc_mult {
        for feature in 0..hidden_size {
            collapsed[feature] +=
                f64::from(pre[stream]) * f64::from(hidden[stream * hidden_size + feature]);
        }
    }
    let mut collapsed = collapsed
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    round_to_bf16_in_place(&mut collapsed)
        .map_err(|error| Hy4MathError::Invalid(error.to_string()))?;
    Ok((collapsed, IdentityHyperConnectionMix { pre, post }))
}

pub fn identity_hyper_connection_post(
    branch: &[f32],
    residual: &[f32],
    hidden_size: usize,
    mix: &IdentityHyperConnectionMix,
) -> Result<Vec<f32>, Hy4MathError> {
    let hc_mult = mix.pre.len();
    if hidden_size == 0
        || branch.len() != hidden_size
        || mix.post.len() != hc_mult
        || residual.len() != hc_mult.saturating_mul(hidden_size)
    {
        return Err(Hy4MathError::Shape(
            "branch/residual/gates do not match identity-HC geometry".to_owned(),
        ));
    }
    let mut output = vec![0.0; residual.len()];
    for stream in 0..hc_mult {
        for feature in 0..hidden_size {
            let value =
                residual[stream * hidden_size + feature] + mix.post[stream] * branch[feature];
            if !value.is_finite() {
                return Err(Hy4MathError::NonFinite);
            }
            output[stream * hidden_size + feature] = value;
        }
    }
    round_to_bf16_in_place(&mut output)
        .map_err(|error| Hy4MathError::Invalid(error.to_string()))?;
    Ok(output)
}

/// Collapses the final iHC streams before the model's output RMSNorm.
#[allow(clippy::too_many_arguments)]
pub fn identity_hyper_connection_head(
    hidden: &[f32],
    hidden_size: usize,
    hc_mult: usize,
    function: &WeightMatrix,
    scale: f32,
    base: &[f32],
    norm_eps: f32,
    hc_eps: f32,
) -> Result<Vec<f32>, Hy4MathError> {
    let width = hidden_size
        .checked_mul(hc_mult)
        .ok_or_else(|| Hy4MathError::Shape("iHC head width overflows".to_owned()))?;
    if hidden_size == 0
        || hc_mult == 0
        || hidden.len() != width
        || function.rows() != hc_mult
        || function.cols() != width
        || base.len() != hc_mult
    {
        return Err(Hy4MathError::Shape(
            "hidden/function/base do not match identity-HC head geometry".to_owned(),
        ));
    }
    if !scale.is_finite()
        || !norm_eps.is_finite()
        || norm_eps <= 0.0
        || !hc_eps.is_finite()
        || hc_eps <= 0.0
        || hidden.iter().chain(base).any(|value| !value.is_finite())
    {
        return Err(Hy4MathError::Invalid(
            "iHC head inputs and scale must be finite and epsilons positive".to_owned(),
        ));
    }
    let mean_square = hidden
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / width as f64;
    let reciprocal_rms = (mean_square + f64::from(norm_eps)).sqrt().recip() as f32;
    let mixes = function.matvec(hidden)?;
    let gates = mixes
        .iter()
        .zip(base)
        .map(|(&mix, &bias)| sigmoid(mix * reciprocal_rms * scale + bias) + hc_eps)
        .collect::<Vec<_>>();
    let mut collapsed = vec![0.0f64; hidden_size];
    for stream in 0..hc_mult {
        for feature in 0..hidden_size {
            collapsed[feature] +=
                f64::from(gates[stream]) * f64::from(hidden[stream * hidden_size + feature]);
        }
    }
    let mut collapsed = collapsed
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    round_to_bf16_in_place(&mut collapsed)
        .map_err(|error| Hy4MathError::Invalid(error.to_string()))?;
    Ok(collapsed)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DenseMatrix;

    #[test]
    fn zero_function_keeps_identity_streams_and_gates_the_branch() {
        let function = WeightMatrix::F32(DenseMatrix::new(4, 4, vec![0.0; 16]).unwrap());
        let (collapsed, mix) = identity_hyper_connection_pre(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &function,
            &[1.0, 1.0],
            &[0.0; 4],
            1e-5,
            1e-6,
            2.0,
        )
        .unwrap();
        assert!((collapsed[0] - 2.0).abs() < 0.02);
        assert!((collapsed[1] - 3.0).abs() < 0.02);
        assert!(mix
            .post
            .iter()
            .all(|value| (*value - 1.000001).abs() < 1e-7));
        assert_eq!(
            identity_hyper_connection_post(&[10.0, 20.0], &[1.0, 2.0, 3.0, 4.0], 2, &mix).unwrap(),
            [11.0, 22.0, 13.0, 24.0]
        );
    }

    #[test]
    fn zero_head_function_averages_two_streams() {
        let function = WeightMatrix::F32(DenseMatrix::new(2, 4, vec![0.0; 8]).unwrap());
        let output = identity_hyper_connection_head(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &function,
            1.0,
            &[0.0; 2],
            1e-5,
            1e-6,
        )
        .unwrap();
        assert!((output[0] - 2.0).abs() < 0.02);
        assert!((output[1] - 3.0).abs() < 0.02);
    }
}
