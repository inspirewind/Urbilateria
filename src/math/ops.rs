use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum MathError {
    EmptyInput,
    Shape {
        operation: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidValue {
        operation: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("operation requires a non-empty input"),
            Self::Shape {
                operation,
                expected,
                got,
            } => {
                write!(f, "{operation}: expected length {expected}, got {got}")
            }
            Self::InvalidValue { operation, reason } => write!(f, "{operation}: {reason}"),
        }
    }
}

impl std::error::Error for MathError {}

#[inline]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// GLM's RMSNorm, accumulated in f64 to make the scalar path a stable reference.
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, MathError> {
    if input.is_empty() {
        return Err(MathError::EmptyInput);
    }
    if input.len() != weight.len() {
        return Err(MathError::Shape {
            operation: "rms_norm",
            expected: input.len(),
            got: weight.len(),
        });
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(MathError::InvalidValue {
            operation: "rms_norm",
            reason: "epsilon must be finite and positive",
        });
    }
    if input.iter().chain(weight).any(|value| !value.is_finite()) {
        return Err(MathError::InvalidValue {
            operation: "rms_norm",
            reason: "input contains NaN or infinity",
        });
    }
    let mean_square = input
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / input.len() as f64;
    let inverse_rms = (mean_square + f64::from(eps)).sqrt().recip() as f32;
    Ok(input
        .iter()
        .zip(weight)
        .map(|(&value, &scale)| value * inverse_rms * scale)
        .collect())
}

pub fn softmax_in_place(values: &mut [f32]) -> Result<(), MathError> {
    if values.is_empty() {
        return Err(MathError::EmptyInput);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MathError::InvalidValue {
            operation: "softmax",
            reason: "input contains NaN or infinity",
        });
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += f64::from(*value);
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err(MathError::InvalidValue {
            operation: "softmax",
            reason: "normalization sum is zero or non-finite",
        });
    }
    let inverse = sum.recip() as f32;
    for value in values {
        *value *= inverse;
    }
    Ok(())
}

/// Applies GLM-5.2's interleaved partial RoPE to one rotary slice in place.
///
/// Input pairs are `(x0,x1), (x2,x3), ...`; output cosine components occupy the first half
/// and sine components the second half, matching the official `apply_rotary_pos_emb_interleave`.
pub fn interleaved_rope(values: &mut [f32], position: usize, theta: f32) -> Result<(), MathError> {
    if values.is_empty() || values.len() % 2 != 0 {
        return Err(MathError::InvalidValue {
            operation: "interleaved_rope",
            reason: "rotary dimension must be non-zero and even",
        });
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(MathError::InvalidValue {
            operation: "interleaved_rope",
            reason: "theta must be finite and positive",
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MathError::InvalidValue {
            operation: "interleaved_rope",
            reason: "input contains NaN or infinity",
        });
    }
    let input = values.to_vec();
    let half = values.len() / 2;
    for pair in 0..half {
        let inverse_frequency = theta.powf(-2.0 * pair as f32 / values.len() as f32);
        let angle = position as f32 * inverse_frequency;
        let (sine, cosine) = angle.sin_cos();
        let even = input[2 * pair];
        let odd = input[2 * pair + 1];
        values[pair] = even * cosine - odd * sine;
        values[half + pair] = odd * cosine + even * sine;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let mut values = [10_000.0, 10_001.0, 9_999.0];
        softmax_in_place(&mut values).unwrap();
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(values[1] > values[0] && values[0] > values[2]);
    }

    #[test]
    fn rope_position_zero_only_deinterleaves() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        interleaved_rope(&mut values, 0, 8_000_000.0).unwrap();
        assert_eq!(values, [1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn rms_norm_matches_hand_calculation() {
        let output = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 1e-6).unwrap();
        let inverse = (12.5f32 + 1e-6).sqrt().recip();
        assert!((output[0] - 3.0 * inverse).abs() < 1e-6);
        assert!((output[1] - 8.0 * inverse).abs() < 1e-6);
    }
}
