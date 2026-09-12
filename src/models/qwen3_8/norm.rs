use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormError {
    EmptyInput,
    Shape {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidEpsilon,
    NonFinite(&'static str),
}

impl fmt::Display for NormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("Qwen RMSNorm requires a non-empty input"),
            Self::Shape {
                name,
                expected,
                got,
            } => write!(f, "Qwen RMSNorm {name} needs {expected} values, got {got}"),
            Self::InvalidEpsilon => f.write_str("Qwen RMSNorm epsilon must be finite and positive"),
            Self::NonFinite(name) => {
                write!(f, "Qwen RMSNorm {name} contains NaN or infinity")
            }
        }
    }
}

impl std::error::Error for NormError {}

/// Qwen3.8's zero-centered RMSNorm.
///
/// Unlike Llama-style RMSNorm, the checkpoint stores an offset from one.  The effective
/// scale is therefore `1 + weight`, not `weight`.
pub fn zero_centered_rms_norm(
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, NormError> {
    validate(input, weight, None, eps)?;
    let inverse_rms = inverse_rms(input, eps)?;
    let output = input
        .iter()
        .zip(weight)
        .map(|(&value, &offset)| value * inverse_rms * (1.0 + offset))
        .collect::<Vec<_>>();
    ensure_finite("output", &output)?;
    Ok(output)
}

/// DeltaNet's per-value-head RMSNorm followed by its SiLU output gate.
///
/// This normalization deliberately uses the checkpoint weight directly.  It is a different
/// parameterization from [`zero_centered_rms_norm`].
pub fn gated_rms_norm(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    eps: f32,
) -> Result<Vec<f32>, NormError> {
    validate(input, weight, Some(gate), eps)?;
    let inverse_rms = inverse_rms(input, eps)?;
    let output = input
        .iter()
        .zip(weight)
        .zip(gate)
        .map(|((&value, &scale), &gate)| {
            value * inverse_rms * scale * (gate / (1.0 + (-gate).exp()))
        })
        .collect::<Vec<_>>();
    ensure_finite("output", &output)?;
    Ok(output)
}

/// BF16-materializing form of DeltaNet's gated RMSNorm used by the release runtime.
///
/// Transformers casts the normalized value to the input dtype before multiplying the direct
/// norm weight, then multiplies that BF16 result by a FP32 SiLU gate and casts once more. These
/// two materialization points are observable with real checkpoint values.
pub(crate) fn gated_rms_norm_bf16(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    eps: f32,
) -> Result<Vec<f32>, NormError> {
    validate(input, weight, Some(gate), eps)?;
    let inverse_rms = inverse_rms(input, eps)?;
    let output = input
        .iter()
        .zip(weight)
        .zip(gate)
        .map(|((&value, &scale), &gate)| {
            let normalized = round_bf16(value * inverse_rms);
            let weighted = round_bf16(normalized * scale);
            round_bf16(weighted * (gate / (1.0 + (-gate).exp())))
        })
        .collect::<Vec<_>>();
    ensure_finite("output", &output)?;
    Ok(output)
}

fn validate(
    input: &[f32],
    weight: &[f32],
    gate: Option<&[f32]>,
    eps: f32,
) -> Result<(), NormError> {
    if input.is_empty() {
        return Err(NormError::EmptyInput);
    }
    if weight.len() != input.len() {
        return Err(NormError::Shape {
            name: "weight",
            expected: input.len(),
            got: weight.len(),
        });
    }
    if let Some(gate) = gate {
        if gate.len() != input.len() {
            return Err(NormError::Shape {
                name: "gate",
                expected: input.len(),
                got: gate.len(),
            });
        }
        ensure_finite("gate", gate)?;
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(NormError::InvalidEpsilon);
    }
    ensure_finite("input", input)?;
    ensure_finite("weight", weight)?;
    Ok(())
}

fn inverse_rms(input: &[f32], eps: f32) -> Result<f32, NormError> {
    // The upstream implementation explicitly evaluates this part in FP32.  Keep the scalar
    // reference in the same precision so tiny PyTorch equation oracles compare directly.
    let mean_square =
        input.iter().fold(0.0f32, |sum, &value| sum + value * value) / input.len() as f32;
    let inverse = (mean_square + eps).sqrt().recip();
    if !inverse.is_finite() {
        return Err(NormError::NonFinite("normalization factor"));
    }
    Ok(inverse)
}

fn ensure_finite(name: &'static str, values: &[f32]) -> Result<(), NormError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NormError::NonFinite(name));
    }
    Ok(())
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_centered_weight_is_an_offset_from_one() {
        let output = zero_centered_rms_norm(&[3.0, 4.0], &[0.0, 1.0], 1e-6).unwrap();
        let inverse = (12.5f32 + 1e-6).sqrt().recip();
        assert!((output[0] - 3.0 * inverse).abs() < 1e-6);
        assert!((output[1] - 8.0 * inverse).abs() < 1e-6);
    }

    #[test]
    fn gated_norm_uses_direct_weight_and_silu_gate() {
        let output = gated_rms_norm(&[3.0, 4.0], &[2.0, 0.5], &[0.0, 1.0], 1e-6).unwrap();
        let inverse = (12.5f32 + 1e-6).sqrt().recip();
        let silu_one = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert_eq!(output[0], 0.0);
        assert!((output[1] - 4.0 * inverse * 0.5 * silu_one).abs() < 1e-6);
    }

    #[test]
    fn gated_bf16_norm_materializes_before_weight_and_after_gate() {
        let input = [1.234_375, -0.542_968_75];
        let weight = [1.101_562_5, 0.902_343_75];
        let gate = [0.333_984_38, -0.777_343_75];
        let output = gated_rms_norm_bf16(&input, &weight, &gate, 1e-6).unwrap();
        let inverse = (input.iter().map(|value| value * value).sum::<f32>() / 2.0 + 1e-6)
            .sqrt()
            .recip();
        let expected = input
            .iter()
            .zip(weight)
            .zip(gate)
            .map(|((&value, scale), gate)| {
                let normalized = round_bf16(value * inverse);
                let weighted = round_bf16(normalized * scale);
                round_bf16(weighted * (gate / (1.0 + (-gate).exp())))
            })
            .collect::<Vec<_>>();
        assert_eq!(output, expected);
    }

    #[test]
    fn non_finite_gate_is_rejected() {
        let error = gated_rms_norm(&[1.0], &[1.0], &[f32::NAN], 1e-6).unwrap_err();
        assert_eq!(error, NormError::NonFinite("gate"));
    }
}
