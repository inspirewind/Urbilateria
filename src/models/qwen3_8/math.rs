//! Qwen-specific execution precision around mixed BF16/block-FP8 projections.

use crate::math::simulate_finegrained_e4m3_activation;
use crate::model::{WeightError, WeightMatrix};

const FP8_ACTIVATION_GROUP: usize = 128;

/// Executes one Qwen projection, including dynamic per-token-group activation quantization for
/// block-FP8 routed-expert weights. Dense tiny-oracle matrices retain their ordinary F32 path.
pub fn linear(weight: &WeightMatrix, input: &[f32]) -> Result<Vec<f32>, WeightError> {
    if weight.uses_mx_activation_quantization() {
        let quantized = simulate_finegrained_e4m3_activation(input, FP8_ACTIVATION_GROUP)?;
        let mut output = weight.matvec(&quantized)?;
        round_to_bf16_in_place(&mut output)?;
        Ok(output)
    } else {
        // Upstream BF16 GEMMs accumulate in FP32. The generic readable matrix contract uses an
        // ordered F64 dot product, so select the existing FP32/FMA kernel explicitly for Qwen's
        // trunks before materializing the BF16 output.
        let mut output = weight.matvec_bf16_fp32(input)?;
        if uses_bf16_output(weight) {
            round_to_bf16_in_place(&mut output)?;
        }
        Ok(output)
    }
}

/// Whether this Qwen projection materializes its result in the model's BF16 activation dtype.
pub fn uses_bf16_output(weight: &WeightMatrix) -> bool {
    matches!(
        weight,
        WeightMatrix::Bf16(_) | WeightMatrix::MxFp8(_) | WeightMatrix::MxFp4(_)
    )
}

pub fn round_to_bf16(value: f32) -> Result<f32, WeightError> {
    if !value.is_finite() {
        return Err(crate::math::MxError::NonFinite.into());
    }
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    let rounded = f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000);
    if !rounded.is_finite() {
        return Err(crate::math::MxError::NonFinite.into());
    }
    Ok(rounded)
}

pub fn round_to_bf16_in_place(values: &mut [f32]) -> Result<(), WeightError> {
    for value in values {
        *value = round_to_bf16(*value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Bf16Matrix, DenseMatrix};

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn bf16_trunk_linear_materializes_bf16_but_dense_oracle_stays_f32() {
        let input = [1.001, 1.001];
        let bf16 =
            WeightMatrix::Bf16(Bf16Matrix::from_le_bytes(1, 2, bf16_bytes(&[1.0, 1.0])).unwrap());
        let dense = WeightMatrix::F32(DenseMatrix::new(1, 2, vec![1.0, 1.0]).unwrap());
        assert_eq!(linear(&bf16, &input).unwrap(), [2.0]);
        assert!((linear(&dense, &input).unwrap()[0] - 2.002).abs() < 1e-6);
    }

    #[test]
    fn bf16_rounding_is_ties_to_even() {
        let midpoint_even = 1.0 + 1.0 / 256.0;
        let midpoint_odd = 1.0 + 3.0 / 256.0;
        assert_eq!(round_to_bf16(midpoint_even).unwrap(), 1.0);
        assert_eq!(round_to_bf16(midpoint_odd).unwrap(), 1.015_625);
    }
}
