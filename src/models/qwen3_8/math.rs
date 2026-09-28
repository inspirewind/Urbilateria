//! Qwen-specific execution precision around mixed BF16/block-FP8 projections.

use crate::math::simulate_finegrained_e4m3_activation;
use crate::model::{MatrixError, WeightError, WeightMatrix};

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

/// Projects `[batch, cols]` input rows, preserving `linear` precision for each token.
/// BF16 weights share decoded lanes; other formats retain their tokenwise path, including
/// independent dynamic activation quantization for block-FP8 routed experts.
pub fn linear_batch(
    weight: &WeightMatrix,
    input: &[f32],
    batch: usize,
) -> Result<Vec<f32>, WeightError> {
    let expected = batch
        .checked_mul(weight.cols())
        .ok_or_else(|| MatrixError::InvalidShape("batch * cols overflows usize".into()))?;
    if batch == 0 || input.len() != expected {
        return Err(MatrixError::InputLength {
            expected,
            got: input.len(),
        }
        .into());
    }
    if let WeightMatrix::Bf16(matrix) = weight {
        let mut output = matrix.matmul_rows_fp32(input, batch)?;
        round_to_bf16_in_place(&mut output)?;
        return Ok(output);
    }
    let mut output = Vec::new();
    for token in input.chunks_exact(weight.cols()) {
        output.extend(linear(weight, token)?);
    }
    Ok(output)
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
    #[cfg(target_arch = "x86_64")]
    let rounded = if values.len() >= 8 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and the helper only accesses complete eight-value chunks.
        unsafe { round_bf16_prefix_avx2(values) }
    } else {
        0
    };
    #[cfg(not(target_arch = "x86_64"))]
    let rounded = 0;
    // The vector path stops before an invalid chunk. Scalar replay preserves the original
    // first error and leaves exactly the same unprocessed suffix, including the invalid value.
    for value in &mut values[rounded..] {
        *value = round_to_bf16(*value)?;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn round_bf16_prefix_avx2(values: &mut [f32]) -> usize {
    use std::arch::x86_64::*;
    let exponent = _mm256_set1_epi32(0x7f80_0000);
    let high_bits = _mm256_set1_epi32(0xffff_0000u32 as i32);
    let bias = _mm256_set1_epi32(0x7fff);
    let one = _mm256_set1_epi32(1);
    let complete = values.len() / 8 * 8;
    let mut offset = 0;
    while offset < complete {
        // SAFETY: offset covers eight in-bounds F32 bit patterns; accesses may be unaligned.
        unsafe {
            let pointer = values.as_mut_ptr().add(offset).cast::<__m256i>();
            let bits = _mm256_loadu_si256(pointer);
            let parity = _mm256_and_si256(_mm256_srli_epi32(bits, 16), one);
            let rounded = _mm256_and_si256(
                _mm256_add_epi32(bits, _mm256_add_epi32(bias, parity)),
                high_bits,
            );
            let invalid_input = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent);
            let invalid_output = _mm256_cmpeq_epi32(_mm256_and_si256(rounded, exponent), exponent);
            if _mm256_movemask_epi8(_mm256_or_si256(invalid_input, invalid_output)) != 0 {
                break;
            }
            _mm256_storeu_si256(pointer, rounded);
        }
        offset += 8;
    }
    offset
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

    #[test]
    fn bulk_bf16_rounding_matches_scalar_at_every_bf16_boundary() {
        let mut values = Vec::new();
        let mut expected = Vec::new();
        for high in 0..=u16::MAX {
            for low in [0, 1, 0x7fff, 0x8000, 0x8001, 0xffff] {
                let value = f32::from_bits((u32::from(high) << 16) | low);
                if let Ok(rounded) = round_to_bf16(value) {
                    values.push(value);
                    expected.push(rounded.to_bits());
                }
            }
        }
        round_to_bf16_in_place(&mut values).unwrap();
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn bulk_bf16_rounding_preserves_unaligned_tails_and_guards() {
        for offset in 0..8 {
            for length in 0..=33 {
                let mut values = (0..offset + length + 8)
                    .map(|i| f32::from_bits(0xbf00_0000 + i as u32 * 0x51e9))
                    .collect::<Vec<_>>();
                let mut expected = values.clone();
                for value in &mut expected[offset..offset + length] {
                    *value = round_to_bf16(*value).unwrap();
                }
                round_to_bf16_in_place(&mut values[offset..offset + length]).unwrap();
                assert_eq!(
                    values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "offset={offset} length={length}"
                );
            }
        }
    }

    #[test]
    fn bulk_bf16_rounding_preserves_first_error_and_partial_mutation() {
        for invalid in [
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7fc0_0000,
            0xffc0_0000,
            0x7fff_ffff,
            0xffff_ffff,
            0x7f7f_8000,
            0xff7f_8000,
            0x7f7f_ffff,
            0xff7f_ffff,
        ] {
            for position in 0..34 {
                let mut values = vec![1.003; 36];
                values[position + 1] = f32::from_bits(invalid);
                let mut expected = values.clone();
                let scalar: Result<(), WeightError> =
                    expected[1..35].iter_mut().try_for_each(|value| {
                        *value = round_to_bf16(*value)?;
                        Ok(())
                    });
                assert!(scalar.is_err());
                assert!(round_to_bf16_in_place(&mut values[1..35]).is_err());
                assert_eq!(
                    values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "invalid={invalid:#x} position={position}"
                );
            }
        }
    }
}
