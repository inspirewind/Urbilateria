use super::MatrixError;
use crate::execution::{install, should_parallelize};
use crate::profiling::{span_with_work, ProfileStage};
use rayon::prelude::*;

/// Row-major `[output, input]` BF16 matrix kept in its on-disk representation.
///
/// Values are decoded to F32 only while a row or dot product is being evaluated. Dot products
/// retain the scalar reference contract and reduce products in F64.
#[derive(Debug, Clone)]
pub struct Bf16Matrix {
    rows: usize,
    cols: usize,
    little_endian_bytes: Vec<u8>,
}

impl Bf16Matrix {
    pub fn from_le_bytes(
        rows: usize,
        cols: usize,
        little_endian_bytes: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        Self::from_le_bytes_impl(rows, cols, little_endian_bytes, true)
    }

    /// Constructs a streamed matvec chunk while deferring payload finiteness validation.
    ///
    /// The streamed caller validates a finite input and every output row. Any BF16 NaN or infinity
    /// necessarily makes its row's ordered dot product non-finite, even when multiplied by zero,
    /// so this removes a redundant full-payload scan without weakening rejection.
    pub(crate) fn from_le_bytes_deferred_finite_check(
        rows: usize,
        cols: usize,
        little_endian_bytes: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        Self::from_le_bytes_impl(rows, cols, little_endian_bytes, false)
    }

    fn from_le_bytes_impl(
        rows: usize,
        cols: usize,
        little_endian_bytes: Vec<u8>,
        validate_finite: bool,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidShape(
                "rows and columns must be non-zero".to_owned(),
            ));
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| MatrixError::InvalidShape("rows * cols overflows usize".to_owned()))?;
        let expected_bytes = elements.checked_mul(2).ok_or_else(|| {
            MatrixError::InvalidShape("BF16 matrix byte length overflows usize".to_owned())
        })?;
        if little_endian_bytes.len() != expected_bytes {
            return Err(MatrixError::InvalidShape(format!(
                "[{rows}, {cols}] BF16 needs {expected_bytes} bytes, got {}",
                little_endian_bytes.len()
            )));
        }
        if validate_finite
            && little_endian_bytes
                .chunks_exact(2)
                .any(|bytes| !decode_bf16(bytes).is_finite())
        {
            return Err(MatrixError::NonFinite);
        }
        Ok(Self {
            rows,
            cols,
            little_endian_bytes,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn resident_bytes(&self) -> usize {
        self.little_endian_bytes.len()
    }

    pub(crate) fn into_le_bytes(self) -> Vec<u8> {
        self.little_endian_bytes
    }

    pub fn write_row(&self, row: usize, output: &mut [f32]) -> Result<(), MatrixError> {
        if output.len() != self.cols {
            return Err(MatrixError::InputLength {
                expected: self.cols,
                got: output.len(),
            });
        }
        if row >= self.rows {
            return Err(MatrixError::RowOutOfBounds {
                row,
                rows: self.rows,
            });
        }
        let row_bytes = self.cols * 2;
        let bytes = &self.little_endian_bytes[row * row_bytes..(row + 1) * row_bytes];
        for (output, value) in output.iter_mut().zip(bytes.chunks_exact(2)) {
            *output = decode_bf16(value);
        }
        Ok(())
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>, MatrixError> {
        let mut output = vec![0.0; self.cols];
        self.write_row(row, &mut output)?;
        Ok(output)
    }

    /// `output = W · input` for `W[rows, cols]`.
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, MatrixError> {
        self.matvec_rows(0, self.rows, input)
    }

    /// `output = W[start..start+count] · input` without widening resident weights.
    pub fn matvec_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, MatrixError> {
        if input.len() != self.cols {
            return Err(MatrixError::InputLength {
                expected: self.cols,
                got: input.len(),
            });
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let end = start.checked_add(count).ok_or_else(|| {
            MatrixError::InvalidShape("matrix row range overflows usize".to_owned())
        })?;
        if end > self.rows {
            return Err(MatrixError::InvalidShape(format!(
                "matrix row range {start}..{end} exceeds 0..{}",
                self.rows
            )));
        }

        let row_bytes = self.cols * 2;
        let rows = &self.little_endian_bytes[start * row_bytes..end * row_bytes];
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecBf16, work);
        let mut output = vec![0.0; count];
        #[cfg(target_arch = "x86_64")]
        let exact_avx2 = std::arch::is_x86_feature_detected!("avx2");
        let dot = |row: &[u8]| {
            #[cfg(target_arch = "x86_64")]
            if exact_avx2 {
                // SAFETY: feature detection proves AVX2 support. Products are generated eight at
                // a time but added to the scalar F64 accumulator in original column order.
                return unsafe { dot_bf16_f64_ordered_avx2(row, input) };
            }
            dot_bf16(row, input)
        };
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .zip(rows.par_chunks_exact(row_bytes))
                    .for_each(|(output, row)| *output = dot(row));
            });
        } else {
            for (output, row) in output.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                *output = dot(row);
            }
        }
        Ok(output)
    }

    /// FP32-accumulating AVX2/FMA GEMV used by Hy4's BF16 output gate.
    pub(crate) fn matvec_fp32(&self, input: &[f32]) -> Result<Vec<f32>, MatrixError> {
        self.matvec_rows_fp32(0, self.rows, input)
    }

    pub(crate) fn matvec_rows_fp32(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, MatrixError> {
        if input.len() != self.cols {
            return Err(MatrixError::InputLength {
                expected: self.cols,
                got: input.len(),
            });
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let end = start
            .checked_add(count)
            .filter(|&end| end <= self.rows)
            .ok_or_else(|| {
                MatrixError::InvalidShape("matrix row range is out of bounds".to_owned())
            })?;
        let row_bytes = self.cols * 2;
        let rows = &self.little_endian_bytes[start * row_bytes..end * row_bytes];
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecBf16, work);
        let mut output = vec![0.0; count];
        #[cfg(target_arch = "x86_64")]
        let avx2_fma = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        let dot = |row: &[u8]| {
            #[cfg(target_arch = "x86_64")]
            if avx2_fma {
                // SAFETY: feature detection proves AVX2/FMA support and the helper bounds every
                // vector load to a complete eight-value group.
                return unsafe { dot_bf16_fp32_avx2_fma(row, input) };
            }
            dot_bf16(row, input)
        };
        if should_parallelize(self.rows, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .zip(rows.par_chunks_exact(row_bytes))
                    .for_each(|(output, row)| *output = dot(row));
            });
        } else {
            for (output, row) in output.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                *output = dot(row);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(output)
    }

    /// `output = Wᵀ · input`.
    pub fn transpose_matvec(&self, input: &[f32]) -> Result<Vec<f32>, MatrixError> {
        if input.len() != self.rows {
            return Err(MatrixError::InputLength {
                expected: self.rows,
                got: input.len(),
            });
        }
        self.transpose_rows_matvec(0, input)
    }

    /// `output = W[start..start+input.len()]ᵀ · input` without widening resident weights.
    pub fn transpose_rows_matvec(
        &self,
        start: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, MatrixError> {
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let end = start.checked_add(input.len()).ok_or_else(|| {
            MatrixError::InvalidShape("matrix row range overflows usize".to_owned())
        })?;
        if end > self.rows {
            return Err(MatrixError::InvalidShape(format!(
                "matrix row range {start}..{end} exceeds 0..{}",
                self.rows
            )));
        }

        let row_bytes = self.cols * 2;
        let rows = &self.little_endian_bytes[start * row_bytes..end * row_bytes];
        let work = input.len().saturating_mul(self.cols);
        let mut output = vec![0.0f64; self.cols];
        if should_parallelize(self.cols, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(column, output)| {
                        let mut sum = 0.0f64;
                        for (row, &value) in rows.chunks_exact(row_bytes).zip(input) {
                            let offset = column * 2;
                            sum +=
                                f64::from(decode_bf16(&row[offset..offset + 2])) * f64::from(value);
                        }
                        *output = sum;
                    });
            });
        } else {
            for (row, &value) in rows.chunks_exact(row_bytes).zip(input) {
                for (column, weight) in row.chunks_exact(2).enumerate() {
                    output[column] += f64::from(decode_bf16(weight)) * f64::from(value);
                }
            }
        }
        Ok(output.into_iter().map(|value| value as f32).collect())
    }
}

#[inline]
fn decode_bf16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
fn dot_bf16(row: &[u8], input: &[f32]) -> f32 {
    row.chunks_exact(2)
        .zip(input)
        .map(|(weight, &value)| f64::from(decode_bf16(weight)) * f64::from(value))
        .sum::<f64>() as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_bf16_f64_ordered_avx2(row: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(row.len(), input.len() * 2);
    let complete = input.len() / 8 * 8;
    // Match Iterator::sum's identity, including a row whose products are all negative zero.
    let mut sum = -0.0f64;
    let mut products = [0.0f64; 8];
    for offset in (0..complete).step_by(8) {
        let packed = _mm_loadu_si128(row.as_ptr().add(offset * 2).cast());
        let bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
        let weights = _mm256_castsi256_ps(bits);
        let inputs = _mm256_loadu_ps(input.as_ptr().add(offset));
        let weight_low = _mm256_cvtps_pd(_mm256_castps256_ps128(weights));
        let weight_high = _mm256_cvtps_pd(_mm256_extractf128_ps(weights, 1));
        let input_low = _mm256_cvtps_pd(_mm256_castps256_ps128(inputs));
        let input_high = _mm256_cvtps_pd(_mm256_extractf128_ps(inputs, 1));
        _mm256_storeu_pd(products.as_mut_ptr(), _mm256_mul_pd(weight_low, input_low));
        _mm256_storeu_pd(
            products.as_mut_ptr().add(4),
            _mm256_mul_pd(weight_high, input_high),
        );
        for product in products {
            sum += product;
        }
    }
    for (column, &input_value) in input.iter().enumerate().skip(complete) {
        let byte = column * 2;
        sum += f64::from(decode_bf16(&row[byte..byte + 2])) * f64::from(input_value);
    }
    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_fp32_avx2_fma(row: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(row.len(), input.len() * 2);
    let complete = input.len() / 8 * 8;
    let mut accumulator = _mm256_setzero_ps();
    for offset in (0..complete).step_by(8) {
        let packed = _mm_loadu_si128(row.as_ptr().add(offset * 2).cast::<__m128i>());
        let weights = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        let values = _mm256_loadu_ps(input.as_ptr().add(offset));
        accumulator = _mm256_fmadd_ps(weights, values, accumulator);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
    let mut sum = lanes.into_iter().sum::<f32>();
    for offset in complete..input.len() {
        sum += decode_bf16(&row[offset * 2..offset * 2 + 2]) * input[offset];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn compact_matrix_preserves_scalar_matrix_contract() {
        let matrix = Bf16Matrix::from_le_bytes(2, 2, bf16_bytes(&[-1.0, 2.0, 3.0, -4.0])).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.resident_bytes(), 8);
        assert_eq!(matrix.row(0).unwrap(), vec![-1.0, 2.0]);
        assert_eq!(matrix.matvec(&[0.5, 2.0]).unwrap(), vec![3.5, -6.5]);
        assert_eq!(matrix.matvec_rows(1, 1, &[0.5, 2.0]).unwrap(), vec![-6.5]);
        assert_eq!(
            matrix.transpose_matvec(&[0.5, 2.0]).unwrap(),
            vec![5.5, -7.0]
        );
        assert_eq!(
            matrix.transpose_rows_matvec(1, &[2.0]).unwrap(),
            vec![6.0, -8.0]
        );
    }

    #[test]
    fn compact_matrix_rejects_invalid_shapes_and_non_finite_values() {
        assert!(matches!(
            Bf16Matrix::from_le_bytes(1, 2, vec![0; 2]),
            Err(MatrixError::InvalidShape(_))
        ));
        assert!(matches!(
            Bf16Matrix::from_le_bytes(1, 1, 0x7f80u16.to_le_bytes().to_vec()),
            Err(MatrixError::NonFinite)
        ));

        let matrix = Bf16Matrix::from_le_bytes(1, 1, bf16_bytes(&[1.0])).unwrap();
        assert!(matches!(
            matrix.matvec(&[f32::NAN]),
            Err(MatrixError::NonFinite)
        ));
        assert!(matches!(
            matrix.row(1),
            Err(MatrixError::RowOutOfBounds { .. })
        ));
    }

    #[test]
    fn compact_matvec_preserves_reference_signed_zero() {
        for columns in [1, 7, 8, 9, 32, 33] {
            let matrix =
                Bf16Matrix::from_le_bytes(2, columns, bf16_bytes(&[-0.0, 0.0].repeat(columns)))
                    .unwrap();
            for input in [vec![1.0; columns], vec![-1.0; columns], vec![0.0; columns]] {
                let actual = matrix.matvec(&input).unwrap();
                for (row, &value) in actual.iter().enumerate() {
                    let expected = dot_bf16(
                        &matrix.little_endian_bytes[row * columns * 2..(row + 1) * columns * 2],
                        &input,
                    );
                    assert_eq!(value.to_bits(), expected.to_bits());
                }
            }
            let negative =
                Bf16Matrix::from_le_bytes(1, columns, bf16_bytes(&vec![-0.0; columns])).unwrap();
            assert_eq!(
                negative.matvec(&vec![1.0; columns]).unwrap()[0].to_bits(),
                (-0.0f32).to_bits()
            );
        }
    }

    #[test]
    fn fp32_matvec_stays_within_ordered_reference_error() {
        let rows = 17usize;
        let cols = 97usize;
        let values = (0..rows * cols)
            .map(|index| ((index.wrapping_mul(29) % 257) as f32 - 128.0) / 64.0)
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|index| ((index.wrapping_mul(17) % 127) as f32 - 63.0) / 32.0)
            .collect::<Vec<_>>();
        let matrix = Bf16Matrix::from_le_bytes(rows, cols, bf16_bytes(&values)).unwrap();
        let reference = matrix.matvec(&input).unwrap();
        let actual = matrix.matvec_fp32(&input).unwrap();
        for row in 0..rows {
            let absolute_sum = values[row * cols..(row + 1) * cols]
                .iter()
                .zip(&input)
                .map(|(&weight, &value)| f64::from(weight * value).abs())
                .sum::<f64>();
            let error = (f64::from(actual[row]) - f64::from(reference[row])).abs();
            let tolerance = 64.0 * f64::from(f32::EPSILON) * absolute_sum.max(1.0);
            assert!(
                error <= tolerance,
                "row {row}: actual={}, reference={}, error={error}, tolerance={tolerance}",
                actual[row],
                reference[row]
            );
        }
    }
}
