use super::{Bf16Matrix, DenseMatrix, MatrixError};
use crate::math::{Int4Matrix, Int8Matrix, MxError, MxFp4Matrix, MxFp8Matrix, QuantError};
use std::fmt;

#[derive(Debug)]
pub enum WeightError {
    Dense(MatrixError),
    Quantized(QuantError),
    Mx(MxError),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(error) => error.fmt(f),
            Self::Quantized(error) => error.fmt(f),
            Self::Mx(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for WeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Dense(error) => Some(error),
            Self::Quantized(error) => Some(error),
            Self::Mx(error) => Some(error),
        }
    }
}

impl From<MatrixError> for WeightError {
    fn from(value: MatrixError) -> Self {
        Self::Dense(value)
    }
}

impl From<QuantError> for WeightError {
    fn from(value: QuantError) -> Self {
        Self::Quantized(value)
    }
}

impl From<MxError> for WeightError {
    fn from(value: MxError) -> Self {
        Self::Mx(value)
    }
}

/// A readable execution abstraction for the mixed GLM checkpoint container.
///
/// Activations remain F32. This is a correctness baseline, not the later SIMD kernel API.
#[derive(Debug, Clone)]
pub enum WeightMatrix {
    F32(DenseMatrix),
    Bf16(Bf16Matrix),
    Int8PerRow(Int8Matrix),
    Int4(Int4Matrix),
    MxFp8(MxFp8Matrix),
    MxFp4(MxFp4Matrix),
}

impl From<DenseMatrix> for WeightMatrix {
    fn from(value: DenseMatrix) -> Self {
        Self::F32(value)
    }
}

impl WeightMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::F32(matrix) => matrix.rows(),
            Self::Bf16(matrix) => matrix.rows(),
            Self::Int8PerRow(matrix) => matrix.rows(),
            Self::Int4(matrix) => matrix.rows(),
            Self::MxFp8(matrix) => matrix.rows(),
            Self::MxFp4(matrix) => matrix.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F32(matrix) => matrix.cols(),
            Self::Bf16(matrix) => matrix.cols(),
            Self::Int8PerRow(matrix) => matrix.cols(),
            Self::Int4(matrix) => matrix.cols(),
            Self::MxFp8(matrix) => matrix.cols(),
            Self::MxFp4(matrix) => matrix.cols(),
        }
    }

    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, WeightError> {
        Ok(match self {
            Self::F32(matrix) => matrix.matvec(input)?,
            Self::Bf16(matrix) => matrix.matvec(input)?,
            Self::Int8PerRow(matrix) => matrix.matvec(input)?,
            Self::Int4(matrix) => matrix.matvec(input)?,
            Self::MxFp8(matrix) => matrix.matvec(input)?,
            Self::MxFp4(matrix) => matrix.matvec(input)?,
        })
    }

    /// Applies one weight matrix to consecutive input rows `[batch, cols]`. Native ModelOpt
    /// matrices use compact multi-input kernels; wider batches are split evenly to avoid a small
    /// scalar tail while keeping AVX2 register pressure bounded.
    pub fn matmul_rows(&self, input: &[f32], batch: usize) -> Result<Vec<f32>, WeightError> {
        let expected = batch
            .checked_mul(self.cols())
            .ok_or_else(|| MatrixError::InvalidShape("batch * cols overflows usize".to_owned()))?;
        if batch == 0 || input.len() != expected {
            return Err(MatrixError::InputLength {
                expected,
                got: input.len(),
            }
            .into());
        }
        if let Self::MxFp8(matrix) = self {
            let chunks = batch.div_ceil(12);
            let base_batch = batch / chunks;
            let larger_chunks = batch % chunks;
            let mut output = Vec::with_capacity(batch.saturating_mul(matrix.rows()));
            let mut token = 0usize;
            for chunk in 0..chunks {
                let chunk_batch = base_batch + usize::from(chunk < larger_chunks);
                let start = token.saturating_mul(matrix.cols());
                let end = start.saturating_add(chunk_batch.saturating_mul(matrix.cols()));
                output.extend(matrix.matmul_rows(&input[start..end], chunk_batch)?);
                token += chunk_batch;
            }
            return Ok(output);
        }
        if let Self::MxFp4(matrix) = self {
            let chunks = batch.div_ceil(12);
            let base_batch = batch / chunks;
            let larger_chunks = batch % chunks;
            let mut output = Vec::with_capacity(batch.saturating_mul(matrix.rows()));
            let mut token = 0usize;
            for chunk in 0..chunks {
                let chunk_batch = base_batch + usize::from(chunk < larger_chunks);
                let start = token.saturating_mul(matrix.cols());
                let end = start.saturating_add(chunk_batch.saturating_mul(matrix.cols()));
                output.extend(matrix.matmul_rows(&input[start..end], chunk_batch)?);
                token += chunk_batch;
            }
            return Ok(output);
        }
        let mut output = Vec::with_capacity(batch.saturating_mul(self.rows()));
        for token in input.chunks_exact(self.cols()) {
            output.extend(self.matvec(token)?);
        }
        Ok(output)
    }

    /// Applies a contiguous output-row range to consecutive input rows `[batch, cols]`.
    pub fn matmul_row_range(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, WeightError> {
        let expected = batch
            .checked_mul(self.cols())
            .ok_or_else(|| MatrixError::InvalidShape("batch * cols overflows usize".to_owned()))?;
        let end = start
            .checked_add(count)
            .filter(|&end| end <= self.rows())
            .ok_or_else(|| MatrixError::InvalidShape("output row range is invalid".to_owned()))?;
        if batch == 0 || input.len() != expected {
            return Err(MatrixError::InputLength {
                expected,
                got: input.len(),
            }
            .into());
        }
        if let Self::MxFp8(matrix) = self {
            let chunks = batch.div_ceil(12);
            let base_batch = batch / chunks;
            let larger_chunks = batch % chunks;
            let mut output = Vec::with_capacity(batch.saturating_mul(count));
            let mut token = 0usize;
            for chunk in 0..chunks {
                let chunk_batch = base_batch + usize::from(chunk < larger_chunks);
                let input_start = token.saturating_mul(matrix.cols());
                let input_end =
                    input_start.saturating_add(chunk_batch.saturating_mul(matrix.cols()));
                output.extend(matrix.matmul_row_range(
                    start,
                    count,
                    &input[input_start..input_end],
                    chunk_batch,
                )?);
                token += chunk_batch;
            }
            return Ok(output);
        }
        let mut output = Vec::with_capacity(batch.saturating_mul(count));
        for token in input.chunks_exact(self.cols()) {
            output.extend(self.matvec_rows(start, count, token)?);
        }
        debug_assert_eq!(end, start + count);
        Ok(output)
    }

    pub(crate) fn matvec_bf16_fp32(&self, input: &[f32]) -> Result<Vec<f32>, WeightError> {
        match self {
            Self::Bf16(matrix) => Ok(matrix.matvec_fp32(input)?),
            _ => self.matvec(input),
        }
    }

    /// FP32 accumulation boundary used by the published DeepSeek-V4.1 kernels.
    pub(crate) fn matvec_fp32_accum(&self, input: &[f32]) -> Result<Vec<f32>, WeightError> {
        match self {
            Self::F32(matrix) => Ok(matrix.matvec_fp32(input)?),
            Self::Bf16(matrix) => Ok(matrix.matvec_fp32(input)?),
            Self::MxFp4(matrix) => Ok(matrix.matvec_fp32(input)?),
            _ => self.matvec(input),
        }
    }

    /// Batched form of [`Self::matvec_fp32_accum`]. F32 and native MX formats use batch kernels
    /// that preserve their respective scalar accumulation order; other formats retain per-token
    /// dispatch until an equivalent batch kernel exists.
    pub(crate) fn matmul_rows_fp32_accum(
        &self,
        input: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, WeightError> {
        let expected = batch
            .checked_mul(self.cols())
            .ok_or_else(|| MatrixError::InvalidShape("batch * cols overflows usize".to_owned()))?;
        if batch == 0 || input.len() != expected {
            return Err(MatrixError::InputLength {
                expected,
                got: input.len(),
            }
            .into());
        }
        if let Self::F32(matrix) = self {
            let chunks = batch.div_ceil(12);
            let base_batch = batch / chunks;
            let larger_chunks = batch % chunks;
            let mut output = Vec::with_capacity(batch.saturating_mul(matrix.rows()));
            let mut token = 0usize;
            for chunk in 0..chunks {
                let chunk_batch = base_batch + usize::from(chunk < larger_chunks);
                let start = token.saturating_mul(matrix.cols());
                let end = start.saturating_add(chunk_batch.saturating_mul(matrix.cols()));
                output.extend(matrix.matmul_rows_fp32(&input[start..end], chunk_batch)?);
                token += chunk_batch;
            }
            return Ok(output);
        }
        if let Self::MxFp8(matrix) = self {
            let chunks = batch.div_ceil(12);
            let base_batch = batch / chunks;
            let larger_chunks = batch % chunks;
            let mut output = Vec::with_capacity(batch.saturating_mul(matrix.rows()));
            let mut token = 0usize;
            for chunk in 0..chunks {
                let chunk_batch = base_batch + usize::from(chunk < larger_chunks);
                let start = token.saturating_mul(matrix.cols());
                let end = start.saturating_add(chunk_batch.saturating_mul(matrix.cols()));
                output.extend(matrix.matmul_rows(&input[start..end], chunk_batch)?);
                token += chunk_batch;
            }
            return Ok(output);
        }
        if let Self::MxFp4(matrix) = self {
            let chunks = batch.div_ceil(12);
            let base_batch = batch / chunks;
            let larger_chunks = batch % chunks;
            let mut output = Vec::with_capacity(batch.saturating_mul(matrix.rows()));
            let mut token = 0usize;
            for chunk in 0..chunks {
                let chunk_batch = base_batch + usize::from(chunk < larger_chunks);
                let start = token.saturating_mul(matrix.cols());
                let end = start.saturating_add(chunk_batch.saturating_mul(matrix.cols()));
                output.extend(matrix.matmul_rows_fp32(&input[start..end], chunk_batch)?);
                token += chunk_batch;
            }
            return Ok(output);
        }
        let mut output = Vec::with_capacity(batch.saturating_mul(self.rows()));
        for token in input.chunks_exact(self.cols()) {
            output.extend(self.matvec_fp32_accum(token)?);
        }
        Ok(output)
    }

    pub(crate) fn matvec_rows_fp32(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, WeightError> {
        Ok(match self {
            Self::F32(matrix) => matrix.matvec_rows_fp32(start, count, input)?,
            Self::Bf16(matrix) => matrix.matvec_rows_fp32(start, count, input)?,
            Self::MxFp4(matrix) => matrix.matvec_rows_fp32(start, count, input)?,
            _ => self.matvec_rows(start, count, input)?,
        })
    }

    pub fn matvec_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, WeightError> {
        Ok(match self {
            Self::F32(matrix) => matrix.matvec_rows(start, count, input)?,
            Self::Bf16(matrix) => matrix.matvec_rows(start, count, input)?,
            Self::Int8PerRow(matrix) => matrix.matvec_rows(start, count, input)?,
            Self::Int4(matrix) => matrix.matvec_rows(start, count, input)?,
            Self::MxFp8(matrix) => matrix.matvec_rows(start, count, input)?,
            Self::MxFp4(matrix) => matrix.matvec_rows(start, count, input)?,
        })
    }

    pub fn transpose_matvec(&self, input: &[f32]) -> Result<Vec<f32>, WeightError> {
        Ok(match self {
            Self::F32(matrix) => matrix.transpose_matvec(input)?,
            Self::Bf16(matrix) => matrix.transpose_matvec(input)?,
            Self::Int8PerRow(matrix) => matrix.transpose_matvec(input)?,
            Self::Int4(matrix) => matrix.transpose_matvec(input)?,
            Self::MxFp8(_) | Self::MxFp4(_) => self.transpose_rows_matvec(0, input)?,
        })
    }

    pub fn transpose_rows_matvec(
        &self,
        start: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, WeightError> {
        Ok(match self {
            Self::F32(matrix) => matrix.transpose_rows_matvec(start, input)?,
            Self::Bf16(matrix) => matrix.transpose_rows_matvec(start, input)?,
            Self::Int8PerRow(matrix) => matrix.transpose_rows_matvec(start, input)?,
            Self::Int4(matrix) => matrix.transpose_rows_matvec(start, input)?,
            Self::MxFp8(matrix) => matrix.transpose_rows_matvec(start, input)?,
            Self::MxFp4(matrix) => matrix.transpose_rows_matvec(start, input)?,
        })
    }

    pub fn write_row(&self, row: usize, output: &mut [f32]) -> Result<(), WeightError> {
        match self {
            Self::F32(matrix) => {
                if output.len() != matrix.cols() {
                    return Err(MatrixError::InputLength {
                        expected: matrix.cols(),
                        got: output.len(),
                    }
                    .into());
                }
                output.copy_from_slice(matrix.row(row)?);
                Ok(())
            }
            Self::Bf16(matrix) => Ok(matrix.write_row(row, output)?),
            Self::Int8PerRow(matrix) => Ok(matrix.write_row(row, output)?),
            Self::Int4(matrix) => Ok(matrix.write_row(row, output)?),
            Self::MxFp8(matrix) => {
                if output.len() != matrix.cols() {
                    return Err(MatrixError::InputLength {
                        expected: matrix.cols(),
                        got: output.len(),
                    }
                    .into());
                }
                output.copy_from_slice(&matrix.row(row)?);
                Ok(())
            }
            Self::MxFp4(matrix) => {
                if output.len() != matrix.cols() {
                    return Err(MatrixError::InputLength {
                        expected: matrix.cols(),
                        got: output.len(),
                    }
                    .into());
                }
                output.copy_from_slice(&matrix.row(row)?);
                Ok(())
            }
        }
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>, WeightError> {
        let mut output = vec![0.0; self.cols()];
        self.write_row(row, &mut output)?;
        Ok(output)
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::F32(matrix) => matrix
                .data()
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
            Self::Bf16(matrix) => matrix.resident_bytes(),
            Self::Int8PerRow(matrix) => matrix.resident_bytes(),
            Self::Int4(matrix) => matrix.resident_bytes(),
            Self::MxFp8(matrix) => matrix.resident_bytes(),
            Self::MxFp4(matrix) => matrix.resident_bytes(),
        }
    }

    pub fn uses_mx_activation_quantization(&self) -> bool {
        matches!(self, Self::MxFp8(_) | Self::MxFp4(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_formats_share_matvec_and_row_contracts() {
        let dense = WeightMatrix::F32(DenseMatrix::new(2, 2, vec![-1.0, 2.0, 3.0, -4.0]).unwrap());
        let int8 = WeightMatrix::Int8PerRow(
            Int8Matrix::from_packed(2, 2, vec![0xff, 2, 3, 0xfc], vec![1.0, 1.0]).unwrap(),
        );
        let int4 = WeightMatrix::Int4(
            Int4Matrix::from_packed(2, 2, 2, vec![0xa7, 0x4b], vec![1.0, 1.0]).unwrap(),
        );
        for matrix in [&dense, &int8, &int4] {
            assert_eq!(matrix.row(0).unwrap(), vec![-1.0, 2.0]);
            assert_eq!(matrix.matvec(&[0.5, 2.0]).unwrap(), vec![3.5, -6.5]);
            assert_eq!(
                matrix.matmul_rows(&[0.5, 2.0, 1.0, -1.0], 2).unwrap(),
                vec![3.5, -6.5, -3.0, 7.0]
            );
            assert_eq!(
                matrix.transpose_matvec(&[0.5, 2.0]).unwrap(),
                vec![5.5, -7.0]
            );
            assert_eq!(
                matrix.transpose_rows_matvec(1, &[2.0]).unwrap(),
                vec![6.0, -8.0]
            );
            assert_eq!(matrix.matvec_rows(1, 1, &[0.5, 2.0]).unwrap(), vec![-6.5]);
        }
    }

    #[test]
    fn mxfp4_weight_batches_split_without_changing_token_order() {
        let matrix = WeightMatrix::MxFp4(
            MxFp4Matrix::from_packed(
                2,
                8,
                8,
                vec![0x21, 0x43, 0x65, 0x87, 0x12, 0x34, 0x56, 0x78],
                vec![127, 126],
            )
            .unwrap(),
        );
        let batch = 25usize;
        let input = (0..batch * matrix.cols())
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 16.0)
            .collect::<Vec<_>>();
        let expected = input
            .chunks_exact(matrix.cols())
            .flat_map(|input| matrix.matvec(input).unwrap())
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let actual = matrix
            .matmul_rows(&input, batch)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn mxfp8_fp32_accum_batches_split_without_changing_token_order() {
        let matrix = WeightMatrix::MxFp8(
            MxFp8Matrix::from_packed(
                3,
                8,
                2,
                8,
                (0..24)
                    .map(|index| [0x00, 0x20, 0x38, 0xb8][index % 4])
                    .collect(),
                vec![127, 126],
            )
            .unwrap(),
        );
        let batch = 25usize;
        let input = (0..batch * matrix.cols())
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 16.0)
            .collect::<Vec<_>>();
        let expected = input
            .chunks_exact(matrix.cols())
            .flat_map(|input| matrix.matvec_fp32_accum(input).unwrap())
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let actual = matrix
            .matmul_rows_fp32_accum(&input, batch)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn large_parallel_kernels_are_bit_exact_against_ordered_scalar_dots() {
        let rows = 65;
        let cols = 1_025;
        let input = (0..cols)
            .map(|column| (column as f32 % 17.0 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let transpose_input = (0..rows)
            .map(|row| (row as f32 % 11.0 - 5.0) / 8.0)
            .collect::<Vec<_>>();
        let dense_values = (0..rows * cols)
            .map(|index| (index as f32 % 23.0 - 11.0) / 32.0)
            .collect::<Vec<_>>();
        let matrices = [
            WeightMatrix::F32(DenseMatrix::new(rows, cols, dense_values).unwrap()),
            WeightMatrix::Int8PerRow(
                Int8Matrix::from_packed(
                    rows,
                    cols,
                    (0..rows * cols)
                        .map(|index| (index as i8 % 31) as u8)
                        .collect(),
                    vec![0.03125; rows],
                )
                .unwrap(),
            ),
            WeightMatrix::Int4(
                Int4Matrix::from_packed(
                    rows,
                    cols,
                    64,
                    (0..rows * cols.div_ceil(2))
                        .map(|index| (index as u8).wrapping_mul(37))
                        .collect(),
                    vec![0.0625; rows * cols.div_ceil(64)],
                )
                .unwrap(),
            ),
            WeightMatrix::MxFp8(
                MxFp8Matrix::from_packed(
                    rows,
                    cols,
                    2,
                    32,
                    (0..rows * cols)
                        .map(|index| [0x00, 0x20, 0x38, 0xb8][index % 4])
                        .collect(),
                    vec![127; rows.div_ceil(2) * cols.div_ceil(32)],
                )
                .unwrap(),
            ),
            WeightMatrix::MxFp4(
                MxFp4Matrix::from_packed(
                    rows,
                    cols,
                    32,
                    vec![0x2a; rows * cols.div_ceil(2)],
                    vec![127; rows * cols.div_ceil(32)],
                )
                .unwrap(),
            ),
        ];

        for matrix in matrices {
            let materialized = (0..rows)
                .map(|row| matrix.row(row).unwrap())
                .collect::<Vec<_>>();
            let expected = materialized
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(&input)
                        .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                        .sum::<f64>() as f32
                })
                .collect::<Vec<_>>();
            let actual = matrix.matvec(&input).unwrap();
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

            let expected_transpose = (0..cols)
                .map(|column| {
                    materialized
                        .iter()
                        .zip(&transpose_input)
                        .map(|(row, &value)| f64::from(value) * f64::from(row[column]))
                        .sum::<f64>() as f32
                })
                .collect::<Vec<_>>();
            let actual_transpose = matrix.transpose_matvec(&transpose_input).unwrap();
            assert_eq!(
                actual_transpose
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_transpose
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}
