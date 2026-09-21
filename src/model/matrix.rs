use rayon::prelude::*;
use serde::Serialize;
use std::fmt;

use crate::execution::{install, should_parallelize};
use crate::profiling::{span_with_work, ProfileStage};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    InvalidShape(String),
    InputLength { expected: usize, got: usize },
    NonFinite,
    RowOutOfBounds { row: usize, rows: usize },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(reason) => write!(f, "invalid matrix shape: {reason}"),
            Self::InputLength { expected, got } => {
                write!(f, "matrix input length: expected {expected}, got {got}")
            }
            Self::NonFinite => f.write_str("matrix data or input contains NaN/infinity"),
            Self::RowOutOfBounds { row, rows } => {
                write!(f, "matrix row {row} is outside 0..{rows}")
            }
        }
    }
}

impl std::error::Error for MatrixError {}

/// Readable row-major `[output, input]` F32 matrix.
///
/// The scalar reference accumulates dot products in f64. Optimized kernels may accumulate
/// differently, but must stay within a documented tolerance against this implementation.
#[derive(Debug, Clone, Serialize)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidShape(
                "rows and columns must be non-zero".to_owned(),
            ));
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| MatrixError::InvalidShape("rows * cols overflows usize".to_owned()))?;
        if data.len() != expected {
            return Err(MatrixError::InvalidShape(format!(
                "[{rows}, {cols}] needs {expected} values, got {}",
                data.len()
            )));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| MatrixError::InvalidShape("rows * cols overflows usize".to_owned()))?;
        Self::new(rows, cols, vec![0.0; len])
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn row(&self, row: usize) -> Result<&[f32], MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::RowOutOfBounds {
                row,
                rows: self.rows,
            });
        }
        Ok(&self.data[row * self.cols..(row + 1) * self.cols])
    }

    /// `output = W · input` for `W[rows, cols]`.
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, MatrixError> {
        self.matvec_rows(0, self.rows, input)
    }

    /// FP32-accumulating GEMV for runtimes whose published kernels materialize this boundary in
    /// FP32. The regular scalar reference deliberately retains its higher-precision accumulator.
    pub(crate) fn matvec_fp32(&self, input: &[f32]) -> Result<Vec<f32>, MatrixError> {
        self.matvec_rows_fp32(0, self.rows, input)
    }

    /// Applies the FP32-accumulating kernel to consecutive input rows while traversing each
    /// resident weight row only once. Every token accumulator observes columns in the same order
    /// as [`Self::matvec_fp32`].
    pub(crate) fn matmul_rows_fp32(
        &self,
        input: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, MatrixError> {
        let expected = batch
            .checked_mul(self.cols)
            .ok_or_else(|| MatrixError::InvalidShape("batch * cols overflows usize".to_owned()))?;
        if batch == 0 || input.len() != expected {
            return Err(MatrixError::InputLength {
                expected,
                got: input.len(),
            });
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let work = batch.saturating_mul(self.rows).saturating_mul(self.cols);
        let mut output_by_row = vec![0.0f32; self.rows.saturating_mul(batch)];
        let compute = |row: &[f32], output: &mut [f32]| {
            for (column, &weight) in row.iter().enumerate() {
                for (token, output) in output.iter_mut().enumerate() {
                    *output += weight * input[token * self.cols + column];
                }
            }
        };
        if should_parallelize(self.rows, work) {
            install(|| {
                self.data
                    .par_chunks_exact(self.cols)
                    .zip(output_by_row.par_chunks_mut(batch))
                    .for_each(|(row, output)| compute(row, output));
            });
        } else {
            for (row, output) in self
                .data
                .chunks_exact(self.cols)
                .zip(output_by_row.chunks_mut(batch))
            {
                compute(row, output);
            }
        }
        let mut output = vec![0.0f32; batch.saturating_mul(self.rows)];
        for (row, values) in output_by_row.chunks_exact(batch).enumerate() {
            for (token, &value) in values.iter().enumerate() {
                output[token * self.rows + row] = value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(output)
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
        let rows = &self.data[start * self.cols..end * self.cols];
        let work = count.saturating_mul(self.cols);
        let mut output = vec![0.0; count];
        let dot = |row: &[f32]| {
            row.iter()
                .zip(input)
                .fold(0.0f32, |sum, (&w, &x)| sum + w * x)
        };
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .zip(rows.par_chunks_exact(self.cols))
                    .for_each(|(out, row)| *out = dot(row))
            });
        } else {
            for (out, row) in output.iter_mut().zip(rows.chunks_exact(self.cols)) {
                *out = dot(row);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(output)
    }

    /// `output = W[start..start+count] · input` without materializing matrix rows.
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
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecF32, work);
        let rows = &self.data[start * self.cols..end * self.cols];
        let mut output = vec![0.0; count];
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .zip(rows.par_chunks_exact(self.cols))
                    .for_each(|(output, row)| {
                        *output = row
                            .iter()
                            .zip(input)
                            .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                            .sum::<f64>() as f32;
                    });
            });
        } else {
            for (output, row) in output.iter_mut().zip(rows.chunks_exact(self.cols)) {
                *output = row
                    .iter()
                    .zip(input)
                    .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                    .sum::<f64>() as f32;
            }
        }
        Ok(output)
    }

    /// `output = Wᵀ · input`, used by MLA weight absorption.
    pub fn transpose_matvec(&self, input: &[f32]) -> Result<Vec<f32>, MatrixError> {
        if input.len() != self.rows {
            return Err(MatrixError::InputLength {
                expected: self.rows,
                got: input.len(),
            });
        }
        self.transpose_rows_matvec(0, input)
    }

    /// `output = W[start..start+input.len()]ᵀ · input` without copying the selected rows.
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
        let work = input.len().saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::TransposeMatvecF32, work);
        let rows = &self.data[start * self.cols..end * self.cols];
        let mut output = vec![0.0f64; self.cols];
        if should_parallelize(self.cols, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(column, output)| {
                        let mut sum = 0.0f64;
                        for (row, &value) in rows.chunks_exact(self.cols).zip(input) {
                            sum += f64::from(row[column]) * f64::from(value);
                        }
                        *output = sum;
                    });
            });
        } else {
            for (row, &value) in rows.chunks_exact(self.cols).zip(input) {
                for (column, &weight) in row.iter().enumerate() {
                    output[column] += f64::from(weight) * f64::from(value);
                }
            }
        }
        Ok(output.into_iter().map(|value| value as f32).collect())
    }

    /// Applies the same matrix to `batch` consecutive input rows `[batch, cols]`.
    pub fn matmul_rows(&self, input: &[f32], batch: usize) -> Result<Vec<f32>, MatrixError> {
        let expected = batch
            .checked_mul(self.cols)
            .ok_or_else(|| MatrixError::InvalidShape("batch * cols overflows usize".to_owned()))?;
        if input.len() != expected {
            return Err(MatrixError::InputLength {
                expected,
                got: input.len(),
            });
        }
        let mut output = Vec::with_capacity(batch.saturating_mul(self.rows));
        for row in input.chunks_exact(self.cols) {
            output.extend(self.matvec(row)?);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_contract_is_y_equals_x_w_transpose() {
        let matrix = DenseMatrix::new(2, 3, vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0]).unwrap();
        assert_eq!(matrix.matvec(&[2.0, 3.0, -1.0]).unwrap(), vec![5.0, -4.5]);
    }

    #[test]
    fn transpose_matvec_matches_explicit_transpose() {
        let matrix = DenseMatrix::new(2, 3, vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0]).unwrap();
        assert_eq!(
            matrix.transpose_matvec(&[2.0, -3.0]).unwrap(),
            vec![5.0, 2.5, -6.0]
        );
        assert_eq!(
            matrix.transpose_rows_matvec(1, &[-3.0]).unwrap(),
            vec![3.0, -1.5, -12.0]
        );
        assert_eq!(
            matrix.matvec_rows(1, 1, &[2.0, 3.0, -1.0]).unwrap(),
            vec![-4.5]
        );
    }

    #[test]
    fn batched_rows_keep_token_order() {
        let matrix = DenseMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        assert_eq!(
            matrix.matmul_rows(&[1.0, 2.0, 3.0, 4.0], 2).unwrap(),
            vec![1.0, 4.0, 3.0, 8.0]
        );
    }

    #[test]
    fn fp32_batched_rows_match_independent_accumulation_bits() {
        let rows = 17;
        let cols = 97;
        let batch = 12;
        let matrix = DenseMatrix::new(
            rows,
            cols,
            (0..rows * cols)
                .map(|index| ((index * 29 % 257) as f32 - 128.0) / 64.0)
                .collect(),
        )
        .unwrap();
        let input = (0..batch * cols)
            .map(|index| ((index * 17 % 127) as f32 - 63.0) / 32.0)
            .collect::<Vec<_>>();
        let expected = input
            .chunks_exact(cols)
            .flat_map(|input| matrix.matvec_fp32(input).unwrap())
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let actual = matrix
            .matmul_rows_fp32(&input, batch)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
