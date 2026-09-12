//! Readable reference implementation of Colibrì-compatible signed INT4 matrices.

use rayon::prelude::*;
use std::fmt;

use crate::execution::{install, should_parallelize};
use crate::profiling::{span_with_work, ProfileStage};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    Shape(String),
    InvalidGroupSize,
    NonFinite,
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(f, "invalid INT4 shape: {reason}"),
            Self::InvalidGroupSize => f.write_str("INT4 group size must be greater than zero"),
            Self::NonFinite => f.write_str("INT4 source contains NaN or infinity"),
        }
    }
}

impl std::error::Error for QuantError {}

/// Row-major `[rows, cols]` matrix with low-nibble-first signed INT4 codes.
///
/// A stored nibble `code` represents integer `code - 8`. `group_size=cols` gives the original
/// per-row Colibrì format; a smaller value (normally 64) gives grouped-scale INT4.
#[derive(Debug, Clone)]
pub struct Int4Matrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
}

impl Int4Matrix {
    pub fn quantize_per_row(rows: usize, cols: usize, values: &[f32]) -> Result<Self, QuantError> {
        Self::quantize_grouped(rows, cols, cols, values)
    }

    pub fn quantize_grouped(
        rows: usize,
        cols: usize,
        group_size: usize,
        values: &[f32],
    ) -> Result<Self, QuantError> {
        validate_geometry(rows, cols, group_size, values.len())?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(QuantError::NonFinite);
        }
        let row_bytes = cols.div_ceil(2);
        let groups_per_row = cols.div_ceil(group_size);
        let mut packed = vec![0x88u8; rows * row_bytes];
        let mut scales = vec![0.0f32; rows * groups_per_row];

        for row in 0..rows {
            let row_values = &values[row * cols..(row + 1) * cols];
            for group in 0..groups_per_row {
                let start = group * group_size;
                let end = (start + group_size).min(cols);
                let maximum = row_values[start..end]
                    .iter()
                    .fold(0.0f32, |acc, value| acc.max(value.abs()));
                scales[row * groups_per_row + group] = (maximum / 7.0).max(1e-8);
            }
            for col in 0..cols {
                let scale = scales[row * groups_per_row + col / group_size];
                let quantized = round_ties_even(row_values[col] / scale).clamp(-8, 7);
                let code = (quantized + 8) as u8;
                let byte = &mut packed[row * row_bytes + col / 2];
                if col % 2 == 0 {
                    *byte = (*byte & 0xf0) | code;
                } else {
                    *byte = (*byte & 0x0f) | (code << 4);
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            packed,
            scales,
        })
    }

    pub fn from_packed(
        rows: usize,
        cols: usize,
        group_size: usize,
        packed: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self, QuantError> {
        validate_geometry(rows, cols, group_size, rows.saturating_mul(cols))?;
        let expected_packed = rows
            .checked_mul(cols.div_ceil(2))
            .ok_or_else(|| QuantError::Shape("packed length overflows usize".to_owned()))?;
        let expected_scales = rows
            .checked_mul(cols.div_ceil(group_size))
            .ok_or_else(|| QuantError::Shape("scale length overflows usize".to_owned()))?;
        if packed.len() != expected_packed {
            return Err(QuantError::Shape(format!(
                "expected {expected_packed} packed bytes, got {}",
                packed.len()
            )));
        }
        if scales.len() != expected_scales {
            return Err(QuantError::Shape(format!(
                "expected {expected_scales} scales, got {}",
                scales.len()
            )));
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(QuantError::NonFinite);
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            packed,
            scales,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn dequantize_row(&self, row: usize) -> Result<Vec<f32>, QuantError> {
        if row >= self.rows {
            return Err(QuantError::Shape(format!(
                "row {row} is outside 0..{}",
                self.rows
            )));
        }
        Ok((0..self.cols).map(|col| self.value(row, col)).collect())
    }

    pub fn write_row(&self, row: usize, output: &mut [f32]) -> Result<(), QuantError> {
        if row >= self.rows {
            return Err(QuantError::Shape(format!(
                "row {row} is outside 0..{}",
                self.rows
            )));
        }
        if output.len() != self.cols {
            return Err(QuantError::Shape(format!(
                "row output needs {} values, got {}",
                self.cols,
                output.len()
            )));
        }
        for (col, value) in output.iter_mut().enumerate() {
            *value = self.value(row, col);
        }
        Ok(())
    }

    /// Scalar `y = W x`, the reference contract for future SIMD kernels.
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, QuantError> {
        self.matvec_rows(0, self.rows, input)
    }

    pub fn matvec_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, QuantError> {
        if input.len() != self.cols {
            return Err(QuantError::Shape(format!(
                "matvec expects {} inputs, got {}",
                self.cols,
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(QuantError::NonFinite);
        }
        let end = self.validate_row_range(start, count)?;
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecInt4, work);
        let mut output = vec![0.0f32; count];
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(offset, output)| {
                        *output = self.dot_row(start + offset, input) as f32;
                    });
            });
        } else {
            for (row, output_value) in (start..end).zip(&mut output) {
                *output_value = self.dot_row(row, input) as f32;
            }
        }
        Ok(output)
    }

    pub fn transpose_matvec(&self, input: &[f32]) -> Result<Vec<f32>, QuantError> {
        if input.len() != self.rows {
            return Err(QuantError::Shape(format!(
                "transpose matvec expects {} inputs, got {}",
                self.rows,
                input.len()
            )));
        }
        self.transpose_rows_matvec(0, input)
    }

    pub fn transpose_rows_matvec(
        &self,
        start: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, QuantError> {
        if input.iter().any(|value| !value.is_finite()) {
            return Err(QuantError::NonFinite);
        }
        let end = self.validate_row_range(start, input.len())?;
        let work = input.len().saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::TransposeMatvecInt4, work);
        let mut output = vec![0.0f64; self.cols];
        if should_parallelize(self.cols, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(column, output)| {
                        let mut sum = 0.0f64;
                        for (row, &input_value) in (start..end).zip(input) {
                            sum += f64::from(self.value(row, column)) * f64::from(input_value);
                        }
                        *output = sum;
                    });
            });
        } else {
            for (row, &input_value) in (start..end).zip(input) {
                let row_bytes = self.cols.div_ceil(2);
                let groups_per_row = self.cols.div_ceil(self.group_size);
                for group in 0..groups_per_row {
                    let group_start = group * self.group_size;
                    let group_end = (group_start + self.group_size).min(self.cols);
                    let scale = self.scales[row * groups_per_row + group];
                    let coefficient = f64::from(input_value);
                    self.for_each_packed_pair(
                        row * row_bytes,
                        group_start,
                        group_end,
                        |col, code| {
                            output[col] +=
                                f64::from((i32::from(code) - 8) as f32 * scale) * coefficient;
                        },
                    );
                }
            }
        }
        Ok(output.into_iter().map(|value| value as f32).collect())
    }

    pub fn resident_bytes(&self) -> usize {
        self.packed
            .len()
            .saturating_add(self.scales.len().saturating_mul(std::mem::size_of::<f32>()))
    }

    fn validate_row_range(&self, start: usize, count: usize) -> Result<usize, QuantError> {
        let end = start
            .checked_add(count)
            .ok_or_else(|| QuantError::Shape("matrix row range overflows usize".to_owned()))?;
        if end > self.rows {
            return Err(QuantError::Shape(format!(
                "matrix row range {start}..{end} exceeds 0..{}",
                self.rows
            )));
        }
        Ok(end)
    }

    fn dot_row(&self, row: usize, input: &[f32]) -> f64 {
        let row_bytes = self.cols.div_ceil(2);
        let groups_per_row = self.cols.div_ceil(self.group_size);
        let mut sum = 0.0f64;
        for group in 0..groups_per_row {
            let group_start = group * self.group_size;
            let group_end = (group_start + self.group_size).min(self.cols);
            let scale = self.scales[row * groups_per_row + group];
            self.for_each_packed_pair(row * row_bytes, group_start, group_end, |col, code| {
                let weight = (i32::from(code) - 8) as f32 * scale;
                sum += f64::from(weight) * f64::from(input[col]);
            });
        }
        sum
    }

    fn for_each_packed_pair(
        &self,
        row_offset: usize,
        start: usize,
        end: usize,
        mut visit: impl FnMut(usize, u8),
    ) {
        let mut col = start;
        if col % 2 == 1 && col < end {
            visit(col, self.packed[row_offset + col / 2] >> 4);
            col += 1;
        }
        while col + 1 < end {
            let byte = self.packed[row_offset + col / 2];
            visit(col, byte & 0x0f);
            visit(col + 1, byte >> 4);
            col += 2;
        }
        if col < end {
            visit(col, self.packed[row_offset + col / 2] & 0x0f);
        }
    }

    fn value(&self, row: usize, col: usize) -> f32 {
        let row_bytes = self.cols.div_ceil(2);
        let byte = self.packed[row * row_bytes + col / 2];
        let code = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        let groups_per_row = self.cols.div_ceil(self.group_size);
        let scale = self.scales[row * groups_per_row + col / self.group_size];
        (i32::from(code) - 8) as f32 * scale
    }
}

/// Row-major symmetric INT8 with one F32 scale per output row.
#[derive(Debug, Clone)]
pub struct Int8Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u8>,
    scales: Vec<f32>,
}

impl Int8Matrix {
    pub fn from_packed(
        rows: usize,
        cols: usize,
        bytes: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self, QuantError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| QuantError::Shape("rows * cols overflows usize".to_owned()))?;
        if rows == 0 || cols == 0 || bytes.len() != expected {
            return Err(QuantError::Shape(format!(
                "INT8 [{rows},{cols}] needs {expected} bytes, got {}",
                bytes.len()
            )));
        }
        if scales.len() != rows {
            return Err(QuantError::Shape(format!(
                "INT8 [{rows},{cols}] needs {rows} scales, got {}",
                scales.len()
            )));
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(QuantError::NonFinite);
        }
        Ok(Self {
            rows,
            cols,
            data: bytes,
            scales,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn packed(&self) -> &[u8] {
        &self.data
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn write_row(&self, row: usize, output: &mut [f32]) -> Result<(), QuantError> {
        if row >= self.rows {
            return Err(QuantError::Shape(format!(
                "row {row} is outside 0..{}",
                self.rows
            )));
        }
        if output.len() != self.cols {
            return Err(QuantError::Shape(format!(
                "row output needs {} values, got {}",
                self.cols,
                output.len()
            )));
        }
        let scale = self.scales[row];
        for (output, &value) in output
            .iter_mut()
            .zip(&self.data[row * self.cols..(row + 1) * self.cols])
        {
            *output = f32::from(value as i8) * scale;
        }
        Ok(())
    }

    pub fn dequantize_row(&self, row: usize) -> Result<Vec<f32>, QuantError> {
        let mut output = vec![0.0; self.cols];
        self.write_row(row, &mut output)?;
        Ok(output)
    }

    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, QuantError> {
        self.matvec_rows(0, self.rows, input)
    }

    pub fn matvec_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, QuantError> {
        if input.len() != self.cols {
            return Err(QuantError::Shape(format!(
                "matvec expects {} inputs, got {}",
                self.cols,
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(QuantError::NonFinite);
        }
        let end = self.validate_row_range(start, count)?;
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecInt8, work);
        let rows = &self.data[start * self.cols..end * self.cols];
        let mut output = vec![0.0; count];
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .zip(rows.par_chunks_exact(self.cols))
                    .enumerate()
                    .for_each(|(offset, (output, values))| {
                        let sum = values
                            .iter()
                            .zip(input)
                            .map(|(&weight, &value)| f64::from(weight as i8) * f64::from(value))
                            .sum::<f64>();
                        *output = (sum * f64::from(self.scales[start + offset])) as f32;
                    });
            });
        } else {
            for (offset, (output, values)) in output
                .iter_mut()
                .zip(rows.chunks_exact(self.cols))
                .enumerate()
            {
                let sum = values
                    .iter()
                    .zip(input)
                    .map(|(&weight, &value)| f64::from(weight as i8) * f64::from(value))
                    .sum::<f64>();
                *output = (sum * f64::from(self.scales[start + offset])) as f32;
            }
        }
        Ok(output)
    }

    pub fn transpose_matvec(&self, input: &[f32]) -> Result<Vec<f32>, QuantError> {
        if input.len() != self.rows {
            return Err(QuantError::Shape(format!(
                "transpose matvec expects {} inputs, got {}",
                self.rows,
                input.len()
            )));
        }
        self.transpose_rows_matvec(0, input)
    }

    pub fn transpose_rows_matvec(
        &self,
        start: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, QuantError> {
        if input.iter().any(|value| !value.is_finite()) {
            return Err(QuantError::NonFinite);
        }
        let end = self.validate_row_range(start, input.len())?;
        let work = input.len().saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::TransposeMatvecInt8, work);
        let rows = &self.data[start * self.cols..end * self.cols];
        let mut output = vec![0.0f64; self.cols];
        if should_parallelize(self.cols, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(column, output)| {
                        let mut sum = 0.0f64;
                        for (offset, (&input_value, weights)) in
                            input.iter().zip(rows.chunks_exact(self.cols)).enumerate()
                        {
                            let scaled =
                                f64::from(input_value) * f64::from(self.scales[start + offset]);
                            sum += scaled * f64::from(weights[column] as i8);
                        }
                        *output = sum;
                    });
            });
        } else {
            for (row, (&input_value, weights)) in
                input.iter().zip(rows.chunks_exact(self.cols)).enumerate()
            {
                let scaled = f64::from(input_value) * f64::from(self.scales[start + row]);
                for (output, &weight) in output.iter_mut().zip(weights) {
                    *output += scaled * f64::from(weight as i8);
                }
            }
        }
        Ok(output.into_iter().map(|value| value as f32).collect())
    }

    pub fn resident_bytes(&self) -> usize {
        self.data
            .len()
            .saturating_add(self.scales.len().saturating_mul(std::mem::size_of::<f32>()))
    }

    fn validate_row_range(&self, start: usize, count: usize) -> Result<usize, QuantError> {
        let end = start
            .checked_add(count)
            .ok_or_else(|| QuantError::Shape("matrix row range overflows usize".to_owned()))?;
        if end > self.rows {
            return Err(QuantError::Shape(format!(
                "matrix row range {start}..{end} exceeds 0..{}",
                self.rows
            )));
        }
        Ok(end)
    }
}

fn validate_geometry(
    rows: usize,
    cols: usize,
    group_size: usize,
    value_len: usize,
) -> Result<(), QuantError> {
    if group_size == 0 {
        return Err(QuantError::InvalidGroupSize);
    }
    if rows == 0 || cols == 0 {
        return Err(QuantError::Shape(
            "rows and columns must be non-zero".to_owned(),
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| QuantError::Shape("rows * cols overflows usize".to_owned()))?;
    if value_len != expected {
        return Err(QuantError::Shape(format!(
            "expected {expected} values, got {value_len}"
        )));
    }
    Ok(())
}

fn round_ties_even(value: f32) -> i32 {
    // Rust's `round_ties_even` is not available on the minimum toolchain used here.
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor as i32
    } else if fraction > 0.5 {
        floor as i32 + 1
    } else {
        let lower = floor as i32;
        if lower % 2 == 0 {
            lower
        } else {
            lower + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_low_nibble_first_with_zero_tail() {
        let matrix = Int4Matrix::quantize_per_row(1, 3, &[-8.0, 7.0, 0.0]).unwrap();
        assert_eq!(matrix.packed().len(), 2);
        // Scale is 8/7, so -8 maps to -7 (code 1), +7 maps to +6 (code 14).
        assert_eq!(matrix.packed()[0] & 0x0f, 1);
        assert_eq!(matrix.packed()[0] >> 4, 14);
        assert_eq!(matrix.packed()[1] & 0x0f, 8);
        assert_eq!(matrix.packed()[1] >> 4, 8);
    }

    #[test]
    fn grouped_scales_reduce_outlier_damage() {
        let values = [100.0, 0.5, 0.5, 0.5];
        let per_row = Int4Matrix::quantize_per_row(1, 4, &values).unwrap();
        let grouped = Int4Matrix::quantize_grouped(1, 4, 2, &values).unwrap();
        let row_error = (per_row.dequantize_row(0).unwrap()[2] - 0.5).abs();
        let grouped_error = (grouped.dequantize_row(0).unwrap()[2] - 0.5).abs();
        assert!(grouped_error < row_error);
    }

    #[test]
    fn matvec_matches_explicit_dequantization() {
        let matrix =
            Int4Matrix::quantize_grouped(2, 4, 2, &[1.0, -2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0])
                .unwrap();
        let input = [0.25, 2.0, -1.0, 0.5];
        let output = matrix.matvec(&input).unwrap();
        for (row, &actual) in output.iter().enumerate() {
            let expected: f32 = matrix
                .dequantize_row(row)
                .unwrap()
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn packed_row_range_kernels_handle_odd_groups_and_tail_nibbles() {
        let values = (0..21)
            .map(|index| (index as f32 - 10.0) * 0.17)
            .collect::<Vec<_>>();
        let matrix = Int4Matrix::quantize_grouped(3, 7, 3, &values).unwrap();
        let input = [0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4];
        let rows = matrix.matvec_rows(1, 2, &input).unwrap();
        for (offset, actual) in rows.into_iter().enumerate() {
            let expected = matrix
                .dequantize_row(offset + 1)
                .unwrap()
                .iter()
                .zip(input)
                .map(|(&weight, value)| f64::from(weight) * f64::from(value))
                .sum::<f64>() as f32;
            assert_eq!(actual, expected);
        }

        let transposed = matrix.transpose_rows_matvec(1, &[0.25, -0.75]).unwrap();
        for (column, actual) in transposed.into_iter().enumerate() {
            let expected = (1..3)
                .zip([0.25f32, -0.75])
                .map(|(row, coefficient)| {
                    f64::from(matrix.value(row, column)) * f64::from(coefficient)
                })
                .sum::<f64>() as f32;
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn int8_uses_twos_complement_bytes_and_row_scales() {
        let matrix = Int8Matrix::from_packed(
            2,
            3,
            vec![0xff, 0x00, 0x7f, 0x80, 0x02, 0xfe],
            vec![0.5, 0.25],
        )
        .unwrap();
        assert_eq!(matrix.dequantize_row(0).unwrap(), vec![-0.5, 0.0, 63.5]);
        assert_eq!(matrix.dequantize_row(1).unwrap(), vec![-32.0, 0.5, -0.5]);
        let input = [2.0, -1.0, 0.5];
        let output = matrix.matvec(&input).unwrap();
        let explicit: Vec<f32> = (0..2)
            .map(|row| {
                matrix
                    .dequantize_row(row)
                    .unwrap()
                    .iter()
                    .zip(input)
                    .map(|(left, right)| left * right)
                    .sum()
            })
            .collect();
        assert_eq!(output, explicit);
    }
}
