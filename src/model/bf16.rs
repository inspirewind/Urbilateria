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
        if validate_finite && !bf16_payload_is_finite(&little_endian_bytes) {
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
        let parallel = should_parallelize(count, work);
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
        #[cfg(target_arch = "x86_64")]
        let tiled = exact_avx2
            && (!parallel || {
                let workers = crate::execution::worker_threads();
                // Keep enough independent tiles to occupy the pool. Larger pools already
                // saturate LM-head memory reads; their original row schedule stays intact.
                workers <= 4 && count / 8 >= workers
            });
        #[cfg(not(target_arch = "x86_64"))]
        let tiled = false;
        if !tiled {
            if parallel {
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
            return Ok(output);
        }
        let compute_rows = |output: &mut [f32], rows: &[u8]| {
            #[cfg(target_arch = "x86_64")]
            if exact_avx2 && output.len() == 8 {
                // SAFETY: AVX2 is detected, and these are eight complete weight rows.
                unsafe { dot_bf16_rows_f64_ordered_avx2::<2>(rows, input, output) };
                return;
            }
            for (output, row) in output.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                *output = dot(row);
            }
        };
        let rows_per_task = 8;
        if parallel {
            install(|| {
                output
                    .par_chunks_mut(rows_per_task)
                    .zip(rows.par_chunks(row_bytes * rows_per_task))
                    .for_each(|(output, rows)| compute_rows(output, rows));
            });
        } else {
            for (output, rows) in output
                .chunks_mut(rows_per_task)
                .zip(rows.chunks(row_bytes * rows_per_task))
            {
                compute_rows(output, rows);
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
        let compute_rows = |output: &mut [f32], rows: &[u8]| {
            #[cfg(target_arch = "x86_64")]
            if avx2_fma && output.len() == 4 {
                // SAFETY: feature detection and the four complete row slices satisfy the kernel.
                unsafe {
                    dot_bf16_four_rows_fp32_avx2_fma(rows, input, output);
                }
                return;
            }
            for (output, row) in output.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                #[cfg(target_arch = "x86_64")]
                if avx2_fma {
                    *output = unsafe { dot_bf16_fp32_avx2_fma(row, input) };
                    continue;
                }
                *output = dot_bf16(row, input);
            }
        };
        if should_parallelize(self.rows, work) {
            install(|| {
                output
                    .par_chunks_mut(4)
                    .zip(rows.par_chunks(row_bytes * 4))
                    .for_each(|(output, rows)| compute_rows(output, rows));
            });
        } else {
            for (output, rows) in output.chunks_mut(4).zip(rows.chunks(row_bytes * 4)) {
                compute_rows(output, rows);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(output)
    }

    /// Applies consecutive input rows without changing the FP32 GEMV reduction tree.
    /// Two weight rows share each input load, with four inputs sharing each weight load.
    pub(crate) fn matmul_rows_fp32(
        &self,
        input: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, MatrixError> {
        let expected = batch
            .checked_mul(self.cols)
            .ok_or_else(|| MatrixError::InvalidShape("batch * cols overflows usize".into()))?;
        if batch == 0 || input.len() != expected {
            return Err(MatrixError::InputLength {
                expected,
                got: input.len(),
            });
        }
        if batch == 1 {
            return self.matvec_fp32(input);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let output_len = batch
            .checked_mul(self.rows)
            .ok_or_else(|| MatrixError::InvalidShape("batch * rows overflows usize".into()))?;
        let work = output_len.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecBf16, work);
        let mut by_row = vec![0.0f32; output_len];
        #[cfg(target_arch = "x86_64")]
        let avx2_fma = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        let compute = |row: &[u8], values: &mut [f32]| {
            #[cfg(target_arch = "x86_64")]
            if avx2_fma {
                let mut offset = 0;
                // Full groups share weights; small tails keep the same vector reduction.
                for width in [8, 4, 2, 1] {
                    while values.len() - offset >= width {
                        let inputs = &input[offset * self.cols..(offset + width) * self.cols];
                        let outputs = &mut values[offset..offset + width];
                        // SAFETY: features are detected and inputs/outputs contain exactly
                        // the selected number of complete rows. The helper bounds every load.
                        unsafe {
                            match width {
                                8 => dot_bf16_inputs_fp32_avx2_fma::<8>(row, inputs, outputs),
                                4 => dot_bf16_inputs_fp32_avx2_fma::<4>(row, inputs, outputs),
                                2 => dot_bf16_inputs_fp32_avx2_fma::<2>(row, inputs, outputs),
                                1 => dot_bf16_inputs_fp32_avx2_fma::<1>(row, inputs, outputs),
                                _ => unreachable!(),
                            }
                        }
                        offset += width;
                    }
                }
                return;
            }
            for (value, token) in values.iter_mut().zip(input.chunks_exact(self.cols)) {
                *value = dot_bf16(row, token);
            }
        };
        let compute_pair = |rows: &[u8], values: &mut [f32]| {
            #[cfg(target_arch = "x86_64")]
            if avx2_fma && values.len() == batch * 2 {
                let (first, second) = values.split_at_mut(batch);
                let mut offset = 0;
                for width in [4, 2, 1] {
                    while batch - offset >= width {
                        let inputs = &input[offset * self.cols..(offset + width) * self.cols];
                        let a = &mut first[offset..offset + width];
                        let b = &mut second[offset..offset + width];
                        // SAFETY: detected features, two complete weight rows, and N complete
                        // input/output slices satisfy the helper's bounded SIMD accesses.
                        unsafe {
                            match width {
                                4 => {
                                    dot_bf16_two_rows_inputs_fp32_avx2_fma::<4>(rows, inputs, a, b)
                                }
                                2 => {
                                    dot_bf16_two_rows_inputs_fp32_avx2_fma::<2>(rows, inputs, a, b)
                                }
                                1 => {
                                    dot_bf16_two_rows_inputs_fp32_avx2_fma::<1>(rows, inputs, a, b)
                                }
                                _ => unreachable!(),
                            }
                        }
                        offset += width;
                    }
                }
                return;
            }
            for (row, values) in rows
                .chunks_exact(self.cols * 2)
                .zip(values.chunks_mut(batch))
            {
                compute(row, values);
            }
        };
        #[cfg(target_arch = "x86_64")]
        let rows_per_task = if avx2_fma { 2 } else { 1 };
        #[cfg(not(target_arch = "x86_64"))]
        let rows_per_task = 1;
        if should_parallelize(self.rows.div_ceil(rows_per_task), work) {
            install(|| {
                self.little_endian_bytes
                    .par_chunks(self.cols * 2 * rows_per_task)
                    .zip(by_row.par_chunks_mut(batch * rows_per_task))
                    .for_each(|(rows, values)| compute_pair(rows, values));
            });
        } else {
            for (rows, values) in self
                .little_endian_bytes
                .chunks(self.cols * 2 * rows_per_task)
                .zip(by_row.chunks_mut(batch * rows_per_task))
            {
                compute_pair(rows, values);
            }
        }
        if by_row.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let mut output = vec![0.0; output_len];
        for (row, values) in by_row.chunks_exact(batch).enumerate() {
            for (token, &value) in values.iter().enumerate() {
                output[token * self.rows + row] = value;
            }
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

/// Transpose four-column register tiles so F64 lanes hold independent output rows.
/// Each row still adds products in column order, with separate multiply/add operations.
/// Native weights remain row-major; no packed copy or widened matrix is allocated.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_bf16_rows_f64_ordered_avx2<const N: usize>(
    rows: &[u8],
    input: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let stride = input.len() * 2;
    debug_assert_eq!(rows.len(), N * 4 * stride);
    debug_assert_eq!(output.len(), N * 4);
    let complete = input.len() / 4 * 4;
    let mut sums = [_mm256_set1_pd(-0.0); N];
    for offset in (0..complete).step_by(4) {
        let inputs = [
            _mm256_set1_pd(f64::from(input[offset])),
            _mm256_set1_pd(f64::from(input[offset + 1])),
            _mm256_set1_pd(f64::from(input[offset + 2])),
            _mm256_set1_pd(f64::from(input[offset + 3])),
        ];
        for (group, sum) in sums.iter_mut().enumerate() {
            let load_row = |row| {
                let ptr = rows.as_ptr().add((group * 4 + row) * stride + offset * 2);
                let packed = _mm_loadl_epi64(ptr.cast());
                _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(packed), 16))
            };
            let a = load_row(0);
            let b = load_row(1);
            let c = load_row(2);
            let d = load_row(3);
            let ab_low = _mm_unpacklo_ps(a, b);
            let ab_high = _mm_unpackhi_ps(a, b);
            let cd_low = _mm_unpacklo_ps(c, d);
            let cd_high = _mm_unpackhi_ps(c, d);
            let columns = [
                _mm_movelh_ps(ab_low, cd_low),
                _mm_movehl_ps(cd_low, ab_low),
                _mm_movelh_ps(ab_high, cd_high),
                _mm_movehl_ps(cd_high, ab_high),
            ];
            for (weights, values) in columns.into_iter().zip(inputs) {
                *sum = _mm256_add_pd(*sum, _mm256_mul_pd(_mm256_cvtps_pd(weights), values));
            }
        }
    }
    for (group, sum) in sums.into_iter().enumerate() {
        let mut lanes = [0.0; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), sum);
        for (lane, mut sum) in lanes.into_iter().enumerate() {
            let row = group * 4 + lane;
            for (column, &value) in input.iter().enumerate().skip(complete) {
                let byte = row * stride + column * 2;
                sum += f64::from(decode_bf16(&rows[byte..byte + 2])) * f64::from(value);
            }
            output[row] = sum as f32;
        }
    }
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

/// One weight row and N input rows, preserving each input's original eight-lane FMA tree.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_inputs_fp32_avx2_fma<const N: usize>(
    row: &[u8],
    input: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let cols = row.len() / 2;
    debug_assert_eq!(input.len(), N * cols);
    debug_assert_eq!(output.len(), N);
    let mut accumulators = [_mm256_setzero_ps(); N];
    let complete = cols / 8 * 8;
    for column in (0..complete).step_by(8) {
        let packed = _mm_loadu_si128(row.as_ptr().add(column * 2).cast());
        let weights = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        for (token, accumulator) in accumulators.iter_mut().enumerate() {
            let values = _mm256_loadu_ps(input.as_ptr().add(token * cols + column));
            *accumulator = _mm256_fmadd_ps(weights, values, *accumulator);
        }
    }
    for (token, accumulator) in accumulators.into_iter().enumerate() {
        let mut lanes = [0.0; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        let mut sum = lanes.into_iter().sum::<f32>();
        for column in complete..cols {
            sum += decode_bf16(&row[column * 2..column * 2 + 2]) * input[token * cols + column];
        }
        output[token] = sum;
    }
}

/// Two weight rows and N input rows share loads along both matrix dimensions. Each output
/// retains the single-row kernel's eight-lane FMA accumulation and final reduction order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_two_rows_inputs_fp32_avx2_fma<const N: usize>(
    rows: &[u8],
    input: &[f32],
    first: &mut [f32],
    second: &mut [f32],
) {
    use std::arch::x86_64::*;
    let cols = rows.len() / 4;
    debug_assert_eq!(input.len(), N * cols);
    debug_assert_eq!(first.len(), N);
    debug_assert_eq!(second.len(), N);
    let stride = cols * 2;
    let complete = cols / 8 * 8;
    let mut a = [_mm256_setzero_ps(); N];
    let mut b = a;
    for column in (0..complete).step_by(8) {
        let packed_a = _mm_loadu_si128(rows.as_ptr().add(column * 2).cast());
        let packed_b = _mm_loadu_si128(rows.as_ptr().add(stride + column * 2).cast());
        let wa = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed_a), 16));
        let wb = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed_b), 16));
        for token in 0..N {
            let values = _mm256_loadu_ps(input.as_ptr().add(token * cols + column));
            a[token] = _mm256_fmadd_ps(wa, values, a[token]);
            b[token] = _mm256_fmadd_ps(wb, values, b[token]);
        }
    }
    for (row, (accumulators, output)) in [(a, first), (b, second)].into_iter().enumerate() {
        for (token, accumulator) in accumulators.into_iter().enumerate() {
            let mut lanes = [0.0; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
            let mut sum = lanes.into_iter().sum::<f32>();
            for column in complete..cols {
                let byte = row * stride + column * 2;
                sum += decode_bf16(&rows[byte..byte + 2]) * input[token * cols + column];
            }
            output[token] = sum;
        }
    }
}

/// Four independent row accumulators hide FMA latency and reuse the input load,
/// without reassociating any row's products or changing its final lane reduction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_four_rows_fp32_avx2_fma(rows: &[u8], input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    let stride = input.len() * 2;
    debug_assert_eq!(rows.len(), stride * 4);
    debug_assert_eq!(output.len(), 4);
    let mut a = _mm256_setzero_ps();
    let mut b = a;
    let mut c = a;
    let mut d = a;
    let complete = input.len() / 8 * 8;
    for offset in (0..complete).step_by(8) {
        let x = _mm256_loadu_ps(input.as_ptr().add(offset));
        let ptr = rows.as_ptr().add(offset * 2);
        let w0 = _mm_loadu_si128(ptr.cast());
        let w1 = _mm_loadu_si128(ptr.add(stride).cast());
        let w2 = _mm_loadu_si128(ptr.add(stride * 2).cast());
        let w3 = _mm_loadu_si128(ptr.add(stride * 3).cast());
        a = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w0), 16)),
            x,
            a,
        );
        b = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w1), 16)),
            x,
            b,
        );
        c = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w2), 16)),
            x,
            c,
        );
        d = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w3), 16)),
            x,
            d,
        );
    }
    for (row, accumulator) in [a, b, c, d].into_iter().enumerate() {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        let mut sum = lanes.into_iter().sum::<f32>();
        for (column, &value) in input.iter().enumerate().skip(complete) {
            sum += decode_bf16(&rows[row * stride + column * 2..row * stride + column * 2 + 2])
                * value;
        }
        output[row] = sum;
    }
}

/// BF16 non-finite values have an all-ones exponent. Check packed bits directly
/// so loading a resident expert does not scalar-decode every element before GEMV.
fn bf16_payload_is_finite(bytes: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the feature is detected and all loads stay within complete 32-byte chunks.
        return unsafe { bf16_payload_is_finite_avx2(bytes) };
    }
    bytes
        .chunks_exact(2)
        .all(|value| decode_bf16(value).is_finite())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bf16_payload_is_finite_avx2(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi16(0x7f80);
    let mut chunks = bytes.chunks_exact(32);
    for chunk in &mut chunks {
        let bits = _mm256_loadu_si256(chunk.as_ptr().cast());
        let invalid = _mm256_cmpeq_epi16(_mm256_and_si256(bits, mask), mask);
        if _mm256_movemask_epi8(invalid) != 0 {
            return false;
        }
    }
    chunks
        .remainder()
        .chunks_exact(2)
        .all(|value| decode_bf16(value).is_finite())
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
    fn ordered_row_tiles_preserve_scalar_bits_ranges_and_tails() {
        for rows in [1, 3, 7, 8, 9, 15, 16, 17, 129] {
            for cols in [1, 3, 4, 5, 7, 8, 9, 31, 65, 512, 2048] {
                let weights = (0..rows * cols)
                    .map(|i| ((i * 137 % 251) as f32 - 125.0) / 37.0)
                    .collect::<Vec<_>>();
                let input = (0..cols)
                    .map(|i| ((i * 19 % 97) as f32 - 48.0) / 31.0)
                    .collect::<Vec<_>>();
                let bytes = bf16_bytes(&weights);
                let expected = bytes
                    .chunks_exact(cols * 2)
                    .map(|row| dot_bf16(row, &input).to_bits())
                    .collect::<Vec<_>>();
                let matrix = Bf16Matrix::from_le_bytes(rows, cols, bytes).unwrap();
                for (start, count) in [(0, rows), (1, rows - 1), (rows, 0)] {
                    let actual = matrix.matvec_rows(start, count, &input).unwrap();
                    assert_eq!(
                        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected[start..start + count],
                        "rows={rows} cols={cols} start={start}"
                    );
                }
            }
        }
    }

    #[test]
    fn ordered_row_tiles_preserve_cancellation_and_signed_zero() {
        let patterns = [
            [-0.0; 8],
            [0.0; 8],
            [1e30, 1.0, -1e30, 2.0, -0.0, 0.0, 1e-30, -1e-30],
            [
                f32::MAX,
                f32::MIN,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                f32::from_bits(0x0001_0000),
                -f32::from_bits(0x0001_0000),
                0.0,
                -0.0,
            ],
        ];
        for cols in [3, 4, 5, 7, 8, 9, 65, 2048] {
            let weights = (0..16 * cols)
                .map(|i| patterns[(i / cols) % patterns.len()][i % cols % 8])
                .collect::<Vec<_>>();
            let bytes = bf16_bytes(&weights);
            let matrix = Bf16Matrix::from_le_bytes(16, cols, bytes.clone()).unwrap();
            for input in [
                vec![1.0; cols],
                vec![-1.0; cols],
                vec![0.0; cols],
                vec![-0.0; cols],
                vec![f32::from_bits(1); cols],
                vec![f32::MIN_POSITIVE; cols],
                vec![f32::MAX; cols],
            ] {
                let expected = bytes
                    .chunks_exact(cols * 2)
                    .map(|row| dot_bf16(row, &input).to_bits())
                    .collect::<Vec<_>>();
                let actual = matrix.matvec(&input).unwrap();
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }

    #[test]
    fn batched_fp32_matches_individual_gemv_bits_across_tiles_and_tails() {
        for rows in [1, 2, 3, 16, 17, 129] {
            for cols in [1, 7, 8, 9, 65, 512] {
                let weights = (0..rows * cols)
                    .map(|i| ((i * 137 % 251) as f32 - 125.0) / 37.0)
                    .collect::<Vec<_>>();
                let matrix = Bf16Matrix::from_le_bytes(rows, cols, bf16_bytes(&weights)).unwrap();
                for batch in [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 19, 31, 32, 33, 63, 64, 65, 128, 129,
                ] {
                    let input = (0..batch * cols)
                        .map(|i| ((i * 19 % 97) as f32 - 48.0) / 31.0)
                        .collect::<Vec<_>>();
                    let expected = input
                        .chunks_exact(cols)
                        .flat_map(|token| matrix.matvec_fp32(token).unwrap())
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    let actual = matrix.matmul_rows_fp32(&input, batch).unwrap();
                    assert_eq!(
                        actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                        expected,
                        "rows={rows}, cols={cols}, batch={batch}"
                    );
                }
            }
        }
    }

    #[test]
    fn batched_fp32_rejects_invalid_shapes_and_nonfinite_values() {
        let matrix = Bf16Matrix::from_le_bytes(2, 2, bf16_bytes(&[1.0; 4])).unwrap();
        assert!(matrix.matmul_rows_fp32(&[], 0).is_err());
        assert!(matrix.matmul_rows_fp32(&[1.0; 3], 2).is_err());
        assert!(matrix.matmul_rows_fp32(&[], usize::MAX).is_err());
        assert!(matrix
            .matmul_rows_fp32(&[1.0, 2.0, 3.0, f32::NAN], 2)
            .is_err());
        assert!(matrix.matmul_rows_fp32(&[f32::MAX; 4], 2).is_err());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn four_row_fp32_kernel_preserves_single_row_bits_and_tails() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        for columns in [1, 7, 8, 9, 31, 65, 512, 2048] {
            for rows in [1, 3, 4, 5, 9, 17] {
                let values = (0..rows * columns)
                    .map(|i| ((i * 137 % 251) as f32 - 125.0) / 37.0)
                    .collect::<Vec<_>>();
                let input = (0..columns)
                    .map(|i| ((i * 19 % 97) as f32 - 48.0) / 31.0)
                    .collect::<Vec<_>>();
                let bytes = bf16_bytes(&values);
                let expected = bytes
                    .chunks_exact(columns * 2)
                    .map(|row| unsafe { dot_bf16_fp32_avx2_fma(row, &input) })
                    .collect::<Vec<_>>();
                let matrix = Bf16Matrix::from_le_bytes(rows, columns, bytes).unwrap();
                let actual = matrix.matvec_fp32(&input).unwrap();
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                if rows > 4 {
                    let slice = matrix.matvec_rows_fp32(1, rows - 2, &input).unwrap();
                    assert_eq!(slice, expected[1..rows - 1]);
                }
            }
        }
    }

    #[test]
    fn packed_finiteness_matches_scalar_for_every_bf16_encoding_and_tail() {
        for count in [1, 15, 16, 17, 31, 32, 33, 97] {
            let mut bytes = vec![0u8; count * 2];
            for position in 0..count {
                for bits in [0x7f80u16, 0xff80, 0x7fc1, 0xffc1] {
                    bytes[position * 2..position * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                    assert!(!bf16_payload_is_finite(&bytes));
                }
                bytes[position * 2..position * 2 + 2].copy_from_slice(&0x7f7fu16.to_le_bytes());
                assert!(bf16_payload_is_finite(&bytes));
            }
        }
        for bits in 0..=u16::MAX {
            let bytes = bits.to_le_bytes().repeat(32);
            assert_eq!(
                bf16_payload_is_finite(&bytes),
                decode_bf16(&bits.to_le_bytes()).is_finite()
            );
        }
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
