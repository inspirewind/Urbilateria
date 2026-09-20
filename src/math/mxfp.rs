//! Scalar reference containers for block-scaled FP8 and MXFP4 weights.
//!
//! DeepSeek stores FP8 block scales as E8M0 bytes, while Qwen's routed experts store
//! `weight_scale_inv` as BF16. Both decode to the same row-block/column-block multiplication,
//! but their constructors remain distinct so the checkpoint ABIs cannot be confused.

use rayon::prelude::*;
use std::fmt;
use std::sync::OnceLock;

use crate::execution::{install, should_parallelize};
use crate::profiling::{span_with_work, ProfileStage};
use crate::storage::ReadBuffer;

static E4M3_TABLE: OnceLock<[f32; 256]> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MxError {
    Shape(String),
    InvalidScale(u8),
    InvalidBf16Scale(u16),
    NonFinite,
}

impl fmt::Display for MxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(f, "invalid MX matrix shape: {reason}"),
            Self::InvalidScale(value) => write!(f, "E8M0 scale byte 0x{value:02x} is NaN"),
            Self::InvalidBf16Scale(value) => {
                write!(
                    f,
                    "BF16 scale bits 0x{value:04x} do not decode to a finite, strictly positive value"
                )
            }
            Self::NonFinite => f.write_str("MX matrix input contains NaN or infinity"),
        }
    }
}

impl std::error::Error for MxError {}

/// Decodes PyTorch/OCP unsigned E8M0. `0xff` is the sole NaN encoding.
pub fn decode_e8m0(value: u8) -> Result<f32, MxError> {
    if value == u8::MAX {
        return Err(MxError::InvalidScale(value));
    }
    Ok(decode_e8m0_unchecked(value))
}

/// Every finite E8M0 code except zero is already the biased IEEE-754 F32 exponent field for the
/// corresponding power of two. Code zero is 2^-127, represented by the midpoint subnormal bit.
#[inline]
fn decode_e8m0_unchecked(value: u8) -> f32 {
    let bits = u32::from(value) << 23;
    f32::from_bits(if bits == 0 { 0x0040_0000 } else { bits })
}

/// Decodes the finite-only E4M3 format used by `torch.float8_e4m3fn`.
pub fn decode_e4m3fn(value: u8) -> f32 {
    let sign = if value & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    if exponent == 0 {
        sign * f32::from(mantissa) * 2.0f32.powi(-9)
    } else if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    }
}

const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub fn decode_e2m1(value: u8) -> f32 {
    E2M1_TABLE[usize::from(value & 0x0f)]
}

/// Simulates the power-of-two-scaled E4M3 activation quantizer used before every MX GEMM.
pub fn simulate_e4m3_activation(values: &[f32], block_size: usize) -> Result<Vec<f32>, MxError> {
    let mut output = values.to_vec();
    simulate_e4m3_activation_in_place(&mut output, block_size)?;
    Ok(output)
}

/// In-place form of [`simulate_e4m3_activation`] for hot paths that already own an activation
/// buffer. Validation completes before mutation, and every block retains the same left-to-right
/// maximum and quantization order as the allocating form.
pub fn simulate_e4m3_activation_in_place(
    values: &mut [f32],
    block_size: usize,
) -> Result<(), MxError> {
    if values.is_empty() || block_size == 0 || values.iter().any(|value| !value.is_finite()) {
        return Err(if block_size == 0 || values.is_empty() {
            MxError::Shape("activation and block size must be non-zero".to_owned())
        } else {
            MxError::NonFinite
        });
    }
    let table = e4m3_table();
    for block in values.chunks_mut(block_size) {
        let absolute_maximum = block
            .iter()
            .map(|value| value.abs())
            .fold(1.0e-4f32, f32::max);
        let scale = power_of_two_ceiling(absolute_maximum / 448.0);
        for value in block {
            let code = encode_e4m3fn_finite(*value / scale);
            *value = table[usize::from(code)] * scale;
        }
    }
    Ok(())
}

/// Simulates Hugging Face's fine-grained FP8 dynamic activation path.
///
/// Each contiguous K block uses the exact floating scale `max(abs(block)) / 448`, rather than
/// the power-of-two E8M0 scale used by [`simulate_e4m3_activation`]. Qwen's block-FP8 experts
/// declare this per-token-group dynamic scheme with groups of 128 values.
pub fn simulate_finegrained_e4m3_activation(
    values: &[f32],
    block_size: usize,
) -> Result<Vec<f32>, MxError> {
    if values.is_empty() || block_size == 0 || values.iter().any(|value| !value.is_finite()) {
        return Err(if block_size == 0 || values.is_empty() {
            MxError::Shape("activation and block size must be non-zero".to_owned())
        } else {
            MxError::NonFinite
        });
    }
    let mut output = Vec::with_capacity(values.len());
    let table = e4m3_table();
    for block in values.chunks(block_size) {
        let absolute_maximum = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        if absolute_maximum == 0.0 {
            output.resize(output.len() + block.len(), 0.0);
            continue;
        }
        let scale = absolute_maximum / 448.0;
        output.extend(
            block
                .iter()
                .map(|&value| table[usize::from(encode_e4m3fn_finite(value / scale))] * scale),
        );
    }
    Ok(output)
}

/// Simulates the E2M1 activation quantizer used by the sparse-attention indexer.
pub fn simulate_e2m1_activation(values: &[f32], group_size: usize) -> Result<Vec<f32>, MxError> {
    if values.is_empty() || group_size == 0 || values.iter().any(|value| !value.is_finite()) {
        return Err(if group_size == 0 || values.is_empty() {
            MxError::Shape("activation and group size must be non-zero".to_owned())
        } else {
            MxError::NonFinite
        });
    }
    let mut output = Vec::with_capacity(values.len());
    for group in values.chunks(group_size) {
        let absolute_maximum = group
            .iter()
            .map(|value| value.abs())
            .fold(6.0 * 2.0f32.powi(-126), f32::max);
        let scale = power_of_two_ceiling(absolute_maximum / 6.0);
        output.extend(group.iter().map(|&value| {
            let code = nearest_finite_code(value / scale, &E2M1_TABLE, 0x0f);
            decode_e2m1(code) * scale
        }));
    }
    Ok(output)
}

/// In-place normalized Walsh-Hadamard transform used by the indexer QAT path.
pub fn normalized_hadamard(values: &mut [f32]) -> Result<(), MxError> {
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(MxError::Shape(
            "Hadamard width must be a non-zero power of two".to_owned(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MxError::NonFinite);
    }
    let mut half = 1;
    while half < values.len() {
        for start in (0..values.len()).step_by(2 * half) {
            for offset in 0..half {
                let left = values[start + offset];
                let right = values[start + half + offset];
                values[start + offset] = left + right;
                values[start + half + offset] = left - right;
            }
        }
        half *= 2;
    }
    let scale = (values.len() as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
    Ok(())
}

#[cfg(test)]
fn encode_e4m3fn(value: f32) -> Result<u8, MxError> {
    if !value.is_finite() {
        return Err(MxError::NonFinite);
    }
    Ok(encode_e4m3fn_finite(value))
}

/// Encodes one finite value in the OCP finite-only E4M3 format.
///
/// This is exposed for non-matrix caches whose scale itself is stored as E4M3, notably the
/// DeepSeek-V4.1 global KV cache. Matrix activation quantization continues to use its dedicated
/// block routines.
pub fn encode_e4m3fn_scalar(value: f32) -> Result<u8, MxError> {
    if !value.is_finite() {
        return Err(MxError::NonFinite);
    }
    Ok(encode_e4m3fn_finite(value))
}

/// Returns the nearest finite E2M1 code, with deterministic low-code tie breaking.
pub fn encode_e2m1_scalar(value: f32) -> Result<u8, MxError> {
    if !value.is_finite() {
        return Err(MxError::NonFinite);
    }
    Ok(nearest_finite_code(value, &E2M1_TABLE, 0x0f))
}

/// Rounds a positive finite scale upward to E8M0, matching `fast_round_scale` in the native
/// DeepSeek kernels. Values outside the finite E8M0 range are rejected.
pub fn encode_e8m0_scalar(value: f32) -> Result<u8, MxError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(MxError::NonFinite);
    }
    let rounded = power_of_two_ceiling(value);
    if !rounded.is_finite() || rounded < 2.0f32.powi(-127) || rounded > 2.0f32.powi(127) {
        return Err(MxError::NonFinite);
    }
    Ok((rounded.to_bits() >> 23) as u8)
}

/// Encodes a finite value after a containing activation slice has been validated once.
#[inline]
fn encode_e4m3fn_finite(value: f32) -> u8 {
    debug_assert!(value.is_finite());
    let value = value.clamp(-448.0, 448.0);
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return 0;
    }

    // E4M3 subnormals form an exact 2^-9 grid. Normal values retain three F32 mantissa bits;
    // adding half-minus-one plus the retained LSB implements round-to-nearest, ties-to-even and
    // naturally carries into the next exponent. Codes above finite 0x7e saturate to 448.
    let code = if magnitude < 2.0f32.powi(-6) {
        (magnitude * 512.0).round_ties_even() as u8
    } else {
        const DISCARD_BITS: u32 = 20;
        const ROUND_BIAS: u32 = (1 << (DISCARD_BITS - 1)) - 1;
        let bits = magnitude.to_bits();
        let retained_lsb = (bits >> DISCARD_BITS) & 1;
        let rounded = bits.wrapping_add(ROUND_BIAS + retained_lsb) & !((1 << DISCARD_BITS) - 1);
        let exponent = ((rounded >> 23) & 0xff).saturating_sub(120);
        let mantissa = (rounded >> DISCARD_BITS) & 0x07;
        ((exponent << 3) | mantissa).min(0x7e) as u8
    };
    if value.is_sign_negative() && code != 0 {
        code | 0x80
    } else {
        code
    }
}

fn e4m3_table() -> &'static [f32; 256] {
    E4M3_TABLE.get_or_init(|| std::array::from_fn(|code| decode_e4m3fn(code as u8)))
}

fn nearest_finite_code(value: f32, table: &[f32], maximum_code: u8) -> u8 {
    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0..=maximum_code {
        let decoded = table[usize::from(code)];
        if !decoded.is_finite() {
            continue;
        }
        let distance = (decoded - value).abs();
        let best_even = best & 1 == 0;
        let code_even = code & 1 == 0;
        if distance < best_distance || (distance == best_distance && code_even && !best_even) {
            best = code;
            best_distance = distance;
        }
    }
    best
}

fn power_of_two_ceiling(value: f32) -> f32 {
    debug_assert!(value.is_finite() && value > 0.0);
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if mantissa == 0 {
        return value;
    }
    // The previous `log2().ceil().powf()` formula rounds its F32 logarithm to an integer for a
    // narrow band on either side of exact powers, especially at extreme exponents. Preserve that
    // established behavior there; ordinary interior mantissas need only increment the exponent.
    const BOUNDARY_MANTISSAS: u32 = 4_096;
    if exponent == 0
        || mantissa <= BOUNDARY_MANTISSAS
        || mantissa >= 0x007f_ffff - BOUNDARY_MANTISSAS
    {
        return 2.0f32.powf(value.log2().ceil());
    }
    f32::from_bits(exponent.saturating_add(0x0080_0000))
}

#[derive(Debug, Clone)]
pub struct MxFp8Matrix {
    rows: usize,
    cols: usize,
    block_rows: usize,
    block_cols: usize,
    values: ReadBuffer,
    scales: MxFp8Scales,
}

#[derive(Debug, Clone)]
enum MxFp8Scales {
    /// Native E8M0 bytes stay compact; the hot kernel expands their IEEE exponent bits directly.
    E8M0(ReadBuffer),
    /// Qwen's BF16 `weight_scale_inv` needs an exact decoded F32 representation.
    Decoded(Vec<f32>),
}

impl MxFp8Matrix {
    pub fn from_packed(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        values: Vec<u8>,
        scale_bytes: Vec<u8>,
    ) -> Result<Self, MxError> {
        Self::from_packed_with_validation(
            rows,
            cols,
            block_rows,
            block_cols,
            values.into(),
            scale_bytes.into(),
            true,
        )
    }

    /// Reconstructs a matrix whose bytes were validated earlier from the same immutable
    /// checkpoint ranges. Shape checks remain mandatory; only the two linear payload scans are
    /// elided when an expert is reloaded after eviction within one runtime state.
    #[cfg(test)]
    pub(crate) fn from_packed_prevalidated(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        values: Vec<u8>,
        scale_bytes: Vec<u8>,
    ) -> Result<Self, MxError> {
        Self::from_packed_with_validation(
            rows,
            cols,
            block_rows,
            block_cols,
            values.into(),
            scale_bytes.into(),
            false,
        )
    }

    pub(crate) fn from_read_buffers(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        values: ReadBuffer,
        scale_bytes: ReadBuffer,
    ) -> Result<Self, MxError> {
        Self::from_packed_with_validation(
            rows,
            cols,
            block_rows,
            block_cols,
            values,
            scale_bytes,
            true,
        )
    }

    pub(crate) fn from_read_buffers_prevalidated(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        values: ReadBuffer,
        scale_bytes: ReadBuffer,
    ) -> Result<Self, MxError> {
        Self::from_packed_with_validation(
            rows,
            cols,
            block_rows,
            block_cols,
            values,
            scale_bytes,
            false,
        )
    }

    fn from_packed_with_validation(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        values: ReadBuffer,
        scale_bytes: ReadBuffer,
        validate_payload: bool,
    ) -> Result<Self, MxError> {
        if rows == 0 || cols == 0 || block_rows == 0 || block_cols == 0 {
            return Err(MxError::Shape(
                "dimensions and block dimensions must be non-zero".to_owned(),
            ));
        }
        let expected_values = rows
            .checked_mul(cols)
            .ok_or_else(|| MxError::Shape("value count overflows".to_owned()))?;
        let expected_scales = rows
            .div_ceil(block_rows)
            .checked_mul(cols.div_ceil(block_cols))
            .ok_or_else(|| MxError::Shape("scale count overflows".to_owned()))?;
        if values.len() != expected_values || scale_bytes.len() != expected_scales {
            return Err(MxError::Shape(format!(
                "expected {expected_values} values and {expected_scales} scales, got {} and {}",
                values.len(),
                scale_bytes.len()
            )));
        }
        if validate_payload && contains_invalid_e4m3(&values) {
            return Err(MxError::NonFinite);
        }
        if validate_payload && contains_invalid_e8m0(&scale_bytes) {
            return Err(MxError::InvalidScale(u8::MAX));
        }
        Ok(Self {
            rows,
            cols,
            block_rows,
            block_cols,
            values,
            scales: MxFp8Scales::E8M0(scale_bytes),
        })
    }

    /// Builds a block-scaled E4M3 matrix from Qwen's BF16 `weight_scale_inv` payload.
    ///
    /// The decoded scale at `[row / block_rows, column / block_cols]` multiplies the E4M3
    /// weight code. Keeping this constructor separate from [`Self::from_packed`] prevents a
    /// BF16 scale payload from being interpreted as DeepSeek's one-byte E8M0 scale format.
    pub fn from_packed_bf16_scale_inv(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        values: Vec<u8>,
        scale_bytes: Vec<u8>,
    ) -> Result<Self, MxError> {
        if rows == 0 || cols == 0 || block_rows == 0 || block_cols == 0 {
            return Err(MxError::Shape(
                "dimensions and block dimensions must be non-zero".to_owned(),
            ));
        }
        let expected_values = rows
            .checked_mul(cols)
            .ok_or_else(|| MxError::Shape("value count overflows".to_owned()))?;
        let expected_scales = rows
            .div_ceil(block_rows)
            .checked_mul(cols.div_ceil(block_cols))
            .ok_or_else(|| MxError::Shape("scale count overflows".to_owned()))?;
        let expected_scale_bytes = expected_scales
            .checked_mul(2)
            .ok_or_else(|| MxError::Shape("BF16 scale byte count overflows".to_owned()))?;
        if values.len() != expected_values || scale_bytes.len() != expected_scale_bytes {
            return Err(MxError::Shape(format!(
                "expected {expected_values} values and {expected_scale_bytes} BF16 scale bytes, got {} and {}",
                values.len(),
                scale_bytes.len()
            )));
        }
        if values.iter().any(|code| code & 0x7f == 0x7f) {
            return Err(MxError::NonFinite);
        }
        let scales = scale_bytes
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                let value = f32::from_bits(u32::from(bits) << 16);
                if value.is_finite() && value > 0.0 {
                    Ok(value)
                } else {
                    Err(MxError::InvalidBf16Scale(bits))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rows,
            cols,
            block_rows,
            block_cols,
            values: values.into(),
            scales: MxFp8Scales::Decoded(scales),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn resident_bytes(&self) -> usize {
        let scale_bytes = match &self.scales {
            MxFp8Scales::E8M0(scales) => scales.len(),
            MxFp8Scales::Decoded(scales) => scales.len().saturating_mul(4),
        };
        self.values.len().saturating_add(scale_bytes)
    }

    /// Returns native E8M0 storage for reuse after an MXFP8 matrix leaves a streamed layer or
    /// expert cache.
    pub(crate) fn into_e8m0_buffers(self) -> Option<(ReadBuffer, ReadBuffer)> {
        match self.scales {
            MxFp8Scales::E8M0(scales) => Some((self.values, scales)),
            MxFp8Scales::Decoded(_) => None,
        }
    }

    #[inline]
    fn scale(&self, index: usize) -> f32 {
        match &self.scales {
            MxFp8Scales::E8M0(scales) => decode_e8m0_unchecked(scales[index]),
            MxFp8Scales::Decoded(scales) => scales[index],
        }
    }

    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, MxError> {
        self.matvec_rows(0, self.rows, input)
    }

    /// Applies the same native MXFP8 matrix to consecutive input rows `[batch, cols]`.
    ///
    /// For two through twelve inputs, the AVX2 paths decode each weight block once and update
    /// independent accumulators. ModelOpt 1x32 matrices retain their FP32 FMA/lane reduction;
    /// larger block layouts retain the exact ordered-F64 product and addition sequence used by
    /// [`Self::matvec`].
    pub fn matmul_rows(&self, input: &[f32], batch: usize) -> Result<Vec<f32>, MxError> {
        self.matmul_row_range(0, self.rows, input, batch)
    }

    /// Batched counterpart of [`Self::matvec_rows`], restricted to one contiguous output range.
    pub fn matmul_row_range(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, MxError> {
        let expected = batch
            .checked_mul(self.cols)
            .ok_or_else(|| MxError::Shape("batch * columns overflows usize".to_owned()))?;
        if batch == 0 || input.len() != expected {
            return Err(MxError::Shape(format!(
                "batched matvec expects a non-zero batch and {expected} inputs, got {}",
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        let end = start
            .checked_add(count)
            .filter(|&end| end <= self.rows)
            .ok_or_else(|| {
                MxError::Shape("batched matvec row range is out of bounds".to_owned())
            })?;
        #[cfg(target_arch = "x86_64")]
        let compact_avx2_fma = (2..=12).contains(&batch)
            && self.block_rows == 1
            && self.block_cols == 32
            && self.cols % 32 == 0
            && matches!(self.scales, MxFp8Scales::E8M0(_))
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let compact_avx2_fma = false;
        #[cfg(target_arch = "x86_64")]
        let ordered_avx2 = (2..=12).contains(&batch)
            && self.block_cols >= 8
            && matches!(self.scales, MxFp8Scales::E8M0(_))
            && std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let ordered_avx2 = false;

        if !compact_avx2_fma && !ordered_avx2 {
            let mut output = Vec::with_capacity(batch.saturating_mul(count));
            for input in input.chunks_exact(self.cols) {
                output.extend(self.matvec_rows(start, count, input)?);
            }
            return Ok(output);
        }

        let work = batch.saturating_mul(count).saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecMxFp8, work);
        let scale_cols = self.cols.div_ceil(self.block_cols);
        let MxFp8Scales::E8M0(scales) = &self.scales else {
            unreachable!("batched AVX2 is selected only for E8M0 scales")
        };
        let mut output_by_row = vec![0.0f32; count.saturating_mul(batch)];
        let compute_row = |row: usize, output: &mut [f32]| {
            let codes = &self.values[row * self.cols..(row + 1) * self.cols];
            let scale_row = row / self.block_rows;
            let row_scales = &scales[scale_row * scale_cols..(scale_row + 1) * scale_cols];
            macro_rules! compute_batch {
                ($batch:literal) => {
                    if compact_avx2_fma {
                        dot_e4m3_e8m0_batch_aligned_avx2_fma::<$batch>(
                            codes, row_scales, input, output,
                        )
                    } else {
                        dot_e4m3_e8m0_batch_f64_ordered_avx2::<$batch>(
                            codes,
                            row_scales,
                            input,
                            self.cols,
                            self.block_cols,
                            output,
                        )
                    }
                };
            }
            // SAFETY: dispatch proves AVX2 support, batch is in 2..=12, and validated matrix/input
            // slices cover every code, scale block, and input row. The compact branch additionally
            // proves FMA support and aligned 1x32 geometry.
            unsafe {
                match batch {
                    2 => compute_batch!(2),
                    3 => compute_batch!(3),
                    4 => compute_batch!(4),
                    5 => compute_batch!(5),
                    6 => compute_batch!(6),
                    7 => compute_batch!(7),
                    8 => compute_batch!(8),
                    9 => compute_batch!(9),
                    10 => compute_batch!(10),
                    11 => compute_batch!(11),
                    12 => compute_batch!(12),
                    _ => unreachable!("MXFP8 batch dispatch is limited to two through twelve"),
                }
            }
        };
        if should_parallelize(self.rows, work) {
            install(|| {
                output_by_row
                    .par_chunks_mut(batch)
                    .enumerate()
                    .for_each(|(offset, output)| compute_row(start + offset, output));
            });
        } else {
            for (row, output) in (start..end).zip(output_by_row.chunks_mut(batch)) {
                compute_row(row, output);
            }
        }
        let mut output = vec![0.0f32; batch.saturating_mul(count)];
        for (row, values) in output_by_row.chunks_exact(batch).enumerate() {
            for (token, &value) in values.iter().enumerate() {
                output[token * count + row] = value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        Ok(output)
    }

    pub fn matvec_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, MxError> {
        if input.len() != self.cols {
            return Err(MxError::Shape(format!(
                "matvec expects {} inputs, got {}",
                self.cols,
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        let end = start
            .checked_add(count)
            .filter(|&end| end <= self.rows)
            .ok_or_else(|| MxError::Shape("matvec row range is out of bounds".to_owned()))?;
        let scale_cols = self.cols.div_ceil(self.block_cols);
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecMxFp8, work);
        let mut output = vec![0.0; count];
        // Resolve OnceLock once per operation, rather than checking its atomic state for every
        // scalar weight in block layouts that do not use the ModelOpt AVX2 kernel.
        let table = e4m3_table();
        #[cfg(target_arch = "x86_64")]
        let compact_avx2_fma = self.block_rows == 1
            && self.block_cols == 32
            && matches!(self.scales, MxFp8Scales::E8M0(_))
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        #[cfg(target_arch = "x86_64")]
        let ordered_avx2 = self.block_cols >= 8
            && matches!(self.scales, MxFp8Scales::E8M0(_))
            && std::arch::is_x86_feature_detected!("avx2");
        let dot_row = |row: usize| {
            let scale_row = row / self.block_rows;
            let row_values = &self.values[row * self.cols..(row + 1) * self.cols];
            #[cfg(target_arch = "x86_64")]
            if compact_avx2_fma {
                let MxFp8Scales::E8M0(scales) = &self.scales else {
                    unreachable!("compact AVX2 is selected only for E8M0 scales")
                };
                let row_scales = &scales[scale_row * scale_cols..(scale_row + 1) * scale_cols];
                // SAFETY: feature detection above proves AVX2/FMA support. The helper bounds every
                // vector load to complete 8-value chunks and handles any final tail scalarly.
                return unsafe { dot_e4m3_e8m0_avx2_fma(row_values, row_scales, input) };
            }
            #[cfg(target_arch = "x86_64")]
            if ordered_avx2 {
                let MxFp8Scales::E8M0(scales) = &self.scales else {
                    unreachable!("ordered AVX2 is selected only for E8M0 scales")
                };
                let row_scales = &scales[scale_row * scale_cols..(scale_row + 1) * scale_cols];
                // SAFETY: AVX2 is detected and validated row/scale slices cover every column.
                return unsafe {
                    dot_e4m3_e8m0_f64_ordered_avx2(row_values, row_scales, input, self.block_cols)
                };
            }
            let mut sum = 0.0f64;
            let mut column = 0;
            for (block, codes) in row_values.chunks(self.block_cols).enumerate() {
                let scale = self.scale(scale_row * scale_cols + block);
                let input_block = &input[column..column + codes.len()];
                for (&code, &input_value) in codes.iter().zip(input_block) {
                    let weight = table[usize::from(code)] * scale;
                    sum += f64::from(weight) * f64::from(input_value);
                }
                column += codes.len();
            }
            sum as f32
        };
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(offset, output)| *output = dot_row(start + offset));
            });
        } else {
            for (row, output) in (start..end).zip(&mut output) {
                *output = dot_row(row);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        Ok(output)
    }

    pub fn transpose_rows_matvec(&self, start: usize, input: &[f32]) -> Result<Vec<f32>, MxError> {
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        let end = start
            .checked_add(input.len())
            .filter(|&end| end <= self.rows)
            .ok_or_else(|| {
                MxError::Shape("transpose matvec row range is out of bounds".to_owned())
            })?;
        let scale_cols = self.cols.div_ceil(self.block_cols);
        let work = input.len().saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::TransposeMatvecMxFp8, work);
        #[cfg(target_arch = "x86_64")]
        if self.block_rows == 1
            && self.block_cols == 32
            && matches!(self.scales, MxFp8Scales::E8M0(_))
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && rayon::current_thread_index().is_some()
        {
            let MxFp8Scales::E8M0(scales) = &self.scales else {
                unreachable!("compact transpose AVX2 is selected only for E8M0 scales")
            };
            // SAFETY: feature detection proves AVX2/FMA support. Complete eight-column groups are
            // bounds checked by slices and any final columns are handled scalarly.
            let output = unsafe {
                transpose_rows_e4m3_e8m0_avx2(
                    &self.values,
                    scales,
                    self.cols,
                    scale_cols,
                    start,
                    input,
                )
            };
            if output.iter().any(|value| !value.is_finite()) {
                return Err(MxError::NonFinite);
            }
            return Ok(output);
        }
        let mut output = vec![0.0f64; self.cols];
        let table = e4m3_table();
        let compute_column = |column: usize| {
            let mut sum = 0.0f64;
            for (row, &coefficient) in (start..end).zip(input) {
                let scale_row = row / self.block_rows;
                let weight = table[usize::from(self.values[row * self.cols + column])]
                    * self.scale(scale_row * scale_cols + column / self.block_cols);
                sum += f64::from(coefficient) * f64::from(weight);
            }
            sum
        };
        if should_parallelize(self.cols, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(column, output)| *output = compute_column(column));
            });
        } else {
            for (column, output) in output.iter_mut().enumerate() {
                *output = compute_column(column);
            }
        }
        let output = output
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        Ok(output)
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>, MxError> {
        if row >= self.rows {
            return Err(MxError::Shape(format!(
                "row {row} is outside 0..{}",
                self.rows
            )));
        }
        let scale_cols = self.cols.div_ceil(self.block_cols);
        let scale_row = row / self.block_rows;
        let table = e4m3_table();
        Ok((0..self.cols)
            .map(|col| {
                table[usize::from(self.values[row * self.cols + col])]
                    * self.scale(scale_row * scale_cols + col / self.block_cols)
            })
            .collect())
    }
}

#[inline]
fn contains_invalid_e4m3(values: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection proves AVX2 support; the helper bounds vector loads to full
        // 32-byte chunks and checks the tail scalarly.
        return unsafe { contains_invalid_e4m3_avx2(values) };
    }
    values.iter().any(|code| code & 0x7f == 0x7f)
}

#[inline]
fn contains_invalid_e8m0(values: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection proves AVX2 support; the helper bounds vector loads to full
        // 32-byte chunks and checks the tail scalarly.
        return unsafe { contains_invalid_e8m0_avx2(values) };
    }
    values.contains(&u8::MAX)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn contains_invalid_e4m3_avx2(values: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let mask = _mm256_set1_epi8(0x7f);
    let mut offset = 0usize;
    while offset + 128 <= values.len() {
        let mut invalid = _mm256_setzero_si256();
        for lane in 0..4 {
            let value = _mm256_loadu_si256(values.as_ptr().add(offset + lane * 32).cast());
            invalid = _mm256_or_si256(
                invalid,
                _mm256_cmpeq_epi8(_mm256_and_si256(value, mask), mask),
            );
        }
        if _mm256_movemask_epi8(invalid) != 0 {
            return true;
        }
        offset += 128;
    }
    while offset + 32 <= values.len() {
        let value = _mm256_loadu_si256(values.as_ptr().add(offset).cast());
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(value, mask), mask)) != 0 {
            return true;
        }
        offset += 32;
    }
    values[offset..].iter().any(|code| code & 0x7f == 0x7f)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn contains_invalid_e8m0_avx2(values: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let invalid_value = _mm256_set1_epi8(-1);
    let mut offset = 0usize;
    while offset + 128 <= values.len() {
        let mut invalid = _mm256_setzero_si256();
        for lane in 0..4 {
            let value = _mm256_loadu_si256(values.as_ptr().add(offset + lane * 32).cast());
            invalid = _mm256_or_si256(invalid, _mm256_cmpeq_epi8(value, invalid_value));
        }
        if _mm256_movemask_epi8(invalid) != 0 {
            return true;
        }
        offset += 128;
    }
    while offset + 32 <= values.len() {
        let value = _mm256_loadu_si256(values.as_ptr().add(offset).cast());
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(value, invalid_value)) != 0 {
            return true;
        }
        offset += 32;
    }
    values[offset..].contains(&u8::MAX)
}

/// Block-layout E4M3/E8M0 counterpart of the scalar ordered-F64 reference. Weight decoding and
/// widening multiplications are vectorized, while the final additions retain column order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_e4m3_e8m0_f64_ordered_avx2(
    codes: &[u8],
    scales: &[u8],
    input: &[f32],
    block_cols: usize,
) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(codes.len(), input.len());
    debug_assert_eq!(scales.len(), codes.len().div_ceil(block_cols));
    let mut sum = 0.0f64;
    let mut products = [0.0f64; 8];
    let mut column = 0usize;
    for (block, block_codes) in codes.chunks(block_cols).enumerate() {
        let scale = _mm256_set1_ps(decode_e8m0_unchecked(scales[block]));
        let block_input = &input[column..column + block_codes.len()];
        let complete = block_codes.len() / 8 * 8;
        for offset in (0..complete).step_by(8) {
            let packed = _mm_loadl_epi64(block_codes.as_ptr().add(offset).cast::<__m128i>());
            let indices = _mm256_cvtepu8_epi32(packed);
            let weights = _mm256_mul_ps(decode_e4m3fn_avx2(indices), scale);
            let inputs = _mm256_loadu_ps(block_input.as_ptr().add(offset));
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
        let scale = decode_e8m0_unchecked(scales[block]);
        for offset in complete..block_codes.len() {
            let weight = e4m3_table()[usize::from(block_codes[offset])] * scale;
            sum += f64::from(weight) * f64::from(block_input[offset]);
        }
        column += block_codes.len();
    }
    sum as f32
}

/// Batched counterpart of [`dot_e4m3_e8m0_f64_ordered_avx2`]. Each E4M3 weight block is decoded
/// once, while every token keeps an independent accumulator with the original ordered-F64
/// multiplication and addition boundaries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_e4m3_e8m0_batch_f64_ordered_avx2<const BATCH: usize>(
    codes: &[u8],
    scales: &[u8],
    input: &[f32],
    columns: usize,
    block_cols: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert!((2..=12).contains(&BATCH));
    debug_assert_eq!(codes.len(), columns);
    debug_assert_eq!(scales.len(), columns.div_ceil(block_cols));
    debug_assert_eq!(input.len(), BATCH * columns);
    debug_assert_eq!(output.len(), BATCH);
    let mut sums = [0.0f64; BATCH];
    let mut products = [0.0f64; 8];
    let mut column = 0usize;
    for (block, block_codes) in codes.chunks(block_cols).enumerate() {
        let scale = _mm256_set1_ps(decode_e8m0_unchecked(scales[block]));
        let complete = block_codes.len() / 8 * 8;
        for offset in (0..complete).step_by(8) {
            let packed = _mm_loadl_epi64(block_codes.as_ptr().add(offset).cast::<__m128i>());
            let indices = _mm256_cvtepu8_epi32(packed);
            let weights = _mm256_mul_ps(decode_e4m3fn_avx2(indices), scale);
            let weight_low = _mm256_cvtps_pd(_mm256_castps256_ps128(weights));
            let weight_high = _mm256_cvtps_pd(_mm256_extractf128_ps(weights, 1));
            for (token, sum) in sums.iter_mut().enumerate() {
                let inputs = _mm256_loadu_ps(input.as_ptr().add(token * columns + column + offset));
                let input_low = _mm256_cvtps_pd(_mm256_castps256_ps128(inputs));
                let input_high = _mm256_cvtps_pd(_mm256_extractf128_ps(inputs, 1));
                _mm256_storeu_pd(products.as_mut_ptr(), _mm256_mul_pd(weight_low, input_low));
                _mm256_storeu_pd(
                    products.as_mut_ptr().add(4),
                    _mm256_mul_pd(weight_high, input_high),
                );
                for product in products {
                    *sum += product;
                }
            }
        }
        let scale = decode_e8m0_unchecked(scales[block]);
        for offset in complete..block_codes.len() {
            let weight = e4m3_table()[usize::from(block_codes[offset])] * scale;
            for (token, sum) in sums.iter_mut().enumerate() {
                *sum += f64::from(weight) * f64::from(input[token * columns + column + offset]);
            }
        }
        column += block_codes.len();
    }
    for (output, sum) in output.iter_mut().zip(sums) {
        *output = sum as f32;
    }
}

/// AVX2/FMA implementation for ModelOpt's native 1x32 E4M3/E8M0 layout.
///
/// ModelOpt GEMMs accumulate into FP32 outputs. Eight lanes reduce independently and are combined
/// once at the end, matching the checkpoint's numerical boundary while avoiding the F64 scalar
/// reference overhead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_e4m3_e8m0_avx2_fma(codes: &[u8], scales: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(codes.len(), input.len());
    debug_assert_eq!(scales.len(), codes.len().div_ceil(32));
    if codes.len() % 32 == 0 {
        return dot_e4m3_e8m0_aligned_avx2_fma(codes, scales, input);
    }
    let mut accumulator = _mm256_setzero_ps();
    let mut scalar_tail = 0.0f32;
    let mut column = 0usize;
    for (block, block_codes) in codes.chunks(32).enumerate() {
        let scale = decode_e8m0_unchecked(scales[block]);
        let scale_vector = _mm256_set1_ps(scale);
        let complete = block_codes.len() / 8 * 8;
        let block_input = &input[column..column + block_codes.len()];
        let all_normal = e4m3_block_all_normal_avx2(block_codes);
        for offset in (0..complete).step_by(8) {
            let packed = _mm_loadl_epi64(block_codes.as_ptr().add(offset).cast::<__m128i>());
            let indices = _mm256_cvtepu8_epi32(packed);
            let decoded = if all_normal {
                decode_normal_e4m3fn_avx2(indices)
            } else {
                decode_e4m3fn_avx2(indices)
            };
            let weights = _mm256_mul_ps(decoded, scale_vector);
            let inputs = _mm256_loadu_ps(block_input.as_ptr().add(offset));
            accumulator = _mm256_fmadd_ps(weights, inputs, accumulator);
        }
        for offset in complete..block_codes.len() {
            let weight = e4m3_table()[usize::from(block_codes[offset])] * scale;
            scalar_tail += weight * block_input[offset];
        }
        column += block_codes.len();
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
    lanes.into_iter().sum::<f32>() + scalar_tail
}

/// Tail-free specialization for the native matrix geometries used by release checkpoints.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_e4m3_e8m0_aligned_avx2_fma(codes: &[u8], scales: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(codes.len(), input.len());
    debug_assert_eq!(codes.len() % 32, 0);
    debug_assert_eq!(scales.len(), codes.len() / 32);
    let mut accumulator = _mm256_setzero_ps();
    for (block, &scale_code) in scales.iter().enumerate() {
        let code_ptr = codes.as_ptr().add(block * 32);
        let input_ptr = input.as_ptr().add(block * 32);
        let packed_block = _mm256_loadu_si256(code_ptr.cast());
        let low_codes = _mm256_castsi256_si128(packed_block);
        let high_codes = _mm256_extracti128_si256(packed_block, 1);
        let scale = _mm256_set1_ps(decode_e8m0_unchecked(scale_code));

        macro_rules! accumulate_group {
            ($packed:expr, $offset:expr, $decoder:ident) => {{
                let indices = _mm256_cvtepu8_epi32($packed);
                let weights = _mm256_mul_ps($decoder(indices), scale);
                let inputs = _mm256_loadu_ps(input_ptr.add($offset));
                accumulator = _mm256_fmadd_ps(weights, inputs, accumulator);
            }};
        }

        if packed_e4m3_block_all_normal_avx2(packed_block) {
            accumulate_group!(low_codes, 0, decode_normal_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(low_codes, 8), 8, decode_normal_e4m3fn_avx2);
            accumulate_group!(high_codes, 16, decode_normal_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(high_codes, 8), 24, decode_normal_e4m3fn_avx2);
        } else {
            accumulate_group!(low_codes, 0, decode_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(low_codes, 8), 8, decode_e4m3fn_avx2);
            accumulate_group!(high_codes, 16, decode_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(high_codes, 8), 24, decode_e4m3fn_avx2);
        }
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
    lanes.into_iter().sum::<f32>()
}

/// Tail-free multi-input counterpart of [`dot_e4m3_e8m0_aligned_avx2_fma`]. The weight decode
/// and scale multiplication are shared while each input retains an independent eight-lane FMA
/// chain. `BATCH` is instantiated only for 2 through 12 by the checked public dispatcher.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_e4m3_e8m0_batch_aligned_avx2_fma<const BATCH: usize>(
    codes: &[u8],
    scales: &[u8],
    input: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert!((2..=12).contains(&BATCH));
    debug_assert_eq!(codes.len() % 32, 0);
    debug_assert_eq!(scales.len(), codes.len() / 32);
    debug_assert_eq!(input.len(), BATCH * codes.len());
    debug_assert_eq!(output.len(), BATCH);
    let columns = codes.len();
    let mut accumulators = [_mm256_setzero_ps(); BATCH];
    for (block, &scale_code) in scales.iter().enumerate() {
        let code_ptr = codes.as_ptr().add(block * 32);
        let packed_block = _mm256_loadu_si256(code_ptr.cast());
        let low_codes = _mm256_castsi256_si128(packed_block);
        let high_codes = _mm256_extracti128_si256(packed_block, 1);
        let scale = _mm256_set1_ps(decode_e8m0_unchecked(scale_code));

        macro_rules! accumulate_group {
            ($packed:expr, $offset:expr, $decoder:ident) => {{
                let indices = _mm256_cvtepu8_epi32($packed);
                let weights = _mm256_mul_ps($decoder(indices), scale);
                for token in 0..BATCH {
                    let inputs =
                        _mm256_loadu_ps(input.as_ptr().add(token * columns + block * 32 + $offset));
                    accumulators[token] = _mm256_fmadd_ps(weights, inputs, accumulators[token]);
                }
            }};
        }

        if packed_e4m3_block_all_normal_avx2(packed_block) {
            accumulate_group!(low_codes, 0, decode_normal_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(low_codes, 8), 8, decode_normal_e4m3fn_avx2);
            accumulate_group!(high_codes, 16, decode_normal_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(high_codes, 8), 24, decode_normal_e4m3fn_avx2);
        } else {
            accumulate_group!(low_codes, 0, decode_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(low_codes, 8), 8, decode_e4m3fn_avx2);
            accumulate_group!(high_codes, 16, decode_e4m3fn_avx2);
            accumulate_group!(_mm_srli_si128(high_codes, 8), 24, decode_e4m3fn_avx2);
        }
    }
    for token in 0..BATCH {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulators[token]);
        output[token] = lanes.into_iter().sum::<f32>();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn e4m3_block_all_normal_avx2(codes: &[u8]) -> bool {
    use std::arch::x86_64::*;

    debug_assert!(codes.len() <= 32);
    if codes.len() == 32 {
        let packed = _mm256_loadu_si256(codes.as_ptr().cast());
        return packed_e4m3_block_all_normal_avx2(packed);
    }
    !codes.iter().any(|code| code & 0x7f < 8)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn packed_e4m3_block_all_normal_avx2(packed: std::arch::x86_64::__m256i) -> bool {
    use std::arch::x86_64::*;

    let magnitude = _mm256_and_si256(packed, _mm256_set1_epi8(0x7f));
    let exponent_zero = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), magnitude);
    _mm256_movemask_epi8(exponent_zero) == 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_normal_e4m3fn_avx2(
    codes: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let magnitude = _mm256_and_si256(codes, _mm256_set1_epi32(0x7f));
    let sign = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(0x80)), 24);

    // Shifting the complete seven-bit magnitude by 20 puts its exponent directly in F32 bits
    // 23..26 and its mantissa in bits 20..22. Adding the 120-point bias delta therefore replaces
    // separate exponent/mantissa extraction and shifts while preserving every normal value.
    let normal_bits = _mm256_or_si256(
        sign,
        _mm256_add_epi32(
            _mm256_slli_epi32(magnitude, 20),
            _mm256_set1_epi32(120 << 23),
        ),
    );
    _mm256_castsi256_ps(normal_bits)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_e4m3fn_avx2(codes: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let magnitude = _mm256_and_si256(codes, _mm256_set1_epi32(0x7f));
    let sign = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(0x80)), 24);
    let normal = _mm256_castsi256_ps(_mm256_or_si256(
        sign,
        _mm256_add_epi32(
            _mm256_slli_epi32(magnitude, 20),
            _mm256_set1_epi32(120 << 23),
        ),
    ));

    // ModelOpt matrices overwhelmingly use normal E4M3 values. Avoid executing the conversion,
    // multiply, and blend needed by the exponent-zero path when all eight lanes are normal. The
    // branch is highly predictable for checkpoint payloads while the uncommon mixed vector still
    // falls through to the bit-identical general path below.
    let is_subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(8), magnitude);
    if _mm256_movemask_ps(_mm256_castsi256_ps(is_subnormal)) == 0 {
        return normal;
    }

    // Exponent zero is sign * mantissa * 2^-9. XOR installs the sign bit exactly, including -0.
    let mantissa = _mm256_and_si256(magnitude, _mm256_set1_epi32(0x07));
    let subnormal = _mm256_xor_ps(
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(mantissa),
            _mm256_set1_ps(2.0f32.powi(-9)),
        ),
        _mm256_castsi256_ps(sign),
    );
    _mm256_blendv_ps(normal, subnormal, _mm256_castsi256_ps(is_subnormal))
}

/// Row-contiguous AVX2/FMA kernel for `W[start..]ᵀ · input` in native ModelOpt layout.
///
/// Each output column still visits input rows in ascending order and accumulates in F64, exactly
/// matching the scalar reference. F32-by-F32 products are exact in F64, so fused multiply-add has
/// the same rounded sum while removing one vector instruction. Traversing weights by row replaces
/// 512 strided column walks with contiguous reads while independent heads run over Rayon.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn transpose_rows_e4m3_e8m0_avx2(
    values: &[u8],
    scales: &[u8],
    columns: usize,
    scale_columns: usize,
    start: usize,
    input: &[f32],
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut output = vec![0.0f64; columns];
    for (offset_row, &coefficient) in input.iter().enumerate() {
        let row = start + offset_row;
        let row_values = &values[row * columns..(row + 1) * columns];
        let row_scales = &scales[row * scale_columns..(row + 1) * scale_columns];
        let coefficient_f64 = f64::from(coefficient);
        let coefficient_vector = _mm256_set1_pd(coefficient_f64);
        for (block, codes) in row_values.chunks(32).enumerate() {
            let scale = _mm256_set1_ps(decode_e8m0_unchecked(row_scales[block]));
            let complete = codes.len() / 8 * 8;
            let column = block * 32;
            let all_normal = e4m3_block_all_normal_avx2(codes);
            for code_offset in (0..complete).step_by(8) {
                let packed = _mm_loadl_epi64(codes.as_ptr().add(code_offset).cast::<__m128i>());
                let indices = _mm256_cvtepu8_epi32(packed);
                let decoded = if all_normal {
                    decode_normal_e4m3fn_avx2(indices)
                } else {
                    decode_e4m3fn_avx2(indices)
                };
                let weights = _mm256_mul_ps(decoded, scale);
                let low = _mm256_cvtps_pd(_mm256_castps256_ps128(weights));
                let high = _mm256_cvtps_pd(_mm256_extractf128_ps(weights, 1));
                let output_ptr = output.as_mut_ptr().add(column + code_offset);
                _mm256_storeu_pd(
                    output_ptr,
                    _mm256_fmadd_pd(low, coefficient_vector, _mm256_loadu_pd(output_ptr)),
                );
                _mm256_storeu_pd(
                    output_ptr.add(4),
                    _mm256_fmadd_pd(high, coefficient_vector, _mm256_loadu_pd(output_ptr.add(4))),
                );
            }
            for code_offset in complete..codes.len() {
                let weight = e4m3_table()[usize::from(codes[code_offset])]
                    * decode_e8m0_unchecked(row_scales[block]);
                output[column + code_offset] =
                    f64::from(weight).mul_add(coefficient_f64, output[column + code_offset]);
            }
        }
    }
    output.into_iter().map(|value| value as f32).collect()
}

#[derive(Debug, Clone)]
pub struct MxFp4Matrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    packed: ReadBuffer,
    scale_bytes: ReadBuffer,
}

impl MxFp4Matrix {
    pub fn from_packed(
        rows: usize,
        cols: usize,
        group_size: usize,
        packed: Vec<u8>,
        scale_bytes: Vec<u8>,
    ) -> Result<Self, MxError> {
        Self::from_read_buffers(rows, cols, group_size, packed.into(), scale_bytes.into())
    }

    pub(crate) fn from_read_buffers(
        rows: usize,
        cols: usize,
        group_size: usize,
        packed: ReadBuffer,
        scale_bytes: ReadBuffer,
    ) -> Result<Self, MxError> {
        Self::from_read_buffers_with_validation(rows, cols, group_size, packed, scale_bytes, true)
    }

    /// Reconstructs an MXFP4 matrix after these immutable checkpoint ranges were validated by an
    /// earlier layer load. Shape checks remain mandatory; only the repeated E8M0 scale scan is
    /// skipped.
    pub(crate) fn from_read_buffers_prevalidated(
        rows: usize,
        cols: usize,
        group_size: usize,
        packed: ReadBuffer,
        scale_bytes: ReadBuffer,
    ) -> Result<Self, MxError> {
        Self::from_read_buffers_with_validation(rows, cols, group_size, packed, scale_bytes, false)
    }

    fn from_read_buffers_with_validation(
        rows: usize,
        cols: usize,
        group_size: usize,
        packed: ReadBuffer,
        scale_bytes: ReadBuffer,
        validate_scales: bool,
    ) -> Result<Self, MxError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(MxError::Shape(
                "dimensions and group size must be non-zero".to_owned(),
            ));
        }
        let expected_packed = rows
            .checked_mul(cols.div_ceil(2))
            .ok_or_else(|| MxError::Shape("packed count overflows".to_owned()))?;
        let expected_scales = rows
            .checked_mul(cols.div_ceil(group_size))
            .ok_or_else(|| MxError::Shape("scale count overflows".to_owned()))?;
        if packed.len() != expected_packed || scale_bytes.len() != expected_scales {
            return Err(MxError::Shape(format!(
                "expected {expected_packed} bytes and {expected_scales} scales, got {} and {}",
                packed.len(),
                scale_bytes.len()
            )));
        }
        if validate_scales {
            for &scale in scale_bytes.iter() {
                decode_e8m0(scale)?;
            }
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            packed,
            scale_bytes,
        })
    }

    /// Returns native packed E2M1/E8M0 storage so a retired layer or evicted expert can donate its
    /// allocations to the next same-shaped checkpoint read.
    pub(crate) fn into_e2m1_buffers(self) -> (ReadBuffer, ReadBuffer) {
        (self.packed, self.scale_bytes)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn resident_bytes(&self) -> usize {
        self.packed.len().saturating_add(self.scale_bytes.len())
    }

    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, MxError> {
        self.matvec_rows(0, self.rows, input)
    }

    /// Applies one native MXFP4 matrix to consecutive input rows `[batch, cols]`.
    ///
    /// The AVX2 path decodes each packed weight group once for two through twelve inputs. Every
    /// input retains the same ordered-F64 product and addition sequence as [`Self::matvec`].
    pub fn matmul_rows(&self, input: &[f32], batch: usize) -> Result<Vec<f32>, MxError> {
        let expected = batch
            .checked_mul(self.cols)
            .ok_or_else(|| MxError::Shape("batch * columns overflows usize".to_owned()))?;
        if batch == 0 || input.len() != expected {
            return Err(MxError::Shape(format!(
                "batched matvec expects a non-zero batch and {expected} inputs, got {}",
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        #[cfg(target_arch = "x86_64")]
        let ordered_avx2 = (2..=12).contains(&batch)
            && self.group_size >= 8
            && self.group_size % 2 == 0
            && std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let ordered_avx2 = false;

        if !ordered_avx2 {
            let mut output = Vec::with_capacity(batch.saturating_mul(self.rows));
            for input in input.chunks_exact(self.cols) {
                output.extend(self.matvec(input)?);
            }
            return Ok(output);
        }

        let work = batch.saturating_mul(self.rows).saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecMxFp4, work);
        let row_bytes = self.cols.div_ceil(2);
        let groups = self.cols.div_ceil(self.group_size);
        let mut output_by_row = vec![0.0f32; self.rows.saturating_mul(batch)];
        let compute_row = |row: usize, output: &mut [f32]| {
            let packed = &self.packed[row * row_bytes..(row + 1) * row_bytes];
            let scales = &self.scale_bytes[row * groups..(row + 1) * groups];
            macro_rules! compute_batch {
                ($batch:literal) => {
                    dot_e2m1_e8m0_batch_f64_ordered_avx2::<$batch>(
                        packed,
                        scales,
                        input,
                        self.cols,
                        self.group_size,
                        output,
                    )
                };
            }
            // SAFETY: dispatch proves AVX2 support, batch is in 2..=12, and validated matrix/input
            // slices cover every packed nibble, scale group, and input row.
            unsafe {
                match batch {
                    2 => compute_batch!(2),
                    3 => compute_batch!(3),
                    4 => compute_batch!(4),
                    5 => compute_batch!(5),
                    6 => compute_batch!(6),
                    7 => compute_batch!(7),
                    8 => compute_batch!(8),
                    9 => compute_batch!(9),
                    10 => compute_batch!(10),
                    11 => compute_batch!(11),
                    12 => compute_batch!(12),
                    _ => unreachable!("MXFP4 batch dispatch is limited to two through twelve"),
                }
            }
        };
        if should_parallelize(self.rows, work) {
            install(|| {
                output_by_row
                    .par_chunks_mut(batch)
                    .enumerate()
                    .for_each(|(row, output)| compute_row(row, output));
            });
        } else {
            for (row, output) in output_by_row.chunks_mut(batch).enumerate() {
                compute_row(row, output);
            }
        }
        let mut output = vec![0.0f32; batch.saturating_mul(self.rows)];
        for (row, values) in output_by_row.chunks_exact(batch).enumerate() {
            for (token, &value) in values.iter().enumerate() {
                output[token * self.rows + row] = value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        Ok(output)
    }

    pub(crate) fn matvec_fp32(&self, input: &[f32]) -> Result<Vec<f32>, MxError> {
        self.matvec_rows_fp32(0, self.rows, input)
    }

    pub(crate) fn matvec_rows_fp32(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, MxError> {
        if input.len() != self.cols || input.iter().any(|value| !value.is_finite()) {
            return Err(if input.len() != self.cols {
                MxError::Shape(format!(
                    "matvec expects {} inputs, got {}",
                    self.cols,
                    input.len()
                ))
            } else {
                MxError::NonFinite
            });
        }
        let end = start
            .checked_add(count)
            .filter(|&end| end <= self.rows)
            .ok_or_else(|| MxError::Shape("matvec row range is out of bounds".to_owned()))?;
        let row_bytes = self.cols.div_ceil(2);
        let groups = self.cols.div_ceil(self.group_size);
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecMxFp4, work);
        let mut output = vec![0.0; count];
        #[cfg(target_arch = "x86_64")]
        let ordered_avx2 = self.group_size >= 8
            && self.group_size % 2 == 0
            && std::arch::is_x86_feature_detected!("avx2");
        let dot_row = |row: usize| {
            let packed = &self.packed[row * row_bytes..(row + 1) * row_bytes];
            let scales = &self.scale_bytes[row * groups..(row + 1) * groups];
            #[cfg(target_arch = "x86_64")]
            if ordered_avx2 {
                // SAFETY: AVX2 is detected; validated row slices cover every input column,
                // and even group widths keep each group's first nibble byte-aligned.
                return unsafe {
                    dot_e2m1_e8m0_fp32_ordered_avx2(packed, scales, input, self.group_size)
                };
            }
            dot_e2m1_e8m0_fp32_ordered(packed, scales, input, self.group_size)
        };
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(offset, out)| *out = dot_row(start + offset))
            });
        } else {
            for (row, out) in (start..end).zip(&mut output) {
                *out = dot_row(row);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        Ok(output)
    }

    pub fn matvec_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, MxError> {
        if input.len() != self.cols {
            return Err(MxError::Shape(format!(
                "matvec expects {} inputs, got {}",
                self.cols,
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        let end = start
            .checked_add(count)
            .filter(|&end| end <= self.rows)
            .ok_or_else(|| MxError::Shape("matvec row range is out of bounds".to_owned()))?;
        let row_bytes = self.cols.div_ceil(2);
        let groups = self.cols.div_ceil(self.group_size);
        let work = count.saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::MatvecMxFp4, work);
        let mut output = vec![0.0; count];
        #[cfg(target_arch = "x86_64")]
        let ordered_avx2 = self.group_size >= 8
            && self.group_size % 2 == 0
            && std::arch::is_x86_feature_detected!("avx2");
        let dot_row = |row: usize| {
            let row_packed = &self.packed[row * row_bytes..(row + 1) * row_bytes];
            let row_scales = &self.scale_bytes[row * groups..(row + 1) * groups];
            #[cfg(target_arch = "x86_64")]
            if ordered_avx2 {
                // SAFETY: AVX2 is detected; validated row slices cover every input column,
                // and even group widths keep each group's first nibble byte-aligned.
                return unsafe {
                    dot_e2m1_e8m0_f64_ordered_avx2(row_packed, row_scales, input, self.group_size)
                };
            }
            dot_e2m1_e8m0_f64_ordered(row_packed, row_scales, input, self.group_size)
        };
        if should_parallelize(count, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(offset, output)| *output = dot_row(start + offset));
            });
        } else {
            for (row, output) in (start..end).zip(&mut output) {
                *output = dot_row(row);
            }
        }
        Ok(output)
    }

    pub fn transpose_rows_matvec(&self, start: usize, input: &[f32]) -> Result<Vec<f32>, MxError> {
        if input.iter().any(|value| !value.is_finite()) {
            return Err(MxError::NonFinite);
        }
        let end = start
            .checked_add(input.len())
            .filter(|&end| end <= self.rows)
            .ok_or_else(|| {
                MxError::Shape("transpose matvec row range is out of bounds".to_owned())
            })?;
        let row_bytes = self.cols.div_ceil(2);
        let groups = self.cols.div_ceil(self.group_size);
        let work = input.len().saturating_mul(self.cols);
        let _profile = span_with_work(ProfileStage::TransposeMatvecMxFp4, work);
        let mut output = vec![0.0f64; self.cols];
        let compute_column = |column: usize| {
            let mut sum = 0.0f64;
            for (row, &coefficient) in (start..end).zip(input) {
                let byte = self.packed[row * row_bytes + column / 2];
                let code = if column % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let scale = decode_e8m0(self.scale_bytes[row * groups + column / self.group_size])
                    .expect("MXFP4 scale bytes were validated during construction");
                let weight = decode_e2m1(code) * scale;
                sum += f64::from(coefficient) * f64::from(weight);
            }
            sum
        };
        if should_parallelize(self.cols, work) {
            install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(column, output)| *output = compute_column(column));
            });
        } else {
            for (column, output) in output.iter_mut().enumerate() {
                *output = compute_column(column);
            }
        }
        Ok(output.into_iter().map(|value| value as f32).collect())
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>, MxError> {
        if row >= self.rows {
            return Err(MxError::Shape(format!(
                "row {row} is outside 0..{}",
                self.rows
            )));
        }
        let row_bytes = self.cols.div_ceil(2);
        let groups = self.cols.div_ceil(self.group_size);
        Ok((0..self.cols)
            .map(|col| {
                let byte = self.packed[row * row_bytes + col / 2];
                let code = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let scale = decode_e8m0(self.scale_bytes[row * groups + col / self.group_size])
                    .expect("MXFP4 scale bytes were validated during construction");
                decode_e2m1(code) * scale
            })
            .collect())
    }
}

#[inline]
fn dot_e2m1_e8m0_f64_ordered(
    packed: &[u8],
    scales: &[u8],
    input: &[f32],
    group_size: usize,
) -> f32 {
    let mut sum = 0.0f64;
    let mut column = 0;
    for (input_group, &scale_byte) in input.chunks(group_size).zip(scales) {
        let scale = decode_e8m0_unchecked(scale_byte);
        for &input_value in input_group {
            let byte = packed[column / 2];
            let code = if column % 2 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            let weight = decode_e2m1(code) * scale;
            sum += f64::from(weight) * f64::from(input_value);
            column += 1;
        }
    }
    sum as f32
}

#[inline]
fn dot_e2m1_e8m0_fp32_ordered(
    packed: &[u8],
    scales: &[u8],
    input: &[f32],
    group_size: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let mut column = 0;
    for (input_group, &scale_byte) in input.chunks(group_size).zip(scales) {
        let scale = decode_e8m0_unchecked(scale_byte);
        for &input_value in input_group {
            let byte = packed[column / 2];
            let code = if column % 2 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            sum += decode_e2m1(code) * scale * input_value;
            column += 1;
        }
    }
    sum
}

/// SIMD-decodes eight weights at a time, then multiplies and accumulates them in the scalar
/// reference's original F64 order. Keeping the multiply scalar is intentional: converting an
/// FP32 product to F64 would not match multiplying the separately widened operands.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_e2m1_e8m0_f64_ordered_avx2(
    packed: &[u8],
    scales: &[u8],
    input: &[f32],
    group_size: usize,
) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(group_size % 2, 0);
    debug_assert_eq!(packed.len(), input.len().div_ceil(2));
    debug_assert_eq!(scales.len(), input.len().div_ceil(group_size));
    let magnitudes = _mm256_loadu_ps(E2M1_TABLE.as_ptr());
    let magnitude_mask = _mm256_set1_epi32(7);
    let sign_mask = _mm256_set1_epi32(8);
    let mut sum = 0.0f64;
    let mut column = 0;
    let mut products = [0.0f64; 8];
    for (input_group, &scale_byte) in input.chunks(group_size).zip(scales) {
        let scale = decode_e8m0_unchecked(scale_byte);
        let scale_vector = _mm256_set1_ps(scale);
        let complete = input_group.len() / 8 * 8;
        for offset in (0..complete).step_by(8) {
            let bytes = _mm_cvtsi32_si128(
                packed
                    .as_ptr()
                    .add(column / 2)
                    .cast::<i32>()
                    .read_unaligned(),
            );
            let nibbles = _mm_unpacklo_epi8(bytes, _mm_srli_epi16(bytes, 4));
            let codes = _mm256_cvtepu8_epi32(nibbles);
            let indices = _mm256_and_si256(codes, magnitude_mask);
            let magnitude = _mm256_permutevar8x32_ps(magnitudes, indices);
            let signs = _mm256_slli_epi32(_mm256_and_si256(codes, sign_mask), 28);
            let signs =
                _mm256_and_si256(signs, _mm256_cmpgt_epi32(indices, _mm256_setzero_si256()));
            let decoded = _mm256_xor_ps(magnitude, _mm256_castsi256_ps(signs));
            let weights = _mm256_mul_ps(decoded, scale_vector);
            let inputs = _mm256_loadu_ps(input_group.as_ptr().add(offset));
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
            column += 8;
        }
        for &input_value in &input_group[complete..] {
            let byte = packed[column / 2];
            let code = if column % 2 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            let weight = decode_e2m1(code) * scale;
            sum += f64::from(weight) * f64::from(input_value);
            column += 1;
        }
    }
    sum as f32
}

/// Batched counterpart of [`dot_e2m1_e8m0_f64_ordered_avx2`]. Packed weights are decoded once,
/// then multiplied into independent token accumulators without changing any token's reduction
/// order or FP64 multiplication boundary.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_e2m1_e8m0_batch_f64_ordered_avx2<const BATCH: usize>(
    packed: &[u8],
    scales: &[u8],
    input: &[f32],
    columns: usize,
    group_size: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert!((2..=12).contains(&BATCH));
    debug_assert_eq!(group_size % 2, 0);
    debug_assert_eq!(packed.len(), columns.div_ceil(2));
    debug_assert_eq!(scales.len(), columns.div_ceil(group_size));
    debug_assert_eq!(input.len(), BATCH * columns);
    debug_assert_eq!(output.len(), BATCH);
    let magnitudes = _mm256_loadu_ps(E2M1_TABLE.as_ptr());
    let magnitude_mask = _mm256_set1_epi32(7);
    let sign_mask = _mm256_set1_epi32(8);
    let mut sums = [0.0f64; BATCH];
    let mut products = [0.0f64; 8];
    let mut column = 0usize;
    for (group, &scale_byte) in scales.iter().enumerate() {
        let group_end = ((group + 1) * group_size).min(columns);
        let group_columns = group_end - column;
        let scale = _mm256_set1_ps(decode_e8m0_unchecked(scale_byte));
        let complete = group_columns / 8 * 8;
        for offset in (0..complete).step_by(8) {
            let bytes = _mm_cvtsi32_si128(
                packed
                    .as_ptr()
                    .add((column + offset) / 2)
                    .cast::<i32>()
                    .read_unaligned(),
            );
            let nibbles = _mm_unpacklo_epi8(bytes, _mm_srli_epi16(bytes, 4));
            let codes = _mm256_cvtepu8_epi32(nibbles);
            let indices = _mm256_and_si256(codes, magnitude_mask);
            let magnitude = _mm256_permutevar8x32_ps(magnitudes, indices);
            let signs = _mm256_slli_epi32(_mm256_and_si256(codes, sign_mask), 28);
            let signs =
                _mm256_and_si256(signs, _mm256_cmpgt_epi32(indices, _mm256_setzero_si256()));
            let weights =
                _mm256_mul_ps(_mm256_xor_ps(magnitude, _mm256_castsi256_ps(signs)), scale);
            let weight_low = _mm256_cvtps_pd(_mm256_castps256_ps128(weights));
            let weight_high = _mm256_cvtps_pd(_mm256_extractf128_ps(weights, 1));
            for (token, sum) in sums.iter_mut().enumerate() {
                let inputs = _mm256_loadu_ps(input.as_ptr().add(token * columns + column + offset));
                let input_low = _mm256_cvtps_pd(_mm256_castps256_ps128(inputs));
                let input_high = _mm256_cvtps_pd(_mm256_extractf128_ps(inputs, 1));
                _mm256_storeu_pd(products.as_mut_ptr(), _mm256_mul_pd(weight_low, input_low));
                _mm256_storeu_pd(
                    products.as_mut_ptr().add(4),
                    _mm256_mul_pd(weight_high, input_high),
                );
                for product in products {
                    *sum += product;
                }
            }
        }
        for offset in complete..group_columns {
            let absolute = column + offset;
            let byte = packed[absolute / 2];
            let code = if absolute % 2 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            let weight = decode_e2m1(code) * decode_e8m0_unchecked(scale_byte);
            for (token, sum) in sums.iter_mut().enumerate() {
                *sum += f64::from(weight) * f64::from(input[token * columns + absolute]);
            }
        }
        column = group_end;
    }
    for (output, sum) in output.iter_mut().zip(sums) {
        *output = sum as f32;
    }
}

/// SIMD decodes and multiplies eight weights at a time, then adds their products in the
/// scalar reference order. No FMA or horizontal reduction may replace these operations:
/// V4.1 routing depends on the original FP32 rounding boundaries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_e2m1_e8m0_fp32_ordered_avx2(
    packed: &[u8],
    scales: &[u8],
    input: &[f32],
    group_size: usize,
) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(group_size % 2, 0);
    debug_assert_eq!(packed.len(), input.len().div_ceil(2));
    debug_assert_eq!(scales.len(), input.len().div_ceil(group_size));
    let magnitudes = _mm256_loadu_ps(E2M1_TABLE.as_ptr());
    let magnitude_mask = _mm256_set1_epi32(7);
    let sign_mask = _mm256_set1_epi32(8);
    let mut sum = 0.0f32;
    let mut column = 0;
    let mut products = [0.0f32; 8];
    for (input_group, &scale_byte) in input.chunks(group_size).zip(scales) {
        let scale = decode_e8m0_unchecked(scale_byte);
        let scale_vector = _mm256_set1_ps(scale);
        let complete = input_group.len() / 8 * 8;
        for offset in (0..complete).step_by(8) {
            let bytes = _mm_cvtsi32_si128(
                packed
                    .as_ptr()
                    .add(column / 2)
                    .cast::<i32>()
                    .read_unaligned(),
            );
            let nibbles = _mm_unpacklo_epi8(bytes, _mm_srli_epi16(bytes, 4));
            let codes = _mm256_cvtepu8_epi32(nibbles);
            let indices = _mm256_and_si256(codes, magnitude_mask);
            let magnitude = _mm256_permutevar8x32_ps(magnitudes, indices);
            let signs = _mm256_slli_epi32(_mm256_and_si256(codes, sign_mask), 28);
            // Both zero nibble encodings decode to positive zero in the scalar table.
            let signs =
                _mm256_and_si256(signs, _mm256_cmpgt_epi32(indices, _mm256_setzero_si256()));
            let decoded = _mm256_xor_ps(magnitude, _mm256_castsi256_ps(signs));
            let weights = _mm256_mul_ps(decoded, scale_vector);
            let inputs = _mm256_loadu_ps(input_group.as_ptr().add(offset));
            _mm256_storeu_ps(products.as_mut_ptr(), _mm256_mul_ps(weights, inputs));
            for product in products {
                sum += product;
            }
            column += 8;
        }
        for &input_value in &input_group[complete..] {
            let byte = packed[column / 2];
            let code = if column % 2 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            sum += decode_e2m1(code) * scale * input_value;
            column += 1;
        }
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
    fn real_checkpoint_fp8_prefix_decodes_exactly() {
        assert_eq!(decode_e4m3fn(0xe0), -32.0);
        assert_eq!(decode_e8m0(115).unwrap(), 2.0f32.powi(-12));
        assert_eq!(decode_e4m3fn(0xe0) * decode_e8m0(115).unwrap(), -0.0078125);
    }

    #[test]
    fn e8m0_bit_decode_matches_the_power_of_two_definition_exhaustively() {
        for code in 0u8..=0xfe {
            assert_eq!(
                decode_e8m0(code).unwrap().to_bits(),
                2.0f32.powi(i32::from(code) - 127).to_bits()
            );
        }
    }

    #[test]
    fn finite_e8m0_scales_roundtrip_through_the_scalar_encoder() {
        for code in 0..u8::MAX {
            let scale = decode_e8m0(code).unwrap();
            assert_eq!(encode_e8m0_scalar(scale).unwrap(), code);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_e4m3_decode_matches_scalar_bits_exhaustively() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::*;

        for base in (0u16..=248).step_by(8) {
            let mut codes = [0i32; 8];
            for (lane, code) in codes.iter_mut().enumerate() {
                *code = i32::from((base + lane as u16) as u8);
            }
            let mut decoded = [0.0f32; 8];
            unsafe {
                let packed = _mm256_loadu_si256(codes.as_ptr().cast::<__m256i>());
                _mm256_storeu_ps(decoded.as_mut_ptr(), decode_e4m3fn_avx2(packed));
            }
            for (lane, actual) in decoded.into_iter().enumerate() {
                let code = (base + lane as u16) as u8;
                if code & 0x7f != 0x7f {
                    assert_eq!(actual.to_bits(), decode_e4m3fn(code).to_bits());
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_normal_block_fast_path_detects_every_exponent_zero_position() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut codes = [0x38u8; 32];
        assert!(unsafe { e4m3_block_all_normal_avx2(&codes) });
        for position in 0..codes.len() {
            for subnormal in [0x00, 0x07, 0x80, 0x87] {
                codes[position] = subnormal;
                assert!(!unsafe { e4m3_block_all_normal_avx2(&codes) });
            }
            codes[position] = 0xb8;
        }

        use std::arch::x86_64::*;
        let normal_codes = (0u8..=u8::MAX)
            .filter(|code| {
                let magnitude = code & 0x7f;
                magnitude >= 8 && magnitude != 0x7f
            })
            .collect::<Vec<_>>();
        for chunk in normal_codes.chunks(8) {
            let mut lanes = [0i32; 8];
            for (lane, code) in lanes.iter_mut().zip(chunk.iter().cycle()) {
                *lane = i32::from(*code);
            }
            let mut decoded = [0.0f32; 8];
            unsafe {
                let packed = _mm256_loadu_si256(lanes.as_ptr().cast::<__m256i>());
                _mm256_storeu_ps(decoded.as_mut_ptr(), decode_normal_e4m3fn_avx2(packed));
            }
            for (actual, code) in decoded.into_iter().zip(lanes) {
                assert_eq!(actual.to_bits(), decode_e4m3fn(code as u8).to_bits());
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn modelopt_avx2_matvec_stays_within_fp32_accumulation_error() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let rows = 17usize;
        let cols = 97usize;
        let finite_codes = [0x00, 0x07, 0x18, 0x38, 0x48, 0x70, 0x7e, 0xb8, 0xfe];
        let values = (0..rows * cols)
            .map(|index| finite_codes[index.wrapping_mul(17) % finite_codes.len()])
            .collect::<Vec<_>>();
        let scale_cols = cols.div_ceil(32);
        let scales = (0..rows * scale_cols)
            .map(|index| 119 + (index.wrapping_mul(13) % 17) as u8)
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
            .collect::<Vec<_>>();
        let matrix =
            MxFp8Matrix::from_packed(rows, cols, 1, 32, values.clone(), scales.clone()).unwrap();

        for (row, actual) in matrix.matvec(&input).unwrap().into_iter().enumerate() {
            let mut expected = 0.0f64;
            let mut absolute_sum = 0.0f64;
            for column in 0..cols {
                let weight = f64::from(decode_e4m3fn(values[row * cols + column]))
                    * f64::from(decode_e8m0_unchecked(
                        scales[row * scale_cols + column / 32],
                    ));
                let product = weight * f64::from(input[column]);
                expected += product;
                absolute_sum += product.abs();
            }
            let error = (f64::from(actual) - expected).abs();
            let tolerance = 64.0 * f64::from(f32::EPSILON) * absolute_sum.max(1.0);
            assert!(
                error <= tolerance,
                "row {row}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn modelopt_batched_avx2_matches_independent_matvec_bits() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let rows = 17usize;
        let cols = 96usize;
        let finite_codes = [0x00, 0x07, 0x18, 0x38, 0x48, 0x70, 0x7e, 0xb8, 0xfe];
        let values = (0..rows * cols)
            .map(|index| finite_codes[index.wrapping_mul(17) % finite_codes.len()])
            .collect::<Vec<_>>();
        let scales = (0..rows * cols.div_ceil(32))
            .map(|index| 119 + (index.wrapping_mul(13) % 17) as u8)
            .collect::<Vec<_>>();
        let matrix = MxFp8Matrix::from_packed(rows, cols, 1, 32, values, scales).unwrap();
        for batch in 2..=12 {
            let input = (0..batch * cols)
                .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
                .collect::<Vec<_>>();
            let expected = input
                .chunks_exact(cols)
                .flat_map(|input| matrix.matvec(input).unwrap())
                .collect::<Vec<_>>();
            let actual = matrix.matmul_rows(&input, batch).unwrap();
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
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn block_mxfp8_batched_avx2_matches_independent_matvec_bits() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let rows = 257usize;
        let cols = 259usize;
        let block_rows = 128usize;
        let block_cols = 128usize;
        let finite_codes = [0x00, 0x07, 0x18, 0x38, 0x48, 0x70, 0x7e, 0xb8, 0xfe];
        let values = (0..rows * cols)
            .map(|index| finite_codes[index.wrapping_mul(17) % finite_codes.len()])
            .collect::<Vec<_>>();
        let scales = (0..rows.div_ceil(block_rows) * cols.div_ceil(block_cols))
            .map(|index| 119 + (index.wrapping_mul(13) % 17) as u8)
            .collect::<Vec<_>>();
        let matrix =
            MxFp8Matrix::from_packed(rows, cols, block_rows, block_cols, values, scales).unwrap();
        for batch in 2..=12 {
            let input = (0..batch * cols)
                .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
                .collect::<Vec<_>>();
            let expected = input
                .chunks_exact(cols)
                .flat_map(|input| matrix.matvec(input).unwrap())
                .collect::<Vec<_>>();
            let actual = matrix.matmul_rows(&input, batch).unwrap();
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
            let start = 63;
            let count = 131;
            let expected_range = input
                .chunks_exact(cols)
                .flat_map(|input| matrix.matvec_rows(start, count, input).unwrap())
                .collect::<Vec<_>>();
            let actual_range = matrix
                .matmul_row_range(start, count, &input, batch)
                .unwrap();
            assert_eq!(
                actual_range
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_range
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn modelopt_avx2_transpose_fma_is_bit_exact_to_ordered_f64_columns() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        for cols in [64usize, 71] {
            let rows = 17usize;
            let start = 2usize;
            let input = (0..11)
                .map(|index| ((index * 19 % 31) as f32 - 15.0) / 32.0)
                .collect::<Vec<_>>();
            let finite_codes = [0x00, 0x07, 0x18, 0x38, 0x48, 0x70, 0x7e, 0xb8, 0xfe];
            let values = (0..rows * cols)
                .map(|index| finite_codes[index.wrapping_mul(17) % finite_codes.len()])
                .collect::<Vec<_>>();
            let scale_cols = cols.div_ceil(32);
            let scales = (0..rows * scale_cols)
                .map(|index| 119 + (index.wrapping_mul(13) % 17) as u8)
                .collect::<Vec<_>>();
            let actual = unsafe {
                transpose_rows_e4m3_e8m0_avx2(&values, &scales, cols, scale_cols, start, &input)
            };

            for column in 0..cols {
                let mut expected = 0.0f64;
                for (offset, &coefficient) in input.iter().enumerate() {
                    let row = start + offset;
                    let weight = decode_e4m3fn(values[row * cols + column])
                        * decode_e8m0_unchecked(scales[row * scale_cols + column / 32]);
                    expected += f64::from(weight) * f64::from(coefficient);
                }
                assert_eq!(actual[column].to_bits(), (expected as f32).to_bits());
            }
        }
    }

    #[test]
    fn fp8_constructors_reject_both_e4m3_nan_encodings() {
        for code in [0x7f, 0xff] {
            assert!(matches!(
                MxFp8Matrix::from_packed(1, 1, 1, 32, vec![code], vec![127]),
                Err(MxError::NonFinite)
            ));
            assert!(matches!(
                MxFp8Matrix::from_packed_bf16_scale_inv(
                    1,
                    1,
                    128,
                    128,
                    vec![code],
                    bf16_bytes(&[1.0]),
                ),
                Err(MxError::NonFinite)
            ));
        }
    }

    #[test]
    fn native_mxfp8_storage_can_be_recovered_for_reuse() {
        let values = vec![0x38, 0xb8];
        let scales = vec![127];
        let matrix = MxFp8Matrix::from_packed(1, 2, 1, 32, values.clone(), scales.clone()).unwrap();
        let (recovered_values, recovered_scales) = matrix.into_e8m0_buffers().unwrap();
        assert_eq!(recovered_values.as_ref(), values);
        assert_eq!(recovered_scales.as_ref(), scales);
    }

    #[test]
    fn native_mxfp4_storage_can_be_recovered_for_reuse() {
        let packed = vec![0x21, 0x43];
        let scales = vec![127];
        let matrix = MxFp4Matrix::from_packed(1, 4, 32, packed.clone(), scales.clone()).unwrap();
        let (recovered_packed, recovered_scales) = matrix.into_e2m1_buffers();
        assert_eq!(recovered_packed.as_ref(), packed);
        assert_eq!(recovered_scales.as_ref(), scales);
    }

    #[test]
    fn prevalidated_mxfp8_constructor_keeps_shape_checks() {
        let matrix =
            MxFp8Matrix::from_packed_prevalidated(1, 2, 1, 32, vec![0x38, 0xb8], vec![127])
                .unwrap();
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 2);
        assert!(matches!(
            MxFp8Matrix::from_packed_prevalidated(1, 2, 1, 32, vec![0x38], vec![127]),
            Err(MxError::Shape(_))
        ));
    }

    #[test]
    fn prevalidated_mxfp4_constructor_keeps_shape_checks() {
        let matrix = MxFp4Matrix::from_read_buffers_prevalidated(
            1,
            4,
            32,
            vec![0x21, 0x43].into(),
            vec![127].into(),
        )
        .unwrap();
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 4);
        assert!(matches!(
            MxFp4Matrix::from_read_buffers_prevalidated(
                1,
                4,
                32,
                vec![0x21].into(),
                vec![127].into(),
            ),
            Err(MxError::Shape(_))
        ));
    }

    #[test]
    fn packed_fp8_validation_matches_scalar_at_every_vector_boundary() {
        for length in [0usize, 1, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 257] {
            let mut codes = vec![0x38u8; length];
            let mut scales = vec![127u8; length];
            assert!(!contains_invalid_e4m3(&codes));
            assert!(!contains_invalid_e8m0(&scales));
            for position in 0..length {
                for invalid in [0x7fu8, 0xff] {
                    codes[position] = invalid;
                    assert!(contains_invalid_e4m3(&codes));
                    codes[position] = 0x38;
                }
                scales[position] = u8::MAX;
                assert!(contains_invalid_e8m0(&scales));
                scales[position] = 127;
            }
        }
    }

    #[test]
    fn real_checkpoint_fp4_prefix_is_low_nibble_first() {
        let matrix = MxFp4Matrix::from_packed(1, 2, 2, vec![0x8c], vec![120]).unwrap();
        assert_eq!(matrix.row(0).unwrap(), vec![-0.015625, 0.0]);
        assert_eq!(matrix.resident_bytes(), 2);
        assert!(matches!(
            MxFp4Matrix::from_packed(1, 2, 2, vec![0x8c], vec![0xff]),
            Err(MxError::InvalidScale(0xff))
        ));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mxfp4_batched_avx2_matches_independent_matvec_bits() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let rows = 17usize;
        let cols = 129usize;
        let group_size = 32usize;
        let row_bytes = cols.div_ceil(2);
        let groups = cols.div_ceil(group_size);
        let packed = (0..rows * row_bytes)
            .map(|index| (index.wrapping_mul(83).wrapping_add(11)) as u8)
            .collect::<Vec<_>>();
        let scales = (0..rows * groups)
            .map(|index| 118 + (index.wrapping_mul(7) % 19) as u8)
            .collect::<Vec<_>>();
        let matrix = MxFp4Matrix::from_packed(rows, cols, group_size, packed, scales).unwrap();
        for batch in 1..=12 {
            let input = (0..batch * cols)
                .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 131.0)
                .collect::<Vec<_>>();
            let expected = input
                .chunks_exact(cols)
                .flat_map(|input| matrix.matvec(input).unwrap())
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let actual = matrix
                .matmul_rows(&input, batch)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "batch={batch}");
        }
    }

    #[test]
    fn fp8_block_scales_follow_both_matrix_axes() {
        let values = vec![0x38; 9];
        let matrix =
            MxFp8Matrix::from_packed(3, 3, 2, 2, values, vec![127, 128, 129, 130]).unwrap();
        assert_eq!(matrix.row(0).unwrap(), vec![1.0, 1.0, 2.0]);
        assert_eq!(matrix.row(2).unwrap(), vec![4.0, 4.0, 8.0]);
        assert_eq!(matrix.resident_bytes(), 9 + 4);
    }

    #[test]
    fn fp8_scalar_blocks_preserve_ordered_dot_bits_across_partial_blocks() {
        let rows = 35usize;
        for (block_rows, block_cols) in [(32usize, 32usize), (128, 128)] {
            for cols in [1usize, 31, 32, 65, 129, 2049] {
                let values = (0..rows * cols)
                    .map(|index| ((index * 79 + 17) % 127) as u8 | ((index % 2) as u8 * 128))
                    .collect::<Vec<_>>();
                let scale_cols = cols.div_ceil(block_cols);
                let scales = (0..rows.div_ceil(block_rows) * scale_cols)
                    .map(|index| 115 + (index % 19) as u8)
                    .collect::<Vec<_>>();
                let input = (0..cols)
                    .map(|index| ((index * 29 % 127) as f32 - 63.0) / 131.0)
                    .collect::<Vec<_>>();
                let matrix = MxFp8Matrix::from_packed(
                    rows,
                    cols,
                    block_rows,
                    block_cols,
                    values.clone(),
                    scales.clone(),
                )
                .unwrap();
                let actual = matrix.matvec_rows(1, rows - 2, &input).unwrap();
                for (row, actual) in (1..rows - 1).zip(actual) {
                    let mut expected = 0.0f64;
                    for column in 0..cols {
                        let weight = decode_e4m3fn(values[row * cols + column])
                            * decode_e8m0(
                                scales[row / block_rows * scale_cols + column / block_cols],
                            )
                            .unwrap();
                        expected += f64::from(weight) * f64::from(input[column]);
                    }
                    assert_eq!(actual.to_bits(), (expected as f32).to_bits());
                }
            }
        }
    }

    #[test]
    fn fp4_fp32_groups_preserve_ordered_dot_bits_across_odd_nibble_boundaries() {
        let rows = 35usize;
        for cols in [1usize, 31, 32, 65, 2049] {
            for group_size in [1usize, 3, 7, 32, 33] {
                let row_bytes = cols.div_ceil(2);
                let groups = cols.div_ceil(group_size);
                let packed = (0..rows * row_bytes)
                    .map(|index| (index * 83 + 11) as u8)
                    .collect::<Vec<_>>();
                let scales = (0..rows * groups)
                    .map(|index| 115 + (index % 19) as u8)
                    .collect::<Vec<_>>();
                let input = (0..cols)
                    .map(|index| ((index * 29 % 127) as f32 - 63.0) / 131.0)
                    .collect::<Vec<_>>();
                let matrix = MxFp4Matrix::from_packed(
                    rows,
                    cols,
                    group_size,
                    packed.clone(),
                    scales.clone(),
                )
                .unwrap();
                let actual = matrix.matvec_rows_fp32(1, rows - 2, &input).unwrap();
                for (row, actual) in (1..rows - 1).zip(actual) {
                    let mut expected = 0.0f32;
                    for column in 0..cols {
                        let byte = packed[row * row_bytes + column / 2];
                        let code = if column % 2 == 0 {
                            byte & 15
                        } else {
                            byte >> 4
                        };
                        let scale =
                            decode_e8m0(scales[row * groups + column / group_size]).unwrap();
                        expected += decode_e2m1(code) * scale * input[column];
                    }
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fp4_ordered_avx2_matches_scalar_for_extreme_scales_and_partial_groups() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let adversarial = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            0.5,
            -0.5,
            f32::MAX / 8.0,
            -f32::MAX / 8.0,
            16_777_216.0,
            1.0,
            -16_777_216.0,
        ];
        for group_size in [8usize, 10, 16, 32, 34, 128] {
            for cols in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129] {
                let input = (0..cols)
                    .map(|index| adversarial[index % adversarial.len()])
                    .collect::<Vec<_>>();
                for scale in [0, 1, 2, 125, 126, 127, 128, 251, 252, 253, 254] {
                    for code in 0..16u8 {
                        let packed = vec![code | ((15 - code) << 4); cols.div_ceil(2)];
                        let scales = vec![scale; cols.div_ceil(group_size)];
                        let expected =
                            dot_e2m1_e8m0_fp32_ordered(&packed, &scales, &input, group_size);
                        let expected_f64 =
                            dot_e2m1_e8m0_f64_ordered(&packed, &scales, &input, group_size);
                        let actual = unsafe {
                            dot_e2m1_e8m0_fp32_ordered_avx2(&packed, &scales, &input, group_size)
                        };
                        let actual_f64 = unsafe {
                            dot_e2m1_e8m0_f64_ordered_avx2(&packed, &scales, &input, group_size)
                        };
                        if expected.is_nan() {
                            assert!(actual.is_nan());
                        } else {
                            assert_eq!(actual.to_bits(), expected.to_bits());
                        }
                        if expected_f64.is_nan() {
                            assert!(actual_f64.is_nan());
                        } else {
                            assert_eq!(actual_f64.to_bits(), expected_f64.to_bits());
                        }
                        let matrix =
                            MxFp4Matrix::from_packed(1, cols, group_size, packed, scales).unwrap();
                        assert_eq!(
                            matrix.matvec(&input).unwrap()[0].to_bits(),
                            expected_f64.to_bits()
                        );
                        if expected.is_finite() {
                            assert_eq!(
                                matrix.matvec_fp32(&input).unwrap()[0].to_bits(),
                                expected.to_bits()
                            );
                        } else {
                            assert_eq!(matrix.matvec_fp32(&input), Err(MxError::NonFinite));
                        }
                    }
                }
            }
        }
        let matrix = MxFp4Matrix::from_packed(1, 32, 32, vec![0x22; 16], vec![127]).unwrap();
        for invalid in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut input = vec![1.0; 32];
            input[7] = invalid;
            assert_eq!(matrix.matvec_fp32(&input), Err(MxError::NonFinite));
        }
    }

    #[test]
    fn qwen_bf16_scale_inv_follows_128_by_128_blocks() {
        let values = vec![0x38; 129 * 129];
        let matrix = MxFp8Matrix::from_packed_bf16_scale_inv(
            129,
            129,
            128,
            128,
            values,
            bf16_bytes(&[1.0, 2.0, 4.0, 8.0]),
        )
        .unwrap();
        let first = matrix.row(0).unwrap();
        let second_block_row = matrix.row(128).unwrap();
        assert_eq!((&first[..2], first[128]), (&[1.0, 1.0][..], 2.0));
        assert_eq!(
            (&second_block_row[..2], second_block_row[128]),
            (&[4.0, 4.0][..], 8.0)
        );
        assert_eq!(matrix.resident_bytes(), 129 * 129 + 4 * 4);
    }

    #[test]
    fn qwen_bf16_scale_inv_rejects_bad_counts_and_non_positive_or_non_finite_values() {
        assert!(matches!(
            MxFp8Matrix::from_packed_bf16_scale_inv(1, 1, 128, 128, vec![0x38], vec![0]),
            Err(MxError::Shape(_))
        ));
        for bits in [0x0000u16, 0xbf80, 0x7fc0] {
            assert!(matches!(
                MxFp8Matrix::from_packed_bf16_scale_inv(
                    1,
                    1,
                    128,
                    128,
                    vec![0x38],
                    bits.to_le_bytes().to_vec(),
                ),
                Err(MxError::InvalidBf16Scale(actual)) if actual == bits
            ));
        }
        assert_eq!(
            MxError::InvalidBf16Scale(0).to_string(),
            "BF16 scale bits 0x0000 do not decode to a finite, strictly positive value"
        );
    }

    #[test]
    fn activation_simulation_uses_power_of_two_blocks() {
        let values = simulate_e4m3_activation(&[448.0, 0.1, 896.0, -896.0], 2).unwrap();
        assert_eq!(values[0], 448.0);
        assert_eq!(values[2], 896.0);
        assert_eq!(values[3], -896.0);
        let fp4 = simulate_e2m1_activation(&[6.0, -6.0, 0.25, 0.75], 2).unwrap();
        assert_eq!(fp4[..2], [6.0, -6.0]);
    }

    #[test]
    fn bitwise_power_of_two_ceiling_matches_the_previous_float_formula() {
        for exponent in 0u32..=254 {
            let boundary = (0u32..=8_192)
                .chain((0x007f_ffff - 8_192)..=0x007f_ffff)
                .chain((0u32..1_024).map(|sample| {
                    sample
                        .wrapping_mul(0x0001_f123)
                        .wrapping_add(exponent.wrapping_mul(0x0000_9e37))
                        & 0x007f_ffff
                }));
            for mantissa in boundary {
                let value = f32::from_bits((exponent << 23) | mantissa);
                if !value.is_finite() || value <= 0.0 {
                    continue;
                }
                let previous = 2.0f32.powf(value.log2().ceil());
                assert_eq!(
                    power_of_two_ceiling(value).to_bits(),
                    previous.to_bits(),
                    "value={value:e} bits=0x{:08x}",
                    value.to_bits()
                );
            }
        }
    }

    #[test]
    fn in_place_activation_simulation_matches_allocating_form_and_validates_first() {
        let input = [448.0, 0.1, -0.0, -17.25, 896.0, -896.0, 1.0e-7];
        for block_size in [1usize, 2, 3, 4, 7, 8] {
            let expected = simulate_e4m3_activation(&input, block_size).unwrap();
            let mut actual = input;
            simulate_e4m3_activation_in_place(&mut actual, block_size).unwrap();
            assert_eq!(
                actual.map(f32::to_bits),
                expected
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
                    .as_slice()
            );
        }

        let mut invalid = [1.0f32, f32::NAN, 2.0];
        let before = invalid.map(f32::to_bits);
        assert_eq!(
            simulate_e4m3_activation_in_place(&mut invalid, 2),
            Err(MxError::NonFinite)
        );
        assert_eq!(invalid.map(f32::to_bits), before);
    }

    #[test]
    fn finegrained_activation_uses_exact_absmax_scale() {
        let input = [1.0, 0.3, 0.0, 0.0];
        let output = simulate_finegrained_e4m3_activation(&input, 2).unwrap();
        assert_eq!(output[0], 1.0);
        assert!((output[1] - decode_e4m3fn(0x70) / 448.0).abs() < 1e-7);
        assert_eq!(&output[2..], &[0.0, 0.0]);
    }

    #[test]
    fn binary_e4m3_encoder_matches_the_exhaustive_reference() {
        let slow = |value: f32| nearest_finite_code(value.clamp(-448.0, 448.0), e4m3_table(), 0xfe);
        for code in 0u8..=0xfe {
            let value = decode_e4m3fn(code);
            if value.is_finite() {
                assert_eq!(encode_e4m3fn(value).unwrap(), slow(value));
            }
        }
        for code in 0usize..0x7e {
            let midpoint = (e4m3_table()[code] + e4m3_table()[code + 1]) * 0.5;
            for value in [midpoint, -midpoint] {
                assert_eq!(encode_e4m3fn(value).unwrap(), slow(value));
            }
        }

        let mut bits = 0x1234_5678u32;
        for _ in 0..10_000 {
            bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = f32::from_bits(bits);
            if value.is_finite() {
                assert_eq!(encode_e4m3fn(value).unwrap(), slow(value));
            }
        }
    }

    #[test]
    fn normalized_hadamard_is_orthonormal() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let energy = values.iter().map(|value| value * value).sum::<f32>();
        normalized_hadamard(&mut values).unwrap();
        let transformed = values.iter().map(|value| value * value).sum::<f32>();
        assert!((energy - transformed).abs() < 1e-5);
    }
}
