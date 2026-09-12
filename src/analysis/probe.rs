//! Bounded, deterministic tensor sampling for model forensics.

use super::classify::expected_matrix_shape;
use crate::config::{ConfigError, ModelConfig};
use crate::math::{decode_e2m1, decode_e4m3fn, decode_e8m0};
use crate::storage::{
    infer_unique_packed_bits, validate_quantized_scale_layout, DType, SafetensorError, TensorIndex,
};
use serde::Serialize;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum ProbeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Unsupported(String),
    Invalid(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(f),
            Self::Checkpoint(error) => error.fmt(f),
            Self::Unsupported(reason) => write!(f, "unsupported tensor probe: {reason}"),
            Self::Invalid(reason) => write!(f, "invalid tensor probe: {reason}"),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<ConfigError> for ProbeError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl From<SafetensorError> for ProbeError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NumericStats {
    pub count: u64,
    pub non_finite: u64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub mean: Option<f64>,
    pub standard_deviation: Option<f64>,
    pub zero_fraction: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantProbe {
    pub bits_per_weight: u8,
    pub scale_layout: String,
    pub group_size: Option<u64>,
    pub code_histogram: Vec<u64>,
    pub saturation_fraction: f64,
    pub scales: Option<NumericStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeReport {
    pub model_path: PathBuf,
    pub tensor: String,
    pub dtype: String,
    pub declared_shape: Vec<u64>,
    pub logical_matrix_shape: Option<[u64; 2]>,
    pub storage_bytes: u64,
    pub sidecar_storage_bytes: u64,
    pub sampled_storage_bytes: u64,
    pub sampled_values: NumericStats,
    pub quantization: Option<QuantProbe>,
    pub sampling_note: String,
}

pub fn probe_tensor(
    model_dir: &Path,
    name: &str,
    max_samples: usize,
) -> Result<ProbeReport, ProbeError> {
    if max_samples == 0 || max_samples > 10_000_000 {
        return Err(ProbeError::Invalid(
            "max_samples must be in 1..=10,000,000".to_owned(),
        ));
    }
    let config = ModelConfig::load(model_dir)?;
    let index = TensorIndex::open(model_dir)?;
    let tensor = index.require(name)?;
    let logical_shape = match &config {
        ModelConfig::Glm52(config) if !name.ends_with(".qs") => expected_matrix_shape(name, config),
        ModelConfig::Glm52(_) => None,
        ModelConfig::DeepseekV4(_) => deepseek_logical_shape(&index, name)?,
        ModelConfig::DeepseekV41(_) => deepseek_logical_shape(&index, name)?,
        ModelConfig::Hy4(_) => hy4_logical_shape(&index, name)?,
        ModelConfig::KimiK3(_) => kimi_k3_logical_shape(&index, name)?,
        ModelConfig::Qwen38(_) => qwen38_logical_shape(&index, name)?,
    };
    let (stats, quantization, sampled_bytes) = match tensor.dtype {
        DType::F32 => {
            let segments = sample_segments(&index, name, max_samples.saturating_mul(4), 4)?;
            let mut accumulator = Accumulator::default();
            let mut bytes = 0u64;
            for (_, segment) in segments {
                bytes += segment.len() as u64;
                for chunk in segment.chunks_exact(4) {
                    accumulator.push(f32::from_le_bytes(chunk.try_into().unwrap()) as f64);
                }
            }
            (accumulator.finish(), None, bytes)
        }
        DType::Bf16 => {
            let segments = sample_segments(&index, name, max_samples.saturating_mul(2), 2)?;
            let mut accumulator = Accumulator::default();
            let mut bytes = 0u64;
            for (_, segment) in segments {
                bytes += segment.len() as u64;
                for chunk in segment.chunks_exact(2) {
                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                    accumulator.push(f32::from_bits(u32::from(bits) << 16) as f64);
                }
            }
            (accumulator.finish(), None, bytes)
        }
        DType::F16 => {
            let segments = sample_segments(&index, name, max_samples.saturating_mul(2), 2)?;
            let mut accumulator = Accumulator::default();
            let mut bytes = 0u64;
            for (_, segment) in segments {
                bytes += segment.len() as u64;
                for chunk in segment.chunks_exact(2) {
                    accumulator
                        .push(f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())) as f64);
                }
            }
            (accumulator.finish(), None, bytes)
        }
        DType::U8 if name.ends_with(".weight_packed") && native_scale(&index, name).is_some() => {
            probe_mx(&index, name, logical_shape, max_samples, true)?
        }
        DType::U8 if logical_shape.is_some() => {
            probe_packed(&index, name, logical_shape.unwrap(), max_samples)?
        }
        DType::F8E4M3 => probe_mx(&index, name, logical_shape, max_samples, false)?,
        DType::I8 if native_scale(&index, name).is_some() => {
            probe_mx(&index, name, logical_shape, max_samples, true)?
        }
        DType::F8E8M0 => {
            let segments = sample_segments(&index, name, max_samples, 1)?;
            let mut accumulator = Accumulator::default();
            let mut bytes = 0u64;
            for (_, segment) in segments {
                bytes += segment.len() as u64;
                for byte in segment {
                    accumulator.push(
                        decode_e8m0(byte).map_err(|error| ProbeError::Invalid(error.to_string()))?
                            as f64,
                    );
                }
            }
            (accumulator.finish(), None, bytes)
        }
        DType::U8 | DType::I8 => {
            let segments = sample_segments(&index, name, max_samples, 1)?;
            let mut accumulator = Accumulator::default();
            let mut bytes = 0u64;
            for (_, segment) in segments {
                bytes += segment.len() as u64;
                for byte in segment {
                    let value = if tensor.dtype == DType::I8 {
                        i8::from_le_bytes([byte]) as f64
                    } else {
                        f64::from(byte)
                    };
                    accumulator.push(value);
                }
            }
            (accumulator.finish(), None, bytes)
        }
        ref dtype => {
            return Err(ProbeError::Unsupported(format!(
                "dtype {dtype}; supported probes are F32/BF16/F16/U8/I8/MXFP8/MXFP4/E8M0"
            )))
        }
    };

    Ok(ProbeReport {
        model_path: model_dir.to_path_buf(),
        tensor: name.to_owned(),
        dtype: tensor.dtype.to_string(),
        declared_shape: tensor.shape.clone(),
        logical_matrix_shape: logical_shape.map(|(rows, cols)| [rows, cols]),
        storage_bytes: tensor.data_len,
        sidecar_storage_bytes: index
            .get(&format!("{name}.qs"))
            .or_else(|| native_scale(&index, name))
            .map(|sidecar| sidecar.data_len)
            .unwrap_or(0),
        sampled_storage_bytes: sampled_bytes,
        sampled_values: stats,
        quantization,
        sampling_note: "deterministic evenly spaced weight windows; quantized probes validate and read the bounded scale sidecar; statistics are not a full weight scan"
            .to_owned(),
    })
}

fn deepseek_logical_shape(
    index: &TensorIndex,
    name: &str,
) -> Result<Option<(u64, u64)>, ProbeError> {
    let tensor = index.require(name)?;
    if name.ends_with(".scale") || tensor.shape.len() != 2 {
        return Ok(None);
    }
    if tensor.dtype == DType::I8 {
        let scale = native_scale(index, name).ok_or_else(|| {
            ProbeError::Invalid(format!("MXFP4 tensor {name:?} has no native scale"))
        })?;
        if scale.shape.len() != 2 || tensor.shape[0] != scale.shape[0] {
            return Err(ProbeError::Invalid(format!(
                "MXFP4 tensor {name:?} has inconsistent weight/scale rows"
            )));
        }
        return Ok(Some((tensor.shape[0], scale.shape[1].saturating_mul(32))));
    }
    Ok(Some((tensor.shape[0], tensor.shape[1])))
}

fn kimi_k3_logical_shape(
    index: &TensorIndex,
    name: &str,
) -> Result<Option<(u64, u64)>, ProbeError> {
    let tensor = index.require(name)?;
    if name.ends_with(".weight_scale") || tensor.shape.len() != 2 {
        return Ok(None);
    }
    if name.ends_with(".weight_packed") {
        if tensor.dtype != DType::U8 {
            return Err(ProbeError::Invalid(format!(
                "Kimi-K3 packed tensor {name:?} has dtype {}, expected U8",
                tensor.dtype
            )));
        }
        let scale = native_scale(index, name).ok_or_else(|| {
            ProbeError::Invalid(format!(
                "Kimi-K3 MXFP4 tensor {name:?} has no .weight_scale sidecar"
            ))
        })?;
        if scale.dtype != DType::U8 || scale.shape.len() != 2 || tensor.shape[0] != scale.shape[0] {
            return Err(ProbeError::Invalid(format!(
                "Kimi-K3 MXFP4 tensor {name:?} has inconsistent weight/scale metadata"
            )));
        }
        let logical_columns = scale.shape[1].checked_mul(32).ok_or_else(|| {
            ProbeError::Invalid(format!(
                "Kimi-K3 MXFP4 tensor {name:?} logical column count overflows"
            ))
        })?;
        if tensor.shape[1] != logical_columns.div_ceil(2) {
            return Err(ProbeError::Invalid(format!(
                "Kimi-K3 MXFP4 tensor {name:?} packed columns do not match its group-32 scale layout"
            )));
        }
        return Ok(Some((tensor.shape[0], logical_columns)));
    }
    Ok(Some((tensor.shape[0], tensor.shape[1])))
}

fn qwen38_logical_shape(index: &TensorIndex, name: &str) -> Result<Option<(u64, u64)>, ProbeError> {
    let tensor = index.require(name)?;
    if name.ends_with(".weight_scale_inv") || tensor.shape.len() != 2 {
        return Ok(None);
    }
    Ok(Some((tensor.shape[0], tensor.shape[1])))
}

fn hy4_logical_shape(index: &TensorIndex, name: &str) -> Result<Option<(u64, u64)>, ProbeError> {
    let tensor = index.require(name)?;
    if name.ends_with("_scale") || tensor.shape.len() != 2 {
        return Ok(None);
    }
    Ok(Some((tensor.shape[0], tensor.shape[1])))
}

fn native_scale<'a>(index: &'a TensorIndex, name: &str) -> Option<&'a crate::storage::TensorInfo> {
    if let Some(prefix) = name.strip_suffix(".weight_packed") {
        index.get(&format!("{prefix}.weight_scale"))
    } else {
        name.strip_suffix(".weight").and_then(|prefix| {
            index
                .get(&format!("{prefix}.scale"))
                .or_else(|| index.get(&format!("{prefix}.weight_scale")))
                .or_else(|| index.get(&format!("{prefix}.weight_scale_inv")))
        })
    }
}

fn probe_mx(
    index: &TensorIndex,
    name: &str,
    logical_shape: Option<(u64, u64)>,
    max_samples: usize,
    fp4: bool,
) -> Result<(NumericStats, Option<QuantProbe>, u64), ProbeError> {
    let (rows, columns) = logical_shape.ok_or_else(|| {
        ProbeError::Invalid(format!("MX tensor {name:?} has no logical matrix shape"))
    })?;
    let scale = native_scale(index, name)
        .ok_or_else(|| ProbeError::Invalid(format!("MX tensor {name:?} has no native scale")))?;
    if !matches!(scale.dtype, DType::F8E8M0 | DType::U8 | DType::Bf16)
        || scale.shape.len() != 2
        || scale.data_len > 64 * 1024 * 1024
        || (fp4 && scale.dtype == DType::Bf16)
    {
        return Err(ProbeError::Invalid(format!(
            "MX/block-FP8 scale {} must be bounded E8M0, or BF16 scale_inv for E4M3",
            scale.name
        )));
    }
    let block32_shape = [rows.div_ceil(32), columns.div_ceil(32)];
    if !fp4 {
        let block128_shape = [rows.div_ceil(128), columns.div_ceil(128)];
        let modelopt_shape = [rows, columns.div_ceil(32)];
        let modelopt = scale.dtype == DType::U8 && scale.shape == modelopt_shape;
        if scale.shape != block32_shape && scale.shape != block128_shape && !modelopt {
            return Err(ProbeError::Invalid(format!(
                "E4M3 tensor {name:?} needs 32x32 grid {block32_shape:?}, 128x128 grid {block128_shape:?}, or ModelOpt 1x32 grid {modelopt_shape:?}, got {:?}",
                scale.shape
            )));
        }
        let expected_dtype = if scale.name.ends_with(".weight_scale_inv") {
            DType::Bf16
        } else if modelopt {
            DType::U8
        } else {
            DType::F8E8M0
        };
        if scale.dtype != expected_dtype {
            return Err(ProbeError::Invalid(format!(
                "scale sidecar {} has dtype {}, expected {} for its naming contract",
                scale.name, scale.dtype, expected_dtype
            )));
        }
    }
    let scale_bytes = index.read_range(&scale.name, 0, scale.data_len as usize)?;
    let decoded_scales = if scale.dtype == DType::Bf16 {
        scale_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes(chunk.try_into().expect("two-byte BF16 chunk"));
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect::<Vec<_>>()
    } else {
        scale_bytes
            .iter()
            .map(|&byte| decode_e8m0(byte).map_err(|error| ProbeError::Invalid(error.to_string())))
            .collect::<Result<Vec<_>, _>>()?
    };
    if decoded_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(ProbeError::Invalid(format!(
            "scale sidecar {} contains a non-finite or non-positive value",
            scale.name
        )));
    }
    let mut scale_accumulator = Accumulator::default();
    for &value in &decoded_scales {
        scale_accumulator.push(f64::from(value));
    }
    let segments = sample_segments(index, name, max_samples.div_ceil(1 + usize::from(fp4)), 1)?;
    let mut accumulator = Accumulator::default();
    let mut histogram = vec![0u64; if fp4 { 16 } else { 256 }];
    let physical_columns = if fp4 { columns.div_ceil(2) } else { columns };
    let scale_columns = scale.shape[1];
    let mut sampled_bytes = scale.data_len;
    for (offset, segment) in segments {
        sampled_bytes += segment.len() as u64;
        for (within, byte) in segment.into_iter().enumerate() {
            let physical = offset + within as u64;
            let row = physical / physical_columns;
            let physical_column = physical % physical_columns;
            if row >= rows {
                continue;
            }
            if fp4 {
                for (half, code) in [(0u64, byte & 0x0f), (1, byte >> 4)] {
                    let column = physical_column * 2 + half;
                    if column >= columns {
                        continue;
                    }
                    histogram[usize::from(code)] += 1;
                    let scale_index = (row * scale_columns + column / 32) as usize;
                    accumulator.push(f64::from(decode_e2m1(code) * decoded_scales[scale_index]));
                }
            } else {
                histogram[usize::from(byte)] += 1;
                let scale_index = if scale.dtype == DType::U8 {
                    (row * scale_columns + physical_column / 32) as usize
                } else {
                    let block = if scale.shape == block32_shape {
                        32
                    } else {
                        128
                    };
                    ((row / block) * scale_columns + physical_column / block) as usize
                };
                accumulator.push(f64::from(decode_e4m3fn(byte) * decoded_scales[scale_index]));
            }
        }
    }
    let sample_count = histogram.iter().sum::<u64>().max(1);
    let saturated = if fp4 {
        histogram[7] + histogram[15]
    } else {
        histogram[0x7e] + histogram[0xfe]
    };
    Ok((
        accumulator.finish(),
        Some(QuantProbe {
            bits_per_weight: if fp4 { 4 } else { 8 },
            scale_layout: if fp4 {
                "mxfp4_group_32"
            } else if scale.dtype == DType::Bf16 {
                "block_fp8_e4m3_bf16_scale_inv_128x128"
            } else if scale.dtype == DType::U8 {
                "modelopt_mxfp8_group_32"
            } else if scale.shape == block32_shape {
                "mxfp8_block_32x32"
            } else {
                "mxfp8_block_128x128"
            }
            .to_owned(),
            group_size: fp4.then_some(32),
            code_histogram: histogram,
            saturation_fraction: saturated as f64 / sample_count as f64,
            scales: Some(scale_accumulator.finish()),
        }),
        sampled_bytes,
    ))
}

fn probe_packed(
    index: &TensorIndex,
    name: &str,
    (rows, cols): (u64, u64),
    max_samples: usize,
) -> Result<(NumericStats, Option<QuantProbe>, u64), ProbeError> {
    let tensor = index.require(name)?;
    let bits = infer_unique_packed_bits(rows, cols, tensor.data_len, &[4, 8])
        .map_err(ProbeError::Unsupported)?;
    let scale_name = format!("{name}.qs");
    let scale_tensor = index.get(&scale_name).ok_or_else(|| {
        ProbeError::Invalid(format!(
            "packed U8 matrix is missing required sidecar {scale_name}"
        ))
    })?;
    if scale_tensor.dtype != DType::F32 {
        return Err(ProbeError::Invalid(format!(
            "{scale_name} must be F32, got {}",
            scale_tensor.dtype
        )));
    }
    let group_size = validate_quantized_scale_layout(bits, rows, cols, scale_tensor)
        .map_err(ProbeError::Invalid)?
        .unwrap_or(cols);
    if scale_tensor.data_len > 64 * 1024 * 1024 {
        return Err(ProbeError::Unsupported(format!(
            "{scale_name} is larger than the bounded 64 MiB scale-read limit"
        )));
    }
    let scale_bytes = index.read_range(&scale_name, 0, scale_tensor.data_len as usize)?;
    let scales = scale_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(ProbeError::Invalid(format!(
            "{scale_name} contains a non-finite or non-positive scale"
        )));
    }

    let scale_layout = if scale_tensor.shape == [rows] {
        "per_row"
    } else {
        "grouped"
    }
    .to_owned();
    let scale_stats = Some({
        let mut accumulator = Accumulator::default();
        for &value in &scales {
            accumulator.push(f64::from(value));
        }
        accumulator.finish()
    });

    let bytes_requested = if bits == 4 {
        max_samples.div_ceil(2)
    } else {
        max_samples
    };
    let segments = sample_segments(index, name, bytes_requested, 1)?;
    let row_bytes = if bits == 4 { cols.div_ceil(2) } else { cols };
    let groups_per_row = scales.len() as u64 / rows;
    let mut histogram = vec![0u64; 1usize << bits];
    let mut accumulator = Accumulator::default();
    let mut sampled_bytes = scale_tensor.data_len;

    for (offset, segment) in segments {
        sampled_bytes += segment.len() as u64;
        for (within, byte) in segment.into_iter().enumerate() {
            let byte_index = offset + within as u64;
            let row = byte_index / row_bytes;
            let byte_col = byte_index % row_bytes;
            if row >= rows {
                continue;
            }
            if bits == 4 {
                for (nibble_index, code) in [(0u64, byte & 0x0f), (1, byte >> 4)] {
                    let col = byte_col * 2 + nibble_index;
                    if col >= cols {
                        continue;
                    }
                    histogram[code as usize] += 1;
                    let integer = i32::from(code) - 8;
                    accumulator.push(dequantized(
                        integer,
                        row,
                        col,
                        cols,
                        groups_per_row,
                        group_size,
                        &scales,
                    ));
                }
            } else {
                let code = byte as usize;
                histogram[code] += 1;
                let integer = i8::from_le_bytes([byte]) as i32;
                accumulator.push(dequantized(
                    integer,
                    row,
                    byte_col,
                    cols,
                    groups_per_row,
                    group_size,
                    &scales,
                ));
            }
        }
    }
    let code_count: u64 = histogram.iter().sum();
    let saturated = if bits == 4 {
        histogram[0] + histogram[15]
    } else {
        histogram[128] + histogram[127]
    };
    let quantization = QuantProbe {
        bits_per_weight: bits,
        scale_layout,
        group_size: Some(group_size),
        code_histogram: histogram,
        saturation_fraction: saturated as f64 / code_count.max(1) as f64,
        scales: scale_stats,
    };
    Ok((accumulator.finish(), Some(quantization), sampled_bytes))
}

#[allow(clippy::too_many_arguments)]
fn dequantized(
    integer: i32,
    row: u64,
    col: u64,
    cols: u64,
    groups_per_row: u64,
    group_size: u64,
    scales: &[f32],
) -> f64 {
    debug_assert!(groups_per_row > 0);
    let group = (col / group_size).min(groups_per_row - 1);
    let index = row.saturating_mul(groups_per_row).saturating_add(group) as usize;
    let scale = scales[index];
    let _ = cols; // retained in the signature to make row/column geometry explicit.
    f64::from(integer as f32 * scale)
}

/// Returns `(absolute tensor-relative byte offset, bytes)` windows.
fn sample_segments(
    index: &TensorIndex,
    name: &str,
    budget_bytes: usize,
    alignment: u64,
) -> Result<Vec<(u64, Vec<u8>)>, ProbeError> {
    let tensor = index.require(name)?;
    let budget = budget_bytes
        .max(alignment as usize)
        .min(tensor.data_len as usize);
    if budget as u64 >= tensor.data_len {
        return Ok(vec![(
            0,
            index.read_range(name, 0, tensor.data_len as usize)?,
        )]);
    }
    let alignment = alignment as usize;
    let count = (budget / alignment).clamp(1, 64);
    let window = (budget / count / alignment * alignment).max(alignment);
    let last_start = tensor.data_len.saturating_sub(window as u64);
    let mut segments = Vec::with_capacity(count);
    for index_number in 0..count {
        let raw_start = if count == 1 {
            last_start / 2
        } else {
            last_start.saturating_mul(index_number as u64) / (count - 1) as u64
        };
        let start = raw_start - raw_start % alignment as u64;
        let remaining_budget = budget.saturating_sub(segments.len() * window);
        let len = window.min(remaining_budget.max(alignment));
        segments.push((start, index.read_range(name, start, len)?));
    }
    Ok(segments)
}

#[derive(Default)]
struct Accumulator {
    finite_count: u64,
    non_finite: u64,
    zeros: u64,
    mean: f64,
    m2: f64,
    minimum: f64,
    maximum: f64,
}

impl Accumulator {
    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            self.non_finite += 1;
            return;
        }
        if self.finite_count == 0 {
            self.minimum = value;
            self.maximum = value;
        } else {
            self.minimum = self.minimum.min(value);
            self.maximum = self.maximum.max(value);
        }
        if value == 0.0 {
            self.zeros += 1;
        }
        self.finite_count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.finite_count as f64;
        self.m2 += delta * (value - self.mean);
    }

    fn finish(self) -> NumericStats {
        let count = self.finite_count + self.non_finite;
        if self.finite_count == 0 {
            return NumericStats {
                count,
                non_finite: self.non_finite,
                minimum: None,
                maximum: None,
                mean: None,
                standard_deviation: None,
                zero_fraction: None,
            };
        }
        NumericStats {
            count,
            non_finite: self.non_finite,
            minimum: Some(self.minimum),
            maximum: Some(self.maximum),
            mean: Some(self.mean),
            standard_deviation: Some((self.m2 / self.finite_count as f64).sqrt()),
            zero_fraction: Some(self.zeros as f64 / self.finite_count as f64),
        }
    }
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = (u32::from(value & 0x8000)) << 16;
    let exponent = (value >> 10) & 0x1f;
    let mantissa = value & 0x03ff;
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut fraction = u32::from(mantissa);
            let mut shift = 0u32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                shift += 1;
            }
            fraction &= 0x03ff;
            let exp = 127u32 - 15 - shift + 1;
            sign | (exp << 23) | (fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(mantissa) << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (u32::from(mantissa) << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_uses_population_standard_deviation() {
        let mut acc = Accumulator::default();
        for value in [1.0, 2.0, 3.0] {
            acc.push(value);
        }
        let stats = acc.finish();
        assert_eq!(stats.mean, Some(2.0));
        assert!((stats.standard_deviation.unwrap() - (2.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn half_decoder_handles_normal_and_special_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
    }
}
