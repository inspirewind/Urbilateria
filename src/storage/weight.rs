use super::load::decode_reference_values;
use super::{DType, SafetensorError, TensorIndex, TensorInfo, TensorLoadError};
use crate::math::{Int4Matrix, Int8Matrix, MxError, MxFp4Matrix, MxFp8Matrix, QuantError};
use crate::model::{Bf16Matrix, DenseMatrix, MatrixError, WeightMatrix};
use std::fmt;

#[derive(Debug)]
pub enum WeightLoadError {
    Checkpoint(SafetensorError),
    Reference(TensorLoadError),
    Matrix(MatrixError),
    Quantized(QuantError),
    Mx(MxError),
    InvalidShape(String),
    Unsupported(String),
    Budget {
        name: String,
        required: u64,
        maximum: u64,
    },
}

impl fmt::Display for WeightLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(f),
            Self::Reference(error) => error.fmt(f),
            Self::Matrix(error) => error.fmt(f),
            Self::Quantized(error) => error.fmt(f),
            Self::Mx(error) => error.fmt(f),
            Self::InvalidShape(reason) => write!(f, "invalid weight shape: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported weight container: {reason}"),
            Self::Budget {
                name,
                required,
                maximum,
            } => write!(
                f,
                "weight {name:?} needs {required} resident bytes, caller budget is {maximum}"
            ),
        }
    }
}

impl std::error::Error for WeightLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            Self::Reference(error) => Some(error),
            Self::Matrix(error) => Some(error),
            Self::Quantized(error) => Some(error),
            Self::Mx(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SafetensorError> for WeightLoadError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<TensorLoadError> for WeightLoadError {
    fn from(value: TensorLoadError) -> Self {
        Self::Reference(value)
    }
}

impl From<MatrixError> for WeightLoadError {
    fn from(value: MatrixError) -> Self {
        Self::Matrix(value)
    }
}

impl From<QuantError> for WeightLoadError {
    fn from(value: QuantError) -> Self {
        Self::Quantized(value)
    }
}

impl From<MxError> for WeightLoadError {
    fn from(value: MxError) -> Self {
        Self::Mx(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    /// Compact BF16 storage, returned only by the explicit compact-BF16 inspector.
    Bf16,
    Int8PerRow,
    Int4 {
        group_size: usize,
    },
    MxFp8E4M3 {
        block_rows: usize,
        block_cols: usize,
    },
    /// E4M3 values with Qwen's BF16 `weight_scale_inv` on a two-dimensional block grid.
    BlockFp8E4M3Bf16ScaleInv {
        block_rows: usize,
        block_cols: usize,
    },
    MxFp4E2M1 {
        group_size: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightLayout {
    pub format: WeightFormat,
    pub resident_bytes: u64,
}

/// Validates one logical matrix using safetensors metadata only; no payload bytes are read.
pub fn inspect_weight_matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightLayout, WeightLoadError> {
    if rows == 0 || cols == 0 {
        return Err(WeightLoadError::InvalidShape(
            "rows and columns must be non-zero".to_owned(),
        ));
    }
    let tensor = index.require(name)?;
    let logical = rows
        .checked_mul(cols)
        .ok_or_else(|| WeightLoadError::InvalidShape("rows * cols overflows usize".to_owned()))?;
    let scale_name = format!("{name}.qs");
    let scale = index.get(&scale_name);
    let mx_scale_name = find_native_scale_name(index, name);
    let mx_scale = index.get(&mx_scale_name);
    let block_fp8_scale_name = block_fp8_scale_inv_name(name);
    let block_fp8_scale = index.get(&block_fp8_scale_name);
    let present_scale_names = [
        scale.as_ref().map(|_| scale_name.as_str()),
        mx_scale.as_ref().map(|_| mx_scale_name.as_str()),
        block_fp8_scale
            .as_ref()
            .map(|_| block_fp8_scale_name.as_str()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if present_scale_names.len() > 1 {
        return Err(WeightLoadError::Unsupported(format!(
            "tensor {name:?} has conflicting scale sidecars: {}",
            present_scale_names.join(", ")
        )));
    }
    if matches!(tensor.dtype, DType::F16 | DType::Bf16 | DType::F32)
        && scale.is_none()
        && mx_scale.is_none()
        && block_fp8_scale.is_none()
    {
        if tensor.shape != [rows as u64, cols as u64] || tensor.declared_elements != logical as u64
        {
            return Err(WeightLoadError::InvalidShape(format!(
                "plain tensor {name:?} must be [{rows},{cols}]"
            )));
        }
        return Ok(WeightLayout {
            format: WeightFormat::F32,
            resident_bytes: (logical as u64).saturating_mul(4),
        });
    }
    if tensor.dtype == DType::F8E4M3 {
        if let Some(block_fp8_scale) = block_fp8_scale {
            let expected_scale_shape = [rows.div_ceil(128) as u64, cols.div_ceil(128) as u64];
            let expected_scale_elements = expected_scale_shape[0]
                .checked_mul(expected_scale_shape[1])
                .ok_or_else(|| {
                    WeightLoadError::InvalidShape(
                        "block-FP8 scale element count overflows u64".to_owned(),
                    )
                })?;
            let expected_scale_bytes = expected_scale_elements.checked_mul(2).ok_or_else(|| {
                WeightLoadError::InvalidShape(
                    "block-FP8 scale payload byte count overflows u64".to_owned(),
                )
            })?;
            if tensor.shape != [rows as u64, cols as u64]
                || tensor.data_len != logical as u64
                || block_fp8_scale.dtype != DType::Bf16
                || block_fp8_scale.shape != expected_scale_shape
                || block_fp8_scale.data_len != expected_scale_bytes
            {
                return Err(WeightLoadError::InvalidShape(format!(
                    "block-FP8 tensor {name:?} needs E4M3 weight [{rows},{cols}] and BF16 scale_inv {expected_scale_shape:?}"
                )));
            }
            let decoded_scale_bytes = expected_scale_elements.checked_mul(4).ok_or_else(|| {
                WeightLoadError::InvalidShape(
                    "decoded block-FP8 scale byte count overflows u64".to_owned(),
                )
            })?;
            let resident_bytes = tensor
                .data_len
                .checked_add(decoded_scale_bytes)
                .ok_or_else(|| {
                    WeightLoadError::InvalidShape(
                        "block-FP8 resident byte count overflows u64".to_owned(),
                    )
                })?;
            return Ok(WeightLayout {
                format: WeightFormat::BlockFp8E4M3Bf16ScaleInv {
                    block_rows: 128,
                    block_cols: 128,
                },
                resident_bytes,
            });
        }
        if scale.is_some() {
            return Err(WeightLoadError::Unsupported(format!(
                "E4M3 tensor {name:?} cannot use a Colibri F32 sidecar {scale_name:?}"
            )));
        }
        let mx_scale = require_mx_scale(mx_scale, &mx_scale_name)?;
        let deepseek_v41_shape = [rows.div_ceil(32) as u64, cols.div_ceil(32) as u64];
        let deepseek_shape = [rows.div_ceil(128) as u64, cols.div_ceil(128) as u64];
        let modelopt_shape = [rows as u64, cols.div_ceil(32) as u64];
        let (block_rows, block_cols) = if mx_scale.dtype == DType::F8E8M0
            && mx_scale.shape == deepseek_v41_shape
        {
            (32, 32)
        } else if mx_scale.dtype == DType::F8E8M0 && mx_scale.shape == deepseek_shape {
            (128, 128)
        } else if mx_scale.dtype == DType::U8 && mx_scale.shape == modelopt_shape {
            (1, 32)
        } else {
            return Err(WeightLoadError::InvalidShape(format!(
                "MXFP8 tensor {name:?} needs E8M0 scale {deepseek_v41_shape:?} (32x32), E8M0 scale {deepseek_shape:?} (128x128), or U8 E8M0 scale {modelopt_shape:?} (1x32), got {} {:?}",
                mx_scale.dtype, mx_scale.shape
            )));
        };
        if tensor.shape != [rows as u64, cols as u64] || tensor.data_len != logical as u64 {
            return Err(WeightLoadError::InvalidShape(format!(
                "MXFP8 tensor {name:?} must be [{rows},{cols}]"
            )));
        }
        return Ok(WeightLayout {
            format: WeightFormat::MxFp8E4M3 {
                block_rows,
                block_cols,
            },
            resident_bytes: tensor.data_len.saturating_add(mx_scale.data_len),
        });
    }
    if block_fp8_scale.is_some() {
        return Err(WeightLoadError::Unsupported(format!(
            "BF16 scale_inv sidecar {block_fp8_scale_name:?} requires an F8_E4M3 weight, got {}",
            tensor.dtype
        )));
    }
    let is_native_mxfp4 = tensor.dtype == DType::I8
        || (tensor.dtype == DType::U8 && name.ends_with(".weight_packed"));
    if is_native_mxfp4 && mx_scale.is_some() {
        let mx_scale = require_mx_scale(mx_scale, &mx_scale_name)?;
        let physical_cols = cols.div_ceil(2);
        let expected_scale_shape = [rows as u64, cols.div_ceil(32) as u64];
        let expected_scale_dtype = if tensor.dtype == DType::U8 {
            DType::U8
        } else {
            DType::F8E8M0
        };
        if tensor.shape != [rows as u64, physical_cols as u64]
            || tensor.data_len != rows.saturating_mul(physical_cols) as u64
            || mx_scale.dtype != expected_scale_dtype
            || mx_scale.shape != expected_scale_shape
        {
            return Err(WeightLoadError::InvalidShape(format!(
                "MXFP4 tensor {name:?} needs packed weight [{rows},{physical_cols}] and {expected_scale_dtype} E8M0 scale {expected_scale_shape:?}"
            )));
        }
        return Ok(WeightLayout {
            format: WeightFormat::MxFp4E2M1 { group_size: 32 },
            resident_bytes: tensor.data_len.saturating_add(mx_scale.data_len),
        });
    }
    if tensor.dtype != DType::U8 {
        return Err(WeightLoadError::Unsupported(format!(
        "tensor {name:?} has dtype {}, expected a supported plain, Colibri, MXFP8, block-FP8, or MXFP4 matrix",
            tensor.dtype
        )));
    }
    let scale = scale.ok_or_else(|| {
        WeightLoadError::Unsupported(format!(
            "packed tensor {name:?} has no required F32 sidecar {scale_name:?}"
        ))
    })?;
    if scale.dtype != DType::F32 {
        return Err(WeightLoadError::Unsupported(format!(
            "scale sidecar {scale_name:?} has dtype {}, expected F32",
            scale.dtype
        )));
    }
    let resident_bytes = tensor
        .data_len
        .checked_add(scale.data_len)
        .ok_or_else(|| WeightLoadError::InvalidShape("resident byte count overflows".to_owned()))?;
    let bits = infer_unique_packed_bits(rows as u64, cols as u64, tensor.data_len, &[4, 8])
        .map_err(|reason| {
            WeightLoadError::InvalidShape(format!("packed tensor {name:?}: {reason}"))
        })?;
    let group_size = validate_quantized_scale_layout(bits, rows as u64, cols as u64, scale)
        .map_err(|reason| {
            WeightLoadError::InvalidShape(format!("scale sidecar {scale_name:?}: {reason}"))
        })?;
    match bits {
        8 => Ok(WeightLayout {
            format: WeightFormat::Int8PerRow,
            resident_bytes,
        }),
        4 => Ok(WeightLayout {
            format: WeightFormat::Int4 {
                group_size: usize::try_from(group_size.expect("INT4 layout has a group size"))
                    .map_err(|_| {
                        WeightLoadError::InvalidShape(
                            "INT4 group size does not fit usize".to_owned(),
                        )
                    })?,
            },
            resident_bytes,
        }),
        _ => unreachable!("runtime inference requested only INT4/INT8"),
    }
}

/// Validates a plain BF16 `[rows, cols]` matrix for compact resident loading.
///
/// This is intentionally separate from [`inspect_weight_matrix`], whose established plain
/// F16/BF16/F32 contract accounts for widening the payload to resident F32 values.
pub fn inspect_compact_bf16_matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightLayout, WeightLoadError> {
    if rows == 0 || cols == 0 {
        return Err(WeightLoadError::InvalidShape(
            "rows and columns must be non-zero".to_owned(),
        ));
    }
    let logical = rows
        .checked_mul(cols)
        .ok_or_else(|| WeightLoadError::InvalidShape("rows * cols overflows usize".to_owned()))?;
    let logical = u64::try_from(logical).map_err(|_| {
        WeightLoadError::InvalidShape("matrix element count does not fit u64".to_owned())
    })?;
    let resident_bytes = logical.checked_mul(2).ok_or_else(|| {
        WeightLoadError::InvalidShape("BF16 resident byte count overflows".to_owned())
    })?;
    let tensor = index.require(name)?;
    if tensor.dtype != DType::Bf16 {
        return Err(WeightLoadError::Unsupported(format!(
            "compact tensor {name:?} has dtype {}, expected BF16",
            tensor.dtype
        )));
    }
    if tensor.shape != [rows as u64, cols as u64]
        || tensor.declared_elements != logical
        || tensor.data_len != resident_bytes
    {
        return Err(WeightLoadError::InvalidShape(format!(
            "compact BF16 tensor {name:?} must be [{rows},{cols}] with {resident_bytes} payload bytes"
        )));
    }
    Ok(WeightLayout {
        format: WeightFormat::Bf16,
        resident_bytes,
    })
}

/// Loads a complete plain BF16 matrix without widening it to resident F32 values.
///
/// The complete resident allocation is checked against `maximum_resident_bytes` before payload
/// I/O. The underlying tensor reader then fills that allocation using reads of at most 64 MiB.
pub fn load_compact_bf16_matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    cols: usize,
    maximum_resident_bytes: u64,
) -> Result<WeightMatrix, WeightLoadError> {
    let layout = inspect_compact_bf16_matrix(index, name, rows, cols)?;
    check_budget(name, layout.resident_bytes, maximum_resident_bytes)?;
    let bytes = index.read_tensor_bounded(name, layout.resident_bytes)?;
    Ok(WeightMatrix::Bf16(Bf16Matrix::from_le_bytes(
        rows, cols, bytes,
    )?))
}

/// Loads a related group of plain BF16 matrices without widening them to resident F32 values.
///
/// Every matrix is validated from metadata and the aggregate resident allocation is authorized
/// before any payload I/O. Payloads are then read in physical shard/offset order while results
/// retain the caller's order. An empty request succeeds with an empty result and performs no I/O,
/// including when `maximum_resident_bytes` is zero.
pub fn load_compact_bf16_matrices(
    index: &TensorIndex,
    matrices: &[(&str, usize, usize)],
    maximum_resident_bytes: u64,
) -> Result<Vec<WeightMatrix>, WeightLoadError> {
    if matrices.is_empty() {
        return Ok(Vec::new());
    }

    let mut total_resident = 0u64;
    let mut prepared = Vec::with_capacity(matrices.len());
    for &(name, rows, cols) in matrices {
        let layout = inspect_compact_bf16_matrix(index, name, rows, cols)?;
        total_resident = total_resident
            .checked_add(layout.resident_bytes)
            .ok_or_else(|| {
                WeightLoadError::InvalidShape(
                    "compact BF16 resident byte count overflows".to_owned(),
                )
            })?;
        prepared.push((name, rows, cols));
    }

    check_budget(matrices[0].0, total_resident, maximum_resident_bytes)?;

    let names = prepared
        .iter()
        .map(|(name, _, _)| *name)
        .collect::<Vec<_>>();
    let payloads = index.read_tensors_bounded(&names, total_resident)?;
    prepared
        .into_iter()
        .zip(payloads)
        .map(|((_, rows, cols), payload)| {
            Ok(WeightMatrix::Bf16(Bf16Matrix::from_le_bytes(
                rows, cols, payload,
            )?))
        })
        .collect()
}

/// Loads a Colibrì packed matrix, or a small plain-F32 matrix, without requantizing it.
/// The caller must authorize the complete resident allocation before payload I/O starts.
pub fn load_weight_matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    cols: usize,
    maximum_resident_bytes: u64,
) -> Result<WeightMatrix, WeightLoadError> {
    load_weight_matrices(index, &[(name, rows, cols)], maximum_resident_bytes).map(
        |mut matrices| {
            matrices
                .pop()
                .expect("a one-element weight request returns one matrix")
        },
    )
}

#[derive(Debug)]
struct PreparedWeight {
    name: String,
    rows: usize,
    cols: usize,
    dtype: DType,
    layout: WeightLayout,
    payload_index: usize,
    scale: Option<(String, usize)>,
}

/// Loads a related group of matrices after sorting all payload and scale reads by shard offset.
///
/// The returned order matches `matrices`. The aggregate resident allocation is authorized before
/// any payload I/O, so expert loaders can safely fetch their gate/up/down projections as one batch.
pub fn load_weight_matrices(
    index: &TensorIndex,
    matrices: &[(&str, usize, usize)],
    maximum_resident_bytes: u64,
) -> Result<Vec<WeightMatrix>, WeightLoadError> {
    let mut total_resident = 0u64;
    let mut read_names = Vec::with_capacity(matrices.len().saturating_mul(2));
    let mut prepared = Vec::with_capacity(matrices.len());
    for &(name, rows, cols) in matrices {
        let layout = inspect_weight_matrix(index, name, rows, cols)?;
        total_resident = total_resident
            .checked_add(layout.resident_bytes)
            .ok_or_else(|| {
                WeightLoadError::InvalidShape("resident byte count overflows".to_owned())
            })?;
        let tensor = index.require(name)?;
        let payload_index = read_names.len();
        read_names.push(name.to_owned());
        let scale_name = match layout.format {
            WeightFormat::F32 | WeightFormat::Bf16 => None,
            WeightFormat::Int8PerRow | WeightFormat::Int4 { .. } => Some(format!("{name}.qs")),
            WeightFormat::MxFp8E4M3 { .. } | WeightFormat::MxFp4E2M1 { .. } => {
                Some(find_native_scale_name(index, name))
            }
            WeightFormat::BlockFp8E4M3Bf16ScaleInv { .. } => Some(block_fp8_scale_inv_name(name)),
        };
        let scale = scale_name.map(|scale_name| {
            let index = read_names.len();
            read_names.push(scale_name.clone());
            (scale_name, index)
        });
        prepared.push(PreparedWeight {
            name: name.to_owned(),
            rows,
            cols,
            dtype: tensor.dtype.clone(),
            layout,
            payload_index,
            scale,
        });
    }

    let budget_name = matrices
        .first()
        .map(|(name, _, _)| *name)
        .unwrap_or("empty weight batch");
    check_budget(budget_name, total_resident, maximum_resident_bytes)?;

    let total_payload = read_names.iter().try_fold(0u64, |total, name| {
        total
            .checked_add(index.require(name)?.data_len)
            .ok_or_else(|| WeightLoadError::InvalidShape("payload byte count overflows".to_owned()))
    })?;
    let read_refs = read_names.iter().map(String::as_str).collect::<Vec<_>>();
    let mut payloads = index
        .read_tensors_bounded(&read_refs, total_payload)?
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();

    prepared
        .into_iter()
        .map(|weight| {
            let payload = payloads[weight.payload_index]
                .take()
                .expect("each prepared payload has one owner");
            let scale = weight.scale.as_ref().map(|(_, index)| {
                payloads[*index]
                    .take()
                    .expect("each prepared scale has one owner")
            });
            finish_weight_matrix(weight, payload, scale)
        })
        .collect()
}

fn finish_weight_matrix(
    weight: PreparedWeight,
    payload: Vec<u8>,
    scale: Option<Vec<u8>>,
) -> Result<WeightMatrix, WeightLoadError> {
    let PreparedWeight {
        name,
        rows,
        cols,
        dtype,
        layout,
        scale: scale_info,
        ..
    } = weight;
    match layout.format {
        WeightFormat::F32 => {
            let values = decode_reference_values(&name, &dtype, &payload)?;
            Ok(WeightMatrix::F32(DenseMatrix::new(rows, cols, values)?))
        }
        WeightFormat::Bf16 => Ok(WeightMatrix::Bf16(Bf16Matrix::from_le_bytes(
            rows, cols, payload,
        )?)),
        WeightFormat::MxFp8E4M3 {
            block_rows,
            block_cols,
        } => Ok(WeightMatrix::MxFp8(MxFp8Matrix::from_packed(
            rows,
            cols,
            block_rows,
            block_cols,
            payload,
            scale.expect("MXFP8 preparation includes its scale"),
        )?)),
        WeightFormat::BlockFp8E4M3Bf16ScaleInv {
            block_rows,
            block_cols,
        } => Ok(WeightMatrix::MxFp8(
            MxFp8Matrix::from_packed_bf16_scale_inv(
                rows,
                cols,
                block_rows,
                block_cols,
                payload,
                scale.expect("block-FP8 preparation includes its BF16 scale_inv"),
            )?,
        )),
        WeightFormat::MxFp4E2M1 { group_size } => {
            Ok(WeightMatrix::MxFp4(MxFp4Matrix::from_packed(
                rows,
                cols,
                group_size,
                payload,
                scale.expect("MXFP4 preparation includes its scale"),
            )?))
        }
        WeightFormat::Int8PerRow | WeightFormat::Int4 { .. } => {
            let (scale_name, _) = scale_info.expect("Colibri preparation includes its scale");
            let scales = decode_reference_values(
                &scale_name,
                &DType::F32,
                &scale.expect("Colibri preparation includes its scale bytes"),
            )?;
            if layout.format == WeightFormat::Int8PerRow {
                if scales.len() != rows {
                    return Err(WeightLoadError::InvalidShape(format!(
                        "INT8 tensor {name:?} needs {rows} row scales, got {}",
                        scales.len()
                    )));
                }
                Ok(WeightMatrix::Int8PerRow(Int8Matrix::from_packed(
                    rows, cols, payload, scales,
                )?))
            } else if let WeightFormat::Int4 { group_size } = layout.format {
                Ok(WeightMatrix::Int4(Int4Matrix::from_packed(
                    rows, cols, group_size, payload, scales,
                )?))
            } else {
                unreachable!("Colibri branch contains only INT4/INT8")
            }
        }
    }
}

fn native_scale_name(name: &str) -> String {
    if let Some(prefix) = name.strip_suffix(".weight_packed") {
        return format!("{prefix}.weight_scale");
    }
    name.strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.scale"))
        .unwrap_or_else(|| format!("{name}.scale"))
}

fn find_native_scale_name(index: &TensorIndex, name: &str) -> String {
    if let Some(prefix) = name.strip_suffix(".weight") {
        let modelopt = format!("{prefix}.weight_scale");
        if index.get(&modelopt).is_some() {
            return modelopt;
        }
    }
    native_scale_name(name)
}

fn block_fp8_scale_inv_name(name: &str) -> String {
    name.strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.weight_scale_inv"))
        .unwrap_or_else(|| format!("{name}_scale_inv"))
}

fn require_mx_scale<'a>(
    scale: Option<&'a TensorInfo>,
    name: &str,
) -> Result<&'a TensorInfo, WeightLoadError> {
    scale.ok_or_else(|| {
        WeightLoadError::Unsupported(format!("MX tensor has no required E8M0 sidecar {name:?}"))
    })
}

fn check_budget(name: &str, required: u64, maximum: u64) -> Result<(), WeightLoadError> {
    if required > maximum {
        Err(WeightLoadError::Budget {
            name: name.to_owned(),
            required,
            maximum,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn infer_int4_group_size(rows: u64, cols: u64, scale_count: u64) -> Result<u64, String> {
    if rows == 0 || cols == 0 || scale_count == 0 {
        return Err("rows, columns, and scale count must be non-zero".to_owned());
    }
    if scale_count == rows {
        return Ok(cols);
    }
    const CANDIDATES: [u64; 8] = [16, 32, 48, 64, 96, 128, 192, 256];
    let candidates = CANDIDATES
        .into_iter()
        .take_while(|group| *group <= cols)
        .filter(|group| rows.checked_mul(cols.div_ceil(*group)) == Some(scale_count))
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [group_size] => Ok(*group_size),
        [] => Err("scale count is incompatible with every supported group size".to_owned()),
        _ => Err(format!(
            "group size is ambiguous among {candidates:?}; store an unambiguous canonical layout"
        )),
    }
}

pub(crate) fn infer_unique_packed_bits(
    rows: u64,
    cols: u64,
    payload_bytes: u64,
    supported_bits: &[u8],
) -> Result<u8, String> {
    if rows == 0 || cols == 0 {
        return Err("logical rows and columns must be non-zero".to_owned());
    }
    let mut matches = Vec::new();
    let mut expected = Vec::new();
    for &bits in supported_bits {
        let row_bytes = match bits {
            2 => cols.div_ceil(4),
            4 => cols.div_ceil(2),
            8 => cols,
            _ => return Err(format!("unsupported packed bit width {bits}")),
        };
        let bytes = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("INT{bits} payload size overflows u64"))?;
        expected.push(format!("INT{bits}={bytes}"));
        if bytes == payload_bytes {
            matches.push(bits);
        }
    }
    match matches.as_slice() {
        [bits] => Ok(*bits),
        [] => Err(format!(
            "payload has {payload_bytes} bytes; expected {}",
            expected.join(", ")
        )),
        _ => Err(format!(
            "payload has {payload_bytes} bytes and is ambiguous among {matches:?}; explicit format metadata is required"
        )),
    }
}

pub(crate) fn validate_quantized_scale_layout(
    bits: u8,
    rows: u64,
    cols: u64,
    scale: &TensorInfo,
) -> Result<Option<u64>, String> {
    let group_size = if bits == 4 {
        Some(infer_int4_group_size(rows, cols, scale.declared_elements)?)
    } else if matches!(bits, 2 | 8) {
        if scale.declared_elements != rows {
            return Err(format!(
                "INT{bits} needs {rows} per-row scales, got {}",
                scale.declared_elements
            ));
        }
        None
    } else {
        return Err(format!("unsupported quantized bit width {bits}"));
    };
    let expected_shape = match group_size {
        Some(group) if group != cols => vec![rows, cols.div_ceil(group)],
        _ => vec![rows],
    };
    if scale.shape != expected_shape {
        return Err(format!(
            "canonical INT{bits} scale shape is {expected_shape:?}, got {:?}",
            scale.shape
        ));
    }
    Ok(group_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn fixture_dir() -> PathBuf {
        crate::test_support::temp_dir("urbilateria_weight_loader")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn write_safetensor_fixture(path: &Path, tensors: Vec<(&str, &str, Vec<u64>, Vec<u8>)>) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut offset = 0u64;
        for (name, dtype, shape, bytes) in tensors {
            let end = offset.checked_add(bytes.len() as u64).unwrap();
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            payload.extend(bytes);
            offset = end;
        }
        let mut header = serde_json::Value::Object(header).to_string().into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(payload);
        fs::write(path, output).unwrap();
    }

    #[test]
    fn grouped_scale_geometry_must_identify_one_group_size() {
        assert_eq!(infer_int4_group_size(2, 6144, 192).unwrap(), 64);
        let error = infer_int4_group_size(1, 130, 3).unwrap_err();
        assert!(error.contains("ambiguous"));
        assert!(infer_int4_group_size(2, 130, 5).is_err());
    }

    #[test]
    fn packed_width_and_scale_shape_must_be_unambiguous() {
        assert_eq!(infer_unique_packed_bits(2, 2, 2, &[4, 8]).unwrap(), 4);
        assert!(infer_unique_packed_bits(2, 1, 2, &[4, 8])
            .unwrap_err()
            .contains("ambiguous"));

        let scale = |shape: Vec<u64>| TensorInfo {
            name: "w.qs".to_owned(),
            dtype: DType::F32,
            declared_elements: shape.iter().product(),
            data_len: shape.iter().product::<u64>() * 4,
            shape,
            shard: PathBuf::from("fixture"),
            data_offset: 0,
        };
        validate_quantized_scale_layout(8, 2, 2, &scale(vec![2])).unwrap();
        assert_eq!(
            validate_quantized_scale_layout(4, 2, 6144, &scale(vec![2, 96])).unwrap(),
            Some(64)
        );
        assert!(validate_quantized_scale_layout(4, 2, 6144, &scale(vec![192])).is_err());
    }

    fn write_fixture(path: &Path) {
        let i8 = vec![0xff, 2, 3, 0xfc];
        let i8_scale = f32_bytes(&[1.0, 1.0]);
        let i4 = vec![0xa7, 0x4b];
        let i4_scale = f32_bytes(&[1.0, 1.0]);
        let dense = f32_bytes(&[-1.0, 2.0, 3.0, -4.0]);
        let bf16 = [-1.0f32, 2.0, 3.0, -4.0]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let mut header = serde_json::json!({
            "i8": {"dtype":"U8", "shape":[4], "data_offsets":[0,4]},
            "i8.qs": {"dtype":"F32", "shape":[2], "data_offsets":[4,12]},
            "i4": {"dtype":"U8", "shape":[2], "data_offsets":[12,14]},
            "i4.qs": {"dtype":"F32", "shape":[2], "data_offsets":[14,22]},
            "dense": {"dtype":"F32", "shape":[2,2], "data_offsets":[22,38]},
            "bf16": {"dtype":"BF16", "shape":[2,2], "data_offsets":[38,46]}
        })
        .to_string()
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(i8);
        output.extend(i8_scale);
        output.extend(i4);
        output.extend(i4_scale);
        output.extend(dense);
        output.extend(bf16);
        fs::write(path, output).unwrap();
    }

    fn write_native_mxfp4_fixture(path: &Path) {
        let packed = [0x21, 0x48];
        let scales = [127, 127];
        let mut header = serde_json::json!({
            "expert.w1.weight_packed": {
                "dtype":"U8", "shape":[2,1], "data_offsets":[0,2]
            },
            "expert.w1.weight_scale": {
                "dtype":"U8", "shape":[2,1], "data_offsets":[2,4]
            }
        })
        .to_string()
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(packed);
        output.extend(scales);
        fs::write(path, output).unwrap();
    }

    fn write_qwen_block_fp8_fixture(path: &Path) {
        write_safetensor_fixture(
            path,
            vec![
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "F8_E4M3",
                    vec![129, 129],
                    vec![0x38; 129 * 129],
                ),
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
                    "BF16",
                    vec![2, 2],
                    bf16_bytes(&[1.0, 2.0, 4.0, 8.0]),
                ),
            ],
        );
    }

    fn write_modelopt_mxfp8_fixture(path: &Path) {
        write_safetensor_fixture(
            path,
            vec![
                (
                    "layer.q_proj.weight",
                    "F8_E4M3",
                    vec![2, 33],
                    vec![0x38; 2 * 33],
                ),
                (
                    "layer.q_proj.weight_scale",
                    "U8",
                    vec![2, 2],
                    vec![127, 128, 129, 130],
                ),
            ],
        );
    }

    fn write_compact_bf16_batch_fixture(path: &Path) {
        let physical_first = [1.0f32, 2.0]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let not_bf16 = 7.0f32.to_le_bytes();
        let physical_second = [-1.0f32, 2.0, 3.0, -4.0]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let wrong_shape = [8.0f32, 9.0]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let mut header = serde_json::json!({
            "physical_first": {
                "dtype":"BF16", "shape":[1,2], "data_offsets":[0,4]
            },
            "not_bf16": {
                "dtype":"F32", "shape":[1,1], "data_offsets":[4,8]
            },
            "physical_second": {
                "dtype":"BF16", "shape":[2,2], "data_offsets":[8,16]
            },
            "wrong_shape": {
                "dtype":"BF16", "shape":[1,2], "data_offsets":[16,20]
            }
        })
        .to_string()
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(physical_first);
        output.extend(not_bf16);
        output.extend(physical_second);
        output.extend(wrong_shape);
        fs::write(path, output).unwrap();
    }

    #[test]
    fn loads_mixed_container_without_requantizing() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        for name in ["i8", "i4", "dense"] {
            let matrix = load_weight_matrix(&index, name, 2, 2, 64).unwrap();
            assert_eq!(matrix.row(0).unwrap(), vec![-1.0, 2.0]);
            assert_eq!(matrix.matvec(&[0.5, 2.0]).unwrap(), vec![3.5, -6.5]);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_related_weights_as_one_order_preserving_batch() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let matrices =
            load_weight_matrices(&index, &[("dense", 2, 2), ("i4", 2, 2), ("i8", 2, 2)], 38)
                .unwrap();
        for matrix in &matrices {
            assert_eq!(matrix.row(0).unwrap(), vec![-1.0, 2.0]);
            assert_eq!(matrix.matvec(&[0.5, 2.0]).unwrap(), vec![3.5, -6.5]);
        }
        assert!(matches!(
            load_weight_matrices(&index, &[("dense", 2, 2), ("i4", 2, 2), ("i8", 2, 2)], 37,),
            Err(WeightLoadError::Budget {
                required: 38,
                maximum: 37,
                ..
            })
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_kimi_native_mxfp4_names_without_widening_scales() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_native_mxfp4_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let name = "expert.w1.weight_packed";
        let layout = inspect_weight_matrix(&index, name, 2, 2).unwrap();
        assert_eq!(
            layout,
            WeightLayout {
                format: WeightFormat::MxFp4E2M1 { group_size: 32 },
                resident_bytes: 4,
            }
        );
        let matrix = load_weight_matrix(&index, name, 2, 2, 4).unwrap();
        assert_eq!(matrix.row(0).unwrap(), vec![0.5, 1.0]);
        assert_eq!(matrix.matvec(&[2.0, 3.0]).unwrap(), vec![4.0, 6.0]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_modelopt_mxfp8_with_u8_one_by_32_scales() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_modelopt_mxfp8_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let name = "layer.q_proj.weight";
        let layout = inspect_weight_matrix(&index, name, 2, 33).unwrap();
        assert_eq!(
            layout,
            WeightLayout {
                format: WeightFormat::MxFp8E4M3 {
                    block_rows: 1,
                    block_cols: 32,
                },
                resident_bytes: 70,
            }
        );
        let matrix = load_weight_matrix(&index, name, 2, 33, 70).unwrap();
        assert_eq!(matrix.matvec(&[1.0; 33]).unwrap(), [34.0, 136.0]);
        assert_eq!(matrix.resident_bytes(), 70);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_qwen_block_fp8_with_bf16_scale_inv_on_both_axes() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_qwen_block_fp8_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let name = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let layout = inspect_weight_matrix(&index, name, 129, 129).unwrap();
        assert_eq!(
            layout,
            WeightLayout {
                format: WeightFormat::BlockFp8E4M3Bf16ScaleInv {
                    block_rows: 128,
                    block_cols: 128,
                },
                resident_bytes: (129 * 129 + 4 * 4) as u64,
            }
        );

        let matrix = load_weight_matrix(&index, name, 129, 129, layout.resident_bytes).unwrap();
        assert!(matches!(&matrix, WeightMatrix::MxFp8(_)));
        let first = matrix.row(0).unwrap();
        let second_block_row = matrix.row(128).unwrap();
        assert_eq!(&first[..2], &[1.0, 1.0]);
        assert_eq!(first[128], 2.0);
        assert_eq!(&second_block_row[..2], &[4.0, 4.0]);
        assert_eq!(second_block_row[128], 8.0);
        assert_eq!(
            matrix.matvec(&vec![1.0; 129]).unwrap(),
            [vec![130.0; 128], vec![520.0]].concat()
        );
        assert_eq!(matrix.resident_bytes() as u64, layout.resident_bytes);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn qwen_block_fp8_rejects_dtype_shape_and_sidecar_conflicts() {
        let root = fixture_dir();
        fs::create_dir_all(&root).unwrap();
        let weight = "expert.weight";
        let scale_inv = "expert.weight_scale_inv";

        let wrong_dtype = root.join("wrong_dtype");
        fs::create_dir_all(&wrong_dtype).unwrap();
        write_safetensor_fixture(
            &wrong_dtype.join("weights.safetensors"),
            vec![
                (weight, "F8_E4M3", vec![1, 1], vec![0x38]),
                (scale_inv, "F32", vec![1, 1], f32_bytes(&[1.0])),
            ],
        );
        let index = TensorIndex::open(&wrong_dtype).unwrap();
        assert!(matches!(
            inspect_weight_matrix(&index, weight, 1, 1),
            Err(WeightLoadError::InvalidShape(reason)) if reason.contains("BF16 scale_inv")
        ));

        let wrong_shape = root.join("wrong_shape");
        fs::create_dir_all(&wrong_shape).unwrap();
        write_safetensor_fixture(
            &wrong_shape.join("weights.safetensors"),
            vec![
                (weight, "F8_E4M3", vec![1, 129], vec![0x38; 129]),
                (scale_inv, "BF16", vec![1, 1], bf16_bytes(&[1.0])),
            ],
        );
        let index = TensorIndex::open(&wrong_shape).unwrap();
        assert!(matches!(
            inspect_weight_matrix(&index, weight, 1, 129),
            Err(WeightLoadError::InvalidShape(reason)) if reason.contains("[1, 2]")
        ));

        let conflict = root.join("conflict");
        fs::create_dir_all(&conflict).unwrap();
        write_safetensor_fixture(
            &conflict.join("weights.safetensors"),
            vec![
                (weight, "F8_E4M3", vec![1, 1], vec![0x38]),
                ("expert.scale", "F8_E8M0", vec![1, 1], vec![127]),
                (scale_inv, "BF16", vec![1, 1], bf16_bytes(&[1.0])),
            ],
        );
        let index = TensorIndex::open(&conflict).unwrap();
        assert!(matches!(
            inspect_weight_matrix(&index, weight, 1, 1),
            Err(WeightLoadError::Unsupported(reason)) if reason.contains("conflicting scale sidecars")
        ));

        let wrong_weight_dtype = root.join("wrong_weight_dtype");
        fs::create_dir_all(&wrong_weight_dtype).unwrap();
        write_safetensor_fixture(
            &wrong_weight_dtype.join("weights.safetensors"),
            vec![
                (weight, "BF16", vec![1, 1], bf16_bytes(&[1.0])),
                (scale_inv, "BF16", vec![1, 1], bf16_bytes(&[1.0])),
            ],
        );
        let index = TensorIndex::open(&wrong_weight_dtype).unwrap();
        assert!(matches!(
            inspect_weight_matrix(&index, weight, 1, 1),
            Err(WeightLoadError::Unsupported(reason)) if reason.contains("requires an F8_E4M3 weight")
        ));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn qwen_block_fp8_budget_is_checked_before_payload_io() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("weights.safetensors");
        write_qwen_block_fp8_fixture(&shard);
        let index = TensorIndex::open(&dir).unwrap();
        let name = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let layout = inspect_weight_matrix(&index, name, 129, 129).unwrap();

        // Poison the indexed payload: an early read would now fail with EOF instead of Budget.
        fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(matches!(
            load_weight_matrix(&index, name, 129, 129, layout.resident_bytes - 1),
            Err(WeightLoadError::Budget {
                required,
                maximum,
                ..
            }) if required == layout.resident_bytes && maximum + 1 == required
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn allocation_budget_is_enforced_before_payload_read() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        assert!(matches!(
            load_weight_matrix(&index, "i8", 2, 2, 11),
            Err(WeightLoadError::Budget {
                required: 12,
                maximum: 11,
                ..
            })
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compact_bf16_loader_preserves_payload_residency_and_old_plain_contract() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();

        let compact = inspect_compact_bf16_matrix(&index, "bf16", 2, 2).unwrap();
        assert_eq!(
            compact,
            WeightLayout {
                format: WeightFormat::Bf16,
                resident_bytes: 8,
            }
        );
        assert!(matches!(
            load_compact_bf16_matrix(&index, "bf16", 2, 2, 7),
            Err(WeightLoadError::Budget {
                required: 8,
                maximum: 7,
                ..
            })
        ));

        let matrix = load_compact_bf16_matrix(&index, "bf16", 2, 2, 8).unwrap();
        assert!(matches!(&matrix, WeightMatrix::Bf16(_)));
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.resident_bytes(), 8);
        assert_eq!(matrix.row(0).unwrap(), vec![-1.0, 2.0]);
        assert_eq!(matrix.matvec(&[0.5, 2.0]).unwrap(), vec![3.5, -6.5]);
        assert_eq!(matrix.matvec_rows(1, 1, &[0.5, 2.0]).unwrap(), vec![-6.5]);
        assert_eq!(
            matrix.transpose_rows_matvec(1, &[2.0]).unwrap(),
            vec![6.0, -8.0]
        );

        let widened = inspect_weight_matrix(&index, "bf16", 2, 2).unwrap();
        assert_eq!(
            widened,
            WeightLayout {
                format: WeightFormat::F32,
                resident_bytes: 16,
            }
        );
        assert!(matches!(
            load_weight_matrix(&index, "bf16", 2, 2, 15),
            Err(WeightLoadError::Budget {
                required: 16,
                maximum: 15,
                ..
            })
        ));
        let widened = load_weight_matrix(&index, "bf16", 2, 2, 16).unwrap();
        assert!(matches!(&widened, WeightMatrix::F32(_)));
        assert_eq!(widened.resident_bytes(), 16);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compact_bf16_batch_returns_caller_order_without_widening() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_compact_bf16_batch_fixture(&dir.join("weights.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();

        let matrices = load_compact_bf16_matrices(
            &index,
            &[("physical_second", 2, 2), ("physical_first", 1, 2)],
            12,
        )
        .unwrap();
        assert_eq!(matrices.len(), 2);
        assert!(matrices
            .iter()
            .all(|matrix| matches!(matrix, WeightMatrix::Bf16(_))));
        assert_eq!(matrices[0].resident_bytes(), 8);
        assert_eq!(matrices[0].row(0).unwrap(), vec![-1.0, 2.0]);
        assert_eq!(matrices[0].matvec(&[0.5, 2.0]).unwrap(), vec![3.5, -6.5]);
        assert_eq!(matrices[1].resident_bytes(), 4);
        assert_eq!(matrices[1].row(0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(matrices[1].matvec(&[0.5, 2.0]).unwrap(), vec![4.5]);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compact_bf16_batch_checks_metadata_and_aggregate_budget_before_io() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("weights.safetensors");
        write_compact_bf16_batch_fixture(&shard);
        let index = TensorIndex::open(&dir).unwrap();

        assert!(matches!(
            load_compact_bf16_matrices(&index, &[("physical_first", 1, 2), ("not_bf16", 1, 1)], 0,),
            Err(WeightLoadError::Unsupported(_))
        ));
        assert!(matches!(
            load_compact_bf16_matrices(
                &index,
                &[("physical_first", 1, 2), ("wrong_shape", 2, 1)],
                u64::MAX,
            ),
            Err(WeightLoadError::InvalidShape(_))
        ));

        // Poison the indexed payload: a premature read would now return EOF instead of Budget.
        fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(matches!(
            load_compact_bf16_matrices(
                &index,
                &[("physical_second", 2, 2), ("physical_first", 1, 2)],
                11,
            ),
            Err(WeightLoadError::Budget {
                required: 12,
                maximum: 11,
                ..
            })
        ));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn empty_compact_bf16_batch_succeeds_with_zero_budget() {
        let missing = fixture_dir();
        fs::create_dir_all(&missing).unwrap();
        write_compact_bf16_batch_fixture(&missing.join("weights.safetensors"));
        let index = TensorIndex::open(&missing).unwrap();
        assert!(load_compact_bf16_matrices(&index, &[], 0)
            .unwrap()
            .is_empty());
        fs::remove_dir_all(missing).unwrap();
    }
}
