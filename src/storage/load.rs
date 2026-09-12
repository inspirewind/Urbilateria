use super::{DType, SafetensorError, TensorIndex};
use crate::execution::{install, should_parallelize};
use crate::model::{Bf16Matrix, DenseMatrix, MatrixError};
use crate::profiling::{span_with_metrics, span_with_work, ProfileStage};
use rayon::prelude::*;
use std::fmt;

#[derive(Debug)]
pub enum TensorLoadError {
    Checkpoint(SafetensorError),
    Matrix(MatrixError),
    UnsupportedDType(DType),
    InvalidShape(String),
    NonFinite(String),
}

impl fmt::Display for TensorLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(f),
            Self::Matrix(error) => error.fmt(f),
            Self::UnsupportedDType(dtype) => write!(
                f,
                "reference tensor loader does not support dtype {dtype}; use F32/BF16/F16"
            ),
            Self::InvalidShape(reason) => write!(f, "invalid loaded tensor shape: {reason}"),
            Self::NonFinite(name) => write!(f, "tensor {name:?} contains NaN or infinity"),
        }
    }
}

impl std::error::Error for TensorLoadError {}

impl From<SafetensorError> for TensorLoadError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<MatrixError> for TensorLoadError {
    fn from(value: MatrixError) -> Self {
        Self::Matrix(value)
    }
}

/// Loads one bounded floating tensor into the scalar F32 reference representation.
///
/// The underlying range reader enforces a 64 MiB single-read ceiling. This is intentionally
/// for tiny fixtures and resident vectors, not a loophole for loading full GLM matrices.
pub fn load_reference_values(index: &TensorIndex, name: &str) -> Result<Vec<f32>, TensorLoadError> {
    let tensor = index.require(name)?;
    let byte_len = usize::try_from(tensor.data_len).map_err(|_| {
        TensorLoadError::InvalidShape(format!("tensor {name:?} byte length does not fit usize"))
    })?;
    let bytes = index.read_range(name, 0, byte_len)?;
    decode_reference_values(name, &tensor.dtype, &bytes)
}

pub(super) fn decode_reference_values(
    name: &str,
    dtype: &DType,
    bytes: &[u8],
) -> Result<Vec<f32>, TensorLoadError> {
    let values: Vec<f32> = match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect(),
        DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes(chunk.try_into().expect("two-byte chunk"));
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes(chunk.try_into().expect("two-byte chunk"));
                f16_to_f32(bits)
            })
            .collect(),
        dtype => return Err(TensorLoadError::UnsupportedDType(dtype.clone())),
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(TensorLoadError::NonFinite(name.to_owned()));
    }
    Ok(values)
}

pub fn load_reference_matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<DenseMatrix, TensorLoadError> {
    let tensor = index.require(name)?;
    let expected_shape = [rows as u64, columns as u64];
    if tensor.shape != expected_shape {
        return Err(TensorLoadError::InvalidShape(format!(
            "tensor {name:?} must declare {:?}, got {:?}",
            expected_shape, tensor.shape
        )));
    }
    Ok(DenseMatrix::new(
        rows,
        columns,
        load_reference_values(index, name)?,
    )?)
}

pub fn load_reference_vector(
    index: &TensorIndex,
    name: &str,
    length: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    let tensor = index.require(name)?;
    if tensor.shape != [length as u64] {
        return Err(TensorLoadError::InvalidShape(format!(
            "tensor {name:?} must declare [{length}], got {:?}",
            tensor.shape
        )));
    }
    load_reference_values(index, name)
}

/// Loads one row from a plain floating matrix without allocating or reading the other rows.
pub fn load_reference_matrix_row(
    index: &TensorIndex,
    name: &str,
    row: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    let tensor = index.require(name)?;
    if tensor.shape != [rows as u64, columns as u64] || row >= rows {
        return Err(TensorLoadError::InvalidShape(format!(
            "tensor {name:?} must declare [{rows},{columns}] and row {row} must be in range"
        )));
    }
    let element_bytes = match tensor.dtype {
        DType::F16 | DType::Bf16 => 2usize,
        DType::F32 => 4,
        ref dtype => return Err(TensorLoadError::UnsupportedDType(dtype.clone())),
    };
    let row_bytes = columns
        .checked_mul(element_bytes)
        .ok_or_else(|| TensorLoadError::InvalidShape("matrix row bytes overflow".to_owned()))?;
    let offset = row
        .checked_mul(row_bytes)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| TensorLoadError::InvalidShape("matrix row offset overflows".to_owned()))?;
    decode_reference_bytes(
        name,
        &tensor.dtype,
        index.read_range(name, offset, row_bytes)?,
    )
}

/// Streams a plain floating `[rows, columns]` matrix in bounded row chunks and computes
/// `matrix · input`. This is used for vocabulary-sized LM heads.
pub fn streamed_reference_matvec(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
    input: &[f32],
    rows_per_chunk: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    let tensor = index.require(name)?;
    if tensor.shape != [rows as u64, columns as u64]
        || columns == 0
        || input.len() != columns
        || rows_per_chunk == 0
    {
        return Err(TensorLoadError::InvalidShape(format!(
            "streamed matvec {name:?} needs [{rows},{columns}], {} inputs, and non-zero chunks",
            input.len()
        )));
    }
    let element_bytes = match tensor.dtype {
        DType::F16 | DType::Bf16 => 2usize,
        DType::F32 => 4,
        ref dtype => return Err(TensorLoadError::UnsupportedDType(dtype.clone())),
    };
    let row_bytes = columns
        .checked_mul(element_bytes)
        .ok_or_else(|| TensorLoadError::InvalidShape("matrix row bytes overflow".to_owned()))?;
    let maximum_rows = (64 * 1024 * 1024 / row_bytes).max(1);
    let chunk_rows = rows_per_chunk.min(maximum_rows);
    let mut output = Vec::with_capacity(rows);
    for start in (0..rows).step_by(chunk_rows) {
        let count = chunk_rows.min(rows - start);
        let byte_count = count.checked_mul(row_bytes).ok_or_else(|| {
            TensorLoadError::InvalidShape("matrix chunk bytes overflow".to_owned())
        })?;
        let offset = start
            .checked_mul(row_bytes)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| {
                TensorLoadError::InvalidShape("matrix chunk offset overflows".to_owned())
            })?;
        let work = count.saturating_mul(columns);
        let chunk_output = if tensor.dtype == DType::Bf16 {
            // Keep vocabulary-sized BF16 chunks compact. The existing BF16 kernel decodes
            // products directly from these bytes and retains the original ordered F64 sum.
            let map_matrix_error = |error| match error {
                MatrixError::NonFinite => TensorLoadError::NonFinite(name.to_owned()),
                error => TensorLoadError::Matrix(error),
            };
            let matrix = {
                let _profile =
                    span_with_metrics(ProfileStage::StreamedMatrixReadDecode, 0, byte_count);
                Bf16Matrix::from_le_bytes(
                    count,
                    columns,
                    index.read_range(name, offset, byte_count)?,
                )
                .map_err(map_matrix_error)?
            };
            let _profile = span_with_work(ProfileStage::StreamedMatrixCompute, work);
            matrix.matvec(input).map_err(map_matrix_error)?
        } else {
            let values = {
                let _profile =
                    span_with_metrics(ProfileStage::StreamedMatrixReadDecode, 0, byte_count);
                decode_reference_bytes(
                    name,
                    &tensor.dtype,
                    index.read_range(name, offset, byte_count)?,
                )?
            };
            let _profile = span_with_work(ProfileStage::StreamedMatrixCompute, work);
            let mut chunk_output = vec![0.0; count];
            if should_parallelize(count, work) {
                install(|| {
                    chunk_output
                        .par_iter_mut()
                        .zip(values.par_chunks_exact(columns))
                        .for_each(|(output, row)| {
                            *output = row
                                .iter()
                                .zip(input)
                                .map(|(&weight, &input)| f64::from(weight) * f64::from(input))
                                .sum::<f64>() as f32;
                        });
                });
            } else {
                for (output, row) in chunk_output.iter_mut().zip(values.chunks_exact(columns)) {
                    *output = row
                        .iter()
                        .zip(input)
                        .map(|(&weight, &input)| f64::from(weight) * f64::from(input))
                        .sum::<f64>() as f32;
                }
            }
            chunk_output
        };
        if chunk_output.iter().any(|value| !value.is_finite()) {
            return Err(TensorLoadError::NonFinite(name.to_owned()));
        }
        output.extend(chunk_output);
    }
    Ok(output)
}

/// Reads one row from an I64 routing table and converts it to platform-sized indices.
pub fn load_i64_matrix_row(
    index: &TensorIndex,
    name: &str,
    row: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<usize>, TensorLoadError> {
    let tensor = index.require(name)?;
    if tensor.dtype != DType::I64 || tensor.shape != [rows as u64, columns as u64] || row >= rows {
        return Err(TensorLoadError::InvalidShape(format!(
            "tensor {name:?} must be I64 [{rows},{columns}] and row {row} must be in range"
        )));
    }
    let row_bytes = columns
        .checked_mul(8)
        .ok_or_else(|| TensorLoadError::InvalidShape("I64 row bytes overflow".to_owned()))?;
    let offset = row
        .checked_mul(row_bytes)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| TensorLoadError::InvalidShape("I64 row offset overflows".to_owned()))?;
    index
        .read_range(name, offset, row_bytes)?
        .chunks_exact(8)
        .map(|chunk| {
            let value = i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
            usize::try_from(value).map_err(|_| {
                TensorLoadError::InvalidShape(format!(
                    "routing table {name:?} contains negative or oversized ID {value}"
                ))
            })
        })
        .collect()
}

fn decode_reference_bytes(
    name: &str,
    dtype: &DType,
    bytes: Vec<u8>,
) -> Result<Vec<f32>, TensorLoadError> {
    let values: Vec<f32> = match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect(),
        DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes(chunk.try_into().expect("two-byte chunk"));
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                f16_to_f32(u16::from_le_bytes(
                    chunk.try_into().expect("two-byte chunk"),
                ))
            })
            .collect(),
        dtype => return Err(TensorLoadError::UnsupportedDType(dtype.clone())),
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(TensorLoadError::NonFinite(name.to_owned()));
    }
    Ok(values)
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = (u32::from(value & 0x8000)) << 16;
    let exponent = (value >> 10) & 0x1f;
    let mantissa = value & 0x03ff;
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut exponent_adjustment = -14i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                exponent_adjustment -= 1;
            }
            normalized &= 0x03ff;
            sign | ((exponent_adjustment + 127) as u32) << 23 | (u32::from(normalized) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(mantissa) << 13),
        _ => sign | (u32::from(exponent + 112) << 23) | (u32::from(mantissa) << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::{ProfileSession, ProfileStage};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "urbilateria_reference_loader_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    fn write_fixture(path: &Path) {
        let matrix: Vec<u8> = [1.0f32, -2.0, 3.5, 0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let bf16: Vec<u8> = [0x3f80u16, 0xc000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let f16: Vec<u8> = [0x0000u16, 0x8000, 0x0001, 0x3e00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let nonfinite_f16: Vec<u8> = [0x7c00u16, 0x7e00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let packed = [0x88u8];
        let mut header = serde_json::json!({
            "matrix": {"dtype":"F32", "shape":[2,2], "data_offsets":[0,16]},
            "vector": {"dtype":"BF16", "shape":[2], "data_offsets":[16,20]},
            "f16": {"dtype":"F16", "shape":[4], "data_offsets":[20,28]},
            "nonfinite_f16": {"dtype":"F16", "shape":[2], "data_offsets":[28,32]},
            "packed": {"dtype":"U8", "shape":[1], "data_offsets":[32,33]}
        })
        .to_string()
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(matrix);
        bytes.extend(bf16);
        bytes.extend(f16);
        bytes.extend(nonfinite_f16);
        bytes.extend(packed);
        fs::write(path, bytes).unwrap();
    }

    fn write_large_stream_fixture(path: &Path, rows: usize, columns: usize) -> Vec<f32> {
        let values = (0..rows * columns)
            .map(|index| (index as f32 % 31.0 - 15.0) / 32.0)
            .collect::<Vec<_>>();
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut header = serde_json::json!({
            "large": {
                "dtype": "F32",
                "shape": [rows, columns],
                "data_offsets": [0, payload.len()]
            }
        })
        .to_string()
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        fs::write(path, bytes).unwrap();
        values
    }

    fn write_bf16_stream_fixture(path: &Path, rows: usize, columns: usize, bits: &[u16]) {
        assert_eq!(bits.len(), rows * columns);
        let mut header = serde_json::json!({
            "bf16": {
                "dtype": "BF16",
                "shape": [rows, columns],
                "data_offsets": [0, bits.len() * 2]
            }
        })
        .to_string()
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(bits.iter().flat_map(|value| value.to_le_bytes()));
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn loads_f32_matrix_and_converts_bf16_vector() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("tiny.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let matrix = load_reference_matrix(&index, "matrix", 2, 2).unwrap();
        assert_eq!(matrix.data(), &[1.0, -2.0, 3.5, 0.25]);
        assert_eq!(
            load_reference_vector(&index, "vector", 2).unwrap(),
            vec![1.0, -2.0]
        );
        let f16 = load_reference_vector(&index, "f16", 4).unwrap();
        assert_eq!(f16[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(f16[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16[2], 2.0f32.powi(-24));
        assert_eq!(f16[3], 1.5);
        assert!(matches!(
            load_reference_values(&index, "nonfinite_f16"),
            Err(TensorLoadError::NonFinite(name)) if name == "nonfinite_f16"
        ));
        assert!(matches!(
            load_reference_values(&index, "packed"),
            Err(TensorLoadError::UnsupportedDType(DType::U8))
        ));
        assert_eq!(
            load_reference_matrix_row(&index, "matrix", 1, 2, 2).unwrap(),
            vec![3.5, 0.25]
        );
        assert_eq!(
            streamed_reference_matvec(&index, "matrix", 2, 2, &[2.0, -1.0], 1).unwrap(),
            vec![4.0, 6.75]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streamed_matvec_parallel_chunks_match_ordered_scalar_rows() {
        let rows = 130;
        let columns = 1_025;
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        let values = write_large_stream_fixture(&dir.join("large.safetensors"), rows, columns);
        let index = TensorIndex::open(&dir).unwrap();
        let input = (0..columns)
            .map(|column| (column as f32 % 13.0 - 6.0) / 16.0)
            .collect::<Vec<_>>();
        let expected = values
            .chunks_exact(columns)
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(&weight, &input)| f64::from(weight) * f64::from(input))
                    .sum::<f64>() as f32
            })
            .collect::<Vec<_>>();

        let profile = ProfileSession::start();
        let actual = streamed_reference_matvec(&index, "large", rows, columns, &input, 65).unwrap();
        let report = profile.finish();
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
        let compute = report.stage(ProfileStage::StreamedMatrixCompute).unwrap();
        assert_eq!(compute.calls, 2);
        assert_eq!(compute.work_items, (rows * columns) as u64);
        let reads = report
            .stage(ProfileStage::StreamedMatrixReadDecode)
            .unwrap();
        assert_eq!(reads.calls, 2);
        assert_eq!(reads.logical_bytes, (rows * columns * 4) as u64);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streamed_bf16_matches_scalar_bits_across_parallel_chunks_and_tails() {
        let rows = 131;
        let columns = 1_025;
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        // Signed zeros, subnormals, large cancellation terms, and irregular mantissas exercise
        // the storage conversion and ordered F64 reduction independently of the BF16 kernel.
        let samples = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x3f81, 0xbf81, 0x4f00, 0xcf00, 0x3201, 0x3eab, 0xc140,
            0x4303,
        ];
        let bits = (0..rows * columns)
            .map(|index| samples[(index * 7 + index / columns) % samples.len()])
            .collect::<Vec<_>>();
        write_bf16_stream_fixture(&dir.join("bf16.safetensors"), rows, columns, &bits);
        let index = TensorIndex::open(&dir).unwrap();
        let input = (0..columns)
            .map(|column| {
                let sign = (column as u32 % 2) << 31;
                f32::from_bits(sign | 0x3f00_0000 | ((column as u32 * 7_919) & 0x007f_ffff))
            })
            .collect::<Vec<_>>();
        let expected = bits
            .chunks_exact(columns)
            .map(|row| {
                (row.iter()
                    .zip(&input)
                    .map(|(&bits, &input)| {
                        f64::from(f32::from_bits(u32::from(bits) << 16)) * f64::from(input)
                    })
                    .sum::<f64>() as f32)
                    .to_bits()
            })
            .collect::<Vec<_>>();
        for chunk_rows in [1, 65, 130, usize::MAX] {
            let actual =
                streamed_reference_matvec(&index, "bf16", rows, columns, &input, chunk_rows)
                    .unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected,
                "chunk rows: {chunk_rows}"
            );
        }

        let profile = ProfileSession::start();
        streamed_reference_matvec(&index, "bf16", rows, columns, &input, 65).unwrap();
        let report = profile.finish();
        let compute = report.stage(ProfileStage::StreamedMatrixCompute).unwrap();
        assert_eq!(compute.calls, 3);
        assert_eq!(compute.work_items, (rows * columns) as u64);
        let reads = report
            .stage(ProfileStage::StreamedMatrixReadDecode)
            .unwrap();
        assert_eq!(reads.calls, 3);
        assert_eq!(reads.logical_bytes, (rows * columns * 2) as u64);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streamed_bf16_retains_negative_zero_scalar_sum() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        for columns in [1, 7, 8, 9, 17] {
            let bits = [vec![0x8000; columns], vec![0; columns]].concat();
            write_bf16_stream_fixture(&dir.join("bf16.safetensors"), 2, columns, &bits);
            let index = TensorIndex::open(&dir).unwrap();
            let actual =
                streamed_reference_matvec(&index, "bf16", 2, columns, &vec![1.0; columns], 1)
                    .unwrap();
            assert_eq!(actual[0].to_bits(), (-0.0f32).to_bits());
            assert_eq!(actual[1].to_bits(), 0.0f32.to_bits());
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streamed_bf16_preserves_validation_and_nonfinite_errors() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bf16.safetensors");
        for nonfinite in [0x7f80, 0xff80, 0x7fc1, 0xffc1] {
            // The invalid value is in the final chunk, after one valid chunk has run.
            write_bf16_stream_fixture(&path, 2, 2, &[0x3f80, 0x4000, 0x4040, nonfinite]);
            let index = TensorIndex::open(&dir).unwrap();
            assert!(matches!(
                streamed_reference_matvec(&index, "bf16", 2, 2, &[1.0, 1.0], 1),
                Err(TensorLoadError::NonFinite(name)) if name == "bf16"
            ));
        }
        write_bf16_stream_fixture(&path, 2, 2, &[0x3f80, 0x4000, 0x4040, 0x4080]);
        let index = TensorIndex::open(&dir).unwrap();
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                streamed_reference_matvec(&index, "bf16", 2, 2, &[1.0, nonfinite], 1),
                Err(TensorLoadError::NonFinite(name)) if name == "bf16"
            ));
        }
        for (rows, columns, input, chunk_rows) in [
            (2, 2, vec![1.0], 1),
            (2, 2, vec![1.0; 3], 1),
            (1, 4, vec![1.0; 4], 1),
            (2, 2, vec![1.0; 2], 0),
        ] {
            assert!(matches!(
                streamed_reference_matvec(&index, "bf16", rows, columns, &input, chunk_rows),
                Err(TensorLoadError::InvalidShape(_))
            ));
        }
        write_bf16_stream_fixture(&path, 1, 1, &[0x7f7f]);
        let index = TensorIndex::open(&dir).unwrap();
        assert!(matches!(
            streamed_reference_matvec(&index, "bf16", 1, 1, &[4.0], 1),
            Err(TensorLoadError::NonFinite(name)) if name == "bf16"
        ));
        write_bf16_stream_fixture(&path, 1, 0, &[]);
        let index = TensorIndex::open(&dir).unwrap();
        assert!(matches!(
            streamed_reference_matvec(&index, "bf16", 1, 0, &[], 1),
            Err(TensorLoadError::InvalidShape(_))
        ));
        fs::remove_dir_all(dir).unwrap();
    }
}
