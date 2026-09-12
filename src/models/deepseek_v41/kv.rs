//! Packed cache formats used by DeepSeek-V4.1 CSA2.

use crate::math::{
    decode_e2m1, decode_e4m3fn, decode_e8m0, encode_e2m1_scalar, encode_e4m3fn_scalar,
    encode_e8m0_scalar, MxError,
};
use std::fmt;

pub const MAIN_KV_GROUP_SIZE: usize = 16;
pub const INDEX_KEY_GROUP_SIZE: usize = 32;
pub const WINDOW_KV_GROUP_SIZE: usize = 32;

#[derive(Debug)]
pub enum KvCacheError {
    Invalid(String),
    Quantization(MxError),
}

/// One indexer-key row: packed E2M1 values and one power-of-two E8M0 scale per 32 channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexKeyRow {
    width: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct IndexKeyCache {
    width: usize,
    rows: Vec<IndexKeyRow>,
}

impl IndexKeyCache {
    pub fn new(width: usize) -> Result<Self, KvCacheError> {
        if width == 0 || width % INDEX_KEY_GROUP_SIZE != 0 {
            return Err(KvCacheError::Invalid(format!(
                "index key width must be a non-zero multiple of {INDEX_KEY_GROUP_SIZE}"
            )));
        }
        Ok(Self {
            width,
            rows: Vec::new(),
        })
    }

    pub fn push(&mut self, values: &[f32]) -> Result<(), KvCacheError> {
        if values.len() != self.width {
            return Err(KvCacheError::Invalid(format!(
                "index key width {} differs from cache width {}",
                values.len(),
                self.width
            )));
        }
        self.rows.push(IndexKeyRow::quantize(values)?);
        Ok(())
    }

    pub fn row(&self, index: usize) -> Option<&IndexKeyRow> {
        self.rows.get(index)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rollback(&mut self, checkpoint: usize) -> Result<(), KvCacheError> {
        if checkpoint > self.rows.len() {
            return Err(KvCacheError::Invalid(
                "index-key rollback checkpoint is beyond the current cache".to_owned(),
            ));
        }
        self.rows.truncate(checkpoint);
        Ok(())
    }

    pub fn storage_bytes(&self) -> usize {
        self.rows.iter().map(IndexKeyRow::storage_bytes).sum()
    }
}

impl IndexKeyRow {
    pub fn quantize(values: &[f32]) -> Result<Self, KvCacheError> {
        let (packed, scales) = quantize_e2m1_e8m0(values, INDEX_KEY_GROUP_SIZE)?;
        Ok(Self {
            width: values.len(),
            packed,
            scales,
        })
    }

    pub fn decode(&self) -> Result<Vec<f32>, KvCacheError> {
        decode_e2m1_e8m0(self.width, &self.packed, &self.scales, INDEX_KEY_GROUP_SIZE)
    }

    pub fn storage_bytes(&self) -> usize {
        self.packed.len() + self.scales.len()
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
}

/// One raw sliding-window KV row in the native block-scaled E4M3/E8M0 layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowKvRow {
    values: Vec<u8>,
    scales: Vec<u8>,
}

impl WindowKvRow {
    pub fn quantize(values: &[f32]) -> Result<Self, KvCacheError> {
        validate_row(values, WINDOW_KV_GROUP_SIZE)?;
        let mut codes = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(values.len() / WINDOW_KV_GROUP_SIZE);
        for group in values.chunks_exact(WINDOW_KV_GROUP_SIZE) {
            let absolute_maximum = group
                .iter()
                .map(|value| value.abs())
                // `act_quant_kernel` clamps the activation amax before deriving its E8M0
                // power-of-two scale. Keep the same floor for an all-zero sliding-window row.
                .fold(1.0e-4, f32::max);
            let scale_code = encode_e8m0_scalar(absolute_maximum / 448.0)?;
            let scale = decode_e8m0(scale_code)?;
            scales.push(scale_code);
            for &value in group {
                codes.push(encode_e4m3fn_scalar((value / scale).clamp(-448.0, 448.0))?);
            }
        }
        Ok(Self {
            values: codes,
            scales,
        })
    }

    pub fn decode(&self) -> Result<Vec<f32>, KvCacheError> {
        self.values
            .iter()
            .enumerate()
            .map(|(column, &code)| {
                let value = decode_e4m3fn(code);
                if !value.is_finite() {
                    return Err(KvCacheError::Invalid(format!(
                        "window KV column {column} contains an E4M3 NaN"
                    )));
                }
                Ok(value * decode_e8m0(self.scales[column / WINDOW_KV_GROUP_SIZE])?)
            })
            .collect()
    }

    pub fn width(&self) -> usize {
        self.values.len()
    }

    pub fn storage_bytes(&self) -> usize {
        self.values.len() + self.scales.len()
    }
}

/// Opaque undo record for one ring write. Dropping it commits the write.
#[derive(Debug)]
pub struct WindowKvUndo {
    position: usize,
    slot: usize,
    previous: Option<(usize, WindowKvRow)>,
}

/// Fixed-memory ring for raw sliding-window KV. Absolute positions guard against stale slots.
#[derive(Debug, Clone)]
pub struct WindowKvCache {
    width: usize,
    next_position: usize,
    slots: Vec<Option<(usize, WindowKvRow)>>,
}

impl WindowKvCache {
    pub fn new(width: usize, capacity: usize) -> Result<Self, KvCacheError> {
        if width == 0 || width % WINDOW_KV_GROUP_SIZE != 0 || capacity == 0 {
            return Err(KvCacheError::Invalid(format!(
                "window width must be a non-zero multiple of {WINDOW_KV_GROUP_SIZE} and capacity must be non-zero"
            )));
        }
        Ok(Self {
            width,
            next_position: 0,
            slots: vec![None; capacity],
        })
    }

    pub fn append(
        &mut self,
        position: usize,
        values: &[f32],
    ) -> Result<WindowKvUndo, KvCacheError> {
        if position != self.next_position || values.len() != self.width {
            return Err(KvCacheError::Invalid(format!(
                "window append expects position {} and width {}, got position {position} and width {}",
                self.next_position,
                self.width,
                values.len()
            )));
        }
        let row = WindowKvRow::quantize(values)?;
        let slot = position % self.slots.len();
        let previous = self.slots[slot].replace((position, row));
        self.next_position += 1;
        Ok(WindowKvUndo {
            position,
            slot,
            previous,
        })
    }

    pub fn rollback(&mut self, undo: WindowKvUndo) -> Result<(), KvCacheError> {
        if self.next_position != undo.position + 1
            || undo.slot != undo.position % self.slots.len()
            || self.slots[undo.slot]
                .as_ref()
                .map(|(position, _)| *position)
                != Some(undo.position)
        {
            return Err(KvCacheError::Invalid(
                "window undo is stale or is not the latest append".to_owned(),
            ));
        }
        self.slots[undo.slot] = undo.previous;
        self.next_position = undo.position;
        Ok(())
    }

    pub fn get(&self, position: usize) -> Option<&WindowKvRow> {
        self.slots[position % self.slots.len()]
            .as_ref()
            .and_then(|(stored_position, row)| (*stored_position == position).then_some(row))
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn storage_bytes(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|(_, row)| row.storage_bytes())
            .sum()
    }
}

fn quantize_e2m1_e8m0(
    values: &[f32],
    group_size: usize,
) -> Result<(Vec<u8>, Vec<u8>), KvCacheError> {
    validate_row(values, group_size)?;
    let mut codes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / group_size);
    for group in values.chunks_exact(group_size) {
        let absolute_maximum = group
            .iter()
            .map(|value| value.abs())
            .fold(6.0 * 2.0f32.powi(-126), f32::max);
        let scale_code = encode_e8m0_scalar(absolute_maximum / 6.0)?;
        let scale = decode_e8m0(scale_code)?;
        scales.push(scale_code);
        for &value in group {
            codes.push(encode_e2m1_scalar((value / scale).clamp(-6.0, 6.0))?);
        }
    }
    Ok((
        codes
            .chunks_exact(2)
            .map(|pair| pair[0] | (pair[1] << 4))
            .collect(),
        scales,
    ))
}

fn decode_e2m1_e8m0(
    width: usize,
    packed: &[u8],
    scales: &[u8],
    group_size: usize,
) -> Result<Vec<f32>, KvCacheError> {
    (0..width)
        .map(|column| {
            let byte = packed[column / 2];
            let code = if column % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            Ok(decode_e2m1(code) * decode_e8m0(scales[column / group_size])?)
        })
        .collect()
}

fn validate_row(values: &[f32], group_size: usize) -> Result<(), KvCacheError> {
    if values.is_empty()
        || values.len() % group_size != 0
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(KvCacheError::Invalid(format!(
            "row width {} must be a non-zero multiple of {group_size} with finite values",
            values.len()
        )));
    }
    Ok(())
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid DeepSeek-V4.1 KV cache: {reason}"),
            Self::Quantization(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for KvCacheError {}

impl From<MxError> for KvCacheError {
    fn from(value: MxError) -> Self {
        Self::Quantization(value)
    }
}

/// One global KV row in the trained cache format: packed E2M1 values and one E4M3 scale for each
/// 16 consecutive channels. The second-level global scale used by NVFP4 is deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MainKvRow {
    width: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
}

impl MainKvRow {
    pub fn quantize(values: &[f32]) -> Result<Self, KvCacheError> {
        if values.is_empty()
            || values.len() % MAIN_KV_GROUP_SIZE != 0
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(KvCacheError::Invalid(format!(
                "row width {} must be a non-zero multiple of {MAIN_KV_GROUP_SIZE} with finite values",
                values.len()
            )));
        }
        let mut codes = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(values.len() / MAIN_KV_GROUP_SIZE);
        for group in values.chunks_exact(MAIN_KV_GROUP_SIZE) {
            let absolute_maximum = group
                .iter()
                .map(|value| value.abs())
                .fold(6.0 * 2.0f32.powi(-9), f32::max);
            // The training/deployment kernel rounds the scale itself to E4M3 before quantizing
            // values. Even an all-zero group therefore receives a non-zero 2^-9 scale.
            let scale_code = encode_e4m3fn_scalar(absolute_maximum / 6.0)?;
            let scale = decode_e4m3fn(scale_code);
            if !scale.is_finite() || scale <= 0.0 {
                return Err(KvCacheError::Invalid(
                    "E4M3 scale rounded to a non-finite or non-positive value".to_owned(),
                ));
            }
            scales.push(scale_code);
            for &value in group {
                codes.push(encode_e2m1_scalar((value / scale).clamp(-6.0, 6.0))?);
            }
        }
        let packed = codes
            .chunks_exact(2)
            .map(|pair| pair[0] | (pair[1] << 4))
            .collect();
        Ok(Self {
            width: values.len(),
            packed,
            scales,
        })
    }

    pub fn from_parts(
        width: usize,
        packed: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<Self, KvCacheError> {
        if width == 0
            || width % MAIN_KV_GROUP_SIZE != 0
            || packed.len() != width.div_ceil(2)
            || scales.len() != width.div_ceil(MAIN_KV_GROUP_SIZE)
        {
            return Err(KvCacheError::Invalid(
                "packed values/scales do not match the declared row width".to_owned(),
            ));
        }
        if scales.iter().any(|&code| {
            let value = decode_e4m3fn(code);
            !value.is_finite() || value <= 0.0
        }) {
            return Err(KvCacheError::Invalid(
                "cache contains a non-finite or non-positive E4M3 scale".to_owned(),
            ));
        }
        Ok(Self {
            width,
            packed,
            scales,
        })
    }

    pub fn decode(&self) -> Vec<f32> {
        (0..self.width)
            .map(|column| {
                let byte = self.packed[column / 2];
                let code = if column % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let scale = decode_e4m3fn(self.scales[column / MAIN_KV_GROUP_SIZE]);
                decode_e2m1(code) * scale
            })
            .collect()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub fn storage_bytes(&self) -> usize {
        self.packed.len() + self.scales.len()
    }
}

/// Append-only storage for one CSA2 global-KV owner. Rows remain packed while resident and can be
/// rolled back transactionally when a later layer or expert read fails.
#[derive(Debug, Clone)]
pub struct MainKvCache {
    width: usize,
    rows: Vec<MainKvRow>,
}

impl MainKvCache {
    pub fn new(width: usize) -> Result<Self, KvCacheError> {
        if width == 0 || width % MAIN_KV_GROUP_SIZE != 0 {
            return Err(KvCacheError::Invalid(format!(
                "cache width must be a non-zero multiple of {MAIN_KV_GROUP_SIZE}"
            )));
        }
        Ok(Self {
            width,
            rows: Vec::new(),
        })
    }

    pub fn push(&mut self, values: &[f32]) -> Result<(), KvCacheError> {
        if values.len() != self.width {
            return Err(KvCacheError::Invalid(format!(
                "row width {} differs from cache width {}",
                values.len(),
                self.width
            )));
        }
        self.rows.push(MainKvRow::quantize(values)?);
        Ok(())
    }

    pub fn row(&self, index: usize) -> Option<&MainKvRow> {
        self.rows.get(index)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn checkpoint(&self) -> usize {
        self.rows.len()
    }

    pub fn rollback(&mut self, checkpoint: usize) -> Result<(), KvCacheError> {
        if checkpoint > self.rows.len() {
            return Err(KvCacheError::Invalid(
                "rollback checkpoint is beyond the current cache".to_owned(),
            ));
        }
        self.rows.truncate(checkpoint);
        Ok(())
    }

    pub fn storage_bytes(&self) -> usize {
        self.rows.iter().map(MainKvRow::storage_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_kv_uses_exact_288_byte_release_layout() {
        let values = (0..512)
            .map(|index| (index as f32 - 256.0) / 19.0)
            .collect::<Vec<_>>();
        let row = MainKvRow::quantize(&values).unwrap();
        assert_eq!(row.packed().len(), 256);
        assert_eq!(row.scales().len(), 32);
        assert_eq!(row.storage_bytes(), 288);
        assert!(row.decode().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn zero_groups_keep_the_required_nonzero_e4m3_scale() {
        let row = MainKvRow::quantize(&[0.0; 16]).unwrap();
        assert_eq!(decode_e4m3fn(row.scales()[0]), 2.0f32.powi(-9));
        assert_eq!(row.decode(), vec![0.0; 16]);
    }

    #[test]
    fn cache_rollback_restores_a_failed_token_boundary() {
        let mut cache = MainKvCache::new(16).unwrap();
        cache.push(&[1.0; 16]).unwrap();
        let checkpoint = cache.checkpoint();
        cache.push(&[2.0; 16]).unwrap();
        cache.rollback(checkpoint).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.storage_bytes(), 9);
    }

    #[test]
    fn index_key_uses_exact_68_byte_release_layout() {
        let values = (0..128)
            .map(|index| (index as f32 - 64.0) / 7.0)
            .collect::<Vec<_>>();
        let row = IndexKeyRow::quantize(&values).unwrap();
        assert_eq!(row.packed().len(), 64);
        assert_eq!(row.scales().len(), 4);
        assert_eq!(row.storage_bytes(), 68);
        assert!(row.decode().unwrap().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn window_ring_uses_528_byte_rows_and_rejects_stale_positions() {
        let mut cache = WindowKvCache::new(512, 2).unwrap();
        let first = cache.append(0, &[1.0; 512]).unwrap();
        assert_eq!(cache.get(0).unwrap().storage_bytes(), 528);
        cache.append(1, &[2.0; 512]).unwrap();
        cache.append(2, &[3.0; 512]).unwrap();
        assert!(cache.get(0).is_none());
        assert!(cache.get(2).is_some());
        assert!(cache.rollback(first).is_err());
    }

    #[test]
    fn window_ring_rollback_restores_overwritten_slot() {
        let mut cache = WindowKvCache::new(32, 1).unwrap();
        cache.append(0, &[1.0; 32]).unwrap();
        let undo = cache.append(1, &[2.0; 32]).unwrap();
        cache.rollback(undo).unwrap();
        assert_eq!(cache.next_position(), 1);
        assert_eq!(cache.get(0).unwrap().decode().unwrap(), vec![1.0; 32]);
    }
}
