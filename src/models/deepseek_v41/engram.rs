//! Exact Engram addressing and bounded table-row reads for DeepSeek-V4.1-Flash.
//!
//! The two embedding tables total almost 189 GiB. Loading either table is therefore forbidden:
//! one token reads only its 24 random rows from each participating layer.

use super::DeepseekV41Config;
use crate::math::{decode_e4m3fn, decode_e8m0};
use crate::storage::{DType, SafetensorError, TensorIndex};
use crate::tokenizer::{ByteBpeTokenizer, TokenizerError};
use std::collections::HashMap;
use std::fmt;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

const RELEASE_COMPRESSED_VOCAB_SIZE: usize = 99_092;
const MAX_ROW_READ_BYTES: usize = 64 * 1024;
const RELEASE_TOKEN_MAP_FNV1A64: u64 = 0x6b5c_fa68_a5f5_7ba6;
const RELEASE_MULTIPLIERS: [[u64; 4]; 2] = [
    [
        76_632_096_046_245,
        4_839_876_093_313,
        35_959_672_319_349,
        73_987_337_458_391,
    ],
    [
        67_716_810_739_261,
        51_510_806_800_915,
        30_921_347_202_721,
        82_619_226_485_591,
    ],
];

#[derive(Debug)]
pub enum EngramError {
    Invalid(String),
    Checkpoint(SafetensorError),
    Tokenizer(TokenizerError),
}

impl fmt::Display for EngramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => {
                write!(formatter, "invalid DeepSeek-V4.1 Engram state: {reason}")
            }
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Tokenizer(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EngramError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            Self::Tokenizer(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<SafetensorError> for EngramError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<TokenizerError> for EngramError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

/// A validated, non-owning view of one enormous native Engram table.
#[derive(Debug)]
pub struct EngramTable<'a> {
    index: &'a TensorIndex,
    weight_name: String,
    scale_name: String,
    rows: usize,
    width: usize,
}

impl<'a> EngramTable<'a> {
    pub fn open(
        index: &'a TensorIndex,
        layer: usize,
        rows: usize,
        width: usize,
    ) -> Result<Self, EngramError> {
        if rows == 0 || width == 0 || width % 32 != 0 {
            return Err(EngramError::Invalid(
                "table rows and width must be non-zero and width must be divisible by 32"
                    .to_owned(),
            ));
        }
        let weight_name = format!("layers.{layer}.engram.embed.weight");
        let scale_name = format!("layers.{layer}.engram.embed.scale");
        let weight = index.require(&weight_name)?;
        let scale = index.require(&scale_name)?;
        let expected_weight_shape = [rows as u64, width as u64];
        let expected_scale_shape = [rows as u64, width.div_ceil(32) as u64];
        if weight.dtype != DType::F8E4M3 || weight.shape != expected_weight_shape {
            return Err(EngramError::Invalid(format!(
                "{weight_name:?} must be F8_E4M3 {expected_weight_shape:?}, got {} {:?}",
                weight.dtype, weight.shape
            )));
        }
        if scale.dtype != DType::F8E8M0 || scale.shape != expected_scale_shape {
            return Err(EngramError::Invalid(format!(
                "{scale_name:?} must be F8_E8M0 {expected_scale_shape:?}, got {} {:?}",
                scale.dtype, scale.shape
            )));
        }
        Ok(Self {
            index,
            weight_name,
            scale_name,
            rows,
            width,
        })
    }

    /// Reads and decodes exactly one row: `width` E4M3 bytes plus `width/32` E8M0 bytes.
    pub fn read_row(&self, row: usize) -> Result<Vec<f32>, EngramError> {
        Ok(self
            .read_rows(&[row])?
            .pop()
            .expect("one row was requested"))
    }

    /// Preserves caller order, but reads unique rows in physical tensor/offset order.
    ///
    /// Grouping value and scale reads avoids alternating between distant regions of
    /// an enormous shard. Adjacent requested rows share a bounded read; gaps are never read.
    pub fn read_rows(&self, row_ids: &[usize]) -> Result<Vec<Vec<f32>>, EngramError> {
        if let Some(&row) = row_ids.iter().find(|&&row| row >= self.rows) {
            return Err(EngramError::Invalid(format!(
                "row {row} is outside table with {} rows",
                self.rows
            )));
        }
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut unique_rows = row_ids.to_vec();
        unique_rows.sort_unstable();
        unique_rows.dedup();
        let scale_width = self.width / 32;
        let read = |name: &str, width| {
            read_sorted_row_bytes(&unique_rows, width, |offset, length| {
                Ok(self.index.read_range(name, offset, length)?)
            })
        };
        let weight = self.index.require(&self.weight_name)?;
        let scale = self.index.require(&self.scale_name)?;
        let (values, scales) =
            if (&weight.shard, weight.data_offset) <= (&scale.shard, scale.data_offset) {
                (
                    read(&self.weight_name, self.width)?,
                    read(&self.scale_name, scale_width)?,
                )
            } else {
                let scales = read(&self.scale_name, scale_width)?;
                (read(&self.weight_name, self.width)?, scales)
            };
        let decoded = unique_rows
            .iter()
            .zip(values.chunks_exact(self.width))
            .zip(scales.chunks_exact(scale_width))
            .map(|((&row, values), scales)| Self::decode_row(row, values, scales))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(row_ids
            .iter()
            .map(|row| decoded[unique_rows.binary_search(row).expect("row was read")].clone())
            .collect())
    }

    fn decode_row(row: usize, values: &[u8], scales: &[u8]) -> Result<Vec<f32>, EngramError> {
        values
            .iter()
            .enumerate()
            .map(|(column, &code)| {
                let value = decode_e4m3fn(code);
                if !value.is_finite() {
                    return Err(EngramError::Invalid(format!(
                        "row {row}, column {column} contains an E4M3 NaN"
                    )));
                }
                Ok(value
                    * decode_e8m0(scales[column / 32]).map_err(|error| {
                        EngramError::Invalid(format!(
                            "row {row}, scale group {}: {error}",
                            column / 32
                        ))
                    })?)
            })
            .collect()
    }

    pub fn row_storage_bytes(&self) -> usize {
        self.width + self.width / 32
    }
}

/// Packs only the requested, sorted unique rows. Temporary reads are at most 64 KiB (or one
/// row for wider tables); total storage is proportional to the requested rows, never their span.
fn read_sorted_row_bytes(
    rows: &[usize],
    width: usize,
    mut read: impl FnMut(u64, usize) -> Result<Vec<u8>, EngramError>,
) -> Result<Vec<u8>, EngramError> {
    let length = rows.len().checked_mul(width).ok_or_else(|| {
        EngramError::Invalid("requested table rows overflow allocation size".to_owned())
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(length).map_err(|error| {
        EngramError::Invalid(format!("cannot reserve {length} row bytes: {error}"))
    })?;
    let max_rows = (MAX_ROW_READ_BYTES / width).max(1);
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len()
            && end - start < max_rows
            && rows[end - 1].checked_add(1) == Some(rows[end])
        {
            end += 1;
        }
        let offset = rows[start]
            .checked_mul(width)
            .and_then(|offset| u64::try_from(offset).ok())
            .ok_or_else(|| EngramError::Invalid("table row offset overflows".to_owned()))?;
        output.extend(read(offset, (end - start) * width)?);
        start = end;
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct HashLayer {
    primes: Vec<Vec<u64>>,
    offsets: Vec<u64>,
    multipliers: [u64; 4],
}

/// Stateful native n-gram hasher. It carries compressed token IDs across prefill/decode calls.
#[derive(Debug, Clone)]
pub struct NgramHashState {
    token_map: Vec<u32>,
    pad_id: u64,
    max_ngram_size: usize,
    n_heads: usize,
    layers: Vec<HashLayer>,
    cache: Vec<Option<u64>>,
}

impl NgramHashState {
    pub fn new(
        config: &DeepseekV41Config,
        tokenizer: &ByteBpeTokenizer,
    ) -> Result<Self, EngramError> {
        let text = &config.text_config;
        let token_map = build_compressed_token_map(tokenizer)?;
        let compressed_size = token_map
            .iter()
            .copied()
            .max()
            .map_or(0usize, |value| value as usize + 1);
        let fingerprint = token_map_fingerprint(&token_map);
        if tokenizer.vocabulary_size() != text.vocab_size
            || compressed_size != RELEASE_COMPRESSED_VOCAB_SIZE
            || compressed_size != text.engram_compressed_vocab_size
            || fingerprint != RELEASE_TOKEN_MAP_FNV1A64
        {
            return Err(EngramError::Invalid(format!(
                "token map mismatch: tokenizer={}, compressed={compressed_size}, fnv1a64={fingerprint:#018x}; expected {}, {}, {RELEASE_TOKEN_MAP_FNV1A64:#018x}",
                tokenizer.vocabulary_size(), text.vocab_size, text.engram_compressed_vocab_size
            )));
        }
        let pad_id = u64::from(token_map[text.engram_pad_token_id as usize]);
        let mut seen = Vec::<u64>::new();
        let mut layers = Vec::with_capacity(text.engram_layer_ids.len());
        for (layer_index, &layer_id) in text.engram_layer_ids.iter().enumerate() {
            if layer_index >= RELEASE_MULTIPLIERS.len() || !matches!(layer_id, 1 | 14) {
                return Err(EngramError::Invalid(format!(
                    "no pinned release multipliers for Engram layer {layer_id}"
                )));
            }
            let mut primes = Vec::with_capacity(text.engram_max_ngram_size - 1);
            let mut flat =
                Vec::with_capacity((text.engram_max_ngram_size - 1) * text.engram_n_heads);
            for _ in 1..text.engram_max_ngram_size {
                let mut current = text.engram_vocab_size as u64 - 1;
                let mut row = Vec::with_capacity(text.engram_n_heads);
                for _ in 0..text.engram_n_heads {
                    current = next_unseen_prime(current, &seen);
                    seen.push(current);
                    flat.push(current);
                    row.push(current);
                }
                primes.push(row);
            }
            let mut running = 0u64;
            let offsets = flat
                .iter()
                .map(|&size| {
                    let offset = running;
                    running = running
                        .checked_add(size)
                        .expect("release bucket sum fits u64");
                    offset
                })
                .collect::<Vec<_>>();
            if running != text.engram_num_embeddings[layer_index] as u64 {
                return Err(EngramError::Invalid(format!(
                    "Engram layer {layer_id} bucket sum {running} differs from table rows {}",
                    text.engram_num_embeddings[layer_index]
                )));
            }
            layers.push(HashLayer {
                primes,
                offsets,
                multipliers: RELEASE_MULTIPLIERS[layer_index],
            });
        }
        Ok(Self {
            token_map,
            pad_id,
            max_ngram_size: text.engram_max_ngram_size,
            n_heads: text.engram_n_heads,
            layers,
            cache: Vec::new(),
        })
    }

    /// Hashes a contiguous token span. `participates=false` marks image-span tokens as DEAD.
    pub fn push(
        &mut self,
        start_pos: usize,
        token_ids: &[u32],
        participates: Option<&[bool]>,
    ) -> Result<Vec<Vec<Vec<u64>>>, EngramError> {
        if participates.is_some_and(|mask| mask.len() != token_ids.len()) {
            return Err(EngramError::Invalid(
                "Engram token mask length differs from token span".to_owned(),
            ));
        }
        if start_pos > self.cache.len() {
            return Err(EngramError::Invalid(format!(
                "start_pos={start_pos} leaves a gap after cached length {}",
                self.cache.len()
            )));
        }
        self.cache.truncate(start_pos);
        let mut output = Vec::with_capacity(token_ids.len());
        for (offset, &token_id) in token_ids.iter().enumerate() {
            let compressed = self
                .token_map
                .get(token_id as usize)
                .copied()
                .ok_or(TokenizerError::UnknownTokenId(token_id))?;
            let live = participates.map_or(true, |mask| mask[offset]);
            self.cache.push(live.then_some(u64::from(compressed)));
            output.push(self.hash_position(self.cache.len() - 1));
        }
        Ok(output)
    }

    fn hash_position(&self, position: usize) -> Vec<Vec<u64>> {
        let mut blocked = false;
        let tokens = (0..self.max_ngram_size)
            .map(|shift| {
                if position < shift || self.cache[position - shift].is_none() {
                    blocked = true;
                }
                if blocked {
                    self.pad_id
                } else {
                    self.cache[position - shift].expect("unblocked token exists")
                }
            })
            .collect::<Vec<_>>();
        self.layers
            .iter()
            .map(|layer| {
                let mut rolling = tokens[0].wrapping_mul(layer.multipliers[0]);
                let mut hashes = Vec::with_capacity((self.max_ngram_size - 1) * self.n_heads);
                for ngram_index in 0..self.max_ngram_size - 1 {
                    rolling ^=
                        tokens[ngram_index + 1].wrapping_mul(layer.multipliers[ngram_index + 1]);
                    for head in 0..self.n_heads {
                        let column = ngram_index * self.n_heads + head;
                        hashes.push(
                            rolling % layer.primes[ngram_index][head] + layer.offsets[column],
                        );
                    }
                }
                hashes
            })
            .collect()
    }

    pub fn cached_tokens(&self) -> usize {
        self.cache.len()
    }

    /// Returns the append-only history length needed to restore this hasher after a failed step.
    pub fn checkpoint(&self) -> usize {
        self.cache.len()
    }

    /// Discards token history appended after `checkpoint`.
    pub fn rollback(&mut self, checkpoint: usize) -> Result<(), EngramError> {
        if checkpoint > self.cache.len() {
            return Err(EngramError::Invalid(
                "Engram rollback checkpoint is beyond the cached token history".to_owned(),
            ));
        }
        self.cache.truncate(checkpoint);
        Ok(())
    }
}

/// Builds the exact trained normalization map. Construction is linear in tokenizer size and
/// should happen once per model state.
pub fn build_compressed_token_map(tokenizer: &ByteBpeTokenizer) -> Result<Vec<u32>, EngramError> {
    let mut keys = HashMap::<String, u32>::with_capacity(tokenizer.vocabulary_size());
    let mut lookup = Vec::with_capacity(tokenizer.vocabulary_size());
    for token_id in 0..tokenizer.vocabulary_size() as u32 {
        let decoded = tokenizer.decode(&[token_id], false)?;
        let key = if decoded.contains('\u{fffd}') {
            tokenizer.raw_token_content(token_id)?.to_owned()
        } else {
            let normalized = normalize_engram_token(&decoded);
            if normalized.is_empty() {
                decoded
            } else {
                normalized
            }
        };
        let next = keys.len() as u32;
        let compressed = *keys.entry(key).or_insert(next);
        lookup.push(compressed);
    }
    Ok(lookup)
}

fn normalize_engram_token(value: &str) -> String {
    let compatible = value.nfkc().collect::<String>();
    let decomposed = compatible.nfd().filter(|character| {
        !matches!(
            get_general_category(*character),
            GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
    });
    let lowered = decomposed.flat_map(char::to_lowercase).collect::<String>();
    let mut collapsed = String::with_capacity(lowered.len());
    let mut in_ascii_space = false;
    for character in lowered.chars() {
        if matches!(character, ' ' | '\t' | '\r' | '\n') {
            if !in_ascii_space {
                collapsed.push(' ');
                in_ascii_space = true;
            }
        } else {
            collapsed.push(character);
            in_ascii_space = false;
        }
    }
    if collapsed == " " {
        collapsed
    } else {
        collapsed.trim().to_owned()
    }
}

fn token_map_fingerprint(token_map: &[u32]) -> u64 {
    token_map
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn next_unseen_prime(mut current: u64, seen: &[u64]) -> u64 {
    loop {
        current += 1;
        if is_prime(current) && !seen.contains(&current) {
            return current;
        }
    }
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if value == prime {
            return true;
        }
        if value % prime == 0 {
            return false;
        }
    }
    // Deterministic Miller-Rabin bases for all u64 values.
    let shifts = (value - 1).trailing_zeros();
    let odd = (value - 1) >> shifts;
    [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022]
        .into_iter()
        .all(|base| miller_rabin_round(value, odd, shifts, base % value))
}

fn miller_rabin_round(modulus: u64, odd: u64, shifts: u32, base: u64) -> bool {
    if base == 0 {
        return true;
    }
    let mut value = modular_pow(base, odd, modulus);
    if value == 1 || value == modulus - 1 {
        return true;
    }
    for _ in 1..shifts {
        value = ((value as u128 * value as u128) % modulus as u128) as u64;
        if value == modulus - 1 {
            return true;
        }
    }
    false
}

fn modular_pow(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut output = 1u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = ((output as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exponent >>= 1;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TableFixture(PathBuf);

    impl TableFixture {
        fn new(scales_first: bool) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "urb_engram_rows_{}_{}",
                std::process::id(),
                nonce
            ));
            fs::create_dir_all(&dir).unwrap();
            let values = (0..6)
                .flat_map(|row| std::iter::repeat_n(if row == 3 { 0x7f } else { 0x38 + row }, 64))
                .collect::<Vec<u8>>();
            let scales = [126, 127].repeat(6);
            let (value_start, scale_start) = if scales_first {
                (scales.len(), 0)
            } else {
                (0, values.len())
            };
            let mut header = serde_json::json!({
                "layers.1.engram.embed.weight": {
                    "dtype": "F8_E4M3", "shape": [6, 64],
                    "data_offsets": [value_start, value_start + values.len()]
                },
                "layers.1.engram.embed.scale": {
                    "dtype": "F8_E8M0", "shape": [6, 2],
                    "data_offsets": [scale_start, scale_start + scales.len()]
                }
            })
            .to_string()
            .into_bytes();
            while header.len() % 8 != 0 {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            if scales_first {
                bytes.extend(scales);
                bytes.extend(values);
            } else {
                bytes.extend(values);
                bytes.extend(scales);
            }
            fs::write(dir.join("table.safetensors"), bytes).unwrap();
            Self(dir)
        }
    }

    impl Drop for TableFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn batched_rows_preserve_native_values_caller_order_and_duplicates() {
        for scales_first in [false, true] {
            let fixture = TableFixture::new(scales_first);
            let index = TensorIndex::open(&fixture.0).unwrap();
            let table = EngramTable::open(&index, 1, 6, 64).unwrap();
            let ids = [5, 1, 2, 1, 0];
            let rows = table.read_rows(&ids).unwrap();
            for (&row, decoded) in ids.iter().zip(&rows) {
                let value = 1.0 + row as f32 / 8.0;
                assert_eq!(&decoded[..32], &[value * 0.5; 32]);
                assert_eq!(&decoded[32..], &[value; 32]);
            }
            assert_eq!(table.read_row(5).unwrap(), rows[0]);
            assert!(table.read_rows(&[]).unwrap().is_empty());
            assert!(table
                .read_rows(&[0, 6])
                .unwrap_err()
                .to_string()
                .contains("row 6"));
            assert!(table
                .read_rows(&[3])
                .unwrap_err()
                .to_string()
                .contains("E4M3 NaN"));
        }
    }

    #[test]
    fn contiguous_reads_are_capped_and_sparse_gaps_are_never_read() {
        let rows = (0..300).chain([10_000_000, 10_000_001]).collect::<Vec<_>>();
        let mut reads = Vec::new();
        let bytes = read_sorted_row_bytes(&rows, 256, |offset, length| {
            reads.push((offset, length));
            Ok((offset..offset + length as u64)
                .map(|offset| (offset / 256) as u8)
                .collect())
        })
        .unwrap();
        assert_eq!(reads, [(0, 65_536), (65_536, 11_264), (2_560_000_000, 512)]);
        assert_eq!(bytes.len(), rows.len() * 256);
        assert_eq!(
            reads.iter().map(|(_, length)| length).sum::<usize>(),
            bytes.len()
        );
        for (&row, bytes) in rows.iter().zip(bytes.chunks_exact(256)) {
            assert!(bytes.iter().all(|&byte| byte == row as u8));
        }
    }

    #[test]
    fn row_offset_overflow_is_rejected_before_reading() {
        let error = read_sorted_row_bytes(&[usize::MAX], 256, |_, _| {
            panic!("overflow must fail before I/O")
        })
        .unwrap_err();
        assert!(error.to_string().contains("offset overflows"));
    }

    #[test]
    fn release_prime_layout_sums_to_both_table_sizes() {
        let config = DeepseekV41Config::from_json_str(include_str!(
            "../../../tests/fixtures/deepseek_v4_1_flash_config.json"
        ))
        .unwrap();
        let text = &config.text_config;
        let mut seen = Vec::new();
        let mut sums = Vec::new();
        for _ in &text.engram_layer_ids {
            let mut sum = 0;
            for _ in 1..text.engram_max_ngram_size {
                let mut current = text.engram_vocab_size as u64 - 1;
                for _ in 0..text.engram_n_heads {
                    current = next_unseen_prime(current, &seen);
                    seen.push(current);
                    sum += current;
                }
            }
            sums.push(sum);
        }
        assert_eq!(sums, vec![384_006_168, 384_016_682]);
    }

    #[test]
    fn normalization_preserves_a_single_space_and_collapses_accents_and_case() {
        assert_eq!(normalize_engram_token(" "), " ");
        assert_eq!(normalize_engram_token("  A\u{301}\tB  "), "a b");
    }

    #[test]
    fn primality_reference_rejects_composites() {
        assert!(is_prime(16_000_057));
        assert!(!is_prime(16_000_057 * 3));
    }
}
