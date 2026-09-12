//! Shared byte-level BPE engine for the supported model-native tokenizer geometries.
//!
//! It reads the Hugging Face `tokenizer.json` directly but implements pre-tokenization, BPE,
//! added-token handling, family-selected pre-tokenization, and byte-level decoding in Rust. The
//! accepted JSON geometries are intentionally narrow so tokenizer drift fails loudly.

use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

pub use crate::models::glm::prompt::{render_chat, ChatMessage, ChatRole, ChatTemplateOptions};

const MAX_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_VOCABULARY: usize = 1_000_000;
const MAX_MERGES: usize = 2_000_000;
const GLM_SPLIT_PATTERN: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN_SPLIT_PATTERN: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Unsupported(String),
    Invalid(String),
    UnknownTokenId(u32),
    InvalidByteToken { token: u32, character: char },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Json(error) => error.fmt(f),
            Self::Unsupported(reason) => write!(f, "unsupported tokenizer: {reason}"),
            Self::Invalid(reason) => write!(f, "invalid tokenizer: {reason}"),
            Self::UnknownTokenId(token) => write!(f, "unknown tokenizer ID {token}"),
            Self::InvalidByteToken { token, character } => write!(
                f,
                "token ID {token} contains non-byte-level character {character:?}"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TokenizerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerFile {
    version: String,
    added_tokens: Vec<AddedTokenFile>,
    normalizer: Option<Value>,
    pre_tokenizer: Value,
    post_processor: Value,
    decoder: Value,
    model: BpeFile,
}

#[derive(Debug, Deserialize)]
struct AddedTokenFile {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    #[serde(rename = "normalized")]
    _normalized: bool,
    special: bool,
}

#[derive(Debug, Deserialize)]
struct BpeFile {
    #[serde(rename = "type")]
    kind: String,
    dropout: Option<f64>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: Option<bool>,
    byte_fallback: bool,
    ignore_merges: Option<bool>,
    vocab: HashMap<String, u32>,
    merges: Vec<MergeFile>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeFile {
    Pair([String; 2]),
    Text(String),
}

impl MergeFile {
    fn into_pair(self, rank: usize) -> Result<[String; 2], TokenizerError> {
        match self {
            Self::Pair(pair) => Ok(pair),
            Self::Text(text) => {
                let (left, right) = text.split_once(' ').ok_or_else(|| {
                    TokenizerError::Invalid(format!(
                        "merge rank {rank} string has no token separator"
                    ))
                })?;
                if left.is_empty() || right.is_empty() || right.contains(' ') {
                    return Err(TokenizerError::Invalid(format!(
                        "merge rank {rank} string does not contain exactly two tokens"
                    )));
                }
                Ok([left.to_owned(), right.to_owned()])
            }
        }
    }
}

#[derive(Debug, Clone)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

#[derive(Debug, Clone, Copy)]
struct Merge {
    rank: usize,
    result: u32,
}

#[derive(Debug, Clone)]
enum TokenEntry {
    ByteLevel(String),
    Added { content: String, special: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PretokenizerKind {
    Glm52,
    DeepseekV4,
    Qwen38,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizerKind {
    None,
    Nfc,
}

/// Byte-level BPE plus model-native pre-tokenization and added vocabulary.
#[derive(Debug, Clone)]
pub struct ByteBpeTokenizer {
    vocab: HashMap<String, u32>,
    tokens: Vec<Option<TokenEntry>>,
    added: Vec<AddedToken>,
    merges: HashMap<(u32, u32), Merge>,
    byte_token_ids: [u32; 256],
    byte_characters: [char; 256],
    character_bytes: HashMap<char, u8>,
    ignore_merges: bool,
    pretokenizer: PretokenizerKind,
    normalizer: NormalizerKind,
}

/// Architecture-neutral name for the shared byte-level BPE engine.
/// Backward-compatible name for callers that used the original GLM-only API.
pub type GlmTokenizer = ByteBpeTokenizer;

impl ByteBpeTokenizer {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let metadata = fs::metadata(&path)?;
        if metadata.len() > MAX_TOKENIZER_BYTES {
            return Err(TokenizerError::Invalid(format!(
                "{} is {} bytes; limit is {MAX_TOKENIZER_BYTES}",
                path.display(),
                metadata.len()
            )));
        }
        let json = fs::read_to_string(path)?;
        Self::from_json_str(&json)
    }

    pub fn from_json_str(json: &str) -> Result<Self, TokenizerError> {
        if json.len() as u64 > MAX_TOKENIZER_BYTES {
            return Err(TokenizerError::Invalid(format!(
                "tokenizer JSON is {} bytes; limit is {MAX_TOKENIZER_BYTES}",
                json.len()
            )));
        }
        let file: TokenizerFile = serde_json::from_str(json)?;
        let (pretokenizer, normalizer) = validate_file_contract(&file)?;

        let BpeFile {
            ignore_merges,
            vocab,
            merges: merge_files,
            ..
        } = file.model;
        let maximum_id = vocab
            .values()
            .copied()
            .chain(file.added_tokens.iter().map(|token| token.id))
            .max()
            .ok_or_else(|| TokenizerError::Invalid("vocabulary is empty".to_owned()))?;
        let token_slots = usize::try_from(maximum_id)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                TokenizerError::Invalid("maximum token ID overflows usize".to_owned())
            })?;
        if token_slots > MAX_VOCABULARY {
            return Err(TokenizerError::Invalid(format!(
                "token ID space has {token_slots} entries; limit is {MAX_VOCABULARY}"
            )));
        }
        let mut tokens = vec![None; token_slots];
        for (token, &id) in &vocab {
            insert_token(&mut tokens, id, TokenEntry::ByteLevel(token.clone()), token)?;
        }

        let mut added = Vec::with_capacity(file.added_tokens.len());
        for token in file.added_tokens {
            let entry = TokenEntry::Added {
                content: token.content.clone(),
                special: token.special,
            };
            let slot = tokens.get_mut(token.id as usize).ok_or_else(|| {
                TokenizerError::Invalid(format!("token ID {} is outside ID space", token.id))
            })?;
            match slot.as_ref() {
                Some(TokenEntry::ByteLevel(content)) if content == &token.content => {
                    *slot = Some(entry);
                }
                Some(_) => {
                    return Err(TokenizerError::Invalid(format!(
                        "token ID {} is assigned incompatible content {:?}",
                        token.id, token.content
                    )));
                }
                None => *slot = Some(entry),
            }
            added.push(AddedToken {
                id: token.id,
                content: token.content,
                special: token.special,
            });
        }
        added.sort_by(|left, right| {
            right
                .content
                .len()
                .cmp(&left.content.len())
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut merge_map = HashMap::with_capacity(merge_files.len());
        for (rank, merge) in merge_files.into_iter().enumerate() {
            let [left, right] = merge.into_pair(rank)?;
            let left_id = *vocab.get(&left).ok_or_else(|| {
                TokenizerError::Invalid(format!(
                    "merge rank {rank} has unknown left token {left:?}"
                ))
            })?;
            let right_id = *vocab.get(&right).ok_or_else(|| {
                TokenizerError::Invalid(format!(
                    "merge rank {rank} has unknown right token {right:?}"
                ))
            })?;
            let result_text = format!("{left}{right}");
            let result = *vocab.get(&result_text).ok_or_else(|| {
                TokenizerError::Invalid(format!(
                    "merge rank {rank} result {result_text:?} is absent from vocabulary"
                ))
            })?;
            if merge_map
                .insert((left_id, right_id), Merge { rank, result })
                .is_some()
            {
                return Err(TokenizerError::Invalid(format!(
                    "duplicate merge pair {left:?} + {right:?}"
                )));
            }
        }

        let (byte_characters, character_bytes) = byte_level_alphabet();
        let mut byte_token_ids = [0u32; 256];
        for (byte, character) in byte_characters.iter().enumerate() {
            byte_token_ids[byte] = *vocab.get(&character.to_string()).ok_or_else(|| {
                TokenizerError::Invalid(format!(
                    "byte-level alphabet token {character:?} for byte {byte} is missing"
                ))
            })?;
        }
        Ok(Self {
            vocab,
            tokens,
            added,
            merges: merge_map,
            byte_token_ids,
            byte_characters,
            character_bytes,
            ignore_merges: ignore_merges.unwrap_or(false),
            pretokenizer,
            normalizer,
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the checkpoint spelling of one vocabulary entry before byte-level decoding.
    ///
    /// DeepSeek-V4.1 Engram uses this only for tokens whose one-token decode contains U+FFFD:
    /// those are partial UTF-8 byte pieces and the trained hash table keys them by this raw form.
    pub(crate) fn raw_token_content(&self, id: u32) -> Result<&str, TokenizerError> {
        match self
            .tokens
            .get(id as usize)
            .and_then(Option::as_ref)
            .ok_or(TokenizerError::UnknownTokenId(id))?
        {
            TokenEntry::ByteLevel(content) => Ok(content),
            TokenEntry::Added { content, .. } => Ok(content),
        }
    }

    /// Returns whether an LM-head row has a tokenizer entry and can therefore be decoded.
    pub fn is_decodable_token_id(&self, id: u32) -> bool {
        self.tokens.get(id as usize).is_some_and(Option::is_some)
    }

    /// Builds a mask over the model vocabulary so reserved LM-head rows are never sampled.
    pub fn decodable_token_mask(&self, model_vocabulary_size: usize) -> Vec<bool> {
        (0..model_vocabulary_size)
            .map(|id| self.tokens.get(id).is_some_and(Option::is_some))
            .collect()
    }

    pub fn token_id(&self, content: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|token| token.content == content)
            .map(|token| token.id)
            .or_else(|| self.vocab.get(content).copied())
    }

    /// Rejects text that would inject a reserved protocol token into a rendered chat turn.
    pub fn validate_chat_content(&self, text: &str) -> Result<(), TokenizerError> {
        if let Some(token) = self
            .added
            .iter()
            .filter(|token| token.special)
            .find(|token| text.contains(&token.content))
        {
            return Err(TokenizerError::Invalid(format!(
                "chat content contains reserved token {:?}; use an explicitly trusted raw prompt if intentional",
                token.content
            )));
        }
        Ok(())
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            let next = self.next_added_token(text, cursor);
            match next {
                Some((position, token)) => {
                    if position > cursor {
                        self.encode_plain(&text[cursor..position], &mut output)?;
                    }
                    output.push(token.id);
                    cursor = position + token.content.len();
                }
                None => {
                    self.encode_plain(&text[cursor..], &mut output)?;
                    cursor = text.len();
                }
            }
        }
        Ok(output)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let mut decoder = self.streaming_decoder(skip_special_tokens);
        let mut output = String::new();
        for &id in ids {
            output.push_str(&decoder.push(id)?);
        }
        output.push_str(&decoder.finish());
        Ok(output)
    }

    /// Creates an incremental decoder that never emits a partial UTF-8 code point.
    pub fn streaming_decoder(&self, skip_special_tokens: bool) -> StreamingDecoder<'_> {
        StreamingDecoder {
            tokenizer: self,
            pending_bytes: Vec::new(),
            skip_special_tokens,
        }
    }

    fn next_added_token<'a>(
        &'a self,
        text: &str,
        cursor: usize,
    ) -> Option<(usize, &'a AddedToken)> {
        let tail = &text[cursor..];
        self.added
            .iter()
            .filter_map(|token| {
                tail.find(&token.content)
                    .map(|offset| (cursor + offset, token))
            })
            .min_by(|(left_position, left), (right_position, right)| {
                left_position
                    .cmp(right_position)
                    .then_with(|| right.content.len().cmp(&left.content.len()))
                    .then_with(|| left.id.cmp(&right.id))
            })
    }

    fn encode_plain(&self, text: &str, output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let normalized = match self.normalizer {
            NormalizerKind::None => None,
            NormalizerKind::Nfc => Some(text.nfc().collect::<String>()),
        };
        let text = normalized.as_deref().unwrap_or(text);
        let mut cursor = 0usize;
        while cursor < text.len() {
            let end = match self.pretokenizer {
                PretokenizerKind::Glm52 => pretoken_end(text, cursor),
                PretokenizerKind::DeepseekV4 => deepseek_pretoken_end(text, cursor),
                PretokenizerKind::Qwen38 => qwen_pretoken_end(text, cursor),
            };
            if end <= cursor || !text.is_char_boundary(end) {
                return Err(TokenizerError::Invalid(
                    "pre-tokenizer failed to make UTF-8 progress".to_owned(),
                ));
            }
            output.extend(self.encode_piece(&text[cursor..end])?);
            cursor = end;
        }
        Ok(())
    }

    fn encode_piece(&self, piece: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoded: String = piece
            .as_bytes()
            .iter()
            .map(|byte| self.byte_characters[*byte as usize])
            .collect();
        if self.ignore_merges {
            if let Some(&token) = self.vocab.get(&encoded) {
                return Ok(vec![token]);
            }
        }
        let mut symbols: Vec<u32> = piece
            .as_bytes()
            .iter()
            .map(|byte| self.byte_token_ids[*byte as usize])
            .collect();
        while symbols.len() > 1 {
            let best = symbols
                .windows(2)
                .filter_map(|pair| self.merges.get(&(pair[0], pair[1])).copied())
                .min_by_key(|merge| merge.rank);
            let Some(best) = best else {
                break;
            };
            let mut merged = Vec::with_capacity(symbols.len());
            let mut index = 0usize;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && self
                        .merges
                        .get(&(symbols[index], symbols[index + 1]))
                        .is_some_and(|candidate| candidate.rank == best.rank)
                {
                    merged.push(best.result);
                    index += 2;
                } else {
                    merged.push(symbols[index]);
                    index += 1;
                }
            }
            symbols = merged;
        }
        Ok(symbols)
    }
}

/// Incremental byte-level decoder for token-at-a-time generation.
///
/// Byte-level BPE tokens can split one Unicode scalar across several token IDs. `push` therefore
/// returns only the complete text available so far, and `finish` lossily flushes a final truncated
/// byte sequence using the same replacement semantics as [`ByteBpeTokenizer::decode`].
#[derive(Debug)]
pub struct StreamingDecoder<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    pending_bytes: Vec<u8>,
    skip_special_tokens: bool,
}

impl StreamingDecoder<'_> {
    pub fn push(&mut self, id: u32) -> Result<String, TokenizerError> {
        let entry = self
            .tokenizer
            .tokens
            .get(id as usize)
            .and_then(Option::as_ref)
            .ok_or(TokenizerError::UnknownTokenId(id))?;
        match entry {
            TokenEntry::ByteLevel(token) => {
                for character in token.chars() {
                    self.pending_bytes.push(
                        *self.tokenizer.character_bytes.get(&character).ok_or(
                            TokenizerError::InvalidByteToken {
                                token: id,
                                character,
                            },
                        )?,
                    );
                }
                Ok(drain_decodable_prefix(&mut self.pending_bytes))
            }
            TokenEntry::Added { content, special } => {
                if self.skip_special_tokens && *special {
                    // Batch decoding removes special tokens before byte-level decoding. Preserve
                    // pending bytes so the surrounding ordinary tokens retain identical UTF-8
                    // semantics in the streaming path.
                    Ok(String::new())
                } else {
                    let mut output = flush_pending_lossy(&mut self.pending_bytes);
                    output.push_str(content);
                    Ok(output)
                }
            }
        }
    }

    pub fn finish(&mut self) -> String {
        flush_pending_lossy(&mut self.pending_bytes)
    }

    pub fn pending_byte_count(&self) -> usize {
        self.pending_bytes.len()
    }
}

fn validate_file_contract(
    file: &TokenizerFile,
) -> Result<(PretokenizerKind, NormalizerKind), TokenizerError> {
    if file.version != "1.0" {
        return Err(TokenizerError::Unsupported(format!(
            "tokenizer JSON version {:?}; expected 1.0",
            file.version
        )));
    }
    let normalizer = validate_normalizer_contract(file.normalizer.as_ref())?;
    let pretokenizer =
        validate_component_contract(&file.pre_tokenizer, &file.post_processor, &file.decoder)?;
    match (pretokenizer, normalizer) {
        (PretokenizerKind::Qwen38, NormalizerKind::Nfc)
        | (PretokenizerKind::Glm52 | PretokenizerKind::DeepseekV4, NormalizerKind::None) => {}
        (PretokenizerKind::Qwen38, NormalizerKind::None) => {
            return Err(TokenizerError::Unsupported(
                "Qwen3.8 tokenizer requires the official NFC normalizer".to_owned(),
            ));
        }
        (_, NormalizerKind::Nfc) => {
            return Err(TokenizerError::Unsupported(
                "NFC normalization is supported only for the Qwen3.8 tokenizer contract".to_owned(),
            ));
        }
    }
    let model = &file.model;
    if model.kind != "BPE"
        || model.dropout.is_some()
        || model.unk_token.is_some()
        || model
            .continuing_subword_prefix
            .as_deref()
            .is_some_and(|prefix| !prefix.is_empty())
        || model
            .end_of_word_suffix
            .as_deref()
            .is_some_and(|suffix| !suffix.is_empty())
        || model.fuse_unk == Some(true)
        || model.byte_fallback
    {
        return Err(TokenizerError::Unsupported(
            "expected deterministic byte-level BPE without unk/dropout/non-empty prefix or suffix/fallback"
                .to_owned(),
        ));
    }
    if model.vocab.is_empty() || model.vocab.len() > MAX_VOCABULARY {
        return Err(TokenizerError::Invalid(format!(
            "vocabulary count {} is outside 1..={MAX_VOCABULARY}",
            model.vocab.len()
        )));
    }
    if model.merges.len() > MAX_MERGES {
        return Err(TokenizerError::Invalid(format!(
            "merge count {} exceeds {MAX_MERGES}",
            model.merges.len()
        )));
    }
    let mut ids = HashSet::with_capacity(model.vocab.len() + file.added_tokens.len());
    for (token, &id) in &model.vocab {
        if !ids.insert(id) {
            return Err(TokenizerError::Invalid(format!(
                "duplicate vocabulary ID {id}, including token {token:?}"
            )));
        }
    }
    let mut contents: HashSet<&str> = model.vocab.keys().map(String::as_str).collect();
    for token in &file.added_tokens {
        if token.content.is_empty() {
            return Err(TokenizerError::Invalid(
                "added token content must not be empty".to_owned(),
            ));
        }
        if token.single_word || token.lstrip || token.rstrip {
            return Err(TokenizerError::Unsupported(format!(
                "added token {:?} requests unsupported boundary behavior",
                token.content
            )));
        }
        let duplicates_matching_vocab = model.vocab.get(&token.content) == Some(&token.id);
        if !duplicates_matching_vocab && (!ids.insert(token.id) || !contents.insert(&token.content))
        {
            return Err(TokenizerError::Invalid(format!(
                "added token {:?} has a duplicate ID or content",
                token.content
            )));
        }
    }
    Ok((pretokenizer, normalizer))
}

fn validate_normalizer_contract(
    normalizer: Option<&Value>,
) -> Result<NormalizerKind, TokenizerError> {
    let Some(normalizer) = normalizer else {
        return Ok(NormalizerKind::None);
    };
    if normalizer.get("type").and_then(Value::as_str) == Some("NFC") {
        return Ok(NormalizerKind::Nfc);
    }
    let empty_sequence = normalizer.get("type").and_then(Value::as_str) == Some("Sequence")
        && normalizer
            .get("normalizers")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty);
    if empty_sequence {
        return Ok(NormalizerKind::None);
    }
    Err(TokenizerError::Unsupported(
        "only no normalizer, an empty normalizer Sequence, or NFC is supported".to_owned(),
    ))
}

fn validate_component_contract(
    pre_tokenizer: &Value,
    post_processor: &Value,
    decoder: &Value,
) -> Result<PretokenizerKind, TokenizerError> {
    let sequence = pre_tokenizer
        .get("pretokenizers")
        .and_then(Value::as_array)
        .ok_or_else(|| TokenizerError::Unsupported("pre-tokenizer is not a Sequence".to_owned()))?;
    if pre_tokenizer.get("type").and_then(Value::as_str) != Some("Sequence") {
        return Err(TokenizerError::Unsupported(
            "pre-tokenizer is not a Sequence".to_owned(),
        ));
    }
    let kind = if sequence.len() == 2
        && valid_split(&sequence[0])
        && sequence[0]
            .pointer("/pattern/Regex")
            .and_then(Value::as_str)
            == Some(GLM_SPLIT_PATTERN)
        && valid_byte_level(&sequence[1])
    {
        PretokenizerKind::Glm52
    } else if sequence.len() == 2
        && valid_split(&sequence[0])
        && sequence[0]
            .pointer("/pattern/Regex")
            .and_then(Value::as_str)
            == Some(QWEN_SPLIT_PATTERN)
        && valid_byte_level(&sequence[1])
    {
        PretokenizerKind::Qwen38
    } else if sequence.len() == 4
        && sequence[..3].iter().all(valid_split)
        && sequence[0]
            .pointer("/pattern/Regex")
            .and_then(Value::as_str)
            == Some("\\p{N}{1,3}")
        && sequence[1]
            .pointer("/pattern/Regex")
            .and_then(Value::as_str)
            == Some("[一-龥぀-ゟ゠-ヿ]+")
        && sequence[2]
            .pointer("/pattern/Regex")
            .and_then(Value::as_str)
            .is_some_and(|pattern| pattern.contains("[\\p{L}\\p{M}]+"))
        && valid_byte_level(&sequence[3])
    {
        PretokenizerKind::DeepseekV4
    } else {
        return Err(TokenizerError::Unsupported(
            "pre-tokenizer is not a supported GLM-5.2, DeepSeek-V4, or Qwen3.8 Split + ByteLevel contract"
                .to_owned(),
        ));
    };
    for (name, component) in [("post-processor", post_processor), ("decoder", decoder)] {
        if component.get("type").and_then(Value::as_str) != Some("ByteLevel") {
            return Err(TokenizerError::Unsupported(format!(
                "{name} must be ByteLevel"
            )));
        }
    }
    Ok(kind)
}

fn valid_split(component: &Value) -> bool {
    component.get("type").and_then(Value::as_str) == Some("Split")
        && component.get("behavior").and_then(Value::as_str) == Some("Isolated")
        && component.get("invert").and_then(Value::as_bool) == Some(false)
}

fn valid_byte_level(component: &Value) -> bool {
    component.get("type").and_then(Value::as_str) == Some("ByteLevel")
        && component.get("add_prefix_space").and_then(Value::as_bool) == Some(false)
        && component.get("use_regex").and_then(Value::as_bool) == Some(false)
}

fn insert_token(
    slots: &mut [Option<TokenEntry>],
    id: u32,
    entry: TokenEntry,
    content: &str,
) -> Result<(), TokenizerError> {
    let slot = slots
        .get_mut(id as usize)
        .ok_or_else(|| TokenizerError::Invalid(format!("token ID {id} is outside ID space")))?;
    if slot.replace(entry).is_some() {
        return Err(TokenizerError::Invalid(format!(
            "token ID {id} is assigned more than once, including {content:?}"
        )));
    }
    Ok(())
}

fn byte_level_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut codepoints: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
    let present: HashSet<u8> = bytes.iter().copied().collect();
    let mut extra = 0u32;
    for byte in 0u8..=u8::MAX {
        if !present.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + extra);
            extra += 1;
        }
    }
    let mut forward = ['\0'; 256];
    let mut reverse = HashMap::with_capacity(256);
    for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
        let character = char::from_u32(codepoint).expect("byte alphabet is valid Unicode");
        forward[byte as usize] = character;
        reverse.insert(character, byte);
    }
    (forward, reverse)
}

fn drain_decodable_prefix(bytes: &mut Vec<u8>) -> String {
    let mut output = String::new();
    let mut consumed = 0usize;
    while consumed < bytes.len() {
        match std::str::from_utf8(&bytes[consumed..]) {
            Ok(text) => {
                output.push_str(text);
                consumed = bytes.len();
            }
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    output.push_str(
                        std::str::from_utf8(&bytes[consumed..consumed + valid])
                            .expect("from_utf8 reports this prefix as valid"),
                    );
                    consumed += valid;
                }
                let Some(invalid) = error.error_len() else {
                    break;
                };
                output.push('\u{fffd}');
                consumed += invalid;
            }
        }
    }
    if consumed > 0 {
        bytes.drain(..consumed);
    }
    output
}

fn flush_pending_lossy(bytes: &mut Vec<u8>) -> String {
    let output = String::from_utf8_lossy(bytes).into_owned();
    bytes.clear();
    output
}

fn pretoken_end(text: &str, start: usize) -> usize {
    let tail = &text[start..];
    if let Some(length) = contraction_length(tail) {
        return start + length;
    }

    let mut chars = tail.char_indices();
    let Some((_, first)) = chars.next() else {
        return start;
    };
    if is_letter(first) {
        return start + consume_while(tail, 0, is_letter);
    }
    if first != '\r' && first != '\n' && !is_letter(first) && !is_number(first) {
        let prefix_end = first.len_utf8();
        if tail[prefix_end..].chars().next().is_some_and(is_letter) {
            return start + consume_while(tail, prefix_end, is_letter);
        }
    }
    if is_number(first) {
        let mut end = first.len_utf8();
        for character in tail[end..].chars().take(2) {
            if !is_number(character) {
                break;
            }
            end += character.len_utf8();
        }
        return start + end;
    }

    let punctuation_start = if first == ' ' {
        let next = tail[first.len_utf8()..].chars().next();
        next.filter(|character| is_non_word_punctuation(*character))
            .map(|_| first.len_utf8())
    } else if is_non_word_punctuation(first) {
        Some(0)
    } else {
        None
    };
    if let Some(punctuation_start) = punctuation_start {
        let mut end = consume_while(tail, punctuation_start, is_non_word_punctuation);
        while let Some(character) = tail[end..].chars().next() {
            if !matches!(character, '\r' | '\n') {
                break;
            }
            end += character.len_utf8();
        }
        return start + end;
    }

    if first.is_whitespace() {
        let mut run_end = 0usize;
        let mut last_newline_end = None;
        let mut boundaries = Vec::new();
        for (offset, character) in tail.char_indices() {
            if !character.is_whitespace() {
                break;
            }
            run_end = offset + character.len_utf8();
            boundaries.push(run_end);
            if matches!(character, '\r' | '\n') {
                last_newline_end = Some(run_end);
            }
        }
        if let Some(end) = last_newline_end {
            return start + end;
        }
        if run_end < tail.len() && boundaries.len() > 1 {
            return start + boundaries[boundaries.len() - 2];
        }
        return start + run_end;
    }

    start + first.len_utf8()
}

fn deepseek_pretoken_end(text: &str, start: usize) -> usize {
    let tail = &text[start..];
    let mut chars = tail.chars();
    let Some(first) = chars.next() else {
        return start;
    };

    if is_number(first) {
        let mut end = 0usize;
        for character in tail.chars().take(3) {
            if !is_number(character) {
                break;
            }
            end += character.len_utf8();
        }
        return start + end;
    }
    if is_deepseek_cjk(first) {
        return start + consume_while(tail, 0, is_deepseek_cjk);
    }

    let first_len = first.len_utf8();
    if first.is_ascii_punctuation()
        && tail[first_len..]
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic())
    {
        let end = consume_while(tail, first_len, |character| character.is_ascii_alphabetic());
        return start + end;
    }

    if is_letter(first) || is_mark(first) {
        return start + consume_while(tail, 0, is_letter_or_mark);
    }
    if !matches!(first, '\r' | '\n')
        && !is_letter(first)
        && !is_punctuation_or_symbol(first)
        && tail[first_len..]
            .chars()
            .next()
            .is_some_and(is_letter_or_mark)
    {
        return start + consume_while(tail, first_len, is_letter_or_mark);
    }

    let punctuation_start = if first == ' '
        && tail[first_len..]
            .chars()
            .next()
            .is_some_and(is_punctuation_or_symbol)
    {
        Some(first_len)
    } else if is_punctuation_or_symbol(first) {
        Some(0)
    } else {
        None
    };
    if let Some(punctuation_start) = punctuation_start {
        let mut end = consume_while(tail, punctuation_start, is_punctuation_or_symbol);
        while let Some(character) = tail[end..].chars().next() {
            if !matches!(character, '\r' | '\n') {
                break;
            }
            end += character.len_utf8();
        }
        return start + end;
    }

    if first.is_whitespace() {
        let mut run_end = 0usize;
        let mut last_newline_end = None;
        let mut boundaries = Vec::new();
        for (offset, character) in tail.char_indices() {
            if !character.is_whitespace() {
                break;
            }
            run_end = offset + character.len_utf8();
            boundaries.push(run_end);
            if matches!(character, '\r' | '\n') {
                last_newline_end = Some(run_end);
            }
        }
        if let Some(end) = last_newline_end {
            return start + end;
        }
        if run_end < tail.len() && boundaries.len() > 1 {
            return start + boundaries[boundaries.len() - 2];
        }
        return start + run_end;
    }

    start + first_len
}

fn qwen_pretoken_end(text: &str, start: usize) -> usize {
    let tail = &text[start..];
    if let Some(length) = contraction_length(tail) {
        return start + length;
    }

    let mut chars = tail.chars();
    let Some(first) = chars.next() else {
        return start;
    };
    let first_len = first.len_utf8();

    if is_letter_or_mark(first) {
        return start + consume_while(tail, 0, is_letter_or_mark);
    }
    if !matches!(first, '\r' | '\n')
        && !is_letter(first)
        && !is_number(first)
        && tail[first_len..]
            .chars()
            .next()
            .is_some_and(is_letter_or_mark)
    {
        return start + consume_while(tail, first_len, is_letter_or_mark);
    }

    // Qwen deliberately isolates every Unicode number, unlike the GLM three-digit grouping.
    if is_number(first) {
        return start + first_len;
    }

    let punctuation_start = if first == ' '
        && tail[first_len..]
            .chars()
            .next()
            .is_some_and(is_qwen_punctuation)
    {
        Some(first_len)
    } else if is_qwen_punctuation(first) {
        Some(0)
    } else {
        None
    };
    if let Some(punctuation_start) = punctuation_start {
        let mut end = consume_while(tail, punctuation_start, is_qwen_punctuation);
        while let Some(character) = tail[end..].chars().next() {
            if !matches!(character, '\r' | '\n') {
                break;
            }
            end += character.len_utf8();
        }
        return start + end;
    }

    if first.is_whitespace() {
        let mut run_end = 0usize;
        let mut last_newline_end = None;
        let mut boundaries = Vec::new();
        for (offset, character) in tail.char_indices() {
            if !character.is_whitespace() {
                break;
            }
            run_end = offset + character.len_utf8();
            boundaries.push(run_end);
            if matches!(character, '\r' | '\n') {
                last_newline_end = Some(run_end);
            }
        }
        if let Some(end) = last_newline_end {
            return start + end;
        }
        if run_end < tail.len() && boundaries.len() > 1 {
            return start + boundaries[boundaries.len() - 2];
        }
        return start + run_end;
    }

    start + first_len
}

fn contraction_length(text: &str) -> Option<usize> {
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let mut input = text.char_indices();
        if input.next().map(|(_, character)| character) != Some('\'') {
            return None;
        }
        let mut end = 1usize;
        let mut matched = true;
        for expected in suffix.chars() {
            let Some((offset, actual)) = input.next() else {
                matched = false;
                break;
            };
            if !simple_fold_matches_ascii(actual, expected) {
                matched = false;
                break;
            }
            end = offset + actual.len_utf8();
        }
        if matched {
            return Some(end);
        }
    }
    None
}

fn simple_fold_matches_ascii(actual: char, expected: char) -> bool {
    actual.eq_ignore_ascii_case(&expected) || (expected == 's' && actual == '\u{017f}')
}

fn consume_while(text: &str, start: usize, predicate: fn(char) -> bool) -> usize {
    let mut end = start;
    for character in text[start..].chars() {
        if !predicate(character) {
            break;
        }
        end += character.len_utf8();
    }
    end
}

fn is_letter(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_number(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn is_mark(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

fn is_letter_or_mark(character: char) -> bool {
    is_letter(character) || is_mark(character)
}

fn is_punctuation_or_symbol(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::MathSymbol
            | GeneralCategory::CurrencySymbol
            | GeneralCategory::ModifierSymbol
            | GeneralCategory::OtherSymbol
    )
}

fn is_deepseek_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{4e00}'..='\u{9fa5}' | '\u{3040}'..='\u{309f}' | '\u{30a0}'..='\u{30ff}'
    )
}

fn is_non_word_punctuation(character: char) -> bool {
    !character.is_whitespace() && !is_letter(character) && !is_number(character)
}

fn is_qwen_punctuation(character: char) -> bool {
    !character.is_whitespace()
        && !is_letter(character)
        && !is_mark(character)
        && !is_number(character)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer_json() -> String {
        let (alphabet, _) = byte_level_alphabet();
        let mut vocab = serde_json::Map::new();
        for (id, character) in alphabet.into_iter().enumerate() {
            vocab.insert(character.to_string(), Value::from(id as u32));
        }
        // A compact merge chain for " hello".
        let mut next = 256u32;
        for token in ["he", "hel", "hell", "hello", "Ġhello"] {
            vocab.insert(token.to_owned(), Value::from(next));
            next += 1;
        }
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": next,
                "content": "<|user|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type":"Split", "pattern":{"Regex":GLM_SPLIT_PATTERN}, "behavior":"Isolated", "invert":false},
                    {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false}
                ]
            },
            "post_processor": {"type":"ByteLevel", "add_prefix_space":true, "trim_offsets":false, "use_regex":true},
            "decoder": {"type":"ByteLevel", "add_prefix_space":true, "trim_offsets":true, "use_regex":true},
            "model": {
                "type":"BPE", "dropout":null, "unk_token":null,
                "continuing_subword_prefix":null, "end_of_word_suffix":null,
                "fuse_unk":false, "byte_fallback":false, "ignore_merges":true,
                "vocab":vocab,
                "merges":[["h","e"],["he","l"],["hel","l"],["hell","o"],["Ġ","hello"]]
            }
        })
        .to_string()
    }

    fn tiny_qwen_tokenizer_json() -> String {
        let mut value: Value = serde_json::from_str(&tiny_tokenizer_json()).unwrap();
        value["normalizer"] = serde_json::json!({"type":"NFC"});
        value["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
            Value::from(QWEN_SPLIT_PATTERN);
        value["pre_tokenizer"]["pretokenizers"][1]["trim_offsets"] = Value::from(false);
        value["post_processor"] = serde_json::json!({
            "type":"ByteLevel", "add_prefix_space":false,
            "trim_offsets":false, "use_regex":false
        });
        value["decoder"] = serde_json::json!({
            "type":"ByteLevel", "add_prefix_space":false,
            "trim_offsets":false, "use_regex":false
        });
        value["model"]["continuing_subword_prefix"] = Value::from("");
        value["model"]["end_of_word_suffix"] = Value::from("");
        value["model"]["ignore_merges"] = Value::from(false);
        value.to_string()
    }

    #[test]
    fn byte_bpe_roundtrips_unicode_and_added_tokens() {
        let tokenizer = GlmTokenizer::from_json_str(&tiny_tokenizer_json()).unwrap();
        let text = "<|user|> hello, 世界🚀";
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
        assert_eq!(tokenizer.decode(&ids, true).unwrap(), " hello, 世界🚀");
        assert_eq!(tokenizer.token_id("<|user|>"), Some(261));
    }

    #[test]
    fn qwen_contract_accepts_empty_affixes_and_normalizes_plain_text_to_nfc() {
        let tokenizer = ByteBpeTokenizer::from_json_str(&tiny_qwen_tokenizer_json()).unwrap();
        let decomposed = "Cafe\u{301} <|user|> nai\u{308}ve";
        let ids = tokenizer.encode(decomposed).unwrap();
        assert_eq!(
            tokenizer.decode(&ids, false).unwrap(),
            "Caf\u{e9} <|user|> na\u{ef}ve"
        );
        assert_eq!(
            tokenizer.decode(&ids, true).unwrap(),
            "Caf\u{e9}  na\u{ef}ve"
        );
    }

    #[test]
    fn qwen_pretokenizer_isolates_numbers_and_keeps_letters_with_marks() {
        let text = "I'm cafe\u{301} 12\u{2163}\r\n next";
        let mut cursor = 0;
        let mut pieces = Vec::new();
        while cursor < text.len() {
            let end = qwen_pretoken_end(text, cursor);
            pieces.push(&text[cursor..end]);
            cursor = end;
        }
        assert_eq!(
            pieces,
            vec![
                "I",
                "'m",
                " cafe\u{301}",
                " ",
                "1",
                "2",
                "\u{2163}",
                "\r\n",
                " next"
            ]
        );
    }

    #[test]
    fn qwen_contract_fails_closed_on_normalizer_or_affix_drift() {
        let mut value: Value = serde_json::from_str(&tiny_qwen_tokenizer_json()).unwrap();
        value["normalizer"] = serde_json::json!({"type":"NFD"});
        assert!(matches!(
            ByteBpeTokenizer::from_json_str(&value.to_string()),
            Err(TokenizerError::Unsupported(reason)) if reason.contains("normalizer")
        ));

        let mut value: Value = serde_json::from_str(&tiny_qwen_tokenizer_json()).unwrap();
        value["normalizer"] = Value::Null;
        assert!(matches!(
            ByteBpeTokenizer::from_json_str(&value.to_string()),
            Err(TokenizerError::Unsupported(reason)) if reason.contains("requires the official NFC")
        ));

        let mut value: Value = serde_json::from_str(&tiny_qwen_tokenizer_json()).unwrap();
        value["model"]["continuing_subword_prefix"] = Value::from("##");
        assert!(matches!(
            ByteBpeTokenizer::from_json_str(&value.to_string()),
            Err(TokenizerError::Unsupported(reason)) if reason.contains("non-empty prefix")
        ));
    }

    #[test]
    fn streaming_decoder_buffers_split_utf8_and_special_boundaries() {
        let tokenizer = GlmTokenizer::from_json_str(&tiny_tokenizer_json()).unwrap();
        let ids = tokenizer.encode("¢世🚀").unwrap();
        let mut decoder = tokenizer.streaming_decoder(false);
        let mut streamed = String::new();
        let mut saw_pending = false;
        for id in ids.iter().copied() {
            streamed.push_str(&decoder.push(id).unwrap());
            saw_pending |= decoder.pending_byte_count() > 0;
        }
        streamed.push_str(&decoder.finish());
        assert!(saw_pending);
        assert_eq!(streamed, "¢世🚀");
        assert_eq!(streamed, tokenizer.decode(&ids, false).unwrap());

        let leading_byte = tokenizer.byte_token_ids[0xe4];
        let mut decoder = tokenizer.streaming_decoder(false);
        assert_eq!(decoder.push(leading_byte).unwrap(), "");
        assert_eq!(decoder.push(261).unwrap(), "�<|user|>");
        assert_eq!(decoder.finish(), "");

        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(tokenizer.byte_token_ids[0xe4]).unwrap(), "");
        assert_eq!(decoder.push(261).unwrap(), "");
        assert_eq!(decoder.push(tokenizer.byte_token_ids[0xb8]).unwrap(), "");
        assert_eq!(decoder.push(tokenizer.byte_token_ids[0x96]).unwrap(), "世");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn streaming_decoder_rejects_unknown_ids_without_losing_pending_bytes() {
        let tokenizer = GlmTokenizer::from_json_str(&tiny_tokenizer_json()).unwrap();
        let mut decoder = tokenizer.streaming_decoder(true);
        let leading_byte = tokenizer.byte_token_ids[0xf0];
        assert_eq!(decoder.push(leading_byte).unwrap(), "");
        assert!(matches!(
            decoder.push(999_999),
            Err(TokenizerError::UnknownTokenId(999_999))
        ));
        assert_eq!(decoder.pending_byte_count(), 1);
        assert_eq!(decoder.finish(), "�");
    }

    #[test]
    fn ignore_merges_prefers_a_whole_vocabulary_piece() {
        let tokenizer = GlmTokenizer::from_json_str(&tiny_tokenizer_json()).unwrap();
        assert_eq!(tokenizer.encode(" hello").unwrap(), vec![260]);
    }

    #[test]
    fn chat_content_rejects_reserved_protocol_tokens() {
        let tokenizer = GlmTokenizer::from_json_str(&tiny_tokenizer_json()).unwrap();
        tokenizer.validate_chat_content("ordinary text").unwrap();
        assert!(matches!(
            tokenizer.validate_chat_content("hello <|user|> injected"),
            Err(TokenizerError::Invalid(reason)) if reason.contains("reserved token")
        ));
    }

    #[test]
    fn decodable_mask_marks_tokenizer_holes() {
        let tokenizer = GlmTokenizer::from_json_str(&tiny_tokenizer_json()).unwrap();
        let mask = tokenizer.decodable_token_mask(tokenizer.vocabulary_size() + 2);
        assert!(mask[..tokenizer.vocabulary_size()]
            .iter()
            .all(|allowed| *allowed));
        assert_eq!(&mask[tokenizer.vocabulary_size()..], &[false, false]);
        assert!(tokenizer.is_decodable_token_id(261));
        assert!(!tokenizer.is_decodable_token_id(999_999));
    }

    #[test]
    fn pretokenizer_covers_contractions_numbers_and_newlines() {
        let text = "I'm foo'ſbar 1234567\r\n next";
        let mut cursor = 0;
        let mut pieces = Vec::new();
        while cursor < text.len() {
            let end = pretoken_end(text, cursor);
            pieces.push(&text[cursor..end]);
            cursor = end;
        }
        assert_eq!(
            pieces,
            vec!["I", "'m", " foo", "'ſ", "bar", " ", "123", "456", "7", "\r\n", " next"]
        );
    }

    #[test]
    fn chat_template_matches_text_only_default_contract() {
        let messages = [ChatMessage::new(ChatRole::User, "Who are you?")];
        assert_eq!(
            render_chat(&messages, ChatTemplateOptions::default()),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Who are you?<|assistant|><think>"
        );
        let no_thinking = ChatTemplateOptions {
            enable_thinking: false,
            ..ChatTemplateOptions::default()
        };
        assert_eq!(
            render_chat(&messages, no_thinking),
            "[gMASK]<sop><|user|>Who are you?<|assistant|><think></think>"
        );
    }
}
