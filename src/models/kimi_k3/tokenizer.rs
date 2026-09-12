//! Native Kimi-K3 tiktoken loader and byte-pair encoder.
//!
//! K3 ships `tiktoken.model` (raw token bytes plus ranks), not a Hugging Face
//! `tokenizer.json`. The 163,584 mergeable ranks are followed by 256 registered tokens.
//! Encoding ordinary message content and encoding trusted XTML controls are separate APIs so a
//! user-provided `<|end_of_msg|>` cannot terminate a message.

use super::prompt::{self, EncodeSegment, Message, PromptOptions};
use crate::tokenizer::TokenizerError;
use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::path::Path;
use unicode_general_category::{get_general_category, GeneralCategory};

pub const BASE_TOKEN_COUNT: usize = 163_584;
pub const ADDED_TOKEN_COUNT: usize = 256;
pub const VOCABULARY_SIZE: usize = BASE_TOKEN_COUNT + ADDED_TOKEN_COUNT;

pub const BOS_TOKEN_ID: u32 = 163_584;
/// The tokenizer metadata's `[EOS]`. K3 generation does not normally stop on this ID.
pub const TOKENIZER_EOS_TOKEN_ID: u32 = 163_585;
/// The language model's message boundary and configured generation stop token.
pub const END_OF_MESSAGE_TOKEN_ID: u32 = 163_586;
pub const OPEN_TOKEN_ID: u32 = 163_587;
pub const CLOSE_TOKEN_ID: u32 = 163_588;
pub const SEP_TOKEN_ID: u32 = 163_589;
pub const PAD_TOKEN_ID: u32 = 163_839;

const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAX_TOKEN_BYTES: usize = 1024 * 1024;

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedTokenConfig>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
}

#[derive(Debug, Clone)]
pub struct KimiK3Tokenizer {
    ranks: HashMap<Vec<u8>, u32>,
    base_tokens: Vec<Option<Vec<u8>>>,
    added_tokens: Vec<String>,
    special_to_id: HashMap<String, u32>,
    specials_longest_first: Vec<String>,
}

impl KimiK3Tokenizer {
    /// Loads the released `tiktoken.model` and `tokenizer_config.json` from a checkpoint.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let model_path = model_dir.as_ref().join("tiktoken.model");
        let config_path = model_dir.as_ref().join("tokenizer_config.json");
        check_size(&model_path, MAX_MODEL_BYTES)?;
        check_size(&config_path, MAX_CONFIG_BYTES)?;
        let model = fs::read_to_string(model_path)?;
        let config = fs::read_to_string(config_path)?;
        Self::from_model_and_config_str(&model, &config)
    }

    /// Parses the exact released K3 tokenizer geometry from in-memory sources.
    pub fn from_model_and_config_str(model: &str, config: &str) -> Result<Self, TokenizerError> {
        Self::from_sources(model, config, BASE_TOKEN_COUNT, true)
    }

    pub fn vocabulary_size(&self) -> usize {
        self.base_tokens.len() + self.added_tokens.len()
    }

    pub fn stop_token_id(&self) -> u32 {
        END_OF_MESSAGE_TOKEN_ID
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.special_to_id
            .get(token)
            .copied()
            .or_else(|| self.ranks.get(token.as_bytes()).copied())
    }

    pub fn is_special_token_id(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.base_tokens.len() && id < self.vocabulary_size()
    }

    pub fn is_decodable_token_id(&self, id: u32) -> bool {
        let id = id as usize;
        if id < self.base_tokens.len() {
            self.base_tokens[id].is_some()
        } else {
            id - self.base_tokens.len() < self.added_tokens.len()
        }
    }

    pub fn decodable_token_mask(&self, vocabulary_size: usize) -> Vec<bool> {
        (0..vocabulary_size)
            .map(|id| u32::try_from(id).is_ok_and(|id| self.is_decodable_token_id(id)))
            .collect()
    }

    /// Encodes untrusted ordinary text. Registered token spellings remain ordinary BPE text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_ordinary(text)
    }

    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_ordinary_into(text, &mut output)?;
        Ok(output)
    }

    /// Encodes a trusted model-native string, matching the official `allowed_special="all"`
    /// path. Never use this method directly on user or tool content.
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            let Some((position, token, id)) = self.next_special(text, cursor) else {
                self.encode_ordinary_into(&text[cursor..], &mut output)?;
                break;
            };
            self.encode_ordinary_into(&text[cursor..position], &mut output)?;
            output.push(id);
            cursor = position + token.len();
        }
        Ok(output)
    }

    /// Encodes XTML segments while preserving their trust boundary.
    pub fn encode_segments(&self, segments: &[EncodeSegment]) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        for segment in segments {
            if segment.allow_special {
                output.extend(self.encode_with_special_tokens(&segment.text)?);
            } else {
                self.encode_ordinary_into(&segment.text, &mut output)?;
            }
        }
        Ok(output)
    }

    /// Renders and tokenizes chat without ever joining trusted controls and untrusted content.
    pub fn encode_chat(
        &self,
        messages: &[Message],
        options: PromptOptions,
    ) -> Result<Vec<u32>, TokenizerError> {
        self.encode_segments(&prompt::render_chat_segments(messages, options)?)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let mut output = String::new();
        let mut bytes = Vec::new();
        for &id in ids {
            if (id as usize) < self.base_tokens.len() {
                let token = self.base_tokens[id as usize]
                    .as_deref()
                    .ok_or(TokenizerError::UnknownTokenId(id))?;
                bytes.extend_from_slice(token);
                continue;
            }
            let added = (id as usize)
                .checked_sub(self.base_tokens.len())
                .and_then(|index| self.added_tokens.get(index))
                .ok_or(TokenizerError::UnknownTokenId(id))?;
            if !skip_special_tokens {
                output.push_str(&String::from_utf8_lossy(&bytes));
                bytes.clear();
                output.push_str(added);
            }
        }
        output.push_str(&String::from_utf8_lossy(&bytes));
        Ok(output)
    }

    pub fn streaming_decoder(&self, skip_special_tokens: bool) -> KimiK3StreamingDecoder<'_> {
        KimiK3StreamingDecoder {
            tokenizer: self,
            pending_bytes: Vec::new(),
            skip_special_tokens,
        }
    }

    fn from_sources(
        model: &str,
        config: &str,
        base_token_count: usize,
        require_dense: bool,
    ) -> Result<Self, TokenizerError> {
        if model.len() as u64 > MAX_MODEL_BYTES {
            return Err(TokenizerError::Invalid(format!(
                "tiktoken.model is {} bytes; limit is {MAX_MODEL_BYTES}",
                model.len()
            )));
        }
        if config.len() as u64 > MAX_CONFIG_BYTES {
            return Err(TokenizerError::Invalid(format!(
                "tokenizer_config.json is {} bytes; limit is {MAX_CONFIG_BYTES}",
                config.len()
            )));
        }
        if base_token_count == 0 || base_token_count > 1_000_000 {
            return Err(TokenizerError::Invalid(format!(
                "invalid Kimi-K3 base token count {base_token_count}"
            )));
        }
        let mut ranks = HashMap::with_capacity(base_token_count);
        let mut base_tokens = vec![None; base_token_count];
        let mut parsed = 0usize;
        for (line_index, line) in model.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split_whitespace();
            let encoded = fields.next().expect("non-empty line has one field");
            let rank_text = fields.next().ok_or_else(|| {
                TokenizerError::Invalid(format!(
                    "tiktoken.model line {} has no rank",
                    line_index + 1
                ))
            })?;
            if fields.next().is_some() {
                return Err(TokenizerError::Invalid(format!(
                    "tiktoken.model line {} has extra fields",
                    line_index + 1
                )));
            }
            let rank: usize = rank_text.parse().map_err(|_| {
                TokenizerError::Invalid(format!(
                    "tiktoken.model line {} has invalid rank {rank_text:?}",
                    line_index + 1
                ))
            })?;
            if rank >= base_token_count {
                return Err(TokenizerError::Invalid(format!(
                    "tiktoken.model rank {rank} is outside 0..{base_token_count}"
                )));
            }
            let bytes = decode_base64(encoded).map_err(|reason| {
                TokenizerError::Invalid(format!(
                    "tiktoken.model line {} has invalid base64: {reason}",
                    line_index + 1
                ))
            })?;
            if bytes.is_empty() || bytes.len() > MAX_TOKEN_BYTES {
                return Err(TokenizerError::Invalid(format!(
                    "tiktoken.model rank {rank} has invalid byte length {}",
                    bytes.len()
                )));
            }
            if base_tokens[rank].replace(bytes.clone()).is_some() {
                return Err(TokenizerError::Invalid(format!(
                    "tiktoken.model assigns rank {rank} more than once"
                )));
            }
            if ranks.insert(bytes, rank as u32).is_some() {
                return Err(TokenizerError::Invalid(
                    "tiktoken.model assigns one byte string more than once".to_owned(),
                ));
            }
            parsed += 1;
        }
        if require_dense && parsed != base_token_count {
            return Err(TokenizerError::Invalid(format!(
                "tiktoken.model has {parsed} ranks; expected exactly {base_token_count}"
            )));
        }
        if require_dense {
            for byte in 0u8..=u8::MAX {
                if !ranks.contains_key([byte].as_slice()) {
                    return Err(TokenizerError::Invalid(format!(
                        "tiktoken.model has no single-byte token for 0x{byte:02x}"
                    )));
                }
            }
        }

        let parsed_config: TokenizerConfig = serde_json::from_str(config)?;
        let mut added_tokens = (0..ADDED_TOKEN_COUNT)
            .map(|offset| format!("<|reserved_token_{}|>", base_token_count + offset))
            .collect::<Vec<_>>();
        for (id_text, entry) in parsed_config.added_tokens_decoder {
            let id: usize = id_text.parse().map_err(|_| {
                TokenizerError::Invalid(format!(
                    "tokenizer_config.json has invalid added token ID {id_text:?}"
                ))
            })?;
            let Some(index) = id.checked_sub(base_token_count) else {
                return Err(TokenizerError::Invalid(format!(
                    "added token ID {id} precedes the base vocabulary"
                )));
            };
            let slot = added_tokens.get_mut(index).ok_or_else(|| {
                TokenizerError::Invalid(format!(
                    "added token ID {id} is outside the 256-token reserved range"
                ))
            })?;
            if entry.content.is_empty() {
                return Err(TokenizerError::Invalid(format!(
                    "added token ID {id} has empty content"
                )));
            }
            *slot = entry.content;
        }

        let mut special_to_id = HashMap::with_capacity(ADDED_TOKEN_COUNT);
        for (index, token) in added_tokens.iter().enumerate() {
            let id = (base_token_count + index) as u32;
            if special_to_id.insert(token.clone(), id).is_some() {
                return Err(TokenizerError::Invalid(format!(
                    "added token content {token:?} is assigned more than once"
                )));
            }
        }
        if base_token_count == BASE_TOKEN_COUNT {
            for (token, id) in [
                ("[BOS]", BOS_TOKEN_ID),
                ("[EOS]", TOKENIZER_EOS_TOKEN_ID),
                (prompt::END_OF_MESSAGE_TOKEN, END_OF_MESSAGE_TOKEN_ID),
                (prompt::OPEN_TOKEN, OPEN_TOKEN_ID),
                (prompt::CLOSE_TOKEN, CLOSE_TOKEN_ID),
                (prompt::SEP_TOKEN, SEP_TOKEN_ID),
                ("[PAD]", PAD_TOKEN_ID),
            ] {
                if special_to_id.get(token) != Some(&id) {
                    return Err(TokenizerError::Invalid(format!(
                        "Kimi-K3 control token {token:?} must have ID {id}"
                    )));
                }
            }
        }
        let mut specials_longest_first = special_to_id.keys().cloned().collect::<Vec<_>>();
        specials_longest_first
            .sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));

        Ok(Self {
            ranks,
            base_tokens,
            added_tokens,
            special_to_id,
            specials_longest_first,
        })
    }

    fn encode_ordinary_into(
        &self,
        text: &str,
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        for chunk in kimi_encoding_chunks(text) {
            let chunk = &text[chunk];
            for range in kimi_pretoken_ranges(chunk) {
                self.encode_piece(&chunk.as_bytes()[range], output)?;
            }
        }
        Ok(())
    }

    fn encode_piece(&self, bytes: &[u8], output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if bytes.is_empty() {
            return Ok(());
        }
        if let Some(&rank) = self.ranks.get(bytes) {
            output.push(rank);
            return Ok(());
        }

        #[derive(Debug, Clone)]
        struct Node {
            start: usize,
            end: usize,
            previous: Option<usize>,
            next: Option<usize>,
            live: bool,
            version: u64,
        }

        type MergeCandidate = Reverse<(u32, usize, usize, u64, u64)>;

        fn push_candidate(
            bytes: &[u8],
            ranks: &HashMap<Vec<u8>, u32>,
            nodes: &[Node],
            left: usize,
            heap: &mut BinaryHeap<MergeCandidate>,
        ) {
            let Some(right) = nodes[left].next else {
                return;
            };
            if let Some(&rank) = ranks.get(&bytes[nodes[left].start..nodes[right].end]) {
                heap.push(Reverse((
                    rank,
                    left,
                    right,
                    nodes[left].version,
                    nodes[right].version,
                )));
            }
        }

        let mut nodes = (0..bytes.len())
            .map(|index| Node {
                start: index,
                end: index + 1,
                previous: index.checked_sub(1),
                next: (index + 1 < bytes.len()).then_some(index + 1),
                live: true,
                version: 0,
            })
            .collect::<Vec<_>>();
        let mut heap = BinaryHeap::new();
        for left in 0..nodes.len().saturating_sub(1) {
            push_candidate(bytes, &self.ranks, &nodes, left, &mut heap);
        }

        while let Some(Reverse((_rank, left, right, left_version, right_version))) = heap.pop() {
            if !nodes[left].live
                || !nodes[right].live
                || nodes[left].next != Some(right)
                || nodes[left].version != left_version
                || nodes[right].version != right_version
            {
                continue;
            }
            let previous = nodes[left].previous;
            let next = nodes[right].next;
            nodes[left].end = nodes[right].end;
            nodes[left].next = next;
            nodes[left].version += 1;
            nodes[right].live = false;
            nodes[right].version += 1;
            if let Some(next) = next {
                nodes[next].previous = Some(left);
            }
            if let Some(previous) = previous {
                push_candidate(bytes, &self.ranks, &nodes, previous, &mut heap);
            }
            push_candidate(bytes, &self.ranks, &nodes, left, &mut heap);
        }

        let mut node = Some(0usize);
        while let Some(index) = node {
            debug_assert!(nodes[index].live);
            let token = &bytes[nodes[index].start..nodes[index].end];
            let rank = self.ranks.get(token).copied().ok_or_else(|| {
                TokenizerError::Invalid(format!(
                    "tiktoken.model cannot encode byte sequence {token:?}"
                ))
            })?;
            output.push(rank);
            node = nodes[index].next;
        }
        Ok(())
    }

    fn next_special<'a>(&self, text: &'a str, start: usize) -> Option<(usize, &'a str, u32)> {
        let mut best: Option<(usize, &str, u32)> = None;
        for token in &self.specials_longest_first {
            let Some(relative) = text[start..].find(token) else {
                continue;
            };
            let position = start + relative;
            let replace = best.is_none_or(|(best_position, best_token, _)| {
                position < best_position
                    || (position == best_position && token.len() > best_token.len())
            });
            if replace {
                let matched = &text[position..position + token.len()];
                best = Some((position, matched, self.special_to_id[token]));
            }
        }
        best
    }
}

#[derive(Debug)]
pub struct KimiK3StreamingDecoder<'a> {
    tokenizer: &'a KimiK3Tokenizer,
    pending_bytes: Vec<u8>,
    skip_special_tokens: bool,
}

impl KimiK3StreamingDecoder<'_> {
    pub fn push(&mut self, id: u32) -> Result<String, TokenizerError> {
        if (id as usize) < self.tokenizer.base_tokens.len() {
            let bytes = self.tokenizer.base_tokens[id as usize]
                .as_deref()
                .ok_or(TokenizerError::UnknownTokenId(id))?;
            self.pending_bytes.extend_from_slice(bytes);
            return Ok(drain_utf8_prefix(&mut self.pending_bytes));
        }
        let token = (id as usize)
            .checked_sub(self.tokenizer.base_tokens.len())
            .and_then(|index| self.tokenizer.added_tokens.get(index))
            .ok_or(TokenizerError::UnknownTokenId(id))?;
        if self.skip_special_tokens {
            Ok(String::new())
        } else {
            let mut output = flush_lossy(&mut self.pending_bytes);
            output.push_str(token);
            Ok(output)
        }
    }

    pub fn finish(&mut self) -> String {
        flush_lossy(&mut self.pending_bytes)
    }

    pub fn pending_byte_count(&self) -> usize {
        self.pending_bytes.len()
    }
}

fn check_size(path: &Path, limit: u64) -> Result<(), TokenizerError> {
    let size = fs::metadata(path)?.len();
    if size > limit {
        return Err(TokenizerError::Invalid(format!(
            "{} is {size} bytes; limit is {limit}",
            path.display()
        )));
    }
    Ok(())
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    if input.len() % 4 != 0 {
        return Err("length is not a multiple of four".to_owned());
    }
    let padding_count = input.bytes().rev().take_while(|&byte| byte == b'=').count();
    if padding_count > 2 {
        return Err("more than two padding characters".to_owned());
    }
    if input.as_bytes()[..input.len() - padding_count].contains(&b'=') {
        return Err("padding occurs before the end".to_owned());
    }
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    let mut padding = false;
    for byte in input.bytes() {
        if byte == b'=' {
            padding = true;
            continue;
        }
        if padding {
            return Err("non-padding data follows '='".to_owned());
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("invalid character 0x{byte:02x}")),
        };
        accumulator = (accumulator << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((accumulator >> bits) & 0xff) as u8);
        }
    }
    if bits > 0 && accumulator & ((1 << bits) - 1) != 0 {
        return Err("non-zero trailing bits".to_owned());
    }
    Ok(output)
}

fn kimi_encoding_chunks(text: &str) -> Vec<std::ops::Range<usize>> {
    const MAX_ENCODE_CHARACTERS: usize = 400_000;
    const MAX_CONSECUTIVE_CLASS_CHARACTERS: usize = 25_000;

    let characters = text.char_indices().collect::<Vec<_>>();
    if characters.is_empty() {
        return Vec::new();
    }
    let byte_boundary = |index: usize| {
        characters
            .get(index)
            .map_or(text.len(), |&(offset, _)| offset)
    };
    let mut output = Vec::new();
    let mut outer_start = 0usize;
    while outer_start < characters.len() {
        let outer_end = (outer_start + MAX_ENCODE_CHARACTERS).min(characters.len());
        let mut slice_start = outer_start;
        let mut current_is_space = characters[outer_start].1.is_whitespace();
        let mut current_length = 0usize;
        for (index, &(_, character)) in characters
            .iter()
            .enumerate()
            .take(outer_end)
            .skip(outer_start)
        {
            let is_space = character.is_whitespace();
            if current_is_space != is_space {
                current_length = 1;
                current_is_space = is_space;
            } else {
                current_length += 1;
                if current_length > MAX_CONSECUTIVE_CLASS_CHARACTERS {
                    output.push(byte_boundary(slice_start)..byte_boundary(index));
                    slice_start = index;
                    current_length = 1;
                }
            }
        }
        output.push(byte_boundary(slice_start)..byte_boundary(outer_end));
        outer_start = outer_end;
    }
    output
}

fn kimi_pretoken_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
    let characters = text.char_indices().collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut index = 0usize;
    while index < characters.len() {
        let start = index;
        let character = characters[index].1;

        if is_han(character) {
            index += 1;
            while index < characters.len() && is_han(characters[index].1) {
                index += 1;
            }
        } else if let Some(end) = letter_piece_end(&characters, index) {
            index = end;
        } else if is_number(character) {
            index += 1;
            while index < characters.len() && index - start < 3 && is_number(characters[index].1) {
                index += 1;
            }
        } else if let Some(end) = punctuation_piece_end(&characters, index) {
            index = end;
        } else if character.is_whitespace() {
            let mut run_end = index;
            let mut last_newline = None;
            while run_end < characters.len() && characters[run_end].1.is_whitespace() {
                if matches!(characters[run_end].1, '\r' | '\n') {
                    last_newline = Some(run_end + 1);
                }
                run_end += 1;
            }
            if let Some(end) = last_newline {
                index = end;
            } else if run_end < characters.len() && run_end - index > 1 {
                index = run_end - 1;
            } else {
                index = run_end;
            }
        } else {
            index += 1;
        }

        let byte_start = characters[start].0;
        let byte_end = characters
            .get(index)
            .map_or(text.len(), |&(offset, _)| offset);
        output.push(byte_start..byte_end);
    }
    output
}

fn letter_piece_end(characters: &[(usize, char)], start: usize) -> Option<usize> {
    let len = characters.len();
    for with_prefix in [true, false] {
        let mut body = start;
        if with_prefix {
            if start + 1 >= len || !can_prefix_letter_piece(characters[start].1) {
                continue;
            }
            body += 1;
        }
        let mut s1_end = body;
        while s1_end < len && is_kimi_s1(characters[s1_end].1) {
            s1_end += 1;
        }
        for s2_start in (body..=s1_end).rev() {
            if s2_start < len && is_kimi_s2(characters[s2_start].1) {
                let mut end = s2_start + 1;
                while end < len && is_kimi_s2(characters[end].1) {
                    end += 1;
                }
                return Some(contraction_end(characters, end));
            }
        }
    }

    for with_prefix in [true, false] {
        let mut body = start;
        if with_prefix {
            if start + 1 >= len || !can_prefix_letter_piece(characters[start].1) {
                continue;
            }
            body += 1;
        }
        let mut end = body;
        while end < len && is_kimi_s1(characters[end].1) {
            end += 1;
        }
        if end == body {
            continue;
        }
        while end < len && is_kimi_s2(characters[end].1) {
            end += 1;
        }
        return Some(contraction_end(characters, end));
    }
    None
}

fn punctuation_piece_end(characters: &[(usize, char)], start: usize) -> Option<usize> {
    let mut index = start;
    if characters[index].1 == ' '
        && characters
            .get(index + 1)
            .is_some_and(|&(_, character)| is_non_word(character))
    {
        index += 1;
    }
    if !is_non_word(characters.get(index)?.1) {
        return None;
    }
    while index < characters.len() && is_non_word(characters[index].1) {
        index += 1;
    }
    while index < characters.len() && matches!(characters[index].1, '\r' | '\n') {
        index += 1;
    }
    Some(index)
}

fn contraction_end(characters: &[(usize, char)], start: usize) -> usize {
    if characters.get(start).map(|&(_, c)| c) != Some('\'') {
        return start;
    }
    for suffix in ["re", "ve", "ll", "s", "t", "m", "d"] {
        let mut end = start + 1;
        let mut matches = true;
        for expected in suffix.chars() {
            let Some(&(_, actual)) = characters.get(end) else {
                matches = false;
                break;
            };
            if !fold_matches_ascii(actual, expected) {
                matches = false;
                break;
            }
            end += 1;
        }
        if matches {
            return end;
        }
    }
    start
}

fn can_prefix_letter_piece(character: char) -> bool {
    !matches!(character, '\r' | '\n') && !is_letter(character) && !is_number(character)
}

fn is_kimi_s1(character: char) -> bool {
    !is_han(character)
        && matches!(
            get_general_category(character),
            GeneralCategory::UppercaseLetter
                | GeneralCategory::TitlecaseLetter
                | GeneralCategory::ModifierLetter
                | GeneralCategory::OtherLetter
                | GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
}

fn is_kimi_s2(character: char) -> bool {
    !is_han(character)
        && matches!(
            get_general_category(character),
            GeneralCategory::LowercaseLetter
                | GeneralCategory::ModifierLetter
                | GeneralCategory::OtherLetter
                | GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
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

fn is_non_word(character: char) -> bool {
    !character.is_whitespace() && !is_letter(character) && !is_number(character)
}

fn fold_matches_ascii(actual: char, expected: char) -> bool {
    actual.eq_ignore_ascii_case(&expected) || (expected == 's' && actual == '\u{017f}')
}

fn is_han(character: char) -> bool {
    matches!(
        character,
        '\u{2e80}'..='\u{2e99}'
            | '\u{2e9b}'..='\u{2ef3}'
            | '\u{2f00}'..='\u{2fd5}'
            | '\u{3005}'
            | '\u{3007}'
            | '\u{3021}'..='\u{3029}'
            | '\u{3038}'..='\u{303b}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{4e00}'..='\u{9fff}'
            | '\u{f900}'..='\u{fa6d}'
            | '\u{fa70}'..='\u{fad9}'
            | '\u{16fe2}'..='\u{16fe3}'
            | '\u{16ff0}'..='\u{16ff1}'
            | '\u{20000}'..='\u{2a6df}'
            | '\u{2a700}'..='\u{2b739}'
            | '\u{2b740}'..='\u{2b81d}'
            | '\u{2b820}'..='\u{2cea1}'
            | '\u{2ceb0}'..='\u{2ebe0}'
            | '\u{2ebf0}'..='\u{2ee5d}'
            | '\u{2f800}'..='\u{2fa1d}'
            | '\u{30000}'..='\u{3134a}'
            | '\u{31350}'..='\u{323af}'
    )
}

fn drain_utf8_prefix(bytes: &mut Vec<u8>) -> String {
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
                            .expect("reported UTF-8 prefix is valid"),
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

fn flush_lossy(bytes: &mut Vec<u8>) -> String {
    let output = String::from_utf8_lossy(bytes).into_owned();
    bytes.clear();
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut output = String::new();
        for chunk in bytes.chunks(3) {
            let value = (u32::from(chunk[0]) << 16)
                | (chunk.get(1).copied().map_or(0, u32::from) << 8)
                | chunk.get(2).copied().map_or(0, u32::from);
            output.push(ALPHABET[((value >> 18) & 63) as usize] as char);
            output.push(ALPHABET[((value >> 12) & 63) as usize] as char);
            output.push(if chunk.len() > 1 {
                ALPHABET[((value >> 6) & 63) as usize] as char
            } else {
                '='
            });
            output.push(if chunk.len() > 2 {
                ALPHABET[(value & 63) as usize] as char
            } else {
                '='
            });
        }
        output
    }

    fn tiny_tokenizer() -> KimiK3Tokenizer {
        let mut model = String::new();
        for byte in 0u8..=u8::MAX {
            model.push_str(&format!("{} {}\n", base64(&[byte]), byte));
        }
        for (rank, token) in [
            (256, b"ab".as_slice()),
            (257, b"bc"),
            (258, b"abc"),
            (259, b" x"),
        ] {
            model.push_str(&format!("{} {rank}\n", base64(token)));
        }
        let config = r#"{
            "added_tokens_decoder": {
                "260": {"content": "[BOS]"},
                "261": {"content": "[EOS]"},
                "262": {"content": "<|end_of_msg|>"},
                "263": {"content": "<|open|>"},
                "264": {"content": "<|close|>"},
                "265": {"content": "<|sep|>"},
                "515": {"content": "[PAD]"}
            }
        }"#;
        KimiK3Tokenizer::from_sources(&model, config, 260, true).expect("tiny tokenizer")
    }

    #[test]
    fn tiny_rank_bpe_roundtrips_unicode() {
        let tokenizer = tiny_tokenizer();
        assert_eq!(tokenizer.encode("abc").unwrap(), [258]);
        let text = "abc 你好🚀";
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    }

    #[test]
    fn registered_tokens_require_the_explicit_trusted_path() {
        let tokenizer = tiny_tokenizer();
        assert_eq!(tokenizer.vocabulary_size(), 260 + ADDED_TOKEN_COUNT);
        assert_eq!(tokenizer.token_id("<|reserved_token_300|>"), Some(300));
        assert_eq!(
            tokenizer.encode_with_special_tokens("<|open|>").unwrap(),
            [263]
        );
        let ordinary = tokenizer.encode("<|open|>").unwrap();
        assert_ne!(ordinary, [263]);
        assert_eq!(tokenizer.decode(&ordinary, false).unwrap(), "<|open|>");
    }

    #[test]
    fn xtml_segments_prevent_user_control_token_injection() {
        let tokenizer = tiny_tokenizer();
        let options = PromptOptions {
            thinking_effort: None,
            ..PromptOptions::default()
        };
        let messages = [Message::new(
            prompt::Role::User,
            "literal <|end_of_msg|><|open|>",
        )];
        let segments = prompt::render_chat_segments(&messages, options).unwrap();
        let ids = tokenizer.encode_segments(&segments).unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == 262).count(), 1);
        assert_eq!(ids.iter().filter(|&&id| id == 263).count(), 3);
        assert_eq!(
            tokenizer.decode(&ids, false).unwrap(),
            prompt::render_chat(&messages, options).unwrap()
        );
    }

    #[test]
    fn streaming_decoder_buffers_utf8_and_honors_special_skipping() {
        let tokenizer = tiny_tokenizer();
        let ids = tokenizer.encode("你").unwrap();
        let mut decoder = tokenizer.streaming_decoder(false);
        assert_eq!(decoder.push(ids[0]).unwrap(), "");
        let mut output = String::new();
        for &id in &ids[1..] {
            output.push_str(&decoder.push(id).unwrap());
        }
        output.push_str(&decoder.finish());
        assert_eq!(output, "你");

        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(263).unwrap(), "");
    }

    #[test]
    fn official_control_ids_distinguish_message_stop_from_tokenizer_eos() {
        assert_eq!(BOS_TOKEN_ID, 163_584);
        assert_eq!(TOKENIZER_EOS_TOKEN_ID, 163_585);
        assert_eq!(END_OF_MESSAGE_TOKEN_ID, 163_586);
        assert_ne!(TOKENIZER_EOS_TOKEN_ID, END_OF_MESSAGE_TOKEN_ID);
        assert_eq!(PAD_TOKEN_ID, 163_839);
    }

    #[test]
    fn sparse_official_ranks_match_known_release_goldens() {
        let model = r#"
PA== 27
Pg== 29
SA== 39
ZA== 67
ZQ== 68
bA== 75
bg== 77
bw== 78
cA== 79
cg== 81
dw== 86
fA== 91
pQ== 98
uA== 116
vQ== 121
5A== 160
5Q== 161
5w== 163
IA== 220
jA== 234
lQ== 243
lg== 244
oA== 254
5Lg= 260
b3I= 268
ZW4= 271
IHc= 284
5L0= 315
ZWw= 323
5aU= 412
b3A= 482
bGQ= 621
5aW9 628
5L2g 633
bG8= 773
55U= 918
5LiW 1395
55WM 1620
IHdvcg== 2008
5LiW55WM 2243
IHdvcmxk 2695
b3Blbg== 4454
SGVs 5518
SGVsbG8= 19180
5L2g5aW9 33845
"#;
        let config = r#"{"added_tokens_decoder":{
            "163584":{"content":"[BOS]"},
            "163585":{"content":"[EOS]"},
            "163586":{"content":"<|end_of_msg|>"},
            "163587":{"content":"<|open|>"},
            "163588":{"content":"<|close|>"},
            "163589":{"content":"<|sep|>"},
            "163839":{"content":"[PAD]"}
        }}"#;
        let tokenizer =
            KimiK3Tokenizer::from_sources(model, config, BASE_TOKEN_COUNT, false).unwrap();
        assert_eq!(tokenizer.encode("Hello world").unwrap(), [19180, 2695]);
        assert_eq!(tokenizer.encode("你好世界").unwrap(), [33845, 2243]);
        assert_eq!(
            tokenizer.encode("<|open|>").unwrap(),
            [27, 91, 4454, 91, 29]
        );
        assert_eq!(
            tokenizer.encode_with_special_tokens("<|open|>").unwrap(),
            [OPEN_TOKEN_ID]
        );
    }

    #[test]
    fn malformed_sources_are_rejected() {
        let config = r#"{"added_tokens_decoder":{}}"#;
        assert!(matches!(
            KimiK3Tokenizer::from_sources("not-base64 0", config, 1, false),
            Err(TokenizerError::Invalid(_))
        ));
    }

    #[test]
    fn official_long_input_guards_split_by_codepoint_and_whitespace_class() {
        let run = "界".repeat(25_001);
        let ranges = kimi_encoding_chunks(&run);
        assert_eq!(ranges.len(), 2);
        assert_eq!(run[ranges[0].clone()].chars().count(), 25_000);
        assert_eq!(run[ranges[1].clone()].chars().count(), 1);

        let alternating = "a ".repeat(200_001);
        let ranges = kimi_encoding_chunks(&alternating);
        assert_eq!(ranges.len(), 2);
        assert_eq!(alternating[ranges[0].clone()].chars().count(), 400_000);
        assert_eq!(alternating[ranges[1].clone()].chars().count(), 2);
    }
}
