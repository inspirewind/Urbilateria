//! Text tokenization and decoding without loading model weights.

use crate::models::deepseek_v4::prompt as deepseek_prompt;
use crate::models::deepseek_v41::prompt as deepseek_v41_prompt;
use crate::models::hy4::prompt as hy4_prompt;
use crate::models::kimi_k3::{prompt as kimi_k3_prompt, tokenizer::KimiK3Tokenizer};
use crate::models::qwen3_8::prompt as qwen38_prompt;
use crate::tokenizer::{render_chat, ByteBpeTokenizer, ChatMessage, ChatRole, ChatTemplateOptions};
use crate::{CommonModelConfig, ModelConfig};
use serde::Serialize;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TokenizeOptions {
    pub chat: bool,
    pub no_thinking: bool,
}

impl TokenizeOptions {
    pub fn validate(self) -> Result<(), String> {
        if self.no_thinking && !self.chat {
            return Err("--no-thinking requires --chat".into());
        }
        Ok(())
    }
}

pub struct Tokenization {
    pub model_path: PathBuf,
    pub model: CommonModelConfig,
    pub report: TokenizeReport,
}

#[derive(Debug, Serialize)]
pub struct TokenizeReport {
    pub prompt: String,
    pub token_ids: Vec<u32>,
    pub token_count: usize,
}

pub struct Decoding {
    pub model_path: PathBuf,
    pub model: CommonModelConfig,
    pub report: DecodeReport,
}

#[derive(Debug, Serialize)]
pub struct DecodeReport {
    pub token_ids: Vec<u32>,
    pub text: String,
}

pub fn tokenize_text(
    model_dir: &Path,
    text: String,
    options: TokenizeOptions,
) -> Result<Tokenization, Box<dyn Error + Send + Sync>> {
    options.validate()?;
    let TokenizeOptions { chat, no_thinking } = options;
    let config = ModelConfig::load(model_dir)?;
    let model = config.common();
    let (prompt, token_ids) = match config {
        ModelConfig::KimiK3(_) => {
            let tokenizer = KimiK3Tokenizer::load(model_dir)?;
            if chat {
                let messages = [kimi_k3_prompt::Message::new(
                    kimi_k3_prompt::Role::User,
                    text,
                )];
                let options = kimi_k3_prompt::PromptOptions {
                    thinking: !no_thinking,
                    thinking_effort: (!no_thinking).then_some(kimi_k3_prompt::ThinkingEffort::Max),
                    ..kimi_k3_prompt::PromptOptions::default()
                };
                let prompt = kimi_k3_prompt::render_chat(&messages, options)?;
                let token_ids = tokenizer.encode_chat(&messages, options)?;
                (prompt, token_ids)
            } else {
                let token_ids = tokenizer.encode(&text)?;
                (text, token_ids)
            }
        }
        config => {
            let tokenizer = ByteBpeTokenizer::load(model_dir)?;
            let prompt = if chat {
                tokenizer.validate_chat_content(&text)?;
                match config {
                    ModelConfig::Glm52(_) => render_chat(
                        &[ChatMessage::new(ChatRole::User, text)],
                        ChatTemplateOptions {
                            enable_thinking: !no_thinking,
                            ..ChatTemplateOptions::default()
                        },
                    ),
                    ModelConfig::DeepseekV4(_) => deepseek_prompt::render_chat(
                        &[deepseek_prompt::Message::new(
                            deepseek_prompt::Role::User,
                            text,
                        )],
                        deepseek_prompt::PromptOptions {
                            thinking_mode: if no_thinking {
                                deepseek_prompt::ThinkingMode::Chat
                            } else {
                                deepseek_prompt::ThinkingMode::Thinking
                            },
                            ..deepseek_prompt::PromptOptions::default()
                        },
                    ),
                    ModelConfig::DeepseekV41(_) => deepseek_v41_prompt::render_chat(
                        &[deepseek_v41_prompt::Message::new(
                            deepseek_v41_prompt::Role::User,
                            text,
                        )],
                        deepseek_v41_prompt::PromptOptions {
                            thinking_mode: if no_thinking {
                                deepseek_v41_prompt::ThinkingMode::Chat
                            } else {
                                deepseek_v41_prompt::ThinkingMode::Thinking
                            },
                            ..deepseek_v41_prompt::PromptOptions::default()
                        },
                    ),
                    ModelConfig::Hy4(_) => hy4_prompt::render_chat(
                        &[hy4_prompt::Message::new(hy4_prompt::Role::User, text)],
                        hy4_prompt::PromptOptions {
                            enable_thinking: !no_thinking,
                            ..hy4_prompt::PromptOptions::default()
                        },
                    ),
                    ModelConfig::KimiK3(_) => unreachable!("handled above"),
                    ModelConfig::Qwen38(_) => {
                        if no_thinking {
                            return Err("Qwen3.8 requires thinking; --no-thinking is unsupported by the official template".into());
                        }
                        qwen38_prompt::render_chat(
                            &[qwen38_prompt::Message::new(qwen38_prompt::Role::User, text)],
                            qwen38_prompt::PromptOptions::default(),
                        )?
                    }
                }
            } else {
                text
            };
            let token_ids = tokenizer.encode(&prompt)?;
            (prompt, token_ids)
        }
    };

    Ok(Tokenization {
        model_path: model_dir.to_owned(),
        model,
        report: TokenizeReport {
            token_count: token_ids.len(),
            prompt,
            token_ids,
        },
    })
}

pub fn decode_tokens(
    model_dir: &Path,
    token_ids: Vec<u32>,
    skip_special: bool,
) -> Result<Decoding, Box<dyn Error + Send + Sync>> {
    let config = ModelConfig::load(model_dir)?;
    let model = config.common();
    let text = match config {
        ModelConfig::KimiK3(_) => {
            KimiK3Tokenizer::load(model_dir)?.decode(&token_ids, skip_special)?
        }
        ModelConfig::Glm52(_)
        | ModelConfig::DeepseekV4(_)
        | ModelConfig::DeepseekV41(_)
        | ModelConfig::Hy4(_)
        | ModelConfig::Qwen38(_) => {
            ByteBpeTokenizer::load(model_dir)?.decode(&token_ids, skip_special)?
        }
    };

    Ok(Decoding {
        model_path: model_dir.to_owned(),
        model,
        report: DecodeReport { text, token_ids },
    })
}

pub fn parse_token_ids(text: &str) -> Result<Vec<u32>, String> {
    if text.trim().is_empty() {
        return Err("TOKEN_IDS must not be empty".into());
    }
    text.split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                Err("TOKEN_IDS contains an empty item".into())
            } else {
                part.parse::<u32>()
                    .map_err(|error| format!("invalid token ID {part:?}: {error}"))
            }
        })
        .collect()
}
