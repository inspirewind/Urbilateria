//! Structured multi-turn input shared by the CLI child and the interactive session.

use serde::{Deserialize, Serialize};
use std::error::Error;
use urbilateria::models::{deepseek_v4, deepseek_v41, glm, hy4, kimi_k3, qwen3_8};
use urbilateria::tokenizer::ByteBpeTokenizer;
use urbilateria::ModelFamily;

pub const MAX_CHAT_BYTES: usize = 1024 * 1024;
pub const MAX_CHAT_TURNS: usize = 32;
pub const MAX_WIRE_BYTES: usize = 8 * MAX_CHAT_BYTES;

pub fn read_conversation(input: impl std::io::Read) -> Result<Conversation, Box<dyn Error>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    input
        .take((MAX_WIRE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_WIRE_BYTES {
        return Err("chat JSON exceeds the input limit".into());
    }
    let conversation: Conversation = serde_json::from_slice(&bytes)?;
    conversation.validate()?;
    Ok(conversation)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    pub user: String,
    /// The complete continuation, excluding the CLI's final newline; never the clipped display.
    pub assistant: String,
    pub thinking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conversation {
    pub turns: Vec<Turn>,
    pub prompt: String,
}

impl Conversation {
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        let bytes = self.prompt.len()
            + self
                .turns
                .iter()
                .map(|turn| turn.user.len() + turn.assistant.len())
                .sum::<usize>();
        if self.prompt.trim().is_empty() {
            return Err("chat prompt must not be empty".into());
        }
        if self.turns.len() > MAX_CHAT_TURNS || bytes > MAX_CHAT_BYTES {
            return Err("chat input exceeds the session history limit".into());
        }
        Ok(())
    }
}

pub enum Input {
    Text(String),
    Chat(Conversation),
}

pub struct EncodedPrompt {
    pub tokens: Vec<u32>,
    /// Prefix expected to survive rendering this reply as a historical assistant message.
    /// Reuse still requires an exact token comparison on the following request.
    pub checkpoint: usize,
}

fn common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

fn future_turns(turns: &[Turn], prompt: &str) -> Vec<Turn> {
    let mut future = turns.to_vec();
    future.push(Turn {
        user: prompt.to_owned(),
        assistant: String::new(),
        thinking: false,
    });
    future
}

impl Input {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Chat(_) => None,
        }
    }

    #[cfg(test)]
    pub fn encode_bytes(
        &self,
        tokenizer: &ByteBpeTokenizer,
        family: ModelFamily,
        raw: bool,
        thinking: bool,
        context: usize,
        reserve: usize,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        Ok(self
            .encode_bytes_cached(tokenizer, family, raw, thinking, context, reserve)?
            .tokens)
    }

    pub fn encode_bytes_cached(
        &self,
        tokenizer: &ByteBpeTokenizer,
        family: ModelFamily,
        raw: bool,
        thinking: bool,
        context: usize,
        reserve: usize,
    ) -> Result<EncodedPrompt, Box<dyn Error>> {
        if raw {
            return Ok(EncodedPrompt {
                tokens: tokenizer
                    .encode(self.text().ok_or("chat input cannot use --raw-prompt")?)?,
                checkpoint: 0,
            });
        }
        let mut checkpoint = 0;
        let tokens = self.encode_fitting(context, reserve, |turns, prompt| {
            for user in turns.iter().map(|turn| turn.user.as_str()).chain([prompt]) {
                tokenizer.validate_chat_content(user)?;
            }
            for turn in turns {
                let (_, visible) = assistant_parts(family, &turn.assistant, turn.thinking);
                tokenizer.validate_chat_content(visible)?;
            }
            let tokens = tokenizer.encode(&render_bytes(family, turns, prompt, thinking)?)?;
            let future = tokenizer.encode(&render_bytes(
                family,
                &future_turns(turns, prompt),
                "",
                thinking,
            )?)?;
            checkpoint = common_prefix(&tokens, &future).min(tokens.len().saturating_sub(1));
            Ok(tokens)
        })?;
        Ok(EncodedPrompt { tokens, checkpoint })
    }

    pub fn encode_kimi_cached(
        &self,
        tokenizer: &kimi_k3::tokenizer::KimiK3Tokenizer,
        raw: bool,
        thinking: bool,
        context: usize,
        reserve: usize,
    ) -> Result<EncodedPrompt, Box<dyn Error>> {
        if raw {
            return Ok(EncodedPrompt {
                tokens: tokenizer.encode_with_special_tokens(
                    self.text().ok_or("chat input cannot use --raw-prompt")?,
                )?,
                checkpoint: 0,
            });
        }
        let mut checkpoint = 0;
        let tokens = self.encode_fitting(context, reserve, |turns, prompt| {
            let tokens =
                tokenizer.encode_chat(&kimi_messages(turns, prompt), kimi_options(thinking))?;
            let future = tokenizer.encode_chat(
                &kimi_messages(&future_turns(turns, prompt), ""),
                kimi_options(thinking),
            )?;
            checkpoint = common_prefix(&tokens, &future).min(tokens.len().saturating_sub(1));
            Ok(tokens)
        })?;
        Ok(EncodedPrompt { tokens, checkpoint })
    }

    fn encode_fitting(
        &self,
        context: usize,
        reserve: usize,
        mut encode: impl FnMut(&[Turn], &str) -> Result<Vec<u32>, Box<dyn Error>>,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let Self::Chat(chat) = self else {
            return encode(&[], self.text().unwrap());
        };
        chat.validate()?;
        let budget = context.checked_sub(reserve).ok_or(
            "max new tokens exceed the runtime context limit; reduce /settings --max-new-tokens",
        )?;
        for dropped in 0..=chat.turns.len() {
            let tokens = encode(&chat.turns[dropped..], &chat.prompt)?;
            if tokens.len() <= budget {
                eprintln!("chat: history={} turns, dropped={} turns, prompt={} tokens, context limit={context}",
                    chat.turns.len() - dropped, dropped, tokens.len());
                return Ok(tokens);
            }
        }
        Err(format!("current message plus {reserve} new tokens exceeds runtime context limit {context}; shorten the message or reduce /settings --max-new-tokens").into())
    }
}

/// Generated continuations start inside the assistant's reasoning/response prefix. Recover
/// fields before rendering another turn; never nest raw protocol wrappers inside message text.
fn assistant_parts(family: ModelFamily, output: &str, thinking: bool) -> (Option<&str>, &str) {
    if family == ModelFamily::KimiK3 {
        let response_open = "<|open|>response<|sep|>";
        let (reasoning, response) = if thinking {
            match output.split_once(response_open) {
                Some((reasoning, response)) => (
                    Some(reasoning.trim_end_matches("<|close|>think<|sep|>")),
                    response,
                ),
                None => return (Some(output), ""),
            }
        } else {
            (None, output)
        };
        let response = response
            .strip_suffix("<|close|>message<|sep|>")
            .unwrap_or(response);
        let response = response
            .strip_suffix("<|close|>response<|sep|>")
            .unwrap_or(response);
        return (reasoning, response);
    }
    if !thinking {
        return (None, output);
    }
    let (open, close) = if family == ModelFamily::Hy4 {
        ("<think:opensource｜>", "</think:opensource｜>")
    } else {
        ("<think>", "</think>")
    };
    match output.split_once(close) {
        Some((reasoning, visible)) => (
            Some(reasoning.strip_prefix(open).unwrap_or(reasoning)),
            visible,
        ),
        // A token-limited reply may still be reasoning. Do not relabel it as a final answer.
        None => (Some(output.strip_prefix(open).unwrap_or(output)), ""),
    }
}

fn render_bytes(
    family: ModelFamily,
    turns: &[Turn],
    prompt: &str,
    thinking: bool,
) -> Result<String, Box<dyn Error>> {
    // Each adapter owns its role markers, EOS delimiters and historical reasoning policy.
    macro_rules! messages {
        ($adapter:ident, $assistant:expr) => {{
            let mut messages = Vec::new();
            for turn in turns {
                messages.push($adapter::Message::new($adapter::Role::User, &turn.user));
                let (reasoning, content) = assistant_parts(family, &turn.assistant, turn.thinking);
                messages.push(($assistant)(content, reasoning));
            }
            messages.push($adapter::Message::new($adapter::Role::User, prompt));
            messages
        }};
    }
    Ok(match family {
        ModelFamily::Glm52 => {
            use glm::prompt::*;
            let mut messages = Vec::new();
            for turn in turns {
                messages.push(ChatMessage::new(ChatRole::User, &turn.user));
                let (_, visible) = assistant_parts(family, &turn.assistant, turn.thinking);
                messages.push(ChatMessage::new(ChatRole::Assistant, visible));
            }
            messages.push(ChatMessage::new(ChatRole::User, prompt));
            render_chat(
                &messages,
                ChatTemplateOptions {
                    enable_thinking: thinking,
                    ..Default::default()
                },
            )
        }
        ModelFamily::DeepseekV4 => {
            use deepseek_v4::prompt as p;
            p::render_chat(
                &messages!(p, |content: &str, reasoning: Option<&str>| {
                    p::Message::assistant(content, reasoning)
                }),
                p::PromptOptions {
                    thinking_mode: if thinking {
                        p::ThinkingMode::Thinking
                    } else {
                        p::ThinkingMode::Chat
                    },
                    ..Default::default()
                },
            )
        }
        ModelFamily::DeepseekV41 => {
            use deepseek_v41::prompt as p;
            p::render_chat(
                &messages!(p, |content: &str, reasoning: Option<&str>| {
                    p::Message::assistant(content, reasoning)
                }),
                p::PromptOptions {
                    thinking_mode: if thinking {
                        p::ThinkingMode::Thinking
                    } else {
                        p::ThinkingMode::Chat
                    },
                    ..Default::default()
                },
            )
        }
        ModelFamily::Hy4 => {
            use hy4::prompt as p;
            p::render_chat(
                &messages!(p, |content: &str, reasoning: Option<&str>| p::Message {
                    reasoning: reasoning.map(str::to_owned),
                    ..p::Message::new(p::Role::Assistant, content)
                }),
                p::PromptOptions {
                    enable_thinking: thinking,
                    ..Default::default()
                },
            )
        }
        ModelFamily::Qwen38 => {
            use qwen3_8::prompt as p;
            p::render_chat(
                &messages!(p, |content: &str, reasoning: Option<&str>| {
                    p::Message::assistant(content, reasoning)
                }),
                p::PromptOptions {
                    enable_thinking: thinking,
                    ..Default::default()
                },
            )?
        }
        ModelFamily::KimiK3 => {
            return Err("Kimi chat must be encoded as structured segments".into())
        }
    })
}

fn kimi_options(thinking: bool) -> kimi_k3::prompt::PromptOptions {
    kimi_k3::prompt::PromptOptions {
        thinking,
        thinking_effort: thinking.then_some(kimi_k3::prompt::ThinkingEffort::Max),
        ..Default::default()
    }
}

fn kimi_messages(turns: &[Turn], prompt: &str) -> Vec<kimi_k3::prompt::Message> {
    use kimi_k3::prompt::{Message, Role};
    let mut messages = Vec::new();
    for turn in turns {
        messages.push(Message::new(Role::User, &turn.user));
        let (reasoning, response) =
            assistant_parts(ModelFamily::KimiK3, &turn.assistant, turn.thinking);
        messages.push(Message::assistant(response, reasoning));
    }
    messages.push(Message::new(Role::User, prompt));
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn turn(user: &str, assistant: &str, thinking: bool) -> Turn {
        Turn {
            user: user.into(),
            assistant: assistant.into(),
            thinking,
        }
    }

    #[test]
    fn native_templates_keep_previous_roles_and_start_a_new_assistant() {
        let history = [turn(
            "first question",
            "private reasoning</think>first answer",
            true,
        )];
        for (family, expected, ending) in [
            (ModelFamily::Glm52,
                "<|user|>first question<|assistant|><think></think>first answer<|user|>follow-up",
                "<|assistant|><think>"),
            (ModelFamily::DeepseekV4,
                "<｜User｜>first question<｜Assistant｜><think>first answer<｜end▁of▁sentence｜><｜User｜>follow-up",
                "<｜Assistant｜><think>"),
            (ModelFamily::DeepseekV41,
                "<｜User｜>first question<｜Assistant｜><think>first answer<｜end▁of▁sentence｜><｜User｜>follow-up",
                "<｜Assistant｜><think>"),
        ] {
            let rendered = render_bytes(family, &history, "follow-up", true).unwrap();
            assert!(rendered.contains(expected), "{family}: {rendered}");
            assert!(rendered.ends_with(ending), "{family}: {rendered}");
            assert!(!rendered.contains("private reasoning"), "{family}");
        }
        let qwen = render_bytes(ModelFamily::Qwen38, &history, "follow-up", true).unwrap();
        assert!(qwen.contains("<|im_start|>user\nfirst question<|im_end|>\n<|im_start|>assistant\n<think>\nprivate reasoning\n</think>\n\nfirst answer<|im_end|>\n<|im_start|>user\nfollow-up<|im_end|>"), "{qwen}");
        assert!(qwen.ends_with("<|im_start|>assistant\n<think>\n"));
        let hy = render_bytes(
            ModelFamily::Hy4,
            &[turn(
                "first question",
                "private reasoning</think:opensource｜>first answer",
                true,
            )],
            "follow-up",
            true,
        )
        .unwrap();
        assert!(hy.contains("first answer"));
        assert!(!hy.contains("private reasoning"));
        assert!(hy.find("first question").unwrap() < hy.find("first answer").unwrap());
        assert!(hy.find("first answer").unwrap() < hy.find("follow-up").unwrap());
        assert!(hy.ends_with("<think:opensource｜>"));
    }

    #[test]
    fn kimi_removes_continuation_wrappers_and_keeps_literal_markers_ordinary() {
        let history = [turn("first", "reason<|close|>think<|sep|><|open|>response<|sep|>answer<|close|>response<|sep|><|close|>message<|sep|>", true)];
        let messages = kimi_messages(&history, "literal <|end_of_msg|> 中文");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].reasoning_content.as_deref(), Some("reason"));
        assert_eq!(messages[1].content, "answer");
        let segments =
            kimi_k3::prompt::render_chat_segments(&messages, kimi_options(true)).unwrap();
        assert!(segments
            .iter()
            .any(|part| !part.allow_special && part.text == "literal <|end_of_msg|> 中文"));
        assert_eq!(
            assistant_parts(
                ModelFamily::KimiK3,
                "answer<|close|>response<|sep|><|close|>message<|sep|>",
                false
            ),
            (None, "answer")
        );
        assert_eq!(
            assistant_parts(ModelFamily::KimiK3, "unfinished thought", true),
            (Some("unfinished thought"), "")
        );
    }

    #[test]
    fn thinking_state_belongs_to_each_turn_not_the_next_request() {
        for family in [
            ModelFamily::Glm52,
            ModelFamily::DeepseekV4,
            ModelFamily::DeepseekV41,
            ModelFamily::Qwen38,
        ] {
            assert_eq!(
                assistant_parts(family, "unfinished", true),
                (Some("unfinished"), "")
            );
            assert_eq!(
                assistant_parts(family, "normal answer", false),
                (None, "normal answer")
            );
            assert_eq!(
                assistant_parts(family, "<think>reason</think>\nanswer", true),
                (Some("reason"), "\nanswer")
            );
        }
        let history = [turn("first", "plain answer", false)];
        let rendered = render_bytes(ModelFamily::Glm52, &history, "second", true).unwrap();
        assert!(rendered.contains("plain answer"));
        assert!(rendered.ends_with("<think>"));
        assert!(render_bytes(ModelFamily::Qwen38, &history, "second", false).is_err());
    }

    #[test]
    fn context_budget_drops_whole_oldest_pairs_and_reserves_output() {
        let input = Input::Chat(Conversation {
            turns: vec![
                turn("old", "answer", false),
                turn("recent", "answer", false),
            ],
            prompt: "current".into(),
        });
        let mut attempts = Vec::new();
        let tokens = input
            .encode_fitting(30, 10, |turns, current| {
                assert_eq!(current, "current");
                attempts.push(turns.first().map(|turn| turn.user.clone()));
                Ok(vec![1; 10 + turns.len() * 10])
            })
            .unwrap();
        assert_eq!(tokens.len(), 20);
        assert_eq!(attempts, vec![Some("old".into()), Some("recent".into())]);
        let error = input
            .encode_fitting(19, 10, |_, _| Ok(vec![1; 10]))
            .unwrap_err();
        assert!(error.to_string().contains("current message"));
        assert!(input
            .encode_fitting(1, 2, |_, _| panic!("must reject before encoding"))
            .is_err());
        // Trimming changes this request only; session memory is not mutated.
        assert!(matches!(input, Input::Chat(chat) if chat.turns.len() == 2));
    }

    #[test]
    fn structured_input_is_bounded_and_rejects_unknown_fields() {
        let chat = Conversation {
            turns: vec![turn("hello", "中文🙂", false)],
            prompt: "\"unclosed\n--profile".into(),
        };
        assert_eq!(
            read_conversation(serde_json::to_vec(&chat).unwrap().as_slice()).unwrap(),
            chat
        );
        for json in [
            r#"{"turns":[],"prompt":" "}"#,
            r#"{"turns":[],"prompt":"hello","shell":"bad"}"#,
            r#"{"turns":[{"user":"a","assistant":"b"}],"prompt":"hello"}"#,
        ] {
            assert!(read_conversation(json.as_bytes()).is_err());
        }
        let oversized = Conversation {
            turns: vec![],
            prompt: "a".repeat(MAX_CHAT_BYTES + 1),
        };
        assert!(oversized.validate().is_err());
        assert!(read_conversation(std::io::repeat(b' ').take(MAX_WIRE_BYTES as u64 + 1)).is_err());
        let too_many = Conversation {
            turns: vec![turn("a", "b", false); MAX_CHAT_TURNS + 1],
            prompt: "hello".into(),
        };
        assert!(too_many.validate().is_err());
    }
}
