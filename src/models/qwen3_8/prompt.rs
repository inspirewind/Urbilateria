//! Injection-safe, text-only Qwen3.8 ChatML rendering.
//!
//! This is the text subset of the tokenizer's published Jinja template. Qwen3.8 does not allow
//! thinking to be disabled and recognizes exactly `xhigh` (default), `medium`, and `low`.

use std::fmt;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";

const XHIGH_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW_INSTRUCTION: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

/// Every published added-token marker in IDs 248044 through 248076.
///
/// The ChatML renderer owns these strings. Literal occurrences in message data are rejected so
/// concatenation cannot turn untrusted text into a role boundary, thinking tag, or multimodal
/// placeholder when a tokenizer is configured to recognize added tokens.
pub const RESERVED_MARKERS: &[&str] = &[
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
        }
    }

    pub fn assistant(
        content: impl Into<String>,
        reasoning_content: Option<impl Into<String>>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            reasoning_content: reasoning_content.map(Into::into),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    XHigh,
    Medium,
    Low,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::XHigh => "xhigh",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Self::XHigh => XHIGH_INSTRUCTION,
            Self::Medium => "",
            Self::Low => LOW_INSTRUCTION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
    pub reasoning_effort: ReasoningEffort,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_thinking: true,
            reasoning_effort: ReasoningEffort::XHigh,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptError {
    NoMessages,
    NoUserQuery,
    ThinkingCannotBeDisabled,
    SystemMessageNotFirst {
        index: usize,
    },
    ReasoningOnNonAssistant {
        index: usize,
    },
    ReservedMarker {
        index: usize,
        field: &'static str,
        marker: &'static str,
    },
}

impl fmt::Display for PromptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMessages => f.write_str("Qwen3.8 prompt has no messages"),
            Self::NoUserQuery => f.write_str("Qwen3.8 prompt has no user query"),
            Self::ThinkingCannotBeDisabled => {
                f.write_str("Qwen3.8 does not support disabling thinking")
            }
            Self::SystemMessageNotFirst { index } => {
                write!(f, "Qwen3.8 system message at index {index} is not first")
            }
            Self::ReasoningOnNonAssistant { index } => write!(
                f,
                "Qwen3.8 message at index {index} has reasoning_content but is not assistant"
            ),
            Self::ReservedMarker {
                index,
                field,
                marker,
            } => write!(
                f,
                "Qwen3.8 message {index} {field} contains reserved marker {marker:?}"
            ),
        }
    }
}

impl std::error::Error for PromptError {}

/// Renders the released tokenizer's text-only ChatML protocol.
///
/// Structured tools, tool responses, and multimodal content are intentionally absent from this
/// phase-one API. Message text is trimmed exactly as in the official Jinja template.
pub fn render_chat(messages: &[Message], options: PromptOptions) -> Result<String, PromptError> {
    if messages.is_empty() {
        return Err(PromptError::NoMessages);
    }
    if !options.enable_thinking {
        return Err(PromptError::ThinkingCannotBeDisabled);
    }
    validate_messages(messages)?;

    let last_user_index = messages
        .iter()
        .rposition(|message| message.role == Role::User)
        .ok_or(PromptError::NoUserQuery)?;
    let mut output = String::new();
    let reasoning_instruction = options.reasoning_effort.instruction();

    let has_system = messages[0].role == Role::System;
    if has_system {
        let system_content = messages[0].content.trim();
        if !system_content.is_empty() || !reasoning_instruction.is_empty() {
            output.push_str(IM_START);
            output.push_str("system\n");
            if !reasoning_instruction.is_empty() {
                output.push_str(reasoning_instruction);
                if !system_content.is_empty() {
                    output.push_str("\n\n");
                }
            }
            output.push_str(system_content);
            output.push_str(IM_END);
            output.push('\n');
        }
    } else if !reasoning_instruction.is_empty() {
        output.push_str(IM_START);
        output.push_str("system\n");
        output.push_str(reasoning_instruction);
        output.push_str(IM_END);
        output.push('\n');
    }

    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System {
            continue;
        }
        output.push_str(IM_START);
        output.push_str(message.role.as_str());
        output.push('\n');
        if message.role == Role::Assistant && (options.preserve_thinking || index > last_user_index)
        {
            output.push_str(THINK_START);
            output.push('\n');
            output.push_str(
                message
                    .reasoning_content
                    .as_deref()
                    .unwrap_or_default()
                    .trim(),
            );
            output.push('\n');
            output.push_str(THINK_END);
            output.push_str("\n\n");
        }
        output.push_str(message.content.trim());
        output.push_str(IM_END);
        output.push('\n');
    }

    if options.add_generation_prompt {
        output.push_str(IM_START);
        output.push_str("assistant\n");
        output.push_str(THINK_START);
        output.push('\n');
    }
    Ok(output)
}

fn validate_messages(messages: &[Message]) -> Result<(), PromptError> {
    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System && index != 0 {
            return Err(PromptError::SystemMessageNotFirst { index });
        }
        if message.role != Role::Assistant && message.reasoning_content.is_some() {
            return Err(PromptError::ReasoningOnNonAssistant { index });
        }
        reject_reserved(index, "content", &message.content)?;
        if let Some(reasoning) = message.reasoning_content.as_deref() {
            reject_reserved(index, "reasoning_content", reasoning)?;
        }
    }
    Ok(())
}

fn reject_reserved(index: usize, field: &'static str, text: &str) -> Result<(), PromptError> {
    if let Some(&marker) = RESERVED_MARKERS
        .iter()
        .find(|&&marker| text.contains(marker))
    {
        return Err(PromptError::ReservedMarker {
            index,
            field,
            marker,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_xhigh_prompt_matches_the_release_template() {
        let prompt = render_chat(
            &[Message::new(Role::User, "  hello  ")],
            PromptOptions::default(),
        )
        .unwrap();
        assert_eq!(
            prompt,
            format!(
                "<|im_start|>system\n{XHIGH_INSTRUCTION}<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
    }

    #[test]
    fn medium_emits_no_reasoning_system_instruction() {
        let prompt = render_chat(
            &[Message::new(Role::User, "hello")],
            PromptOptions {
                reasoning_effort: ReasoningEffort::Medium,
                ..PromptOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn low_instruction_is_prepended_to_the_first_system_message() {
        let prompt = render_chat(
            &[
                Message::new(Role::System, "  Be terse. "),
                Message::new(Role::User, "question"),
            ],
            PromptOptions {
                reasoning_effort: ReasoningEffort::Low,
                add_generation_prompt: false,
                ..PromptOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            prompt,
            format!(
                "<|im_start|>system\n{LOW_INSTRUCTION}\n\nBe terse.<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n"
            )
        );
    }

    #[test]
    fn preserve_thinking_false_only_strips_assistants_before_last_query() {
        let prompt = render_chat(
            &[
                Message::new(Role::User, "first"),
                Message::assistant("answer one", Some("reason one")),
                Message::new(Role::User, "second"),
                Message::assistant("answer two", Some("reason two")),
            ],
            PromptOptions {
                add_generation_prompt: false,
                preserve_thinking: false,
                reasoning_effort: ReasoningEffort::Medium,
                ..PromptOptions::default()
            },
        )
        .unwrap();
        assert!(prompt.contains("assistant\nanswer one<|im_end|>"));
        assert!(!prompt.contains("reason one"));
        assert!(prompt.contains("assistant\n<think>\nreason two\n</think>\n\nanswer two"));
    }

    #[test]
    fn thinking_is_mandatory_and_a_user_query_is_required() {
        let error = render_chat(
            &[Message::new(Role::User, "hello")],
            PromptOptions {
                enable_thinking: false,
                ..PromptOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, PromptError::ThinkingCannotBeDisabled);

        assert_eq!(
            render_chat(
                &[Message::new(Role::System, "system")],
                PromptOptions::default()
            )
            .unwrap_err(),
            PromptError::NoUserQuery
        );
    }

    #[test]
    fn rejects_every_reserved_token_in_content_and_reasoning() {
        for &marker in RESERVED_MARKERS {
            let error = render_chat(
                &[Message::new(Role::User, format!("prefix {marker} suffix"))],
                PromptOptions::default(),
            )
            .unwrap_err();
            assert!(
                matches!(error, PromptError::ReservedMarker { .. }),
                "{marker}"
            );
        }

        let error = render_chat(
            &[
                Message::new(Role::User, "hello"),
                Message::assistant("answer", Some("bad </think> escape")),
            ],
            PromptOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PromptError::ReservedMarker {
                field: "reasoning_content",
                ..
            }
        ));
    }

    #[test]
    fn system_must_be_first_and_reasoning_is_assistant_only() {
        let error = render_chat(
            &[
                Message::new(Role::User, "hello"),
                Message::new(Role::System, "late"),
            ],
            PromptOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error, PromptError::SystemMessageNotFirst { index: 1 });

        let mut user = Message::new(Role::User, "hello");
        user.reasoning_content = Some("not allowed".to_owned());
        assert_eq!(
            render_chat(&[user], PromptOptions::default()).unwrap_err(),
            PromptError::ReasoningOnNonAssistant { index: 0 }
        );
    }
}
