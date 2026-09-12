//! Text-only DeepSeek-V4 chat protocol.

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
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
pub enum ThinkingMode {
    Thinking,
    Chat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    High,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptOptions {
    pub thinking_mode: ThinkingMode,
    pub reasoning_effort: ReasoningEffort,
    pub drop_history_reasoning: bool,
    pub add_bos: bool,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            thinking_mode: ThinkingMode::Thinking,
            reasoning_effort: ReasoningEffort::Low,
            drop_history_reasoning: true,
            add_bos: true,
        }
    }
}

/// Renders the official text-only protocol. Tool/DSML objects remain a separate milestone.
pub fn render_chat(messages: &[Message], options: PromptOptions) -> String {
    let last_user = messages
        .iter()
        .rposition(|message| message.role == Role::User);
    let mut prompt = String::new();
    if options.add_bos {
        prompt.push_str(BOS);
    }
    if options.thinking_mode == ThinkingMode::Thinking {
        prompt.push_str(reasoning_prefix(options.reasoning_effort));
    }

    for (index, message) in messages.iter().enumerate() {
        match message.role {
            Role::System => prompt.push_str(&message.content),
            Role::User => {
                prompt.push_str(USER);
                prompt.push_str(&message.content);
            }
            Role::Assistant => {
                if options.thinking_mode == ThinkingMode::Thinking {
                    let preserve = !options.drop_history_reasoning
                        || last_user.is_some_and(|last| index > last);
                    if preserve {
                        if let Some(reasoning) = &message.reasoning_content {
                            prompt.push_str(reasoning);
                        }
                        prompt.push_str(THINK_END);
                    }
                }
                prompt.push_str(&message.content);
                prompt.push_str(EOS);
            }
        }

        let next_role = messages.get(index + 1).map(|next| next.role);
        if message.role == Role::User && (next_role.is_none() || next_role == Some(Role::Assistant))
        {
            prompt.push_str(ASSISTANT);
            prompt.push_str(match options.thinking_mode {
                ThinkingMode::Thinking => THINK_START,
                ThinkingMode::Chat => THINK_END,
            });
        }
    }
    prompt
}

fn reasoning_prefix(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Low => "",
        ReasoningEffort::High => concat!(
            "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
            "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
            "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
        ),
        ReasoningEffort::Max => concat!(
            "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n",
            "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n",
            "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readme_prompt_matches_official_reference() {
        let messages = [
            Message::new(Role::System, "You are a helpful assistant."),
            Message::new(Role::User, "What is 2+2?"),
        ];
        assert_eq!(
            render_chat(&messages, PromptOptions::default()),
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"
        );
    }

    #[test]
    fn chat_mode_closes_thinking_immediately() {
        let options = PromptOptions {
            thinking_mode: ThinkingMode::Chat,
            ..PromptOptions::default()
        };
        assert_eq!(
            render_chat(&[Message::new(Role::User, "hello")], options),
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
    }
}
