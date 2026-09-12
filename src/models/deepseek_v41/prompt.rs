//! Text prompt protocol for DeepSeek-V4.1.
//!
//! Tool/DSML and interleaved image blocks remain separate milestones. This module pins the native
//! text conversation, mid-conversation system messages, thinking markers, and numeric effort ABI.

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const SYSTEM: &str = "<｜System｜>";
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
pub struct ReasoningEffort(u8);

impl ReasoningEffort {
    pub const LOW: Self = Self(50);
    pub const HIGH: Self = Self(75);
    pub const MAX: Self = Self(100);

    pub fn new(value: u8) -> Option<Self> {
        (1..=100).contains(&value).then_some(Self(value))
    }

    pub fn get(self) -> u8 {
        self.0
    }
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
            reasoning_effort: ReasoningEffort::HIGH,
            drop_history_reasoning: true,
            add_bos: true,
        }
    }
}

pub fn render_chat(messages: &[Message], options: PromptOptions) -> String {
    let mut prompt = String::new();
    if options.add_bos {
        prompt.push_str(BOS);
    }

    let last_generation_input = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            (message.role == Role::User || (message.role == Role::System && index > 0))
                .then_some(index)
        });

    if !messages.is_empty()
        && (options.thinking_mode == ThinkingMode::Thinking || messages[0].role == Role::System)
    {
        prompt.push_str(SYSTEM);
    }
    if options.thinking_mode == ThinkingMode::Thinking && !messages.is_empty() {
        prompt.push_str("Reasoning Effort: ");
        prompt.push_str(&options.reasoning_effort.get().to_string());
        prompt
            .push_str(" (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
    }

    for (index, message) in messages.iter().enumerate() {
        match message.role {
            Role::System => {
                if index > 0 {
                    prompt.push_str(SYSTEM);
                }
                prompt.push_str(&message.content);
            }
            Role::User => {
                prompt.push_str(USER);
                prompt.push_str(&message.content);
            }
            Role::Assistant => {
                if options.thinking_mode == ThinkingMode::Thinking {
                    let preserve = !options.drop_history_reasoning
                        || last_generation_input.is_some_and(|last| index > last);
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

        let triggers_generation =
            message.role == Role::User || (message.role == Role::System && index > 0);
        let next_is_assistant = messages
            .get(index + 1)
            .is_some_and(|next| next.role == Role::Assistant);
        if triggers_generation && (next_is_assistant || index + 1 == messages.len()) {
            prompt.push_str(ASSISTANT);
            prompt.push_str(match options.thinking_mode {
                ThinkingMode::Thinking => THINK_START,
                ThinkingMode::Chat => THINK_END,
            });
        }
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_default_numeric_effort_and_thinking_header() {
        let messages = [
            Message::new(Role::System, "You are helpful."),
            Message::new(Role::User, "Hello"),
        ];
        assert_eq!(
            render_chat(&messages, PromptOptions::default()),
            concat!(
                "<｜begin▁of▁sentence｜><｜System｜>",
                "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n",
                "You are helpful.<｜User｜>Hello<｜Assistant｜><think>"
            )
        );
    }

    #[test]
    fn chat_mode_and_mid_conversation_system_follow_reference_transitions() {
        let messages = [
            Message::new(Role::User, "one"),
            Message::assistant("two", None::<String>),
            Message::new(Role::System, "new rule"),
        ];
        let options = PromptOptions {
            thinking_mode: ThinkingMode::Chat,
            ..PromptOptions::default()
        };
        assert_eq!(
            render_chat(&messages, options),
            concat!(
                "<｜begin▁of▁sentence｜><｜User｜>one<｜Assistant｜></think>",
                "two<｜end▁of▁sentence｜><｜System｜>new rule<｜Assistant｜></think>"
            )
        );
    }

    #[test]
    fn numeric_effort_accepts_only_one_through_one_hundred() {
        assert_eq!(ReasoningEffort::new(0), None);
        assert_eq!(ReasoningEffort::new(1).unwrap().get(), 1);
        assert_eq!(ReasoningEffort::new(100).unwrap().get(), 100);
        assert_eq!(ReasoningEffort::new(101), None);
    }
}
