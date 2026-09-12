//! Text-only Hy4 chat protocol (tool declarations/calls remain outside the CLI surface).

const START: &str = "<｜hy_start:opensource｜>";
const MIDDLE: &str = "<｜hy_middle:opensource｜>";
const END: &str = "<｜hy_end:opensource｜>";
const THINK_START: &str = "<think:opensource｜>";
const THINK_END: &str = "</think:opensource｜>";
const REASONING_MODE: &str = "<｜reasoning_mode:opensource｜>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub preserve_history_reasoning: bool,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_history_reasoning: false,
        }
    }
}

pub fn render_chat(messages: &[Message], options: PromptOptions) -> String {
    let last_user = messages
        .iter()
        .rposition(|message| message.role == Role::User);
    let mut output = String::new();
    let leading_system = messages
        .first()
        .is_some_and(|message| message.role == Role::System);
    if !leading_system {
        push_open(&mut output, Role::System);
        push_reasoning_mode(&mut output, options.enable_thinking);
        output.push_str(END);
    }

    for (index, message) in messages.iter().enumerate() {
        push_open(&mut output, message.role);
        match message.role {
            Role::System => {
                output.push_str(&message.content);
                if index == 0 {
                    push_reasoning_mode(&mut output, options.enable_thinking);
                }
            }
            Role::Assistant => {
                let preserve = options.enable_thinking
                    && (options.preserve_history_reasoning
                        || last_user.is_some_and(|last| index > last));
                output.push_str(THINK_START);
                if preserve {
                    if let Some(reasoning) = &message.reasoning {
                        output.push_str(reasoning);
                    }
                }
                output.push_str(THINK_END);
                output.push_str(&message.content);
            }
            Role::User | Role::Tool => output.push_str(&message.content),
        }
        output.push_str(END);
    }

    if options.add_generation_prompt
        && !messages
            .last()
            .is_some_and(|message| message.role == Role::Assistant)
    {
        push_open(&mut output, Role::Assistant);
        output.push_str(THINK_START);
        if !options.enable_thinking {
            output.push_str(THINK_END);
        }
    }
    output
}

fn push_open(output: &mut String, role: Role) {
    output.push_str(START);
    output.push_str(role.name());
    output.push_str(MIDDLE);
}

fn push_reasoning_mode(output: &mut String, enable_thinking: bool) {
    output.push_str(REASONING_MODE);
    output.push_str(if enable_thinking {
        "reasoning_effort:high"
    } else {
        "reasoning_effort:no_think"
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_user_turn_matches_the_release_template() {
        assert_eq!(
            render_chat(
                &[Message::new(Role::User, "hello")],
                PromptOptions::default()
            ),
            concat!(
                "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>",
                "<｜reasoning_mode:opensource｜>reasoning_effort:high",
                "<｜hy_end:opensource｜>",
                "<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>hello",
                "<｜hy_end:opensource｜>",
                "<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>",
                "<think:opensource｜>"
            )
        );
    }

    #[test]
    fn no_think_closes_the_empty_reasoning_span() {
        let options = PromptOptions {
            enable_thinking: false,
            ..PromptOptions::default()
        };
        let prompt = render_chat(&[Message::new(Role::User, "hello")], options);
        assert!(prompt.contains("reasoning_effort:no_think"));
        assert!(prompt.ends_with("<think:opensource｜></think:opensource｜>"));
    }
}
