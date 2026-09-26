//! Text subset of the shipped Qwen3.6 Jinja template (tools/vision are not accepted).
use crate::models::qwen3_8::prompt::{validate_messages, IM_END, IM_START};
pub use crate::models::qwen3_8::prompt::{Message, PromptError, Role};
#[derive(Debug, Clone, Copy)]
pub struct PromptOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
}
impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_thinking: false,
        }
    }
}
pub fn render_chat(messages: &[Message], options: PromptOptions) -> Result<String, PromptError> {
    if messages.is_empty() {
        return Err(PromptError::NoMessages);
    }
    validate_messages(messages)?;
    let last_user = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .ok_or(PromptError::NoUserQuery)?;
    let mut output = String::new();
    for (index, message) in messages.iter().enumerate() {
        let role = match message.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        output.push_str(IM_START);
        output.push_str(role);
        output.push('\n');
        if message.role == Role::Assistant && (options.preserve_thinking || index > last_user) {
            output.push_str("<think>\n");
            output.push_str(
                message
                    .reasoning_content
                    .as_deref()
                    .unwrap_or_default()
                    .trim(),
            );
            output.push_str("\n</think>\n\n");
        }
        output.push_str(message.content.trim());
        output.push_str(IM_END);
        output.push('\n');
    }
    if options.add_generation_prompt {
        output.push_str("<|im_start|>assistant\n<think>\n");
        if !options.enable_thinking {
            output.push_str("\n</think>\n\n");
        }
    }
    Ok(output)
}
