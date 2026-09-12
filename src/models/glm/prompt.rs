//! Text-only GLM-5.2 chat protocol.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub preserve_history_reasoning: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_history_reasoning: false,
        }
    }
}

/// Renders the text-only portion of the official GLM-5.2 chat template.
pub fn render_chat(messages: &[ChatMessage], options: ChatTemplateOptions) -> String {
    let last_user = messages
        .iter()
        .rposition(|message| message.role == ChatRole::User);
    let mut prompt = String::from("[gMASK]<sop>");
    if options.enable_thinking {
        prompt.push_str("<|system|>Reasoning Effort: Max");
    }
    let mut previous_was_tool = false;
    for (index, message) in messages.iter().enumerate() {
        match message.role {
            ChatRole::System => {
                prompt.push_str("<|system|>");
                prompt.push_str(&message.content);
            }
            ChatRole::User => {
                prompt.push_str("<|user|>");
                prompt.push_str(&message.content);
            }
            ChatRole::Assistant => {
                prompt.push_str("<|assistant|>");
                let (reasoning, visible) = split_reasoning(&message.content);
                let preserve = options.preserve_history_reasoning
                    || last_user.is_some_and(|last| index > last);
                if preserve {
                    if let Some(reasoning) = reasoning {
                        prompt.push_str("<think>");
                        prompt.push_str(reasoning);
                        prompt.push_str("</think>");
                    } else {
                        prompt.push_str("<think></think>");
                    }
                } else {
                    prompt.push_str("<think></think>");
                }
                prompt.push_str(visible.trim());
            }
            ChatRole::Tool => {
                if !previous_was_tool {
                    prompt.push_str("<|observation|>");
                }
                prompt.push_str("<tool_response>");
                prompt.push_str(&message.content);
                prompt.push_str("</tool_response>");
            }
        }
        previous_was_tool = message.role == ChatRole::Tool;
    }
    if options.add_generation_prompt {
        prompt.push_str("<|assistant|>");
        if options.enable_thinking {
            prompt.push_str("<think>");
        } else {
            prompt.push_str("<think></think>");
        }
    }
    prompt
}

fn split_reasoning(content: &str) -> (Option<&str>, &str) {
    if let Some(end) = content.find("</think>") {
        let reasoning_start = content[..end]
            .rfind("<think>")
            .map(|start| start + "<think>".len())
            .unwrap_or(0);
        (
            Some(&content[reasoning_start..end]),
            &content[end + "</think>".len()..],
        )
    } else {
        (None, content)
    }
}
