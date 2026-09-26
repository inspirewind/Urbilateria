use super::commands::{GenerateOptions, SettingsUpdate};
use crate::chat::{Conversation, Turn, MAX_CHAT_TURNS};
use std::collections::VecDeque;
use urbilateria::analysis::detect_available_ram;

pub const HISTORY_BYTES: usize = 512 * 1024;

pub struct ChatSettings {
    detected_ram: Option<String>,
    pub ram_gib: Option<String>,
    pub max_new_tokens: usize,
    pub threads: Option<usize>,
    pub thinking: bool,
}

impl Default for ChatSettings {
    fn default() -> Self {
        Self {
            detected_ram: None,
            ram_gib: None,
            max_new_tokens: 512,
            threads: None,
            thinking: true,
        }
    }
}

impl ChatSettings {
    pub fn update(&mut self, update: SettingsUpdate) {
        if let Some(ram) = update.ram_gib {
            self.ram_gib = ram;
            self.detected_ram = None;
        }
        if let Some(threads) = update.threads {
            self.threads = threads;
        }
        if let Some(tokens) = update.max_new_tokens {
            self.max_new_tokens = tokens;
        }
        if let Some(thinking) = update.thinking {
            self.thinking = thinking;
        }
    }

    pub fn remember(&mut self, options: &GenerateOptions) {
        self.ram_gib = options.value("--ram-gib").map(str::to_owned);
        self.threads = options
            .value("--threads")
            .and_then(|value| value.parse().ok());
        self.max_new_tokens = options
            .value("--max-new-tokens")
            .and_then(|value| value.parse().ok())
            .unwrap_or(1);
        if !options.flag("--raw-prompt") {
            self.thinking = !options.flag("--no-thinking");
        }
    }

    pub fn generation(&mut self, prompt: &str) -> Result<GenerateOptions, String> {
        let ram = self.ram_gib.clone().or_else(|| self.detected_ram.clone()).or_else(|| {
            detect_available_ram().filter(|bytes| *bytes > 0)
                .map(|bytes| (bytes as f64 / 1_073_741_824.0).to_string())
        }).ok_or("Cannot detect available RAM. Set /settings --ram-gib N once, then send your message again.")?;
        if self.ram_gib.is_none() {
            self.detected_ram = Some(ram.clone());
        }
        let mut args = vec![
            "--prompt".into(),
            prompt.into(),
            "--ram-gib".into(),
            ram.into(),
            "--max-new-tokens".into(),
            self.max_new_tokens.to_string().into(),
            "--allow-large-model".into(),
        ];
        if let Some(threads) = self.threads {
            args.extend(["--threads".into(), threads.to_string().into()]);
        }
        if !self.thinking {
            args.push("--no-thinking".into());
        }
        Ok(GenerateOptions {
            args,
            conversation: None,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "RAM: {} · max new tokens: {} · threads: {} · thinking: {}",
            self.ram_gib
                .as_ref()
                .map(|ram| format!("{ram} GiB"))
                .unwrap_or_else(|| "auto".into()),
            self.max_new_tokens,
            self.threads
                .map(|threads| threads.to_string())
                .unwrap_or_else(|| "auto".into()),
            if self.thinking { "on" } else { "off" }
        )
    }
}

#[derive(Default)]
pub struct History {
    pub turns: VecDeque<Turn>,
    pub epoch: u64,
}

impl History {
    pub fn clear(&mut self) {
        self.turns.clear();
        self.epoch += 1;
    }

    pub fn conversation(&self, prompt: String, same_model: bool) -> Conversation {
        Conversation {
            turns: if same_model {
                self.turns.iter().cloned().collect()
            } else {
                vec![]
            },
            prompt,
        }
    }

    pub fn commit(&mut self, turn: Turn) {
        self.turns.push_back(turn);
        let mut bytes: usize = self
            .turns
            .iter()
            .map(|turn| turn.user.len() + turn.assistant.len())
            .sum();
        while bytes > HISTORY_BYTES || self.turns.len() > MAX_CHAT_TURNS {
            let turn = self.turns.pop_front().unwrap();
            bytes -= turn.user.len() + turn.assistant.len();
        }
    }
}

pub struct PendingTurn {
    pub user: String,
    pub thinking: bool,
    pub epoch: u64,
    pub response: String,
    pub overflow: bool,
}

impl PendingTurn {
    pub fn append(&mut self, text: &str) {
        if self.overflow {
            return;
        }
        if self.response.len() + self.user.len() + text.len() > HISTORY_BYTES {
            self.response.clear();
            self.overflow = true;
        } else {
            self.response.push_str(text);
        }
    }

    pub fn finish(mut self) -> Turn {
        if self.response.ends_with('\n') {
            self.response.pop();
        }
        Turn {
            user: self.user,
            assistant: self.response,
            thinking: self.thinking,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::commands::{parse, Command};
    use super::*;

    #[test]
    fn session_settings_remember_runtime_options_but_not_profile_destinations() {
        let mut settings = ChatSettings::default();
        assert_eq!(settings.max_new_tokens, 512);
        let Command::Generate(_, options) = parse("/generate --ram-gib 2 --allow-large-model --threads 3 --max-new-tokens 128 --no-thinking --profile-json run.json -- '--profile'").unwrap()
            else { panic!("expected generation") };
        assert!(!options.flag("--profile")); // Literal prompt must not become a flag.
        settings.remember(&options);
        let options = settings.generation("don't quote \"this\n中文🙂").unwrap();
        assert_eq!(
            options.value("--prompt"),
            Some("don't quote \"this\n中文🙂")
        );
        assert_eq!(options.value("--ram-gib"), Some("2"));
        assert_eq!(options.value("--threads"), Some("3"));
        assert_eq!(options.value("--max-new-tokens"), Some("128"));
        assert!(options.flag("--no-thinking"));
        assert!(!options.flag("--profile-json"));
        let Command::Settings(update) =
            parse("/settings --ram-gib auto --threads auto --thinking").unwrap()
        else {
            panic!("expected settings")
        };
        settings.update(update);
        assert_eq!(settings.ram_gib, None);
        assert_eq!(settings.threads, None);
        assert!(settings.thinking);
        assert_eq!(settings.max_new_tokens, 128);
    }

    #[test]
    fn history_is_bounded_by_complete_turns_and_independent_of_display_clipping() {
        let mut history = History::default();
        for index in 0..40 {
            history.commit(Turn {
                user: index.to_string(),
                assistant: "answer".into(),
                thinking: false,
            });
        }
        assert_eq!(history.turns.len(), MAX_CHAT_TURNS);
        assert_eq!(history.turns.front().unwrap().user, "8");
        history.commit(Turn {
            user: "large".into(),
            assistant: "中".repeat(100_000),
            thinking: false,
        });
        history.commit(Turn {
            user: "latest".into(),
            assistant: "中".repeat(100_000),
            thinking: false,
        });
        assert_eq!(history.turns.len(), 1);
        assert_eq!(history.turns[0].user, "latest");
        assert!(history.conversation("next".into(), false).turns.is_empty());
        assert_eq!(history.conversation("next".into(), true).turns.len(), 1);
        history.clear();
        assert!(history.turns.is_empty());
        assert_eq!(history.epoch, 1);
    }

    #[test]
    fn pending_response_preserves_model_newlines_and_never_commits_a_clipped_tail() {
        let mut turn = PendingTurn {
            user: "hello".into(),
            response: String::new(),
            thinking: false,
            epoch: 0,
            overflow: false,
        };
        turn.append("first\n");
        turn.append("中文🙂\n\n"); // One model newline, then the CLI's final newline.
        assert_eq!(turn.finish().assistant, "first\n中文🙂\n");
        let mut turn = PendingTurn {
            user: "hello".into(),
            response: String::new(),
            thinking: true,
            epoch: 0,
            overflow: false,
        };
        turn.append(&"中".repeat(HISTORY_BYTES / 3 + 1));
        turn.append("tail");
        assert!(turn.overflow);
        assert!(turn.response.is_empty());
    }
}
