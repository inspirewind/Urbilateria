use super::commands::{clean_text, parse, takes_arguments, Command, COMMANDS};
use super::report::{
    decode_entry, explain_entry, help_entry, inspection_entry, list_entry, plan_entry,
    preflight_entry, probe_entry, tokenize_entry, Detail, Entry, Kind,
};
use super::worker::{Finished, Report, Request, Task};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui_textarea::{CursorMove, TextArea, WrapMode};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

const INPUT_LIMIT: usize = 16_384;
const HISTORY_LIMIT: usize = 100;

pub struct Pending {
    pub id: u64,
    pub started: Instant,
    pub task: Task,
}

pub struct App {
    pub editor: TextArea<'static>,
    pub entries: VecDeque<Entry>,
    pub model_path: Option<PathBuf>,
    pub model_family: Option<String>,
    pub pending: Option<Pending>,
    pub quit: bool,
    /// None follows the latest output; Some anchors a row from the start.
    pub scroll: Option<usize>,
    pub max_scroll: usize,
    pub page_size: usize,
    pub completion_index: usize,
    completion_dismissed: bool,
    history: VecDeque<String>,
    history_index: Option<usize>,
    draft: String,
    next_id: u64,
}

impl App {
    pub fn new(model_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            editor: editor(""),
            entries: VecDeque::new(),
            model_path,
            model_family: None,
            pending: None,
            quit: false,
            scroll: None,
            max_scroll: 0,
            page_size: 10,
            completion_index: 0,
            completion_dismissed: false,
            history: VecDeque::new(),
            history_index: None,
            draft: String::new(),
            next_id: 0,
        };
        app.push(Entry::new(
            Kind::Info,
            "Welcome to Urbilateria",
            vec![
                Detail::text("Explore your model checkpoints from the terminal."),
                Detail::text("Type / for commands, or /help for keyboard shortcuts."),
                Detail::text(match &app.model_path {
                    Some(path) => format!(
                        "Current path: {}\nRun /inspect to read its metadata.",
                        path.display()
                    ),
                    None => "Start with /inspect \"/path/to/model\". Use /probe to sample tensor values.".into(),
                }),
            ],
        ));
        app
    }

    pub fn input(&self) -> String {
        self.editor.lines().join("\n")
    }

    pub fn completions(&self) -> Vec<(&'static str, &'static str)> {
        let lines = self.editor.lines();
        if self.completion_dismissed || lines.len() != 1 {
            return Vec::new();
        }
        let text = &lines[0];
        if !text.starts_with('/') || text.chars().any(char::is_whitespace) {
            return Vec::new();
        }
        COMMANDS
            .iter()
            .copied()
            .filter(|(name, _)| name.starts_with(text) && *name != text)
            .collect()
    }

    pub fn paste(&mut self, text: &str) {
        let remaining = INPUT_LIMIT.saturating_sub(self.input_len());
        self.editor.insert_str(clean_text(text, remaining));
        self.edited();
    }

    pub fn key(&mut self, key: KeyEvent) -> Option<Request> {
        if key.kind == KeyEventKind::Release {
            return None;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let completions = self.completions();
        match key.code {
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Char('d') if ctrl && self.input().is_empty() => self.quit = true,
            KeyCode::Char('j') if ctrl => self.newline(),
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
            {
                self.newline()
            }
            KeyCode::Enter => return self.submit(),
            KeyCode::Esc => self.completion_dismissed = true,
            KeyCode::Tab if !completions.is_empty() => {
                let (name, _) = completions[self.completion_index % completions.len()];
                self.editor = editor(&format!(
                    "{name}{}",
                    if takes_arguments(name) { " " } else { "" }
                ));
                self.edited();
            }
            KeyCode::Tab => {}
            KeyCode::Up if !completions.is_empty() => {
                self.completion_index =
                    (self.completion_index + completions.len() - 1) % completions.len();
            }
            KeyCode::Down if !completions.is_empty() => {
                self.completion_index = (self.completion_index + 1) % completions.len();
            }
            KeyCode::Char('p') if ctrl => self.previous_history(),
            KeyCode::Char('n') if ctrl => self.next_history(),
            KeyCode::Up if self.editor.cursor().0 == 0 => self.previous_history(),
            KeyCode::Down if self.editor.cursor().0 + 1 == self.editor.lines().len() => {
                self.next_history()
            }
            KeyCode::PageUp => {
                self.scroll = Some(
                    self.scroll
                        .unwrap_or(self.max_scroll)
                        .saturating_sub(self.page_size),
                );
            }
            KeyCode::PageDown => {
                let row = self
                    .scroll
                    .unwrap_or(self.max_scroll)
                    .saturating_add(self.page_size);
                self.scroll = (row < self.max_scroll).then_some(row);
            }
            KeyCode::Home if ctrl => self.scroll = Some(0),
            KeyCode::End if ctrl => self.scroll = None,
            _ => {
                // At the limit, retain navigation/deletion and reject only input that grows text.
                if (self.input_len() < INPUT_LIMIT || !matches!(key.code, KeyCode::Char(_)) || ctrl)
                    && self.editor.input(key)
                {
                    self.edited();
                }
            }
        }
        None
    }

    fn newline(&mut self) {
        if self.input_len() < INPUT_LIMIT {
            self.editor.insert_newline();
            self.edited();
        }
    }

    fn input_len(&self) -> usize {
        self.editor
            .lines()
            .iter()
            .map(|line| line.chars().count())
            .sum::<usize>()
            + self.editor.lines().len().saturating_sub(1)
    }

    fn edited(&mut self) {
        if self.input_len() > INPUT_LIMIT {
            self.editor = editor(&clean_text(&self.input(), INPUT_LIMIT));
        }
        self.completion_dismissed = false;
        self.completion_index = 0;
        self.history_index = None;
    }

    fn previous_history(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let index = match self.history_index {
            None => {
                self.draft = self.input();
                self.history.len() - 1
            }
            Some(index) => index.saturating_sub(1),
        };
        self.history_index = Some(index);
        self.editor = editor(&self.history[index]);
        self.completion_dismissed = true;
    }

    fn next_history(&mut self) {
        if let Some(index) = self.history_index {
            if index + 1 < self.history.len() {
                self.history_index = Some(index + 1);
                self.editor = editor(&self.history[index + 1]);
            } else {
                self.history_index = None;
                self.editor = editor(&self.draft);
            }
            self.completion_dismissed = true;
        }
    }

    pub fn submit(&mut self) -> Option<Request> {
        let input = self.input();
        let input = input.trim();
        if input.is_empty() {
            return None;
        }
        if self.history.back().is_none_or(|last| last != input) {
            if self.history.len() == HISTORY_LIMIT {
                self.history.pop_front();
            }
            self.history.push_back(input.into());
        }
        self.editor = editor("");
        self.draft.clear();
        self.edited();
        self.scroll = None;
        self.push(Entry::message(Kind::Input, "You", input));
        match parse(input) {
            Ok(Command::Help) => self.push(help_entry()),
            Ok(Command::Version) => self.push(Entry::message(
                Kind::Info,
                "Version",
                format!("urb {}", env!("CARGO_PKG_VERSION")),
            )),
            Ok(Command::Clear) => {
                self.entries.clear();
                self.scroll = None;
            }
            Ok(Command::Quit) => self.quit = true,
            Ok(Command::Inspect(path)) => return self.start(path, Task::Inspect),
            Ok(Command::Plan(path, options)) => return self.start(path, Task::Plan(options)),
            Ok(Command::Preflight(path, options)) => {
                return self.start(path, Task::Preflight(options))
            }
            Ok(Command::List(path, options)) => return self.start(path, Task::List(options)),
            Ok(Command::Explain(path)) => return self.start(path, Task::Explain),
            Ok(Command::Probe(path, tensor, samples)) => {
                return self.start(path, Task::Probe { tensor, samples })
            }
            Ok(Command::Tokenize(path, text, options)) => {
                return self.start(path, Task::Tokenize { text, options })
            }
            Ok(Command::Decode(path, ids, skip_special)) => {
                return self.start(path, Task::Decode { ids, skip_special })
            }
            Err(error) => self.error(error),
        }
        None
    }

    fn start(&mut self, path: Option<PathBuf>, task: Task) -> Option<Request> {
        if let Some(pending) = &self.pending {
            self.error(format!(
                "{} is already running. You can keep editing or quit with Ctrl+C.",
                pending.task.command()
            ));
        } else if let Some(path) = path.or_else(|| self.model_path.clone()) {
            self.next_id += 1;
            self.pending = Some(Pending {
                id: self.next_id,
                started: Instant::now(),
                task: task.clone(),
            });
            return Some(Request {
                id: self.next_id,
                path,
                task,
            });
        } else {
            self.error(
                "No model selected. Use /inspect MODEL_DIR first (quote paths with spaces).",
            );
        }
        None
    }

    pub fn finished(&mut self, event: Finished) {
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.id != event.id)
        {
            return;
        }
        let pending = self.pending.take().expect("matching pending analysis");
        match event.result {
            Ok(report) => {
                let (path, family) = report.identity();
                if family.is_some() || self.model_path.as_deref() != Some(path) {
                    self.model_family = family.map(|family| family.to_string());
                }
                self.model_path = Some(path.to_owned());
                let elapsed = pending.started.elapsed();
                self.push(match report {
                    Report::Inspection(result) => inspection_entry(&result, elapsed),
                    Report::Planning(result) => plan_entry(&result, elapsed),
                    Report::Preflight(result) => preflight_entry(&result, elapsed),
                    Report::Listing(result) => list_entry(&result, elapsed),
                    Report::Explanation(result) => explain_entry(&result, elapsed),
                    Report::Probe(result) => probe_entry(&result, elapsed),
                    Report::Tokenization(result) => tokenize_entry(&result, elapsed),
                    Report::Decoding(result) => decode_entry(&result, elapsed),
                });
            }
            Err(error) => self.error(error),
        }
    }

    pub fn error(&mut self, error: impl Into<String>) {
        self.push(Entry::message(Kind::Error, "Error", error));
    }

    pub fn push(&mut self, entry: Entry) {
        if self.entries.len() == HISTORY_LIMIT {
            self.entries.pop_front();
            // Recompute from the oldest retained entry after eviction.
            self.scroll = self.scroll.map(|_| 0);
        }
        self.entries.push_back(entry);
    }
}

fn editor(text: &str) -> TextArea<'static> {
    let mut editor = TextArea::from(text.split('\n'));
    editor.set_wrap_mode(WrapMode::Glyph);
    editor.set_placeholder_text("Type / for commands");
    editor.move_cursor(CursorMove::Bottom);
    editor.move_cursor(CursorMove::End);
    editor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn paste_is_editable_and_never_executes() {
        let mut app = App::new(None);
        app.paste("/quit\r\n中文🙂");
        assert!(!app.quit);
        assert_eq!(app.input(), "/quit\n中文🙂");
        app.key(key(KeyCode::Backspace));
        assert_eq!(app.input(), "/quit\n中文");
        app.key(key(KeyCode::Enter));
        assert!(!app.quit);
        assert!(matches!(app.entries.back().unwrap().kind, Kind::Error));
        app.paste("/help");
        app.key(key(KeyCode::Enter));
        assert_eq!(app.entries.back().unwrap().title, "Commands & keys");
    }

    #[test]
    fn completion_and_history_restore_the_draft() {
        let mut app = App::new(None);
        app.paste("/he");
        app.key(key(KeyCode::Tab));
        assert_eq!(app.input(), "/help");
        app.submit();
        app.paste("unfinished 中文");
        app.key(key(KeyCode::Up));
        assert_eq!(app.input(), "/help");
        app.key(key(KeyCode::Down));
        assert_eq!(app.input(), "unfinished 中文");
        app.key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        app.paste("second line");
        app.key(key(KeyCode::Up));
        assert!(app.input().ends_with("second line"));
        assert_eq!(app.editor.cursor().0, 0);
    }

    #[test]
    fn inspection_survives_clear_and_failure_allows_retry() {
        let mut app = App::new(Some(PathBuf::from("/old/model")));
        app.paste("/inspect /new/model");
        let request = app.submit().unwrap();
        app.paste("/inspect /another/model");
        assert!(app.submit().is_none());
        assert_eq!(app.pending.as_ref().unwrap().id, request.id);
        app.paste("/clear");
        app.submit();
        assert!(app.entries.is_empty());
        app.finished(Finished {
            id: request.id + 1,
            result: Err("stale".into()),
        });
        assert!(app.pending.is_some());
        app.finished(Finished {
            id: request.id,
            result: Err("missing config".into()),
        });
        assert!(app.pending.is_none());
        assert_eq!(app.model_path, Some(PathBuf::from("/old/model")));
        assert_eq!(
            app.entries.back().unwrap().details[0].text,
            "missing config"
        );
        app.paste("/inspect");
        assert_eq!(app.submit().unwrap().path, PathBuf::from("/old/model"));
    }

    #[test]
    fn paste_is_bounded_on_character_boundaries() {
        let mut app = App::new(None);
        app.paste(&"中".repeat(INPUT_LIMIT + 20));
        assert_eq!(app.input_len(), INPUT_LIMIT);
        app.paste("🙂");
        assert_eq!(app.input_len(), INPUT_LIMIT);
    }

    #[test]
    fn planning_and_preflight_reuse_the_model_and_serialize_background_work() {
        let mut app = App::new(Some(PathBuf::from("/current/model")));
        app.paste("/pl");
        app.key(key(KeyCode::Tab));
        assert_eq!(app.input(), "/plan ");
        app.paste("--ram-gib 4 --context 512");
        let request = app.submit().unwrap();
        assert_eq!(request.path, PathBuf::from("/current/model"));
        assert!(
            matches!(request.task, Task::Plan(options) if options.context == 512 && options.ram_bytes == Some(4 * 1024 * 1024 * 1024))
        );
        app.paste("/preflight --context 64");
        assert!(app.submit().is_none());
        assert!(app.entries.back().unwrap().details[0]
            .text
            .contains("/plan is already running"));
        app.finished(Finished {
            id: request.id,
            result: Err("budget check failed".into()),
        });
        app.paste("/pre");
        app.key(key(KeyCode::Tab));
        assert_eq!(app.input(), "/preflight ");
        app.paste("--context 64 --expert-slots 2");
        let request = app.submit().unwrap();
        assert_eq!(request.path, PathBuf::from("/current/model"));
        assert!(
            matches!(request.task, Task::Preflight(options) if options.context == 64 && options.expert_slots == 2)
        );
        app.paste("/clear");
        app.submit();
        app.finished(Finished {
            id: request.id,
            result: Err("invalid schema".into()),
        });
        assert_eq!(app.entries.len(), 1);
        assert_eq!(app.model_path, Some(PathBuf::from("/current/model")));
        app.paste("/plan --context 0");
        assert!(app.submit().is_none());
        assert!(app.pending.is_none());
    }

    #[test]
    fn version_works_without_a_model_and_during_background_work() {
        let mut app = App::new(None);
        app.paste("/ver");
        app.key(key(KeyCode::Tab));
        assert_eq!(app.input(), "/version");
        assert!(app.submit().is_none());
        assert_eq!(app.entries.back().unwrap().title, "Version");
        app.paste("/tokenize hello");
        assert!(app.submit().is_none());
        assert!(app.entries.back().unwrap().details[0]
            .text
            .contains("No model selected"));
        app.paste("/tokenize hello --model '/model one'");
        let request = app.submit().unwrap();
        assert_eq!(request.path, PathBuf::from("/model one"));
        app.paste("/version");
        assert!(app.submit().is_none());
        assert_eq!(app.pending.as_ref().unwrap().id, request.id);
        assert_eq!(app.entries.back().unwrap().title, "Version");
        app.paste("/decode 1");
        assert!(app.submit().is_none());
        assert!(app.entries.back().unwrap().details[0]
            .text
            .contains("/tokenize is already running"));
    }

    #[test]
    fn listing_a_different_checkpoint_clears_the_previous_family_and_errors_keep_selection() {
        let mut app = App::new(Some("/glm".into()));
        app.model_family = Some("GLM-5.2".into());
        app.paste("/li");
        app.key(key(KeyCode::Tab));
        assert_eq!(app.input(), "/list ");
        app.paste("weight --model /unknown");
        let request = app.submit().unwrap();
        app.finished(Finished {
            id: request.id,
            result: Ok(Report::Listing(Box::new(
                urbilateria::analysis::TensorListing {
                    model_path: request.path,
                    options: urbilateria::analysis::ListOptions::default(),
                    total_matches: 0,
                    tensors: vec![],
                },
            ))),
        });
        assert_eq!(
            app.model_path.as_deref(),
            Some(std::path::Path::new("/unknown"))
        );
        assert_eq!(app.model_family, None);
        app.paste("/probe missing --model /another");
        let request = app.submit().unwrap();
        app.finished(Finished {
            id: request.id,
            result: Err("missing tensor".into()),
        });
        app.paste("/explain");
        assert_eq!(app.submit().unwrap().path, PathBuf::from("/unknown"));
    }
}
