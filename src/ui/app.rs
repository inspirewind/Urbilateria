use super::chat::{ChatSettings, History, PendingTurn};
use super::commands::{clean_text, parse, takes_arguments, Command, GenerateOptions, COMMANDS};
use super::generate::{Event as GenerationEvent, Outcome, Stream};
use super::report::{
    decode_entry, explain_entry, help_entry, inspection_entry, list_entry, plan_entry,
    preflight_entry, probe_entry, tokenize_entry, Detail, Entry, GenerationTranscript, Kind,
    ModelSummary, RuntimeInfo,
};
use super::worker::{Finished, Report, Request, Task};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui_textarea::{CursorMove, TextArea, WrapMode};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

const INPUT_LIMIT: usize = 16_384;
const HISTORY_LIMIT: usize = 100;
// Also bound long generation transcripts. Together with each entry's line limit, this keeps
// wrapped output within Paragraph's u16 scroll range at the minimum supported terminal width.
const TRANSCRIPT_BYTES: usize = 512 * 1024;

pub struct Pending {
    pub id: u64,
    pub started: Instant,
    pub task: Task,
    pub cancelling: bool,
    path: PathBuf,
    output: Option<GenerationTranscript>,
    turn: Option<PendingTurn>,
}

pub struct App {
    pub editor: TextArea<'static>,
    pub entries: VecDeque<Entry>,
    pub model_path: Option<PathBuf>,
    pub model_family: Option<String>,
    pub model_summary: Option<ModelSummary>,
    pub runtime: Option<RuntimeInfo>,
    pub runtime_open: bool,
    pub runtime_scroll: Option<usize>,
    pub runtime_max_scroll: usize,
    pub runtime_page_size: usize,
    pub settings: ChatSettings,
    pub conversation: History,
    pub pending: Option<Pending>,
    pub quit: bool,
    pub cancel_requested: bool,
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
            model_summary: None,
            runtime: None,
            runtime_open: false,
            runtime_scroll: None,
            runtime_max_scroll: 0,
            runtime_page_size: 1,
            settings: ChatSettings::default(),
            conversation: History::default(),
            pending: None,
            quit: false,
            cancel_requested: false,
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
                Detail::text("Type a message to chat, / for commands, or /help for keyboard shortcuts."),
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
        if self.runtime_open {
            return;
        }
        let remaining = INPUT_LIMIT.saturating_sub(self.input_len());
        self.editor.insert_str(clean_text(text, remaining));
        self.edited();
    }

    pub fn key(&mut self, key: KeyEvent) -> Option<Request> {
        if key.kind == KeyEventKind::Release {
            return None;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        // The full runtime view works on narrow terminals too, without inserting logs into chat.
        match key.code {
            KeyCode::F(2) => {
                self.runtime_open = !self.runtime_open;
                return None;
            }
            KeyCode::Esc if self.runtime_open => {
                self.runtime_open = false;
                return None;
            }
            KeyCode::PageUp if self.runtime_open => {
                self.runtime_scroll = Some(
                    self.runtime_scroll
                        .unwrap_or(self.runtime_max_scroll)
                        .saturating_sub(self.runtime_page_size),
                );
                return None;
            }
            KeyCode::PageDown if self.runtime_open => {
                let row = self
                    .runtime_scroll
                    .unwrap_or(self.runtime_max_scroll)
                    .saturating_add(self.runtime_page_size);
                self.runtime_scroll = (row < self.runtime_max_scroll).then_some(row);
                return None;
            }
            KeyCode::Home if ctrl && self.runtime_open => {
                self.runtime_scroll = Some(0);
                return None;
            }
            KeyCode::End if ctrl && self.runtime_open => {
                self.runtime_scroll = None;
                return None;
            }
            _ => {}
        }
        if self.runtime_open {
            if ctrl
                && (key.code == KeyCode::Char('c')
                    || (key.code == KeyCode::Char('d') && self.input().is_empty()))
            {
                self.quit = true;
            }
            return None;
        }
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
            KeyCode::Esc => {
                if completions.is_empty() {
                    if let Some(pending) = &mut self.pending {
                        if matches!(pending.task, Task::Generate(_)) && !pending.cancelling {
                            pending.cancelling = true;
                            self.cancel_requested = true;
                        }
                    }
                }
                self.completion_dismissed = true;
            }
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
        let original = self.input();
        let input = if original.trim_start().starts_with('/') {
            original.trim()
        } else {
            original.as_str()
        };
        if input.trim().is_empty() {
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
            Ok(Command::Message(text)) => {
                if self.model_path.is_none() {
                    self.error("No model selected. Use /inspect MODEL_DIR first (quote paths with spaces).");
                } else if let Some(pending) = &self.pending {
                    self.error(format!("{} is already running. Wait for it to finish, or press Esc to stop generation.", pending.task.command()));
                } else {
                    match self.settings.generation(&text) {
                        Ok(options) => return self.start_generation(None, options, false),
                        Err(error) => self.error(error),
                    }
                }
            }
            Ok(Command::Settings(update)) => {
                self.settings.update(update);
                self.push(Entry::message(Kind::Info, "Conversation settings", format!(
                    "{}\nSend plain text to chat. Options: --ram-gib N|auto --max-new-tokens N --threads N|auto --thinking | --no-thinking.",
                    self.settings.describe()
                )));
            }
            Ok(Command::Help) => self.push(help_entry()),
            Ok(Command::Version) => self.push(Entry::message(
                Kind::Info,
                "Version",
                format!("urb {}", env!("CARGO_PKG_VERSION")),
            )),
            Ok(Command::Clear) => {
                self.entries.clear();
                self.scroll = None;
                self.conversation.clear();
                if let Some(runtime) = &mut self.runtime {
                    runtime.clear_log();
                }
                self.runtime_scroll = None;
                if let Some(pending) = &mut self.pending {
                    if let Some(output) = &mut pending.output {
                        *output = GenerationTranscript::default();
                    }
                }
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
            Ok(Command::Generate(path, options)) => {
                return self.start_generation(path, options, true)
            }
            Err(error) => self.error(error),
        }
        None
    }

    fn start_generation(
        &mut self,
        path: Option<PathBuf>,
        mut options: GenerateOptions,
        remember: bool,
    ) -> Option<Request> {
        let path = path.or_else(|| self.model_path.clone());
        if self.pending.is_some() || path.is_none() {
            return self.start(path, Task::Generate(options));
        }
        let remembered = options.clone();
        if !options.flag("--raw-prompt") {
            if let Some(prompt) = options.take_prompt() {
                options.conversation = Some(
                    self.conversation
                        .conversation(prompt, path == self.model_path),
                );
            }
        }
        let request = self.start(path, Task::Generate(options));
        if request.is_some() && remember {
            self.settings.remember(&remembered);
        }
        request
    }

    fn select_model(&mut self, path: PathBuf) {
        if self.model_path.as_ref() != Some(&path) {
            self.conversation.clear();
            self.model_summary = None;
            self.model_family = None;
            self.runtime = None;
        }
        self.model_path = Some(path);
    }

    fn start(&mut self, path: Option<PathBuf>, task: Task) -> Option<Request> {
        if let Some(pending) = &self.pending {
            self.error(format!(
                "{} is already running. You can keep editing or quit with Ctrl+C.",
                pending.task.command()
            ));
        } else if let Some(path) = path.or_else(|| self.model_path.clone()) {
            self.next_id += 1;
            if matches!(task, Task::Generate(_)) {
                self.runtime = Some(RuntimeInfo::new(&path));
                self.runtime_scroll = None;
            }
            self.pending = Some(Pending {
                id: self.next_id,
                started: Instant::now(),
                task: task.clone(),
                path: path.clone(),
                cancelling: false,
                output: matches!(task, Task::Generate(_)).then(GenerationTranscript::default),
                turn: match &task {
                    Task::Generate(options) => {
                        options.conversation.as_ref().map(|chat| PendingTurn {
                            user: chat.prompt.clone(),
                            thinking: !options.flag("--no-thinking"),
                            epoch: self.conversation.epoch,
                            response: String::new(),
                            overflow: false,
                        })
                    }
                    _ => None,
                },
            });
            if let Some(output) = self
                .pending
                .as_ref()
                .and_then(|pending| pending.output.as_ref())
            {
                self.push(output.entry(
                    self.next_id,
                    Kind::Info,
                    "Generating · Esc to stop".into(),
                    None,
                ));
            }
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
        if matches!(self.pending.as_ref().unwrap().task, Task::Generate(_)) {
            // A process-spawn failure arrives through the same path as other submission errors.
            if let Err(error) = event.result {
                self.generation_event(event.id, GenerationEvent::Finished(Outcome::Failed(error)));
            }
            return;
        }
        let pending = self.pending.take().expect("matching pending analysis");
        match event.result {
            Ok(report) => {
                let (path, family) = report.identity();
                self.select_model(path.to_owned());
                if family.is_some() {
                    self.model_family = family.map(|family| family.to_string());
                }
                let elapsed = pending.started.elapsed();
                let entry = match report {
                    Report::Inspection(result) => {
                        self.model_summary = Some(ModelSummary::from_inspection(&result));
                        inspection_entry(&result, elapsed)
                    }
                    Report::Planning(result) => plan_entry(&result, elapsed),
                    Report::Preflight(result) => preflight_entry(&result, elapsed),
                    Report::Listing(result) => list_entry(&result, elapsed),
                    Report::Explanation(result) => explain_entry(&result, elapsed),
                    Report::Probe(result) => probe_entry(&result, elapsed),
                    Report::Tokenization(result) => tokenize_entry(&result, elapsed),
                    Report::Decoding(result) => decode_entry(&result, elapsed),
                };
                self.push(entry);
            }
            Err(error) => self.error(error),
        }
    }

    pub fn generation_event(&mut self, id: u64, event: GenerationEvent) {
        let Some(mut pending) = self.pending.take() else {
            return;
        };
        if pending.id != id {
            self.pending = Some(pending);
            return;
        }
        let Some(output) = &mut pending.output else {
            self.pending = Some(pending);
            return;
        };
        if let GenerationEvent::Progress(snapshot) = event {
            if let Some(runtime) = &mut self.runtime {
                runtime.metrics = snapshot;
            }
            self.pending = Some(pending);
            return;
        }
        let mut history_warning = false;
        let (kind, title, error, finished): (Kind, String, Option<String>, bool) = match event {
            GenerationEvent::Output(stream, text) => {
                if stream == Stream::Text {
                    if let Some(turn) = &mut pending.turn {
                        turn.append(&text);
                    }
                }
                if stream == Stream::Log {
                    if let Some(runtime) = &mut self.runtime {
                        runtime.append(&text);
                    }
                    self.pending = Some(pending);
                    return;
                }
                output.append(stream, &text);
                (
                    Kind::Info,
                    "Generating · Esc to stop".to_owned(),
                    None,
                    false,
                )
            }
            GenerationEvent::Finished(outcome) => {
                if let Some(runtime) = &mut self.runtime {
                    if runtime.metrics.status == crate::progress::Status::Running {
                        runtime.metrics.elapsed_seconds = runtime
                            .metrics
                            .elapsed_seconds
                            .max(pending.started.elapsed().as_secs_f64());
                    }
                    runtime.status = match &outcome {
                        Outcome::Complete => "Complete",
                        Outcome::Cancelled => "Cancelled",
                        Outcome::Failed(_) => "Failed",
                    };
                    if let Outcome::Failed(error) = &outcome {
                        runtime.append(&format!("\n{error}"));
                    }
                }
                let (kind, title, error) = match outcome {
                    Outcome::Complete => {
                        let turn = pending.turn.take().filter(|turn| turn.epoch == self.conversation.epoch);
                        let runtime = self.runtime.take();
                        self.select_model(pending.path.clone());
                        self.runtime = runtime;
                        if let Some(turn) = turn {
                            if turn.overflow { history_warning = true; }
                            else { self.conversation.commit(turn.finish()); }
                        }
                        (Kind::Success, "Generation complete", None)
                    }
                    Outcome::Cancelled => (Kind::Warning, "Generation cancelled", Some(
                        "Partial text kept. Model resources released; profile files may be incomplete.".into()
                    )),
                    Outcome::Failed(_) => (Kind::Error, "Generation failed", Some(
                        "See Runtime (F2) for details.".into()
                    )),
                };
                (
                    kind,
                    format!("{title} · {:.2}s", pending.started.elapsed().as_secs_f64()),
                    error,
                    true,
                )
            }
            GenerationEvent::Progress(_) => unreachable!("progress is handled separately"),
        };
        let entry = output.entry(id, kind, title, error.as_deref());
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|entry| entry.task_id == Some(id))
        {
            *existing = entry;
        } else {
            self.push(entry); // /clear or history eviction removed the previous display block.
        }
        self.trim_transcript();
        if finished {
            self.cancel_requested = false;
            if history_warning {
                self.push(Entry::message(Kind::Warning, "Conversation history",
                    "This response exceeded the history limit and was not added to the next prompt."));
            }
        } else {
            self.pending = Some(pending);
        }
    }

    pub fn error(&mut self, error: impl Into<String>) {
        self.push(Entry::message(Kind::Error, "Error", error));
    }

    pub fn push(&mut self, entry: Entry) {
        self.entries.push_back(entry);
        self.trim_transcript();
    }

    fn trim_transcript(&mut self) {
        let size = |entry: &Entry| {
            entry.title.len()
                + entry
                    .details
                    .iter()
                    .map(|detail| detail.text.len() + detail.label.map_or(0, str::len))
                    .sum::<usize>()
        };
        let mut bytes: usize = self.entries.iter().map(size).sum();
        while self.entries.len() > HISTORY_LIMIT || bytes > TRANSCRIPT_BYTES {
            bytes -= size(self.entries.front().expect("nonempty transcript"));
            self.entries.pop_front();
            // Recompute from the oldest retained entry after eviction.
            self.scroll = self.scroll.map(|_| 0);
        }
    }
}

fn editor(text: &str) -> TextArea<'static> {
    let mut editor = TextArea::from(text.split('\n'));
    editor.set_wrap_mode(WrapMode::Glyph);
    editor.set_placeholder_text("Message the model, or / for commands");
    editor.move_cursor(CursorMove::Bottom);
    editor.move_cursor(CursorMove::End);
    editor
}

#[cfg(test)]
mod tests {
    use super::super::generate::Stream;
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn message(app: &mut App, text: &str) -> Request {
        app.paste(text);
        app.submit().expect("message starts generation")
    }

    fn finish_reply(app: &mut App, id: u64, text: &str, outcome: Outcome) {
        app.generation_event(id, GenerationEvent::Output(Stream::Text, text.into()));
        app.generation_event(
            id,
            GenerationEvent::Output(Stream::Log, "diagnostics".into()),
        );
        app.generation_event(id, GenerationEvent::Finished(outcome));
    }

    fn chat_app() -> App {
        let mut app = App::new(Some("/model".into()));
        app.paste("/settings --ram-gib 2 --max-new-tokens 128 --no-thinking");
        app.submit();
        app.model_summary = Some(ModelSummary {
            name: "model".into(),
            family: "GLM-5.2".into(),
            path: "/model".into(),
            fields: vec![],
        });
        app
    }

    #[test]
    fn runtime_metrics_ignore_stale_events_and_reset_with_each_generation() {
        let mut app = chat_app();
        let request = message(&mut app, "hello");
        let snapshot = crate::progress::Snapshot {
            prompt_tokens: Some(6),
            generated_tokens: 3,
            total_tokens: Some(9),
            elapsed_seconds: 4.0,
            ttft_seconds: Some(2.0),
            decode_tokens_per_second: Some(1.0),
            ..Default::default()
        };
        app.generation_event(request.id + 1, GenerationEvent::Progress(snapshot.clone()));
        assert_eq!(app.runtime.as_ref().unwrap().metrics.generated_tokens, 0);
        app.generation_event(request.id, GenerationEvent::Progress(snapshot.clone()));
        assert_eq!(app.runtime.as_ref().unwrap().metrics, snapshot);
        app.key(key(KeyCode::F(2)));
        assert!(app.runtime_open);
        app.key(key(KeyCode::Esc));
        assert!(!app.runtime_open);
        assert!(!app.cancel_requested);
        finish_reply(&mut app, request.id, "reply\n", Outcome::Cancelled);
        assert_eq!(app.runtime.as_ref().unwrap().status, "Cancelled");
        assert_eq!(app.runtime.as_ref().unwrap().metrics.generated_tokens, 3);
        assert!(app.runtime.as_ref().unwrap().text().contains("diagnostics"));
        let next = message(&mut app, "retry");
        assert_eq!(app.runtime.as_ref().unwrap().metrics.generated_tokens, 0);
        assert_eq!(app.runtime.as_ref().unwrap().metrics.ttft_seconds, None);
        assert_eq!(
            app.runtime.as_ref().unwrap().text(),
            "Waiting for runtime diagnostics…"
        );
        app.generation_event(request.id, GenerationEvent::Progress(snapshot));
        assert_eq!(app.pending.as_ref().unwrap().id, next.id);
        assert_eq!(app.runtime.as_ref().unwrap().metrics.generated_tokens, 0);
    }

    #[test]
    fn plain_input_reuses_complete_responses_without_display_truncation_or_logs() {
        let mut app = chat_app();
        let first = message(&mut app, "  don't quote \"this\n中文🙂  ");
        let Task::Generate(options) = &first.task else {
            panic!("expected generation")
        };
        assert!(options.value("--prompt").is_none());
        assert_eq!(
            options.conversation.as_ref().unwrap().prompt,
            "  don't quote \"this\n中文🙂  "
        );
        assert!(options.conversation.as_ref().unwrap().turns.is_empty());
        let response = format!("START{}END\n", "中".repeat(20_000));
        finish_reply(&mut app, first.id, &response, Outcome::Complete);
        assert!(!app.entries.back().unwrap().details[0]
            .text
            .contains("START"));
        assert_eq!(
            app.conversation.turns[0].assistant,
            response.strip_suffix('\n').unwrap()
        );
        let second = message(&mut app, "What was my question?");
        let Task::Generate(options) = &second.task else {
            panic!("expected generation")
        };
        let chat = options.conversation.as_ref().unwrap();
        assert_eq!(chat.turns.len(), 1);
        assert!(chat.turns[0].assistant.starts_with("START"));
        assert!(!chat.turns[0].assistant.contains("diagnostics"));
        assert_eq!(options.value("--max-new-tokens"), Some("128"));
        finish_reply(&mut app, second.id, "unfinished", Outcome::Cancelled);
        assert_eq!(app.conversation.turns.len(), 1);
        let third = message(&mut app, "Retry");
        finish_reply(
            &mut app,
            third.id,
            "partial",
            Outcome::Failed("runtime failure".into()),
        );
        assert_eq!(app.conversation.turns.len(), 1);
        assert_eq!(app.model_summary.as_ref().unwrap().name, "model");
    }

    #[test]
    fn clear_during_generation_starts_a_new_context_and_keeps_the_model_card() {
        let mut app = chat_app();
        let first = message(&mut app, "first");
        finish_reply(&mut app, first.id, "answer\n", Outcome::Complete);
        let second = message(&mut app, "second");
        app.paste("/clear");
        app.submit();
        finish_reply(
            &mut app,
            second.id,
            "answer after clear\n",
            Outcome::Complete,
        );
        assert!(app.conversation.turns.is_empty());
        assert!(app.model_summary.is_some());
        let third = message(&mut app, "new topic");
        let Task::Generate(options) = third.task else {
            panic!("expected generation")
        };
        assert!(options.conversation.unwrap().turns.is_empty());
    }

    #[test]
    fn model_changes_commit_only_on_success_and_raw_prompts_do_not_join_history() {
        let mut app = chat_app();
        let first = message(&mut app, "first");
        finish_reply(&mut app, first.id, "answer\n", Outcome::Complete);
        let failed = message(
            &mut app,
            "/generate next --model /other --ram-gib 2 --allow-large-model",
        );
        let Task::Generate(options) = failed.task else {
            panic!("expected generation")
        };
        assert!(options.conversation.unwrap().turns.is_empty());
        finish_reply(
            &mut app,
            failed.id,
            "",
            Outcome::Failed("missing model".into()),
        );
        assert_eq!(app.conversation.turns.len(), 1);
        assert!(app.model_summary.is_some());
        let raw = message(
            &mut app,
            "/generate raw --ram-gib 2 --allow-large-model --raw-prompt",
        );
        let Task::Generate(options) = &raw.task else {
            panic!("expected generation")
        };
        assert!(options.conversation.is_none());
        finish_reply(&mut app, raw.id, "raw output\n", Outcome::Complete);
        assert_eq!(app.conversation.turns.len(), 1);
        let switched = message(
            &mut app,
            "/generate next --model /other --ram-gib 2 --allow-large-model --no-thinking",
        );
        finish_reply(&mut app, switched.id, "new answer\n", Outcome::Complete);
        assert_eq!(app.model_path, Some("/other".into()));
        assert!(app.model_summary.is_none());
        assert_eq!(app.conversation.turns.len(), 1);
        assert_eq!(app.conversation.turns[0].user, "next");
    }

    #[test]
    fn overflowing_reply_is_not_added_to_history_and_ordinary_text_needs_a_model() {
        let mut app = App::new(None);
        app.paste("hello");
        assert!(app.submit().is_none());
        assert!(app.entries.back().unwrap().details[0]
            .text
            .contains("No model selected"));
        let mut app = chat_app();
        let request = message(&mut app, "hello");
        finish_reply(
            &mut app,
            request.id,
            &"x".repeat(super::super::chat::HISTORY_BYTES),
            Outcome::Complete,
        );
        assert!(app.conversation.turns.is_empty());
        assert!(app.entries.back().unwrap().details[0]
            .text
            .contains("not added"));
    }

    fn generate(app: &mut App, options: &str) -> Request {
        app.paste(&format!(
            "/generate hello --ram-gib 2 --allow-large-model {options}"
        ));
        app.submit().unwrap()
    }

    #[test]
    fn generation_updates_one_block_and_keeps_partial_output_on_cancel() {
        let mut app = App::new(Some("/original".into()));
        let request = generate(&mut app, "--model /new");
        app.generation_event(
            request.id + 1,
            GenerationEvent::Output(Stream::Text, "stale".into()),
        );
        app.generation_event(
            request.id,
            GenerationEvent::Output(Stream::Text, "中文".into()),
        );
        app.paste("/version");
        app.submit();
        app.generation_event(
            request.id,
            GenerationEvent::Output(Stream::Text, "🙂".into()),
        );
        app.generation_event(
            request.id,
            GenerationEvent::Output(Stream::Log, "preflight: ready".into()),
        );
        assert_eq!(
            app.entries
                .iter()
                .filter(|entry| entry.task_id == Some(request.id))
                .count(),
            1
        );
        let output = app
            .entries
            .iter()
            .find(|entry| entry.task_id == Some(request.id))
            .unwrap();
        assert_eq!(output.details[0].text, "中文🙂");
        assert_eq!(output.details.len(), 1);
        assert_eq!(app.runtime.as_ref().unwrap().text(), "preflight: ready");
        app.key(key(KeyCode::Esc));
        assert!(app.cancel_requested);
        assert!(!app.quit);
        app.generation_event(request.id, GenerationEvent::Finished(Outcome::Cancelled));
        assert!(app.pending.is_none());
        assert_eq!(app.model_path, Some("/original".into()));
        let output = app
            .entries
            .iter()
            .find(|entry| entry.task_id == Some(request.id))
            .unwrap();
        assert_eq!(output.details[0].text, "中文🙂");
        assert!(output.title.starts_with("Generation cancelled"));
        let next = generate(&mut app, "--model /new");
        app.generation_event(request.id, GenerationEvent::Finished(Outcome::Complete));
        assert_eq!(app.pending.as_ref().unwrap().id, next.id);
        app.generation_event(next.id, GenerationEvent::Finished(Outcome::Complete));
        assert_eq!(app.model_path, Some("/new".into()));
    }

    #[test]
    fn generation_handles_clear_spawn_failure_and_completion_menu_escape() {
        let mut app = App::new(Some("/model".into()));
        let request = generate(&mut app, "");
        app.generation_event(
            request.id,
            GenerationEvent::Output(Stream::Text, "old".into()),
        );
        app.paste("/clear");
        app.submit();
        assert!(app.entries.is_empty());
        app.generation_event(
            request.id,
            GenerationEvent::Output(Stream::Text, "new".into()),
        );
        assert_eq!(app.entries.back().unwrap().details[0].text, "new");
        app.paste("/");
        app.key(key(KeyCode::Esc));
        assert!(!app.cancel_requested); // First Escape dismisses the command menu.
        app.key(key(KeyCode::Esc));
        assert!(app.cancel_requested);
        app.generation_event(
            request.id,
            GenerationEvent::Finished(Outcome::Failed("runtime error".into())),
        );
        assert!(app.pending.is_none());
        assert!(app
            .entries
            .back()
            .unwrap()
            .title
            .starts_with("Generation failed"));
        app.key(key(KeyCode::Backspace));
        let request = generate(&mut app, "");
        app.finished(Finished {
            id: request.id,
            result: Err("spawn failed".into()),
        });
        assert!(app.pending.is_none());
        assert!(app
            .runtime
            .as_ref()
            .unwrap()
            .text()
            .contains("spawn failed"));
        assert!(app
            .entries
            .back()
            .unwrap()
            .details
            .iter()
            .any(|detail| detail.text.contains("F2")));
    }

    #[test]
    fn long_generation_history_remains_scrollable_at_the_minimum_width() {
        let mut app = App::new(Some("/model".into()));
        for _ in 0..25 {
            let request = generate(&mut app, "");
            app.generation_event(
                request.id,
                GenerationEvent::Output(
                    Stream::Text,
                    format!("{}latest text", "long line ".repeat(4000)),
                ),
            );
            app.generation_event(request.id, GenerationEvent::Finished(Outcome::Complete));
        }
        let bytes: usize = app
            .entries
            .iter()
            .flat_map(|entry| &entry.details)
            .map(|detail| detail.text.len())
            .sum();
        assert!(bytes <= TRANSCRIPT_BYTES);
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(36, 10)).unwrap();
        terminal
            .draw(|frame| super::super::view::draw(frame, &mut app))
            .unwrap();
        assert!(app.max_scroll < usize::from(u16::MAX));
        let display: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(display.contains("latest text"));
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
