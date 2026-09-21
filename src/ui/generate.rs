//! A managed instance of the existing CLI keeps model I/O and CPU work off the UI thread.
//! Only pipe readers use threads; the UI owns, cancels, and reaps the child process.

use super::commands::GenerateOptions;
use crate::progress::{Snapshot, JSON_PREFIX};
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Text,
    Log,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Complete,
    Cancelled,
    Failed(String),
}

pub enum Event {
    Output(Stream, String),
    Progress(Snapshot),
    Finished(Outcome),
}

pub struct Generation {
    pub id: u64,
    child: Child,
    output: Receiver<io::Result<Event>>,
    cancelled: bool,
}

impl Generation {
    pub fn start(id: u64, path: &Path, options: &GenerateOptions) -> io::Result<Self> {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("generate")
            .arg(path)
            .args(&options.args)
            .arg("--progress-json");
        let input = options
            .conversation
            .as_ref()
            .map(serde_json::to_vec)
            .transpose()
            .map_err(io::Error::other)?;
        if input.is_some() {
            command.arg("--chat-stdin");
        }
        Self::spawn_with_input(id, &mut command, input)
    }

    #[cfg(test)]
    fn spawn(id: u64, command: &mut Command) -> io::Result<Self> {
        Self::spawn_with_input(id, command, None)
    }

    fn spawn_with_input(
        id: u64,
        command: &mut Command,
        input: Option<Vec<u8>>,
    ) -> io::Result<Self> {
        let child = command
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        // Bound queued bytes even if the renderer is slower than generation/profiling output.
        let (send, output) = mpsc::sync_channel(32);
        let mut generation = Self {
            id,
            child,
            output,
            cancelled: false,
        };
        let stdout = generation.child.stdout.take().expect("piped stdout");
        let stderr = generation.child.stderr.take().expect("piped stderr");
        if let Some(input) = input {
            let mut stdin = generation.child.stdin.take().expect("piped stdin");
            thread::Builder::new()
                .name("urb-chat-input".into())
                .spawn(move || {
                    // Bounded JSON travels over stdin, avoiding argv size limits and shell parsing.
                    // The child reports malformed/partial input through its normal error stream.
                    let _ = stdin.write_all(&input);
                })?;
        }
        let text_send = send.clone();
        thread::Builder::new()
            .name("urb-generate-text".into())
            .spawn(move || read_pipe(stdout, Stream::Text, text_send))?;
        thread::Builder::new()
            .name("urb-generate-log".into())
            .spawn(move || read_pipe(stderr, Stream::Log, send))?;
        // On reader setup failure, Drop also terminates and reaps the already-started child.
        Ok(generation)
    }

    pub fn cancel(&mut self) -> io::Result<()> {
        if !self.cancelled && self.child.try_wait()?.is_none() {
            self.child.kill()?;
            self.cancelled = true;
        }
        Ok(())
    }

    pub fn poll(&mut self) -> io::Result<Option<Event>> {
        match self.output.try_recv() {
            Ok(Ok(event)) => Ok(Some(event)),
            Ok(Err(error)) => Err(error),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                // Drain both streams, including final diagnostics, before announcing completion.
                Ok(self.child.try_wait()?.map(|status| {
                    Event::Finished(if self.cancelled {
                        Outcome::Cancelled
                    } else if status.success() {
                        Outcome::Complete
                    } else {
                        Outcome::Failed(format!(
                            "generate exited with {status}; see runtime output."
                        ))
                    })
                }))
            }
        }
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        // Covers /quit, Ctrl+C/D, handled termination signals, UI errors and panic unwinding.
        // No join of reader threads: dropping the receiver releases any blocked senders.
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn read_pipe(mut pipe: impl Read, stream: Stream, send: SyncSender<io::Result<Event>>) {
    let mut bytes = [0; 4096];
    let mut decoder = Utf8Stream::default();
    let mut logs = LogLines::default();
    loop {
        let (text, eof) = match pipe.read(&mut bytes) {
            Ok(0) => (String::from_utf8_lossy(&decoder.pending).into_owned(), true),
            Ok(count) => (decoder.push(&bytes[..count]), false),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                let _ = send.send(Err(error));
                return;
            }
        };
        let events = if stream == Stream::Log {
            logs.push(&text, eof)
        } else if !text.is_empty() {
            vec![Event::Output(stream, text)]
        } else {
            vec![]
        };
        for event in events {
            if send.send(Ok(event)).is_err() {
                return;
            }
        }
        if eof {
            return;
        }
    }
}

/// Stderr contains both ordinary diagnostics and newline-delimited metric records. Never
/// expose protocol records as chat text, and never buffer an unbounded unterminated log line.
#[derive(Default)]
struct LogLines {
    pending: String,
    continued: bool,
}

impl LogLines {
    fn push(&mut self, text: &str, eof: bool) -> Vec<Event> {
        const MAX_LINE_BYTES: usize = 8192;
        self.pending.push_str(text);
        let mut events = Vec::new();
        let mut consumed = 0;
        for line in self.pending.split_inclusive('\n') {
            if !line.ends_with('\n') && !eof {
                break;
            }
            let snapshot = (!self.continued)
                .then(|| {
                    line.strip_prefix(JSON_PREFIX)
                        .and_then(|json| serde_json::from_str::<Snapshot>(json).ok())
                        .filter(Snapshot::is_valid)
                })
                .flatten();
            events.push(match snapshot {
                Some(snapshot) => Event::Progress(snapshot),
                None => Event::Output(Stream::Log, line.to_owned()),
            });
            consumed += line.len();
            self.continued = false;
        }
        self.pending.drain(..consumed);
        if self.pending.len() > MAX_LINE_BYTES {
            events.push(Event::Output(
                Stream::Log,
                std::mem::take(&mut self.pending),
            ));
            self.continued = true;
        }
        events
    }
}

#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut text = String::new();
        let mut consumed = 0;
        while consumed < self.pending.len() {
            match std::str::from_utf8(&self.pending[consumed..]) {
                Ok(valid) => {
                    text.push_str(valid);
                    consumed = self.pending.len();
                }
                Err(error) => {
                    let valid_end = consumed + error.valid_up_to();
                    text.push_str(std::str::from_utf8(&self.pending[consumed..valid_end]).unwrap());
                    consumed = valid_end;
                    if let Some(length) = error.error_len() {
                        text.push('\u{fffd}');
                        consumed += length;
                    } else {
                        break; // Keep an incomplete code point until the next pipe read.
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn split_stderr_records_keep_metrics_out_of_logs_and_bound_partial_lines() {
        let snapshot = Snapshot {
            prompt_tokens: Some(6),
            generated_tokens: 3,
            total_tokens: Some(9),
            elapsed_seconds: 4.0,
            ttft_seconds: Some(2.0),
            decode_tokens_per_second: Some(1.0),
            ..Snapshot::default()
        };
        let wire = format!(
            "preflight: 中文\n{JSON_PREFIX}{}\ndone\n",
            serde_json::to_string(&snapshot).unwrap()
        );
        let mut utf8 = Utf8Stream::default();
        let mut lines = LogLines::default();
        let mut logs = String::new();
        let mut snapshots = Vec::new();
        for byte in wire.as_bytes() {
            for event in lines.push(&utf8.push(&[*byte]), false) {
                match event {
                    Event::Output(Stream::Log, text) => logs.push_str(&text),
                    Event::Progress(snapshot) => snapshots.push(snapshot),
                    _ => panic!("unexpected stderr event"),
                }
            }
        }
        assert_eq!(logs, "preflight: 中文\ndone\n");
        assert_eq!(snapshots, vec![snapshot.clone()]);
        let mut invalid = snapshot;
        invalid.total_tokens = Some(100);
        for record in [
            format!("{JSON_PREFIX}{{broken}}\n"),
            format!(
                "{JSON_PREFIX}{}\n",
                serde_json::to_string(&invalid).unwrap()
            ),
        ] {
            assert!(
                matches!(&lines.push(&record, false)[0], Event::Output(Stream::Log, text) if text == &record)
            );
        }
        assert_eq!(lines.push(&"x".repeat(9000), false).len(), 1);
        assert!(lines.pending.is_empty());
        assert!(
            matches!(&lines.push("tail", true)[0], Event::Output(Stream::Log, text) if text == "tail")
        );
    }

    #[test]
    fn pipe_reads_preserve_split_utf8_and_replace_invalid_bytes() {
        let mut decoder = Utf8Stream::default();
        let mut text = String::new();
        for byte in "中文🙂".as_bytes() {
            text.push_str(&decoder.push(&[*byte]));
            assert!(decoder.pending.len() <= 3);
        }
        assert_eq!(text, "中文🙂");
        assert_eq!(decoder.push(b"a\xffb"), "a�b");
        assert_eq!(decoder.push(&[0xe4, 0xb8]), "");
        assert_eq!(decoder.push(b"!"), "�!");
    }

    #[cfg(unix)]
    fn collect(generation: &mut Generation) -> (String, String, Outcome) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let (mut text, mut log) = (String::new(), String::new());
        loop {
            assert!(Instant::now() < deadline, "child did not finish");
            match generation.poll().unwrap() {
                Some(Event::Output(Stream::Text, chunk)) => text.push_str(&chunk),
                Some(Event::Output(Stream::Log, chunk)) => log.push_str(&chunk),
                Some(Event::Progress(_)) => {}
                Some(Event::Finished(outcome)) => return (text, log, outcome),
                None => thread::sleep(Duration::from_millis(5)),
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn large_stdin_is_written_without_blocking_the_ui_or_deadlocking_stdout() {
        let input = "中文🙂".repeat(32_768).into_bytes();
        let mut generation =
            Generation::spawn_with_input(1, &mut Command::new("/bin/cat"), Some(input.clone()))
                .unwrap();
        let (text, log, outcome) = collect(&mut generation);
        assert_eq!(text.as_bytes(), input);
        assert!(log.is_empty());
        assert_eq!(outcome, Outcome::Complete);

        let mut command = Command::new("/bin/sh");
        command.args(["-c", "printf ready; exec sleep 60"]);
        let mut generation =
            Generation::spawn_with_input(2, &mut command, Some(vec![b'a'; 1024 * 1024])).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            assert!(Instant::now() < deadline, "stdin writer blocked UI startup");
            if let Some(Event::Output(Stream::Text, text)) = generation.poll().unwrap() {
                assert_eq!(text, "ready");
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        generation.cancel().unwrap();
        assert_eq!(collect(&mut generation).2, Outcome::Cancelled);
    }

    #[cfg(unix)]
    #[test]
    fn completion_waits_for_both_pipes_and_reports_nonzero_exit() {
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            "printf '中文🙂'; printf 'runtime failure' >&2; exit 2",
        ]);
        let mut generation = Generation::spawn(1, &mut command).unwrap();
        let (text, log, outcome) = collect(&mut generation);
        assert_eq!(text, "中文🙂");
        assert_eq!(log, "runtime failure");
        assert!(matches!(outcome, Outcome::Failed(_)));
    }

    #[cfg(unix)]
    #[test]
    fn streams_before_exit_and_cancels_a_busy_child() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "printf 'first chunk'; exec sleep 60"]);
        let mut generation = Generation::spawn(1, &mut command).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            assert!(Instant::now() < deadline, "output was not streamed");
            if let Some(Event::Output(Stream::Text, text)) = generation.poll().unwrap() {
                assert_eq!(text, "first chunk");
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(generation.child.try_wait().unwrap().is_none());
        generation.cancel().unwrap();
        assert_eq!(collect(&mut generation).2, Outcome::Cancelled);
        assert!(generation.child.try_wait().unwrap().is_some());
    }
}
