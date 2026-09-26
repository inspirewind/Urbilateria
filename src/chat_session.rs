//! Persistent CLI conversation and bounded JSON-lines transport used by the TUI.

use crate::chat::{Conversation, Input, Turn, MAX_CHAT_BYTES, MAX_CHAT_TURNS, MAX_WIRE_BYTES};
use crate::{run_generate_cached, session, GenerateArguments};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::Path;

pub const END_MARKER: &str = "URB_SESSION_END";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub args: Vec<String>,
    pub conversation: Conversation,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Record {
    Text { text: String },
    Finished { error: Option<String> },
}

pub fn read_line(reader: &mut impl BufRead, limit: usize) -> io::Result<Option<Vec<u8>>> {
    let mut bytes = Vec::new();
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Ok((!bytes.is_empty()).then_some(bytes));
        }
        let count = buffer
            .iter()
            .position(|&b| b == b'\n')
            .map_or(buffer.len(), |p| p + 1);
        if bytes.len().saturating_add(count) > limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "chat input exceeds the input limit",
            ));
        }
        let ended = buffer[count - 1] == b'\n';
        bytes.extend_from_slice(&buffer[..count]);
        reader.consume(count);
        if ended {
            return Ok(Some(bytes));
        }
    }
}

fn record(output: &mut impl Write, value: &Record) -> io::Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    output.flush()
}

struct JsonOutput<'a, W>(&'a mut W);
impl<W: Write> Write for JsonOutput<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let text = std::str::from_utf8(bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        record(
            self.0,
            &Record::Text {
                text: text.to_owned(),
            },
        )?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

pub fn run(model: &Path, args: &[String]) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let interactive = stdin.is_terminal();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    if args == ["--session-json"] {
        return serve(model, &mut input, &mut output);
    }
    let parsed = GenerateArguments::parse(args)?;
    if parsed.value("--prompt").is_some()
        || parsed.flag("--chat-stdin")
        || parsed.flag("--raw-prompt")
    {
        return Err("chat reads messages from stdin; --prompt, --chat-stdin and --raw-prompt are not supported".into());
    }
    if !parsed.flag("--allow-large-model") || parsed.value("--ram-gib").is_none() {
        return Err("chat requires --ram-gib N --allow-large-model".into());
    }
    let thinking = !parsed.flag("--no-thinking");
    let mut args = args.to_vec();
    if parsed.value("--max-new-tokens").is_none() {
        args.extend(["--max-new-tokens".into(), "512".into()]);
    }
    let mut cache = session::Cache::resident();
    let mut turns = Vec::new();
    loop {
        if interactive {
            eprint!("you> ");
            io::stderr().flush()?;
        }
        let Some(line) = read_line(&mut input, MAX_CHAT_BYTES)? else {
            break;
        };
        let prompt = String::from_utf8(line)?
            .trim_end_matches(['\r', '\n'])
            .to_owned();
        if prompt == "/quit" {
            break;
        }
        if prompt == "/clear" {
            cache.clear();
            turns.clear();
            continue;
        }
        if prompt.trim().is_empty() {
            continue;
        }
        let conversation = Conversation {
            turns: turns.clone(),
            prompt: prompt.clone(),
        };
        conversation.validate()?;
        let mut captured = Capture {
            output: &mut output,
            text: String::new(),
            overflow: false,
        };
        let result = run_generate_cached(
            model,
            &args,
            None,
            &mut captured,
            &mut cache,
            Some(Input::Chat(conversation)),
        );
        match result {
            Ok(()) if !captured.overflow => {
                captured.text.pop(); // CLI's final newline.
                turns.push(Turn {
                    user: prompt,
                    assistant: captured.text,
                    thinking,
                });
                while turns.len() > MAX_CHAT_TURNS
                    || turns
                        .iter()
                        .map(|t| t.user.len() + t.assistant.len())
                        .sum::<usize>()
                        > MAX_CHAT_BYTES / 2
                {
                    turns.remove(0);
                }
            }
            Ok(()) => {
                cache.clear();
                eprintln!("chat: reply exceeds history limit; excluded from conversation");
            }
            Err(error) => {
                cache.clear();
                eprintln!("error: {error}");
            }
        }
    }
    Ok(())
}

struct Capture<'a, W> {
    output: &'a mut W,
    text: String,
    overflow: bool,
}
impl<W: Write> Write for Capture<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.output.write_all(bytes)?;
        if self.text.len() + bytes.len() <= MAX_CHAT_BYTES / 2 && !self.overflow {
            self.text
                .push_str(std::str::from_utf8(bytes).map_err(io::Error::other)?);
        } else {
            self.text.clear();
            self.overflow = true;
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

fn serve(
    model: &Path,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    let mut cache = session::Cache::resident();
    while let Some(line) = read_line(input, MAX_WIRE_BYTES)? {
        let result = (|| {
            let request: Request = serde_json::from_slice(&line)?;
            request.conversation.validate()?;
            let parsed = GenerateArguments::parse(&request.args)?;
            if parsed.flag("--raw-prompt")
                || parsed.flag("--chat-stdin")
                || parsed.value("--prompt").is_some()
            {
                return Err("session requests require structured conversation input".into());
            }
            run_generate_cached(
                model,
                &request.args,
                None,
                &mut JsonOutput(output),
                &mut cache,
                Some(Input::Chat(request.conversation)),
            )
        })();
        let error = result.err().map(|error: Box<dyn Error>| error.to_string());
        if let Some(error) = &error {
            cache.clear();
            eprintln!("error: {error}");
        }
        // A boundary on BOTH pipes prevents completion from racing with buffered text/metrics.
        eprintln!("{END_MARKER}");
        record(output, &Record::Finished { error })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn line_protocol_bounds_partial_input_and_preserves_multibyte_text() {
        let mut input = std::io::Cursor::new("中文\nnext".as_bytes());
        assert_eq!(
            read_line(&mut input, 7).unwrap().unwrap(),
            "中文\n".as_bytes()
        );
        assert_eq!(read_line(&mut input, 7).unwrap().unwrap(), b"next");
        assert!(read_line(&mut input, 7).unwrap().is_none());
        assert!(read_line(&mut std::io::Cursor::new(b"12345678"), 7).is_err());
    }
}
