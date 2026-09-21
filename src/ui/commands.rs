use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::PathBuf;
use urbilateria::analysis::{
    parse_token_ids, ListOptions, PlanOptions, PreflightOptions, TokenizeOptions,
};

pub const COMMANDS: [(&str, &str); 14] = [
    ("/help", "Commands and keyboard shortcuts"),
    ("/inspect", "Inspect checkpoint metadata"),
    ("/plan", "Estimate memory and expert-cache budgets"),
    (
        "/preflight",
        "Validate checkpoint schema and runtime requirements",
    ),
    ("/list", "Find tensors by name"),
    ("/explain", "Explain the model architecture and token path"),
    ("/probe", "Sample a tensor and inspect numeric statistics"),
    ("/tokenize", "Encode text or a model-native chat prompt"),
    ("/decode", "Decode comma-separated token IDs"),
    ("/generate", "Stream a response with an explicit RAM budget"),
    (
        "/settings",
        "Show or change conversation generation settings",
    ),
    ("/version", "Show the program version"),
    ("/clear", "Clear the transcript and conversation context"),
    ("/quit", "Return to your shell"),
];

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Message(String),
    Help,
    Inspect(Option<PathBuf>),
    Plan(Option<PathBuf>, PlanOptions),
    Preflight(Option<PathBuf>, PreflightOptions),
    List(Option<PathBuf>, ListOptions),
    Explain(Option<PathBuf>),
    Probe(Option<PathBuf>, String, usize),
    Tokenize(Option<PathBuf>, String, TokenizeOptions),
    Decode(Option<PathBuf>, Vec<u32>, bool),
    Generate(Option<PathBuf>, GenerateOptions),
    Settings(SettingsUpdate),
    Version,
    Clear,
    Quit,
}

/// Validated CLI arguments, passed directly to the current executable without a shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateOptions {
    pub args: Vec<OsString>,
    pub conversation: Option<crate::chat::Conversation>,
}

impl GenerateOptions {
    fn index(&self, name: &str) -> Option<usize> {
        let mut index = 0;
        while index < self.args.len() {
            if self.args[index] == name {
                return Some(index);
            }
            index += if matches!(
                self.args[index].to_str(),
                Some("--allow-large-model" | "--raw-prompt" | "--no-thinking" | "--profile")
            ) {
                1
            } else {
                2
            };
        }
        None
    }

    pub fn flag(&self, name: &str) -> bool {
        self.index(name).is_some()
    }

    pub fn value(&self, name: &str) -> Option<&str> {
        self.args.get(self.index(name)? + 1)?.to_str()
    }

    pub fn take_prompt(&mut self) -> Option<String> {
        let index = self.index("--prompt")?;
        let text = self.args[index + 1].to_str()?.to_owned();
        self.args.drain(index..index + 2);
        Some(text)
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SettingsUpdate {
    pub ram_gib: Option<Option<String>>,
    pub threads: Option<Option<usize>>,
    pub max_new_tokens: Option<usize>,
    pub thinking: Option<bool>,
}

pub fn takes_arguments(name: &str) -> bool {
    !matches!(name, "/help" | "/version" | "/clear" | "/quit")
}

pub fn parse(input: &str) -> Result<Command, String> {
    if !input.trim_start().starts_with('/') {
        return Ok(Command::Message(input.to_owned()));
    }
    let words = shlex::split(input)
        .ok_or("Unclosed quote. Put paths and text with spaces inside quotes.")?;
    let (name, words) = words
        .split_first()
        .ok_or("Enter a slash command. Use /help to see available commands.")?;
    match name.as_str() {
        "/help" | "/version" | "/clear" | "/quit" => {
            if !words.is_empty() {
                return Err(format!("{name} does not take arguments."));
            }
            Ok(match name.as_str() {
                "/help" => Command::Help,
                "/version" => Command::Version,
                "/clear" => Command::Clear,
                _ => Command::Quit,
            })
        }
        "/inspect" | "/explain" => {
            // These commands take only a directory; no option parsing is needed.
            let path = match words {
                [] => None,
                [path] => Some(model_path(path)?),
                _ => {
                    return Err(format!(
                        "Usage: {name} [MODEL_DIR]. Quote paths containing spaces."
                    ))
                }
            };
            Ok(if name == "/inspect" {
                Command::Inspect(path)
            } else {
                Command::Explain(path)
            })
        }
        "/plan" => {
            let args = parse_options(words, &["--ram-gib", "--context", "--kv-bytes"], &[])?;
            let defaults = PlanOptions::default();
            let options = PlanOptions {
                ram_bytes: args
                    .number::<f64>("--ram-gib")?
                    .map(crate::gib_to_bytes)
                    .transpose()
                    .map_err(|error| error.to_string())?,
                context: args.number("--context")?.unwrap_or(defaults.context),
                kv_bytes: args.number("--kv-bytes")?.unwrap_or(defaults.kv_bytes),
            };
            options.validate()?;
            Ok(Command::Plan(
                args.positional.as_deref().map(model_path).transpose()?,
                options,
            ))
        }
        "/preflight" => {
            let args = parse_options(words, &["--context", "--expert-slots"], &["--partial"])?;
            let defaults = PreflightOptions::default();
            let options = PreflightOptions {
                context: args.number("--context")?.unwrap_or(defaults.context),
                expert_slots: args
                    .number("--expert-slots")?
                    .unwrap_or(defaults.expert_slots),
                partial: args.flags.contains("--partial"),
            };
            options.validate()?;
            Ok(Command::Preflight(
                args.positional.as_deref().map(model_path).transpose()?,
                options,
            ))
        }
        "/list" => {
            let args = parse_options(words, &["--model", "--limit"], &[])?;
            let path = args.model()?;
            let options = ListOptions {
                limit: args.number("--limit")?.unwrap_or(100),
                filter: args.positional,
            };
            options.validate()?;
            Ok(Command::List(path, options))
        }
        "/probe" => {
            let args = parse_options(words, &["--model", "--samples"], &[])?;
            let path = args.model()?;
            let samples = args.number("--samples")?.unwrap_or(8192);
            if !(1..=10_000_000).contains(&samples) {
                return Err("--samples must be in 1..=10000000".into());
            }
            let tensor = args
                .positional
                .filter(|text| !text.is_empty())
                .ok_or("Usage: /probe TENSOR_NAME [--model MODEL_DIR] [--samples N]")?;
            Ok(Command::Probe(path, tensor, samples))
        }
        "/tokenize" => {
            let args = parse_options(words, &["--model"], &["--chat", "--no-thinking"])?;
            let path = args.model()?;
            let options = TokenizeOptions {
                chat: args.flags.contains("--chat"),
                no_thinking: args.flags.contains("--no-thinking"),
            };
            options.validate()?;
            let text = args
                .positional
                .ok_or("Usage: /tokenize \"TEXT\" [--model MODEL_DIR] [--chat] [--no-thinking]")?;
            Ok(Command::Tokenize(path, text, options))
        }
        "/decode" => {
            let args = parse_options(words, &["--model"], &["--skip-special"])?;
            let path = args.model()?;
            let ids = parse_token_ids(
                args.positional
                    .as_deref()
                    .ok_or("Usage: /decode TOKEN_IDS [--model MODEL_DIR] [--skip-special]")?,
            )?;
            Ok(Command::Decode(
                path,
                ids,
                args.flags.contains("--skip-special"),
            ))
        }
        "/generate" => parse_generate(words),
        "/settings" => {
            let args = parse_options(
                words,
                &["--ram-gib", "--max-new-tokens", "--threads"],
                &["--thinking", "--no-thinking"],
            )?;
            if args.positional.is_some() {
                return Err("/settings takes options only.".into());
            }
            let ram_gib = args
                .values
                .get("--ram-gib")
                .map(|value| {
                    if value == "auto" {
                        return Ok(None);
                    }
                    let ram = value
                        .parse::<f64>()
                        .map_err(|_| "Invalid --ram-gib value.")?;
                    if crate::gib_to_bytes(ram).map_err(|error| error.to_string())? == 0 {
                        return Err("--ram-gib must provide at least one byte.".to_owned());
                    }
                    Ok(Some(value.clone()))
                })
                .transpose()?;
            let threads = args
                .values
                .get("--threads")
                .map(|value| {
                    if value == "auto" {
                        return Ok(None);
                    }
                    let count = value
                        .parse::<usize>()
                        .map_err(|_| "Invalid --threads value.")?;
                    if count == 0 {
                        return Err("--threads must be greater than zero.");
                    }
                    Ok(Some(count))
                })
                .transpose()?;
            let max_new_tokens = args.number("--max-new-tokens")?;
            if let Some(count) = max_new_tokens {
                urbilateria::generation::GenerationConfig::greedy(count, vec![])
                    .validate()
                    .map_err(|e| e.to_string())?;
            }
            if args.flags.contains("--thinking") && args.flags.contains("--no-thinking") {
                return Err("Choose either --thinking or --no-thinking.".into());
            }
            Ok(Command::Settings(SettingsUpdate {
                ram_gib,
                threads,
                max_new_tokens,
                thinking: if args.flags.contains("--thinking") {
                    Some(true)
                } else if args.flags.contains("--no-thinking") {
                    Some(false)
                } else {
                    None
                },
            }))
        }
        _ => Err("Enter a slash command. Use /help to see available commands.".into()),
    }
}

fn parse_generate(words: &[String]) -> Result<Command, String> {
    let mut args = parse_options(
        words,
        &[
            "--model",
            "--prompt",
            "--ram-gib",
            "--max-new-tokens",
            "--threads",
            "--profile-json",
            "--profile-trace",
        ],
        &[
            "--allow-large-model",
            "--raw-prompt",
            "--no-thinking",
            "--profile",
        ],
    )?;
    let path = args.model()?;
    args.values.remove("--model");
    if let Some(prompt) = args.positional.take() {
        if args.values.insert("--prompt".into(), prompt).is_some() {
            return Err("Supply the prompt once: /generate \"TEXT\" [--model MODEL_DIR], or use --prompt TEXT.".into());
        }
    }
    if !args.flags.contains("--allow-large-model") {
        return Err("/generate requires --allow-large-model; no model weights were loaded.".into());
    }
    if args.values.get("--prompt").is_none_or(String::is_empty) {
        return Err(
            "Usage: /generate \"TEXT\" --ram-gib N --allow-large-model [--model MODEL_DIR]".into(),
        );
    }
    let ram = args
        .number::<f64>("--ram-gib")?
        .ok_or("/generate requires an explicit --ram-gib N.")?;
    if crate::gib_to_bytes(ram).map_err(|error| error.to_string())? == 0 {
        return Err("--ram-gib must provide at least one byte.".into());
    }
    // Use the same token-count limits as the CLI without configuring its global CPU pool.
    urbilateria::generation::GenerationConfig::greedy(
        args.number("--max-new-tokens")?.unwrap_or(1),
        vec![],
    )
    .validate()
    .map_err(|error| error.to_string())?;
    if args.number::<usize>("--threads")? == Some(0) {
        return Err("--threads must be greater than zero.".into());
    }
    if args.flags.contains("--raw-prompt") && args.flags.contains("--no-thinking") {
        return Err("--no-thinking applies to chat mode and conflicts with --raw-prompt.".into());
    }
    let json = args
        .values
        .get("--profile-json")
        .map(|value| expand_home(value));
    let trace = args
        .values
        .get("--profile-trace")
        .map(|value| expand_home(value));
    if json.is_some() && json == trace {
        return Err("--profile-json and --profile-trace must use different paths.".into());
    }
    let mut forwarded = Vec::new();
    for (name, value) in args.values {
        forwarded.push(OsString::from(&name));
        if matches!(name.as_str(), "--profile-json" | "--profile-trace") {
            if value.is_empty() {
                return Err(format!("{name} requires a nonempty path."));
            }
            forwarded.push(expand_home(&value).into_os_string());
        } else {
            forwarded.push(OsString::from(value));
        }
    }
    forwarded.extend(args.flags.into_iter().map(OsString::from));
    Ok(Command::Generate(
        path,
        GenerateOptions {
            args: forwarded,
            conversation: None,
        },
    ))
}

fn model_path(path: &str) -> Result<PathBuf, String> {
    if path.is_empty() {
        return Err("MODEL_DIR must not be empty.".into());
    }
    Ok(expand_home(path))
}

struct Arguments {
    positional: Option<String>,
    values: BTreeMap<String, String>,
    flags: BTreeSet<String>,
}

impl Arguments {
    fn model(&self) -> Result<Option<PathBuf>, String> {
        self.values
            .get("--model")
            .map(|path| model_path(path))
            .transpose()
    }

    fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        self.values
            .get(name)
            .map(|value| {
                value
                    .parse()
                    .map_err(|_| format!("Invalid value {value:?} for {name}."))
            })
            .transpose()
    }
}

fn parse_options(
    words: &[String],
    value_flags: &[&str],
    bool_flags: &[&str],
) -> Result<Arguments, String> {
    let mut args = Arguments {
        positional: None,
        values: BTreeMap::new(),
        flags: BTreeSet::new(),
    };
    let mut words = words.iter();
    let mut positional_only = false;
    while let Some(word) = words.next() {
        if !positional_only && word == "--" {
            positional_only = true;
        } else if !positional_only && value_flags.contains(&word.as_str()) {
            let value = words
                .next()
                .filter(|value| !value.starts_with("--"))
                .ok_or_else(|| format!("{word} requires a value."))?;
            if args.values.insert(word.clone(), value.clone()).is_some() {
                return Err(format!("{word} was specified more than once."));
            }
        } else if !positional_only && bool_flags.contains(&word.as_str()) {
            if !args.flags.insert(word.clone()) {
                return Err(format!("{word} was specified more than once."));
            }
        } else if !positional_only && word.starts_with('-') {
            return Err(format!(
                "Unknown option {word:?}. Use /help for supported options."
            ));
        } else if args.positional.replace(word.clone()).is_some() {
            return Err("Expected at most one argument besides options. Quote paths or text containing spaces.".into());
        }
    }
    Ok(args)
}

pub fn expand_home(path: &str) -> PathBuf {
    if path == "~" || path.starts_with("~/") {
        if let Some(home_dir) = std::env::var_os("HOME") {
            return PathBuf::from(home_dir).join(path.strip_prefix("~/").unwrap_or(""));
        }
    }
    PathBuf::from(path)
}

/// Never forward terminal control characters from pasted input or model metadata.
pub fn clean_text(text: &str, limit: usize) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .take(limit)
        .map(|c| if c == '\t' { ' ' } else { c })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_messages_are_verbatim_and_settings_are_validated() {
        for text in [
            "hello",
            "don't close \"the quote",
            "  中文🙂\n--profile\n/quit  ",
        ] {
            assert_eq!(parse(text).unwrap(), Command::Message(text.into()));
        }
        assert_eq!(
            parse("/settings").unwrap(),
            Command::Settings(SettingsUpdate::default())
        );
        assert_eq!(
            parse("/settings --ram-gib 2 --threads 3 --max-new-tokens 128 --no-thinking").unwrap(),
            Command::Settings(SettingsUpdate {
                ram_gib: Some(Some("2".into())),
                threads: Some(Some(3)),
                max_new_tokens: Some(128),
                thinking: Some(false)
            })
        );
        for text in [
            "/settings 2",
            "/settings --ram-gib NaN",
            "/settings --ram-gib 0",
            "/settings --max-new-tokens 0",
            "/settings --max-new-tokens 1000001",
            "/settings --threads 0",
            "/settings --thinking --no-thinking",
            "/settings --ram-gib 2 --ram-gib 3",
            "/settings --profile",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn generation_preserves_literal_prompts_and_forwards_cli_options() {
        let Command::Generate(path, options) = parse(
            "/generate '中文\n$(echo hi)' --model '/模型/one two' --ram-gib 2 --allow-large-model --max-new-tokens 32 --threads 2 --profile --profile-json 'profile one.json' --profile-trace trace.json"
        ).unwrap() else { panic!("expected generation"); };
        assert_eq!(path, Some(PathBuf::from("/模型/one two")));
        let strings: Vec<_> = options
            .args
            .iter()
            .map(|arg| arg.to_str().unwrap().to_owned())
            .collect();
        let forwarded = crate::GenerateArguments::parse(&strings).unwrap();
        assert_eq!(forwarded.value("--prompt"), Some("中文\n$(echo hi)"));
        assert_eq!(forwarded.value("--threads"), Some("2"));
        assert_eq!(forwarded.value("--max-new-tokens"), Some("32"));
        assert_eq!(forwarded.value("--ram-gib"), Some("2"));
        assert_eq!(forwarded.value("--profile-json"), Some("profile one.json"));
        assert_eq!(forwarded.value("--profile-trace"), Some("trace.json"));
        assert!(forwarded.flag("--profile"));
        assert!(forwarded.flag("--allow-large-model"));
        assert!(parse("/generate --prompt hi --ram-gib 2 --allow-large-model").is_ok());
        assert!(parse("/generate --ram-gib 2 --allow-large-model -- '--profile'").is_ok());
    }

    #[test]
    fn generation_requires_explicit_authorization_budget_and_valid_options() {
        for input in [
            "/generate hello --ram-gib 2",
            "/generate hello --allow-large-model",
            "/generate '' --ram-gib 2 --allow-large-model",
            "/generate hello --prompt hi --ram-gib 2 --allow-large-model",
            "/generate hello --ram-gib NaN --allow-large-model",
            "/generate hello --ram-gib 0.00000000001 --allow-large-model",
            "/generate hello --ram-gib 2 --allow-large-model --max-new-tokens 0",
            "/generate hello --ram-gib 2 --allow-large-model --max-new-tokens 1000001",
            "/generate hello --ram-gib 2 --allow-large-model --threads 0",
            "/generate hello --ram-gib 2 --allow-large-model --threads -1",
            "/generate hello --ram-gib 2 --allow-large-model --raw-prompt --no-thinking",
            "/generate hello --ram-gib 2 --allow-large-model --ram-gib 4",
            "/generate hello --ram-gib 2 --allow-large-model --allow-large-model",
            "/generate hello --ram-gib 2 --allow-large-model --profile-json same --profile-trace same",
            "/generate hello --ram-gib 2 --allow-large-model --profile-json ''",
            "/generate hello --ram-gib 2 --allow-large-model --model ''",
            "/generate hello --ram-gib 2 --allow-large-model --json",
        ] {
            assert!(parse(input).is_err(), "accepted {input}");
        }
    }

    #[test]
    fn parses_quoted_unicode_paths_without_running_shell_syntax() {
        assert_eq!(
            parse("/inspect '/模型/checkpoint one'"),
            Ok(Command::Inspect(Some(PathBuf::from(
                "/模型/checkpoint one"
            ))))
        );
        assert_eq!(
            parse("/inspect '$(echo test)'"),
            Ok(Command::Inspect(Some(PathBuf::from("$(echo test)"))))
        );
        assert!(parse("/inspect 'unclosed").is_err());
        assert!(parse("/help\n/quit").is_err());
        assert!(parse("/inspect one two").is_err());
    }

    #[test]
    fn strips_terminal_controls_and_normalizes_paste() {
        assert_eq!(clean_text("你\r\n好\u{1b}\0\t!", 20), "你\n好 !");
        assert_eq!(clean_text("中文🙂", 2), "中文");
    }

    #[test]
    fn analysis_commands_parse_paths_flags_and_defaults() {
        assert_eq!(
            parse("/plan"),
            Ok(Command::Plan(None, PlanOptions::default()))
        );
        assert_eq!(
            parse("/preflight"),
            Ok(Command::Preflight(None, PreflightOptions::default()))
        );
        assert_eq!(
            parse("/plan --ram-gib 1.5 '/模型/model one' --context 512 --kv-bytes 2"),
            Ok(Command::Plan(
                Some(PathBuf::from("/模型/model one")),
                PlanOptions {
                    ram_bytes: Some(1_610_612_736),
                    context: 512,
                    kv_bytes: 2,
                }
            ))
        );
        assert_eq!(
            parse("/preflight 'model one' --partial --expert-slots 0 --context 32"),
            Ok(Command::Preflight(
                Some(PathBuf::from("model one")),
                PreflightOptions {
                    context: 32,
                    expert_slots: 0,
                    partial: true,
                }
            ))
        );
        assert_eq!(
            parse("/preflight -- -checkpoint"),
            Ok(Command::Preflight(
                Some(PathBuf::from("-checkpoint")),
                PreflightOptions::default()
            ))
        );
    }

    #[test]
    fn analysis_commands_reject_invalid_parameters_before_starting_work() {
        for input in [
            "/plan --ram-gib",
            "/plan --ram-gib NaN",
            "/plan --ram-gib inf",
            "/plan --ram-gib -2",
            "/plan --ram-gib 0",
            "/plan --ram-gib 0.00000000001",
            "/plan --context 0",
            "/plan --context -1",
            "/plan --context 1.5",
            "/plan --context 2 --context 4",
            "/plan --kv-bytes 3",
            "/plan --json",
            "/plan --partial",
            "/plan one two",
            "/plan --context --kv-bytes 4",
            "/preflight --expert-slots -1",
            "/preflight --expert-slots 1.5",
            "/preflight --expert-slots 18446744073709551616",
            "/preflight --context 0",
            "/preflight --partial --partial",
            "/preflight --ram-gib 32",
        ] {
            assert!(parse(input).is_err(), "accepted {input}");
        }
    }

    #[test]
    fn browsing_commands_disambiguate_models_from_text_and_tensor_names() {
        assert_eq!(
            parse("/list"),
            Ok(Command::List(None, ListOptions::default()))
        );
        assert_eq!(
            parse("/list --model '/模型/one two' attn --limit 3"),
            Ok(Command::List(
                Some("/模型/one two".into()),
                ListOptions {
                    filter: Some("attn".into()),
                    limit: 3
                }
            ))
        );
        assert_eq!(
            parse("/explain '/模型/one two'"),
            Ok(Command::Explain(Some("/模型/one two".into())))
        );
        assert_eq!(
            parse("/probe model.norm.weight --samples 4"),
            Ok(Command::Probe(None, "model.norm.weight".into(), 4))
        );
        assert_eq!(
            parse(
                "/tokenize --model '/模型/one two' '中文\\n/quit $(echo hi)' --chat --no-thinking"
            ),
            Ok(Command::Tokenize(
                Some("/模型/one two".into()),
                "中文\\n/quit $(echo hi)".into(),
                TokenizeOptions {
                    chat: true,
                    no_thinking: true
                }
            ))
        );
        assert_eq!(
            parse("/tokenize ''"),
            Ok(Command::Tokenize(
                None,
                "".into(),
                TokenizeOptions::default()
            ))
        );
        assert_eq!(
            parse("/tokenize -- '--chat'"),
            Ok(Command::Tokenize(
                None,
                "--chat".into(),
                TokenizeOptions::default()
            ))
        );
        assert_eq!(
            parse("/decode '0, 42,4294967295' --skip-special"),
            Ok(Command::Decode(None, vec![0, 42, u32::MAX], true))
        );
        assert_eq!(parse("/version"), Ok(Command::Version));
    }

    #[test]
    fn browsing_commands_reject_ambiguous_or_invalid_arguments() {
        for input in [
            "/list --limit 0",
            "/list --limit 100001",
            "/list --limit -1",
            "/list --limit 1 --limit 2",
            "/list --model",
            "/list --model ''",
            "/list one two",
            "/list --model one --model two",
            "/list --json",
            "/probe",
            "/probe ''",
            "/probe x --samples 0",
            "/probe x --samples 10000001",
            "/probe x --samples 1.5",
            "/probe x --samples --model path",
            "/tokenize",
            "/tokenize unquoted words",
            "/tokenize x --no-thinking",
            "/tokenize x --chat --chat",
            "/tokenize x --chat --partial",
            "/decode",
            "/decode ''",
            "/decode 1,",
            "/decode 1,,2",
            "/decode 1,abc",
            "/decode 4294967296",
            "/decode -1",
            "/decode 1 --skip-special --skip-special",
            "/explain ''",
            "/explain one two",
            "/version extra",
            "/generate",
        ] {
            assert!(parse(input).is_err(), "accepted {input}");
        }
    }
}
