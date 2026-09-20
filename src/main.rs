use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use urbilateria::analysis::{
    analyze_checkpoint, build_deepseek_resource_plan, build_deepseek_v41_resource_plan,
    build_hy4_resource_plan, build_kimi_k3_resource_plan, build_resource_plan, decode_tokens,
    detect_available_ram, explain_checkpoint, inspect_checkpoint, list_tensors, parse_token_ids,
    plan_checkpoint, preflight_checkpoint, probe_tensor, tokenize_text, CheckpointReport,
    DecodeReport, InspectionReport, ListOptions, PlanOptions, PreflightOptions, PreflightReport,
    ProbeReport, ResourcePlan, TokenizeOptions, TokenizeReport,
};
use urbilateria::execution::{
    configure_threads, enable_streamed_weight_allocation_reuse, worker_threads,
};
use urbilateria::generation::{
    try_generate_with_state, GenerationConfig, GenerationError, StopReason,
};
use urbilateria::models::deepseek_v4::runtime::DeepseekRuntimeModel;
use urbilateria::models::deepseek_v4::{prompt as deepseek_prompt, schema as deepseek_schema};
use urbilateria::models::deepseek_v41::runtime::DeepseekV41RuntimeModel;
use urbilateria::models::deepseek_v41::{
    prompt as deepseek_v41_prompt, schema as deepseek_v41_schema,
};
use urbilateria::models::glm::runtime::GlmRuntimeModel;
use urbilateria::models::hy4::runtime::Hy4RuntimeModel;
use urbilateria::models::hy4::{prompt as hy4_prompt, schema as hy4_schema};
use urbilateria::models::kimi_k3::runtime::KimiK3RuntimeModel;
use urbilateria::models::kimi_k3::{prompt as kimi_k3_prompt, tokenizer::KimiK3Tokenizer};
use urbilateria::models::qwen3_8::{
    prompt as qwen38_prompt, schema as qwen38_schema, Qwen38Config, Qwen38RuntimeModel,
};
use urbilateria::profiling::{span, ProfileSession, ProfileStage};
use urbilateria::runtime::RuntimeLoadOptions;
use urbilateria::storage::TensorIndex;
use urbilateria::tokenizer::{
    render_chat, ByteBpeTokenizer, ChatMessage, ChatRole, ChatTemplateOptions, TokenizerError,
};
use urbilateria::{
    DeepseekV41Config, DeepseekV4Config, Hy4Config, KimiK3Config, ModelConfig, ModelFamily,
};

#[cfg(feature = "ui")]
mod ui;

#[cfg(test)]
mod test_support;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return Ok(());
    };
    match command.as_str() {
        "help" | "--help" | "-h" => print_help(),
        "--version" | "-V" | "version" => println!("urb {}", env!("CARGO_PKG_VERSION")),
        "ui" => {
            #[cfg(feature = "ui")]
            ui::run_args(args.collect())?;
            #[cfg(not(feature = "ui"))]
            return Err("this build does not include the TUI; rebuild with --features ui".into());
        }
        "inspect" => {
            let model = required_path(args.next(), "inspect requires MODEL_DIR")?;
            let flags: Vec<String> = args.collect();
            reject_unknown(&flags, &["--json"])?;
            let json = flags.iter().any(|flag| flag == "--json");
            let inspection =
                inspect_checkpoint(&model).map_err(|error| -> Box<dyn Error> { error })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&inspection.report)?);
            } else {
                match &inspection.report {
                    InspectionReport::Checkpoint(report) => print_checkpoint(report),
                    InspectionReport::Qwen38(report) => print_qwen38_manifest(report),
                    InspectionReport::Hy4(report) => print_hy4_manifest(report),
                    InspectionReport::DeepseekV41(report) => print_deepseek_v41_manifest(report),
                }
            }
        }
        "plan" => {
            let model = required_path(args.next(), "plan requires MODEL_DIR")?;
            let remaining: Vec<String> = args.collect();
            let json = has_flag(&remaining, "--json");
            let ram_gib = option_value(&remaining, "--ram-gib")?
                .map(|value| value.parse::<f64>())
                .transpose()?
                .map(gib_to_bytes)
                .transpose()?
                .or_else(detect_available_ram)
                .ok_or("could not detect RAM; pass --ram-gib N")?;
            let context = option_value(&remaining, "--context")?
                .map(|value| value.parse::<u64>())
                .transpose()?
                .unwrap_or(2048);
            if context == 0 {
                return Err("--context must be greater than zero".into());
            }
            let kv_bytes = option_value(&remaining, "--kv-bytes")?
                .map(|value| value.parse::<u8>())
                .transpose()?
                .unwrap_or(4);
            if !matches!(kv_bytes, 2 | 4) {
                return Err("--kv-bytes must be 2 or 4".into());
            }
            reject_unknown_with_values(
                &remaining,
                &["--json"],
                &["--ram-gib", "--context", "--kv-bytes"],
            )?;
            let planning = plan_checkpoint(
                &model,
                PlanOptions {
                    ram_bytes: Some(ram_gib),
                    context,
                    kv_bytes,
                },
            )
            .map_err(|error| -> Box<dyn Error> { error })?;
            let plan = planning.report;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                print_plan(&plan);
            }
        }
        "preflight" => {
            let model = required_path(args.next(), "preflight requires MODEL_DIR")?;
            let remaining: Vec<String> = args.collect();
            let json = has_flag(&remaining, "--json");
            let partial = has_flag(&remaining, "--partial");
            let context = option_value(&remaining, "--context")?
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(1);
            let expert_slots = option_value(&remaining, "--expert-slots")?
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(0);
            reject_unknown_with_values(
                &remaining,
                &["--json", "--partial"],
                &["--context", "--expert-slots"],
            )?;
            let preflight = preflight_checkpoint(
                &model,
                PreflightOptions {
                    context,
                    expert_slots,
                    partial,
                },
            )
            .map_err(|error| -> Box<dyn Error> { error })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&preflight.report)?);
            } else {
                print_preflight(&preflight.report);
            }
        }
        "probe" => {
            let model = required_path(args.next(), "probe requires MODEL_DIR")?;
            let tensor = args.next().ok_or("probe requires an exact TENSOR_NAME")?;
            let remaining: Vec<String> = args.collect();
            let json = has_flag(&remaining, "--json");
            let samples = option_value(&remaining, "--samples")?
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(8192);
            reject_unknown_with_values(&remaining, &["--json"], &["--samples"])?;
            let report = probe_tensor(&model, &tensor, samples)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_probe(&report);
            }
        }
        "tokenize" => {
            let model = required_path(args.next(), "tokenize requires MODEL_DIR")?;
            let text = args.next().ok_or("tokenize requires quoted TEXT")?;
            let remaining: Vec<String> = args.collect();
            let json = has_flag(&remaining, "--json");
            let chat = has_flag(&remaining, "--chat");
            let no_thinking = has_flag(&remaining, "--no-thinking");
            reject_unknown(&remaining, &["--json", "--chat", "--no-thinking"])?;
            if no_thinking && !chat {
                return Err("--no-thinking requires --chat".into());
            }
            let tokenization = tokenize_text(&model, text, TokenizeOptions { chat, no_thinking })
                .map_err(|error| -> Box<dyn Error> { error })?;
            let TokenizeReport {
                prompt, token_ids, ..
            } = tokenization.report;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "prompt": prompt,
                        "token_ids": token_ids,
                        "token_count": token_ids.len(),
                    }))?
                );
            } else {
                println!("prompt: {prompt}");
                println!(
                    "tokens ({}): {}",
                    token_ids.len(),
                    token_ids
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        "decode" => {
            let model = required_path(args.next(), "decode requires MODEL_DIR")?;
            let token_text = args
                .next()
                .ok_or("decode requires comma-separated TOKEN_IDS")?;
            let remaining: Vec<String> = args.collect();
            let json = has_flag(&remaining, "--json");
            let skip_special = has_flag(&remaining, "--skip-special");
            reject_unknown(&remaining, &["--json", "--skip-special"])?;
            let token_ids = parse_token_ids(&token_text)?;
            let decoding = decode_tokens(&model, token_ids, skip_special)
                .map_err(|error| -> Box<dyn Error> { error })?;
            let DecodeReport { text, token_ids } = decoding.report;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "token_ids": token_ids,
                        "text": text,
                    }))?
                );
            } else {
                println!("{text}");
            }
        }
        "generate" => {
            let model = required_path(args.next(), "generate requires MODEL_DIR")?;
            let remaining: Vec<String> = args.collect();
            run_generate(&model, &remaining)?;
        }
        "list" => {
            let model = required_path(args.next(), "list requires MODEL_DIR")?;
            let mut remaining: Vec<String> = args.collect();
            let filter = if remaining
                .first()
                .is_some_and(|value| !value.starts_with('-'))
            {
                Some(remaining.remove(0))
            } else {
                None
            };
            let json = has_flag(&remaining, "--json");
            let limit = option_value(&remaining, "--limit")?
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(100);
            if limit == 0 || limit > 100_000 {
                return Err("--limit must be in 1..=100000".into());
            }
            reject_unknown_with_values(&remaining, &["--json"], &["--limit"])?;
            let listing = list_tensors(&model, ListOptions { filter, limit })
                .map_err(|error| -> Box<dyn Error> { error })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&listing.tensors)?);
            } else {
                println!(
                    "{} tensor(s) match; showing at most {limit}",
                    listing.total_matches
                );
                for tensor in &listing.tensors {
                    println!(
                        "{:>10}  {:<10}  {:?}  {}",
                        human_bytes(tensor.data_len),
                        tensor.dtype,
                        tensor.shape,
                        tensor.name
                    );
                }
            }
        }
        "explain" => {
            let model = required_path(args.next(), "explain requires MODEL_DIR")?;
            let remaining: Vec<String> = args.collect();
            reject_unknown(&remaining, &[])?;
            let explanation =
                explain_checkpoint(&model).map_err(|error| -> Box<dyn Error> { error })?;
            for line in explanation.lines {
                println!("{line}");
            }
        }
        other => return Err(format!("unknown command {other:?}; run `urb help`").into()),
    }
    Ok(())
}

fn print_help() {
    println!(
        "Urbilateria — a small pure-Rust model inference and analysis framework\n\n\
Usage:\n  \
  urb ui [MODEL_DIR]\n  \
  urb inspect MODEL_DIR [--json]\n  \
  urb plan MODEL_DIR [--ram-gib N] [--context N] [--kv-bytes 2|4] [--json]\n  \
  urb preflight MODEL_DIR [--context N] [--expert-slots N] [--partial] [--json]\n  \
  urb list MODEL_DIR [FILTER] [--limit N] [--json]\n  \
  urb probe MODEL_DIR TENSOR_NAME [--samples N] [--json]\n  \
  urb tokenize MODEL_DIR TEXT [--chat] [--no-thinking] [--json]\n  \
  urb decode MODEL_DIR TOKEN_IDS [--skip-special] [--json]\n  \
  urb generate MODEL_DIR --prompt TEXT --ram-gib N --allow-large-model\n    \
      [--max-new-tokens N] [--threads N] [--profile] [--profile-json PATH]\n    \
      [--profile-trace PATH]\n    \
      [--raw-prompt | --no-thinking]\n  \
  urb explain MODEL_DIR\n\n\
Commands:\n  \
  ui       Open the interactive checkpoint explorer (requires the ui feature)\n  \
  inspect  Validate config/shard headers and build a static parameter X-ray\n  \
  plan     Estimate resident, KV, scratch, and per-layer expert-cache budgets\n  \
  preflight Validate every runtime tensor without payload reads; Kimi `--partial` validates completed layers during rsync\n  \
  list     Find exact tensor names without reading payloads\n  \
  probe    Sample one tensor without scanning or loading the whole checkpoint\n  \
  tokenize Encode raw text or one model-native text-only user turn with byte-level BPE\n  \
  decode   Decode comma-separated token IDs, preserving special tokens by default\n  \
  generate Run explicitly authorized, RAM-planned greedy generation (experimental)\n  \
  explain  Print the model's token path and tensor geometry\n\n\
`--raw-prompt` accepts an already-rendered model-native prompt, not bare user text.\n\
For ordinary text omit it; add `--no-thinking` for non-reasoning chat.\n\n\
Large matrix output rows run on one persistent CPU pool. `--threads 1` is the serial baseline;\n\
without `--threads`, the pool uses one worker per available physical core. Profile text goes to\n\
stderr and profile JSON to the requested file, never to streamed stdout.\n\n\
The public `generate` path currently drives GLM, DeepSeek-V4, Kimi-K3, Qwen3.8, and Hy4 (Hy4 is
exact through 2,048 total tokens, where DSA top-k selects the complete causal history). Qwen3.8
supports text-only, always-thinking generation. Kimi vision inputs are not accepted."
    );
}

#[derive(Debug)]
enum StreamError {
    Tokenizer(TokenizerError),
    Io(io::Error),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenizer(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl Error for StreamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tokenizer(error) => Some(error),
            Self::Io(error) => Some(error),
        }
    }
}

impl From<TokenizerError> for StreamError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

impl From<io::Error> for StreamError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

fn run_generate(model_dir: &Path, args: &[String]) -> Result<(), Box<dyn Error>> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    run_generate_to(model_dir, args, None, &mut output)
}

fn run_generate_to<W: Write>(
    model_dir: &Path,
    args: &[String],
    available_ram_override: Option<u64>,
    output: &mut W,
) -> Result<(), Box<dyn Error>> {
    reject_duplicate_flag(args, "--profile")?;
    reject_unknown_with_values(
        args,
        &[
            "--allow-large-model",
            "--raw-prompt",
            "--no-thinking",
            "--profile",
        ],
        &[
            "--prompt",
            "--ram-gib",
            "--max-new-tokens",
            "--threads",
            "--profile-json",
            "--profile-trace",
        ],
    )?;
    let requested_threads = option_value(args, "--threads")?
        .map(str::parse::<usize>)
        .transpose()
        .map_err(|error| format!("invalid --threads value: {error}"))?;
    if let Some(threads) = requested_threads {
        configure_threads(threads)?;
    }
    let print_profile = has_flag(args, "--profile");
    let profile_json = option_value(args, "--profile-json")?.map(PathBuf::from);
    let profile_trace = option_value(args, "--profile-trace")?.map(PathBuf::from);
    if profile_json.is_some() && profile_json == profile_trace {
        return Err("--profile-json and --profile-trace must use different paths".into());
    }
    let profile = (print_profile || profile_json.is_some() || profile_trace.is_some()).then(|| {
        ProfileSession::start_with_threads_and_trace(requested_threads, profile_trace.is_some())
    });
    let result = {
        let _profile = span(ProfileStage::GenerateTotal);
        run_generate_inner(model_dir, args, available_ram_override, output)
    };
    let profile_result = if let Some(profile) = profile {
        let report = profile.finish();
        if print_profile {
            eprint!("{}", report.render_text());
        }
        (|| {
            if let Some(path) = profile_json {
                let json = serde_json::to_vec_pretty(&report)
                    .map_err(|error| -> Box<dyn Error> { Box::new(error) })?;
                fs::write(&path, json).map_err(|error| -> Box<dyn Error> {
                    format!("could not write profile {}: {error}", path.display()).into()
                })?;
            }
            if let Some(path) = profile_trace {
                let json = report
                    .chrome_trace_json_pretty()
                    .map_err(|error| -> Box<dyn Error> { Box::new(error) })?
                    .ok_or("profile trace collection was not enabled")?;
                fs::write(&path, json).map_err(|error| -> Box<dyn Error> {
                    format!("could not write profile trace {}: {error}", path.display()).into()
                })?;
                if report.trace_dropped_events() != 0 {
                    eprintln!(
                        "warning: profile trace reached its event limit; {} span events were dropped",
                        report.trace_dropped_events()
                    );
                }
            }
            Ok(())
        })()
    } else {
        Ok(())
    };
    match result {
        Err(error) => {
            if let Err(profile_error) = profile_result {
                eprintln!("warning: {profile_error}");
            }
            Err(error)
        }
        Ok(()) => profile_result,
    }
}

fn run_generate_inner<W: Write>(
    model_dir: &Path,
    args: &[String],
    available_ram_override: Option<u64>,
    output: &mut W,
) -> Result<(), Box<dyn Error>> {
    for flag in ["--allow-large-model", "--raw-prompt", "--no-thinking"] {
        reject_duplicate_flag(args, flag)?;
    }
    reject_unknown_with_values(
        args,
        &[
            "--allow-large-model",
            "--raw-prompt",
            "--no-thinking",
            "--profile",
        ],
        &[
            "--prompt",
            "--ram-gib",
            "--max-new-tokens",
            "--threads",
            "--profile-json",
            "--profile-trace",
        ],
    )?;
    if !has_flag(args, "--allow-large-model") {
        return Err("generate requires --allow-large-model; no model weights were loaded".into());
    }
    let prompt = option_value(args, "--prompt")?
        .ok_or("generate requires --prompt TEXT")?
        .to_owned();
    if prompt.is_empty() {
        return Err("--prompt must not be empty".into());
    }
    let requested_ram = option_value(args, "--ram-gib")?
        .ok_or("generate requires an explicit --ram-gib N")?
        .parse::<f64>()
        .map_err(|error| format!("invalid --ram-gib value: {error}"))
        .and_then(|value| gib_to_bytes(value).map_err(|error| error.to_string()))?;
    let max_new_tokens = option_value(args, "--max-new-tokens")?
        .map(str::parse::<usize>)
        .transpose()?
        .unwrap_or(1);
    let raw_prompt = has_flag(args, "--raw-prompt");
    let no_thinking = has_flag(args, "--no-thinking");
    if raw_prompt && no_thinking {
        return Err("--no-thinking applies to chat mode and conflicts with --raw-prompt".into());
    }

    let model_config = {
        let _profile = span(ProfileStage::ConfigLoad);
        ModelConfig::load(model_dir)?
    };
    if raw_prompt {
        validate_native_raw_prompt(model_config.family(), &prompt)?;
    }
    if let ModelConfig::Qwen38(config) = model_config {
        return run_generate_qwen38(
            model_dir,
            config,
            prompt,
            requested_ram,
            max_new_tokens,
            raw_prompt,
            no_thinking,
            available_ram_override,
            output,
        );
    }
    if let ModelConfig::Hy4(config) = model_config {
        return run_generate_hy4(
            model_dir,
            config,
            prompt,
            requested_ram,
            max_new_tokens,
            raw_prompt,
            no_thinking,
            available_ram_override,
            output,
        );
    }
    if let ModelConfig::DeepseekV4(config) = model_config {
        return run_generate_deepseek(
            model_dir,
            config,
            prompt,
            requested_ram,
            max_new_tokens,
            raw_prompt,
            no_thinking,
            available_ram_override,
            output,
        );
    }
    if let ModelConfig::DeepseekV41(config) = model_config {
        return run_generate_deepseek_v41(
            model_dir,
            config,
            prompt,
            requested_ram,
            max_new_tokens,
            raw_prompt,
            no_thinking,
            available_ram_override,
            output,
        );
    }
    if let ModelConfig::KimiK3(config) = model_config {
        return run_generate_kimi(
            model_dir,
            config,
            prompt,
            requested_ram,
            max_new_tokens,
            raw_prompt,
            no_thinking,
            available_ram_override,
            output,
        );
    }
    let ModelConfig::Glm52(config) = model_config else {
        unreachable!("DeepSeek-V4/V4.1, Hy4, Kimi-K3, and Qwen3.8 returned above")
    };
    let prompt_profile = span(ProfileStage::TokenizerPrompt);
    let tokenizer = ByteBpeTokenizer::load(model_dir)?;
    let rendered = if raw_prompt {
        prompt
    } else {
        tokenizer.validate_chat_content(&prompt)?;
        render_chat(
            &[ChatMessage::new(ChatRole::User, prompt)],
            ChatTemplateOptions {
                enable_thinking: !no_thinking,
                ..ChatTemplateOptions::default()
            },
        )
    };
    let prompt_tokens = tokenizer.encode(&rendered)?;
    drop(prompt_profile);
    if prompt_tokens.is_empty() {
        return Err("rendered prompt encoded to zero tokens".into());
    }
    if let Some(&token) = prompt_tokens
        .iter()
        .find(|&&token| token as usize >= config.vocab_size)
    {
        return Err(format!(
            "tokenizer produced ID {token} outside model vocabulary {}",
            config.vocab_size
        )
        .into());
    }
    let eos = config.eos_token_id.as_slice().to_vec();
    if eos.is_empty() {
        return Err("config.eos_token_id is empty; refusing unbounded stop semantics".into());
    }
    if let Some(&token) = eos
        .iter()
        .find(|&&token| !tokenizer.is_decodable_token_id(token))
    {
        return Err(format!(
            "EOS token ID {token} has no tokenizer entry; refusing undecodable stop semantics"
        )
        .into());
    }
    let decodable_mask = tokenizer.decodable_token_mask(config.vocab_size);
    let suppressed_rows = decodable_mask.iter().filter(|allowed| !**allowed).count();
    let mut generation = GenerationConfig::greedy(max_new_tokens, eos.clone());
    generation.allowed_token_mask = Some(decodable_mask);
    generation.validate()?;
    let required_context = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or("prompt + generation length overflows usize")?;
    let exact_limit = if config.index_topk > 0 {
        config.index_topk.min(config.max_position_embeddings)
    } else {
        config.max_position_embeddings
    };
    if required_context > exact_limit {
        return Err(format!(
            "prompt ({}) + max new tokens ({max_new_tokens}) = {required_context}, exceeding exact dense-MLA limit {exact_limit}",
            prompt_tokens.len()
        )
        .into());
    }

    let preflight_profile = span(ProfileStage::CheckpointPreflight);
    let report = analyze_checkpoint(model_dir)?;
    let expected_experts = config
        .sparse_layer_count()
        .checked_mul(config.n_routed_experts)
        .ok_or("expected expert count overflows usize")?;
    if report.missing_base_tensor_count != 0
        || report.sidecars_without_weight != 0
        || report.unknown_packed_bytes != 0
        || report.routed_expert_count_found != expected_experts
    {
        return Err(format!(
            "checkpoint is not runnable: missing={} orphan_sidecars={} unknown_packed_bytes={} complete_experts={}/{}",
            report.missing_base_tensor_count,
            report.sidecars_without_weight,
            report.unknown_packed_bytes,
            report.routed_expert_count_found,
            expected_experts
        )
        .into());
    }
    let effective_ram = available_ram_override
        .or_else(detect_available_ram)
        .map(|available| available.min(requested_ram))
        .unwrap_or(requested_ram);
    let plan = build_resource_plan(&config, &report, effective_ram, required_context as u64, 4);
    drop(preflight_profile);
    if !plan.feasible {
        return Err(format!(
            "RAM plan is infeasible for {} bytes and {required_context} tokens: {}",
            effective_ram,
            plan.notes.join("; ")
        )
        .into());
    }
    let expert_slots = usize::try_from(plan.expert_slots_per_sparse_layer)
        .map_err(|_| "planned expert slots do not fit usize")?;
    eprintln!(
        "preflight: prompt={} tokens, context={}, RAM={}, resident={}, KV={}, expert slots/layer={}, suppressed LM-head rows={}, CPU workers={} (experimental runtime)",
        prompt_tokens.len(),
        required_context,
        human_bytes(effective_ram),
        human_bytes(plan.resident_core_bytes),
        human_bytes(plan.kv_cache_bytes),
        expert_slots,
        suppressed_rows,
        worker_threads()
    );
    let model = {
        let _profile = span(ProfileStage::ModelLoad);
        GlmRuntimeModel::load(
            model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: plan.resident_core_bytes,
                expert_cache_budget_bytes: plan.expert_cache_budget_bytes,
                kv_cache_budget_bytes: plan.kv_cache_bytes,
                expert_slots_per_layer: expert_slots,
                maximum_expert_bytes: plan.maximum_expert_bytes.max(1),
                context_limit: required_context,
            },
        )?
    };
    let mut state = {
        let _profile = span(ProfileStage::StateInit);
        model.new_state()?
    };
    let mut decoder = tokenizer.streaming_decoder(false);
    let result = try_generate_with_state(
        &model,
        &prompt_tokens,
        &generation,
        &mut state,
        |token| -> Result<(), StreamError> {
            if eos.contains(&token) {
                return Ok(());
            }
            let text = decoder.push(token)?;
            output.write_all(text.as_bytes())?;
            output.flush()?;
            Ok(())
        },
    );
    let generated = match result {
        Ok(generated) => generated,
        Err(GenerationError::Callback(StreamError::Io(error)))
            if error.kind() == io::ErrorKind::BrokenPipe =>
        {
            return Ok(())
        }
        Err(error) => return Err(Box::new(error)),
    };
    let tail = decoder.finish();
    output.write_all(tail.as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    let telemetry = state.expert_telemetry();
    eprintln!(
        "done: new_tokens={}, stop={}, expert hits={}, misses={}, evictions={}, expert_payload_bytes_read={}",
        generated.generated_tokens.len(),
        match generated.stop_reason {
            StopReason::Eos(token) => format!("eos:{token}"),
            StopReason::MaxNewTokens => "max_new_tokens".to_owned(),
        },
        telemetry.hits,
        telemetry.misses,
        telemetry.evictions,
        telemetry.bytes_read
    );
    Ok(())
}

fn validate_native_raw_prompt(family: ModelFamily, prompt: &str) -> Result<(), Box<dyn Error>> {
    let required_prefix = match family {
        ModelFamily::Glm52 => "[gMASK]<sop>",
        ModelFamily::DeepseekV4 => "<｜begin▁of▁sentence｜>",
        ModelFamily::DeepseekV41 => "<｜begin▁of▁sentence｜>",
        ModelFamily::Hy4 => "<｜hy_start:opensource｜>",
        ModelFamily::KimiK3 => "<|open|>",
        ModelFamily::Qwen38 => "<|im_start|>",
    };
    if prompt.starts_with(required_prefix) {
        Ok(())
    } else {
        Err(format!(
            "--raw-prompt for {family} expects an already-rendered model-native prompt beginning {required_prefix:?}; bare user text can produce immediate EOS or unrelated output. Omit --raw-prompt for normal chat, and add --no-thinking if reasoning should be disabled"
        )
        .into())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generate_qwen38<W: Write>(
    model_dir: &Path,
    config: Qwen38Config,
    prompt: String,
    requested_ram: u64,
    max_new_tokens: usize,
    raw_prompt: bool,
    no_thinking: bool,
    available_ram_override: Option<u64>,
    output: &mut W,
) -> Result<(), Box<dyn Error>> {
    if no_thinking {
        return Err("Qwen3.8 requires thinking; --no-thinking is unsupported".into());
    }
    let prompt_profile = span(ProfileStage::TokenizerPrompt);
    let tokenizer = ByteBpeTokenizer::load(model_dir)?;
    let rendered = if raw_prompt {
        prompt
    } else {
        tokenizer.validate_chat_content(&prompt)?;
        qwen38_prompt::render_chat(
            &[qwen38_prompt::Message::new(
                qwen38_prompt::Role::User,
                prompt,
            )],
            qwen38_prompt::PromptOptions::default(),
        )?
    };
    let prompt_tokens = tokenizer.encode(&rendered)?;
    drop(prompt_profile);
    if prompt_tokens.is_empty() {
        return Err("rendered prompt encoded to zero tokens".into());
    }
    if let Some(&token) = prompt_tokens
        .iter()
        .find(|&&token| token as usize >= config.vocab_size)
    {
        return Err(format!(
            "tokenizer produced ID {token} outside model vocabulary {}",
            config.vocab_size
        )
        .into());
    }
    let generation_config = urbilateria::models::qwen3_8::Qwen38GenerationConfig::load(model_dir)?;
    let eos = generation_config.eos_token_id;
    for &token in &eos {
        if !tokenizer.is_decodable_token_id(token) {
            return Err(format!(
                "Qwen3.8 EOS token ID {token} has no tokenizer entry; refusing undecodable stop semantics"
            )
            .into());
        }
    }
    let decodable_mask = tokenizer.decodable_token_mask(config.vocab_size);
    let suppressed_rows = decodable_mask.iter().filter(|allowed| !**allowed).count();
    let mut generation = GenerationConfig::greedy(max_new_tokens, eos.clone());
    generation.allowed_token_mask = Some(decodable_mask);
    generation.validate()?;
    let required_context = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or("prompt + generation length overflows usize")?;
    if required_context > config.max_position_embeddings {
        return Err(format!(
            "prompt ({}) + max new tokens ({max_new_tokens}) = {required_context}, exceeding Qwen3.8 limit {}",
            prompt_tokens.len(),
            config.max_position_embeddings
        )
        .into());
    }

    let effective_ram = available_ram_override
        .or_else(detect_available_ram)
        .map(|available| available.min(requested_ram))
        .unwrap_or(requested_ram);
    let preflight_profile = span(ProfileStage::CheckpointPreflight);
    // Start with zero retained expert slots, which is the lowest-memory correct execution mode.
    // A later planner can trade memory for throughput without changing forward semantics.
    let requirements = Qwen38RuntimeModel::inspect_requirements(model_dir, required_context, 0)?;
    let state_bytes = requirements
        .kv_cache_bytes
        .checked_add(requirements.recurrent_state_bytes)
        .and_then(|value| value.checked_add(requirements.convolution_state_bytes))
        .ok_or("Qwen3.8 state byte count overflows")?;
    let total_required = requirements
        .peak_resident_bytes
        .checked_add(state_bytes)
        .ok_or("Qwen3.8 total resident byte count overflows")?;
    if total_required > effective_ram {
        return Err(format!(
            "Qwen3.8 runtime needs at least {} for streamed layer, state, scratch, and one transient expert; effective RAM is {}",
            human_bytes(total_required),
            human_bytes(effective_ram)
        )
        .into());
    }
    drop(preflight_profile);
    eprintln!(
        "preflight: Qwen3.8 prompt={} tokens, context={}, RAM={}, streamed-layer peak={}, recurrent+conv state={}, full-GQA KV={}, expert cache slots/layer=0, transient expert={}, suppressed LM-head rows={}, CPU workers={} (scalar correctness runtime)",
        prompt_tokens.len(),
        required_context,
        human_bytes(effective_ram),
        human_bytes(requirements.streamed_layer_bytes),
        human_bytes(
            requirements
                .recurrent_state_bytes
                .saturating_add(requirements.convolution_state_bytes)
        ),
        human_bytes(requirements.kv_cache_bytes),
        human_bytes(requirements.expert_bytes),
        suppressed_rows,
        worker_threads()
    );
    let model = {
        let _profile = span(ProfileStage::ModelLoad);
        Qwen38RuntimeModel::load(
            model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: requirements.peak_resident_bytes,
                expert_cache_budget_bytes: 0,
                kv_cache_budget_bytes: state_bytes,
                expert_slots_per_layer: 0,
                maximum_expert_bytes: requirements.expert_bytes,
                context_limit: required_context,
            },
        )?
    };
    let mut state = {
        let _profile = span(ProfileStage::StateInit);
        model.new_state()?
    };
    let mut decoder = tokenizer.streaming_decoder(false);
    let result = try_generate_with_state(
        &model,
        &prompt_tokens,
        &generation,
        &mut state,
        |token| -> Result<(), StreamError> {
            if eos.contains(&token) {
                return Ok(());
            }
            let text = decoder.push(token)?;
            output.write_all(text.as_bytes())?;
            output.flush()?;
            Ok(())
        },
    );
    let generated = match result {
        Ok(generated) => generated,
        Err(GenerationError::Callback(StreamError::Io(error)))
            if error.kind() == io::ErrorKind::BrokenPipe =>
        {
            return Ok(())
        }
        Err(error) => return Err(Box::new(error)),
    };
    output.write_all(decoder.finish().as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    let telemetry = state.expert_telemetry();
    eprintln!(
        "done: new_tokens={}, stop={}, expert hits={}, misses={}, evictions={}, expert_payload_bytes_read={}",
        generated.generated_tokens.len(),
        match generated.stop_reason {
            StopReason::Eos(token) => format!("eos:{token}"),
            StopReason::MaxNewTokens => "max_new_tokens".to_owned(),
        },
        telemetry.hits,
        telemetry.misses,
        telemetry.evictions,
        telemetry.bytes_read
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_generate_kimi<W: Write>(
    model_dir: &Path,
    config: KimiK3Config,
    prompt: String,
    requested_ram: u64,
    max_new_tokens: usize,
    raw_prompt: bool,
    no_thinking: bool,
    available_ram_override: Option<u64>,
    output: &mut W,
) -> Result<(), Box<dyn Error>> {
    let prompt_profile = span(ProfileStage::TokenizerPrompt);
    let tokenizer = KimiK3Tokenizer::load(model_dir)?;
    let text = &config.text_config;
    if tokenizer.vocabulary_size() != text.vocab_size {
        return Err(format!(
            "Kimi-K3 tokenizer has {} entries, model vocabulary has {}",
            tokenizer.vocabulary_size(),
            text.vocab_size
        )
        .into());
    }
    let prompt_tokens = if raw_prompt {
        // `--raw-prompt` is the explicitly trusted escape hatch; preserve its XTML controls.
        tokenizer.encode_with_special_tokens(&prompt)?
    } else {
        let messages = [kimi_k3_prompt::Message::new(
            kimi_k3_prompt::Role::User,
            prompt,
        )];
        let options = kimi_k3_prompt::PromptOptions {
            thinking: !no_thinking,
            thinking_effort: (!no_thinking).then_some(kimi_k3_prompt::ThinkingEffort::Max),
            ..kimi_k3_prompt::PromptOptions::default()
        };
        tokenizer.encode_chat(&messages, options)?
    };
    drop(prompt_profile);
    if prompt_tokens.is_empty() {
        return Err("rendered prompt encoded to zero tokens".into());
    }
    if let Some(&token) = prompt_tokens
        .iter()
        .find(|&&token| token as usize >= text.vocab_size)
    {
        return Err(format!(
            "tokenizer produced ID {token} outside model vocabulary {}",
            text.vocab_size
        )
        .into());
    }
    let eos_token = config.eos_token_id;
    if tokenizer.stop_token_id() != eos_token {
        return Err(format!(
            "Kimi-K3 config stops on token {eos_token}, tokenizer message stop is {}",
            tokenizer.stop_token_id()
        )
        .into());
    }
    if !tokenizer.is_decodable_token_id(eos_token) {
        return Err(format!(
            "EOS token ID {eos_token} has no tokenizer entry; refusing undecodable stop semantics"
        )
        .into());
    }
    let eos = vec![eos_token];
    let decodable_mask = tokenizer.decodable_token_mask(text.vocab_size);
    let suppressed_rows = decodable_mask.iter().filter(|allowed| !**allowed).count();
    let mut generation = GenerationConfig::greedy(max_new_tokens, eos.clone());
    generation.allowed_token_mask = Some(decodable_mask);
    generation.validate()?;
    let required_context = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or("prompt + generation length overflows usize")?;
    if required_context > text.max_position_embeddings {
        return Err(format!(
            "prompt ({}) + max new tokens ({max_new_tokens}) = {required_context}, exceeding model limit {}",
            prompt_tokens.len(),
            text.max_position_embeddings
        )
        .into());
    }

    let effective_ram = available_ram_override
        .or_else(detect_available_ram)
        .map(|available| available.min(requested_ram))
        .unwrap_or(requested_ram);
    let preflight_profile = span(ProfileStage::CheckpointPreflight);
    let report = analyze_checkpoint(model_dir)?;
    let plan =
        build_kimi_k3_resource_plan(&config, &report, effective_ram, required_context as u64, 4);
    if !plan.feasible {
        return Err(format!(
            "RAM plan is infeasible for {} bytes and {required_context} tokens: {}",
            effective_ram,
            plan.notes.join("; ")
        )
        .into());
    }
    let expert_slots = usize::try_from(plan.expert_slots_per_sparse_layer)
        .map_err(|_| "planned expert slots do not fit usize")?;
    // This is also the strict 96-shard, 497,220-tensor execution gate.
    let requirements =
        KimiK3RuntimeModel::inspect_requirements(model_dir, required_context, expert_slots)?;
    let planned_resident = plan
        .resident_core_bytes
        .checked_add(plan.scratch_bytes)
        .ok_or("planned Kimi-K3 resident bytes overflow u64")?;
    for (component, required, maximum) in [
        (
            "resident core plus prefill scratch",
            requirements.resident_bytes,
            planned_resident,
        ),
        (
            "compressed MLA cache",
            requirements.mla_cache_bytes,
            plan.kv_cache_bytes,
        ),
        (
            "expert cache",
            requirements.expert_cache_bytes,
            plan.expert_cache_budget_bytes,
        ),
        (
            "largest routed expert",
            requirements.routed_expert_bytes,
            plan.maximum_expert_bytes,
        ),
    ] {
        if required > maximum {
            return Err(format!(
                "Kimi-K3 planner/runtime mismatch: {component} needs {required} bytes, plan authorizes {maximum}"
            )
            .into());
        }
    }
    drop(preflight_profile);
    eprintln!(
        "preflight: Kimi-K3 prompt={} tokens, context={}, RAM={}, streamed-layer peak={}, resident+scratch={}, KDA state={}, MLA KV={}, expert slots/layer={}, suppressed LM-head rows={}, CPU workers={} (text correctness runtime)",
        prompt_tokens.len(),
        required_context,
        human_bytes(effective_ram),
        human_bytes(requirements.streamed_layer_bytes),
        human_bytes(requirements.resident_bytes),
        human_bytes(requirements.kda_state_bytes),
        human_bytes(requirements.mla_cache_bytes),
        expert_slots,
        suppressed_rows,
        worker_threads()
    );
    let model = {
        let _profile = span(ProfileStage::ModelLoad);
        KimiK3RuntimeModel::load(
            model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: requirements.resident_bytes,
                expert_cache_budget_bytes: requirements.expert_cache_bytes,
                kv_cache_budget_bytes: requirements.mla_cache_bytes,
                expert_slots_per_layer: expert_slots,
                maximum_expert_bytes: requirements.routed_expert_bytes,
                context_limit: required_context,
            },
        )?
    };
    let mut state = {
        let _profile = span(ProfileStage::StateInit);
        model.new_state()?
    };
    let mut decoder = tokenizer.streaming_decoder(false);
    let result = try_generate_with_state(
        &model,
        &prompt_tokens,
        &generation,
        &mut state,
        |token| -> Result<(), StreamError> {
            if eos.contains(&token) {
                return Ok(());
            }
            let text = decoder.push(token)?;
            output.write_all(text.as_bytes())?;
            output.flush()?;
            Ok(())
        },
    );
    let generated = match result {
        Ok(generated) => generated,
        Err(GenerationError::Callback(StreamError::Io(error)))
            if error.kind() == io::ErrorKind::BrokenPipe =>
        {
            return Ok(())
        }
        Err(error) => return Err(Box::new(error)),
    };
    output.write_all(decoder.finish().as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    let telemetry = state.expert_telemetry();
    eprintln!(
        "done: new_tokens={}, stop={}, expert hits={}, misses={}, evictions={}, expert_payload_bytes_read={}",
        generated.generated_tokens.len(),
        match generated.stop_reason {
            StopReason::Eos(token) => format!("eos:{token}"),
            StopReason::MaxNewTokens => "max_new_tokens".to_owned(),
        },
        telemetry.hits,
        telemetry.misses,
        telemetry.evictions,
        telemetry.bytes_read
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_generate_hy4<W: Write>(
    model_dir: &Path,
    config: Hy4Config,
    prompt: String,
    requested_ram: u64,
    max_new_tokens: usize,
    raw_prompt: bool,
    no_thinking: bool,
    available_ram_override: Option<u64>,
    output: &mut W,
) -> Result<(), Box<dyn Error>> {
    let prompt_profile = span(ProfileStage::TokenizerPrompt);
    let tokenizer = ByteBpeTokenizer::load(model_dir)?;
    let rendered = if raw_prompt {
        prompt
    } else {
        tokenizer.validate_chat_content(&prompt)?;
        hy4_prompt::render_chat(
            &[hy4_prompt::Message::new(hy4_prompt::Role::User, prompt)],
            hy4_prompt::PromptOptions {
                enable_thinking: !no_thinking,
                ..hy4_prompt::PromptOptions::default()
            },
        )
    };
    let prompt_tokens = tokenizer.encode(&rendered)?;
    drop(prompt_profile);
    if prompt_tokens.is_empty() {
        return Err("rendered prompt encoded to zero tokens".into());
    }
    if let Some(&token) = prompt_tokens
        .iter()
        .find(|&&token| token as usize >= config.vocab_size)
    {
        return Err(format!(
            "tokenizer produced ID {token} outside model vocabulary {}",
            config.vocab_size
        )
        .into());
    }
    let eos = vec![config.eos_token_id];
    if !tokenizer.is_decodable_token_id(config.eos_token_id) {
        return Err(format!(
            "EOS token ID {} has no tokenizer entry; refusing undecodable stop semantics",
            config.eos_token_id
        )
        .into());
    }
    let decodable_mask = tokenizer.decodable_token_mask(config.vocab_size);
    let suppressed_rows = decodable_mask.iter().filter(|allowed| !**allowed).count();
    let mut generation = GenerationConfig::greedy(max_new_tokens, eos.clone());
    generation.allowed_token_mask = Some(decodable_mask);
    generation.validate()?;
    let required_context = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or("prompt + generation length overflows usize")?;
    if required_context > config.exact_dense_context_ceiling() {
        return Err(format!(
            "Hy4 exact base runtime currently supports at most {} prompt+generation tokens (DSA top-k); requested {required_context}",
            config.exact_dense_context_ceiling()
        )
        .into());
    }
    let effective_ram = available_ram_override
        .or_else(detect_available_ram)
        .map(|available| available.min(requested_ram))
        .unwrap_or(requested_ram);
    let preflight_profile = span(ProfileStage::CheckpointPreflight);
    let index = TensorIndex::open(model_dir)?;
    let requirements = hy4_schema::inspect_requirements(&config, &index, required_context, 0)?;
    let plan = build_hy4_resource_plan(
        &config,
        &requirements,
        effective_ram,
        required_context as u64,
        4,
    );
    drop(preflight_profile);
    if !plan.feasible {
        return Err(format!(
            "RAM plan is infeasible for {} bytes and {required_context} tokens: {}",
            effective_ram,
            plan.notes.join("; ")
        )
        .into());
    }
    let expert_slots = usize::try_from(plan.expert_slots_per_sparse_layer)
        .map_err(|_| "planned expert slots do not fit usize")?;
    let base_expert_slots = plan
        .expert_slots_per_sparse_layer
        .saturating_mul(config.sparse_layer_count() as u64);
    let spare_expert_slots = plan.expert_slots_total.saturating_sub(base_expert_slots);
    let cached_layers =
        requirements.cached_decoder_layers_for_resident_budget(plan.resident_core_bytes);
    let lm_head_mode = if requirements.caches_lm_head_for_resident_budget(plan.resident_core_bytes)
    {
        "cached"
    } else {
        "streamed"
    };
    eprintln!(
        "preflight: Hy4-preview-FP8 prompt={} tokens, context={}, RAM={}, layer resident={} (cached={}/{}, LM-head={}), KV={}, expert slots/layer={} (+{} distributed spare), suppressed LM-head rows={}, CPU workers={} (scalar correctness runtime; exact dense DSA fallback)",
        prompt_tokens.len(),
        required_context,
        human_bytes(effective_ram),
        human_bytes(plan.resident_core_bytes),
        cached_layers,
        config.num_hidden_layers,
        lm_head_mode,
        human_bytes(plan.kv_cache_bytes),
        expert_slots,
        spare_expert_slots,
        suppressed_rows,
        worker_threads()
    );
    let model = {
        let _profile = span(ProfileStage::ModelLoad);
        Hy4RuntimeModel::load(
            model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: plan.resident_core_bytes,
                expert_cache_budget_bytes: plan.expert_cache_budget_bytes,
                kv_cache_budget_bytes: plan.kv_cache_bytes,
                expert_slots_per_layer: expert_slots,
                maximum_expert_bytes: plan.maximum_expert_bytes.max(1),
                context_limit: required_context,
            },
        )?
    };
    let mut state = {
        let _profile = span(ProfileStage::StateInit);
        model.new_state()?
    };
    let mut decoder = tokenizer.streaming_decoder(false);
    let result = try_generate_with_state(
        &model,
        &prompt_tokens,
        &generation,
        &mut state,
        |token| -> Result<(), StreamError> {
            if eos.contains(&token) {
                return Ok(());
            }
            let text = decoder.push(token)?;
            output.write_all(text.as_bytes())?;
            output.flush()?;
            Ok(())
        },
    );
    let generated = match result {
        Ok(generated) => generated,
        Err(GenerationError::Callback(StreamError::Io(error)))
            if error.kind() == io::ErrorKind::BrokenPipe =>
        {
            return Ok(())
        }
        Err(error) => return Err(Box::new(error)),
    };
    output.write_all(decoder.finish().as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    let telemetry = state.expert_telemetry();
    eprintln!(
        "done: new_tokens={}, stop={}, expert hits={}, misses={}, evictions={}, expert_payload_bytes_read={}",
        generated.generated_tokens.len(),
        match generated.stop_reason {
            StopReason::Eos(token) => format!("eos:{token}"),
            StopReason::MaxNewTokens => "max_new_tokens".to_owned(),
        },
        telemetry.hits,
        telemetry.misses,
        telemetry.evictions,
        telemetry.bytes_read
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_generate_deepseek<W: Write>(
    model_dir: &Path,
    config: DeepseekV4Config,
    prompt: String,
    requested_ram: u64,
    max_new_tokens: usize,
    raw_prompt: bool,
    no_thinking: bool,
    available_ram_override: Option<u64>,
    output: &mut W,
) -> Result<(), Box<dyn Error>> {
    let prompt_profile = span(ProfileStage::TokenizerPrompt);
    let tokenizer = ByteBpeTokenizer::load(model_dir)?;
    let rendered = if raw_prompt {
        prompt
    } else {
        tokenizer.validate_chat_content(&prompt)?;
        deepseek_prompt::render_chat(
            &[deepseek_prompt::Message::new(
                deepseek_prompt::Role::User,
                prompt,
            )],
            deepseek_prompt::PromptOptions {
                thinking_mode: if no_thinking {
                    deepseek_prompt::ThinkingMode::Chat
                } else {
                    deepseek_prompt::ThinkingMode::Thinking
                },
                ..deepseek_prompt::PromptOptions::default()
            },
        )
    };
    let prompt_tokens = tokenizer.encode(&rendered)?;
    drop(prompt_profile);
    if prompt_tokens.is_empty() {
        return Err("rendered prompt encoded to zero tokens".into());
    }
    if let Some(&token) = prompt_tokens
        .iter()
        .find(|&&token| token as usize >= config.vocab_size)
    {
        return Err(format!(
            "tokenizer produced ID {token} outside model vocabulary {}",
            config.vocab_size
        )
        .into());
    }
    let eos = vec![config.eos_token_id];
    if !tokenizer.is_decodable_token_id(config.eos_token_id) {
        return Err(format!(
            "EOS token ID {} has no tokenizer entry; refusing undecodable stop semantics",
            config.eos_token_id
        )
        .into());
    }
    let decodable_mask = tokenizer.decodable_token_mask(config.vocab_size);
    let suppressed_rows = decodable_mask.iter().filter(|allowed| !**allowed).count();
    let mut generation = GenerationConfig::greedy(max_new_tokens, eos.clone());
    generation.allowed_token_mask = Some(decodable_mask);
    generation.validate()?;
    let required_context = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or("prompt + generation length overflows usize")?;
    if required_context > config.max_position_embeddings {
        return Err(format!(
            "prompt ({}) + max new tokens ({max_new_tokens}) = {required_context}, exceeding model limit {}",
            prompt_tokens.len(),
            config.max_position_embeddings
        )
        .into());
    }
    let effective_ram = available_ram_override
        .or_else(detect_available_ram)
        .map(|available| available.min(requested_ram))
        .unwrap_or(requested_ram);
    let preflight_profile = span(ProfileStage::CheckpointPreflight);
    let report = analyze_checkpoint(model_dir)?;
    let index = TensorIndex::open(model_dir)?;
    let requirements = deepseek_schema::inspect_requirements(&config, &index, required_context, 0)?;
    let plan = build_deepseek_resource_plan(
        &config,
        &report,
        &requirements,
        effective_ram,
        required_context as u64,
        4,
    );
    drop(preflight_profile);
    if !plan.feasible {
        return Err(format!(
            "RAM plan is infeasible for {} bytes and {required_context} tokens: {}",
            effective_ram,
            plan.notes.join("; ")
        )
        .into());
    }
    let expert_slots = usize::try_from(plan.expert_slots_per_sparse_layer)
        .map_err(|_| "planned expert slots do not fit usize")?;
    let (cached_layers, layer_prefetch_depth) =
        requirements.decoder_layer_pipeline_for_resident_budget(plan.resident_core_bytes);
    if cached_layers < config.num_hidden_layers {
        enable_streamed_weight_allocation_reuse();
    }
    eprintln!(
        "preflight: DeepSeek-V4 prompt={} tokens, context={}, RAM={}, resident core={} (cached layers={}/{}, read lookahead={}), KV={}, expert slots/layer={}, suppressed LM-head rows={}, CPU workers={} (correctness runtime; DSpark disabled)",
        prompt_tokens.len(),
        required_context,
        human_bytes(effective_ram),
        human_bytes(plan.resident_core_bytes),
        cached_layers,
        config.num_hidden_layers,
        layer_prefetch_depth,
        human_bytes(plan.kv_cache_bytes),
        expert_slots,
        suppressed_rows,
        worker_threads()
    );
    let model = {
        let _profile = span(ProfileStage::ModelLoad);
        DeepseekRuntimeModel::load(
            model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: plan.resident_core_bytes,
                expert_cache_budget_bytes: plan.expert_cache_budget_bytes,
                kv_cache_budget_bytes: plan.kv_cache_bytes,
                expert_slots_per_layer: expert_slots,
                maximum_expert_bytes: plan.maximum_expert_bytes.max(1),
                context_limit: required_context,
            },
        )?
    };
    let mut state = {
        let _profile = span(ProfileStage::StateInit);
        model.new_state()?
    };
    let mut decoder = tokenizer.streaming_decoder(false);
    let result = try_generate_with_state(
        &model,
        &prompt_tokens,
        &generation,
        &mut state,
        |token| -> Result<(), StreamError> {
            if eos.contains(&token) {
                return Ok(());
            }
            let text = decoder.push(token)?;
            output.write_all(text.as_bytes())?;
            output.flush()?;
            Ok(())
        },
    );
    let generated = match result {
        Ok(generated) => generated,
        Err(GenerationError::Callback(StreamError::Io(error)))
            if error.kind() == io::ErrorKind::BrokenPipe =>
        {
            return Ok(())
        }
        Err(error) => return Err(Box::new(error)),
    };
    output.write_all(decoder.finish().as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    let telemetry = state.expert_telemetry();
    eprintln!(
        "done: new_tokens={}, stop={}, expert hits={}, misses={}, evictions={}, expert_payload_bytes_read={}",
        generated.generated_tokens.len(),
        match generated.stop_reason {
            StopReason::Eos(token) => format!("eos:{token}"),
            StopReason::MaxNewTokens => "max_new_tokens".to_owned(),
        },
        telemetry.hits,
        telemetry.misses,
        telemetry.evictions,
        telemetry.bytes_read
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_generate_deepseek_v41<W: Write>(
    model_dir: &Path,
    config: DeepseekV41Config,
    prompt: String,
    requested_ram: u64,
    max_new_tokens: usize,
    raw_prompt: bool,
    no_thinking: bool,
    available_ram_override: Option<u64>,
    output: &mut W,
) -> Result<(), Box<dyn Error>> {
    let text = &config.text_config;
    let prompt_profile = span(ProfileStage::TokenizerPrompt);
    let tokenizer = ByteBpeTokenizer::load(model_dir)?;
    let rendered = if raw_prompt {
        prompt
    } else {
        tokenizer.validate_chat_content(&prompt)?;
        deepseek_v41_prompt::render_chat(
            &[deepseek_v41_prompt::Message::new(
                deepseek_v41_prompt::Role::User,
                prompt,
            )],
            deepseek_v41_prompt::PromptOptions {
                thinking_mode: if no_thinking {
                    deepseek_v41_prompt::ThinkingMode::Chat
                } else {
                    deepseek_v41_prompt::ThinkingMode::Thinking
                },
                ..deepseek_v41_prompt::PromptOptions::default()
            },
        )
    };
    let prompt_tokens = tokenizer.encode(&rendered)?;
    drop(prompt_profile);
    if prompt_tokens.is_empty() {
        return Err("rendered prompt encoded to zero tokens".into());
    }
    if let Some(&token) = prompt_tokens
        .iter()
        .find(|&&token| token as usize >= text.vocab_size)
    {
        return Err(format!(
            "tokenizer produced ID {token} outside model vocabulary {}",
            text.vocab_size
        )
        .into());
    }
    if !tokenizer.is_decodable_token_id(config.eos_token_id) {
        return Err(format!(
            "EOS token ID {} has no tokenizer entry; refusing undecodable stop semantics",
            config.eos_token_id
        )
        .into());
    }
    let eos = vec![config.eos_token_id];
    let decodable_mask = tokenizer.decodable_token_mask(text.vocab_size);
    let suppressed_rows = decodable_mask.iter().filter(|allowed| !**allowed).count();
    let mut generation = GenerationConfig::greedy(max_new_tokens, eos.clone());
    generation.allowed_token_mask = Some(decodable_mask);
    generation.validate()?;
    let required_context = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or("prompt + generation length overflows usize")?;
    if required_context > text.max_position_embeddings {
        return Err(format!(
            "prompt ({}) + max new tokens ({max_new_tokens}) = {required_context}, exceeding model limit {}",
            prompt_tokens.len(), text.max_position_embeddings
        )
        .into());
    }
    let effective_ram = available_ram_override
        .or_else(detect_available_ram)
        .map(|available| available.min(requested_ram))
        .unwrap_or(requested_ram);
    let preflight_profile = span(ProfileStage::CheckpointPreflight);
    let index = TensorIndex::open(model_dir)?;
    let zero_cache =
        deepseek_v41_schema::inspect_requirements(&config, &index, required_context, 0)?;
    let plan = build_deepseek_v41_resource_plan(
        &config,
        &zero_cache,
        effective_ram,
        required_context as u64,
    );
    drop(preflight_profile);
    if !plan.feasible {
        return Err(format!(
            "RAM plan is infeasible for {} bytes and {required_context} tokens: {}",
            effective_ram,
            plan.notes.join("; ")
        )
        .into());
    }
    let expert_slots = usize::try_from(plan.expert_slots_per_sparse_layer)
        .map_err(|_| "planned expert slots do not fit usize")?;
    eprintln!(
        "preflight: DeepSeek-V4.1 prompt={} tokens, context={}, RAM={}, streamed-layer peak={}, packed KV={}, expert slots/layer={}, suppressed LM-head rows={}, CPU workers={} (scalar base runtime; tokenwise prefill)",
        prompt_tokens.len(),
        required_context,
        human_bytes(effective_ram),
        human_bytes(plan.resident_core_bytes),
        human_bytes(plan.kv_cache_bytes),
        expert_slots,
        suppressed_rows,
        worker_threads()
    );
    let model = {
        let _profile = span(ProfileStage::ModelLoad);
        DeepseekV41RuntimeModel::load(
            model_dir,
            RuntimeLoadOptions {
                resident_budget_bytes: plan.resident_core_bytes,
                expert_cache_budget_bytes: plan.expert_cache_budget_bytes,
                kv_cache_budget_bytes: plan.kv_cache_bytes,
                expert_slots_per_layer: expert_slots,
                maximum_expert_bytes: plan.maximum_expert_bytes,
                context_limit: required_context,
            },
        )?
    };
    let mut state = {
        let _profile = span(ProfileStage::StateInit);
        model.new_state()?
    };
    let mut decoder = tokenizer.streaming_decoder(false);
    let result = try_generate_with_state(
        &model,
        &prompt_tokens,
        &generation,
        &mut state,
        |token| -> Result<(), StreamError> {
            if eos.contains(&token) {
                return Ok(());
            }
            let text = decoder.push(token)?;
            output.write_all(text.as_bytes())?;
            output.flush()?;
            Ok(())
        },
    );
    let generated = match result {
        Ok(generated) => generated,
        Err(GenerationError::Callback(StreamError::Io(error)))
            if error.kind() == io::ErrorKind::BrokenPipe =>
        {
            return Ok(())
        }
        Err(error) => return Err(Box::new(error)),
    };
    output.write_all(decoder.finish().as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    let telemetry = state.expert_telemetry();
    eprintln!(
        "done: new_tokens={}, stop={}, cache={}, expert hits={}, misses={}, evictions={}, expert_payload_bytes_read={}",
        generated.generated_tokens.len(),
        match generated.stop_reason {
            StopReason::Eos(token) => format!("eos:{token}"),
            StopReason::MaxNewTokens => "max_new_tokens".to_owned(),
        },
        human_bytes(state.cache_bytes() as u64),
        telemetry.hits,
        telemetry.misses,
        telemetry.evictions,
        telemetry.bytes_read
    );
    Ok(())
}

fn print_preflight(report: &PreflightReport) {
    match report {
        PreflightReport::Glm(requirements) => {
            print_runtime_requirements(
                "GLM-5.2",
                requirements.context_limit,
                requirements.exact_context_ceiling,
                requirements.resident_bytes,
                requirements.kv_cache_bytes,
                requirements.maximum_expert_bytes,
                requirements.transient_expert_bytes,
                requirements.expert_cache_bytes,
                requirements.expert_slots_per_layer,
            );
        }
        PreflightReport::DeepseekV4(requirements) => {
            print_runtime_requirements(
                "DeepSeek-V4",
                requirements.context_limit,
                requirements.exact_context_ceiling,
                requirements.resident_bytes,
                requirements.kv_cache_bytes,
                requirements.maximum_expert_bytes,
                requirements.transient_expert_bytes,
                requirements.expert_cache_bytes,
                requirements.expert_slots_per_layer,
            );
            println!(
                "  schema tensors      {} required / {} present ({} unexpected)",
                requirements.required_tensor_count,
                requirements.checkpoint_tensor_count,
                requirements.unexpected_tensor_count
            );
            println!(
                "  streamed rows       embedding {}, LM head {}",
                human_bytes(requirements.streamed_embedding_bytes),
                human_bytes(requirements.streamed_lm_head_bytes)
            );
            println!(
                "  architecture        {} indexed layers, {} DSpark stages",
                requirements.indexed_layer_count, requirements.dspark_stage_count
            );
        }
        PreflightReport::DeepseekV41(requirements) => {
            print_deepseek_v41_manifest(requirements);
        }
        PreflightReport::Hy4(requirements) => {
            print_runtime_requirements(
                "Hy4-preview-FP8",
                requirements.context_limit,
                requirements.exact_context_ceiling,
                requirements.resident_bytes,
                requirements.kv_cache_bytes,
                requirements.maximum_expert_bytes,
                requirements.transient_expert_bytes,
                requirements.expert_cache_bytes,
                requirements.expert_slots_per_layer,
            );
            println!(
                "  schema tensors      {} required / {} present across {} shards",
                requirements.required_tensor_count,
                requirements.checkpoint_tensor_count,
                requirements.checkpoint_shard_count
            );
            println!(
                            "  architecture        {} full IndexCache layers · MTP {} (validated, base forward excludes it)",
                            requirements.full_indexer_layer_count,
                            human_bytes(requirements.mtp_payload_bytes)
                        );
        }
        PreflightReport::KimiK3Partial(requirements) => {
            let range = match (
                requirements.validated_layers.first(),
                requirements.validated_layers.last(),
            ) {
                (Some(first), Some(last)) => format!("{first}..={last}"),
                _ => "none".to_owned(),
            };
            println!("Kimi-K3 partial checkpoint preflight");
            println!(
                "  validated layers    {} ({range})",
                requirements.validated_layers.len()
            );
            println!(
                "  validated tensors   {} decoder / {} visible total",
                requirements.validated_tensor_count, requirements.checkpoint_tensor_count
            );
            println!(
                "  non-layer tensors   {} (not asserted in partial mode)",
                requirements.non_layer_tensor_count
            );
            println!(
                "  complete shards     {} exact .safetensors files",
                requirements.checkpoint_shard_count
            );
        }
        PreflightReport::KimiK3(requirements) => {
            println!("Kimi-K3 complete multimodal checkpoint preflight");
            println!(
                "  schema tensors      {} required / {} present",
                requirements.required_tensor_count, requirements.checkpoint_tensor_count
            );
            println!(
                "  decoder tensors     {} ({} KDA, {} MLA; {} dense, {} MoE layers)",
                requirements.decoder_tensor_count,
                requirements.kda_layer_count,
                requirements.mla_layer_count,
                requirements.dense_layer_count,
                requirements.moe_layer_count
            );
            println!(
                "  multimodal tensors  {} projector + {} vision",
                requirements.projector_tensor_count, requirements.vision_tensor_count
            );
            println!(
                "  logical parameters  {}",
                requirements.logical_parameter_count
            );
            println!(
                "  checkpoint payload  {} across {} shards",
                human_bytes(requirements.checkpoint_payload_bytes),
                requirements.checkpoint_shard_count
            );
        }
        PreflightReport::Qwen38(requirements) => {
            println!("Qwen3.8 complete text-runtime preflight");
            println!(
                "  checkpoint          {} tensors across {} shards ({})",
                requirements.schema.checkpoint_tensor_count,
                requirements.schema.checkpoint_shard_count,
                human_bytes(requirements.schema.checkpoint_payload_bytes)
            );
            println!(
                "  streamed layer      {} peak",
                human_bytes(requirements.streamed_layer_bytes)
            );
            println!(
                "  recurrent + conv    {} + {}",
                human_bytes(requirements.recurrent_state_bytes),
                human_bytes(requirements.convolution_state_bytes)
            );
            println!(
                "  full-GQA KV         {} for {} tokens",
                human_bytes(requirements.kv_cache_bytes),
                requirements.context_limit
            );
            println!(
                "  execution scratch   {} (includes layer-wise prompt snapshots)",
                human_bytes(requirements.scratch_bytes)
            );
            println!(
                "  routed expert       {} each · {} cache slots/layer · {} cache",
                human_bytes(requirements.expert_bytes),
                requirements.expert_slots_per_layer,
                human_bytes(requirements.expert_cache_bytes)
            );
            println!(
                "  resident execution  {} persistent · {} miss peak (state separate)",
                human_bytes(requirements.resident_bytes),
                human_bytes(requirements.peak_resident_bytes)
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn print_runtime_requirements(
    family: &str,
    context_limit: usize,
    exact_context_ceiling: usize,
    resident_bytes: u64,
    kv_cache_bytes: u64,
    maximum_expert_bytes: u64,
    transient_expert_bytes: u64,
    expert_cache_bytes: u64,
    expert_slots_per_layer: usize,
) {
    println!("Urbilateria {family} runtime preflight (header-only)");
    println!("  context             {context_limit} / {exact_context_ceiling} exact tokens");
    println!("  resident/peak core  {}", human_bytes(resident_bytes));
    println!("  KV and state        {}", human_bytes(kv_cache_bytes));
    println!(
        "  largest expert      {}",
        human_bytes(maximum_expert_bytes)
    );
    println!(
        "  transient expert    {} minimum working set",
        human_bytes(transient_expert_bytes)
    );
    println!(
        "  expert working set  {} ({} persistent slot(s) per sparse layer)",
        human_bytes(expert_cache_bytes),
        expert_slots_per_layer
    );
    println!("  result               runtime schema is load-compatible");
}

fn print_checkpoint(report: &CheckpointReport) {
    println!("Urbilateria model X-ray");
    println!("  path               {}", report.model_path.display());
    println!(
        "  architecture       {} · {} layers ({} dense + {} MoE)",
        report.model.model_type,
        report.model.layers,
        report.model.dense_layers,
        report.model.sparse_layers
    );
    println!(
        "  routing            {} experts/layer · top-{} per token",
        report.model.routed_experts_per_layer, report.model.active_experts_per_token
    );
    println!(
        "  checkpoint         {} shards · {} tensors · {}",
        report.shard_count,
        report.tensor_count,
        human_bytes(report.file_bytes)
    );
    println!(
        "  known parameters   {} total · {} estimated active/token",
        human_count(report.known_logical_parameters),
        human_count(report.estimated_active_parameters_per_token)
    );
    println!(
        "  base primary names {}/{} present (present known tensors passed schema checks)",
        report.required_base_tensor_count - report.missing_base_tensor_count,
        report.required_base_tensor_count
    );
    println!();
    println!(
        "  {:<18} {:>9} {:>14} {:>14} {:>12}",
        "component", "tensors", "parameters", "weights", "scales"
    );
    for stats in &report.categories {
        println!(
            "  {:<18} {:>9} {:>14} {:>14} {:>12}",
            stats.category,
            stats.tensor_count,
            human_count(stats.logical_parameters),
            human_bytes(stats.payload_bytes),
            human_bytes(stats.scale_bytes)
        );
    }
    println!();
    println!("  quantization");
    for quant in &report.quantization {
        println!(
            "    {:>2}-bit  {:>7} tensors · {:>10} params · {} + {} scales",
            quant.bits_per_weight,
            quant.matrix_count,
            human_count(quant.logical_parameters),
            human_bytes(quant.payload_bytes),
            human_bytes(quant.scale_bytes)
        );
    }
    let kv = &report.kv_cache_f32;
    println!();
    println!(
        "  planned MLA KV     {}/token F32 · configured indexer adds {}/token",
        human_bytes(kv.mla_compressed_bytes_per_token),
        human_bytes(kv.configured_indexer_bytes_per_token)
    );
    println!(
        "  configured KV      {}/token total · {} at 32K",
        human_bytes(kv.compressed_bytes_per_token),
        human_bytes(kv.bytes_at_32k_context)
    );
    println!(
        "  KV reduction       {:.1}× vs conventional full per-head K/V",
        kv.compression_ratio
    );
    println!(
        "  routed experts     {} complete gate/up/down sets · {}..{} each",
        report.routed_expert_count_found,
        human_bytes(report.minimum_routed_expert_bytes),
        human_bytes(report.maximum_routed_expert_bytes)
    );
    for note in &report.format_notes {
        println!("  format             {note}");
    }
    if !report.warnings.is_empty() {
        println!();
        println!("Warnings:");
        for warning in &report.warnings {
            println!("  - {warning}");
        }
    }
}

fn print_hy4_manifest(report: &hy4_schema::Hy4Requirements) {
    println!("Hy4-preview-FP8 native checkpoint X-ray");
    println!(
        "  checkpoint          {} tensors across {} shards · {} payload ({} files)",
        report.checkpoint_tensor_count,
        report.checkpoint_shard_count,
        human_bytes(report.checkpoint_payload_bytes),
        human_bytes(report.checkpoint_file_bytes)
    );
    println!(
        "  schema              {} base + {} MTP tensors · exact name/dtype/shape/shard-map match",
        report.base_tensor_count, report.mtp_tensor_count
    );
    println!(
        "  architecture        {} layers ({} dense + {} MoE) · {} full IndexCache layers",
        report.dense_layer_count + report.moe_layer_count,
        report.dense_layer_count,
        report.moe_layer_count,
        report.full_indexer_layer_count
    );
    println!(
        "  routing             {}/{} routed experts selected per token + one shared expert",
        report.selected_experts_per_token, report.routed_experts_per_layer
    );
    println!(
        "  logical parameters  {} (quantization sidecars excluded)",
        human_count(report.logical_parameter_count)
    );
    println!(
        "  ModelOpt MXFP8      {} E4M3 matrices · {} U8 E8M0 scale tensors · 1×32 groups",
        report.fp8_matrix_count, report.scale_tensor_count
    );
    println!(
        "  streamed execution  {} peak non-expert layer · {} resident core",
        human_bytes(report.streamed_layer_bytes),
        human_bytes(report.resident_bytes)
    );
    println!(
        "  routed expert       {} decoded transient working set each",
        human_bytes(report.maximum_expert_bytes)
    );
    println!(
        "  native MTP          {} validated; excluded from the base decode path",
        human_bytes(report.mtp_payload_bytes)
    );
    println!(
        "  context boundary    checkpoint 1,048,576 · exact dense Gated-MLA fallback ≤{}",
        report.exact_context_ceiling
    );
}

fn print_deepseek_v41_manifest(report: &deepseek_v41_schema::DeepseekV41Requirements) {
    println!("DeepSeek-V4.1-Flash native checkpoint");
    println!(
        "  checkpoint          {} tensors across {} shards ({})",
        report.checkpoint_tensor_count,
        report.checkpoint_shard_count,
        human_bytes(report.checkpoint_payload_bytes)
    );
    println!(
        "  CED                 {} encoder + {} decoder layers",
        report.encoder_layer_count, report.decoder_layer_count
    );
    println!(
        "  CSA2                {} KV sources · {} index sources · global cache {} at {} tokens",
        report.kv_source_layer_count,
        report.index_source_layer_count,
        human_bytes(report.global_kv_cache_bytes),
        report.context_limit
    );
    println!(
        "  SWA cache           {} ({}-token window reconstructed per layer)",
        human_bytes(report.sliding_window_cache_bytes),
        report.approximate_replay_window
    );
    println!(
        "  streamed layer      {} peak · expert {} each",
        human_bytes(report.streamed_layer_bytes),
        human_bytes(report.maximum_expert_bytes)
    );
    println!(
        "  expert cache        {} slots/layer · {} total working set",
        report.expert_slots_per_layer,
        human_bytes(report.expert_cache_bytes)
    );
    println!(
        "  Engram              {} tables · {} lookup bytes/token",
        human_bytes(report.engram_table_bytes),
        report.engram_lookup_bytes_per_token
    );
    println!(
        "  logical parameters  backbone {} · Engram {} · vision {} · DSpark {}",
        report.backbone_logical_parameters,
        report.engram_logical_parameters,
        report.vision_logical_parameters,
        report.dspark_logical_parameters
    );
}

fn print_plan(plan: &ResourcePlan) {
    println!("Urbilateria constrained-memory plan");
    println!(
        "  RAM budget          {}",
        human_bytes(plan.ram_budget_bytes)
    );
    println!(
        "  safety reserve      {}",
        human_bytes(plan.safety_reserve_bytes)
    );
    println!(
        "  resident core       {}",
        human_bytes(plan.resident_core_bytes)
    );
    println!(
        "  KV cache            {} · {} tokens · {}-byte state · indexer={}",
        human_bytes(plan.kv_cache_bytes),
        plan.context_tokens,
        plan.kv_state_bytes,
        if plan.kv_includes_configured_indexer {
            "included"
        } else {
            "unavailable"
        }
    );
    println!("  scratch             {}", human_bytes(plan.scratch_bytes));
    println!(
        "  expert work budget  {} (persistent cache plus transient load)",
        human_bytes(plan.expert_cache_budget_bytes)
    );
    println!(
        "  largest expert      {} (used for conservative slot planning)",
        human_bytes(plan.maximum_expert_bytes)
    );
    println!(
        "  transient expert    {} minimum working set",
        human_bytes(plan.transient_expert_bytes)
    );
    println!(
        "  expert slots        {} total · {} per sparse layer · {:.1}% capacity coverage",
        plan.expert_slots_total,
        plan.expert_slots_per_sparse_layer,
        plan.expert_capacity_fraction * 100.0
    );
    println!(
        "  cold expert traffic {} per token (upper bound before cache reuse)",
        human_bytes(plan.cold_routed_bytes_per_token)
    );
    println!(
        "  context ceiling     {} tokens under this conservative budget",
        plan.maximum_context_under_budget
    );
    println!(
        "  result              {}",
        if plan.feasible {
            "feasible"
        } else {
            "NOT FEASIBLE"
        }
    );
    for note in &plan.notes {
        println!("  - {note}");
    }
}

fn print_probe(report: &ProbeReport) {
    println!("Urbilateria tensor probe");
    println!("  tensor              {}", report.tensor);
    println!(
        "  dtype / declared    {} {:?}",
        report.dtype, report.declared_shape
    );
    if let Some(shape) = report.logical_matrix_shape {
        println!("  logical [O,I]       [{}, {}]", shape[0], shape[1]);
    }
    println!(
        "  sampled             {} of {}",
        human_bytes(report.sampled_storage_bytes),
        human_bytes(
            report
                .storage_bytes
                .saturating_add(report.sidecar_storage_bytes)
        )
    );
    print_stats("values", &report.sampled_values);
    if let Some(quant) = &report.quantization {
        println!(
            "  quantization        {}-bit · {} · group {:?}",
            quant.bits_per_weight, quant.scale_layout, quant.group_size
        );
        println!(
            "  saturation          {:.4}%",
            quant.saturation_fraction * 100.0
        );
        if quant.bits_per_weight == 4 {
            println!("  code histogram      {:?}", quant.code_histogram);
        }
        if let Some(scales) = &quant.scales {
            print_stats("scales", scales);
        }
    }
    println!("  note                {}", report.sampling_note);
}

fn print_qwen38_manifest(report: &qwen38_schema::Qwen38ManifestReport) {
    println!("Qwen3.8-2.4T-A95B-FP8 manifest preflight");
    println!(
        "  schema tensors      {} required / {} indexed",
        report.required_tensor_count, report.checkpoint_tensor_count
    );
    println!(
        "  hybrid layers       {} linear attention + {} full GQA ({} base total)",
        report.linear_attention_layer_count,
        report.full_attention_layer_count,
        report.base_layer_count
    );
    println!(
        "  routed experts      {} per base layer; {} MTP layer",
        report.experts_per_layer, report.mtp_layer_count
    );
    println!(
        "  checkpoint payload  {} across {} indexed shards",
        human_bytes(report.checkpoint_payload_bytes),
        report.checkpoint_shard_count
    );
    println!(
        "  logical parameters  {} (scale sidecars excluded)",
        report.logical_parameter_count
    );
    println!("  generation stops    {:?}", report.stop_token_ids);
    println!("  runtime status      experimental text-only generation; independent real-weight logits oracle passed");
}

fn print_stats(label: &str, stats: &urbilateria::analysis::NumericStats) {
    println!(
        "  {:<18} n={} range={:?}..{:?} mean={:?} std={:?} zero={:?} nonfinite={}",
        label,
        stats.count,
        stats.minimum,
        stats.maximum,
        stats.mean,
        stats.standard_deviation,
        stats.zero_fraction,
        stats.non_finite
    );
}

fn required_path(value: Option<String>, message: &'static str) -> Result<PathBuf, Box<dyn Error>> {
    let path = PathBuf::from(value.ok_or(message)?);
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()).into());
    }
    Ok(path)
}

fn option_value<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>, Box<dyn Error>> {
    let positions: Vec<usize> = args
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value == name).then_some(index))
        .collect();
    if positions.len() > 1 {
        return Err(format!("{name} was supplied more than once").into());
    }
    let Some(position) = positions.first().copied() else {
        return Ok(None);
    };
    args.get(position + 1)
        .map(String::as_str)
        .map(Some)
        .ok_or_else(|| format!("{name} requires a value").into())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|value| value == name)
}

fn reject_duplicate_flag(args: &[String], name: &str) -> Result<(), Box<dyn Error>> {
    if args.iter().filter(|value| value.as_str() == name).count() > 1 {
        Err(format!("{name} was supplied more than once").into())
    } else {
        Ok(())
    }
}

fn reject_unknown(args: &[String], flags: &[&str]) -> Result<(), Box<dyn Error>> {
    for arg in args {
        if !flags.contains(&arg.as_str()) {
            return Err(format!("unexpected argument {arg:?}").into());
        }
    }
    Ok(())
}

fn reject_unknown_with_values(
    args: &[String],
    flags: &[&str],
    options: &[&str],
) -> Result<(), Box<dyn Error>> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if flags.contains(&arg) {
            index += 1;
        } else if options.contains(&arg) {
            if index + 1 >= args.len() {
                return Err(format!("{arg} requires a value").into());
            }
            index += 2;
        } else {
            return Err(format!("unexpected argument {arg:?}").into());
        }
    }
    Ok(())
}

fn gib_to_bytes(value: f64) -> Result<u64, Box<dyn Error>> {
    if !value.is_finite() || value <= 0.0 || value > 1_048_576.0 {
        return Err("--ram-gib must be finite and in (0, 1048576]".into());
    }
    Ok((value * 1024.0 * 1024.0 * 1024.0) as u64)
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn human_count(count: u64) -> String {
    if count >= 1_000_000_000_000 {
        format!("{:.3}T", count as f64 / 1e12)
    } else if count >= 1_000_000_000 {
        format!("{:.3}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.3}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.3}K", count as f64 / 1e3)
    } else {
        count.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const TEST_SPLIT_PATTERN: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    fn empty_model_dir() -> PathBuf {
        test_support::temp_dir("urbilateria_cli")
    }

    #[test]
    fn parallel_model_fixtures_do_not_share_directories_when_timestamps_repeat() {
        let barrier = std::sync::Barrier::new(8);
        let paths = std::thread::scope(|scope| {
            let workers = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        test_support::temp_dir_with_nonce("urbilateria_cli", 0)
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        let unique = paths.iter().collect::<std::collections::BTreeSet<_>>();
        for path in &unique {
            fs::remove_dir_all(path).unwrap();
        }
        assert_eq!(unique.len(), paths.len());
    }

    fn add_f32_tensor(
        tensors: &mut BTreeMap<String, serde_json::Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let start = payload.len();
        payload.extend(values.into_iter().flat_map(f32::to_le_bytes));
        tensors.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }

    fn tiny_tokenizer_json() -> String {
        let mut bytes: Vec<u8> = (b'!'..=b'~')
            .chain(0xa1..=0xac)
            .chain(0xae..=0xff)
            .collect();
        let mut codepoints: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
        let mut present = [false; 256];
        for &byte in &bytes {
            present[byte as usize] = true;
        }
        let mut extra = 0u32;
        for byte in 0u8..=u8::MAX {
            if !present[byte as usize] {
                bytes.push(byte);
                codepoints.push(256 + extra);
                extra += 1;
            }
        }
        let mut alphabet = ['\0'; 256];
        for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
            alphabet[byte as usize] = char::from_u32(codepoint).unwrap();
        }
        let mut vocab = serde_json::Map::new();
        for (id, character) in alphabet.into_iter().enumerate() {
            vocab.insert(character.to_string(), serde_json::Value::from(id as u32));
        }
        let mut next = 256u32;
        for token in ["he", "hel", "hell", "hello", "Ġhello"] {
            vocab.insert(token.to_owned(), serde_json::Value::from(next));
            next += 1;
        }
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": next,
                "content": "<|user|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type":"Split", "pattern":{"Regex":TEST_SPLIT_PATTERN}, "behavior":"Isolated", "invert":false},
                    {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false}
                ]
            },
            "post_processor": {"type":"ByteLevel", "add_prefix_space":true, "trim_offsets":false, "use_regex":true},
            "decoder": {"type":"ByteLevel", "add_prefix_space":true, "trim_offsets":true, "use_regex":true},
            "model": {
                "type":"BPE", "dropout":null, "unk_token":null,
                "continuing_subword_prefix":null, "end_of_word_suffix":null,
                "fuse_unk":false, "byte_fallback":false, "ignore_merges":true,
                "vocab":vocab,
                "merges":[["h","e"],["he","l"],["hel","l"],["hell","o"],["Ġ","hello"]]
            }
        })
        .to_string()
    }

    #[test]
    fn shared_text_analysis_roundtrips_unicode_and_preserves_chat_and_special_tokens() {
        let path = empty_model_dir();
        write_tiny_generate_fixture(&path);
        // Text analysis must work without reading tensor files.
        for entry in fs::read_dir(&path).unwrap() {
            let file = entry.unwrap().path();
            if file
                .extension()
                .is_some_and(|extension| extension == "safetensors")
            {
                fs::remove_file(file).unwrap();
            }
        }
        let tokenizer = ByteBpeTokenizer::load(&path).unwrap();
        let text = "hello 中文🙂\n/quit <|user|>";
        let encoded = tokenize_text(&path, text.into(), TokenizeOptions::default()).unwrap();
        assert_eq!(encoded.report.token_ids, tokenizer.encode(text).unwrap());
        assert_eq!(encoded.report.token_count, encoded.report.token_ids.len());
        let decoded = decode_tokens(&path, encoded.report.token_ids.clone(), false).unwrap();
        assert_eq!(decoded.report.text, text);
        let decoded = decode_tokens(&path, encoded.report.token_ids, true).unwrap();
        assert_eq!(decoded.report.text, "hello 中文🙂\n/quit ");
        let chat = tokenize_text(
            &path,
            "你好".into(),
            TokenizeOptions {
                chat: true,
                no_thinking: true,
            },
        )
        .unwrap();
        assert_eq!(
            chat.report.prompt,
            render_chat(
                &[ChatMessage::new(ChatRole::User, "你好")],
                ChatTemplateOptions {
                    enable_thinking: false,
                    ..ChatTemplateOptions::default()
                }
            )
        );
        assert_eq!(
            chat.report.token_ids,
            tokenizer.encode(&chat.report.prompt).unwrap()
        );
        assert!(tokenize_text(
            &path,
            "<|user|>".into(),
            TokenizeOptions {
                chat: true,
                no_thinking: false
            }
        )
        .is_err());
        assert!(tokenize_text(
            &path,
            "hello".into(),
            TokenizeOptions {
                chat: false,
                no_thinking: true
            }
        )
        .is_err());
        assert!(decode_tokens(&path, vec![u32::MAX], false).is_err());
        assert!(
            tokenize_text(&path, String::new(), TokenizeOptions::default())
                .unwrap()
                .report
                .token_ids
                .is_empty()
        );
        fs::remove_dir_all(path).unwrap();
    }

    fn write_tiny_generate_fixture(path: &Path) {
        let config = serde_json::json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "attention_bias": false,
            "mlp_bias": false,
            "rope_interleave": true,
            "indexer_rope_interleave": true,
            "tie_word_embeddings": false,
            "vocab_size": 262,
            "intermediate_size": 4,
            "moe_intermediate_size": 2,
            "first_k_dense_replace": 1,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "q_lora_rank": 2,
            "kv_lora_rank": 2,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "qk_head_dim": 4,
            "v_head_dim": 2,
            "mlp_layer_types": ["dense"],
            "rms_norm_eps": 0.00001,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "max_position_embeddings": 256,
            "eos_token_id": [67],
            "pad_token_id": 67,
            "hidden_act": "silu",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc"
        });
        fs::write(
            path.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        fs::write(path.join("tokenizer.json"), tiny_tokenizer_json()).unwrap();

        let mut tensors = BTreeMap::new();
        let mut payload = Vec::new();
        let mut embedding = vec![0.0; 262 * 4];
        embedding[65 * 4] = 1.0;
        embedding[66 * 4 + 1] = 1.0;
        add_f32_tensor(
            &mut tensors,
            &mut payload,
            "model.embed_tokens.weight",
            &[262, 4],
            embedding,
        );
        for name in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
        ] {
            add_f32_tensor(&mut tensors, &mut payload, name, &[4], vec![1.0; 4]);
        }
        for (name, shape) in [
            ("model.layers.0.self_attn.q_a_proj.weight", [2, 4]),
            ("model.layers.0.self_attn.q_b_proj.weight", [4, 2]),
            ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", [4, 4]),
            ("model.layers.0.self_attn.kv_b_proj.weight", [4, 2]),
            ("model.layers.0.self_attn.o_proj.weight", [4, 2]),
            ("model.layers.0.mlp.gate_proj.weight", [4, 4]),
            ("model.layers.0.mlp.up_proj.weight", [4, 4]),
            ("model.layers.0.mlp.down_proj.weight", [4, 4]),
        ] {
            add_f32_tensor(
                &mut tensors,
                &mut payload,
                name,
                &shape,
                vec![0.0; shape[0] * shape[1]],
            );
        }
        for name in [
            "model.layers.0.self_attn.q_a_layernorm.weight",
            "model.layers.0.self_attn.kv_a_layernorm.weight",
        ] {
            add_f32_tensor(&mut tensors, &mut payload, name, &[2], vec![1.0; 2]);
        }
        let mut lm_head = vec![0.0; 262 * 4];
        lm_head[66 * 4] = 1.0;
        lm_head[67 * 4 + 1] = 1.0;
        add_f32_tensor(
            &mut tensors,
            &mut payload,
            "lm_head.weight",
            &[262, 4],
            lm_head,
        );

        let mut header = serde_json::to_vec(&tensors).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        fs::write(path.join("model.safetensors"), file).unwrap();
    }

    #[test]
    fn generate_requires_opt_in_before_reading_model_files() {
        let dir = empty_model_dir();
        let args = [
            "--prompt".to_owned(),
            "hello".to_owned(),
            "--ram-gib".to_owned(),
            "1".to_owned(),
        ];
        let error = run_generate(&dir, &args).unwrap_err().to_string();
        assert!(error.contains("--allow-large-model"));
        assert!(error.contains("no model weights were loaded"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn generate_rejects_duplicate_authorization_flags() {
        let dir = empty_model_dir();
        let args = [
            "--allow-large-model".to_owned(),
            "--allow-large-model".to_owned(),
        ];
        let error = run_generate(&dir, &args).unwrap_err().to_string();
        assert!(error.contains("supplied more than once"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn raw_prompt_requires_a_complete_model_native_protocol() {
        for family in [
            ModelFamily::Glm52,
            ModelFamily::DeepseekV4,
            ModelFamily::DeepseekV41,
            ModelFamily::Hy4,
            ModelFamily::KimiK3,
            ModelFamily::Qwen38,
        ] {
            let error = validate_native_raw_prompt(family, "Hello")
                .unwrap_err()
                .to_string();
            assert!(error.contains("bare user text"));
            assert!(error.contains("Omit --raw-prompt"));
        }
        validate_native_raw_prompt(ModelFamily::Glm52, "[gMASK]<sop><|user|>Hello").unwrap();
        validate_native_raw_prompt(
            ModelFamily::DeepseekV4,
            "<｜begin▁of▁sentence｜><｜User｜>Hello",
        )
        .unwrap();
        validate_native_raw_prompt(
            ModelFamily::DeepseekV41,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75",
        )
        .unwrap();
        validate_native_raw_prompt(ModelFamily::Hy4, "<｜hy_start:opensource｜>Hello").unwrap();
        validate_native_raw_prompt(ModelFamily::KimiK3, "<|open|>message role=\"user\"").unwrap();
        validate_native_raw_prompt(ModelFamily::Qwen38, "<|im_start|>user\nHello").unwrap();
    }

    #[test]
    fn generate_success_path_runs_tokenizer_runtime_eos_and_stream_output() {
        let dir = empty_model_dir();
        write_tiny_generate_fixture(&dir);
        let args = [
            "--prompt".to_owned(),
            "[gMASK]<sop>A".to_owned(),
            "--ram-gib".to_owned(),
            "2".to_owned(),
            "--max-new-tokens".to_owned(),
            "3".to_owned(),
            "--allow-large-model".to_owned(),
            "--raw-prompt".to_owned(),
        ];
        let mut output = Vec::new();
        run_generate_to(&dir, &args, Some(2 * 1024 * 1024 * 1024), &mut output).unwrap();
        assert_eq!(output, b"B\n");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn generate_profile_json_is_structured_and_does_not_touch_stream_output() {
        let dir = empty_model_dir();
        write_tiny_generate_fixture(&dir);
        let profile_path = dir.join("profile.json");
        let trace_path = dir.join("profile.trace.json");
        let args = [
            "--prompt".to_owned(),
            "[gMASK]<sop>A".to_owned(),
            "--ram-gib".to_owned(),
            "2".to_owned(),
            "--max-new-tokens".to_owned(),
            "2".to_owned(),
            "--allow-large-model".to_owned(),
            "--raw-prompt".to_owned(),
            "--profile-json".to_owned(),
            profile_path.display().to_string(),
            "--profile-trace".to_owned(),
            trace_path.display().to_string(),
        ];
        let mut output = Vec::new();
        run_generate_to(&dir, &args, Some(2 * 1024 * 1024 * 1024), &mut output).unwrap();
        assert_eq!(output, b"B\n");
        let profile: serde_json::Value =
            serde_json::from_slice(&fs::read(&profile_path).unwrap()).unwrap();
        assert_eq!(profile["schema_version"], 2);
        assert!(profile["worker_threads"].as_u64().unwrap() >= 1);
        #[cfg(target_os = "linux")]
        {
            let resources = profile["resources"].as_object().unwrap();
            assert_eq!(resources["source"], "linux_procfs");
            assert!(resources["samples"].as_array().unwrap().len() >= 2);
            assert!(resources["summary"]["peak_rss_bytes"].as_u64().unwrap() > 0);
        }
        #[cfg(not(target_os = "linux"))]
        assert_eq!(profile.get("resources"), Some(&serde_json::Value::Null));
        let stages = profile["stages"].as_array().unwrap();
        assert!(stages
            .iter()
            .any(|entry| entry["stage"] == "generate.total"));
        assert!(stages
            .iter()
            .any(|entry| entry["stage"] == "kernel.matvec.f32"));
        let trace: serde_json::Value =
            serde_json::from_slice(&fs::read(&trace_path).unwrap()).unwrap();
        assert_eq!(trace["metadata"]["source"], "urbilateria");
        assert!(trace["metadata"]["recorded_span_events"].as_u64().unwrap() > 0);
        let events = trace["traceEvents"].as_array().unwrap();
        assert!(events
            .iter()
            .any(|event| event["ph"] == "X" && event["name"] == "generate.total"));
        // Resource counters require Linux procfs; span events remain available on macOS.
        #[cfg(target_os = "linux")]
        assert!(events
            .iter()
            .any(|event| event["ph"] == "C" && event["name"] == "RSS MiB"));
        #[cfg(not(target_os = "linux"))]
        assert!(events.iter().all(|event| event["ph"] != "C"));
        fs::remove_dir_all(dir).unwrap();
    }
}
