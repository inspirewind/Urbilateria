use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use urbilateria::analysis::{
    analyze_checkpoint, build_deepseek_resource_plan, build_deepseek_v41_resource_plan,
    build_hy4_resource_plan, build_kimi_k3_resource_plan, build_resource_plan, probe_tensor,
    CheckpointReport, ProbeReport, ResourcePlan,
};
use urbilateria::execution::{configure_threads, worker_threads};
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
use urbilateria::models::kimi_k3::schema as kimi_k3_schema;
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
    DeepseekV41Config, DeepseekV4Config, GlmConfig, Hy4Config, KimiK3Config, ModelConfig,
    ModelFamily,
};

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
        "inspect" => {
            let model = required_path(args.next(), "inspect requires MODEL_DIR")?;
            let flags: Vec<String> = args.collect();
            reject_unknown(&flags, &["--json"])?;
            let json = flags.iter().any(|flag| flag == "--json");
            match ModelConfig::load(&model)? {
                ModelConfig::Qwen38(config) => {
                    let report = qwen38_schema::inspect_manifest(&config, &model)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    } else {
                        print_qwen38_manifest(&report);
                    }
                }
                ModelConfig::Hy4(config) => {
                    let index = TensorIndex::open(&model)?;
                    let report = hy4_schema::inspect_requirements(&config, &index, 1, 0)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    } else {
                        print_hy4_manifest(&report);
                    }
                }
                ModelConfig::DeepseekV41(config) => {
                    let index = TensorIndex::open(&model)?;
                    let report = deepseek_v41_schema::inspect_requirements(&config, &index, 1, 0)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    } else {
                        print_deepseek_v41_manifest(&report);
                    }
                }
                _ => {
                    let report = analyze_checkpoint(&model)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    } else {
                        print_checkpoint(&report);
                    }
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
            let config = ModelConfig::load(&model)?;
            if matches!(config, ModelConfig::Qwen38(_)) {
                return Err("the generic `urb plan` report does not represent Qwen3.8 hybrid state; use `urb preflight MODEL_DIR --context N --expert-slots N` for its exact streamed-layer, recurrent, convolution, GQA-KV, scratch, and expert-cache budgets".into());
            }
            let plan = match &config {
                ModelConfig::Glm52(config) => {
                    let report = analyze_checkpoint(&model)?;
                    build_resource_plan(config, &report, ram_gib, context, kv_bytes)
                }
                ModelConfig::DeepseekV4(config) => {
                    let report = analyze_checkpoint(&model)?;
                    let context_usize = usize::try_from(context)
                        .map_err(|_| "--context does not fit this platform")?;
                    let index = TensorIndex::open(&model)?;
                    let requirements =
                        deepseek_schema::inspect_requirements(config, &index, context_usize, 0)?;
                    build_deepseek_resource_plan(
                        config,
                        &report,
                        &requirements,
                        ram_gib,
                        context,
                        kv_bytes,
                    )
                }
                ModelConfig::DeepseekV41(config) => {
                    let context_usize = usize::try_from(context)
                        .map_err(|_| "--context does not fit this platform")?;
                    let index = TensorIndex::open(&model)?;
                    let requirements = deepseek_v41_schema::inspect_requirements(
                        config,
                        &index,
                        context_usize,
                        0,
                    )?;
                    build_deepseek_v41_resource_plan(config, &requirements, ram_gib, context)
                }
                ModelConfig::KimiK3(config) => {
                    let report = analyze_checkpoint(&model)?;
                    build_kimi_k3_resource_plan(config, &report, ram_gib, context, kv_bytes)
                }
                ModelConfig::Hy4(config) => {
                    let context_usize = usize::try_from(context)
                        .map_err(|_| "--context does not fit this platform")?;
                    let index = TensorIndex::open(&model)?;
                    let requirements =
                        hy4_schema::inspect_requirements(config, &index, context_usize, 0)?;
                    build_hy4_resource_plan(config, &requirements, ram_gib, context, kv_bytes)
                }
                ModelConfig::Qwen38(_) => unreachable!("Qwen3.8 returned before generic analysis"),
            };
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
            match ModelConfig::load(&model)? {
                ModelConfig::Glm52(_) => {
                    if partial {
                        return Err(
                            "--partial is currently specific to an in-progress Kimi-K3 transfer"
                                .into(),
                        );
                    }
                    let requirements =
                        GlmRuntimeModel::inspect_requirements(&model, context, expert_slots)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&requirements)?);
                    } else {
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
                }
                ModelConfig::DeepseekV4(config) => {
                    if partial {
                        return Err(
                            "--partial is currently specific to an in-progress Kimi-K3 transfer"
                                .into(),
                        );
                    }
                    let index = TensorIndex::open(&model)?;
                    let requirements = deepseek_schema::inspect_requirements(
                        &config,
                        &index,
                        context,
                        expert_slots,
                    )?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&requirements)?);
                    } else {
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
                }
                ModelConfig::DeepseekV41(config) => {
                    if partial {
                        return Err("DeepSeek-V4.1 preflight validates the complete native release; --partial applies only to Kimi-K3 shard transfer".into());
                    }
                    let index = TensorIndex::open(&model)?;
                    let requirements = deepseek_v41_schema::inspect_requirements(
                        &config,
                        &index,
                        context,
                        expert_slots,
                    )?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&requirements)?);
                    } else {
                        print_deepseek_v41_manifest(&requirements);
                    }
                }
                ModelConfig::Hy4(config) => {
                    if partial {
                        return Err("Hy4 preflight validates the complete 130-shard release; --partial is Kimi-K3-specific".into());
                    }
                    let index = TensorIndex::open(&model)?;
                    let requirements =
                        hy4_schema::inspect_requirements(&config, &index, context, expert_slots)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&requirements)?);
                    } else {
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
                }
                ModelConfig::KimiK3(config) => {
                    let index = TensorIndex::open(&model)?;
                    if partial {
                        let requirements =
                            kimi_k3_schema::inspect_available_layers(&config, &index)?;
                        if json {
                            println!("{}", serde_json::to_string_pretty(&requirements)?);
                        } else {
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
                                requirements.validated_tensor_count,
                                requirements.checkpoint_tensor_count
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
                    } else {
                        let requirements = kimi_k3_schema::inspect_requirements(&config, &index)?;
                        if json {
                            println!("{}", serde_json::to_string_pretty(&requirements)?);
                        } else {
                            println!("Kimi-K3 complete multimodal checkpoint preflight");
                            println!(
                                "  schema tensors      {} required / {} present",
                                requirements.required_tensor_count,
                                requirements.checkpoint_tensor_count
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
                                requirements.projector_tensor_count,
                                requirements.vision_tensor_count
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
                    }
                }
                ModelConfig::Qwen38(_) => {
                    if partial {
                        return Err("Qwen3.8 runtime preflight validates the complete official checkpoint; --partial applies only to Kimi-K3 shard transfer".into());
                    }
                    let requirements =
                        Qwen38RuntimeModel::inspect_requirements(&model, context, expert_slots)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&requirements)?);
                    } else {
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
            let config = ModelConfig::load(&model)?;
            let (prompt, token_ids) = match config {
                ModelConfig::KimiK3(_) => {
                    let tokenizer = KimiK3Tokenizer::load(&model)?;
                    if chat {
                        let messages = [kimi_k3_prompt::Message::new(
                            kimi_k3_prompt::Role::User,
                            text,
                        )];
                        let options = kimi_k3_prompt::PromptOptions {
                            thinking: !no_thinking,
                            thinking_effort: (!no_thinking)
                                .then_some(kimi_k3_prompt::ThinkingEffort::Max),
                            ..kimi_k3_prompt::PromptOptions::default()
                        };
                        let prompt = kimi_k3_prompt::render_chat(&messages, options)?;
                        let token_ids = tokenizer.encode_chat(&messages, options)?;
                        (prompt, token_ids)
                    } else {
                        let token_ids = tokenizer.encode(&text)?;
                        (text, token_ids)
                    }
                }
                config => {
                    let tokenizer = ByteBpeTokenizer::load(&model)?;
                    let prompt = if chat {
                        tokenizer.validate_chat_content(&text)?;
                        match config {
                            ModelConfig::Glm52(_) => render_chat(
                                &[ChatMessage::new(ChatRole::User, text)],
                                ChatTemplateOptions {
                                    enable_thinking: !no_thinking,
                                    ..ChatTemplateOptions::default()
                                },
                            ),
                            ModelConfig::DeepseekV4(_) => deepseek_prompt::render_chat(
                                &[deepseek_prompt::Message::new(
                                    deepseek_prompt::Role::User,
                                    text,
                                )],
                                deepseek_prompt::PromptOptions {
                                    thinking_mode: if no_thinking {
                                        deepseek_prompt::ThinkingMode::Chat
                                    } else {
                                        deepseek_prompt::ThinkingMode::Thinking
                                    },
                                    ..deepseek_prompt::PromptOptions::default()
                                },
                            ),
                            ModelConfig::DeepseekV41(_) => deepseek_v41_prompt::render_chat(
                                &[deepseek_v41_prompt::Message::new(
                                    deepseek_v41_prompt::Role::User,
                                    text,
                                )],
                                deepseek_v41_prompt::PromptOptions {
                                    thinking_mode: if no_thinking {
                                        deepseek_v41_prompt::ThinkingMode::Chat
                                    } else {
                                        deepseek_v41_prompt::ThinkingMode::Thinking
                                    },
                                    ..deepseek_v41_prompt::PromptOptions::default()
                                },
                            ),
                            ModelConfig::Hy4(_) => hy4_prompt::render_chat(
                                &[hy4_prompt::Message::new(hy4_prompt::Role::User, text)],
                                hy4_prompt::PromptOptions {
                                    enable_thinking: !no_thinking,
                                    ..hy4_prompt::PromptOptions::default()
                                },
                            ),
                            ModelConfig::KimiK3(_) => unreachable!("handled above"),
                            ModelConfig::Qwen38(_) => {
                                if no_thinking {
                                    return Err("Qwen3.8 requires thinking; --no-thinking is unsupported by the official template".into());
                                }
                                qwen38_prompt::render_chat(
                                    &[qwen38_prompt::Message::new(qwen38_prompt::Role::User, text)],
                                    qwen38_prompt::PromptOptions::default(),
                                )?
                            }
                        }
                    } else {
                        text
                    };
                    let token_ids = tokenizer.encode(&prompt)?;
                    (prompt, token_ids)
                }
            };
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
            let text = match ModelConfig::load(&model)? {
                ModelConfig::KimiK3(_) => {
                    KimiK3Tokenizer::load(&model)?.decode(&token_ids, skip_special)?
                }
                ModelConfig::Glm52(_)
                | ModelConfig::DeepseekV4(_)
                | ModelConfig::DeepseekV41(_)
                | ModelConfig::Hy4(_)
                | ModelConfig::Qwen38(_) => {
                    ByteBpeTokenizer::load(&model)?.decode(&token_ids, skip_special)?
                }
            };
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
            let index = TensorIndex::open(&model)?;
            let matches: Vec<_> = index
                .tensors()
                .filter(|tensor| {
                    filter
                        .as_deref()
                        .map(|pattern| tensor.name.contains(pattern))
                        .unwrap_or(true)
                })
                .collect();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&matches.iter().take(limit).collect::<Vec<_>>())?
                );
            } else {
                println!("{} tensor(s) match; showing at most {limit}", matches.len());
                for tensor in matches.iter().take(limit) {
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
            match ModelConfig::load(&model)? {
                ModelConfig::Glm52(config) => print_glm_explanation(&config),
                ModelConfig::DeepseekV4(config) => print_deepseek_explanation(&config),
                ModelConfig::DeepseekV41(config) => print_deepseek_v41_explanation(&config),
                ModelConfig::Hy4(config) => print_hy4_explanation(&config),
                ModelConfig::KimiK3(config) => print_kimi_k3_explanation(&config),
                ModelConfig::Qwen38(config) => print_qwen38_explanation(&config),
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
  urb inspect MODEL_DIR [--json]\n  \
  urb plan MODEL_DIR [--ram-gib N] [--context N] [--kv-bytes 2|4] [--json]\n  \
  urb preflight MODEL_DIR [--context N] [--expert-slots N] [--partial] [--json]\n  \
  urb list MODEL_DIR [FILTER] [--limit N] [--json]\n  \
  urb probe MODEL_DIR TENSOR_NAME [--samples N] [--json]\n  \
  urb tokenize MODEL_DIR TEXT [--chat] [--no-thinking] [--json]\n  \
  urb decode MODEL_DIR TOKEN_IDS [--skip-special] [--json]\n  \
  urb generate MODEL_DIR --prompt TEXT --ram-gib N --allow-large-model\n    \
      [--max-new-tokens N] [--threads N] [--profile] [--profile-json PATH]\n    \
      [--raw-prompt | --no-thinking]\n  \
  urb explain MODEL_DIR\n\n\
Commands:\n  \
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
without `--threads`, the pool uses the platform's available parallelism. Profile text goes to\n\
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
    let profile = (print_profile || profile_json.is_some())
        .then(|| ProfileSession::start_with_threads(requested_threads));
    let result = {
        let _profile = span(ProfileStage::GenerateTotal);
        run_generate_inner(model_dir, args, available_ram_override, output)
    };
    let profile_result = if let Some(profile) = profile {
        let report = profile.finish();
        if print_profile {
            eprint!("{}", report.render_text());
        }
        if let Some(path) = profile_json {
            serde_json::to_vec_pretty(&report)
                .map_err(|error| -> Box<dyn Error> { Box::new(error) })
                .and_then(|json| {
                    fs::write(&path, json).map_err(|error| {
                        format!("could not write profile {}: {error}", path.display()).into()
                    })
                })
        } else {
            Ok(())
        }
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
    eprintln!(
        "preflight: DeepSeek-V4 prompt={} tokens, context={}, RAM={}, streamed-layer peak={}, KV={}, expert slots/layer={}, suppressed LM-head rows={}, CPU workers={} (correctness runtime; DSpark disabled)",
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

fn parse_token_ids(text: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    if text.trim().is_empty() {
        return Err("TOKEN_IDS must not be empty".into());
    }
    text.split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                Err("TOKEN_IDS contains an empty item".into())
            } else {
                part.parse::<u32>()
                    .map_err(|error| format!("invalid token ID {part:?}: {error}").into())
            }
        })
        .collect()
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

fn print_glm_explanation(config: &GlmConfig) {
    let q_lora = config.q_lora_rank_value();
    println!("GLM-5.2 token path (row-major weights use Y = X · Wᵀ)");
    println!("  token → embedding [{}]", config.hidden_size);
    println!(
        "  × {} decoder blocks: RMSNorm → MLA/IndexShare → residual → RMSNorm → FFN → residual",
        config.num_hidden_layers
    );
    println!(
        "  MLA query: {} → {} → {} heads × {} dims ({} non-RoPE + {} RoPE)",
        config.hidden_size,
        q_lora,
        config.num_attention_heads,
        config.qk_head_dim,
        config.qk_nope_head_dim,
        config.qk_rope_head_dim
    );
    println!(
        "  compressed KV/token/layer: {} latent + {} shared RoPE values",
        config.kv_lora_rank, config.qk_rope_head_dim
    );
    println!(
        "  FFN: {} dense layers, then {} MoE layers; each token selects {}/{} routed experts plus {} shared",
        config.num_hidden_layers - config.sparse_layer_count(),
        config.sparse_layer_count(),
        config.num_experts_per_tok,
        config.n_routed_experts,
        config.n_shared_experts
    );
    println!(
        "  router: choose by sigmoid(logit)+bias; mix by unbiased sigmoid, normalize={}, scale={}",
        config.norm_topk_prob, config.routed_scaling_factor
    );
    println!(
        "  final RMSNorm → LM head [{} logits] → sampler",
        config.vocab_size
    );
}

fn print_deepseek_explanation(config: &DeepseekV4Config) {
    println!("DeepSeek-V4 token path (row-major weights use Y = X · Wᵀ)");
    println!(
        "  token → streamed embedding [{}] → {} Hyper-Connection streams",
        config.hidden_size, config.hc_mult
    );
    println!(
        "  × {} decoder blocks: HC/Sinkhorn → RMSNorm → MLA → HC merge → HC/Sinkhorn → RMSNorm → MoE → HC merge",
        config.num_hidden_layers
    );
    println!(
        "  MLA query: {} → Q-LoRA {} → {} heads × {} dims ({} paired-RoPE dims)",
        config.hidden_size,
        config.q_lora_rank,
        config.num_attention_heads,
        config.head_dim,
        config.qk_rope_head_dim
    );
    println!(
        "  attention memory: {}-token local window; compression schedule has {} indexed ratio-4 layers",
        config.sliding_window,
        config.indexed_layer_count()
    );
    println!(
        "  MoE: every base layer selects {}/{} routed FP4 experts plus {} shared FP8 expert; first {} layers use token-hash IDs",
        config.num_experts_per_tok,
        config.n_routed_experts,
        config.n_shared_experts,
        config.num_hash_layers
    );
    println!(
        "  router: sqrt(softplus(logit)); correction bias selects only; normalized × {}",
        config.routed_scaling_factor
    );
    println!(
        "  HC head → final RMSNorm → streamed LM head [{} logits] → sampler",
        config.vocab_size
    );
    println!(
        "  checkpoint also contains {} DSpark stage(s), validated independently of base decoding",
        config.declared_dspark_stage_count()
    );
}

fn print_deepseek_v41_explanation(config: &DeepseekV41Config) {
    let text = &config.text_config;
    println!("DeepSeek-V4.1-Flash execution model");
    println!(
        "  CED stack: {} causal-encoder + {} decoder layers; hidden {} x {} Single-Pass mHC streams",
        config.encoder_layer_count(),
        config.decoder_layer_count(),
        text.hidden_size,
        text.hc_mult
    );
    println!(
        "  CSA2: KV sources {:?}; index sources {:?}; top-{}; decoder candidates {} blocks x {} positions",
        text.kv_source_layer_ids,
        text.index_source_layer_ids,
        text.index_topk,
        text.candidate_topk_blocks,
        text.candidate_block_size
    );
    println!(
        "  MoE: {} routed experts, top-{}, plus {} shared expert; intermediate {}",
        text.n_routed_experts,
        text.num_experts_per_tok,
        text.n_shared_experts,
        text.moe_intermediate_size
    );
    println!(
        "  Engram: layers {:?}, table rows {:?}, 2-gram through {}-gram with {} heads",
        text.engram_layer_ids,
        text.engram_num_embeddings,
        text.engram_max_ngram_size,
        text.engram_n_heads
    );
    println!(
        "  checkpoint: 32x32 E4M3/E8M0 trunk, packed E2M1 experts; context {}",
        text.max_position_embeddings
    );
    println!(
        "  execution status: scalar base-text CED/CSA2/Engram generation is enabled; vision and DSpark remain schema-only"
    );
}

fn print_hy4_explanation(config: &Hy4Config) {
    println!("Hy4-preview token path (row-major weights use Y = X · Wᵀ)");
    println!(
        "  token → streamed BF16 embedding [{}] → {} identity-HC streams",
        config.hidden_size, config.hc_mult
    );
    println!(
        "  × {} blocks: iHC → RMSNorm → Gated DSA/MLA → iHC merge → iHC → RMSNorm → FFN/MoE → iHC merge",
        config.num_hidden_layers
    );
    println!(
        "  MLA: Q-LoRA {} · KV-LoRA {} · {} heads × ({} NoPE + {} RoPE) → {}-dim values",
        config.q_lora_rank,
        config.kv_lora_rank,
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.qk_rope_head_dim,
        config.v_head_dim
    );
    println!(
        "  Gated attention: elementwise {}-wide gate + {} learnable per-head sinks",
        config.num_attention_heads * config.v_head_dim,
        config.num_attention_heads
    );
    println!(
        "  DSA IndexCache: {} heads × {} dims · top-{} positions · {} full indexer layers",
        config.index_n_heads,
        config.index_head_dim,
        config.index_topk,
        config.full_indexer_layer_count()
    );
    println!(
        "  FFN: {} dense + {} MoE layers; each token selects {}/{} routed experts plus {} shared",
        config.num_hidden_layers - config.sparse_layer_count(),
        config.sparse_layer_count(),
        config.num_experts_per_tok,
        config.n_routed_experts,
        config.n_shared_experts
    );
    println!("  storage: ModelOpt MXFP8 E4M3 matrices with U8 E8M0 scales per 1×32 input block");
    println!(
        "  final iHC head → RMSNorm → streamed BF16 LM head [{} logits]; {} native MTP layer is validation-only",
        config.vocab_size, config.num_nextn_predict_layers
    );
    println!(
        "  context: checkpoint advertises {}; exact dense fallback covers ≤{} tokens until IndexCache execution lands",
        config.max_position_embeddings,
        config.exact_dense_context_ceiling()
    );
}

fn print_kimi_k3_explanation(config: &KimiK3Config) {
    let text = &config.text_config;
    println!("Kimi-K3 text token path (row-major weights use Y = X · Wᵀ)");
    println!(
        "  XTML/tiktoken token → embedding [{}] → {} decoder blocks",
        text.hidden_size, text.num_hidden_layers
    );
    println!(
        "  hybrid attention: {} KDA layers ({} heads × {} state dims, causal conv {}) + {} gated NoPE-MLA layers",
        text.kda_layer_count(),
        text.linear_attn_config.num_heads,
        text.linear_attn_config.head_dim,
        text.linear_attn_config.short_conv_kernel_size,
        text.full_attention_layer_count()
    );
    println!(
        "  MLA: Q-LoRA {} · KV-LoRA {} · {} heads × ({} NoPE + {} reserved RoPE) → gated {}-dim values",
        text.q_lora_rank,
        text.kv_lora_rank,
        text.num_attention_heads,
        text.qk_nope_head_dim,
        text.qk_rope_head_dim,
        text.v_head_dim
    );
    println!(
        "  AttnRes block size {} mixes normalized keys with raw residual values",
        text.attn_res_block_size
    );
    println!(
        "  LatentMoE: {} dense layer, then {} sparse layers; each token selects {}/{} MXFP4 routed experts plus {} shared experts",
        text.first_k_dense_replace,
        text.num_hidden_layers.saturating_sub(text.first_k_dense_replace),
        text.num_experts_per_token,
        text.num_experts,
        text.num_shared_experts
    );
    println!(
        "  routed expert: {} → latent {} → SiTU hidden {} → latent → {}, with normalized aggregate",
        text.hidden_size,
        text.routed_expert_hidden_size,
        text.moe_intermediate_size,
        text.hidden_size
    );
    println!(
        "  final RMSNorm + output AttnRes → LM head [{} logits]; generation stop token is {}",
        text.vocab_size, config.eos_token_id
    );
    println!(
        "  MoonViT-V2 has {} vision blocks; vision execution is a separate milestone from text-only generation",
        config.vision_config.vt_num_hidden_layers
    );
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

fn print_qwen38_explanation(config: &Qwen38Config) {
    let linear = config
        .layer_types
        .iter()
        .filter(|kind| kind.as_str() == "linear_attention")
        .count();
    let full = config.num_hidden_layers.saturating_sub(linear);
    println!("Qwen3.8-2.4T-A95B-FP8 token path (Y = X · Wᵀ)");
    println!(
        "  byte-BPE/ChatML token → embedding [{}] → {} hybrid decoder blocks",
        config.hidden_size, config.num_hidden_layers
    );
    println!(
        "  hybrid attention: {linear} Gated DeltaNet layers ({} key heads, {} value heads, dim {}, conv {}) + {full} gated GQA layers",
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_conv_kernel_dim
    );
    println!(
        "  full GQA: {} query heads / {} KV heads × {} dims; RoPE covers {:.0}% of each head",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.partial_rotary_factor * 100.0
    );
    println!(
        "  MoE: every layer selects {}/{} routed E4M3 experts plus one BF16 shared expert (intermediate {})",
        config.num_experts_per_tok,
        config.num_experts,
        config.moe_intermediate_size
    );
    println!(
        "  routed storage: 128×128 E4M3 blocks with BF16 weight_scale_inv; trunk/router/shared expert stay BF16"
    );
    println!(
        "  final RMSNorm → independent LM head [{} logits]; native context {}",
        config.vocab_size, config.max_position_embeddings
    );
    println!(
        "  runtime boundary: full scalar forward, state, streaming, and preflight are implemented; public generate awaits real-weight and independent-logits gates"
    );
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

fn detect_available_ram() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let host_available = kib.checked_mul(1024)?;
    Some(
        cgroup_memory_remaining()
            .map(|remaining| remaining.min(host_available))
            .unwrap_or(host_available),
    )
}

fn cgroup_memory_remaining() -> Option<u64> {
    let pairs = [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ];
    for (limit_path, usage_path) in pairs {
        let Ok(limit_text) = fs::read_to_string(limit_path) else {
            continue;
        };
        if limit_text.trim() == "max" {
            continue;
        }
        let Ok(limit) = limit_text.trim().parse::<u64>() else {
            continue;
        };
        let Ok(usage_text) = fs::read_to_string(usage_path) else {
            continue;
        };
        let Ok(usage) = usage_text.trim().parse::<u64>() else {
            continue;
        };
        return Some(limit.saturating_sub(usage));
    }
    None
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
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEST_SPLIT_PATTERN: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    fn empty_model_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("urbilateria_cli_{}_{}", std::process::id(), nonce));
        fs::create_dir_all(&path).unwrap();
        path
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
        ];
        let mut output = Vec::new();
        run_generate_to(&dir, &args, Some(2 * 1024 * 1024 * 1024), &mut output).unwrap();
        assert_eq!(output, b"B\n");
        let profile: serde_json::Value =
            serde_json::from_slice(&fs::read(&profile_path).unwrap()).unwrap();
        assert_eq!(profile["schema_version"], 1);
        assert!(profile["worker_threads"].as_u64().unwrap() >= 1);
        let stages = profile["stages"].as_array().unwrap();
        assert!(stages
            .iter()
            .any(|entry| entry["stage"] == "generate.total"));
        assert!(stages
            .iter()
            .any(|entry| entry["stage"] == "kernel.matvec.f32"));
        fs::remove_dir_all(dir).unwrap();
    }
}
