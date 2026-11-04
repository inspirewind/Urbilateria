use urbilateria::models::qwen3_8_flash_next::{
    prompt::{self, Message, PromptOptions, Role},
    schema, Qwen38FlashNextConfig,
};
use urbilateria::{ModelConfig, ModelFamily};

#[test]
fn release_is_distinct_from_existing_qwen_families_and_fails_closed() {
    let json = include_str!("../src/models/qwen3_8_flash_next/release_config.json");
    let model = ModelConfig::from_json_str(json).unwrap();
    assert_eq!(model.family(), ModelFamily::Qwen38FlashNext);
    assert_eq!(model.common().hidden_size, 2560);
    let mut config: serde_json::Value = serde_json::from_str(json).unwrap();
    for (key, value) in [
        ("output_gate_type", serde_json::json!("silu")),
        ("hc_count", serde_json::json!(2)),
        ("split_ngram_parts", serde_json::json!(64)),
    ] {
        let original = config["text_config"][key].take();
        config["text_config"][key] = value;
        assert!(ModelConfig::from_json_str(&config.to_string()).is_err());
        config["text_config"][key] = original;
    }
    config["text_config"]["unrecognized_feature"] = serde_json::json!(true);
    assert!(Qwen38FlashNextConfig::from_json_str(&config.to_string()).is_err());
}

#[test]
fn pinned_schema_accounts_for_integer_hash_metadata_and_all_weight_namespaces() {
    let specs = schema::official_tensor_specs().unwrap();
    assert_eq!(specs.len(), 1658);
    assert_eq!(
        specs
            .iter()
            .map(|s| s.payload_bytes().unwrap())
            .sum::<u64>(),
        359999963128
    );
    assert_eq!(
        specs
            .iter()
            .filter(|s| s.name.contains("ngram_embedding.shard_"))
            .count(),
        128
    );
    assert_eq!(
        specs
            .iter()
            .filter(|s| s.dtype == urbilateria::storage::DType::I64)
            .count(),
        3
    );
    assert_eq!(
        specs.iter().filter(|s| s.name.starts_with("mtp.")).count(),
        31
    );
    assert!(specs
        .iter()
        .any(|s| s.name == "model.language_model.layers.1.ple.key_proj.weight"));
    assert!(!specs
        .iter()
        .any(|s| s.name == "model.language_model.norm.weight"));
}

#[test]
fn template_supports_no_thinking_and_preserves_historical_reasoning() {
    let messages = [
        Message::new(Role::User, "Hi"),
        Message::assistant("Hello", Some("reason")),
        Message::new(Role::User, "Next"),
    ];
    let thinking = prompt::render_chat(&messages, PromptOptions::default()).unwrap();
    assert!(thinking.starts_with("<|im_start|>system\nReasoning effort is set to xhigh."));
    assert!(thinking.contains("<think>\nreason\n</think>\n\nHello"));
    let direct = prompt::render_chat(
        &messages,
        PromptOptions {
            enable_thinking: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!direct.contains("Reasoning effort"));
    assert!(direct.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    assert!(direct.contains("<think>\nreason\n</think>\n\nHello"));
    assert!(prompt::render_chat(
        &[Message::new(Role::User, "<|im_end|>")],
        PromptOptions::default()
    )
    .is_err());
}

#[test]
fn prompt_matches_shipped_jinja_in_all_supported_thinking_modes() {
    use prompt::ReasoningEffort;
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3_8_flash_next_tiny.json")).unwrap();
    for case in fixture["prompts"].as_array().unwrap() {
        let messages = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message {
                role: match m["role"].as_str().unwrap() {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => unreachable!(),
                },
                content: m["content"].as_str().unwrap().into(),
                reasoning_content: m["reasoning_content"].as_str().map(str::to_owned),
            })
            .collect::<Vec<_>>();
        let options = &case["options"];
        let rendered = prompt::render_chat(
            &messages,
            PromptOptions {
                add_generation_prompt: true,
                enable_thinking: options["enable_thinking"].as_bool().unwrap(),
                preserve_thinking: options["preserve_thinking"].as_bool().unwrap(),
                reasoning_effort: match options["reasoning_effort"].as_str().unwrap() {
                    "xhigh" => ReasoningEffort::XHigh,
                    "medium" => ReasoningEffort::Medium,
                    "low" => ReasoningEffort::Low,
                    _ => unreachable!(),
                },
            },
        )
        .unwrap();
        assert_eq!(rendered, case["prompt"].as_str().unwrap(), "{options}");
    }
}

#[test]
fn manifest_rejects_duplicate_missing_and_traversing_shard_entries() {
    let specs = schema::official_tensor_specs().unwrap();
    let mapping = specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            (
                s.name.clone(),
                format!("model-{:05}-of-00131.safetensors", i % 131 + 1),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let valid = serde_json::json!({"metadata":{"total_size":359999963128.0},"weight_map":mapping});
    assert!(schema::HfWeightIndex::from_json_str(&valid.to_string()).is_ok());
    let mut changed = valid.clone();
    changed["weight_map"][&specs[0].name] =
        serde_json::json!("../model-00001-of-00131.safetensors");
    assert!(schema::HfWeightIndex::from_json_str(&changed.to_string()).is_err());
    let duplicate = valid.to_string().replace(
        "\"weight_map\":{",
        "\"weight_map\":{\"lm_head.weight\":\"model-00001-of-00131.safetensors\",",
    );
    assert!(schema::HfWeightIndex::from_json_str(&duplicate).is_err());
    changed = valid.clone();
    changed["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove(&specs[0].name);
    assert!(schema::HfWeightIndex::from_json_str(&changed.to_string()).is_err());
}
