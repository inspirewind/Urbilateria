use urbilateria::models::qwen3_6::{prompt::*, schema::*, Qwen36Config};
use urbilateria::{ModelConfig, ModelFamily};
const CONFIG: &str = include_str!("../src/models/qwen3_6/release_config.json");
#[test]
fn release_detection_and_strict_geometry() {
    let model = ModelConfig::from_json_str(CONFIG).unwrap();
    assert_eq!(model.family(), ModelFamily::Qwen36);
    assert_eq!(model.common().hidden_size, 2048);
    assert_eq!(model.common().num_hidden_layers, 40);
    for (field, bad) in [
        ("num_experts", serde_json::json!(128)),
        ("attn_output_gate", serde_json::json!(false)),
        ("dtype", serde_json::json!("float16")),
    ] {
        let mut value: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        value["text_config"][field] = bad;
        assert!(Qwen36Config::from_json_str(&value.to_string()).is_err());
    }
}
#[test]
fn schema_accounts_for_packed_experts_and_auxiliary_weights() {
    let specs = official_tensor_specs().unwrap();
    assert_eq!(specs.len(), 1045);
    let bytes: u64 = specs.iter().map(|s| s.payload_bytes().unwrap()).sum();
    assert_eq!(bytes, 71_903_645_408);
    let gate_up = specs
        .iter()
        .find(|s| s.name == "model.language_model.layers.39.mlp.experts.gate_up_proj")
        .unwrap();
    assert_eq!(gate_up.shape, [256, 1024, 2048]);
    assert_eq!(
        specs.iter().filter(|s| s.name.starts_with("mtp.")).count(),
        MTP_TENSOR_COUNT
    );
    assert!(specs.iter().any(|s| s.name.starts_with("model.visual.")));
}
#[test]
fn prompt_supports_both_thinking_modes_without_reasoning_instructions() {
    let messages = [Message::new(Role::User, " Hello ")];
    assert_eq!(
        render_chat(&messages, PromptOptions::default()).unwrap(),
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    assert_eq!(
        render_chat(
            &messages,
            PromptOptions {
                enable_thinking: false,
                ..Default::default()
            }
        )
        .unwrap(),
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let messages = [
        Message::new(Role::User, "first"),
        Message::assistant("answer", Some("private reasoning")),
        Message::new(Role::User, "next"),
    ];
    assert!(!render_chat(&messages, PromptOptions::default())
        .unwrap()
        .contains("private reasoning"));
    assert!(render_chat(
        &[Message::new(Role::User, "<|image_pad|>")],
        PromptOptions::default()
    )
    .is_err());
}

#[test]
fn manifest_rejects_missing_names_wrong_shards_and_fractional_size() {
    let specs = official_tensor_specs().unwrap();
    let mut map = serde_json::Map::new();
    for (i, spec) in specs.iter().enumerate() {
        map.insert(
            spec.name.clone(),
            serde_json::json!(format!("model-{:05}-of-00026.safetensors", i % 26 + 1)),
        );
    }
    let value = serde_json::json!({"metadata":{"total_size":71903645408.0},"weight_map":map});
    HfWeightIndex::from_json_str(&value.to_string()).unwrap();
    let mut missing = value.clone();
    missing["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove("lm_head.weight");
    assert!(HfWeightIndex::from_json_str(&missing.to_string()).is_err());
    let mut wrong = value.clone();
    wrong["weight_map"]["lm_head.weight"] =
        serde_json::json!("../model-00001-of-00026.safetensors");
    assert!(HfWeightIndex::from_json_str(&wrong.to_string()).is_err());
    let mut fractional = value;
    fractional["metadata"]["total_size"] = serde_json::json!(71903645408.5);
    assert!(HfWeightIndex::from_json_str(&fractional.to_string()).is_err());
    assert!(HfWeightIndex::from_json_str(r#"{"metadata":{"total_size":71903645408.0},"weight_map":{"duplicate":"a","duplicate":"b"}}"#).unwrap_err().to_string().contains("duplicate"));
}

#[test]
fn prompt_matches_the_shipped_jinja_in_both_modes() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3_6_prompts.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let messages: Vec<Message> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let role = match m["role"].as_str().unwrap() {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => unreachable!(),
                };
                if role == Role::Assistant {
                    Message::assistant(
                        m["content"].as_str().unwrap(),
                        m["reasoning_content"].as_str(),
                    )
                } else {
                    Message::new(role, m["content"].as_str().unwrap())
                }
            })
            .collect();
        assert_eq!(
            render_chat(
                &messages,
                PromptOptions {
                    enable_thinking: case["thinking"].as_bool().unwrap(),
                    ..Default::default()
                }
            )
            .unwrap(),
            case["rendered"].as_str().unwrap()
        );
    }
}
