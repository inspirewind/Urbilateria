use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use urbilateria::analysis::{
    analyze_checkpoint, build_resource_plan, explain_checkpoint, inspect_checkpoint, list_tensors,
    plan_checkpoint, preflight_checkpoint, InspectionReport, ListOptions, PlanOptions,
    PreflightOptions, PreflightReport,
};
use urbilateria::models::glm::runtime::GlmRuntimeModel;
use urbilateria::GlmConfig;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("urb-inspect-{}-{nonce} 中文", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        let config = serde_json::json!({
            "model_type": "glm_moe_dsa", "hidden_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "vocab_size": 16, "intermediate_size": 12,
            "moe_intermediate_size": 4, "first_k_dense_replace": 1, "n_routed_experts": 8,
            "n_shared_experts": 1, "num_experts_per_tok": 2, "q_lora_rank": 4,
            "kv_lora_rank": 3, "qk_nope_head_dim": 2, "qk_rope_head_dim": 2,
            "qk_head_dim": 4, "v_head_dim": 3, "max_position_embeddings": 64
        });
        fs::write(path.join("config.json"), config.to_string()).unwrap();
        let mut header = serde_json::json!({
            "model.embed_tokens.weight": {"dtype": "F32", "shape": [16, 8], "data_offsets": [0, 512]}
        }).to_string().into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend(header);
        shard.resize(shard.len() + 512, 0);
        fs::write(path.join("model.safetensors"), shard).unwrap();
        Self(path)
    }

    fn complete() -> Self {
        let fixture = Self::new();
        let config_path = fixture.0.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
        config["num_hidden_layers"] = 1.into();
        fs::write(config_path, config.to_string()).unwrap();
        let tensors = [
            ("model.embed_tokens.weight", vec![16, 8]),
            ("model.norm.weight", vec![8]),
            ("lm_head.weight", vec![16, 8]),
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.post_attention_layernorm.weight", vec![8]),
            ("model.layers.0.self_attn.q_a_proj.weight", vec![4, 8]),
            ("model.layers.0.self_attn.q_b_proj.weight", vec![8, 4]),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                vec![5, 8],
            ),
            ("model.layers.0.self_attn.kv_b_proj.weight", vec![10, 3]),
            ("model.layers.0.self_attn.o_proj.weight", vec![8, 6]),
            ("model.layers.0.self_attn.q_a_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.kv_a_layernorm.weight", vec![3]),
            ("model.layers.0.mlp.gate_proj.weight", vec![12, 8]),
            ("model.layers.0.mlp.up_proj.weight", vec![12, 8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 12]),
        ];
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        for (name, shape) in tensors {
            let size = shape.iter().product::<usize>() * 4;
            header.insert(name.into(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}));
            offset += size;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend(header);
        shard.resize(shard.len() + offset, 0);
        fs::write(fixture.0.join("model.safetensors"), shard).unwrap();
        fixture
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn shared_inspection_preserves_cli_json_and_incomplete_checkpoint_warnings() {
    let fixture = Fixture::new();
    let inspection = inspect_checkpoint(&fixture.0).unwrap();
    let InspectionReport::Checkpoint(report) = &inspection.report else {
        panic!("expected GLM report")
    };
    assert_eq!(report.tensor_count, 1);
    assert_eq!(report.known_logical_parameters, 128);
    assert!(report.missing_base_tensor_count > 0);
    assert!(!report.warnings.is_empty());
    let expected = serde_json::to_value(analyze_checkpoint(&fixture.0).unwrap()).unwrap();
    assert_eq!(serde_json::to_value(&inspection.report).unwrap(), expected);
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("inspect")
        .arg(&fixture.0)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        expected
    );
}

#[test]
fn inspection_errors_remain_recoverable() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("config.json"), "not json").unwrap();
    assert!(inspect_checkpoint(&fixture.0).is_err());
}

#[test]
fn listing_filters_and_limits_results_without_requiring_a_model_config() {
    let fixture = Fixture::complete();
    fs::remove_file(fixture.0.join("config.json")).unwrap();
    let options = ListOptions {
        filter: Some("q_a".into()),
        limit: 1,
    };
    let listing = list_tensors(&fixture.0, options).unwrap();
    assert_eq!(listing.total_matches, 2);
    assert_eq!(listing.tensors.len(), 1);
    assert_eq!(
        listing.tensors[0].name,
        "model.layers.0.self_attn.q_a_layernorm.weight"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("list")
        .arg(&fixture.0)
        .args(["q_a", "--limit", "1", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        serde_json::to_value(&listing.tensors).unwrap()
    );
    let empty = list_tensors(
        &fixture.0,
        ListOptions {
            filter: Some("absent".into()),
            limit: 10,
        },
    )
    .unwrap();
    assert_eq!(empty.total_matches, 0);
    assert!(empty.tensors.is_empty());
    assert!(list_tensors(
        &fixture.0,
        ListOptions {
            limit: 0,
            ..ListOptions::default()
        }
    )
    .is_err());
}

#[test]
fn explanations_only_need_config_and_match_plain_cli_output() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.0.join("model.safetensors")).unwrap();
    let explanation = explain_checkpoint(&fixture.0).unwrap();
    assert!(explanation
        .lines
        .iter()
        .any(|line| line.contains("token → embedding [8]")));
    assert!(explanation
        .lines
        .iter()
        .any(|line| line.contains("× 2 decoder blocks")));
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("explain")
        .arg(&fixture.0)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        explanation.lines.join("\n") + "\n"
    );
    fs::write(fixture.0.join("config.json"), "{}").unwrap();
    assert!(explain_checkpoint(&fixture.0).is_err());
}

#[test]
fn plan_service_matches_existing_planner_and_cli_json() {
    let fixture = Fixture::new();
    let options = PlanOptions {
        ram_bytes: Some(2 * 1024 * 1024 * 1024),
        context: 32,
        kv_bytes: 2,
    };
    let planning = plan_checkpoint(&fixture.0, options).unwrap();
    let config = GlmConfig::load(&fixture.0).unwrap();
    let checkpoint = analyze_checkpoint(&fixture.0).unwrap();
    let expected = serde_json::to_value(build_resource_plan(
        &config,
        &checkpoint,
        options.ram_bytes.unwrap(),
        32,
        2,
    ))
    .unwrap();
    assert_eq!(serde_json::to_value(planning.report).unwrap(), expected);
    assert_eq!(planning.warnings, checkpoint.warnings);
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("plan")
        .arg(&fixture.0)
        .args([
            "--ram-gib",
            "2",
            "--context",
            "32",
            "--kv-bytes",
            "2",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        expected
    );
    let insufficient = plan_checkpoint(
        &fixture.0,
        PlanOptions {
            ram_bytes: Some(1),
            ..options
        },
    )
    .unwrap();
    assert!(!insufficient.report.feasible);
}

#[test]
fn preflight_service_preserves_runtime_options_and_cli_json() {
    let fixture = Fixture::complete();
    let options = PreflightOptions {
        context: 16,
        expert_slots: 2,
        partial: false,
    };
    let preflight = preflight_checkpoint(&fixture.0, options).unwrap();
    let PreflightReport::Glm(report) = &preflight.report else {
        panic!("expected GLM runtime preflight")
    };
    assert_eq!(report.context_limit, 16);
    assert_eq!(report.expert_slots_per_layer, 2);
    let expected =
        serde_json::to_value(GlmRuntimeModel::inspect_requirements(&fixture.0, 16, 2).unwrap())
            .unwrap();
    assert_eq!(serde_json::to_value(&preflight.report).unwrap(), expected);
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("preflight")
        .arg(&fixture.0)
        .args(["--context", "16", "--expert-slots", "2", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        expected
    );
    assert!(preflight_checkpoint(
        &fixture.0,
        PreflightOptions {
            context: 65,
            ..options
        }
    )
    .is_err());
    assert!(preflight_checkpoint(
        &fixture.0,
        PreflightOptions {
            expert_slots: 9,
            ..options
        }
    )
    .is_err());
    let error = preflight_checkpoint(
        &fixture.0,
        PreflightOptions {
            partial: true,
            ..options
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("Kimi-K3"));
    let incomplete = Fixture::new();
    assert!(preflight_checkpoint(&incomplete.0, options).is_err());
}

#[test]
#[cfg(feature = "ui")]
fn ui_help_works_in_pipes_but_interactive_ui_refuses_them() {
    let binary = env!("CARGO_BIN_EXE_urb");
    let help = Command::new(binary)
        .args(["ui", "--help"])
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("/inspect"));
    let output = Command::new(binary).arg("ui").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("interactive terminal"));
}

#[test]
#[cfg(not(feature = "ui"))]
fn lean_build_explains_how_to_enable_ui() {
    let output = Command::new(env!("CARGO_BIN_EXE_urb"))
        .arg("ui")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features ui"));
}
