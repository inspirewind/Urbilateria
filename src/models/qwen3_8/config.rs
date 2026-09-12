//! Strict configuration for the released Qwen3.8-2.4T-A95B-FP8 checkpoint.
//!
//! Qwen3.8 currently advertises the Transformers architecture name
//! `Qwen3_5MoeForCausalLM`.  Detection must use the release's actual `model_type`,
//! `qwen3_5_moe_text`, rather than guessing a future `qwen3_8` identifier.

use crate::config::ConfigError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub const MODEL_TYPE: &str = "qwen3_5_moe_text";
pub const ARCHITECTURE: &str = "Qwen3_5MoeForCausalLM";
pub const BOS_TOKEN_ID: u32 = 248_044;
pub const IM_END_TOKEN_ID: u32 = 248_046;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RopeParameters {
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub activation_scheme: String,
    pub weight_per_tensor: bool,
    pub act_per_tensor: bool,
    pub weight_block_size: [usize; 2],
    pub modules_to_not_convert: Vec<String>,
}

/// The exact text-model config published with Qwen3.8-2.4T-A95B-FP8.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen38Config {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attn_output_gate: bool,
    pub bos_token_id: u32,
    pub dtype: String,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub mamba_ssm_dtype: String,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub moe_intermediate_size: usize,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_gate_type: String,
    pub output_router_logits: bool,
    pub pad_token_id: Option<u32>,
    pub partial_rotary_factor: f64,
    pub quantization_config: QuantizationConfig,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub router_aux_loss_coef: f64,
    pub shared_expert_intermediate_size: usize,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

impl Qwen38Config {
    pub fn load(model_dir: &Path) -> Result<Self, ConfigError> {
        load_json(&model_dir.join("config.json"), Self::validate)
    }

    pub fn from_json_str(json: &str) -> Result<Self, ConfigError> {
        let config: Self = serde_json::from_str(json)
            .map_err(|source| ConfigError::Json { path: None, source })?;
        config.validate()?;
        Ok(config)
    }

    /// Rejects compatible-looking variants: phase one is tied to one immutable release ABI.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != MODEL_TYPE {
            return invalid(format!(
                "model_type={:?}; Qwen3.8 uses {MODEL_TYPE:?}",
                self.model_type
            ));
        }
        if self.architectures.as_slice() != [ARCHITECTURE] {
            return invalid(format!(
                "architectures={:?}; expected [{ARCHITECTURE:?}]",
                self.architectures
            ));
        }

        for (name, actual, expected) in [
            (
                "bos_token_id",
                self.bos_token_id as usize,
                BOS_TOKEN_ID as usize,
            ),
            (
                "eos_token_id",
                self.eos_token_id as usize,
                BOS_TOKEN_ID as usize,
            ),
            ("full_attention_interval", self.full_attention_interval, 4),
            ("head_dim", self.head_dim, 256),
            ("hidden_size", self.hidden_size, 8_192),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim, 4),
            ("linear_key_head_dim", self.linear_key_head_dim, 128),
            ("linear_num_key_heads", self.linear_num_key_heads, 16),
            ("linear_num_value_heads", self.linear_num_value_heads, 128),
            ("linear_value_head_dim", self.linear_value_head_dim, 128),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                262_144,
            ),
            ("moe_intermediate_size", self.moe_intermediate_size, 2_048),
            ("mtp_num_hidden_layers", self.mtp_num_hidden_layers, 1),
            ("num_attention_heads", self.num_attention_heads, 64),
            ("num_experts", self.num_experts, 512),
            ("num_experts_per_tok", self.num_experts_per_tok, 10),
            ("num_hidden_layers", self.num_hidden_layers, 92),
            ("num_key_value_heads", self.num_key_value_heads, 4),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
                2_048,
            ),
            ("vocab_size", self.vocab_size, 248_320),
        ] {
            if actual != expected {
                return invalid(format!(
                    "released Qwen3.8 requires {name}={expected}, got {actual}"
                ));
            }
        }

        for (name, actual, expected) in [
            ("attention_dropout", self.attention_dropout, 0.0),
            ("initializer_range", self.initializer_range, 0.02),
            ("partial_rotary_factor", self.partial_rotary_factor, 0.25),
            ("rms_norm_eps", self.rms_norm_eps, 1e-6),
            (
                "rope_parameters.partial_rotary_factor",
                self.rope_parameters.partial_rotary_factor,
                0.25,
            ),
            (
                "rope_parameters.rope_theta",
                self.rope_parameters.rope_theta,
                10_000_000.0,
            ),
            ("router_aux_loss_coef", self.router_aux_loss_coef, 0.001),
        ] {
            if !actual.is_finite() || actual != expected {
                return invalid(format!(
                    "released Qwen3.8 requires {name}={expected}, got {actual}"
                ));
            }
        }

        if self.attention_bias
            || !self.attn_output_gate
            || self.mtp_use_dedicated_embeddings
            || self.output_router_logits
            || self.pad_token_id.is_some()
            || self.tie_word_embeddings
            || !self.use_cache
        {
            return invalid(
                "attention/gating/MTP/token/cache flags differ from the released Qwen3.8 config",
            );
        }
        if self.dtype != "bfloat16"
            || self.hidden_act != "silu"
            || self.mamba_ssm_dtype != "float32"
            || self.output_gate_type != "swish"
            || self.rope_parameters.rope_type != "default"
            || self.transformers_version != "4.57.3"
        {
            return invalid(
                "dtype, activation, RoPE, or Transformers metadata differs from the release",
            );
        }

        let expected_layers = expected_layer_types();
        if self.layer_types != expected_layers {
            return invalid(
                "layer_types must be the released 92-layer [linear, linear, linear, full] map",
            );
        }

        let quant = &self.quantization_config;
        if quant.quant_method != "fp8"
            || quant.activation_scheme != "dynamic"
            || quant.weight_per_tensor
            || quant.act_per_tensor
            || quant.weight_block_size != [128, 128]
        {
            return invalid("quantization_config must use the released dynamic block-FP8 layout");
        }
        validate_unquantized_modules(&quant.modules_to_not_convert)?;
        Ok(())
    }

    pub fn is_full_attention_layer(&self, layer: usize) -> Option<bool> {
        self.layer_types
            .get(layer)
            .map(|kind| kind == "full_attention")
    }
}

/// The exact `generation_config.json` shipped beside the checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen38GenerationConfig {
    pub bos_token_id: u32,
    pub do_sample: bool,
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: u32,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
}

impl Qwen38GenerationConfig {
    pub fn load(model_dir: &Path) -> Result<Self, ConfigError> {
        load_json(&model_dir.join("generation_config.json"), Self::validate)
    }

    pub fn from_json_str(json: &str) -> Result<Self, ConfigError> {
        let config: Self = serde_json::from_str(json)
            .map_err(|source| ConfigError::Json { path: None, source })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.bos_token_id != BOS_TOKEN_ID
            || self.pad_token_id != BOS_TOKEN_ID
            || self.eos_token_id != [IM_END_TOKEN_ID, BOS_TOKEN_ID]
            || !self.do_sample
            || self.temperature != 1.0
            || self.top_k != 20
            || self.top_p != 0.95
        {
            return invalid("generation_config.json differs from the Qwen3.8 release defaults");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen38ReleaseConfig {
    pub model: Qwen38Config,
    pub generation: Qwen38GenerationConfig,
}

impl Qwen38ReleaseConfig {
    pub fn load(model_dir: &Path) -> Result<Self, ConfigError> {
        let release = Self {
            model: Qwen38Config::load(model_dir)?,
            generation: Qwen38GenerationConfig::load(model_dir)?,
        };
        release.validate()?;
        Ok(release)
    }

    pub fn from_json_strs(config_json: &str, generation_json: &str) -> Result<Self, ConfigError> {
        let release = Self {
            model: Qwen38Config::from_json_str(config_json)?,
            generation: Qwen38GenerationConfig::from_json_str(generation_json)?,
        };
        release.validate()?;
        Ok(release)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.model.validate()?;
        self.generation.validate()?;
        if self.model.bos_token_id != self.generation.bos_token_id
            || self.model.eos_token_id != self.generation.pad_token_id
        {
            return invalid("config.json and generation_config.json token IDs disagree");
        }
        Ok(())
    }
}

fn load_json<T>(path: &Path, validate: fn(&T) -> Result<(), ConfigError>) -> Result<T, ConfigError>
where
    T: for<'de> Deserialize<'de>,
{
    let json = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let value = serde_json::from_str(&json).map_err(|source| ConfigError::Json {
        path: Some(path.to_path_buf()),
        source,
    })?;
    validate(&value)?;
    Ok(value)
}

fn expected_layer_types() -> Vec<String> {
    (0..92)
        .map(|layer| {
            if (layer + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
            .to_owned()
        })
        .collect()
}

fn expected_unquantized_modules() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert("lm_head".to_owned());
    names.insert("model.embed_tokens".to_owned());
    for layer in 0..92 {
        let root = format!("model.layers.{layer}");
        if (layer + 1) % 4 == 0 {
            for projection in ["k_proj", "o_proj", "q_proj", "v_proj"] {
                names.insert(format!("{root}.self_attn.{projection}"));
            }
        } else {
            for module in [
                "conv1d",
                "in_proj_a",
                "in_proj_b",
                "in_proj_qkv",
                "in_proj_z",
                "out_proj",
            ] {
                names.insert(format!("{root}.linear_attn.{module}"));
            }
        }
        for module in [
            "gate",
            "shared_expert.down_proj",
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert_gate",
        ] {
            names.insert(format!("{root}.mlp.{module}"));
        }
    }
    for module in [
        "mtp.fc",
        "mtp.layers.0.mlp.gate",
        "mtp.layers.0.mlp.shared_expert.down_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj",
        "mtp.layers.0.mlp.shared_expert.up_proj",
        "mtp.layers.0.mlp.shared_expert_gate",
        "mtp.layers.0.self_attn.k_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.v_proj",
    ] {
        names.insert(module.to_owned());
    }
    names
}

fn validate_unquantized_modules(actual: &[String]) -> Result<(), ConfigError> {
    let actual_set: BTreeSet<_> = actual.iter().cloned().collect();
    if actual_set.len() != actual.len() {
        return invalid("quantization_config.modules_to_not_convert contains a duplicate");
    }
    let expected = expected_unquantized_modules();
    if actual_set != expected {
        let missing = expected.difference(&actual_set).next();
        let unexpected = actual_set.difference(&expected).next();
        return invalid(format!(
            "modules_to_not_convert differs from the 978-entry release set (first missing={missing:?}, first unexpected={unexpected:?})"
        ));
    }
    Ok(())
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(reason.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn official_config_value() -> Value {
        let mut value = json!({
            "architectures": [ARCHITECTURE], "attention_bias": false,
            "attention_dropout": 0, "attn_output_gate": true,
            "bos_token_id": BOS_TOKEN_ID, "dtype": "bfloat16",
            "eos_token_id": BOS_TOKEN_ID, "full_attention_interval": 4,
            "head_dim": 256, "hidden_act": "silu", "hidden_size": 8192,
            "initializer_range": 0.02, "layer_types": expected_layer_types(),
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 128,
            "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144, "model_type": MODEL_TYPE,
            "moe_intermediate_size": 2048, "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false, "num_attention_heads": 64,
        });
        let tail = json!({
            "num_experts": 512, "num_experts_per_tok": 10, "num_hidden_layers": 92,
            "num_key_value_heads": 4, "output_gate_type": "swish",
            "output_router_logits": false, "pad_token_id": null,
            "partial_rotary_factor": 0.25,
            "quantization_config": {
                "quant_method": "fp8", "activation_scheme": "dynamic",
                "weight_per_tensor": false, "act_per_tensor": false,
                "weight_block_size": [128, 128],
                "modules_to_not_convert": expected_unquantized_modules(),
            },
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                "rope_type": "default",
            },
            "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 2048,
            "tie_word_embeddings": false, "transformers_version": "4.57.3",
            "use_cache": true, "vocab_size": 248320,
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(tail.as_object().unwrap().clone());
        value
    }

    fn official_generation_json() -> String {
        json!({
            "bos_token_id": BOS_TOKEN_ID,
            "do_sample": true,
            "eos_token_id": [IM_END_TOKEN_ID, BOS_TOKEN_ID],
            "pad_token_id": BOS_TOKEN_ID,
            "temperature": 1,
            "top_k": 20,
            "top_p": 0.95,
        })
        .to_string()
    }

    #[test]
    fn accepts_the_exact_release_configs() {
        let model_json = official_config_value().to_string();
        let release =
            Qwen38ReleaseConfig::from_json_strs(&model_json, &official_generation_json()).unwrap();
        assert_eq!(release.model.model_type, MODEL_TYPE);
        assert_eq!(release.model.layer_types.len(), 92);
        assert_eq!(
            release
                .model
                .quantization_config
                .modules_to_not_convert
                .len(),
            978
        );
        assert_eq!(release.model.is_full_attention_layer(3), Some(true));
        assert_eq!(release.model.is_full_attention_layer(4), Some(false));
    }

    #[test]
    fn rejects_a_guessed_qwen38_model_type() {
        let mut value = official_config_value();
        value["model_type"] = json!("qwen3_8");
        let error = Qwen38Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("qwen3_5_moe_text"));
    }

    #[test]
    fn rejects_layer_or_quantization_drift() {
        let mut layers = official_config_value();
        layers["layer_types"][3] = json!("linear_attention");
        assert!(Qwen38Config::from_json_str(&layers.to_string()).is_err());

        let mut modules = official_config_value();
        modules["quantization_config"]["modules_to_not_convert"][0] = json!("bad.module");
        let error = Qwen38Config::from_json_str(&modules.to_string()).unwrap_err();
        assert!(error.to_string().contains("978-entry release set"));
    }

    #[test]
    fn generation_defaults_are_release_specific() {
        let generation =
            Qwen38GenerationConfig::from_json_str(&official_generation_json()).unwrap();
        assert_eq!(generation.eos_token_id, [IM_END_TOKEN_ID, BOS_TOKEN_ID]);

        let mut value: Value = serde_json::from_str(&official_generation_json()).unwrap();
        value["top_k"] = json!(0);
        assert!(Qwen38GenerationConfig::from_json_str(&value.to_string()).is_err());
    }
}
