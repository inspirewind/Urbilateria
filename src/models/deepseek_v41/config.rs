//! Strict parsing and validation of the official DeepSeek-V4.1-Flash configuration.

use crate::config::ConfigError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub const MODEL_TYPE: &str = "deepseek_v41";
pub const TEXT_MODEL_TYPE: &str = "deepseek_v41_text";
pub const VISION_MODEL_TYPE: &str = "deepseek_v41_vision";
pub const ARCHITECTURE: &str = "DeepseekV41ForCausalLM";
pub const ENCODER_LAYER_COUNT: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV41QuantizationConfig {
    pub quant_method: String,
    pub activation_scheme: String,
    pub weight_block_size: [usize; 2],
    pub scale_fmt: String,
    pub expert_dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: usize,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV41TextConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hidden_act: String,
    pub swiglu_limit: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f64,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer_id: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<usize>,
    pub engram_max_ngram_size: usize,
    pub engram_vocab_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub engram_pad_token_id: u32,
    pub engram_compressed_vocab_size: usize,
    pub num_nextn_predict_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: u32,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_num_experts_per_tok: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV41VisionConfig {
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    #[serde(default)]
    pub max_wh_ratio: Option<f64>,
}

/// Fields that determine the native V4.1 checkpoint ABI and inference graph.
///
/// Generic Transformers annotations remain forward-compatible, while every field affecting tensor
/// geometry, CED/CSA2 ownership, routing, Engram addressing, or cache precision is typed here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV41Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub dtype: String,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub image_token_id: u32,
    pub quantization_config: DeepseekV41QuantizationConfig,
    pub text_config: DeepseekV41TextConfig,
    pub vision_config: DeepseekV41VisionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    SlidingWindowOnly,
    Full,
    Reindex,
    Reuse,
}

impl DeepseekV41Config {
    pub fn load(model_dir: &Path) -> Result<Self, ConfigError> {
        let path = model_dir.join("config.json");
        let json = fs::read_to_string(&path).map_err(|source| ConfigError::Read {
            path: path.clone(),
            source,
        })?;
        let config: Self = serde_json::from_str(&json).map_err(|source| ConfigError::Json {
            path: Some(path),
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_json_str(json: &str) -> Result<Self, ConfigError> {
        let config: Self = serde_json::from_str(json)
            .map_err(|source| ConfigError::Json { path: None, source })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != MODEL_TYPE
            || self.text_config.model_type != TEXT_MODEL_TYPE
            || self.vision_config.model_type != VISION_MODEL_TYPE
        {
            return invalid("top-level, text, or vision model_type is not DeepSeek-V4.1");
        }
        if self.architectures != [ARCHITECTURE.to_owned()] {
            return invalid(format!(
                "architectures={:?}; expected [{ARCHITECTURE:?}]",
                self.architectures
            ));
        }
        if self.dtype != "bfloat16" {
            return invalid(format!("dtype={:?}; expected bfloat16", self.dtype));
        }
        self.validate_tokens()?;
        self.quantization_config.validate()?;
        self.text_config.validate()?;
        self.vision_config.validate()?;
        Ok(())
    }

    fn validate_tokens(&self) -> Result<(), ConfigError> {
        let vocab = self.text_config.vocab_size;
        for (name, id) in [
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
            ("image_token_id", self.image_token_id),
        ] {
            if id as usize >= vocab {
                return invalid(format!("{name}={id} is outside vocab_size={vocab}"));
            }
        }
        if self.engram_pad_token_id() != self.pad_token_id {
            return invalid("engram_pad_token_id must match the top-level pad token");
        }
        Ok(())
    }

    pub fn engram_pad_token_id(&self) -> u32 {
        self.text_config.engram_pad_token_id
    }

    pub fn encoder_layer_count(&self) -> usize {
        ENCODER_LAYER_COUNT
    }

    pub fn decoder_layer_count(&self) -> usize {
        self.text_config.num_hidden_layers - ENCODER_LAYER_COUNT
    }

    pub fn attention_mode(&self, layer: usize) -> Option<AttentionMode> {
        if layer >= self.text_config.num_hidden_layers {
            return None;
        }
        if self.text_config.compress_ratios[layer] == 0 {
            return Some(AttentionMode::SlidingWindowOnly);
        }
        if self.text_config.kv_source_layer_ids.contains(&layer) {
            return Some(AttentionMode::Full);
        }
        if self.text_config.index_source_layer_ids.contains(&layer) {
            return Some(AttentionMode::Reindex);
        }
        Some(AttentionMode::Reuse)
    }

    /// Returns the most recent global-KV owner available to a layer.
    pub fn kv_source_for(&self, layer: usize) -> Option<usize> {
        (layer < self.text_config.num_hidden_layers)
            .then(|| {
                self.text_config
                    .kv_source_layer_ids
                    .iter()
                    .rev()
                    .copied()
                    .find(|&source| source <= layer)
            })
            .flatten()
    }

    /// Returns the most recent sparse-index selection owner available to a layer.
    pub fn index_source_for(&self, layer: usize) -> Option<usize> {
        (layer < self.text_config.num_hidden_layers)
            .then(|| {
                self.text_config
                    .index_source_layer_ids
                    .iter()
                    .rev()
                    .copied()
                    .find(|&source| source <= layer)
            })
            .flatten()
    }
}

impl DeepseekV41QuantizationConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.quant_method != "fp8"
            || self.activation_scheme != "dynamic"
            || self.weight_block_size != [32, 32]
            || self.scale_fmt != "ue8m0"
            || self.expert_dtype != "fp4"
        {
            return invalid(
                "checkpoint quantization must be 32x32 E4M3/E8M0 with packed FP4 experts",
            );
        }
        Ok(())
    }
}

impl DeepseekV41TextConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        for (name, value, maximum) in [
            ("vocab_size", self.vocab_size, 1 << 24),
            ("hidden_size", self.hidden_size, 1 << 20),
            ("moe_intermediate_size", self.moe_intermediate_size, 1 << 24),
            ("num_hidden_layers", self.num_hidden_layers, 256),
            ("num_attention_heads", self.num_attention_heads, 4096),
            ("head_dim", self.head_dim, 1 << 16),
            ("qk_rope_head_dim", self.qk_rope_head_dim, 1 << 16),
            ("q_lora_rank", self.q_lora_rank, 1 << 20),
            ("o_lora_rank", self.o_lora_rank, 1 << 20),
            ("o_groups", self.o_groups, 4096),
            ("n_routed_experts", self.n_routed_experts, 1 << 16),
            ("n_shared_experts", self.n_shared_experts, 1024),
            ("num_experts_per_tok", self.num_experts_per_tok, 1 << 16),
            ("sliding_window", self.sliding_window, 1 << 20),
            ("index_n_heads", self.index_n_heads, 4096),
            ("index_head_dim", self.index_head_dim, 1 << 16),
            ("index_topk", self.index_topk, 1 << 24),
            ("hc_mult", self.hc_mult, 64),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters, 1024),
            ("engram_max_ngram_size", self.engram_max_ngram_size, 32),
            ("engram_n_heads", self.engram_n_heads, 1024),
            ("engram_head_dim", self.engram_head_dim, 1 << 16),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                1 << 30,
            ),
        ] {
            if value == 0 || value > maximum {
                return invalid(format!(
                    "text_config.{name}={value} is outside 1..={maximum}"
                ));
            }
        }
        if self.num_hidden_layers != 40 || ENCODER_LAYER_COUNT * 2 != self.num_hidden_layers {
            return invalid("the supported CED release requires 20 encoder and 20 decoder layers");
        }
        if self.num_key_value_heads != 1
            || self.qk_rope_head_dim > self.head_dim
            || self.qk_rope_head_dim % 2 != 0
            || self.num_attention_heads % self.o_groups != 0
        {
            return invalid("attention head, RoPE, or output-group geometry is inconsistent");
        }
        if self.num_experts_per_tok > self.n_routed_experts || self.n_shared_experts != 1 {
            return invalid(
                "the supported MoE requires one shared expert and a valid routed top-k",
            );
        }
        if self.hidden_act != "silu"
            || self.scoring_func != "sqrtsoftplus"
            || self.topk_method != "noaux_tc"
            || !self.norm_topk_prob
            || self.attention_bias
            || self.attention_dropout != 0.0
            || !self.use_cache
            || self.tie_word_embeddings
        {
            return invalid("unsupported activation, routing, attention, or embedding semantics");
        }
        for (name, value) in [
            ("swiglu_limit", self.swiglu_limit),
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
            ("routed_scaling_factor", self.routed_scaling_factor),
            ("hc_eps", self.hc_eps),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return invalid(format!("text_config.{name} must be finite and positive"));
            }
        }
        if self.rope_scaling.rope_type != "yarn"
            || self.rope_scaling.factor != 16
            || self.rope_scaling.beta_fast == 0
            || self.rope_scaling.beta_slow == 0
            || self.rope_scaling.original_max_position_embeddings == 0
            || self.rope_scaling.original_max_position_embeddings > self.max_position_embeddings
        {
            return invalid("unsupported YaRN configuration");
        }
        self.validate_csa2()?;
        self.validate_engram()?;
        self.validate_dspark()?;
        Ok(())
    }

    fn validate_csa2(&self) -> Result<(), ConfigError> {
        let expected_len = self
            .num_hidden_layers
            .checked_add(self.num_nextn_predict_layers)
            .ok_or_else(|| ConfigError::Invalid("V4.1 layer count overflow".to_owned()))?;
        if self.compress_ratios.len() != expected_len {
            return invalid(format!(
                "compress_ratios has {} entries; expected {expected_len}",
                self.compress_ratios.len()
            ));
        }
        for (layer, &ratio) in self.compress_ratios.iter().enumerate() {
            let expected = if layer < 2 {
                0
            } else if layer < ENCODER_LAYER_COUNT {
                2
            } else if layer < self.num_hidden_layers {
                1
            } else {
                0
            };
            if ratio != expected {
                return invalid(format!(
                    "compress_ratios[{layer}]={ratio}; expected {expected} for the CED layout"
                ));
            }
        }
        if self.kv_source_layer_ids != [2, 8, 14, 20]
            || self.index_source_layer_ids != [2, 8, 14, 20, 24, 28, 32, 36]
            || self.candidate_source_layer_id != ENCODER_LAYER_COUNT
        {
            return invalid("CSA2 KV/index/candidate source layout differs from the release");
        }
        if self.index_topk == 0
            || self.candidate_topk_blocks == 0
            || self.candidate_block_size == 0
            || self.candidate_source_layer_id >= self.num_hidden_layers
        {
            return invalid("hierarchical sparse indexer geometry is invalid");
        }
        Ok(())
    }

    fn validate_engram(&self) -> Result<(), ConfigError> {
        if self.engram_layer_ids != [1, 14]
            || self.engram_num_embeddings.len() != self.engram_layer_ids.len()
            || self.engram_num_embeddings.contains(&0)
            || self.engram_max_ngram_size != 4
            || self.engram_vocab_size == 0
            || self.engram_compressed_vocab_size == 0
        {
            return invalid("Engram layer, table, or n-gram geometry differs from the release");
        }
        Ok(())
    }

    fn validate_dspark(&self) -> Result<(), ConfigError> {
        if self.num_nextn_predict_layers != 3
            || self.dspark_block_size != 5
            || self.dspark_target_layer_ids != [37, 38, 39]
            || self.dspark_n_routed_experts == 0
            || self.dspark_num_experts_per_tok == 0
            || self.dspark_num_experts_per_tok > self.dspark_n_routed_experts
            || self.dspark_noise_token_id as usize >= self.vocab_size
        {
            return invalid("DSpark geometry differs from the three-stage release module");
        }
        Ok(())
    }
}

impl DeepseekV41VisionConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.hidden_size % self.num_attention_heads != 0
            || self.intermediate_size == 0
            || self.patch_size == 0
            || self.downsample_ratio == 0
            || self.max_image_tokens == 0
            || self.min_pixels == 0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || self
                .max_wh_ratio
                .is_some_and(|ratio| !ratio.is_finite() || ratio <= 0.0)
        {
            return invalid("vision tower geometry is invalid");
        }
        Ok(())
    }
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(format!(
        "DeepSeek-V4.1: {}",
        reason.into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASE_CONFIG: &str =
        include_str!("../../../tests/fixtures/deepseek_v4_1_flash_config.json");

    #[test]
    fn parses_release_config_and_exposes_ced_ownership() {
        let config = DeepseekV41Config::from_json_str(RELEASE_CONFIG).unwrap();
        assert_eq!(config.encoder_layer_count(), 20);
        assert_eq!(config.decoder_layer_count(), 20);
        assert_eq!(
            config.attention_mode(0),
            Some(AttentionMode::SlidingWindowOnly)
        );
        assert_eq!(config.attention_mode(2), Some(AttentionMode::Full));
        assert_eq!(config.attention_mode(21), Some(AttentionMode::Reuse));
        assert_eq!(config.attention_mode(24), Some(AttentionMode::Reindex));
        assert_eq!(config.kv_source_for(36), Some(20));
        assert_eq!(config.index_source_for(35), Some(32));
    }

    #[test]
    fn rejects_old_v4_weight_blocks_and_ced_drift() {
        let mut value: serde_json::Value = serde_json::from_str(RELEASE_CONFIG).unwrap();
        value["quantization_config"]["weight_block_size"] = serde_json::json!([128, 128]);
        assert!(DeepseekV41Config::from_json_str(&value.to_string()).is_err());

        let mut value: serde_json::Value = serde_json::from_str(RELEASE_CONFIG).unwrap();
        value["text_config"]["kv_source_layer_ids"] = serde_json::json!([2, 8, 14, 21]);
        assert!(DeepseekV41Config::from_json_str(&value.to_string()).is_err());
    }
}
