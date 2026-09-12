//! Strict configuration for DeepSeek-V4-Flash.

use crate::config::ConfigError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const SUPPORTED_MODEL_TYPE: &str = "deepseek_v4";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub scale_fmt: String,
    pub weight_block_size: [usize; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    pub rope_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV4Config {
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f64,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hash_layers: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub scoring_func: String,
    pub topk_method: String,
    pub swiglu_limit: f64,
    pub expert_dtype: String,
    pub quantization_config: QuantizationConfig,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: u32,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

fn default_hidden_act() -> String {
    "silu".to_owned()
}

impl DeepseekV4Config {
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
        if self.model_type != SUPPORTED_MODEL_TYPE {
            return invalid(format!(
                "model_type={:?}; expected {SUPPORTED_MODEL_TYPE:?}",
                self.model_type
            ));
        }
        if self.attention_bias || self.tie_word_embeddings {
            return invalid("attention bias and tied embeddings are not supported");
        }
        for (name, value, maximum) in [
            ("hidden_size", self.hidden_size, 1 << 20),
            ("num_hidden_layers", self.num_hidden_layers, 256),
            ("num_attention_heads", self.num_attention_heads, 4096),
            ("head_dim", self.head_dim, 1 << 16),
            ("q_lora_rank", self.q_lora_rank, 1 << 20),
            ("qk_rope_head_dim", self.qk_rope_head_dim, 1 << 16),
            ("o_groups", self.o_groups, 4096),
            ("o_lora_rank", self.o_lora_rank, 1 << 20),
            ("sliding_window", self.sliding_window, 1 << 20),
            ("vocab_size", self.vocab_size, 1 << 24),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                1 << 30,
            ),
            ("hc_mult", self.hc_mult, 64),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters, 1024),
            ("moe_intermediate_size", self.moe_intermediate_size, 1 << 24),
            ("n_routed_experts", self.n_routed_experts, 1 << 16),
            ("n_shared_experts", self.n_shared_experts, 1024),
            ("num_experts_per_tok", self.num_experts_per_tok, 1 << 16),
            ("index_n_heads", self.index_n_heads, 4096),
            ("index_head_dim", self.index_head_dim, 1 << 16),
            ("index_topk", self.index_topk, 1 << 24),
        ] {
            if value == 0 || value > maximum {
                return invalid(format!("{name}={value} is outside 1..={maximum}"));
            }
        }
        if self.num_key_value_heads != 1 {
            return invalid(format!(
                "num_key_value_heads={} but DeepSeek-V4 uses one shared KV head",
                self.num_key_value_heads
            ));
        }
        if self.qk_rope_head_dim > self.head_dim || self.qk_rope_head_dim % 2 != 0 {
            return invalid("qk_rope_head_dim must be even and no larger than head_dim");
        }
        if self.num_attention_heads % self.o_groups != 0 {
            return invalid("num_attention_heads must be divisible by o_groups");
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return invalid("num_experts_per_tok exceeds n_routed_experts");
        }
        if self.num_hash_layers > self.num_hidden_layers {
            return invalid("num_hash_layers exceeds num_hidden_layers");
        }
        if self.compress_ratios.len() < self.num_hidden_layers {
            return invalid(format!(
                "compress_ratios has {} entries but the base model needs {}",
                self.compress_ratios.len(),
                self.num_hidden_layers
            ));
        }
        for (layer, &ratio) in self
            .compress_ratios
            .iter()
            .take(self.num_hidden_layers)
            .enumerate()
        {
            if !matches!(ratio, 0 | 4 | 128) || ratio > self.max_position_embeddings {
                return invalid(format!("compress_ratios[{layer}]={ratio} is unsupported"));
            }
        }
        if self.compress_ratios[self.num_hidden_layers..]
            .iter()
            .any(|&ratio| ratio != 0)
        {
            return invalid("attached DSpark stages must use compression ratio zero");
        }
        if self.index_head_dim < self.qk_rope_head_dim {
            return invalid("index_head_dim is smaller than qk_rope_head_dim");
        }
        if self.hidden_act != "silu"
            || self.scoring_func != "sqrtsoftplus"
            || self.topk_method != "noaux_tc"
            || !self.norm_topk_prob
        {
            return invalid("unsupported activation or DeepSeek-V4 routing semantics");
        }
        if self.expert_dtype != "fp4"
            || self.quantization_config.activation_scheme != "dynamic"
            || self.quantization_config.quant_method != "fp8"
            || self.quantization_config.fmt != "e4m3"
            || self.quantization_config.scale_fmt != "ue8m0"
            || self.quantization_config.weight_block_size != [128, 128]
        {
            return invalid("checkpoint quantization must be FP8 E4M3/E8M0 plus FP4 experts");
        }
        if self.n_shared_experts != 1 {
            return invalid("the reference runtime currently requires one shared expert");
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("hc_eps", self.hc_eps),
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
            ("routed_scaling_factor", self.routed_scaling_factor),
            ("swiglu_limit", self.swiglu_limit),
            ("rope_scaling.factor", self.rope_scaling.factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return invalid(format!("{name} must be finite and positive"));
            }
        }
        if self.rope_scaling.rope_type != "yarn"
            || self.rope_scaling.original_max_position_embeddings == 0
            || self.rope_scaling.original_max_position_embeddings > self.max_position_embeddings
            || self.rope_scaling.beta_fast == 0
            || self.rope_scaling.beta_slow == 0
        {
            return invalid("DeepSeek-V4 requires a valid YaRN rope_scaling block");
        }
        if usize::try_from(self.bos_token_id).unwrap_or(usize::MAX) >= self.vocab_size
            || usize::try_from(self.eos_token_id).unwrap_or(usize::MAX) >= self.vocab_size
            || usize::try_from(self.dspark_noise_token_id).unwrap_or(usize::MAX) >= self.vocab_size
        {
            return invalid("a configured token ID is outside the vocabulary");
        }
        if self
            .dspark_target_layer_ids
            .iter()
            .any(|&layer| layer >= self.num_hidden_layers)
        {
            return invalid("a DSpark target layer is outside the base model");
        }
        Ok(())
    }

    pub fn base_compress_ratio(&self, layer: usize) -> Option<usize> {
        (layer < self.num_hidden_layers).then(|| self.compress_ratios[layer])
    }

    pub fn indexed_layer_count(&self) -> usize {
        self.compress_ratios
            .iter()
            .take(self.num_hidden_layers)
            .filter(|&&ratio| ratio == 4)
            .count()
    }

    pub fn declared_dspark_stage_count(&self) -> usize {
        self.compress_ratios
            .len()
            .saturating_sub(self.num_hidden_layers)
    }
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(format!(
        "DeepSeek-V4: {}",
        reason.into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLASH_CONFIG: &str =
        include_str!("../../../tests/fixtures/deepseek_v4_flash_0731_config.json");

    #[test]
    fn parses_flash_0731_geometry() {
        let config = DeepseekV4Config::from_json_str(FLASH_CONFIG).unwrap();
        assert_eq!(config.num_hidden_layers, 43);
        assert_eq!(config.num_hash_layers, 3);
        assert_eq!(config.indexed_layer_count(), 21);
        assert_eq!(config.declared_dspark_stage_count(), 3);
        assert_eq!(config.base_compress_ratio(42), Some(4));
    }

    #[test]
    fn rejects_short_compression_schedule() {
        let mut value: serde_json::Value = serde_json::from_str(FLASH_CONFIG).unwrap();
        value["compress_ratios"] = serde_json::json!([0, 0, 4]);
        let error = DeepseekV4Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("compress_ratios"));
    }
}
