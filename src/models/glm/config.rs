//! Strict parsing of the Hugging Face `glm_moe_dsa` configuration.

use crate::config::ConfigError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const SUPPORTED_MODEL_TYPE: &str = "glm_moe_dsa";
/// The official attention module constructs its two latent RMSNorms without passing the
/// decoder block epsilon, so they use this constructor default.
pub const MLA_LATENT_NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl Default for TokenIds {
    fn default() -> Self {
        Self::Many(Vec::new())
    }
}

impl TokenIds {
    pub fn as_slice(&self) -> &[u32] {
        match self {
            Self::One(id) => std::slice::from_ref(id),
            Self::Many(ids) => ids,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_rope_type() -> String {
    "default".to_owned()
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: default_rope_theta(),
            rope_type: default_rope_type(),
        }
    }
}

/// Fields that determine GLM-5.2 inference math or memory use.
///
/// Unknown JSON fields are intentionally accepted so a newer upstream config remains
/// inspectable. Every field used for allocation is range-checked by [`GlmConfig::validate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmConfig {
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_true")]
    pub rope_interleave: bool,
    #[serde(default = "default_true")]
    pub indexer_rope_interleave: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub first_k_dense_replace: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_one")]
    pub n_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_one_f64")]
    pub routed_scaling_factor: f64,
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    #[serde(default)]
    pub index_n_heads: usize,
    #[serde(default)]
    pub index_head_dim: usize,
    #[serde(default)]
    pub index_topk: usize,
    #[serde(default = "default_index_freq")]
    pub index_topk_freq: usize,
    #[serde(default)]
    pub index_skip_topk_offset: usize,
    #[serde(default)]
    pub indexer_types: Vec<String>,
    #[serde(default)]
    pub mlp_layer_types: Vec<String>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub eos_token_id: TokenIds,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default = "default_act")]
    pub hidden_act: String,
    #[serde(default = "default_scoring")]
    pub scoring_func: String,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default)]
    pub dtype: Option<String>,
}

fn default_one() -> usize {
    1
}

fn default_true() -> bool {
    true
}

fn default_one_f64() -> f64 {
    1.0
}

fn default_index_freq() -> usize {
    1
}

fn default_eps() -> f64 {
    1e-5
}

fn default_act() -> String {
    "silu".to_owned()
}

fn default_scoring() -> String {
    "sigmoid".to_owned()
}

fn default_topk_method() -> String {
    "noaux_tc".to_owned()
}

impl GlmConfig {
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
            return Err(ConfigError::Invalid(format!(
                "model_type={:?}; expected {SUPPORTED_MODEL_TYPE:?}",
                self.model_type
            )));
        }
        if self.attention_bias || self.mlp_bias {
            return Err(ConfigError::Invalid(format!(
                "bias-bearing projections are not implemented (attention_bias={}, mlp_bias={})",
                self.attention_bias, self.mlp_bias
            )));
        }
        if !self.rope_interleave || !self.indexer_rope_interleave {
            return Err(ConfigError::Invalid(format!(
                "only interleaved GLM RoPE is implemented (rope_interleave={}, indexer_rope_interleave={})",
                self.rope_interleave, self.indexer_rope_interleave
            )));
        }
        if self.tie_word_embeddings {
            return Err(ConfigError::Invalid(
                "tie_word_embeddings=true is not implemented; an independent LM head is required"
                    .to_owned(),
            ));
        }

        check_range("hidden_size", self.hidden_size, 1, 1 << 20)?;
        check_range("num_hidden_layers", self.num_hidden_layers, 1, 256)?;
        check_range("num_attention_heads", self.num_attention_heads, 1, 4096)?;
        check_range("vocab_size", self.vocab_size, 1, 1 << 24)?;
        check_range("intermediate_size", self.intermediate_size, 1, 1 << 24)?;
        check_range(
            "moe_intermediate_size",
            self.moe_intermediate_size,
            1,
            1 << 24,
        )?;
        check_range("n_routed_experts", self.n_routed_experts, 1, 1 << 16)?;
        check_range("n_shared_experts", self.n_shared_experts, 1, 1024)?;
        check_range(
            "num_experts_per_tok",
            self.num_experts_per_tok,
            1,
            self.n_routed_experts,
        )?;
        check_range(
            "first_k_dense_replace",
            self.first_k_dense_replace,
            0,
            self.num_hidden_layers,
        )?;
        check_range("n_group", self.n_group, 1, self.n_routed_experts)?;
        check_range("topk_group", self.topk_group, 1, self.n_group)?;
        if self.n_routed_experts % self.n_group != 0 {
            return Err(ConfigError::Invalid(format!(
                "n_routed_experts={} is not divisible by n_group={}",
                self.n_routed_experts, self.n_group
            )));
        }
        let experts_per_group = self.n_routed_experts / self.n_group;
        if experts_per_group < 2 {
            return Err(ConfigError::Invalid(
                "noaux_tc requires at least two experts in each group".to_owned(),
            ));
        }
        let exposed_experts = self
            .topk_group
            .checked_mul(experts_per_group)
            .ok_or_else(|| ConfigError::Invalid("selected expert count overflows".to_owned()))?;
        if exposed_experts < self.num_experts_per_tok {
            return Err(ConfigError::Invalid(format!(
                "topk_group exposes {exposed_experts} experts, fewer than num_experts_per_tok={}",
                self.num_experts_per_tok
            )));
        }
        if !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 {
            return Err(ConfigError::Invalid(
                "routed_scaling_factor must be finite and positive".to_owned(),
            ));
        }
        if self.q_lora_rank.is_none() {
            return Err(ConfigError::Invalid(
                "GLM-5.2 requires q_lora_rank for its two-stage query projection".to_owned(),
            ));
        }
        check_range("q_lora_rank", self.q_lora_rank.unwrap_or(0), 1, 1 << 20)?;
        check_range("kv_lora_rank", self.kv_lora_rank, 1, 1 << 20)?;
        check_range("qk_nope_head_dim", self.qk_nope_head_dim, 1, 1 << 16)?;
        check_range("qk_rope_head_dim", self.qk_rope_head_dim, 2, 1 << 16)?;
        check_range("v_head_dim", self.v_head_dim, 1, 1 << 16)?;
        if self.qk_rope_head_dim % 2 != 0 {
            return Err(ConfigError::Invalid(
                "qk_rope_head_dim must be even for interleaved RoPE".to_owned(),
            ));
        }
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim {
            return Err(ConfigError::Invalid(format!(
                "qk_head_dim={} but qk_nope_head_dim + qk_rope_head_dim = {}",
                self.qk_head_dim,
                self.qk_nope_head_dim + self.qk_rope_head_dim
            )));
        }
        if self.num_key_value_heads != 0 && self.num_key_value_heads != self.num_attention_heads {
            return Err(ConfigError::Invalid(format!(
                "num_key_value_heads={} but GLM MLA expects 0 or num_attention_heads={}",
                self.num_key_value_heads, self.num_attention_heads
            )));
        }
        check_range(
            "max_position_embeddings",
            self.max_position_embeddings,
            1,
            1 << 30,
        )?;
        check_range(
            "num_nextn_predict_layers",
            self.num_nextn_predict_layers,
            0,
            16,
        )?;
        if !self.indexer_types.is_empty() && self.indexer_types.len() != self.num_hidden_layers {
            return Err(ConfigError::Invalid(format!(
                "indexer_types has {} entries; expected {}",
                self.indexer_types.len(),
                self.num_hidden_layers
            )));
        }
        if !self.mlp_layer_types.is_empty() && self.mlp_layer_types.len() != self.num_hidden_layers
        {
            return Err(ConfigError::Invalid(format!(
                "mlp_layer_types has {} entries; expected {}",
                self.mlp_layer_types.len(),
                self.num_hidden_layers
            )));
        }
        for (layer, kind) in self.mlp_layer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "dense" | "sparse") {
                return Err(ConfigError::Invalid(format!(
                    "mlp_layer_types[{layer}]={kind:?}; expected dense or sparse"
                )));
            }
            let expected = if layer < self.first_k_dense_replace {
                "dense"
            } else {
                "sparse"
            };
            if kind != expected {
                return Err(ConfigError::Invalid(format!(
                    "mlp_layer_types[{layer}]={kind:?} conflicts with first_k_dense_replace={} (expected {expected:?})",
                    self.first_k_dense_replace
                )));
            }
        }
        for (layer, kind) in self.indexer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "full" | "shared") {
                return Err(ConfigError::Invalid(format!(
                    "indexer_types[{layer}]={kind:?}; expected full or shared"
                )));
            }
        }

        let index_dimensions = [self.index_n_heads, self.index_head_dim, self.index_topk];
        let index_enabled = index_dimensions.iter().any(|&value| value != 0);
        if index_enabled && index_dimensions.contains(&0) {
            return Err(ConfigError::Invalid(
                "index_n_heads, index_head_dim, and index_topk must be all zero or all non-zero"
                    .to_owned(),
            ));
        }
        if index_enabled {
            check_range("index_n_heads", self.index_n_heads, 1, 4096)?;
            check_range("index_head_dim", self.index_head_dim, 1, 1 << 16)?;
            check_range(
                "index_topk",
                self.index_topk,
                1,
                self.max_position_embeddings,
            )?;
            check_range(
                "index_topk_freq",
                self.index_topk_freq,
                1,
                self.num_hidden_layers,
            )?;
            check_range(
                "index_skip_topk_offset",
                self.index_skip_topk_offset,
                0,
                self.num_hidden_layers,
            )?;
            if self.index_head_dim < self.qk_rope_head_dim {
                return Err(ConfigError::Invalid(format!(
                    "index_head_dim={} is smaller than the {}-value rotary prefix",
                    self.index_head_dim, self.qk_rope_head_dim
                )));
            }
        } else if !self.indexer_types.is_empty() {
            return Err(ConfigError::Invalid(
                "indexer_types is present while indexer dimensions are disabled".to_owned(),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(ConfigError::Invalid(format!(
                "hidden_act={:?}; only silu is understood",
                self.hidden_act
            )));
        }
        if self.scoring_func != "sigmoid" || self.topk_method != "noaux_tc" {
            return Err(ConfigError::Invalid(format!(
                "router is scoring_func={:?}, topk_method={:?}; expected sigmoid/noaux_tc",
                self.scoring_func, self.topk_method
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(ConfigError::Invalid(
                "rms_norm_eps must be finite and positive".to_owned(),
            ));
        }
        if !self.rope_parameters.rope_theta.is_finite() || self.rope_parameters.rope_theta <= 0.0 {
            return Err(ConfigError::Invalid(
                "rope_parameters.rope_theta must be finite and positive".to_owned(),
            ));
        }
        if self.rope_parameters.rope_type != "default" {
            return Err(ConfigError::Invalid(format!(
                "rope_type={:?}; only default RoPE is implemented",
                self.rope_parameters.rope_type
            )));
        }
        if self
            .eos_token_id
            .as_slice()
            .iter()
            .any(|&token| token as usize >= self.vocab_size)
        {
            return Err(ConfigError::Invalid(
                "eos_token_id contains an ID outside the vocabulary".to_owned(),
            ));
        }
        if self
            .pad_token_id
            .is_some_and(|token| token as usize >= self.vocab_size)
        {
            return Err(ConfigError::Invalid(
                "pad_token_id is outside the vocabulary".to_owned(),
            ));
        }

        // Fail before downstream code can overflow a matrix allocation.
        checked_product("embedding matrix", &[self.vocab_size, self.hidden_size])?;
        checked_product(
            "query projection",
            &[
                self.num_attention_heads,
                self.qk_head_dim,
                self.q_lora_rank_value(),
            ],
        )?;
        checked_product(
            "KV reconstruction",
            &[
                self.num_attention_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            ],
        )?;
        checked_product(
            "shared experts",
            &[
                self.hidden_size,
                self.moe_intermediate_size,
                self.n_shared_experts,
            ],
        )?;
        checked_product(
            "routed experts",
            &[
                self.sparse_layer_count(),
                self.n_routed_experts,
                3,
                self.hidden_size,
                self.moe_intermediate_size,
            ],
        )?;
        Ok(())
    }

    pub fn q_lora_rank_value(&self) -> usize {
        // `validate` makes this invariant explicit at construction time.
        self.q_lora_rank
            .expect("validated GLM config has q_lora_rank")
    }

    pub fn layer_is_sparse(&self, layer: usize) -> bool {
        self.mlp_layer_types
            .get(layer)
            .map(|kind| kind == "sparse")
            .unwrap_or(layer >= self.first_k_dense_replace)
    }

    pub fn sparse_layer_count(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&layer| self.layer_is_sparse(layer))
            .count()
    }

    pub fn layer_has_full_indexer(&self, layer: usize) -> bool {
        if self.index_n_heads == 0 || self.index_head_dim == 0 || self.index_topk == 0 {
            return false;
        }
        self.indexer_types
            .get(layer)
            .map(|kind| kind == "full")
            .unwrap_or_else(|| {
                let offset = self.index_skip_topk_offset;
                layer.saturating_add(1).saturating_sub(offset) % self.index_topk_freq.max(1) == 0
            })
    }

    pub fn full_indexer_layer_count(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&layer| self.layer_has_full_indexer(layer))
            .count()
    }
}

fn check_range(name: &str, value: usize, min: usize, max: usize) -> Result<(), ConfigError> {
    if value < min || value > max {
        return Err(ConfigError::Invalid(format!(
            "{name}={value} is outside [{min}, {max}]"
        )));
    }
    Ok(())
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize, ConfigError> {
    factors.iter().try_fold(1usize, |acc, &factor| {
        acc.checked_mul(factor).ok_or_else(|| {
            ConfigError::Invalid(format!("{name} dimensions overflow usize: {factors:?}"))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> String {
        serde_json::json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "attention_bias": false,
            "mlp_bias": false,
            "rope_interleave": true,
            "indexer_rope_interleave": true,
            "tie_word_embeddings": false,
            "vocab_size": 32,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "first_k_dense_replace": 1,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "qk_head_dim": 4,
            "v_head_dim": 3,
            "index_n_heads": 2,
            "index_head_dim": 2,
            "index_topk": 8,
            "indexer_types": ["full", "shared", "full", "shared"],
            "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
            "rms_norm_eps": 0.00001,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc"
        })
        .to_string()
    }

    #[test]
    fn parses_and_derives_layer_kinds() {
        let cfg = GlmConfig::from_json_str(&tiny_config()).unwrap();
        assert_eq!(cfg.sparse_layer_count(), 3);
        assert_eq!(cfg.full_indexer_layer_count(), 2);
        assert_eq!(cfg.q_lora_rank_value(), 4);
    }

    #[test]
    fn rejects_inconsistent_head_dimensions() {
        let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
        value["qk_head_dim"] = 5.into();
        let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("qk_head_dim"));
    }

    #[test]
    fn rejects_hostile_group_geometry() {
        let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
        value["n_group"] = 3.into();
        let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("divisible"));
    }

    #[test]
    fn rejects_unknown_layer_kinds_and_unsupported_rope() {
        let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
        value["mlp_layer_types"][2] = "mystery".into();
        let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("expected dense or sparse"));

        let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
        value["rope_parameters"]["rope_type"] = "yarn".into();
        let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("only default RoPE"));
    }

    #[test]
    fn rejects_router_geometry_that_cannot_expose_topk() {
        let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
        value["n_group"] = 4.into();
        value["topk_group"] = 1.into();
        value["num_experts_per_tok"] = 3.into();
        let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("fewer than num_experts_per_tok"));

        let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
        value["routed_scaling_factor"] = (-1.0).into();
        let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("finite and positive"));
    }

    #[test]
    fn rejects_projection_and_rope_semantic_drift() {
        for field in ["attention_bias", "mlp_bias", "tie_word_embeddings"] {
            let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
            value[field] = true.into();
            let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
            assert!(
                error.to_string().contains(field)
                    || error.to_string().contains("bias-bearing projections"),
                "unexpected error for {field}: {error}"
            );
        }

        for field in ["rope_interleave", "indexer_rope_interleave"] {
            let mut value: serde_json::Value = serde_json::from_str(&tiny_config()).unwrap();
            value[field] = false.into();
            let error = GlmConfig::from_json_str(&value.to_string()).unwrap_err();
            assert!(
                error.to_string().contains(field),
                "unexpected error for {field}: {error}"
            );
        }
    }
}
