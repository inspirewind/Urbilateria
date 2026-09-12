//! Strict configuration for Tencent Hy4-preview.

use crate::config::ConfigError;
use crate::models::glm::RopeParameters;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub const MODEL_TYPE: &str = "hy_v4";
pub const ARCHITECTURE: &str = "HYV4ForCausalLM";
pub const RELEASE_LAYER_COUNT: usize = 78;
pub const RELEASE_HIDDEN_SIZE: usize = 6_144;
pub const RELEASE_VOCAB_SIZE: usize = 120_832;
pub const RELEASE_ROUTED_EXPERTS: usize = 256;
pub const RELEASE_FULL_INDEXER_LAYERS: usize = 21;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptQuantization {
    pub quant_algo: String,
    #[serde(default)]
    pub kv_cache_quant_algo: Option<String>,
    #[serde(default)]
    pub exclude_modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub quantization: ModelOptQuantization,
}

/// Fields which determine Hy4 inference math, tensor shapes, or memory use.
///
/// Unknown fields remain accepted so release metadata can add non-ABI annotations without
/// making an otherwise compatible checkpoint unreadable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hy4Config {
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub indexer_types: Vec<String>,
    pub layer_types: Vec<String>,
    pub mlp_layer_types: Vec<String>,
    pub hc_mult: usize,
    pub hc_eps: f64,
    pub hc_magnitude: f64,
    pub enable_ihc: bool,
    pub gated_mla: bool,
    pub gating_type: String,
    pub learnable_sink: bool,
    pub use_dsa: bool,
    pub use_mla: bool,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub tie_word_embeddings: bool,
    pub enable_lm_head_fp32: bool,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub num_nextn_predict_layers: usize,
    pub quantization_config: QuantizationConfig,
    #[serde(default)]
    pub dtype: Option<String>,
}

impl Hy4Config {
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
        if self.model_type != MODEL_TYPE {
            return invalid(format!(
                "model_type={:?}; expected {MODEL_TYPE:?}",
                self.model_type
            ));
        }
        if !self.architectures.is_empty() && self.architectures != [ARCHITECTURE.to_owned()] {
            return invalid(format!(
                "architectures={:?}; expected [{ARCHITECTURE:?}]",
                self.architectures
            ));
        }
        for (name, value, maximum) in [
            ("hidden_size", self.hidden_size, 1 << 20),
            ("num_hidden_layers", self.num_hidden_layers, 256),
            ("num_attention_heads", self.num_attention_heads, 4096),
            ("q_lora_rank", self.q_lora_rank, 1 << 20),
            ("kv_lora_rank", self.kv_lora_rank, 1 << 20),
            ("qk_head_dim", self.qk_head_dim, 1 << 16),
            ("qk_nope_head_dim", self.qk_nope_head_dim, 1 << 16),
            ("qk_rope_head_dim", self.qk_rope_head_dim, 1 << 16),
            ("v_head_dim", self.v_head_dim, 1 << 16),
            ("intermediate_size", self.intermediate_size, 1 << 24),
            ("moe_intermediate_size", self.moe_intermediate_size, 1 << 24),
            ("n_routed_experts", self.n_routed_experts, 1 << 16),
            ("n_shared_experts", self.n_shared_experts, 1024),
            ("num_experts_per_tok", self.num_experts_per_tok, 1 << 16),
            ("index_n_heads", self.index_n_heads, 4096),
            ("index_head_dim", self.index_head_dim, 1 << 16),
            ("index_topk", self.index_topk, 1 << 24),
            ("hc_mult", self.hc_mult, 64),
            ("vocab_size", self.vocab_size, 1 << 24),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                1 << 30,
            ),
        ] {
            if value == 0 || value > maximum {
                return invalid(format!("{name}={value} is outside 1..={maximum}"));
            }
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return invalid("num_experts_per_tok exceeds n_routed_experts");
        }
        if self.n_group == 0
            || self.topk_group == 0
            || self.n_routed_experts % self.n_group != 0
            || self.topk_group > self.n_group
        {
            return invalid("router group geometry is inconsistent");
        }
        let exposed = self
            .topk_group
            .checked_mul(self.n_routed_experts / self.n_group)
            .ok_or_else(|| ConfigError::Invalid("router exposure overflows".to_owned()))?;
        if exposed < self.num_experts_per_tok {
            return invalid("topk_group exposes fewer experts than num_experts_per_tok");
        }
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim {
            return invalid("qk_head_dim differs from qk_nope_head_dim + qk_rope_head_dim");
        }
        if self.qk_rope_head_dim % 2 != 0 {
            return invalid("qk_rope_head_dim must be even");
        }
        if self.indexer_types.len() != self.num_hidden_layers
            || self.layer_types.len() != self.num_hidden_layers
            || self.mlp_layer_types.len() != self.num_hidden_layers
        {
            return invalid("indexer_types/layer_types/mlp_layer_types must cover every layer");
        }
        for (layer, kind) in self.indexer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "full" | "shared") {
                return invalid(format!(
                    "indexer_types[{layer}]={kind:?}; expected full or shared"
                ));
            }
        }
        for (layer, kind) in self.layer_types.iter().enumerate() {
            if kind != "deepseek_sparse_attention" {
                return invalid(format!(
                    "layer_types[{layer}]={kind:?}; expected deepseek_sparse_attention"
                ));
            }
        }
        for (layer, kind) in self.mlp_layer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "dense" | "sparse") {
                return invalid(format!(
                    "mlp_layer_types[{layer}]={kind:?}; expected dense or sparse"
                ));
            }
        }
        if !self.enable_ihc
            || !self.gated_mla
            || self.gating_type != "elementwise"
            || !self.learnable_sink
            || !self.use_dsa
            || !self.use_mla
        {
            return invalid("Hy4 requires identity-HC and elementwise-gated DSA/MLA with sinks");
        }
        if self.attention_bias
            || self.attention_dropout != 0.0
            || self.tie_word_embeddings
            || !self.enable_lm_head_fp32
        {
            return invalid(
                "unsupported attention bias/dropout, tied embeddings, or LM-head dtype",
            );
        }
        if self.hidden_act != "silu" {
            return invalid(format!("hidden_act={:?}; expected silu", self.hidden_act));
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("hc_eps", self.hc_eps),
            ("hc_magnitude", self.hc_magnitude),
            ("routed_scaling_factor", self.routed_scaling_factor),
            ("swiglu_limit", self.swiglu_limit),
            ("rope_theta", self.rope_parameters.rope_theta),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return invalid(format!("{name} must be finite and positive"));
            }
        }
        if self.rope_parameters.rope_type != "default" {
            return invalid("only default RoPE is supported");
        }
        if [self.bos_token_id, self.eos_token_id, self.pad_token_id]
            .iter()
            .any(|&token| token as usize >= self.vocab_size)
        {
            return invalid("special token ID lies outside the vocabulary");
        }
        if self.quantization_config.quant_method != "modelopt"
            || self.quantization_config.quantization.quant_algo != "MXFP8"
            || self
                .quantization_config
                .quantization
                .kv_cache_quant_algo
                .is_some()
        {
            return invalid("only ModelOpt MXFP8 with an unquantized KV cache is supported");
        }
        checked_product("HC state", &[self.hc_mult, self.hidden_size])?;
        checked_product(
            "query projection",
            &[self.num_attention_heads, self.qk_head_dim, self.q_lora_rank],
        )?;
        checked_product(
            "routed experts",
            &[
                self.sparse_layer_count(),
                self.n_routed_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ],
        )?;
        Ok(())
    }

    /// Pins the public preview checkpoint ABI rather than silently accepting another Hy4 scale.
    pub fn validate_release(&self) -> Result<(), ConfigError> {
        self.validate()?;
        for (name, got, expected) in [
            ("hidden_size", self.hidden_size, RELEASE_HIDDEN_SIZE),
            (
                "num_hidden_layers",
                self.num_hidden_layers,
                RELEASE_LAYER_COUNT,
            ),
            ("num_attention_heads", self.num_attention_heads, 64),
            ("num_key_value_heads", self.num_key_value_heads, 8),
            ("head_dim", self.head_dim, 64),
            ("q_lora_rank", self.q_lora_rank, 2_048),
            ("kv_lora_rank", self.kv_lora_rank, 512),
            ("qk_head_dim", self.qk_head_dim, 256),
            ("qk_nope_head_dim", self.qk_nope_head_dim, 192),
            ("qk_rope_head_dim", self.qk_rope_head_dim, 64),
            ("v_head_dim", self.v_head_dim, 256),
            ("intermediate_size", self.intermediate_size, 18_432),
            ("moe_intermediate_size", self.moe_intermediate_size, 2_048),
            (
                "n_routed_experts",
                self.n_routed_experts,
                RELEASE_ROUTED_EXPERTS,
            ),
            ("n_shared_experts", self.n_shared_experts, 1),
            ("num_experts_per_tok", self.num_experts_per_tok, 8),
            ("index_n_heads", self.index_n_heads, 32),
            ("index_head_dim", self.index_head_dim, 128),
            ("index_topk", self.index_topk, 2_048),
            ("hc_mult", self.hc_mult, 4),
            ("vocab_size", self.vocab_size, RELEASE_VOCAB_SIZE),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                1_048_576,
            ),
            ("num_nextn_predict_layers", self.num_nextn_predict_layers, 1),
        ] {
            if got != expected {
                return invalid(format!(
                    "release {name}={got}; expected {expected} for Hy4-preview-FP8"
                ));
            }
        }
        if self.mlp_layer_types.first().map(String::as_str) != Some("dense")
            || self
                .mlp_layer_types
                .iter()
                .skip(1)
                .any(|kind| kind != "sparse")
        {
            return invalid("release MLP map must be 1 dense + 77 MoE layers");
        }
        if self.indexer_types.iter().enumerate().any(|(layer, kind)| {
            let expected = if layer == 0 || layer % 4 == 1 {
                "full"
            } else {
                "shared"
            };
            kind != expected
        }) || self.full_indexer_layer_count() != RELEASE_FULL_INDEXER_LAYERS
        {
            return invalid(
                "release indexer map must be full at layer 0 and layers congruent to 1 modulo 4",
            );
        }
        if self.n_group != 1
            || self.topk_group != 1
            || !self.norm_topk_prob
            || self.routed_scaling_factor != 2.827
            || self.swiglu_limit != 10.0
            || self.hc_eps != 1e-6
            || self.hc_magnitude != 2.0
            || self.rms_norm_eps != 1e-5
            || self.rope_parameters.rope_theta != 10_000_000.0
        {
            return invalid("release router, SwiGLU, iHC, RMSNorm, or RoPE semantics drifted");
        }
        if (self.bos_token_id, self.eos_token_id, self.pad_token_id) != (120_000, 120_025, 120_002)
            || self.dtype.as_deref() != Some("bfloat16")
        {
            return invalid("release token IDs or activation dtype drifted");
        }
        Ok(())
    }

    pub fn layer_is_sparse(&self, layer: usize) -> bool {
        self.mlp_layer_types
            .get(layer)
            .is_some_and(|kind| kind == "sparse")
    }

    pub fn layer_has_full_indexer(&self, layer: usize) -> bool {
        self.indexer_types
            .get(layer)
            .is_some_and(|kind| kind == "full")
    }

    pub fn sparse_layer_count(&self) -> usize {
        self.mlp_layer_types
            .iter()
            .filter(|kind| kind.as_str() == "sparse")
            .count()
    }

    pub fn full_indexer_layer_count(&self) -> usize {
        self.indexer_types
            .iter()
            .filter(|kind| kind.as_str() == "full")
            .count()
    }

    pub fn exact_dense_context_ceiling(&self) -> usize {
        self.index_topk.min(self.max_position_embeddings)
    }
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize, ConfigError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            ConfigError::Invalid(format!("{name} dimensions overflow usize: {factors:?}"))
        })
    })
}

#[cfg(test)]
pub(crate) fn release_test_config() -> Hy4Config {
    Hy4Config {
        model_type: MODEL_TYPE.to_owned(),
        architectures: vec![ARCHITECTURE.to_owned()],
        hidden_size: 6_144,
        num_hidden_layers: 78,
        num_attention_heads: 64,
        num_key_value_heads: 8,
        head_dim: 64,
        q_lora_rank: 2_048,
        kv_lora_rank: 512,
        qk_head_dim: 256,
        qk_nope_head_dim: 192,
        qk_rope_head_dim: 64,
        v_head_dim: 256,
        intermediate_size: 18_432,
        moe_intermediate_size: 2_048,
        n_routed_experts: 256,
        n_shared_experts: 1,
        num_experts_per_tok: 8,
        n_group: 1,
        topk_group: 1,
        norm_topk_prob: true,
        routed_scaling_factor: 2.827,
        swiglu_limit: 10.0,
        index_n_heads: 32,
        index_head_dim: 128,
        index_topk: 2_048,
        indexer_types: (0..78)
            .map(|layer| {
                if layer == 0 || layer % 4 == 1 {
                    "full"
                } else {
                    "shared"
                }
                .to_owned()
            })
            .collect(),
        layer_types: vec!["deepseek_sparse_attention".to_owned(); 78],
        mlp_layer_types: std::iter::once("dense".to_owned())
            .chain(std::iter::repeat_n("sparse".to_owned(), 77))
            .collect(),
        hc_mult: 4,
        hc_eps: 1e-6,
        hc_magnitude: 2.0,
        enable_ihc: true,
        gated_mla: true,
        gating_type: "elementwise".to_owned(),
        learnable_sink: true,
        use_dsa: true,
        use_mla: true,
        attention_bias: false,
        attention_dropout: 0.0,
        tie_word_embeddings: false,
        enable_lm_head_fp32: true,
        hidden_act: "silu".to_owned(),
        rms_norm_eps: 1e-5,
        rope_parameters: RopeParameters {
            rope_theta: 10_000_000.0,
            rope_type: "default".to_owned(),
        },
        max_position_embeddings: 1_048_576,
        vocab_size: 120_832,
        bos_token_id: 120_000,
        eos_token_id: 120_025,
        pad_token_id: 120_002,
        num_nextn_predict_layers: 1,
        quantization_config: QuantizationConfig {
            quant_method: "modelopt".to_owned(),
            quantization: ModelOptQuantization {
                quant_algo: "MXFP8".to_owned(),
                kv_cache_quant_algo: None,
                exclude_modules: Vec::new(),
            },
        },
        dtype: Some("bfloat16".to_owned()),
    }
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(reason.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_config_parses_and_derives_maps() {
        let fixture = release_test_config();
        let config = Hy4Config::from_json_str(&serde_json::to_string(&fixture).unwrap()).unwrap();
        config.validate_release().unwrap();
        assert_eq!(config.sparse_layer_count(), 77);
        assert_eq!(config.full_indexer_layer_count(), 21);
        assert_eq!(config.exact_dense_context_ceiling(), 2_048);
    }

    #[test]
    fn release_gate_rejects_indexcache_or_numerical_semantic_drift() {
        let mut config = release_test_config();
        config.indexer_types.swap(2, 5);
        assert!(config.validate_release().is_err());

        let mut config = release_test_config();
        config.hc_magnitude = 1.0;
        assert!(config.validate_release().is_err());
    }
}
