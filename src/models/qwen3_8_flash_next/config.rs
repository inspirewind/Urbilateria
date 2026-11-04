//! Strict configuration for the local Qwen3.8-Flash-Next BF16 release.
use crate::config::ConfigError;
pub use crate::models::qwen3_8::{
    Qwen38GenerationConfig as Qwen38FlashNextGenerationConfig, BOS_TOKEN_ID, IM_END_TOKEN_ID,
};
use serde::{Deserialize, Serialize};
use std::{fs, ops::Deref, path::Path};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RopeParameters {
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rope_type: String,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen38FlashNextTextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub bos_token_id: u32,
    pub dtype: String,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub heads_per_ngram: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub indexer_head_dim: usize,
    pub indexer_kv_heads: usize,
    pub indexer_n_heads: usize,
    pub make_ngram_vocab_size_divisible_by: usize,
    pub mtp: serde_json::Value,
    pub ngram_size: usize,
    pub ngram_vocab_size_base: usize,
    pub output_gate_type: String,
    pub ple_conv_kernel_size: usize,
    pub ple_embed_dim: usize,
    pub ple_layer_ids: Vec<usize>,
    pub split_ngram_parts: usize,
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
    pub output_router_logits: bool,
    pub pad_token_id: Option<u32>,
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub router_aux_loss_coef: f64,
    pub shared_expert_intermediate_size: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen38FlashNextConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: Qwen38FlashNextTextConfig,
    pub language_model_only: bool,
    pub vision_config: serde_json::Value,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
}
impl Deref for Qwen38FlashNextConfig {
    type Target = Qwen38FlashNextTextConfig;
    fn deref(&self) -> &Self::Target {
        &self.text_config
    }
}
impl Qwen38FlashNextConfig {
    pub fn load(directory: &Path) -> Result<Self, ConfigError> {
        let path = directory.join("config.json");
        let json =
            fs::read_to_string(&path).map_err(|source| ConfigError::Read { path, source })?;
        Self::from_json_str(&json)
    }
    pub fn from_json_str(json: &str) -> Result<Self, ConfigError> {
        let config: Self = serde_json::from_str(json)
            .map_err(|source| ConfigError::Json { path: None, source })?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Keep this adapter tied to the inspected release, including unsupported vision/MTP metadata.
        let expected: Self = serde_json::from_str(include_str!("release_config.json"))
            .expect("bundled release config");
        if self != &expected {
            return Err(ConfigError::Invalid(
                "configuration differs from Qwen3.8-Flash-Next BF16 release".into(),
            ));
        }
        Ok(())
    }
    pub fn is_full_attention_layer(&self, layer: usize) -> Option<bool> {
        self.layer_types
            .get(layer)
            .map(|kind| kind == "full_attention")
    }
}
