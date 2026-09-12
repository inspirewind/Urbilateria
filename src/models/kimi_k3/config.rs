//! Strict parsing and validation of the official Hugging Face `kimi_k3` configuration.

use crate::config::ConfigError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const SUPPORTED_MODEL_TYPE: &str = "kimi_k3";
const SUPPORTED_TEXT_MODEL_TYPE: &str = "kimi_linear";
const SUPPORTED_ARCHITECTURE: &str = "KimiK3ForConditionalGeneration";
const SUPPORTED_TEXT_ARCHITECTURE: &str = "KimiLinearForCausalLM";
const MXFP4_FORMAT: &str = "mxfp4-pack-quantized";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3AutoMap {
    #[serde(rename = "AutoConfig")]
    pub auto_config: String,
    #[serde(rename = "AutoModel")]
    pub auto_model: String,
    #[serde(rename = "AutoModelForCausalLM")]
    pub auto_model_for_causal_lm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3LinearAttentionConfig {
    /// One-based layer numbers, matching the upstream Python configuration.
    pub full_attn_layers: Vec<usize>,
    /// One-based layer numbers, matching the upstream Python configuration.
    pub kda_layers: Vec<usize>,
    pub gate_lower_bound: f64,
    pub head_dim: usize,
    pub num_heads: usize,
    pub short_conv_kernel_size: usize,
    pub use_full_rank_gate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3QuantizedWeights {
    pub dynamic: bool,
    pub group_size: usize,
    pub num_bits: usize,
    pub observer: String,
    pub scale_dtype: String,
    pub strategy: String,
    pub symmetric: bool,
    #[serde(rename = "type")]
    pub weight_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3QuantizationGroup {
    pub format: String,
    pub targets: Vec<String>,
    pub weights: KimiK3QuantizedWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3QuantizationConfig {
    pub config_groups: BTreeMap<String, KimiK3QuantizationGroup>,
    pub format: String,
    pub ignore: Vec<String>,
    pub quant_method: String,
    pub quantization_status: String,
}

/// The nested `text_config` describing the hybrid KDA/Gated-MLA decoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3TextConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub auto_map: KimiK3AutoMap,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub activation_situ_beta: f64,
    pub activation_situ_linear_beta: f64,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub attn_res_block_size: usize,
    pub linear_attn_config: KimiK3LinearAttentionConfig,
    pub first_k_dense_replace: usize,
    pub moe_intermediate_size: usize,
    pub moe_layer_freq: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub num_shared_experts: usize,
    pub num_expert_group: usize,
    pub topk_group: usize,
    pub topk_method: String,
    pub use_grouped_topk: bool,
    pub moe_renormalize: bool,
    pub moe_router_activation_func: String,
    pub routed_scaling_factor: f64,
    pub routed_expert_hidden_size: usize,
    pub latent_moe_use_norm: bool,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub mla_use_nope: bool,
    pub mla_use_output_gate: bool,
    pub quantization_config: KimiK3QuantizationConfig,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub dtype: String,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
}

impl KimiK3TextConfig {
    /// Returns whether a zero-based decoder layer uses Kimi Delta Attention.
    pub fn is_kda_layer(&self, layer: usize) -> bool {
        layer
            .checked_add(1)
            .is_some_and(|one_based| self.linear_attn_config.kda_layers.contains(&one_based))
    }

    pub fn kda_layer_count(&self) -> usize {
        self.linear_attn_config.kda_layers.len()
    }

    pub fn full_attention_layer_count(&self) -> usize {
        self.linear_attn_config.full_attn_layers.len()
    }
}

/// The nested `vision_config` for the MoonViT-V2 tower and multimodal projector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3VisionConfig {
    #[serde(rename = "_attn_implementation")]
    pub attention_implementation: String,
    pub patch_size: usize,
    pub init_pos_emb_height: usize,
    pub init_pos_emb_width: usize,
    pub init_pos_emb_time: usize,
    pub pos_emb_type: String,
    pub vt_num_attention_heads: usize,
    pub vt_num_hidden_layers: usize,
    pub vt_hidden_size: usize,
    pub vt_intermediate_size: usize,
    pub qkv_hidden_size: usize,
    pub merge_kernel_size: [usize; 2],
    pub merge_type: String,
    pub mm_projector_type: String,
    pub mm_hidden_size: usize,
    pub projector_hidden_act: String,
    pub projector_ln_eps: f64,
    pub text_hidden_size: usize,
    pub norm_type: String,
    pub mlp_type: String,
    pub activation_func: String,
    pub pos_emb_interpolation_mode: String,
    pub attn_bias: bool,
    pub patch_embed_proj_bias: bool,
    pub linear_bias: bool,
}

/// Fields that define the official Kimi-K3 checkpoint geometry and preprocessing contract.
///
/// Unknown generic Transformers fields are accepted, while every field that changes tensor
/// geometry, layer assignment, token identity, or quantization is typed and validated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiK3Config {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub auto_map: KimiK3AutoMap,
    pub text_config: KimiK3TextConfig,
    pub vision_config: KimiK3VisionConfig,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub media_placeholder_token_id: u32,
    pub image_placeholder: String,
    pub ignore_index: i64,
    pub dtype: String,
    pub tie_word_embeddings: bool,
}

impl KimiK3Config {
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
        require_single_architecture("architectures", &self.architectures, SUPPORTED_ARCHITECTURE)?;
        validate_auto_map(
            "auto_map",
            &self.auto_map,
            "configuration_kimi_k3.KimiK3Config",
            "modeling_kimi_k3.KimiK3ForConditionalGeneration",
            "modeling_kimi_k3.KimiK3ForConditionalGeneration",
        )?;
        if self.dtype != "bfloat16" || self.text_config.dtype != "bfloat16" {
            return invalid(format!(
                "dtype must be bfloat16 at both levels (top={:?}, text={:?})",
                self.dtype, self.text_config.dtype
            ));
        }
        if self.tie_word_embeddings || self.text_config.tie_word_embeddings {
            return invalid(
                "tied embeddings are incompatible with the official independent LM head",
            );
        }
        if self.ignore_index != -100 {
            return invalid(format!(
                "ignore_index={} but the K3 loss contract uses -100",
                self.ignore_index
            ));
        }
        if self.image_placeholder != "<|kimi_image_placeholder|>" {
            return invalid(format!(
                "image_placeholder={:?}; expected \"<|kimi_image_placeholder|>\"",
                self.image_placeholder
            ));
        }

        self.text_config.validate()?;
        self.vision_config.validate(self.text_config.hidden_size)?;

        for (name, top, nested) in [
            (
                "bos_token_id",
                self.bos_token_id,
                self.text_config.bos_token_id,
            ),
            (
                "eos_token_id",
                self.eos_token_id,
                self.text_config.eos_token_id,
            ),
            (
                "pad_token_id",
                self.pad_token_id,
                self.text_config.pad_token_id,
            ),
        ] {
            if top != nested {
                return invalid(format!(
                    "{name} differs between top-level ({top}) and text_config ({nested})"
                ));
            }
        }
        let token_ids = [
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
            (
                "media_placeholder_token_id",
                self.media_placeholder_token_id,
            ),
        ];
        for (name, id) in token_ids {
            if id as usize >= self.text_config.vocab_size {
                return invalid(format!(
                    "{name}={id} is outside vocab_size={}",
                    self.text_config.vocab_size
                ));
            }
        }
        for left in 0..token_ids.len() {
            for right in (left + 1)..token_ids.len() {
                if token_ids[left].1 == token_ids[right].1 {
                    return invalid(format!(
                        "{} and {} both use token ID {}",
                        token_ids[left].0, token_ids[right].0, token_ids[left].1
                    ));
                }
            }
        }
        Ok(())
    }
}

impl KimiK3TextConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != SUPPORTED_TEXT_MODEL_TYPE {
            return invalid(format!(
                "text_config.model_type={:?}; expected {SUPPORTED_TEXT_MODEL_TYPE:?}",
                self.model_type
            ));
        }
        require_single_architecture(
            "text_config.architectures",
            &self.architectures,
            SUPPORTED_TEXT_ARCHITECTURE,
        )?;
        validate_auto_map(
            "text_config.auto_map",
            &self.auto_map,
            "configuration_kimi_k3.KimiLinearConfig",
            "modeling_kimi_linear.KimiLinearModel",
            "modeling_kimi_linear.KimiLinearForCausalLM",
        )?;

        for (name, value, maximum) in [
            ("text_config.hidden_size", self.hidden_size, 1 << 20),
            (
                "text_config.num_hidden_layers",
                self.num_hidden_layers,
                1024,
            ),
            (
                "text_config.num_attention_heads",
                self.num_attention_heads,
                4096,
            ),
            (
                "text_config.num_key_value_heads",
                self.num_key_value_heads,
                4096,
            ),
            ("text_config.vocab_size", self.vocab_size, 1 << 24),
            (
                "text_config.intermediate_size",
                self.intermediate_size,
                1 << 24,
            ),
            (
                "text_config.attn_res_block_size",
                self.attn_res_block_size,
                1024,
            ),
            (
                "text_config.moe_intermediate_size",
                self.moe_intermediate_size,
                1 << 24,
            ),
            ("text_config.moe_layer_freq", self.moe_layer_freq, 1024),
            ("text_config.num_experts", self.num_experts, 1 << 16),
            (
                "text_config.num_experts_per_token",
                self.num_experts_per_token,
                1 << 16,
            ),
            (
                "text_config.num_shared_experts",
                self.num_shared_experts,
                1024,
            ),
            (
                "text_config.num_expert_group",
                self.num_expert_group,
                1 << 16,
            ),
            ("text_config.topk_group", self.topk_group, 1 << 16),
            (
                "text_config.routed_expert_hidden_size",
                self.routed_expert_hidden_size,
                1 << 24,
            ),
            ("text_config.q_lora_rank", self.q_lora_rank, 1 << 20),
            ("text_config.kv_lora_rank", self.kv_lora_rank, 1 << 20),
            (
                "text_config.qk_nope_head_dim",
                self.qk_nope_head_dim,
                1 << 16,
            ),
            (
                "text_config.qk_rope_head_dim",
                self.qk_rope_head_dim,
                1 << 16,
            ),
            ("text_config.v_head_dim", self.v_head_dim, 1 << 16),
            (
                "text_config.max_position_embeddings",
                self.max_position_embeddings,
                1 << 30,
            ),
        ] {
            check_range(name, value, 1, maximum)?;
        }
        check_range(
            "text_config.first_k_dense_replace",
            self.first_k_dense_replace,
            0,
            self.num_hidden_layers,
        )?;
        if self.attn_res_block_size > self.num_hidden_layers {
            return invalid("text_config.attn_res_block_size exceeds num_hidden_layers");
        }
        if self.moe_layer_freq > self.num_hidden_layers {
            return invalid("text_config.moe_layer_freq exceeds num_hidden_layers");
        }
        if self.num_key_value_heads != self.num_attention_heads {
            return invalid(format!(
                "text_config.num_key_value_heads={} but K3 expects one KV head per attention head ({})",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        if self.num_experts_per_token > self.num_experts {
            return invalid("text_config.num_experts_per_token exceeds num_experts");
        }
        if self.num_experts % self.num_expert_group != 0 {
            return invalid("text_config.num_experts is not divisible by num_expert_group");
        }
        if self.topk_group > self.num_expert_group {
            return invalid("text_config.topk_group exceeds num_expert_group");
        }
        let experts_per_group = self.num_experts / self.num_expert_group;
        let exposed = self
            .topk_group
            .checked_mul(experts_per_group)
            .ok_or_else(|| {
                ConfigError::Invalid("Kimi-K3: selected expert count overflows usize".to_owned())
            })?;
        if exposed < self.num_experts_per_token {
            return invalid(format!(
                "text_config.topk_group exposes {exposed} experts, fewer than num_experts_per_token={}",
                self.num_experts_per_token
            ));
        }
        if self.qk_rope_head_dim % 2 != 0 {
            return invalid("text_config.qk_rope_head_dim must be even");
        }
        if self.hidden_act != "situ"
            || self.moe_router_activation_func != "sigmoid"
            || self.topk_method != "noaux_tc"
        {
            return invalid(format!(
                "unsupported activation/router semantics: hidden_act={:?}, router={:?}, topk_method={:?}",
                self.hidden_act, self.moe_router_activation_func, self.topk_method
            ));
        }
        if !self.use_grouped_topk
            || !self.moe_renormalize
            || !self.latent_moe_use_norm
            || !self.mla_use_nope
            || !self.mla_use_output_gate
            || !self.use_cache
        {
            return invalid("official K3 grouped routing, LatentMoE norm, gated MLA, and cache flags must be enabled");
        }
        if self.num_nextn_predict_layers != 0 {
            return invalid("text_config.num_nextn_predict_layers must be zero for the released base checkpoint");
        }
        for (name, value) in [
            (
                "text_config.activation_situ_beta",
                self.activation_situ_beta,
            ),
            (
                "text_config.activation_situ_linear_beta",
                self.activation_situ_linear_beta,
            ),
            ("text_config.initializer_range", self.initializer_range),
            ("text_config.rms_norm_eps", self.rms_norm_eps),
            (
                "text_config.routed_scaling_factor",
                self.routed_scaling_factor,
            ),
        ] {
            require_finite_positive(name, value)?;
        }

        self.linear_attn_config.validate(self)?;
        self.quantization_config.validate()?;

        checked_product("text embedding", &[self.vocab_size, self.hidden_size])?;
        checked_product("dense MLP", &[self.hidden_size, self.intermediate_size, 3])?;
        checked_product(
            "routed experts",
            &[
                self.num_hidden_layers
                    .saturating_sub(self.first_k_dense_replace),
                self.num_experts,
                self.hidden_size,
                self.moe_intermediate_size,
                3,
            ],
        )?;
        Ok(())
    }
}

impl KimiK3LinearAttentionConfig {
    fn validate(&self, text: &KimiK3TextConfig) -> Result<(), ConfigError> {
        for (name, value, maximum) in [
            (
                "text_config.linear_attn_config.head_dim",
                self.head_dim,
                1 << 16,
            ),
            (
                "text_config.linear_attn_config.num_heads",
                self.num_heads,
                4096,
            ),
            (
                "text_config.linear_attn_config.short_conv_kernel_size",
                self.short_conv_kernel_size,
                1024,
            ),
        ] {
            check_range(name, value, 1, maximum)?;
        }
        if self.num_heads != text.num_attention_heads {
            return invalid(format!(
                "linear_attn_config.num_heads={} differs from text_config.num_attention_heads={}",
                self.num_heads, text.num_attention_heads
            ));
        }
        // The released checkpoint stores KDA A_log with `head_dim` entries and consumes only
        // the first `num_heads`; do not equate these fields or reject the padded tail.
        if self.head_dim < self.num_heads {
            return invalid("linear_attn_config.head_dim is smaller than num_heads");
        }
        if self.head_dim != text.v_head_dim {
            return invalid(format!(
                "linear_attn_config.head_dim={} differs from text_config.v_head_dim={}",
                self.head_dim, text.v_head_dim
            ));
        }
        if !self.use_full_rank_gate {
            return invalid("linear_attn_config.use_full_rank_gate must be true");
        }
        if !self.gate_lower_bound.is_finite() {
            return invalid("linear_attn_config.gate_lower_bound must be finite");
        }
        validate_layer_partition(
            text.num_hidden_layers,
            &self.kda_layers,
            &self.full_attn_layers,
        )
    }
}

impl KimiK3QuantizationConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.format != MXFP4_FORMAT
            || self.quant_method != "compressed-tensors"
            || self.quantization_status != "compressed"
        {
            return invalid(format!(
                "unsupported quantization: format={:?}, method={:?}, status={:?}",
                self.format, self.quant_method, self.quantization_status
            ));
        }
        if self.config_groups.len() != 1 || !self.config_groups.contains_key("group_0") {
            return invalid("quantization_config must contain only config_groups.group_0");
        }
        let group = &self.config_groups["group_0"];
        if group.format != MXFP4_FORMAT || group.targets != ["Linear"] {
            return invalid("quantization group_0 must target Linear with the MXFP4 packed format");
        }
        let weights = &group.weights;
        if weights.dynamic
            || weights.group_size != 32
            || weights.num_bits != 4
            || weights.observer != "minmax"
            || weights.scale_dtype != "torch.uint8"
            || weights.strategy != "group"
            || !weights.symmetric
            || weights.weight_type != "float"
        {
            return invalid(format!(
                "group_0 weights are not the released symmetric MXFP4 group-32 encoding (num_bits={}, group_size={})",
                weights.num_bits, weights.group_size
            ));
        }
        const REQUIRED_IGNORES: &[&str] = &[
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
            "re:.*lm_head.*",
            "re:.*vision_tower.*",
            "re:.*mm_projector.*",
        ];
        for required in REQUIRED_IGNORES {
            if !self.ignore.iter().any(|entry| entry == required) {
                return invalid(format!(
                    "quantization_config.ignore is missing required pattern {required:?}"
                ));
            }
        }
        Ok(())
    }
}

impl KimiK3VisionConfig {
    fn validate(&self, text_hidden_size: usize) -> Result<(), ConfigError> {
        for (name, value, maximum) in [
            ("vision_config.patch_size", self.patch_size, 1 << 16),
            (
                "vision_config.init_pos_emb_height",
                self.init_pos_emb_height,
                1 << 16,
            ),
            (
                "vision_config.init_pos_emb_width",
                self.init_pos_emb_width,
                1 << 16,
            ),
            (
                "vision_config.init_pos_emb_time",
                self.init_pos_emb_time,
                1 << 16,
            ),
            (
                "vision_config.vt_num_attention_heads",
                self.vt_num_attention_heads,
                4096,
            ),
            (
                "vision_config.vt_num_hidden_layers",
                self.vt_num_hidden_layers,
                1024,
            ),
            ("vision_config.vt_hidden_size", self.vt_hidden_size, 1 << 20),
            (
                "vision_config.vt_intermediate_size",
                self.vt_intermediate_size,
                1 << 24,
            ),
            (
                "vision_config.qkv_hidden_size",
                self.qkv_hidden_size,
                1 << 20,
            ),
            ("vision_config.mm_hidden_size", self.mm_hidden_size, 1 << 20),
            (
                "vision_config.text_hidden_size",
                self.text_hidden_size,
                1 << 20,
            ),
        ] {
            check_range(name, value, 1, maximum)?;
        }
        for (axis, value) in self.merge_kernel_size.into_iter().enumerate() {
            check_range(
                &format!("vision_config.merge_kernel_size[{axis}]"),
                value,
                1,
                1024,
            )?;
        }
        if self.qkv_hidden_size % self.vt_num_attention_heads != 0 {
            return invalid(
                "vision_config.qkv_hidden_size is not divisible by vt_num_attention_heads",
            );
        }
        if self.mm_hidden_size != self.vt_hidden_size {
            return invalid(format!(
                "vision_config.mm_hidden_size={} differs from vt_hidden_size={}",
                self.mm_hidden_size, self.vt_hidden_size
            ));
        }
        if self.text_hidden_size != text_hidden_size {
            return invalid(format!(
                "vision_config.text_hidden_size={} differs from text_config.hidden_size={text_hidden_size}",
                self.text_hidden_size
            ));
        }
        if self.attention_implementation != "flash_attention_2"
            || self.pos_emb_type != "divided_fixed"
            || self.merge_type != "sd2_tpool"
            || self.mm_projector_type != "patchmergerv2"
            || self.projector_hidden_act != "gelu"
            || self.norm_type != "rmsnorm"
            || self.mlp_type != "mlp2"
            || self.activation_func != "gelu_pytorch_tanh"
            || self.pos_emb_interpolation_mode != "bilinear"
        {
            return invalid("unsupported MoonViT-V2 or multimodal-projector semantics");
        }
        if self.attn_bias || self.patch_embed_proj_bias || self.linear_bias {
            return invalid("vision_config projection biases must be disabled");
        }
        require_finite_positive("vision_config.projector_ln_eps", self.projector_ln_eps)?;
        checked_product(
            "vision positional grid",
            &[
                self.init_pos_emb_time,
                self.init_pos_emb_height,
                self.init_pos_emb_width,
                self.vt_hidden_size,
            ],
        )?;
        Ok(())
    }
}

fn validate_layer_partition(
    num_hidden_layers: usize,
    kda_layers: &[usize],
    full_attn_layers: &[usize],
) -> Result<(), ConfigError> {
    if kda_layers.is_empty() || full_attn_layers.is_empty() {
        return invalid("KDA and full-attention layer lists must both be non-empty");
    }
    for (name, layers) in [
        ("kda_layers", kda_layers),
        ("full_attn_layers", full_attn_layers),
    ] {
        if layers.windows(2).any(|pair| pair[0] >= pair[1]) {
            return invalid(format!(
                "linear_attn_config.{name} must be strictly increasing and unique"
            ));
        }
        if let Some(layer) = layers
            .iter()
            .copied()
            .find(|&layer| layer == 0 || layer > num_hidden_layers)
        {
            return invalid(format!(
                "linear_attn_config.{name} contains one-based layer {layer} outside 1..={num_hidden_layers}"
            ));
        }
    }

    let mut assigned = vec![false; num_hidden_layers + 1];
    for (kind, layers) in [("KDA", kda_layers), ("full attention", full_attn_layers)] {
        for &layer in layers {
            if std::mem::replace(&mut assigned[layer], true) {
                return invalid(format!(
                    "one-based layer {layer} is assigned more than once across the KDA/full-attention map ({kind})"
                ));
            }
        }
    }
    if let Some(layer) = (1..=num_hidden_layers).find(|&layer| !assigned[layer]) {
        return invalid(format!(
            "one-based layer {layer} is absent from the KDA/full-attention map"
        ));
    }
    Ok(())
}

fn require_single_architecture(
    name: &str,
    architectures: &[String],
    expected: &str,
) -> Result<(), ConfigError> {
    if architectures.len() != 1 || architectures[0] != expected {
        return invalid(format!(
            "{name}={architectures:?}; expected exactly [{expected:?}]"
        ));
    }
    Ok(())
}

fn validate_auto_map(
    name: &str,
    map: &KimiK3AutoMap,
    auto_config: &str,
    auto_model: &str,
    auto_model_for_causal_lm: &str,
) -> Result<(), ConfigError> {
    if map.auto_config != auto_config
        || map.auto_model != auto_model
        || map.auto_model_for_causal_lm != auto_model_for_causal_lm
    {
        return invalid(format!(
            "{name} does not name the official Kimi-K3 remote classes"
        ));
    }
    Ok(())
}

fn check_range(name: &str, value: usize, min: usize, max: usize) -> Result<(), ConfigError> {
    if value < min || value > max {
        return invalid(format!("{name}={value} is outside [{min}, {max}]"));
    }
    Ok(())
}

fn require_finite_positive(name: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return invalid(format!("{name} must be finite and positive"));
    }
    Ok(())
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize, ConfigError> {
    factors.iter().try_fold(1usize, |acc, &factor| {
        acc.checked_mul(factor).ok_or_else(|| {
            ConfigError::Invalid(format!(
                "Kimi-K3: {name} dimensions overflow usize: {factors:?}"
            ))
        })
    })
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(format!("Kimi-K3: {}", reason.into())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelFamily};
    use serde_json::{json, Value};

    fn official_config_value() -> Value {
        let full_attn_layers: Vec<usize> = (4..=92).step_by(4).chain(std::iter::once(93)).collect();
        let kda_layers: Vec<usize> = (1..=93)
            .filter(|layer| !full_attn_layers.contains(layer))
            .collect();
        let linear_attn_config = json!({
            "full_attn_layers": full_attn_layers,
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "kda_layers": kda_layers,
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": true
        });
        let quantization_config = json!({
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "dynamic": false,
                        "group_size": 32,
                        "num_bits": 4,
                        "observer": "minmax",
                        "scale_dtype": "torch.uint8",
                        "strategy": "group",
                        "symmetric": true,
                        "type": "float"
                    }
                }
            },
            "format": "mxfp4-pack-quantized",
            "ignore": [
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
                "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
                "re:.*lm_head.*",
                "re:.*vision_tower.*",
                "re:.*mm_projector.*"
            ],
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed"
        });
        let mut text_config: Value = serde_json::from_str(
            r#"{
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "architectures": ["KimiLinearForCausalLM"],
                "attn_res_block_size": 12,
                "auto_map": {
                    "AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
                    "AutoModel": "modeling_kimi_linear.KimiLinearModel",
                    "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM"
                },
                "bos_token_id": 163584,
                "dtype": "bfloat16",
                "eos_token_id": 163586,
                "first_k_dense_replace": 1,
                "hidden_act": "situ",
                "hidden_size": 7168,
                "initializer_range": 0.02,
                "intermediate_size": 33792,
                "kv_lora_rank": 512,
                "latent_moe_use_norm": true,
                "max_position_embeddings": 1048576,
                "mla_use_nope": true,
                "mla_use_output_gate": true,
                "model_type": "kimi_linear",
                "moe_intermediate_size": 3072,
                "moe_layer_freq": 1,
                "moe_renormalize": true,
                "moe_router_activation_func": "sigmoid",
                "num_attention_heads": 96,
                "num_expert_group": 1,
                "num_experts": 896,
                "num_experts_per_token": 16,
                "num_hidden_layers": 93,
                "num_key_value_heads": 96,
                "num_nextn_predict_layers": 0,
                "num_shared_experts": 2,
                "pad_token_id": 163839,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "rms_norm_eps": 0.00001,
                "routed_expert_hidden_size": 3584,
                "routed_scaling_factor": 1.0,
                "tie_word_embeddings": false,
                "topk_group": 1,
                "topk_method": "noaux_tc",
                "use_cache": true,
                "use_grouped_topk": true,
                "v_head_dim": 128,
                "vocab_size": 163840
            }"#,
        )
        .unwrap();
        text_config["linear_attn_config"] = linear_attn_config;
        text_config["quantization_config"] = quantization_config;
        let vision_config = json!({
            "_attn_implementation": "flash_attention_2",
            "activation_func": "gelu_pytorch_tanh",
            "attn_bias": false,
            "init_pos_emb_height": 64,
            "init_pos_emb_time": 4,
            "init_pos_emb_width": 64,
            "linear_bias": false,
            "merge_kernel_size": [2, 2],
            "merge_type": "sd2_tpool",
            "mlp_type": "mlp2",
            "mm_hidden_size": 1024,
            "mm_projector_type": "patchmergerv2",
            "norm_type": "rmsnorm",
            "patch_embed_proj_bias": false,
            "patch_size": 14,
            "pos_emb_interpolation_mode": "bilinear",
            "pos_emb_type": "divided_fixed",
            "projector_hidden_act": "gelu",
            "projector_ln_eps": 0.00001,
            "qkv_hidden_size": 1536,
            "text_hidden_size": 7168,
            "vt_hidden_size": 1024,
            "vt_intermediate_size": 4096,
            "vt_num_attention_heads": 12,
            "vt_num_hidden_layers": 27
        });
        json!({
            "architectures": ["KimiK3ForConditionalGeneration"],
            "auto_map": {
                "AutoConfig": "configuration_kimi_k3.KimiK3Config",
                "AutoModel": "modeling_kimi_k3.KimiK3ForConditionalGeneration",
                "AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration"
            },
            "bos_token_id": 163584,
            "dtype": "bfloat16",
            "eos_token_id": 163586,
            "ignore_index": -100,
            "image_placeholder": "<|kimi_image_placeholder|>",
            "media_placeholder_token_id": 163605,
            "model_type": "kimi_k3",
            "pad_token_id": 163839,
            "tie_word_embeddings": false,
            "text_config": text_config,
            "vision_config": vision_config
        })
    }

    #[test]
    fn parses_official_geometry_and_registers_model_family() {
        let json = official_config_value().to_string();
        let config = KimiK3Config::from_json_str(&json).unwrap();
        assert_eq!(config.text_config.kda_layer_count(), 69);
        assert_eq!(config.text_config.full_attention_layer_count(), 24);
        assert!(config.text_config.is_kda_layer(0));
        assert!(!config.text_config.is_kda_layer(3));
        assert!(!config.text_config.is_kda_layer(92));

        let detected = ModelConfig::from_json_str(&json).unwrap();
        assert_eq!(detected.family(), ModelFamily::KimiK3);
        assert!(detected.as_kimi_k3().is_some());
        assert!(detected.as_glm52().is_none());
        let common = detected.common();
        assert_eq!(common.family, ModelFamily::KimiK3);
        assert_eq!(common.hidden_size, 7168);
        assert_eq!(common.num_hidden_layers, 93);
        assert_eq!(common.max_position_embeddings, 1_048_576);
        assert_eq!(common.eos_token_ids, vec![163586]);
    }

    #[test]
    fn rejects_unsorted_duplicate_or_incomplete_layer_maps() {
        let mut value = official_config_value();
        value["text_config"]["linear_attn_config"]["kda_layers"][1] = json!(1);
        let error = KimiK3Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("strictly increasing"));

        let mut value = official_config_value();
        value["text_config"]["linear_attn_config"]["kda_layers"]
            .as_array_mut()
            .unwrap()
            .insert(3, json!(4));
        let error = KimiK3Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("assigned more than once"));

        let mut value = official_config_value();
        value["text_config"]["linear_attn_config"]["kda_layers"]
            .as_array_mut()
            .unwrap()
            .remove(0);
        let error = KimiK3Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("absent from"));
    }

    #[test]
    fn rejects_cross_config_geometry_and_token_mismatches() {
        let mut value = official_config_value();
        value["vision_config"]["text_hidden_size"] = json!(4096);
        let error = KimiK3Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("text_hidden_size"));

        let mut value = official_config_value();
        value["text_config"]["eos_token_id"] = json!(7);
        let error = KimiK3Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("eos_token_id differs"));
    }

    #[test]
    fn rejects_non_official_quantization_and_malformed_vision_shape() {
        let mut value = official_config_value();
        value["text_config"]["quantization_config"]["config_groups"]["group_0"]["weights"]
            ["num_bits"] = json!(8);
        let error = KimiK3Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("num_bits=8"));

        let mut value = official_config_value();
        value["vision_config"]["merge_kernel_size"] = json!([2]);
        let error = KimiK3Config::from_json_str(&value.to_string()).unwrap_err();
        assert!(matches!(error, ConfigError::Json { .. }));
    }
}
