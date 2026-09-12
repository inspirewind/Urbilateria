//! Strict bridge from the serialized Kimi K3 text configuration to runtime geometry.
//!
//! Config parsing establishes the checkpoint contract. This module repeats the invariants that
//! directly affect scalar execution so callers cannot construct public config structs by hand and
//! bypass them. It also centralizes checked per-sequence state sizing for resource planners.

use super::attention::{KdaGeometry, MlaGeometry};
use super::config::KimiK3TextConfig;
use super::moe::{KimiK3MoeError, LatentMoeGeometry, NoAuxTcConfig};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionLayerKind {
    Kda,
    Mla,
}

/// F32 element counts for one sequence through the whole text decoder.
///
/// KDA state is independent of sequence length. MLA cache values are reported per token; callers
/// multiply by the requested context length with their own checked policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceStateGeometry {
    pub kda_layer_count: usize,
    pub mla_layer_count: usize,

    pub kda_recurrent_f32_per_layer: usize,
    pub kda_conv_history_f32_per_layer: usize,
    pub kda_state_f32_per_layer: usize,
    pub kda_recurrent_f32_per_sequence: usize,
    pub kda_conv_history_f32_per_sequence: usize,
    pub kda_state_f32_per_sequence: usize,

    /// Normalized KV latent plus the one shared, unrotated NoPE slot.
    pub mla_cache_f32_per_token_per_layer: usize,
    /// Compressed cache across every MLA layer for one token.
    pub mla_cache_f32_per_token: usize,

    /// Maximum block snapshots retained by AttnRes. Layer zero is a boundary.
    pub attn_res_max_snapshots: usize,
    /// Snapshot count plus the current running prefix used by an AttnRes softmax.
    pub attn_res_max_sources: usize,
}

/// All scalar component geometry derived from one validated text configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct KimiK3RuntimeGeometry {
    pub kda: KdaGeometry,
    pub mla: MlaGeometry,
    pub latent_moe: LatentMoeGeometry,
    pub attention_layers: Vec<AttentionLayerKind>,
    pub sequence_state: SequenceStateGeometry,
}

impl KimiK3RuntimeGeometry {
    pub fn from_text_config(config: &KimiK3TextConfig) -> Result<Self, GeometryError> {
        validate_execution_flags(config)?;
        let attention_layers = attention_layer_assignment(config)?;
        let kda = build_kda_geometry(config)?;
        let mla = build_mla_geometry(config)?;
        let latent_moe = build_latent_moe_geometry(config)?;
        let sequence_state = build_sequence_state(config, &attention_layers, kda, mla)?;
        Ok(Self {
            kda,
            mla,
            latent_moe,
            attention_layers,
            sequence_state,
        })
    }

    pub fn layer_kind(&self, zero_based_layer: usize) -> Option<AttentionLayerKind> {
        self.attention_layers.get(zero_based_layer).copied()
    }
}

/// Strictly derives every runtime component and resource count.
pub fn runtime_geometry_from_config(
    config: &KimiK3TextConfig,
) -> Result<KimiK3RuntimeGeometry, GeometryError> {
    KimiK3RuntimeGeometry::from_text_config(config)
}

/// Strictly derives KDA geometry. Full config validation is intentional: a malformed MLA or layer
/// assignment must not yield a partially trusted runtime plan.
pub fn kda_geometry_from_config(config: &KimiK3TextConfig) -> Result<KdaGeometry, GeometryError> {
    Ok(runtime_geometry_from_config(config)?.kda)
}

/// Strictly derives gated NoPE-MLA geometry.
pub fn mla_geometry_from_config(config: &KimiK3TextConfig) -> Result<MlaGeometry, GeometryError> {
    Ok(runtime_geometry_from_config(config)?.mla)
}

/// Strictly derives LatentMoE geometry and router semantics.
pub fn latent_moe_geometry_from_config(
    config: &KimiK3TextConfig,
) -> Result<LatentMoeGeometry, GeometryError> {
    Ok(runtime_geometry_from_config(config)?.latent_moe)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryError {
    Invalid {
        field: &'static str,
        reason: String,
    },
    Overflow {
        expression: &'static str,
    },
    Allocation {
        value: &'static str,
        elements: usize,
    },
    Component {
        component: &'static str,
        message: String,
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { field, reason } => {
                write!(f, "invalid Kimi K3 runtime geometry `{field}`: {reason}")
            }
            Self::Overflow { expression } => write!(
                f,
                "integer overflow while computing Kimi K3 runtime geometry {expression}"
            ),
            Self::Allocation { value, elements } => write!(
                f,
                "cannot allocate Kimi K3 runtime geometry {value} with {elements} elements"
            ),
            Self::Component { component, message } => {
                write!(f, "invalid Kimi K3 {component} geometry: {message}")
            }
        }
    }
}

impl std::error::Error for GeometryError {}

fn validate_execution_flags(config: &KimiK3TextConfig) -> Result<(), GeometryError> {
    require_nonzero("num_hidden_layers", config.num_hidden_layers)?;
    require_nonzero("hidden_size", config.hidden_size)?;
    require_nonzero("num_attention_heads", config.num_attention_heads)?;
    require_nonzero("num_key_value_heads", config.num_key_value_heads)?;
    require_nonzero("attn_res_block_size", config.attn_res_block_size)?;
    if config.attn_res_block_size > config.num_hidden_layers {
        return invalid("attn_res_block_size", "must not exceed num_hidden_layers");
    }
    if config.first_k_dense_replace > config.num_hidden_layers {
        return invalid("first_k_dense_replace", "must not exceed num_hidden_layers");
    }
    require_nonzero("moe_layer_freq", config.moe_layer_freq)?;
    if config.moe_layer_freq > config.num_hidden_layers {
        return invalid("moe_layer_freq", "must not exceed num_hidden_layers");
    }
    if config.num_key_value_heads != config.num_attention_heads {
        return invalid(
            "num_key_value_heads",
            format!(
                "must equal num_attention_heads={} for the released MLA expansion, got {}",
                config.num_attention_heads, config.num_key_value_heads
            ),
        );
    }
    if config.num_nextn_predict_layers != 0 {
        return invalid(
            "num_nextn_predict_layers",
            "the released base checkpoint does not contain MTP layers",
        );
    }

    for (field, enabled) in [
        (
            "linear_attn_config.use_full_rank_gate",
            config.linear_attn_config.use_full_rank_gate,
        ),
        ("mla_use_nope", config.mla_use_nope),
        ("mla_use_output_gate", config.mla_use_output_gate),
        ("use_cache", config.use_cache),
        ("use_grouped_topk", config.use_grouped_topk),
        ("moe_renormalize", config.moe_renormalize),
        ("latent_moe_use_norm", config.latent_moe_use_norm),
    ] {
        if !enabled {
            return invalid(field, "must be enabled for the supported Kimi K3 formulas");
        }
    }
    if config.hidden_act != "situ" {
        return invalid("hidden_act", "must be `situ`");
    }
    if config.moe_router_activation_func != "sigmoid" {
        return invalid("moe_router_activation_func", "must be `sigmoid`");
    }
    if config.topk_method != "noaux_tc" {
        return invalid("topk_method", "must be `noaux_tc`");
    }
    if config.qk_rope_head_dim % 2 != 0 {
        return invalid(
            "qk_rope_head_dim",
            "the retained NoPE slot width must still be even",
        );
    }
    Ok(())
}

fn build_kda_geometry(config: &KimiK3TextConfig) -> Result<KdaGeometry, GeometryError> {
    let linear = &config.linear_attn_config;
    for (field, value) in [
        ("linear_attn_config.num_heads", linear.num_heads),
        ("linear_attn_config.head_dim", linear.head_dim),
        (
            "linear_attn_config.short_conv_kernel_size",
            linear.short_conv_kernel_size,
        ),
    ] {
        require_nonzero(field, value)?;
    }
    if linear.num_heads != config.num_attention_heads {
        return invalid(
            "linear_attn_config.num_heads",
            format!(
                "must equal num_attention_heads={}, got {}",
                config.num_attention_heads, linear.num_heads
            ),
        );
    }
    if linear.head_dim < linear.num_heads {
        return invalid(
            "linear_attn_config.head_dim",
            "must contain at least num_heads entries for the padded A_log tensor",
        );
    }
    if linear.head_dim != config.v_head_dim {
        return invalid(
            "linear_attn_config.head_dim",
            format!(
                "must equal v_head_dim={}, got {}",
                config.v_head_dim, linear.head_dim
            ),
        );
    }
    let gate_lower_bound = narrow_strictly_negative(
        "linear_attn_config.gate_lower_bound",
        linear.gate_lower_bound,
    )?;
    let rms_epsilon = narrow_strictly_positive("rms_norm_eps", config.rms_norm_eps)?;

    // Validate every shape expression needed by the scalar component before exposing it.
    let channels = checked_mul(
        linear.num_heads,
        linear.head_dim,
        "KDA num_heads * head_dim",
    )?;
    checked_mul(
        channels,
        linear.short_conv_kernel_size,
        "KDA channels * conv_kernel_size",
    )?;
    checked_product(
        &[linear.num_heads, linear.head_dim, linear.head_dim],
        "KDA num_heads * head_dim * head_dim",
    )?;

    Ok(KdaGeometry {
        hidden_size: config.hidden_size,
        num_heads: linear.num_heads,
        head_dim: linear.head_dim,
        conv_kernel_size: linear.short_conv_kernel_size,
        rms_epsilon,
        gate_lower_bound,
    })
}

fn build_mla_geometry(config: &KimiK3TextConfig) -> Result<MlaGeometry, GeometryError> {
    for (field, value) in [
        ("q_lora_rank", config.q_lora_rank),
        ("kv_lora_rank", config.kv_lora_rank),
        ("qk_nope_head_dim", config.qk_nope_head_dim),
        ("qk_rope_head_dim", config.qk_rope_head_dim),
        ("v_head_dim", config.v_head_dim),
    ] {
        require_nonzero(field, value)?;
    }
    let rms_epsilon = narrow_strictly_positive("rms_norm_eps", config.rms_norm_eps)?;
    let query_head = checked_add(
        config.qk_nope_head_dim,
        config.qk_rope_head_dim,
        "MLA qk_nope_head_dim + qk_rope_head_dim",
    )?;
    checked_mul(
        config.num_attention_heads,
        query_head,
        "MLA num_attention_heads * query_head_dim",
    )?;
    let kv_head = checked_add(
        config.qk_nope_head_dim,
        config.v_head_dim,
        "MLA qk_nope_head_dim + v_head_dim",
    )?;
    checked_mul(
        config.num_attention_heads,
        kv_head,
        "MLA num_attention_heads * expanded_kv_head_dim",
    )?;
    checked_add(
        config.kv_lora_rank,
        config.qk_rope_head_dim,
        "MLA kv_lora_rank + shared NoPE slot",
    )?;
    checked_mul(
        config.num_attention_heads,
        config.v_head_dim,
        "MLA num_attention_heads * v_head_dim",
    )?;

    Ok(MlaGeometry {
        hidden_size: config.hidden_size,
        num_heads: config.num_attention_heads,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        qk_nope_head_dim: config.qk_nope_head_dim,
        qk_nope_slot_dim: config.qk_rope_head_dim,
        value_head_dim: config.v_head_dim,
        rms_epsilon,
    })
}

fn build_latent_moe_geometry(
    config: &KimiK3TextConfig,
) -> Result<LatentMoeGeometry, GeometryError> {
    for (field, value) in [
        (
            "routed_expert_hidden_size",
            config.routed_expert_hidden_size,
        ),
        ("moe_intermediate_size", config.moe_intermediate_size),
        ("num_shared_experts", config.num_shared_experts),
        ("num_experts", config.num_experts),
        ("num_experts_per_token", config.num_experts_per_token),
        ("num_expert_group", config.num_expert_group),
        ("topk_group", config.topk_group),
    ] {
        require_nonzero(field, value)?;
    }
    let geometry = LatentMoeGeometry {
        hidden_size: config.hidden_size,
        latent_size: config.routed_expert_hidden_size,
        expert_intermediate_size: config.moe_intermediate_size,
        shared_expert_count: config.num_shared_experts,
        routing: NoAuxTcConfig {
            expert_count: config.num_experts,
            top_k: config.num_experts_per_token,
            expert_group_count: config.num_expert_group,
            selected_group_count: config.topk_group,
            routed_scaling_factor: narrow_strictly_positive(
                "routed_scaling_factor",
                config.routed_scaling_factor,
            )?,
        },
        rms_norm_epsilon: narrow_strictly_positive("rms_norm_eps", config.rms_norm_eps)?,
        situ_beta: narrow_strictly_positive("activation_situ_beta", config.activation_situ_beta)?,
        situ_linear_beta: narrow_strictly_positive(
            "activation_situ_linear_beta",
            config.activation_situ_linear_beta,
        )?,
    };
    geometry.validate().map_err(map_moe_error)?;
    Ok(geometry)
}

fn attention_layer_assignment(
    config: &KimiK3TextConfig,
) -> Result<Vec<AttentionLayerKind>, GeometryError> {
    let kda = &config.linear_attn_config.kda_layers;
    let mla = &config.linear_attn_config.full_attn_layers;
    if kda.is_empty() || mla.is_empty() {
        return invalid(
            "linear_attn_config layer assignment",
            "KDA and MLA layer lists must both be non-empty",
        );
    }
    validate_one_based_layer_list(
        "linear_attn_config.kda_layers",
        kda,
        config.num_hidden_layers,
    )?;
    validate_one_based_layer_list(
        "linear_attn_config.full_attn_layers",
        mla,
        config.num_hidden_layers,
    )?;
    let assigned = checked_add(kda.len(), mla.len(), "KDA layer count + MLA layer count")?;
    if assigned != config.num_hidden_layers {
        return invalid(
            "linear_attn_config layer assignment",
            format!(
                "assigns {assigned} entries for {} decoder layers",
                config.num_hidden_layers
            ),
        );
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(config.num_hidden_layers)
        .map_err(|_| GeometryError::Allocation {
            value: "attention layer assignment",
            elements: config.num_hidden_layers,
        })?;
    let mut kda_index = 0;
    let mut mla_index = 0;
    for layer in 1..=config.num_hidden_layers {
        let is_kda = kda.get(kda_index).copied() == Some(layer);
        let is_mla = mla.get(mla_index).copied() == Some(layer);
        match (is_kda, is_mla) {
            (true, false) => {
                output.push(AttentionLayerKind::Kda);
                kda_index += 1;
            }
            (false, true) => {
                output.push(AttentionLayerKind::Mla);
                mla_index += 1;
            }
            (true, true) => {
                return invalid(
                    "linear_attn_config layer assignment",
                    format!("one-based layer {layer} is assigned to both KDA and MLA"),
                );
            }
            (false, false) => {
                return invalid(
                    "linear_attn_config layer assignment",
                    format!("one-based layer {layer} is unassigned"),
                );
            }
        }
    }
    Ok(output)
}

fn validate_one_based_layer_list(
    field: &'static str,
    layers: &[usize],
    layer_count: usize,
) -> Result<(), GeometryError> {
    if layers.windows(2).any(|pair| pair[0] >= pair[1]) {
        return invalid(field, "must be strictly increasing and unique");
    }
    if let Some(layer) = layers
        .iter()
        .copied()
        .find(|&layer| layer == 0 || layer > layer_count)
    {
        return invalid(
            field,
            format!("contains one-based layer {layer} outside 1..={layer_count}"),
        );
    }
    Ok(())
}

fn build_sequence_state(
    config: &KimiK3TextConfig,
    layers: &[AttentionLayerKind],
    kda: KdaGeometry,
    mla: MlaGeometry,
) -> Result<SequenceStateGeometry, GeometryError> {
    let kda_layer_count = layers
        .iter()
        .filter(|&&kind| kind == AttentionLayerKind::Kda)
        .count();
    let mla_layer_count = layers
        .iter()
        .filter(|&&kind| kind == AttentionLayerKind::Mla)
        .count();

    let kda_recurrent_f32_per_layer = checked_product(
        &[kda.num_heads, kda.head_dim, kda.head_dim],
        "KDA recurrent F32 per layer",
    )?;
    let channels = checked_mul(
        kda.num_heads,
        kda.head_dim,
        "KDA num_heads * head_dim for convolution state",
    )?;
    let kda_conv_history_f32_per_layer = checked_product(
        &[3, channels, kda.conv_kernel_size - 1],
        "three KDA convolution histories per layer",
    )?;
    let kda_state_f32_per_layer = checked_add(
        kda_recurrent_f32_per_layer,
        kda_conv_history_f32_per_layer,
        "KDA recurrent + convolution state per layer",
    )?;
    let kda_recurrent_f32_per_sequence = checked_mul(
        kda_recurrent_f32_per_layer,
        kda_layer_count,
        "KDA recurrent F32 per layer * KDA layers",
    )?;
    let kda_conv_history_f32_per_sequence = checked_mul(
        kda_conv_history_f32_per_layer,
        kda_layer_count,
        "KDA convolution F32 per layer * KDA layers",
    )?;
    let kda_state_f32_per_sequence = checked_add(
        kda_recurrent_f32_per_sequence,
        kda_conv_history_f32_per_sequence,
        "KDA recurrent + convolution state per sequence",
    )?;

    let mla_cache_f32_per_token_per_layer = checked_add(
        mla.kv_lora_rank,
        mla.qk_nope_slot_dim,
        "MLA latent + shared NoPE slot per token per layer",
    )?;
    let mla_cache_f32_per_token = checked_mul(
        mla_cache_f32_per_token_per_layer,
        mla_layer_count,
        "MLA cache F32 per token per layer * MLA layers",
    )?;

    // Boundaries fire at zero-based layers 0, block, 2*block, ... below layer_count.
    let attn_res_max_snapshots = checked_add(
        (config.num_hidden_layers - 1) / config.attn_res_block_size,
        1,
        "AttnRes boundary quotient + layer-zero snapshot",
    )?;
    let attn_res_max_sources = checked_add(
        attn_res_max_snapshots,
        1,
        "AttnRes snapshots + current prefix",
    )?;

    Ok(SequenceStateGeometry {
        kda_layer_count,
        mla_layer_count,
        kda_recurrent_f32_per_layer,
        kda_conv_history_f32_per_layer,
        kda_state_f32_per_layer,
        kda_recurrent_f32_per_sequence,
        kda_conv_history_f32_per_sequence,
        kda_state_f32_per_sequence,
        mla_cache_f32_per_token_per_layer,
        mla_cache_f32_per_token,
        attn_res_max_snapshots,
        attn_res_max_sources,
    })
}

fn map_moe_error(error: KimiK3MoeError) -> GeometryError {
    match error {
        KimiK3MoeError::Overflow { expression } => GeometryError::Overflow { expression },
        error => GeometryError::Component {
            component: "LatentMoE",
            message: error.to_string(),
        },
    }
}

fn narrow_strictly_positive(field: &'static str, value: f64) -> Result<f32, GeometryError> {
    let narrowed = value as f32;
    if value.is_finite() && value > 0.0 && narrowed.is_finite() && narrowed > 0.0 {
        Ok(narrowed)
    } else {
        invalid(
            field,
            "must be finite, strictly positive, and representable as positive F32",
        )
    }
}

fn narrow_strictly_negative(field: &'static str, value: f64) -> Result<f32, GeometryError> {
    let narrowed = value as f32;
    if value.is_finite() && value < 0.0 && narrowed.is_finite() && narrowed < 0.0 {
        Ok(narrowed)
    } else {
        invalid(
            field,
            "must be finite, strictly negative, and representable as negative F32",
        )
    }
}

fn require_nonzero(field: &'static str, value: usize) -> Result<(), GeometryError> {
    if value == 0 {
        invalid(field, "must be non-zero")
    } else {
        Ok(())
    }
}

fn checked_product(factors: &[usize], expression: &'static str) -> Result<usize, GeometryError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        checked_mul(product, factor, expression)
    })
}

fn checked_mul(
    left: usize,
    right: usize,
    expression: &'static str,
) -> Result<usize, GeometryError> {
    left.checked_mul(right)
        .ok_or(GeometryError::Overflow { expression })
}

fn checked_add(
    left: usize,
    right: usize,
    expression: &'static str,
) -> Result<usize, GeometryError> {
    left.checked_add(right)
        .ok_or(GeometryError::Overflow { expression })
}

fn invalid<T>(field: &'static str, reason: impl Into<String>) -> Result<T, GeometryError> {
    Err(GeometryError::Invalid {
        field,
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::kimi_k3::config::{
        KimiK3AutoMap, KimiK3LinearAttentionConfig, KimiK3QuantizationConfig,
        KimiK3QuantizationGroup, KimiK3QuantizedWeights,
    };
    use std::collections::BTreeMap;

    fn official_text_config() -> KimiK3TextConfig {
        let full_attn_layers = (4..=92)
            .step_by(4)
            .chain(std::iter::once(93))
            .collect::<Vec<_>>();
        let kda_layers = (1..=93)
            .filter(|layer| !full_attn_layers.contains(layer))
            .collect();
        let auto_map = KimiK3AutoMap {
            auto_config: "configuration_kimi_k3.KimiLinearConfig".to_owned(),
            auto_model: "modeling_kimi_linear.KimiLinearModel".to_owned(),
            auto_model_for_causal_lm: "modeling_kimi_linear.KimiLinearForCausalLM".to_owned(),
        };
        let quantization_config = KimiK3QuantizationConfig {
            config_groups: BTreeMap::from([(
                "group_0".to_owned(),
                KimiK3QuantizationGroup {
                    format: "mxfp4-pack-quantized".to_owned(),
                    targets: vec!["Linear".to_owned()],
                    weights: KimiK3QuantizedWeights {
                        dynamic: false,
                        group_size: 32,
                        num_bits: 4,
                        observer: "minmax".to_owned(),
                        scale_dtype: "torch.uint8".to_owned(),
                        strategy: "group".to_owned(),
                        symmetric: true,
                        weight_type: "float".to_owned(),
                    },
                },
            )]),
            format: "mxfp4-pack-quantized".to_owned(),
            ignore: Vec::new(),
            quant_method: "compressed-tensors".to_owned(),
            quantization_status: "compressed".to_owned(),
        };
        KimiK3TextConfig {
            model_type: "kimi_linear".to_owned(),
            architectures: vec!["KimiLinearForCausalLM".to_owned()],
            auto_map,
            hidden_size: 7_168,
            num_hidden_layers: 93,
            num_attention_heads: 96,
            num_key_value_heads: 96,
            vocab_size: 163_840,
            intermediate_size: 33_792,
            hidden_act: "situ".to_owned(),
            activation_situ_beta: 4.0,
            activation_situ_linear_beta: 25.0,
            initializer_range: 0.02,
            rms_norm_eps: 1.0e-5,
            max_position_embeddings: 1_048_576,
            attn_res_block_size: 12,
            linear_attn_config: KimiK3LinearAttentionConfig {
                full_attn_layers,
                kda_layers,
                gate_lower_bound: -5.0,
                head_dim: 128,
                num_heads: 96,
                short_conv_kernel_size: 4,
                use_full_rank_gate: true,
            },
            first_k_dense_replace: 1,
            moe_intermediate_size: 3_072,
            moe_layer_freq: 1,
            num_experts: 896,
            num_experts_per_token: 16,
            num_shared_experts: 2,
            num_expert_group: 1,
            topk_group: 1,
            topk_method: "noaux_tc".to_owned(),
            use_grouped_topk: true,
            moe_renormalize: true,
            moe_router_activation_func: "sigmoid".to_owned(),
            routed_scaling_factor: 1.0,
            routed_expert_hidden_size: 3_584,
            latent_moe_use_norm: true,
            q_lora_rank: 1_536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            mla_use_nope: true,
            mla_use_output_gate: true,
            quantization_config,
            bos_token_id: 163_584,
            eos_token_id: 163_586,
            pad_token_id: 163_839,
            dtype: "bfloat16".to_owned(),
            tie_word_embeddings: false,
            use_cache: true,
            num_nextn_predict_layers: 0,
        }
    }

    fn assert_invalid_field(error: GeometryError, expected: &'static str) {
        match error {
            GeometryError::Invalid { field, .. } => assert_eq!(field, expected),
            other => panic!("expected invalid field {expected}, got {other}"),
        }
    }

    #[test]
    fn official_component_geometry_and_resource_counts_are_exact() {
        let config = official_text_config();
        let geometry = runtime_geometry_from_config(&config).unwrap();
        assert_eq!(
            geometry.kda,
            KdaGeometry {
                hidden_size: 7_168,
                num_heads: 96,
                head_dim: 128,
                conv_kernel_size: 4,
                rms_epsilon: 1.0e-5,
                gate_lower_bound: -5.0,
            }
        );
        assert_eq!(
            geometry.mla,
            MlaGeometry {
                hidden_size: 7_168,
                num_heads: 96,
                q_lora_rank: 1_536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_nope_slot_dim: 64,
                value_head_dim: 128,
                rms_epsilon: 1.0e-5,
            }
        );
        assert_eq!(geometry.latent_moe, LatentMoeGeometry::KIMI_K3);
        assert_eq!(geometry.attention_layers.len(), 93);
        assert_eq!(geometry.layer_kind(0), Some(AttentionLayerKind::Kda));
        assert_eq!(geometry.layer_kind(3), Some(AttentionLayerKind::Mla));
        assert_eq!(geometry.layer_kind(91), Some(AttentionLayerKind::Mla));
        assert_eq!(geometry.layer_kind(92), Some(AttentionLayerKind::Mla));
        assert_eq!(geometry.layer_kind(93), None);

        let state = geometry.sequence_state;
        assert_eq!(state.kda_layer_count, 69);
        assert_eq!(state.mla_layer_count, 24);
        assert_eq!(state.kda_recurrent_f32_per_layer, 1_572_864);
        assert_eq!(state.kda_conv_history_f32_per_layer, 110_592);
        assert_eq!(state.kda_state_f32_per_layer, 1_683_456);
        assert_eq!(state.kda_recurrent_f32_per_sequence, 108_527_616);
        assert_eq!(state.kda_conv_history_f32_per_sequence, 7_630_848);
        assert_eq!(state.kda_state_f32_per_sequence, 116_158_464);
        assert_eq!(state.mla_cache_f32_per_token_per_layer, 576);
        assert_eq!(state.mla_cache_f32_per_token, 13_824);
        assert_eq!(state.attn_res_max_snapshots, 8);
        assert_eq!(state.attn_res_max_sources, 9);

        assert_eq!(kda_geometry_from_config(&config).unwrap(), geometry.kda);
        assert_eq!(mla_geometry_from_config(&config).unwrap(), geometry.mla);
        assert_eq!(
            latent_moe_geometry_from_config(&config).unwrap(),
            geometry.latent_moe
        );
    }

    #[test]
    fn attn_res_counts_layer_zero_and_partial_final_block() {
        let mut config = official_text_config();
        config.num_hidden_layers = 13;
        config.attn_res_block_size = 3;
        config.first_k_dense_replace = 1;
        config.linear_attn_config.full_attn_layers = vec![4, 8, 12, 13];
        config.linear_attn_config.kda_layers = (1..=13)
            .filter(|layer| !config.linear_attn_config.full_attn_layers.contains(layer))
            .collect();
        let state = runtime_geometry_from_config(&config)
            .unwrap()
            .sequence_state;
        assert_eq!(state.kda_layer_count, 9);
        assert_eq!(state.mla_layer_count, 4);
        // Zero-based boundaries 0, 3, 6, 9, 12.
        assert_eq!(state.attn_res_max_snapshots, 5);
        assert_eq!(state.attn_res_max_sources, 6);
    }

    #[test]
    fn every_execution_semantics_flag_is_revalidated() {
        type Mutation = fn(&mut KimiK3TextConfig);
        let cases: [(&str, Mutation); 7] = [
            ("linear_attn_config.use_full_rank_gate", |config| {
                config.linear_attn_config.use_full_rank_gate = false
            }),
            ("mla_use_nope", |config| config.mla_use_nope = false),
            ("mla_use_output_gate", |config| {
                config.mla_use_output_gate = false
            }),
            ("use_cache", |config| config.use_cache = false),
            ("use_grouped_topk", |config| config.use_grouped_topk = false),
            ("moe_renormalize", |config| config.moe_renormalize = false),
            ("latent_moe_use_norm", |config| {
                config.latent_moe_use_norm = false
            }),
        ];
        for (field, mutate) in cases {
            let mut config = official_text_config();
            mutate(&mut config);
            assert_invalid_field(runtime_geometry_from_config(&config).unwrap_err(), field);
        }
    }

    #[test]
    fn malformed_layer_partitions_are_rejected_without_partial_geometry() {
        let mut config = official_text_config();
        config.linear_attn_config.kda_layers.swap(0, 1);
        assert_invalid_field(
            runtime_geometry_from_config(&config).unwrap_err(),
            "linear_attn_config.kda_layers",
        );

        let mut config = official_text_config();
        config.linear_attn_config.kda_layers.pop();
        assert_invalid_field(
            runtime_geometry_from_config(&config).unwrap_err(),
            "linear_attn_config layer assignment",
        );

        let mut config = official_text_config();
        let last = config.linear_attn_config.kda_layers.len() - 1;
        config.linear_attn_config.kda_layers[last] = 92;
        let error = runtime_geometry_from_config(&config).unwrap_err();
        assert!(error.to_string().contains("unassigned") || error.to_string().contains("both"));
    }

    #[test]
    fn invalid_numeric_geometry_and_overflow_are_reported() {
        let mut config = official_text_config();
        config.linear_attn_config.gate_lower_bound = 0.0;
        assert_invalid_field(
            runtime_geometry_from_config(&config).unwrap_err(),
            "linear_attn_config.gate_lower_bound",
        );

        let mut config = official_text_config();
        config.rms_norm_eps = f64::MIN_POSITIVE;
        assert_invalid_field(
            runtime_geometry_from_config(&config).unwrap_err(),
            "rms_norm_eps",
        );

        let mut config = official_text_config();
        config.num_experts_per_token = config.num_experts + 1;
        assert!(matches!(
            runtime_geometry_from_config(&config),
            Err(GeometryError::Component {
                component: "LatentMoE",
                ..
            })
        ));

        let mut config = official_text_config();
        config.qk_nope_head_dim = usize::MAX;
        assert!(matches!(
            runtime_geometry_from_config(&config),
            Err(GeometryError::Overflow { .. })
        ));
        assert!(matches!(
            checked_product(&[usize::MAX, 2], "test product"),
            Err(GeometryError::Overflow {
                expression: "test product"
            })
        ));
    }
}
