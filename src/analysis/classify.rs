use crate::config::GlmConfig;
use serde::Serialize;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterCategory {
    Embedding,
    LmHead,
    Attention,
    Indexer,
    DenseMlp,
    Router,
    SharedExpert,
    RoutedExpert,
    Normalization,
    Mtp,
    Other,
}

impl fmt::Display for ParameterCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Embedding => "embedding",
            Self::LmHead => "lm_head",
            Self::Attention => "attention",
            Self::Indexer => "dsa_indexer",
            Self::DenseMlp => "dense_mlp",
            Self::Router => "router",
            Self::SharedExpert => "shared_expert",
            Self::RoutedExpert => "routed_expert",
            Self::Normalization => "normalization",
            Self::Mtp => "mtp",
            Self::Other => "other",
        })
    }
}

pub fn classify_tensor(name: &str, config: &GlmConfig) -> ParameterCategory {
    let name = name.strip_suffix(".qs").unwrap_or(name);
    if name.starts_with("mtp.")
        || parse_layer(name).is_some_and(|layer| layer >= config.num_hidden_layers)
    {
        return ParameterCategory::Mtp;
    }
    if name == "model.embed_tokens.weight" {
        ParameterCategory::Embedding
    } else if name == "lm_head.weight" {
        ParameterCategory::LmHead
    } else if name.contains(".self_attn.indexer.") || name.contains(".self_attn.indexers_proj") {
        ParameterCategory::Indexer
    } else if name.contains(".self_attn.") {
        ParameterCategory::Attention
    } else if name.contains(".mlp.experts.") {
        ParameterCategory::RoutedExpert
    } else if name.contains(".mlp.shared_experts.") {
        ParameterCategory::SharedExpert
    } else if name.contains(".mlp.gate.") {
        ParameterCategory::Router
    } else if name.contains(".mlp.") {
        ParameterCategory::DenseMlp
    } else if name.contains("norm") || name.contains("layernorm") {
        ParameterCategory::Normalization
    } else {
        ParameterCategory::Other
    }
}

/// Returns the logical row-major `[output, input]` matrix shape implied by GLM-5.2.
///
/// Converted Colibrì tensors deliberately store packed U8 payload shape rather than this
/// logical shape, so name + validated config are the authoritative source here.
pub fn expected_matrix_shape(name: &str, config: &GlmConfig) -> Option<(u64, u64)> {
    let name = name.strip_suffix(".qs").unwrap_or(name);
    let h = config.hidden_size as u64;
    let heads = config.num_attention_heads as u64;
    let q_lora = config.q_lora_rank_value() as u64;
    let kv_lora = config.kv_lora_rank as u64;
    let qk = config.qk_head_dim as u64;
    let rope = config.qk_rope_head_dim as u64;
    let nope = config.qk_nope_head_dim as u64;
    let value = config.v_head_dim as u64;

    if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        return Some((config.vocab_size as u64, h));
    }
    if name.ends_with(".eh_proj.weight") {
        // Native MTP fuses normalized token embedding and previous hidden state: [2D] -> [D].
        return Some((h, 2 * h));
    }
    if name.ends_with(".self_attn.q_a_proj.weight") {
        return Some((q_lora, h));
    }
    if name.ends_with(".self_attn.q_b_proj.weight") {
        return Some((heads * qk, q_lora));
    }
    if name.ends_with(".self_attn.kv_a_proj_with_mqa.weight") {
        return Some((kv_lora + rope, h));
    }
    if name.ends_with(".self_attn.kv_b_proj.weight") {
        return Some((heads * (nope + value), kv_lora));
    }
    if name.ends_with(".self_attn.o_proj.weight") {
        return Some((h, heads * value));
    }
    if name.ends_with(".self_attn.indexer.wq_b.weight") {
        return Some((
            (config.index_n_heads * config.index_head_dim) as u64,
            q_lora,
        ));
    }
    if name.ends_with(".self_attn.indexer.wk.weight") {
        return Some((config.index_head_dim as u64, h));
    }
    if name.ends_with(".self_attn.indexer.weights_proj.weight")
        || name.ends_with(".self_attn.indexers_proj.weight")
    {
        return Some((config.index_n_heads as u64, h));
    }

    let layer = parse_layer(name)?;
    let is_mtp = layer >= config.num_hidden_layers;
    if name.ends_with(".mlp.gate.weight") {
        return Some((config.n_routed_experts as u64, h));
    }
    if name.contains(".mlp.experts.") {
        if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
            return Some((config.moe_intermediate_size as u64, h));
        }
        if name.ends_with(".gate_up_proj.weight") {
            return Some(((2 * config.moe_intermediate_size) as u64, h));
        }
        if name.ends_with(".down_proj.weight") {
            return Some((h, config.moe_intermediate_size as u64));
        }
    }
    if name.contains(".mlp.shared_experts.") {
        let shared = (config.moe_intermediate_size * config.n_shared_experts) as u64;
        if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
            return Some((shared, h));
        }
        if name.ends_with(".down_proj.weight") {
            return Some((h, shared));
        }
    }
    if !is_mtp && !config.layer_is_sparse(layer) {
        if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
            return Some((config.intermediate_size as u64, h));
        }
        if name.ends_with(".mlp.down_proj.weight") {
            return Some((h, config.intermediate_size as u64));
        }
    }
    None
}

pub(crate) fn expected_logical_elements(name: &str, config: &GlmConfig) -> Option<u64> {
    if let Some((rows, cols)) = expected_matrix_shape(name, config) {
        return rows.checked_mul(cols);
    }
    let name = name.strip_suffix(".qs").unwrap_or(name);
    if name == "model.norm.weight" {
        return Some(config.hidden_size as u64);
    }
    if name.ends_with(".enorm.weight")
        || name.ends_with(".hnorm.weight")
        || name.ends_with(".shared_head.norm.weight")
    {
        return Some(config.hidden_size as u64);
    }
    if name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
    {
        return Some(config.hidden_size as u64);
    }
    if name.ends_with(".self_attn.q_a_layernorm.weight") {
        return Some(config.q_lora_rank_value() as u64);
    }
    if name.ends_with(".self_attn.kv_a_layernorm.weight") {
        return Some(config.kv_lora_rank as u64);
    }
    if name.ends_with(".mlp.gate.e_score_correction_bias") {
        return Some(config.n_routed_experts as u64);
    }
    if name.ends_with(".self_attn.indexer.k_norm.weight")
        || name.ends_with(".self_attn.indexer.k_norm.bias")
    {
        return Some(config.index_head_dim as u64);
    }
    None
}

/// Resolves the grouped-scale layouts supported by the planned Colibrì-compatible loader.
/// A single scale per row is represented by `cols`; otherwise the scale cardinality must identify
/// exactly one supported group size. Tail groups can make cardinality ambiguous, which is rejected
/// instead of guessed.
pub(crate) fn supported_group_size(rows: u64, cols: u64, scale_count: u64) -> Option<u64> {
    crate::storage::infer_int4_group_size(rows, cols, scale_count).ok()
}

pub(crate) fn parse_layer(name: &str) -> Option<usize> {
    let tail = name.strip_prefix("model.layers.")?;
    tail.split('.').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GlmConfig {
        GlmConfig::from_json_str(
            &serde_json::json!({
                "model_type":"glm_moe_dsa", "hidden_size":8, "num_hidden_layers":2,
                "num_attention_heads":2, "vocab_size":16, "intermediate_size":12,
                "moe_intermediate_size":4, "first_k_dense_replace":1,
                "n_routed_experts":8, "n_shared_experts":1, "num_experts_per_tok":2,
                "q_lora_rank":4, "kv_lora_rank":3, "qk_nope_head_dim":2,
                "qk_rope_head_dim":2, "qk_head_dim":4, "v_head_dim":3,
                "max_position_embeddings":64,
                "hidden_act":"silu", "scoring_func":"sigmoid", "topk_method":"noaux_tc"
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn recovers_shapes_hidden_by_packed_container() {
        let cfg = config();
        assert_eq!(
            expected_matrix_shape("model.layers.1.mlp.experts.3.gate_proj.weight", &cfg),
            Some((4, 8))
        );
        assert_eq!(
            expected_matrix_shape("model.layers.0.self_attn.q_b_proj.weight", &cfg),
            Some((8, 4))
        );
    }

    #[test]
    fn mtp_layer_is_not_mixed_into_base_experts() {
        let cfg = config();
        assert_eq!(
            classify_tensor("model.layers.2.mlp.experts.0.down_proj.weight", &cfg),
            ParameterCategory::Mtp
        );
    }

    #[test]
    fn group_size_resolution_rejects_arbitrary_scale_counts() {
        assert_eq!(supported_group_size(2, 130, 2), Some(130));
        assert_eq!(supported_group_size(2, 130, 6), None);
        assert_eq!(supported_group_size(2, 130, 10), Some(32));
        assert_eq!(supported_group_size(2, 130, 8), None);
    }
}
