//! Header-only validation of the released Kimi-K3 checkpoint tensor ABI.
//!
//! The validator deliberately checks rank, shape, dtype, and payload length without reading
//! tensor payloads.  The full preflight requires the exact multimodal tensor set.  The partial
//! preflight is intended for an in-progress rsync: [`TensorIndex`] only opens files whose names
//! end in `.safetensors`, and every decoder layer that is visible is still validated in full.

use super::KimiK3Config;
use crate::storage::{DType, SafetensorError, TensorIndex};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;

pub const TEXT_ROOT_TENSOR_COUNT: usize = 5;
pub const RELEASE_KDA_DENSE_LAYER_TENSOR_COUNT: usize = 23;
pub const RELEASE_KDA_MOE_LAYER_TENSOR_COUNT: usize = 5_404;
pub const RELEASE_MLA_MOE_LAYER_TENSOR_COUNT: usize = 5_398;
pub const PROJECTOR_TENSOR_COUNT: usize = 3;
pub const RELEASE_VISION_TENSOR_COUNT: usize = 165;
pub const RELEASE_TENSOR_COUNT: usize = 497_220;

const TEXT_PREFIX: &str = "language_model.model.";
const LAYER_PREFIX: &str = "language_model.model.layers.";
const MXFP4_GROUP_SIZE: usize = 32;

#[derive(Debug)]
pub enum SchemaError {
    Checkpoint(SafetensorError),
    Invalid(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "invalid Kimi-K3 checkpoint: {reason}"),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<SafetensorError> for SchemaError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

/// Exact header-only result for a complete text-and-vision Kimi-K3 checkpoint.
#[derive(Debug, Clone, Serialize)]
pub struct KimiK3Requirements {
    pub required_tensor_count: usize,
    pub checkpoint_tensor_count: usize,
    pub checkpoint_shard_count: usize,
    pub text_root_tensor_count: usize,
    pub decoder_tensor_count: usize,
    pub projector_tensor_count: usize,
    pub vision_tensor_count: usize,
    pub kda_layer_count: usize,
    pub mla_layer_count: usize,
    pub dense_layer_count: usize,
    pub moe_layer_count: usize,
    /// Logical model parameters. MXFP4 scale sidecars are storage metadata, not parameters.
    pub logical_parameter_count: u64,
    pub checkpoint_payload_bytes: u64,
}

/// Result of validating all decoder layers currently visible to [`TensorIndex`].
#[derive(Debug, Clone, Serialize)]
pub struct KimiK3PartialLayers {
    /// Zero-based decoder layer IDs that were present and validated in full.
    pub validated_layers: Vec<usize>,
    pub validated_tensor_count: usize,
    pub checkpoint_tensor_count: usize,
    pub non_layer_tensor_count: usize,
    pub checkpoint_shard_count: usize,
}

#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    dtype: DType,
    shape: Vec<u64>,
    logical_parameters: u64,
}

impl TensorSpec {
    fn plain(name: impl Into<String>, dtype: DType, shape: &[usize]) -> Result<Self, SchemaError> {
        let shape = checked_shape(shape)?;
        let logical_parameters = checked_product_u64(&shape, "tensor elements")?;
        Ok(Self {
            name: name.into(),
            dtype,
            shape,
            logical_parameters,
        })
    }

    fn storage(
        name: impl Into<String>,
        dtype: DType,
        shape: &[usize],
        logical_parameters: u64,
    ) -> Result<Self, SchemaError> {
        Ok(Self {
            name: name.into(),
            dtype,
            shape: checked_shape(shape)?,
            logical_parameters,
        })
    }

    fn physical_elements(&self) -> Result<u64, SchemaError> {
        checked_product_u64(&self.shape, &format!("tensor {:?} elements", self.name))
    }
}

#[derive(Debug, Clone)]
struct Geometry {
    hidden: usize,
    layers: usize,
    vocab: usize,
    dense_intermediate: usize,
    moe_intermediate: usize,
    routed_hidden: usize,
    experts: usize,
    shared_experts: usize,
    attention_heads: usize,
    kda_heads: usize,
    kda_head_dim: usize,
    conv_kernel: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    first_dense_layers: usize,
    is_kda: Vec<bool>,
    vision_layers: usize,
    vision_hidden: usize,
    vision_intermediate: usize,
    vision_qkv_hidden: usize,
    vision_pos_height: usize,
    vision_pos_width: usize,
    vision_patch: usize,
    merge_height: usize,
    merge_width: usize,
}

impl Geometry {
    fn from_config(config: &KimiK3Config) -> Result<Self, SchemaError> {
        let text = &config.text_config;
        let vision = &config.vision_config;
        if text.routed_expert_hidden_size % MXFP4_GROUP_SIZE != 0
            || text.moe_intermediate_size % MXFP4_GROUP_SIZE != 0
        {
            return Err(SchemaError::Invalid(format!(
                "MXFP4 expert dimensions must be divisible by {MXFP4_GROUP_SIZE} (routed_hidden={}, moe_intermediate={})",
                text.routed_expert_hidden_size, text.moe_intermediate_size
            )));
        }
        if text.routed_expert_hidden_size % 2 != 0 || text.moe_intermediate_size % 2 != 0 {
            return Err(SchemaError::Invalid(
                "MXFP4 logical expert columns must be even".to_owned(),
            ));
        }
        let is_kda = (0..text.num_hidden_layers)
            .map(|layer| text.is_kda_layer(layer))
            .collect();
        let geometry = Self {
            hidden: text.hidden_size,
            layers: text.num_hidden_layers,
            vocab: text.vocab_size,
            dense_intermediate: text.intermediate_size,
            moe_intermediate: text.moe_intermediate_size,
            routed_hidden: text.routed_expert_hidden_size,
            experts: text.num_experts,
            shared_experts: text.num_shared_experts,
            attention_heads: text.num_attention_heads,
            kda_heads: text.linear_attn_config.num_heads,
            kda_head_dim: text.linear_attn_config.head_dim,
            conv_kernel: text.linear_attn_config.short_conv_kernel_size,
            q_lora: text.q_lora_rank,
            kv_lora: text.kv_lora_rank,
            qk_nope: text.qk_nope_head_dim,
            qk_rope: text.qk_rope_head_dim,
            v_head: text.v_head_dim,
            first_dense_layers: text.first_k_dense_replace,
            is_kda,
            vision_layers: vision.vt_num_hidden_layers,
            vision_hidden: vision.vt_hidden_size,
            vision_intermediate: vision.vt_intermediate_size,
            vision_qkv_hidden: vision.qkv_hidden_size,
            vision_pos_height: vision.init_pos_emb_height,
            vision_pos_width: vision.init_pos_emb_width,
            vision_patch: vision.patch_size,
            merge_height: vision.merge_kernel_size[0],
            merge_width: vision.merge_kernel_size[1],
        };
        geometry.validate_release_geometry()?;
        Ok(geometry)
    }

    fn kda_layers(&self) -> usize {
        self.is_kda.iter().filter(|&&value| value).count()
    }

    fn dense_layers(&self) -> usize {
        self.first_dense_layers.min(self.layers)
    }

    fn validate_release_geometry(&self) -> Result<(), SchemaError> {
        for (name, actual, expected) in [
            ("hidden_size", self.hidden, 7_168),
            ("num_hidden_layers", self.layers, 93),
            ("vocab_size", self.vocab, 163_840),
            ("intermediate_size", self.dense_intermediate, 33_792),
            ("moe_intermediate_size", self.moe_intermediate, 3_072),
            ("routed_expert_hidden_size", self.routed_hidden, 3_584),
            ("num_experts", self.experts, 896),
            ("num_shared_experts", self.shared_experts, 2),
            ("num_attention_heads", self.attention_heads, 96),
            ("linear_attn_config.num_heads", self.kda_heads, 96),
            ("linear_attn_config.head_dim", self.kda_head_dim, 128),
            ("short_conv_kernel_size", self.conv_kernel, 4),
            ("q_lora_rank", self.q_lora, 1_536),
            ("kv_lora_rank", self.kv_lora, 512),
            ("qk_nope_head_dim", self.qk_nope, 128),
            ("qk_rope_head_dim", self.qk_rope, 64),
            ("v_head_dim", self.v_head, 128),
            ("first_k_dense_replace", self.first_dense_layers, 1),
            ("vision.vt_num_hidden_layers", self.vision_layers, 27),
            ("vision.vt_hidden_size", self.vision_hidden, 1_024),
            (
                "vision.vt_intermediate_size",
                self.vision_intermediate,
                4_096,
            ),
            ("vision.qkv_hidden_size", self.vision_qkv_hidden, 1_536),
            ("vision.init_pos_emb_height", self.vision_pos_height, 64),
            ("vision.init_pos_emb_width", self.vision_pos_width, 64),
            ("vision.patch_size", self.vision_patch, 14),
            ("vision.merge_kernel_size[0]", self.merge_height, 2),
            ("vision.merge_kernel_size[1]", self.merge_width, 2),
        ] {
            if actual != expected {
                return Err(SchemaError::Invalid(format!(
                    "released Kimi-K3 requires {name}={expected}, got {actual}"
                )));
            }
        }
        let expected_kda: Vec<bool> = (1..=self.layers)
            .map(|one_based| one_based % 4 != 0 && one_based != 93)
            .collect();
        if self.is_kda != expected_kda {
            return Err(SchemaError::Invalid(
                "KDA/MLA layer assignment differs from the released 69-KDA/24-MLA map".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct Totals {
    tensors: usize,
    logical_parameters: u64,
}

impl Totals {
    fn add_spec(&mut self, spec: &TensorSpec) -> Result<(), SchemaError> {
        self.tensors = self
            .tensors
            .checked_add(1)
            .ok_or_else(|| SchemaError::Invalid("tensor count overflows usize".to_owned()))?;
        self.logical_parameters = self
            .logical_parameters
            .checked_add(spec.logical_parameters)
            .ok_or_else(|| {
                SchemaError::Invalid("logical parameter count overflows u64".to_owned())
            })?;
        Ok(())
    }
}

/// Validates every required tensor and rejects any extra tensor.
///
/// This is safe to run before payload loading: it reads safetensors headers only.
pub fn inspect_requirements(
    config: &KimiK3Config,
    index: &TensorIndex,
) -> Result<KimiK3Requirements, SchemaError> {
    let geometry = Geometry::from_config(config)?;
    let mut totals = Totals::default();

    let root = root_specs(&geometry)?;
    validate_specs(index, &root, &mut totals)?;

    let mut decoder_tensor_count = 0usize;
    for layer in 0..geometry.layers {
        let specs = layer_specs(&geometry, layer)?;
        decoder_tensor_count =
            checked_add_usize(decoder_tensor_count, specs.len(), "decoder tensor count")?;
        validate_specs(index, &specs, &mut totals)?;
    }

    let projector = projector_specs(&geometry)?;
    validate_specs(index, &projector, &mut totals)?;
    let vision = vision_specs(&geometry)?;
    validate_specs(index, &vision, &mut totals)?;

    let checkpoint_tensor_count = index.names().count();
    if checkpoint_tensor_count != totals.tensors {
        let relation = if checkpoint_tensor_count > totals.tensors {
            format!(
                "{} unexpected tensor(s) remain after every required tensor was found",
                checkpoint_tensor_count - totals.tensors
            )
        } else {
            // A missing required tensor normally fails in `validate_specs`; retain a clear
            // defensive error if the index implementation ever changes.
            format!(
                "checkpoint is short by {} tensor(s)",
                totals.tensors - checkpoint_tensor_count
            )
        };
        return Err(SchemaError::Invalid(format!(
            "checkpoint contains {checkpoint_tensor_count} tensors but the schema requires {}; {relation}",
            totals.tensors
        )));
    }

    Ok(KimiK3Requirements {
        required_tensor_count: totals.tensors,
        checkpoint_tensor_count,
        checkpoint_shard_count: index.shards().len(),
        text_root_tensor_count: root.len(),
        decoder_tensor_count,
        projector_tensor_count: projector.len(),
        vision_tensor_count: vision.len(),
        kda_layer_count: geometry.kda_layers(),
        mla_layer_count: geometry.layers.saturating_sub(geometry.kda_layers()),
        dense_layer_count: geometry.dense_layers(),
        moe_layer_count: geometry.layers.saturating_sub(geometry.dense_layers()),
        logical_parameter_count: totals.logical_parameters,
        checkpoint_payload_bytes: index.total_payload_bytes(),
    })
}

/// Validates every decoder layer currently present in exact `.safetensors` files.
///
/// A visible layer is never treated as best-effort: one missing, extra, mistyped, or misshaped
/// tensor fails validation. Non-layer tensors are intentionally ignored so this remains useful
/// while the official 96 shards are being transferred in order.
pub fn inspect_available_layers(
    config: &KimiK3Config,
    index: &TensorIndex,
) -> Result<KimiK3PartialLayers, SchemaError> {
    let geometry = Geometry::from_config(config)?;
    let mut layers = BTreeSet::new();
    let mut observed_layer_tensors = 0usize;
    for name in index.names() {
        if let Some(layer) = parse_layer_id(name)? {
            if layer >= geometry.layers {
                return Err(SchemaError::Invalid(format!(
                    "tensor {name:?} names layer {layer}, outside 0..{}",
                    geometry.layers
                )));
            }
            layers.insert(layer);
            observed_layer_tensors = checked_add_usize(
                observed_layer_tensors,
                1,
                "observed partial layer tensor count",
            )?;
        }
    }

    let mut validated_tensor_count = 0usize;
    let mut totals = Totals::default();
    for &layer in &layers {
        let specs = layer_specs(&geometry, layer)?;
        validated_tensor_count = checked_add_usize(
            validated_tensor_count,
            specs.len(),
            "validated partial layer tensor count",
        )?;
        validate_specs(index, &specs, &mut totals)?;
    }
    if observed_layer_tensors != validated_tensor_count {
        return Err(SchemaError::Invalid(format!(
            "visible decoder layers contain {observed_layer_tensors} tensors but their exact schemas require {validated_tensor_count}; unexpected layer tensor(s) are present"
        )));
    }

    let checkpoint_tensor_count = index.names().count();
    let non_layer_tensor_count = checkpoint_tensor_count
        .checked_sub(observed_layer_tensors)
        .ok_or_else(|| SchemaError::Invalid("partial tensor accounting underflows".to_owned()))?;
    Ok(KimiK3PartialLayers {
        validated_layers: layers.into_iter().collect(),
        validated_tensor_count,
        checkpoint_tensor_count,
        non_layer_tensor_count,
        checkpoint_shard_count: index.shards().len(),
    })
}

fn validate_specs(
    index: &TensorIndex,
    specs: &[TensorSpec],
    totals: &mut Totals,
) -> Result<(), SchemaError> {
    for spec in specs {
        let tensor = index.require(&spec.name)?;
        if tensor.dtype != spec.dtype || tensor.shape != spec.shape {
            return Err(SchemaError::Invalid(format!(
                "tensor {:?} must be {} {:?}, got {} {:?}",
                spec.name, spec.dtype, spec.shape, tensor.dtype, tensor.shape
            )));
        }
        let physical_elements = spec.physical_elements()?;
        let element_bytes = spec.dtype.element_bytes().ok_or_else(|| {
            SchemaError::Invalid(format!("tensor {:?} uses an unknown dtype", spec.name))
        })?;
        let expected_bytes = physical_elements
            .checked_mul(element_bytes)
            .ok_or_else(|| {
                SchemaError::Invalid(format!("tensor {:?} payload size overflows u64", spec.name))
            })?;
        if tensor.declared_elements != physical_elements || tensor.data_len != expected_bytes {
            return Err(SchemaError::Invalid(format!(
                "tensor {:?} metadata describes {} elements/{} bytes; expected {physical_elements} elements/{expected_bytes} bytes",
                spec.name, tensor.declared_elements, tensor.data_len
            )));
        }
        totals.add_spec(spec)?;
    }
    Ok(())
}

fn root_specs(geometry: &Geometry) -> Result<Vec<TensorSpec>, SchemaError> {
    let h = geometry.hidden;
    let v = geometry.vocab;
    Ok(vec![
        TensorSpec::plain(
            format!("{TEXT_PREFIX}embed_tokens.weight"),
            DType::Bf16,
            &[v, h],
        )?,
        TensorSpec::plain(format!("{TEXT_PREFIX}norm.weight"), DType::Bf16, &[h])?,
        TensorSpec::plain(
            format!("{TEXT_PREFIX}output_attn_res_norm.weight"),
            DType::Bf16,
            &[h],
        )?,
        TensorSpec::plain(
            format!("{TEXT_PREFIX}output_attn_res_proj.weight"),
            DType::Bf16,
            &[1, h],
        )?,
        TensorSpec::plain("language_model.lm_head.weight", DType::Bf16, &[v, h])?,
    ])
}

fn layer_specs(geometry: &Geometry, layer: usize) -> Result<Vec<TensorSpec>, SchemaError> {
    if layer >= geometry.layers {
        return Err(SchemaError::Invalid(format!(
            "cannot build schema for layer {layer}; configured layer count is {}",
            geometry.layers
        )));
    }
    let prefix = format!("{LAYER_PREFIX}{layer}");
    let h = geometry.hidden;
    let mut specs = Vec::with_capacity(layer_tensor_count(geometry, layer)?);

    for name in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attention_res_norm.weight",
        "mlp_res_norm.weight",
    ] {
        specs.push(TensorSpec::plain(
            format!("{prefix}.{name}"),
            DType::Bf16,
            &[h],
        )?);
    }
    for name in ["self_attention_res_proj.weight", "mlp_res_proj.weight"] {
        specs.push(TensorSpec::plain(
            format!("{prefix}.{name}"),
            DType::Bf16,
            &[1, h],
        )?);
    }

    if geometry.is_kda[layer] {
        append_kda_specs(&mut specs, geometry, &prefix)?;
    } else {
        append_mla_specs(&mut specs, geometry, &prefix)?;
    }
    if layer < geometry.first_dense_layers {
        append_dense_specs(&mut specs, geometry, &prefix)?;
    } else {
        append_moe_specs(&mut specs, geometry, &prefix)?;
    }

    let expected = layer_tensor_count(geometry, layer)?;
    if specs.len() != expected {
        return Err(SchemaError::Invalid(format!(
            "internal layer {layer} schema produced {} tensors; expected {expected}",
            specs.len()
        )));
    }
    Ok(specs)
}

fn append_kda_specs(
    specs: &mut Vec<TensorSpec>,
    geometry: &Geometry,
    prefix: &str,
) -> Result<(), SchemaError> {
    let h = geometry.hidden;
    let p = checked_mul_usize(
        geometry.kda_heads,
        geometry.kda_head_dim,
        "KDA projection width",
    )?;
    for projection in ["q", "k", "v", "g"] {
        specs.push(TensorSpec::plain(
            format!("{prefix}.self_attn.{projection}_proj.weight"),
            DType::Bf16,
            &[p, h],
        )?);
    }
    specs.push(TensorSpec::plain(
        format!("{prefix}.self_attn.o_proj.weight"),
        DType::Bf16,
        &[h, p],
    )?);
    for projection in ["q", "k", "v"] {
        specs.push(TensorSpec::plain(
            format!("{prefix}.self_attn.{projection}_conv1d.weight"),
            DType::F32,
            &[p, 1, geometry.conv_kernel],
        )?);
    }
    specs.push(TensorSpec::plain(
        format!("{prefix}.self_attn.f_a_proj.weight"),
        DType::Bf16,
        &[geometry.kda_head_dim, h],
    )?);
    specs.push(TensorSpec::plain(
        format!("{prefix}.self_attn.f_b_proj.weight"),
        DType::Bf16,
        &[p, geometry.kda_head_dim],
    )?);
    specs.push(TensorSpec::plain(
        format!("{prefix}.self_attn.b_proj.weight"),
        DType::Bf16,
        &[geometry.kda_heads, h],
    )?);
    // The released file stores a 128-element F32 vector although only the first 96
    // (one per head) are consumed. The padded tail is part of the checkpoint ABI.
    specs.push(TensorSpec::plain(
        format!("{prefix}.self_attn.A_log"),
        DType::F32,
        &[geometry.kda_head_dim],
    )?);
    specs.push(TensorSpec::plain(
        format!("{prefix}.self_attn.dt_bias"),
        DType::F32,
        &[p],
    )?);
    specs.push(TensorSpec::plain(
        format!("{prefix}.self_attn.o_norm.weight"),
        DType::F32,
        &[geometry.kda_head_dim],
    )?);
    Ok(())
}

fn append_mla_specs(
    specs: &mut Vec<TensorSpec>,
    geometry: &Geometry,
    prefix: &str,
) -> Result<(), SchemaError> {
    let h = geometry.hidden;
    let q_head = checked_add_usize(geometry.qk_nope, geometry.qk_rope, "MLA Q head width")?;
    let q_width = checked_mul_usize(geometry.attention_heads, q_head, "MLA Q width")?;
    let kv_a_width = checked_add_usize(geometry.kv_lora, geometry.qk_rope, "MLA KV-A width")?;
    let kv_head = checked_add_usize(geometry.qk_nope, geometry.v_head, "MLA KV head width")?;
    let kv_b_width = checked_mul_usize(geometry.attention_heads, kv_head, "MLA KV-B width")?;
    let output_width = checked_mul_usize(
        geometry.attention_heads,
        geometry.v_head,
        "MLA output width",
    )?;
    for spec in [
        TensorSpec::plain(
            format!("{prefix}.self_attn.q_a_proj.weight"),
            DType::Bf16,
            &[geometry.q_lora, h],
        )?,
        TensorSpec::plain(
            format!("{prefix}.self_attn.q_a_layernorm.weight"),
            DType::Bf16,
            &[geometry.q_lora],
        )?,
        TensorSpec::plain(
            format!("{prefix}.self_attn.q_b_proj.weight"),
            DType::Bf16,
            &[q_width, geometry.q_lora],
        )?,
        TensorSpec::plain(
            format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
            DType::Bf16,
            &[kv_a_width, h],
        )?,
        TensorSpec::plain(
            format!("{prefix}.self_attn.kv_a_layernorm.weight"),
            DType::Bf16,
            &[geometry.kv_lora],
        )?,
        TensorSpec::plain(
            format!("{prefix}.self_attn.kv_b_proj.weight"),
            DType::Bf16,
            &[kv_b_width, geometry.kv_lora],
        )?,
        TensorSpec::plain(
            format!("{prefix}.self_attn.o_proj.weight"),
            DType::Bf16,
            &[h, output_width],
        )?,
        TensorSpec::plain(
            format!("{prefix}.self_attn.g_proj.weight"),
            DType::Bf16,
            &[output_width, h],
        )?,
    ] {
        specs.push(spec);
    }
    Ok(())
}

fn append_dense_specs(
    specs: &mut Vec<TensorSpec>,
    geometry: &Geometry,
    prefix: &str,
) -> Result<(), SchemaError> {
    for projection in ["gate", "up"] {
        specs.push(TensorSpec::plain(
            format!("{prefix}.mlp.{projection}_proj.weight"),
            DType::Bf16,
            &[geometry.dense_intermediate, geometry.hidden],
        )?);
    }
    specs.push(TensorSpec::plain(
        format!("{prefix}.mlp.down_proj.weight"),
        DType::Bf16,
        &[geometry.hidden, geometry.dense_intermediate],
    )?);
    Ok(())
}

fn append_moe_specs(
    specs: &mut Vec<TensorSpec>,
    geometry: &Geometry,
    prefix: &str,
) -> Result<(), SchemaError> {
    let moe = format!("{prefix}.block_sparse_moe");
    let h = geometry.hidden;
    let latent = geometry.routed_hidden;
    let shared = checked_mul_usize(
        geometry.moe_intermediate,
        geometry.shared_experts,
        "shared expert intermediate width",
    )?;
    for spec in [
        TensorSpec::plain(
            format!("{moe}.gate.weight"),
            DType::Bf16,
            &[geometry.experts, h],
        )?,
        TensorSpec::plain(
            format!("{moe}.gate.e_score_correction_bias"),
            DType::F32,
            &[geometry.experts],
        )?,
        TensorSpec::plain(
            format!("{moe}.routed_expert_down_proj.weight"),
            DType::Bf16,
            &[latent, h],
        )?,
        TensorSpec::plain(
            format!("{moe}.routed_expert_up_proj.weight"),
            DType::Bf16,
            &[h, latent],
        )?,
        TensorSpec::plain(
            format!("{moe}.routed_expert_norm.weight"),
            DType::Bf16,
            &[latent],
        )?,
        TensorSpec::plain(
            format!("{moe}.shared_experts.gate_proj.weight"),
            DType::Bf16,
            &[shared, h],
        )?,
        TensorSpec::plain(
            format!("{moe}.shared_experts.up_proj.weight"),
            DType::Bf16,
            &[shared, h],
        )?,
        TensorSpec::plain(
            format!("{moe}.shared_experts.down_proj.weight"),
            DType::Bf16,
            &[h, shared],
        )?,
    ] {
        specs.push(spec);
    }

    for expert in 0..geometry.experts {
        let expert_prefix = format!("{moe}.experts.{expert}");
        append_mxfp4_matrix(
            specs,
            format!("{expert_prefix}.w1"),
            geometry.moe_intermediate,
            latent,
        )?;
        append_mxfp4_matrix(
            specs,
            format!("{expert_prefix}.w2"),
            latent,
            geometry.moe_intermediate,
        )?;
        append_mxfp4_matrix(
            specs,
            format!("{expert_prefix}.w3"),
            geometry.moe_intermediate,
            latent,
        )?;
    }
    Ok(())
}

fn append_mxfp4_matrix(
    specs: &mut Vec<TensorSpec>,
    prefix: String,
    rows: usize,
    logical_cols: usize,
) -> Result<(), SchemaError> {
    if logical_cols % MXFP4_GROUP_SIZE != 0 || logical_cols % 2 != 0 {
        return Err(SchemaError::Invalid(format!(
            "MXFP4 matrix {prefix:?} has {logical_cols} columns; expected divisibility by {MXFP4_GROUP_SIZE}"
        )));
    }
    let logical_parameters = checked_product_usize(
        &[rows, logical_cols],
        &format!("{prefix} logical parameters"),
    )?;
    specs.push(TensorSpec::storage(
        format!("{prefix}.weight_packed"),
        DType::U8,
        &[rows, logical_cols / 2],
        logical_parameters,
    )?);
    specs.push(TensorSpec::storage(
        format!("{prefix}.weight_scale"),
        DType::U8,
        &[rows, logical_cols / MXFP4_GROUP_SIZE],
        0,
    )?);
    Ok(())
}

fn projector_specs(geometry: &Geometry) -> Result<Vec<TensorSpec>, SchemaError> {
    let kernel_area = checked_mul_usize(
        geometry.merge_height,
        geometry.merge_width,
        "projector merge area",
    )?;
    let merged_hidden = checked_mul_usize(
        geometry.vision_hidden,
        kernel_area,
        "projector merged hidden size",
    )?;
    Ok(vec![
        TensorSpec::plain(
            "mm_projector.proj.0.weight",
            DType::Bf16,
            &[merged_hidden, merged_hidden],
        )?,
        TensorSpec::plain(
            "mm_projector.proj.2.weight",
            DType::Bf16,
            &[geometry.hidden, merged_hidden],
        )?,
        TensorSpec::plain(
            "mm_projector.post_norm.weight",
            DType::Bf16,
            &[geometry.hidden],
        )?,
    ])
}

fn vision_specs(geometry: &Geometry) -> Result<Vec<TensorSpec>, SchemaError> {
    let mut specs = Vec::with_capacity(
        geometry
            .vision_layers
            .checked_mul(6)
            .and_then(|value| value.checked_add(3))
            .ok_or_else(|| {
                SchemaError::Invalid("vision tensor count overflows usize".to_owned())
            })?,
    );
    let qkv_width = checked_mul_usize(3, geometry.vision_qkv_hidden, "vision QKV width")?;
    for layer in 0..geometry.vision_layers {
        let prefix = format!("vision_tower.encoder.blocks.{layer}");
        for spec in [
            TensorSpec::plain(
                format!("{prefix}.mlp.fc0.weight"),
                DType::Bf16,
                &[geometry.vision_intermediate, geometry.vision_hidden],
            )?,
            TensorSpec::plain(
                format!("{prefix}.mlp.fc1.weight"),
                DType::Bf16,
                &[geometry.vision_hidden, geometry.vision_intermediate],
            )?,
            TensorSpec::plain(
                format!("{prefix}.norm0.weight"),
                DType::Bf16,
                &[geometry.vision_hidden],
            )?,
            TensorSpec::plain(
                format!("{prefix}.norm1.weight"),
                DType::Bf16,
                &[geometry.vision_hidden],
            )?,
            TensorSpec::plain(
                format!("{prefix}.wo.weight"),
                DType::Bf16,
                &[geometry.vision_hidden, geometry.vision_qkv_hidden],
            )?,
            TensorSpec::plain(
                format!("{prefix}.wqkv.weight"),
                DType::Bf16,
                &[qkv_width, geometry.vision_hidden],
            )?,
        ] {
            specs.push(spec);
        }
    }
    specs.push(TensorSpec::plain(
        "vision_tower.encoder.final_layernorm.weight",
        DType::Bf16,
        &[geometry.vision_hidden],
    )?);
    specs.push(TensorSpec::plain(
        "vision_tower.patch_embed.pos_emb.weight",
        DType::Bf16,
        &[
            geometry.vision_pos_height,
            geometry.vision_pos_width,
            geometry.vision_hidden,
        ],
    )?);
    specs.push(TensorSpec::plain(
        "vision_tower.patch_embed.proj.weight",
        DType::Bf16,
        &[
            geometry.vision_hidden,
            3,
            geometry.vision_patch,
            geometry.vision_patch,
        ],
    )?);
    Ok(specs)
}

fn layer_tensor_count(geometry: &Geometry, layer: usize) -> Result<usize, SchemaError> {
    if layer >= geometry.layers {
        return Err(SchemaError::Invalid(format!(
            "layer {layer} is outside 0..{}",
            geometry.layers
        )));
    }
    let attention = if geometry.is_kda[layer] { 14 } else { 8 };
    let mlp = if layer < geometry.first_dense_layers {
        3
    } else {
        checked_add_usize(
            8,
            checked_mul_usize(geometry.experts, 6, "expert tensor count")?,
            "MoE tensor count",
        )?
    };
    checked_add_usize(
        checked_add_usize(6, attention, "layer attention tensor count")?,
        mlp,
        "layer tensor count",
    )
}

fn parse_layer_id(name: &str) -> Result<Option<usize>, SchemaError> {
    let Some(rest) = name.strip_prefix(LAYER_PREFIX) else {
        return Ok(None);
    };
    let (raw_layer, suffix) = rest.split_once('.').ok_or_else(|| {
        SchemaError::Invalid(format!(
            "tensor {name:?} has a malformed decoder-layer namespace"
        ))
    })?;
    if raw_layer.is_empty()
        || suffix.is_empty()
        || !raw_layer.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(SchemaError::Invalid(format!(
            "tensor {name:?} has a malformed decoder layer ID"
        )));
    }
    let layer = raw_layer.parse::<usize>().map_err(|_| {
        SchemaError::Invalid(format!(
            "tensor {name:?} has an overflowing decoder layer ID"
        ))
    })?;
    Ok(Some(layer))
}

fn checked_shape(shape: &[usize]) -> Result<Vec<u64>, SchemaError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(SchemaError::Invalid(format!(
            "schema generated an empty or zero dimension: {shape:?}"
        )));
    }
    shape
        .iter()
        .map(|&dimension| {
            u64::try_from(dimension).map_err(|_| {
                SchemaError::Invalid(format!("dimension {dimension} does not fit u64"))
            })
        })
        .collect()
}

fn checked_product_u64(values: &[u64], label: &str) -> Result<u64, SchemaError> {
    values.iter().try_fold(1u64, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| SchemaError::Invalid(format!("{label} overflows u64")))
    })
}

fn checked_product_usize(values: &[usize], label: &str) -> Result<u64, SchemaError> {
    let values = checked_shape(values)?;
    checked_product_u64(&values, label)
}

fn checked_add_usize(left: usize, right: usize, label: &str) -> Result<usize, SchemaError> {
    left.checked_add(right)
        .ok_or_else(|| SchemaError::Invalid(format!("{label} overflows usize")))
}

fn checked_mul_usize(left: usize, right: usize, label: &str) -> Result<usize, SchemaError> {
    left.checked_mul(right)
        .ok_or_else(|| SchemaError::Invalid(format!("{label} overflows usize")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn release_geometry() -> Geometry {
        let layers = 93;
        let full_attention: BTreeSet<usize> =
            (4..=92).step_by(4).chain(std::iter::once(93)).collect();
        Geometry {
            hidden: 7_168,
            layers,
            vocab: 163_840,
            dense_intermediate: 33_792,
            moe_intermediate: 3_072,
            routed_hidden: 3_584,
            experts: 896,
            shared_experts: 2,
            attention_heads: 96,
            kda_heads: 96,
            kda_head_dim: 128,
            conv_kernel: 4,
            q_lora: 1_536,
            kv_lora: 512,
            qk_nope: 128,
            qk_rope: 64,
            v_head: 128,
            first_dense_layers: 1,
            is_kda: (1..=layers)
                .map(|one_based| !full_attention.contains(&one_based))
                .collect(),
            vision_layers: 27,
            vision_hidden: 1_024,
            vision_intermediate: 4_096,
            vision_qkv_hidden: 1_536,
            vision_pos_height: 64,
            vision_pos_width: 64,
            vision_patch: 14,
            merge_height: 2,
            merge_width: 2,
        }
    }

    fn find<'a>(specs: &'a [TensorSpec], name: &str) -> &'a TensorSpec {
        specs
            .iter()
            .find(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("missing generated tensor spec {name}"))
    }

    #[test]
    fn release_tensor_count_is_exact() {
        let geometry = release_geometry();
        assert_eq!(geometry.kda_layers(), 69);
        assert_eq!(geometry.layers - geometry.kda_layers(), 24);
        assert_eq!(
            layer_tensor_count(&geometry, 0).unwrap(),
            RELEASE_KDA_DENSE_LAYER_TENSOR_COUNT
        );
        assert_eq!(
            layer_tensor_count(&geometry, 1).unwrap(),
            RELEASE_KDA_MOE_LAYER_TENSOR_COUNT
        );
        assert_eq!(
            layer_tensor_count(&geometry, 3).unwrap(),
            RELEASE_MLA_MOE_LAYER_TENSOR_COUNT
        );
        let decoder = RELEASE_KDA_DENSE_LAYER_TENSOR_COUNT
            + 68 * RELEASE_KDA_MOE_LAYER_TENSOR_COUNT
            + 24 * RELEASE_MLA_MOE_LAYER_TENSOR_COUNT;
        assert_eq!(
            TEXT_ROOT_TENSOR_COUNT + decoder + PROJECTOR_TENSOR_COUNT + RELEASE_VISION_TENSOR_COUNT,
            RELEASE_TENSOR_COUNT
        );
    }

    #[test]
    fn release_representative_names_shapes_and_dtypes_match_abi() {
        let geometry = release_geometry();
        let dense_kda = layer_specs(&geometry, 0).unwrap();
        let conv = find(
            &dense_kda,
            "language_model.model.layers.0.self_attn.q_conv1d.weight",
        );
        assert_eq!(conv.dtype, DType::F32);
        assert_eq!(conv.shape, [12_288, 1, 4]);
        let a_log = find(&dense_kda, "language_model.model.layers.0.self_attn.A_log");
        assert_eq!(a_log.dtype, DType::F32);
        assert_eq!(a_log.shape, [128]);
        assert_eq!(
            find(
                &dense_kda,
                "language_model.model.layers.0.mlp.down_proj.weight"
            )
            .shape,
            [7_168, 33_792]
        );

        let kda_moe = layer_specs(&geometry, 1).unwrap();
        let packed = find(
            &kda_moe,
            "language_model.model.layers.1.block_sparse_moe.experts.895.w1.weight_packed",
        );
        assert_eq!(packed.dtype, DType::U8);
        assert_eq!(packed.shape, [3_072, 1_792]);
        assert_eq!(packed.logical_parameters, 3_072 * 3_584);
        let scale = find(
            &kda_moe,
            "language_model.model.layers.1.block_sparse_moe.experts.895.w2.weight_scale",
        );
        assert_eq!(scale.dtype, DType::U8);
        assert_eq!(scale.shape, [3_584, 96]);
        assert_eq!(scale.logical_parameters, 0);

        let mla = layer_specs(&geometry, 3).unwrap();
        assert_eq!(
            find(
                &mla,
                "language_model.model.layers.3.self_attn.q_b_proj.weight"
            )
            .shape,
            [18_432, 1_536]
        );
        assert_eq!(
            find(
                &mla,
                "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight"
            )
            .shape,
            [576, 7_168]
        );
    }

    #[test]
    fn release_projector_and_vision_schema_are_exact() {
        let geometry = release_geometry();
        let projector = projector_specs(&geometry).unwrap();
        assert_eq!(projector.len(), PROJECTOR_TENSOR_COUNT);
        assert_eq!(
            find(&projector, "mm_projector.proj.0.weight").shape,
            [4_096, 4_096]
        );
        assert_eq!(
            find(&projector, "mm_projector.proj.2.weight").shape,
            [7_168, 4_096]
        );
        assert_eq!(
            find(&projector, "mm_projector.post_norm.weight").shape,
            [7_168]
        );

        let vision = vision_specs(&geometry).unwrap();
        assert_eq!(vision.len(), RELEASE_VISION_TENSOR_COUNT);
        assert_eq!(
            find(&vision, "vision_tower.encoder.blocks.26.wqkv.weight").shape,
            [4_608, 1_024]
        );
        assert_eq!(
            find(&vision, "vision_tower.encoder.blocks.26.wo.weight").shape,
            [1_024, 1_536]
        );
        assert_eq!(
            find(&vision, "vision_tower.patch_embed.pos_emb.weight").shape,
            [64, 64, 1_024]
        );
        assert_eq!(
            find(&vision, "vision_tower.patch_embed.proj.weight").shape,
            [1_024, 3, 14, 14]
        );
    }

    #[test]
    fn layer_namespace_parser_is_strict() {
        assert_eq!(
            parse_layer_id("language_model.model.layers.92.input_layernorm.weight").unwrap(),
            Some(92)
        );
        assert_eq!(
            parse_layer_id("language_model.model.norm.weight").unwrap(),
            None
        );
        assert!(parse_layer_id("language_model.model.layers.x.norm.weight").is_err());
        assert!(parse_layer_id("language_model.model.layers.1").is_err());
    }

    #[test]
    #[ignore = "requires KIMI_K3_MODEL_DIR pointing at a local complete or partial official checkpoint"]
    fn local_checkpoint_headers_match_schema() {
        let directory = std::env::var("KIMI_K3_MODEL_DIR")
            .expect("set KIMI_K3_MODEL_DIR for the ignored real-checkpoint test");
        let directory = Path::new(&directory);
        let config = KimiK3Config::load(directory).unwrap();
        let index = TensorIndex::open(directory).unwrap();
        let partial = inspect_available_layers(&config, &index).unwrap();
        assert!(!partial.validated_layers.is_empty());
        if index.names().count() == RELEASE_TENSOR_COUNT {
            let full = inspect_requirements(&config, &index).unwrap();
            assert_eq!(full.required_tensor_count, RELEASE_TENSOR_COUNT);
        }
    }
}
