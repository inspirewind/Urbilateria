//! Strict, layer-streamed loading of Kimi-K3 decoder trunk weights.
//!
//! Routed expert payloads are intentionally excluded: [`KimiK3LayerWeights`] contains only the
//! resident norms, AttnRes vectors, attention projections, and the dense or MoE trunk for one
//! decoder layer. Every tensor is validated from safetensors metadata and the aggregate resident
//! allocation is authorized before any payload is read. Large BF16 matrices stay in compact
//! two-byte storage through the shared batch loader; small BF16/F32 tensors are expanded to F32.

use super::KimiK3Config;
use crate::config::ConfigError;
use crate::model::WeightMatrix;
use crate::storage::{
    inspect_compact_bf16_matrix, load_compact_bf16_matrices, load_reference_values, DType,
    SafetensorError, TensorIndex, TensorLoadError, WeightLoadError,
};
use std::collections::BTreeMap;
use std::fmt;

const LAYER_PREFIX: &str = "language_model.model.layers.";

#[derive(Debug)]
pub enum KimiK3LayerWeightError {
    Config(ConfigError),
    InvalidGeometry(String),
    InvalidLayer {
        layer: usize,
        layers: usize,
    },
    Checkpoint {
        name: String,
        source: SafetensorError,
    },
    MatrixMetadata {
        name: String,
        source: WeightLoadError,
    },
    MatrixBatch(WeightLoadError),
    VectorPayload {
        name: String,
        source: TensorLoadError,
    },
    Budget {
        layer: usize,
        required: u64,
        maximum: u64,
    },
    Accounting(String),
}

impl fmt::Display for KimiK3LayerWeightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::InvalidGeometry(reason) => {
                write!(formatter, "invalid Kimi-K3 layer geometry: {reason}")
            }
            Self::InvalidLayer { layer, layers } => {
                write!(formatter, "Kimi-K3 layer {layer} is outside 0..{layers}")
            }
            Self::Checkpoint { name, source } => {
                write!(formatter, "Kimi-K3 layer tensor {name:?}: {source}")
            }
            Self::MatrixMetadata { name, source } => {
                write!(formatter, "Kimi-K3 compact matrix {name:?}: {source}")
            }
            Self::MatrixBatch(source) => {
                write!(
                    formatter,
                    "cannot batch-load Kimi-K3 layer matrices: {source}"
                )
            }
            Self::VectorPayload { name, source } => {
                write!(
                    formatter,
                    "cannot load Kimi-K3 layer vector {name:?}: {source}"
                )
            }
            Self::Budget {
                layer,
                required,
                maximum,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} needs {required} resident bytes, layer budget is {maximum}"
            ),
            Self::Accounting(reason) => {
                write!(
                    formatter,
                    "invalid Kimi-K3 layer weight accounting: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for KimiK3LayerWeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Checkpoint { source, .. } => Some(source),
            Self::MatrixMetadata { source, .. } | Self::MatrixBatch(source) => Some(source),
            Self::VectorPayload { source, .. } => Some(source),
            Self::InvalidGeometry(_)
            | Self::InvalidLayer { .. }
            | Self::Budget { .. }
            | Self::Accounting(_) => None,
        }
    }
}

/// Stable identifiers for every resident matrix projection in one Kimi-K3 layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KimiK3WeightProjection {
    KdaQuery,
    KdaKey,
    KdaValue,
    KdaOutputGate,
    KdaOutput,
    KdaDecayA,
    KdaDecayB,
    KdaBeta,
    MlaQueryA,
    MlaQueryB,
    MlaKvA,
    MlaKvB,
    MlaOutputGate,
    MlaOutput,
    DenseGate,
    DenseUp,
    DenseDown,
    MoeRouter,
    MoeRoutedDown,
    MoeRoutedUp,
    MoeSharedGate,
    MoeSharedUp,
    MoeSharedDown,
}

/// Six layer-level vectors shared by both attention families and feed-forward variants.
#[derive(Debug)]
pub struct KimiK3CommonLayerWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub self_attention_res_norm: Vec<f32>,
    /// Logical `[1, hidden_size]`, stored as a flat vector for elementwise AttnRes folding.
    pub self_attention_res_projection: Vec<f32>,
    pub mlp_res_norm: Vec<f32>,
    /// Logical `[1, hidden_size]`, stored as a flat vector for elementwise AttnRes folding.
    pub mlp_res_projection: Vec<f32>,
}

#[derive(Debug)]
pub struct KimiK3KdaWeights {
    pub query: WeightMatrix,
    pub key: WeightMatrix,
    pub value: WeightMatrix,
    pub output_gate: WeightMatrix,
    pub output: WeightMatrix,
    pub decay_a: WeightMatrix,
    pub decay_b: WeightMatrix,
    pub beta: WeightMatrix,
    /// Flattened `[heads * head_dim, 1, conv_kernel_size]`, oldest tap first.
    pub query_conv: Vec<f32>,
    pub key_conv: Vec<f32>,
    pub value_conv: Vec<f32>,
    /// The released ABI stores `head_dim` values; only the first `num_heads` are consumed.
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub output_norm: Vec<f32>,
}

#[derive(Debug)]
pub struct KimiK3MlaWeights {
    pub query_a: WeightMatrix,
    pub query_a_norm: Vec<f32>,
    pub query_b: WeightMatrix,
    pub kv_a: WeightMatrix,
    pub kv_a_norm: Vec<f32>,
    pub kv_b: WeightMatrix,
    pub output: WeightMatrix,
    pub output_gate: WeightMatrix,
}

#[derive(Debug)]
pub enum KimiK3AttentionWeights {
    Kda(Box<KimiK3KdaWeights>),
    Mla(Box<KimiK3MlaWeights>),
}

#[derive(Debug)]
pub struct KimiK3DenseWeights {
    pub gate: WeightMatrix,
    pub up: WeightMatrix,
    pub down: WeightMatrix,
}

/// Resident MoE projections shared across tokens; routed MXFP4 experts are loaded separately.
#[derive(Debug)]
pub struct KimiK3MoeTrunkWeights {
    pub router: WeightMatrix,
    pub correction_bias: Vec<f32>,
    pub routed_down: WeightMatrix,
    pub routed_up: WeightMatrix,
    pub routed_norm: Vec<f32>,
    pub shared_gate: WeightMatrix,
    pub shared_up: WeightMatrix,
    pub shared_down: WeightMatrix,
}

#[derive(Debug)]
pub enum KimiK3FeedForwardWeights {
    Dense(Box<KimiK3DenseWeights>),
    Moe(Box<KimiK3MoeTrunkWeights>),
}

/// Complete non-expert resident weights for one decoder layer.
#[derive(Debug)]
pub struct KimiK3LayerWeights {
    pub layer: usize,
    pub common: KimiK3CommonLayerWeights,
    pub attention: KimiK3AttentionWeights,
    pub feed_forward: KimiK3FeedForwardWeights,
    resident_bytes: u64,
}

impl KimiK3LayerWeights {
    /// Loads one official layer after validating the complete config and authorizing all resident
    /// matrices and expanded vectors as one allocation boundary.
    pub fn load(
        config: &KimiK3Config,
        index: &TensorIndex,
        layer: usize,
        maximum_resident_bytes: u64,
    ) -> Result<Self, KimiK3LayerWeightError> {
        let geometry = checked_geometry(config, layer)?;
        Self::load_with_geometry(index, geometry, maximum_resident_bytes)
    }

    /// Performs the exact layer metadata preflight used by [`Self::load`] and returns the peak
    /// non-expert resident allocation. This reads retained safetensors headers only; it is safe to
    /// call while planning a layer-streamed runtime before any payload is authorized.
    pub fn inspect_resident_bytes(
        config: &KimiK3Config,
        index: &TensorIndex,
        layer: usize,
    ) -> Result<u64, KimiK3LayerWeightError> {
        let geometry = checked_geometry(config, layer)?;
        Self::inspect_resident_bytes_with_geometry(index, &geometry)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Returns a compact matrix by its semantic projection, or `None` when that projection does
    /// not belong to this layer's attention/feed-forward variants.
    pub fn matrix(&self, projection: KimiK3WeightProjection) -> Option<&WeightMatrix> {
        self.attention
            .matrix(projection)
            .or_else(|| self.feed_forward.matrix(projection))
    }

    fn load_with_geometry(
        index: &TensorIndex,
        geometry: LayerGeometry,
        maximum_resident_bytes: u64,
    ) -> Result<Self, KimiK3LayerWeightError> {
        geometry.validate()?;
        let manifest = LayerManifest::inspect(index, &geometry)?;
        if manifest.resident_bytes > maximum_resident_bytes {
            return Err(KimiK3LayerWeightError::Budget {
                layer: geometry.layer,
                required: manifest.resident_bytes,
                maximum: maximum_resident_bytes,
            });
        }

        // No payload operation occurs before the aggregate layer authorization above.
        let matrix_requests = manifest
            .matrices
            .iter()
            .map(|spec| (spec.name.as_str(), spec.rows, spec.cols))
            .collect::<Vec<_>>();
        let matrices =
            load_compact_bf16_matrices(index, &matrix_requests, manifest.matrix_resident_bytes)
                .map_err(KimiK3LayerWeightError::MatrixBatch)?;
        let mut loaded_matrices = BTreeMap::new();
        for (spec, matrix) in manifest.matrices.into_iter().zip(matrices) {
            if loaded_matrices.insert(spec.slot, matrix).is_some() {
                return Err(KimiK3LayerWeightError::Accounting(format!(
                    "duplicate matrix slot {:?}",
                    spec.slot
                )));
            }
        }

        let mut loaded_vectors = BTreeMap::new();
        for spec in manifest.vectors {
            let values = load_reference_values(index, &spec.name).map_err(|source| {
                KimiK3LayerWeightError::VectorPayload {
                    name: spec.name.clone(),
                    source,
                }
            })?;
            if values.len() != spec.elements {
                return Err(KimiK3LayerWeightError::Accounting(format!(
                    "vector {:?} decoded {} values, expected {}",
                    spec.name,
                    values.len(),
                    spec.elements
                )));
            }
            if loaded_vectors.insert(spec.slot, values).is_some() {
                return Err(KimiK3LayerWeightError::Accounting(format!(
                    "duplicate vector slot {:?}",
                    spec.slot
                )));
            }
        }

        let loaded_resident_bytes = loaded_matrices.values().try_fold(0u64, |total, matrix| {
            checked_add_u64(
                total,
                u64::try_from(matrix.resident_bytes()).map_err(|_| {
                    KimiK3LayerWeightError::Accounting(
                        "matrix resident bytes do not fit u64".to_owned(),
                    )
                })?,
                "loaded matrix resident bytes",
            )
        })?;
        let loaded_resident_bytes =
            loaded_vectors
                .values()
                .try_fold(loaded_resident_bytes, |total, vector| {
                    let bytes = vector
                        .len()
                        .checked_mul(std::mem::size_of::<f32>())
                        .and_then(|bytes| u64::try_from(bytes).ok())
                        .ok_or_else(|| {
                            KimiK3LayerWeightError::Accounting(
                                "loaded vector resident bytes overflow".to_owned(),
                            )
                        })?;
                    checked_add_u64(total, bytes, "loaded vector resident bytes")
                })?;
        if loaded_resident_bytes != manifest.resident_bytes {
            return Err(KimiK3LayerWeightError::Accounting(format!(
                "metadata authorized {} resident bytes but loaded values use {loaded_resident_bytes}",
                manifest.resident_bytes
            )));
        }

        let common = KimiK3CommonLayerWeights {
            input_layernorm: take_vector(&mut loaded_vectors, VectorSlot::InputLayerNorm)?,
            post_attention_layernorm: take_vector(
                &mut loaded_vectors,
                VectorSlot::PostAttentionLayerNorm,
            )?,
            self_attention_res_norm: take_vector(
                &mut loaded_vectors,
                VectorSlot::SelfAttentionResNorm,
            )?,
            self_attention_res_projection: take_vector(
                &mut loaded_vectors,
                VectorSlot::SelfAttentionResProjection,
            )?,
            mlp_res_norm: take_vector(&mut loaded_vectors, VectorSlot::MlpResNorm)?,
            mlp_res_projection: take_vector(&mut loaded_vectors, VectorSlot::MlpResProjection)?,
        };
        let attention = match geometry.attention {
            AttentionKind::Kda => KimiK3AttentionWeights::Kda(Box::new(KimiK3KdaWeights {
                query: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::KdaQuery)?,
                key: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::KdaKey)?,
                value: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::KdaValue)?,
                output_gate: take_matrix(
                    &mut loaded_matrices,
                    KimiK3WeightProjection::KdaOutputGate,
                )?,
                output: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::KdaOutput)?,
                decay_a: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::KdaDecayA)?,
                decay_b: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::KdaDecayB)?,
                beta: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::KdaBeta)?,
                query_conv: take_vector(&mut loaded_vectors, VectorSlot::KdaQueryConv)?,
                key_conv: take_vector(&mut loaded_vectors, VectorSlot::KdaKeyConv)?,
                value_conv: take_vector(&mut loaded_vectors, VectorSlot::KdaValueConv)?,
                a_log: take_vector(&mut loaded_vectors, VectorSlot::KdaALog)?,
                dt_bias: take_vector(&mut loaded_vectors, VectorSlot::KdaDtBias)?,
                output_norm: take_vector(&mut loaded_vectors, VectorSlot::KdaOutputNorm)?,
            })),
            AttentionKind::Mla => KimiK3AttentionWeights::Mla(Box::new(KimiK3MlaWeights {
                query_a: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::MlaQueryA)?,
                query_a_norm: take_vector(&mut loaded_vectors, VectorSlot::MlaQueryANorm)?,
                query_b: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::MlaQueryB)?,
                kv_a: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::MlaKvA)?,
                kv_a_norm: take_vector(&mut loaded_vectors, VectorSlot::MlaKvANorm)?,
                kv_b: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::MlaKvB)?,
                output: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::MlaOutput)?,
                output_gate: take_matrix(
                    &mut loaded_matrices,
                    KimiK3WeightProjection::MlaOutputGate,
                )?,
            })),
        };
        let feed_forward = match geometry.feed_forward {
            FeedForwardKind::Dense => {
                KimiK3FeedForwardWeights::Dense(Box::new(KimiK3DenseWeights {
                    gate: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::DenseGate)?,
                    up: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::DenseUp)?,
                    down: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::DenseDown)?,
                }))
            }
            FeedForwardKind::Moe => {
                KimiK3FeedForwardWeights::Moe(Box::new(KimiK3MoeTrunkWeights {
                    router: take_matrix(&mut loaded_matrices, KimiK3WeightProjection::MoeRouter)?,
                    correction_bias: take_vector(
                        &mut loaded_vectors,
                        VectorSlot::MoeCorrectionBias,
                    )?,
                    routed_down: take_matrix(
                        &mut loaded_matrices,
                        KimiK3WeightProjection::MoeRoutedDown,
                    )?,
                    routed_up: take_matrix(
                        &mut loaded_matrices,
                        KimiK3WeightProjection::MoeRoutedUp,
                    )?,
                    routed_norm: take_vector(&mut loaded_vectors, VectorSlot::MoeRoutedNorm)?,
                    shared_gate: take_matrix(
                        &mut loaded_matrices,
                        KimiK3WeightProjection::MoeSharedGate,
                    )?,
                    shared_up: take_matrix(
                        &mut loaded_matrices,
                        KimiK3WeightProjection::MoeSharedUp,
                    )?,
                    shared_down: take_matrix(
                        &mut loaded_matrices,
                        KimiK3WeightProjection::MoeSharedDown,
                    )?,
                }))
            }
        };
        if !loaded_matrices.is_empty() || !loaded_vectors.is_empty() {
            return Err(KimiK3LayerWeightError::Accounting(format!(
                "{} matrix and {} vector slot(s) were not assembled",
                loaded_matrices.len(),
                loaded_vectors.len()
            )));
        }

        Ok(Self {
            layer: geometry.layer,
            common,
            attention,
            feed_forward,
            resident_bytes: loaded_resident_bytes,
        })
    }

    fn inspect_resident_bytes_with_geometry(
        index: &TensorIndex,
        geometry: &LayerGeometry,
    ) -> Result<u64, KimiK3LayerWeightError> {
        geometry.validate()?;
        Ok(LayerManifest::inspect(index, geometry)?.resident_bytes)
    }
}

impl KimiK3AttentionWeights {
    fn matrix(&self, projection: KimiK3WeightProjection) -> Option<&WeightMatrix> {
        match (self, projection) {
            (Self::Kda(weights), KimiK3WeightProjection::KdaQuery) => Some(&weights.query),
            (Self::Kda(weights), KimiK3WeightProjection::KdaKey) => Some(&weights.key),
            (Self::Kda(weights), KimiK3WeightProjection::KdaValue) => Some(&weights.value),
            (Self::Kda(weights), KimiK3WeightProjection::KdaOutputGate) => {
                Some(&weights.output_gate)
            }
            (Self::Kda(weights), KimiK3WeightProjection::KdaOutput) => Some(&weights.output),
            (Self::Kda(weights), KimiK3WeightProjection::KdaDecayA) => Some(&weights.decay_a),
            (Self::Kda(weights), KimiK3WeightProjection::KdaDecayB) => Some(&weights.decay_b),
            (Self::Kda(weights), KimiK3WeightProjection::KdaBeta) => Some(&weights.beta),
            (Self::Mla(weights), KimiK3WeightProjection::MlaQueryA) => Some(&weights.query_a),
            (Self::Mla(weights), KimiK3WeightProjection::MlaQueryB) => Some(&weights.query_b),
            (Self::Mla(weights), KimiK3WeightProjection::MlaKvA) => Some(&weights.kv_a),
            (Self::Mla(weights), KimiK3WeightProjection::MlaKvB) => Some(&weights.kv_b),
            (Self::Mla(weights), KimiK3WeightProjection::MlaOutputGate) => {
                Some(&weights.output_gate)
            }
            (Self::Mla(weights), KimiK3WeightProjection::MlaOutput) => Some(&weights.output),
            (Self::Kda(_), _) | (Self::Mla(_), _) => None,
        }
    }
}

impl KimiK3FeedForwardWeights {
    fn matrix(&self, projection: KimiK3WeightProjection) -> Option<&WeightMatrix> {
        match (self, projection) {
            (Self::Dense(weights), KimiK3WeightProjection::DenseGate) => Some(&weights.gate),
            (Self::Dense(weights), KimiK3WeightProjection::DenseUp) => Some(&weights.up),
            (Self::Dense(weights), KimiK3WeightProjection::DenseDown) => Some(&weights.down),
            (Self::Moe(weights), KimiK3WeightProjection::MoeRouter) => Some(&weights.router),
            (Self::Moe(weights), KimiK3WeightProjection::MoeRoutedDown) => {
                Some(&weights.routed_down)
            }
            (Self::Moe(weights), KimiK3WeightProjection::MoeRoutedUp) => Some(&weights.routed_up),
            (Self::Moe(weights), KimiK3WeightProjection::MoeSharedGate) => {
                Some(&weights.shared_gate)
            }
            (Self::Moe(weights), KimiK3WeightProjection::MoeSharedUp) => Some(&weights.shared_up),
            (Self::Moe(weights), KimiK3WeightProjection::MoeSharedDown) => {
                Some(&weights.shared_down)
            }
            (Self::Dense(_), _) | (Self::Moe(_), _) => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AttentionKind {
    Kda,
    Mla,
}

#[derive(Debug, Clone, Copy)]
enum FeedForwardKind {
    Dense,
    Moe,
}

#[derive(Debug, Clone, Copy)]
struct LayerGeometry {
    layer: usize,
    hidden: usize,
    attention: AttentionKind,
    feed_forward: FeedForwardKind,
    attention_heads: usize,
    kda_heads: usize,
    kda_head_dim: usize,
    conv_kernel: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_nope: usize,
    qk_nope_slot: usize,
    value_head: usize,
    dense_intermediate: usize,
    moe_intermediate: usize,
    latent: usize,
    experts: usize,
    shared_experts: usize,
}

impl LayerGeometry {
    fn from_config(config: &KimiK3Config, layer: usize) -> Self {
        let text = &config.text_config;
        Self {
            layer,
            hidden: text.hidden_size,
            attention: if text.is_kda_layer(layer) {
                AttentionKind::Kda
            } else {
                AttentionKind::Mla
            },
            feed_forward: if layer < text.first_k_dense_replace {
                FeedForwardKind::Dense
            } else {
                FeedForwardKind::Moe
            },
            attention_heads: text.num_attention_heads,
            kda_heads: text.linear_attn_config.num_heads,
            kda_head_dim: text.linear_attn_config.head_dim,
            conv_kernel: text.linear_attn_config.short_conv_kernel_size,
            q_lora: text.q_lora_rank,
            kv_lora: text.kv_lora_rank,
            qk_nope: text.qk_nope_head_dim,
            qk_nope_slot: text.qk_rope_head_dim,
            value_head: text.v_head_dim,
            dense_intermediate: text.intermediate_size,
            moe_intermediate: text.moe_intermediate_size,
            latent: text.routed_expert_hidden_size,
            experts: text.num_experts,
            shared_experts: text.num_shared_experts,
        }
    }

    fn validate(self) -> Result<(), KimiK3LayerWeightError> {
        if self.hidden == 0 {
            return invalid_geometry("hidden_size must be non-zero");
        }
        match self.attention {
            AttentionKind::Kda => {
                for (name, value) in [
                    ("KDA num_heads", self.kda_heads),
                    ("KDA head_dim", self.kda_head_dim),
                    ("KDA convolution kernel", self.conv_kernel),
                ] {
                    if value == 0 {
                        return invalid_geometry(format!("{name} must be non-zero"));
                    }
                }
            }
            AttentionKind::Mla => {
                for (name, value) in [
                    ("MLA num_attention_heads", self.attention_heads),
                    ("MLA q_lora_rank", self.q_lora),
                    ("MLA kv_lora_rank", self.kv_lora),
                    ("MLA qk_nope_head_dim", self.qk_nope),
                    ("MLA qk_nope_slot_dim", self.qk_nope_slot),
                    ("MLA value_head_dim", self.value_head),
                ] {
                    if value == 0 {
                        return invalid_geometry(format!("{name} must be non-zero"));
                    }
                }
            }
        }
        match self.feed_forward {
            FeedForwardKind::Dense if self.dense_intermediate == 0 => {
                invalid_geometry("dense intermediate size must be non-zero")
            }
            FeedForwardKind::Moe => {
                for (name, value) in [
                    ("MoE intermediate size", self.moe_intermediate),
                    ("MoE latent size", self.latent),
                    ("MoE expert count", self.experts),
                    ("MoE shared expert count", self.shared_experts),
                ] {
                    if value == 0 {
                        return invalid_geometry(format!("{name} must be non-zero"));
                    }
                }
                checked_mul_usize(
                    self.moe_intermediate,
                    self.shared_experts,
                    "shared expert intermediate width",
                )?;
                Ok(())
            }
            FeedForwardKind::Dense => Ok(()),
        }
    }
}

fn checked_geometry(
    config: &KimiK3Config,
    layer: usize,
) -> Result<LayerGeometry, KimiK3LayerWeightError> {
    config.validate().map_err(KimiK3LayerWeightError::Config)?;
    if layer >= config.text_config.num_hidden_layers {
        return Err(KimiK3LayerWeightError::InvalidLayer {
            layer,
            layers: config.text_config.num_hidden_layers,
        });
    }
    let geometry = LayerGeometry::from_config(config, layer);
    geometry.validate()?;
    Ok(geometry)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum VectorSlot {
    InputLayerNorm,
    PostAttentionLayerNorm,
    SelfAttentionResNorm,
    SelfAttentionResProjection,
    MlpResNorm,
    MlpResProjection,
    KdaQueryConv,
    KdaKeyConv,
    KdaValueConv,
    KdaALog,
    KdaDtBias,
    KdaOutputNorm,
    MlaQueryANorm,
    MlaKvANorm,
    MoeCorrectionBias,
    MoeRoutedNorm,
}

#[derive(Debug)]
struct MatrixSpec {
    slot: KimiK3WeightProjection,
    name: String,
    rows: usize,
    cols: usize,
}

#[derive(Debug)]
struct VectorSpec {
    slot: VectorSlot,
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    elements: usize,
}

#[derive(Debug)]
struct LayerManifest {
    matrices: Vec<MatrixSpec>,
    vectors: Vec<VectorSpec>,
    matrix_resident_bytes: u64,
    resident_bytes: u64,
}

impl LayerManifest {
    fn inspect(
        index: &TensorIndex,
        geometry: &LayerGeometry,
    ) -> Result<Self, KimiK3LayerWeightError> {
        let (matrices, mut vectors) = layer_specs(geometry)?;
        let mut matrix_resident_bytes = 0u64;
        for spec in &matrices {
            let layout = inspect_compact_bf16_matrix(index, &spec.name, spec.rows, spec.cols)
                .map_err(|source| KimiK3LayerWeightError::MatrixMetadata {
                    name: spec.name.clone(),
                    source,
                })?;
            matrix_resident_bytes = checked_add_u64(
                matrix_resident_bytes,
                layout.resident_bytes,
                "matrix manifest resident bytes",
            )?;
        }

        let mut resident_bytes = matrix_resident_bytes;
        for spec in &mut vectors {
            spec.elements = inspect_vector(index, spec)?;
            let bytes = u64::try_from(spec.elements)
                .ok()
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| {
                    KimiK3LayerWeightError::Accounting(format!(
                        "expanded vector {:?} resident bytes overflow",
                        spec.name
                    ))
                })?;
            resident_bytes =
                checked_add_u64(resident_bytes, bytes, "layer manifest resident bytes")?;
        }
        Ok(Self {
            matrices,
            vectors,
            matrix_resident_bytes,
            resident_bytes,
        })
    }
}

fn layer_specs(
    geometry: &LayerGeometry,
) -> Result<(Vec<MatrixSpec>, Vec<VectorSpec>), KimiK3LayerWeightError> {
    let prefix = format!("{LAYER_PREFIX}{}", geometry.layer);
    let h = geometry.hidden;
    let mut matrices = Vec::new();
    let mut vectors = vec![
        vector_spec(
            VectorSlot::InputLayerNorm,
            format!("{prefix}.input_layernorm.weight"),
            DType::Bf16,
            &[h],
        ),
        vector_spec(
            VectorSlot::PostAttentionLayerNorm,
            format!("{prefix}.post_attention_layernorm.weight"),
            DType::Bf16,
            &[h],
        ),
        vector_spec(
            VectorSlot::SelfAttentionResNorm,
            format!("{prefix}.self_attention_res_norm.weight"),
            DType::Bf16,
            &[h],
        ),
        vector_spec(
            VectorSlot::MlpResNorm,
            format!("{prefix}.mlp_res_norm.weight"),
            DType::Bf16,
            &[h],
        ),
        vector_spec(
            VectorSlot::SelfAttentionResProjection,
            format!("{prefix}.self_attention_res_proj.weight"),
            DType::Bf16,
            &[1, h],
        ),
        vector_spec(
            VectorSlot::MlpResProjection,
            format!("{prefix}.mlp_res_proj.weight"),
            DType::Bf16,
            &[1, h],
        ),
    ];

    let attention = format!("{prefix}.self_attn");
    match geometry.attention {
        AttentionKind::Kda => {
            let width = checked_mul_usize(
                geometry.kda_heads,
                geometry.kda_head_dim,
                "KDA projection width",
            )?;
            for (slot, projection) in [
                (KimiK3WeightProjection::KdaQuery, "q"),
                (KimiK3WeightProjection::KdaKey, "k"),
                (KimiK3WeightProjection::KdaValue, "v"),
                (KimiK3WeightProjection::KdaOutputGate, "g"),
            ] {
                matrices.push(matrix_spec(
                    slot,
                    format!("{attention}.{projection}_proj.weight"),
                    width,
                    h,
                ));
            }
            matrices.push(matrix_spec(
                KimiK3WeightProjection::KdaOutput,
                format!("{attention}.o_proj.weight"),
                h,
                width,
            ));
            matrices.push(matrix_spec(
                KimiK3WeightProjection::KdaDecayA,
                format!("{attention}.f_a_proj.weight"),
                geometry.kda_head_dim,
                h,
            ));
            matrices.push(matrix_spec(
                KimiK3WeightProjection::KdaDecayB,
                format!("{attention}.f_b_proj.weight"),
                width,
                geometry.kda_head_dim,
            ));
            matrices.push(matrix_spec(
                KimiK3WeightProjection::KdaBeta,
                format!("{attention}.b_proj.weight"),
                geometry.kda_heads,
                h,
            ));
            for (slot, projection) in [
                (VectorSlot::KdaQueryConv, "q"),
                (VectorSlot::KdaKeyConv, "k"),
                (VectorSlot::KdaValueConv, "v"),
            ] {
                vectors.push(vector_spec(
                    slot,
                    format!("{attention}.{projection}_conv1d.weight"),
                    DType::F32,
                    &[width, 1, geometry.conv_kernel],
                ));
            }
            vectors.extend([
                vector_spec(
                    VectorSlot::KdaALog,
                    format!("{attention}.A_log"),
                    DType::F32,
                    &[geometry.kda_head_dim],
                ),
                vector_spec(
                    VectorSlot::KdaDtBias,
                    format!("{attention}.dt_bias"),
                    DType::F32,
                    &[width],
                ),
                vector_spec(
                    VectorSlot::KdaOutputNorm,
                    format!("{attention}.o_norm.weight"),
                    DType::F32,
                    &[geometry.kda_head_dim],
                ),
            ]);
        }
        AttentionKind::Mla => {
            let query_head = checked_add_usize(
                geometry.qk_nope,
                geometry.qk_nope_slot,
                "MLA query head width",
            )?;
            let query_width = checked_mul_usize(
                geometry.attention_heads,
                query_head,
                "MLA query projection width",
            )?;
            let kv_a_width =
                checked_add_usize(geometry.kv_lora, geometry.qk_nope_slot, "MLA KV-A width")?;
            let kv_head =
                checked_add_usize(geometry.qk_nope, geometry.value_head, "MLA KV head width")?;
            let kv_b_width =
                checked_mul_usize(geometry.attention_heads, kv_head, "MLA KV-B width")?;
            let output_width = checked_mul_usize(
                geometry.attention_heads,
                geometry.value_head,
                "MLA output width",
            )?;
            matrices.extend([
                matrix_spec(
                    KimiK3WeightProjection::MlaQueryA,
                    format!("{attention}.q_a_proj.weight"),
                    geometry.q_lora,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MlaQueryB,
                    format!("{attention}.q_b_proj.weight"),
                    query_width,
                    geometry.q_lora,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MlaKvA,
                    format!("{attention}.kv_a_proj_with_mqa.weight"),
                    kv_a_width,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MlaKvB,
                    format!("{attention}.kv_b_proj.weight"),
                    kv_b_width,
                    geometry.kv_lora,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MlaOutput,
                    format!("{attention}.o_proj.weight"),
                    h,
                    output_width,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MlaOutputGate,
                    format!("{attention}.g_proj.weight"),
                    output_width,
                    h,
                ),
            ]);
            vectors.extend([
                vector_spec(
                    VectorSlot::MlaQueryANorm,
                    format!("{attention}.q_a_layernorm.weight"),
                    DType::Bf16,
                    &[geometry.q_lora],
                ),
                vector_spec(
                    VectorSlot::MlaKvANorm,
                    format!("{attention}.kv_a_layernorm.weight"),
                    DType::Bf16,
                    &[geometry.kv_lora],
                ),
            ]);
        }
    }

    match geometry.feed_forward {
        FeedForwardKind::Dense => {
            let mlp = format!("{prefix}.mlp");
            matrices.extend([
                matrix_spec(
                    KimiK3WeightProjection::DenseGate,
                    format!("{mlp}.gate_proj.weight"),
                    geometry.dense_intermediate,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::DenseUp,
                    format!("{mlp}.up_proj.weight"),
                    geometry.dense_intermediate,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::DenseDown,
                    format!("{mlp}.down_proj.weight"),
                    h,
                    geometry.dense_intermediate,
                ),
            ]);
        }
        FeedForwardKind::Moe => {
            let moe = format!("{prefix}.block_sparse_moe");
            let shared = checked_mul_usize(
                geometry.moe_intermediate,
                geometry.shared_experts,
                "shared expert intermediate width",
            )?;
            matrices.extend([
                matrix_spec(
                    KimiK3WeightProjection::MoeRouter,
                    format!("{moe}.gate.weight"),
                    geometry.experts,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MoeRoutedDown,
                    format!("{moe}.routed_expert_down_proj.weight"),
                    geometry.latent,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MoeRoutedUp,
                    format!("{moe}.routed_expert_up_proj.weight"),
                    h,
                    geometry.latent,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MoeSharedGate,
                    format!("{moe}.shared_experts.gate_proj.weight"),
                    shared,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MoeSharedUp,
                    format!("{moe}.shared_experts.up_proj.weight"),
                    shared,
                    h,
                ),
                matrix_spec(
                    KimiK3WeightProjection::MoeSharedDown,
                    format!("{moe}.shared_experts.down_proj.weight"),
                    h,
                    shared,
                ),
            ]);
            vectors.extend([
                vector_spec(
                    VectorSlot::MoeCorrectionBias,
                    format!("{moe}.gate.e_score_correction_bias"),
                    DType::F32,
                    &[geometry.experts],
                ),
                vector_spec(
                    VectorSlot::MoeRoutedNorm,
                    format!("{moe}.routed_expert_norm.weight"),
                    DType::Bf16,
                    &[geometry.latent],
                ),
            ]);
        }
    }
    Ok((matrices, vectors))
}

fn matrix_spec(slot: KimiK3WeightProjection, name: String, rows: usize, cols: usize) -> MatrixSpec {
    MatrixSpec {
        slot,
        name,
        rows,
        cols,
    }
}

fn vector_spec(slot: VectorSlot, name: String, dtype: DType, shape: &[usize]) -> VectorSpec {
    VectorSpec {
        slot,
        name,
        dtype,
        shape: shape.to_vec(),
        elements: 0,
    }
}

fn inspect_vector(index: &TensorIndex, spec: &VectorSpec) -> Result<usize, KimiK3LayerWeightError> {
    let tensor =
        index
            .require(&spec.name)
            .map_err(|source| KimiK3LayerWeightError::Checkpoint {
                name: spec.name.clone(),
                source,
            })?;
    let shape = spec
        .shape
        .iter()
        .map(|&dimension| {
            u64::try_from(dimension).map_err(|_| {
                KimiK3LayerWeightError::Accounting(format!(
                    "tensor {:?} dimension does not fit u64",
                    spec.name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tensor.dtype != spec.dtype || tensor.shape != shape {
        return Err(KimiK3LayerWeightError::Accounting(format!(
            "tensor {:?} must be {} {:?}, got {} {:?}",
            spec.name, spec.dtype, shape, tensor.dtype, tensor.shape
        )));
    }
    let elements = spec.shape.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            KimiK3LayerWeightError::Accounting(format!(
                "tensor {:?} element count overflows usize",
                spec.name
            ))
        })
    })?;
    let elements_u64 = u64::try_from(elements).map_err(|_| {
        KimiK3LayerWeightError::Accounting(format!(
            "tensor {:?} element count does not fit u64",
            spec.name
        ))
    })?;
    let element_bytes = spec.dtype.element_bytes().ok_or_else(|| {
        KimiK3LayerWeightError::Accounting(format!(
            "tensor {:?} has unknown dtype {}",
            spec.name, spec.dtype
        ))
    })?;
    let payload_bytes = elements_u64.checked_mul(element_bytes).ok_or_else(|| {
        KimiK3LayerWeightError::Accounting(format!("tensor {:?} payload bytes overflow", spec.name))
    })?;
    if tensor.declared_elements != elements_u64 || tensor.data_len != payload_bytes {
        return Err(KimiK3LayerWeightError::Accounting(format!(
            "tensor {:?} metadata declares {} elements/{} bytes, expected {elements_u64} elements/{payload_bytes} bytes",
            spec.name, tensor.declared_elements, tensor.data_len
        )));
    }
    Ok(elements)
}

fn take_matrix(
    matrices: &mut BTreeMap<KimiK3WeightProjection, WeightMatrix>,
    slot: KimiK3WeightProjection,
) -> Result<WeightMatrix, KimiK3LayerWeightError> {
    matrices.remove(&slot).ok_or_else(|| {
        KimiK3LayerWeightError::Accounting(format!("missing loaded matrix slot {slot:?}"))
    })
}

fn take_vector(
    vectors: &mut BTreeMap<VectorSlot, Vec<f32>>,
    slot: VectorSlot,
) -> Result<Vec<f32>, KimiK3LayerWeightError> {
    vectors.remove(&slot).ok_or_else(|| {
        KimiK3LayerWeightError::Accounting(format!("missing loaded vector slot {slot:?}"))
    })
}

fn checked_add_usize(
    left: usize,
    right: usize,
    expression: &'static str,
) -> Result<usize, KimiK3LayerWeightError> {
    left.checked_add(right).ok_or_else(|| {
        KimiK3LayerWeightError::InvalidGeometry(format!("{expression} overflows usize"))
    })
}

fn checked_mul_usize(
    left: usize,
    right: usize,
    expression: &'static str,
) -> Result<usize, KimiK3LayerWeightError> {
    left.checked_mul(right).ok_or_else(|| {
        KimiK3LayerWeightError::InvalidGeometry(format!("{expression} overflows usize"))
    })
}

fn checked_add_u64(
    left: u64,
    right: u64,
    expression: &'static str,
) -> Result<u64, KimiK3LayerWeightError> {
    left.checked_add(right)
        .ok_or_else(|| KimiK3LayerWeightError::Accounting(format!("{expression} overflows u64")))
}

fn invalid_geometry<T>(reason: impl Into<String>) -> Result<T, KimiK3LayerWeightError> {
    Err(KimiK3LayerWeightError::InvalidGeometry(reason.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn fixture_dir() -> PathBuf {
        crate::test_support::temp_dir("urbilateria_kimi_k3_layer_weights")
    }

    fn tiny_geometry(
        layer: usize,
        attention: AttentionKind,
        feed_forward: FeedForwardKind,
    ) -> LayerGeometry {
        LayerGeometry {
            layer,
            hidden: 2,
            attention,
            feed_forward,
            attention_heads: 1,
            kda_heads: 1,
            kda_head_dim: 2,
            conv_kernel: 2,
            q_lora: 1,
            kv_lora: 1,
            qk_nope: 1,
            qk_nope_slot: 1,
            value_head: 1,
            dense_intermediate: 2,
            moe_intermediate: 2,
            latent: 2,
            experts: 2,
            shared_experts: 1,
        }
    }

    fn write_fixture(
        path: &Path,
        geometry: &LayerGeometry,
        shape_override: Option<(&str, Vec<usize>)>,
    ) {
        let (matrices, vectors) = layer_specs(geometry).unwrap();
        let mut tensors = matrices
            .into_iter()
            .map(|spec| (spec.name, DType::Bf16, vec![spec.rows, spec.cols]))
            .chain(
                vectors
                    .into_iter()
                    .map(|spec| (spec.name, spec.dtype, spec.shape)),
            )
            .collect::<Vec<_>>();
        // Make physical order differ from the loader's semantic projection order.
        tensors.reverse();

        let mut payload = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, dtype, mut shape) in tensors {
            if let Some((override_name, override_shape)) = &shape_override {
                if name == *override_name {
                    shape = override_shape.clone();
                }
            }
            let elements = shape.iter().product::<usize>();
            let start = payload.len();
            for element in 0..elements {
                let value = if name.ends_with(".self_attn.q_proj.weight")
                    || name.ends_with(".block_sparse_moe.gate.weight")
                {
                    (element + 1) as f32
                } else {
                    1.0
                };
                match dtype {
                    DType::Bf16 => payload.extend(((value.to_bits() >> 16) as u16).to_le_bytes()),
                    DType::F32 => payload.extend(value.to_le_bytes()),
                    _ => unreachable!("tiny fixture uses only BF16 and F32"),
                }
            }
            let end = payload.len();
            let dtype_name = match dtype {
                DType::Bf16 => "BF16",
                DType::F32 => "F32",
                _ => unreachable!("tiny fixture uses only BF16 and F32"),
            };
            header.insert(
                name,
                serde_json::json!({
                    "dtype": dtype_name,
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn tiny_kda_dense_layer_is_compact_and_budgeted_before_io() {
        let geometry = tiny_geometry(0, AttentionKind::Kda, FeedForwardKind::Dense);
        let (matrices, vectors) = layer_specs(&geometry).unwrap();
        assert_eq!(matrices.len(), 11);
        assert_eq!(vectors.len(), 12);
        assert!(matrices.iter().any(|spec| {
            spec.name == "language_model.model.layers.0.self_attn.f_b_proj.weight"
        }));
        assert!(vectors.iter().any(|spec| {
            spec.name == "language_model.model.layers.0.self_attention_res_proj.weight"
                && spec.shape == [1, 2]
        }));

        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("weights.safetensors");
        write_fixture(&shard, &geometry, None);
        let index = TensorIndex::open(&dir).unwrap();
        let manifest = LayerManifest::inspect(&index, &geometry).unwrap();
        assert_eq!(manifest.matrix_resident_bytes, 84);
        assert_eq!(manifest.resident_bytes, 204);

        let layer =
            KimiK3LayerWeights::load_with_geometry(&index, geometry, manifest.resident_bytes)
                .unwrap();
        assert_eq!(layer.resident_bytes(), 204);
        assert_eq!(layer.common.input_layernorm, vec![1.0, 1.0]);
        assert_eq!(layer.common.self_attention_res_projection.len(), 2);
        assert!(matches!(layer.attention, KimiK3AttentionWeights::Kda(_)));
        assert!(matches!(
            layer.feed_forward,
            KimiK3FeedForwardWeights::Dense(_)
        ));
        let query = layer.matrix(KimiK3WeightProjection::KdaQuery).unwrap();
        assert!(matches!(query, WeightMatrix::Bf16(_)));
        assert_eq!(query.matvec(&[1.0, 1.0]).unwrap(), vec![3.0, 7.0]);
        assert!(layer.matrix(KimiK3WeightProjection::MoeRouter).is_none());

        // Poison payload access after indexing. An early read would return EOF, not Budget.
        fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(0)
            .unwrap();
        assert_eq!(
            KimiK3LayerWeights::inspect_resident_bytes_with_geometry(&index, &geometry).unwrap(),
            204
        );
        assert!(matches!(
            KimiK3LayerWeights::load_with_geometry(&index, geometry, 203),
            Err(KimiK3LayerWeightError::Budget {
                required: 204,
                maximum: 203,
                ..
            })
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tiny_mla_moe_layer_loads_only_router_latent_and_shared_trunk() {
        let geometry = tiny_geometry(1, AttentionKind::Mla, FeedForwardKind::Moe);
        let (matrices, vectors) = layer_specs(&geometry).unwrap();
        assert_eq!(matrices.len(), 12);
        assert_eq!(vectors.len(), 10);
        assert!(matrices.iter().any(|spec| {
            spec.name
                == "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight"
        }));
        assert!(matrices.iter().all(|spec| !spec.name.contains(".experts.")));

        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("weights.safetensors"), &geometry, None);
        let index = TensorIndex::open(&dir).unwrap();
        let manifest = LayerManifest::inspect(&index, &geometry).unwrap();
        assert_eq!(manifest.matrix_resident_bytes, 76);
        assert_eq!(manifest.resident_bytes, 148);
        let layer = KimiK3LayerWeights::load_with_geometry(&index, geometry, 148).unwrap();
        assert_eq!(layer.resident_bytes(), 148);
        assert!(matches!(layer.attention, KimiK3AttentionWeights::Mla(_)));
        let KimiK3FeedForwardWeights::Moe(moe) = &layer.feed_forward else {
            panic!("expected MoE trunk")
        };
        assert_eq!(moe.correction_bias, vec![1.0, 1.0]);
        assert_eq!(moe.routed_norm, vec![1.0, 1.0]);
        let router = layer.matrix(KimiK3WeightProjection::MoeRouter).unwrap();
        assert!(matches!(router, WeightMatrix::Bf16(_)));
        assert_eq!(router.matvec(&[1.0, 1.0]).unwrap(), vec![3.0, 7.0]);
        assert!(layer.matrix(KimiK3WeightProjection::DenseGate).is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn vector_rank_drift_is_rejected_during_layer_metadata_preflight() {
        let geometry = tiny_geometry(0, AttentionKind::Kda, FeedForwardKind::Dense);
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        let name = "language_model.model.layers.0.input_layernorm.weight";
        write_fixture(
            &dir.join("weights.safetensors"),
            &geometry,
            Some((name, vec![1, 2])),
        );
        let index = TensorIndex::open(&dir).unwrap();
        assert!(matches!(
            LayerManifest::inspect(&index, &geometry),
            Err(KimiK3LayerWeightError::Accounting(reason))
                if reason.contains("must be BF16 [2]")
        ));
        fs::remove_dir_all(dir).unwrap();
    }
}
