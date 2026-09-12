//! In-memory Qwen3.8 correctness model for tiny end-to-end oracles.
//!
//! This module deliberately owns no architecture equations. It composes the scalar
//! full-attention, Gated DeltaNet, zero-centered RMSNorm, and MoE implementations in
//! the exact decoder-block order and supplies the small [`CausalDecoder`] boundary.
//! Unlike the streamed release runtime, the complete tiny checkpoint is resident.

use super::attention::{AttentionError, FullAttention, FullAttentionCache};
use super::linear_attention::{DeltaNetError, DeltaNetState, GatedDeltaNet};
use super::math::{linear, round_to_bf16_in_place, uses_bf16_output};
use super::moe::{Qwen38Moe, Qwen38MoeError, Qwen38Route};
use super::norm::{zero_centered_rms_norm, NormError};
use crate::generation::CausalDecoder;
use crate::model::{WeightError, WeightMatrix};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_REFERENCE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen38ReferenceLayerKind {
    FullAttention,
    GatedDeltaNet,
}

#[derive(Debug, Clone)]
enum Qwen38ReferenceMixer {
    Full(FullAttention),
    Linear(GatedDeltaNet),
}

impl Qwen38ReferenceMixer {
    fn kind(&self) -> Qwen38ReferenceLayerKind {
        match self {
            Self::Full(_) => Qwen38ReferenceLayerKind::FullAttention,
            Self::Linear(_) => Qwen38ReferenceLayerKind::GatedDeltaNet,
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            Self::Full(attention) => attention.geometry().hidden_size,
            Self::Linear(attention) => attention.geometry().hidden_size,
        }
    }

    fn uses_bf16_activations(&self) -> bool {
        match self {
            Self::Full(attention) => attention.uses_bf16_activations(),
            Self::Linear(attention) => attention.uses_bf16_activations(),
        }
    }

    fn new_state(&self, layer: usize) -> Result<Qwen38ReferenceLayerState, Qwen38ReferenceError> {
        match self {
            Self::Full(attention) => attention
                .new_cache()
                .map(Qwen38ReferenceLayerState::Full)
                .map_err(|source| Qwen38ReferenceError::Attention { layer, source }),
            Self::Linear(attention) => attention
                .new_state()
                .map(Qwen38ReferenceLayerState::Linear)
                .map_err(|source| Qwen38ReferenceError::DeltaNet {
                    layer,
                    source: Box::new(source),
                }),
        }
    }

    fn forward_token(
        &self,
        input: &[f32],
        position: usize,
        state: &mut Qwen38ReferenceLayerState,
        layer: usize,
    ) -> Result<Vec<f32>, Qwen38ReferenceError> {
        match (self, state) {
            (Self::Full(attention), Qwen38ReferenceLayerState::Full(cache)) => attention
                .forward_token(input, position, cache)
                .map_err(|source| Qwen38ReferenceError::Attention { layer, source }),
            (Self::Linear(attention), Qwen38ReferenceLayerState::Linear(state)) => attention
                .forward_token(input, state)
                .map_err(|source| Qwen38ReferenceError::DeltaNet {
                    layer,
                    source: Box::new(source),
                }),
            (mixer, state) => Err(Qwen38ReferenceError::State(format!(
                "layer {layer} is {:?}, but its state is {:?}",
                mixer.kind(),
                state.kind()
            ))),
        }
    }
}

/// One Qwen pre-norm block with either a full-GQA or Gated-DeltaNet token mixer.
#[derive(Debug, Clone)]
pub struct Qwen38ReferenceLayer {
    input_norm: Vec<f32>,
    mixer: Qwen38ReferenceMixer,
    post_attention_norm: Vec<f32>,
    moe: Qwen38Moe,
    bf16_activations: bool,
}

impl Qwen38ReferenceLayer {
    pub fn full(
        input_norm: Vec<f32>,
        attention: FullAttention,
        post_attention_norm: Vec<f32>,
        moe: Qwen38Moe,
    ) -> Result<Self, Qwen38ReferenceError> {
        Self::new(
            input_norm,
            Qwen38ReferenceMixer::Full(attention),
            post_attention_norm,
            moe,
        )
    }

    pub fn linear(
        input_norm: Vec<f32>,
        attention: GatedDeltaNet,
        post_attention_norm: Vec<f32>,
        moe: Qwen38Moe,
    ) -> Result<Self, Qwen38ReferenceError> {
        Self::new(
            input_norm,
            Qwen38ReferenceMixer::Linear(attention),
            post_attention_norm,
            moe,
        )
    }

    fn new(
        input_norm: Vec<f32>,
        mixer: Qwen38ReferenceMixer,
        post_attention_norm: Vec<f32>,
        moe: Qwen38Moe,
    ) -> Result<Self, Qwen38ReferenceError> {
        let hidden_size = mixer.hidden_size();
        validate_vector("layer input norm", &input_norm, hidden_size)?;
        validate_vector(
            "layer post-attention norm",
            &post_attention_norm,
            hidden_size,
        )?;
        if moe.hidden_size() != hidden_size {
            return Err(Qwen38ReferenceError::InvalidShape(format!(
                "{:?} hidden size {hidden_size} differs from MoE hidden size {}",
                mixer.kind(),
                moe.hidden_size()
            )));
        }
        Ok(Self {
            input_norm,
            bf16_activations: mixer.uses_bf16_activations(),
            mixer,
            post_attention_norm,
            moe,
        })
    }

    pub fn kind(&self) -> Qwen38ReferenceLayerKind {
        self.mixer.kind()
    }

    pub fn hidden_size(&self) -> usize {
        self.mixer.hidden_size()
    }

    fn uses_bf16_activations(&self) -> bool {
        self.bf16_activations
    }

    fn new_state(&self, layer: usize) -> Result<Qwen38ReferenceLayerState, Qwen38ReferenceError> {
        self.mixer.new_state(layer)
    }

    /// Decoder order: input norm -> token mixer -> residual -> post norm -> MoE -> residual.
    fn forward_token(
        &self,
        hidden: &[f32],
        position: usize,
        state: &mut Qwen38ReferenceLayerState,
        norm_eps: f32,
        layer: usize,
    ) -> Result<(Vec<f32>, Vec<Qwen38Route>), Qwen38ReferenceError> {
        let mut normalized =
            zero_centered_rms_norm(hidden, &self.input_norm, norm_eps).map_err(|source| {
                Qwen38ReferenceError::Norm {
                    layer: Some(layer),
                    name: "input norm",
                    source,
                }
            })?;
        if self.bf16_activations {
            round_to_bf16_in_place(&mut normalized).map_err(|source| {
                Qwen38ReferenceError::Weight {
                    component: "layer input RMSNorm BF16 cast",
                    source,
                }
            })?;
        }
        let mixed = self
            .mixer
            .forward_token(&normalized, position, state, layer)?;
        let after_mixer =
            residual_add(hidden, &mixed, layer, "token mixer", self.bf16_activations)?;

        let mut normalized =
            zero_centered_rms_norm(&after_mixer, &self.post_attention_norm, norm_eps).map_err(
                |source| Qwen38ReferenceError::Norm {
                    layer: Some(layer),
                    name: "post-attention norm",
                    source,
                },
            )?;
        if self.bf16_activations {
            round_to_bf16_in_place(&mut normalized).map_err(|source| {
                Qwen38ReferenceError::Weight {
                    component: "layer post-attention RMSNorm BF16 cast",
                    source,
                }
            })?;
        }
        let moe = self
            .moe
            .forward(&normalized)
            .map_err(|source| Qwen38ReferenceError::Moe { layer, source })?;
        let hidden = residual_add(
            &after_mixer,
            &moe.hidden,
            layer,
            "MoE",
            self.bf16_activations,
        )?;
        Ok((hidden, moe.routes))
    }
}

#[derive(Debug)]
pub enum Qwen38ReferenceError {
    Weight {
        component: &'static str,
        source: WeightError,
    },
    Attention {
        layer: usize,
        source: AttentionError,
    },
    DeltaNet {
        layer: usize,
        source: Box<DeltaNetError>,
    },
    Moe {
        layer: usize,
        source: Qwen38MoeError,
    },
    Norm {
        layer: Option<usize>,
        name: &'static str,
        source: NormError,
    },
    InvalidShape(String),
    InvalidValue(String),
    State(String),
    StateInstance,
    TokenOutOfRange {
        token: u32,
        vocabulary: usize,
    },
    NonFinite {
        component: &'static str,
        index: usize,
    },
}

impl fmt::Display for Qwen38ReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight { component, source } => {
                write!(formatter, "Qwen3.8 reference {component}: {source}")
            }
            Self::Attention { layer, source } => {
                write!(
                    formatter,
                    "Qwen3.8 reference layer {layer} full attention: {source}"
                )
            }
            Self::DeltaNet { layer, source } => {
                write!(
                    formatter,
                    "Qwen3.8 reference layer {layer} Gated DeltaNet: {source}"
                )
            }
            Self::Moe { layer, source } => {
                write!(formatter, "Qwen3.8 reference layer {layer} MoE: {source}")
            }
            Self::Norm {
                layer: Some(layer),
                name,
                source,
            } => write!(
                formatter,
                "Qwen3.8 reference layer {layer} {name}: {source}"
            ),
            Self::Norm {
                layer: None,
                name,
                source,
            } => write!(formatter, "Qwen3.8 reference {name}: {source}"),
            Self::InvalidShape(reason) => {
                write!(formatter, "invalid Qwen3.8 reference shape: {reason}")
            }
            Self::InvalidValue(reason) => {
                write!(formatter, "invalid Qwen3.8 reference value: {reason}")
            }
            Self::State(reason) => write!(formatter, "invalid Qwen3.8 reference state: {reason}"),
            Self::StateInstance => {
                formatter.write_str("Qwen3.8 reference state belongs to another model")
            }
            Self::TokenOutOfRange { token, vocabulary } => write!(
                formatter,
                "Qwen3.8 token {token} is outside vocabulary 0..{vocabulary}"
            ),
            Self::NonFinite { component, index } => write!(
                formatter,
                "Qwen3.8 reference {component} contains NaN or infinity at index {index}"
            ),
        }
    }
}

impl std::error::Error for Qwen38ReferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight { source, .. } => Some(source),
            Self::Attention { source, .. } => Some(source),
            Self::DeltaNet { source, .. } => Some(source.as_ref()),
            Self::Moe { source, .. } => Some(source),
            Self::Norm { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Per-layer causal state. The variant must match the corresponding model layer.
#[derive(Debug, Clone)]
pub enum Qwen38ReferenceLayerState {
    Full(FullAttentionCache),
    Linear(DeltaNetState),
}

impl Qwen38ReferenceLayerState {
    pub fn kind(&self) -> Qwen38ReferenceLayerKind {
        match self {
            Self::Full(_) => Qwen38ReferenceLayerKind::FullAttention,
            Self::Linear(_) => Qwen38ReferenceLayerKind::GatedDeltaNet,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Full(cache) => cache.len(),
            Self::Linear(state) => state.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stored_f32_elements(&self) -> usize {
        match self {
            Self::Full(cache) => cache.stored_f32_elements(),
            Self::Linear(state) => state.stored_f32_elements(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen38ReferenceState {
    instance_id: u64,
    position: usize,
    layers: Vec<Qwen38ReferenceLayerState>,
    routes_by_layer: Vec<Vec<Qwen38Route>>,
}

impl Qwen38ReferenceState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_position(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).map(Qwen38ReferenceLayerState::len)
    }

    pub fn layer_kind(&self, layer: usize) -> Option<Qwen38ReferenceLayerKind> {
        self.layers.get(layer).map(Qwen38ReferenceLayerState::kind)
    }

    pub fn routes_by_layer(&self) -> &[Vec<Qwen38Route>] {
        &self.routes_by_layer
    }

    pub fn cached_f32_elements(&self) -> usize {
        self.layers
            .iter()
            .map(Qwen38ReferenceLayerState::stored_f32_elements)
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen38ReferenceStep {
    pub logits: Vec<f32>,
    /// Hidden state after each complete block, before the final model RMSNorm.
    pub layer_hidden_states: Vec<Vec<f32>>,
    pub routes_by_layer: Vec<Vec<Qwen38Route>>,
}

/// A fully resident, batch-one Qwen3.8 model for tiny correctness fixtures.
#[derive(Debug, Clone)]
pub struct Qwen38ReferenceModel {
    instance_id: u64,
    embedding: WeightMatrix,
    layers: Vec<Qwen38ReferenceLayer>,
    final_norm: Vec<f32>,
    lm_head: WeightMatrix,
    norm_eps: f32,
    bf16_activations: bool,
}

impl Qwen38ReferenceModel {
    pub fn new(
        embedding: WeightMatrix,
        layers: Vec<Qwen38ReferenceLayer>,
        final_norm: Vec<f32>,
        lm_head: WeightMatrix,
        norm_eps: f32,
    ) -> Result<Self, Qwen38ReferenceError> {
        if embedding.rows() == 0 || embedding.cols() == 0 {
            return Err(Qwen38ReferenceError::InvalidShape(
                "embedding dimensions must be non-zero".to_owned(),
            ));
        }
        if layers.is_empty() {
            return Err(Qwen38ReferenceError::InvalidShape(
                "at least one decoder layer is required".to_owned(),
            ));
        }
        let hidden_size = embedding.cols();
        if let Some((layer, value)) = layers
            .iter()
            .enumerate()
            .find(|(_, value)| value.hidden_size() != hidden_size)
        {
            return Err(Qwen38ReferenceError::InvalidShape(format!(
                "embedding hidden size {hidden_size} differs from layer {layer} hidden size {}",
                value.hidden_size()
            )));
        }
        validate_vector("final norm", &final_norm, hidden_size)?;
        if lm_head.rows() != embedding.rows() || lm_head.cols() != hidden_size {
            return Err(Qwen38ReferenceError::InvalidShape(format!(
                "embedding is [{}, {}], LM head is [{}, {}]",
                embedding.rows(),
                embedding.cols(),
                lm_head.rows(),
                lm_head.cols()
            )));
        }
        if !norm_eps.is_finite() || norm_eps <= 0.0 {
            return Err(Qwen38ReferenceError::InvalidValue(
                "RMSNorm epsilon must be finite and positive".to_owned(),
            ));
        }
        let bf16_activations = uses_bf16_output(&lm_head);
        if uses_bf16_output(&embedding) != bf16_activations {
            return Err(Qwen38ReferenceError::InvalidShape(
                "embedding and LM head must share one activation dtype".to_owned(),
            ));
        }
        if layers
            .iter()
            .any(|layer| layer.uses_bf16_activations() != bf16_activations)
        {
            return Err(Qwen38ReferenceError::InvalidShape(
                "decoder layers and LM head must share one activation dtype".to_owned(),
            ));
        }
        Ok(Self {
            instance_id: NEXT_REFERENCE_ID.fetch_add(1, Ordering::Relaxed),
            embedding,
            layers,
            final_norm,
            lm_head,
            norm_eps,
            bf16_activations,
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.embedding.rows()
    }

    pub fn hidden_size(&self) -> usize {
        self.embedding.cols()
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn new_state(&self) -> Result<Qwen38ReferenceState, Qwen38ReferenceError> {
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(layer, value)| value.new_state(layer))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Qwen38ReferenceState {
            instance_id: self.instance_id,
            position: 0,
            routes_by_layer: vec![Vec::new(); layers.len()],
            layers,
        })
    }

    /// Runs one token transactionally. No cache, route, or position mutation is
    /// committed unless the embedding, every block, final norm, and LM head succeed.
    pub fn forward_token(
        &self,
        token: u32,
        state: &mut Qwen38ReferenceState,
    ) -> Result<Qwen38ReferenceStep, Qwen38ReferenceError> {
        self.validate_state(state)?;
        let mut candidate = state.clone();
        let step = self.forward_token_inner(token, &mut candidate)?;
        *state = candidate;
        Ok(step)
    }

    fn forward_token_inner(
        &self,
        token: u32,
        state: &mut Qwen38ReferenceState,
    ) -> Result<Qwen38ReferenceStep, Qwen38ReferenceError> {
        let token_index = usize::try_from(token).unwrap_or(usize::MAX);
        if token_index >= self.vocabulary_size() {
            return Err(Qwen38ReferenceError::TokenOutOfRange {
                token,
                vocabulary: self.vocabulary_size(),
            });
        }
        let mut hidden =
            self.embedding
                .row(token_index)
                .map_err(|source| Qwen38ReferenceError::Weight {
                    component: "embedding",
                    source,
                })?;
        ensure_finite("embedding output", &hidden)?;

        let position = state.position;
        let mut layer_hidden_states = Vec::with_capacity(self.layers.len());
        let mut routes_by_layer = Vec::with_capacity(self.layers.len());
        for (layer, (block, layer_state)) in self.layers.iter().zip(&mut state.layers).enumerate() {
            let (next, routes) =
                block.forward_token(&hidden, position, layer_state, self.norm_eps, layer)?;
            hidden = next;
            layer_hidden_states.push(hidden.clone());
            routes_by_layer.push(routes);
        }

        let mut final_hidden = zero_centered_rms_norm(&hidden, &self.final_norm, self.norm_eps)
            .map_err(|source| Qwen38ReferenceError::Norm {
                layer: None,
                name: "final norm",
                source,
            })?;
        if self.bf16_activations {
            round_to_bf16_in_place(&mut final_hidden).map_err(|source| {
                Qwen38ReferenceError::Weight {
                    component: "final RMSNorm BF16 cast",
                    source,
                }
            })?;
        }
        let logits = linear(&self.lm_head, &final_hidden).map_err(|source| {
            Qwen38ReferenceError::Weight {
                component: "LM head",
                source,
            }
        })?;
        ensure_finite("LM-head logits", &logits)?;

        state.position = position
            .checked_add(1)
            .ok_or_else(|| Qwen38ReferenceError::State("position overflows usize".to_owned()))?;
        state.routes_by_layer = routes_by_layer.clone();
        Ok(Qwen38ReferenceStep {
            logits,
            layer_hidden_states,
            routes_by_layer,
        })
    }

    fn validate_state(&self, state: &Qwen38ReferenceState) -> Result<(), Qwen38ReferenceError> {
        if state.instance_id != self.instance_id {
            return Err(Qwen38ReferenceError::StateInstance);
        }
        if state.layers.len() != self.layers.len()
            || state.routes_by_layer.len() != self.layers.len()
        {
            return Err(Qwen38ReferenceError::State(format!(
                "state has {} layer caches and {} route lists, model has {} layers",
                state.layers.len(),
                state.routes_by_layer.len(),
                self.layers.len()
            )));
        }
        for (layer, (block, layer_state)) in self.layers.iter().zip(&state.layers).enumerate() {
            if block.kind() != layer_state.kind() {
                return Err(Qwen38ReferenceError::State(format!(
                    "layer {layer} is {:?}, state is {:?}",
                    block.kind(),
                    layer_state.kind()
                )));
            }
            if layer_state.len() != state.position {
                return Err(Qwen38ReferenceError::State(format!(
                    "layer {layer} cache position {} differs from model position {}",
                    layer_state.len(),
                    state.position
                )));
            }
        }
        Ok(())
    }
}

impl CausalDecoder for Qwen38ReferenceModel {
    type State = Qwen38ReferenceState;
    type Error = Qwen38ReferenceError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        Qwen38ReferenceModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(Qwen38ReferenceModel::forward_token(self, token, state)?.logits)
    }
}

fn residual_add(
    residual: &[f32],
    delta: &[f32],
    layer: usize,
    component: &'static str,
    bf16_activations: bool,
) -> Result<Vec<f32>, Qwen38ReferenceError> {
    if residual.len() != delta.len() {
        return Err(Qwen38ReferenceError::InvalidShape(format!(
            "layer {layer} {component} returned {} values for residual width {}",
            delta.len(),
            residual.len()
        )));
    }
    let mut output = residual
        .iter()
        .zip(delta)
        .map(|(&residual, &delta)| residual + delta)
        .collect::<Vec<_>>();
    if bf16_activations {
        round_to_bf16_in_place(&mut output).map_err(|source| Qwen38ReferenceError::Weight {
            component: "residual BF16 cast",
            source,
        })?;
    }
    ensure_finite("residual output", &output)?;
    Ok(output)
}

fn validate_vector(
    name: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), Qwen38ReferenceError> {
    if values.len() != expected {
        return Err(Qwen38ReferenceError::InvalidShape(format!(
            "{name} has {} values, expected {expected}",
            values.len()
        )));
    }
    ensure_finite(name, values)
}

fn ensure_finite(component: &'static str, values: &[f32]) -> Result<(), Qwen38ReferenceError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(Qwen38ReferenceError::NonFinite { component, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::attention::FullAttentionGeometry;
    use super::super::expert::Qwen38Expert;
    use super::super::linear_attention::DeltaNetGeometry;
    use super::super::moe::Qwen38MoeGeometry;
    use super::*;
    use crate::generation::{generate, GenerationConfig, StopReason};
    use crate::model::{Bf16Matrix, DenseMatrix};

    const EPS: f32 = 1e-6;

    fn dense(rows: usize, columns: usize, values: Vec<f32>) -> WeightMatrix {
        WeightMatrix::F32(DenseMatrix::new(rows, columns, values).unwrap())
    }

    fn zeros(rows: usize, columns: usize) -> WeightMatrix {
        dense(rows, columns, vec![0.0; rows * columns])
    }

    fn bf16_dense(rows: usize, columns: usize, values: Vec<f32>) -> WeightMatrix {
        let bytes = values
            .into_iter()
            .flat_map(|value| {
                let bits = value.to_bits();
                let bias = 0x7fff + ((bits >> 16) & 1);
                ((bits.wrapping_add(bias) >> 16) as u16).to_le_bytes()
            })
            .collect();
        WeightMatrix::Bf16(Bf16Matrix::from_le_bytes(rows, columns, bytes).unwrap())
    }

    fn bf16_vec(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| {
                let bits = value.to_bits();
                let bias = 0x7fff + ((bits >> 16) & 1);
                f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
            })
            .collect()
    }

    fn zero_expert() -> Qwen38Expert {
        Qwen38Expert::new(zeros(1, 2), zeros(1, 2), zeros(2, 1)).unwrap()
    }

    fn zero_moe() -> Qwen38Moe {
        Qwen38Moe::new(
            zeros(2, 2),
            vec![zero_expert(), zero_expert()],
            zero_expert(),
            zeros(1, 2),
            Qwen38MoeGeometry { top_k: 1 },
        )
        .unwrap()
    }

    fn tiny_expert(gate: [f32; 2], up: [f32; 2], down: [f32; 2]) -> Qwen38Expert {
        Qwen38Expert::new(
            dense(1, 2, gate.to_vec()),
            dense(1, 2, up.to_vec()),
            dense(2, 1, down.to_vec()),
        )
        .unwrap()
    }

    fn nonzero_moe() -> Qwen38Moe {
        Qwen38Moe::new(
            dense(2, 2, vec![0.8, -0.3, -0.2, 0.7]),
            vec![
                tiny_expert([0.5, 0.1], [0.3, -0.2], [0.4, -0.1]),
                tiny_expert([-0.4, 0.6], [0.2, 0.5], [-0.3, 0.7]),
            ],
            tiny_expert([0.25, -0.15], [0.4, 0.2], [0.6, -0.2]),
            dense(1, 2, vec![0.3, -0.4]),
            Qwen38MoeGeometry { top_k: 1 },
        )
        .unwrap()
    }

    fn zero_delta_net() -> GatedDeltaNet {
        let geometry = DeltaNetGeometry {
            hidden_size: 2,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 1,
            value_head_dim: 1,
            conv_kernel_size: 2,
            norm_eps: EPS,
        };
        GatedDeltaNet::new_mixed(
            geometry,
            zeros(3, 2),
            zeros(1, 2),
            zeros(1, 2),
            zeros(1, 2),
            vec![0.0; 6],
            vec![0.0],
            vec![0.0],
            vec![1.0],
            zeros(2, 1),
        )
        .unwrap()
    }

    fn nonzero_delta_net() -> GatedDeltaNet {
        let geometry = DeltaNetGeometry {
            hidden_size: 2,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 1,
            value_head_dim: 1,
            conv_kernel_size: 2,
            norm_eps: EPS,
        };
        GatedDeltaNet::new_mixed(
            geometry,
            dense(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.6, 0.4]),
            dense(1, 2, vec![0.5, -0.25]),
            dense(1, 2, vec![0.2, 0.1]),
            dense(1, 2, vec![-0.1, 0.3]),
            vec![0.25, 0.75, -0.2, 0.8, 0.1, 0.9],
            vec![0.1],
            vec![-0.2],
            vec![1.1],
            dense(2, 1, vec![0.7, -0.3]),
        )
        .unwrap()
    }

    fn zero_full_attention() -> FullAttention {
        let geometry = FullAttentionGeometry {
            hidden_size: 2,
            num_query_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            norm_eps: EPS,
            rope_theta: 10_000.0,
        };
        FullAttention::new_mixed(
            geometry,
            zeros(4, 2),
            zeros(2, 2),
            zeros(2, 2),
            zeros(2, 2),
            vec![0.0; 2],
            vec![0.0; 2],
        )
        .unwrap()
    }

    fn nonzero_full_attention() -> FullAttention {
        let geometry = FullAttentionGeometry {
            hidden_size: 2,
            num_query_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            norm_eps: EPS,
            rope_theta: 10_000.0,
        };
        FullAttention::new_mixed(
            geometry,
            dense(4, 2, vec![1.0, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, -0.25]),
            dense(2, 2, vec![0.8, 0.1, -0.2, 0.9]),
            dense(2, 2, vec![0.7, -0.3, 0.2, 0.6]),
            dense(2, 2, vec![0.5, 0.1, -0.2, 0.8]),
            vec![0.0; 2],
            vec![0.0; 2],
        )
        .unwrap()
    }

    fn tiny_model(failing_lm_head: bool) -> Qwen38ReferenceModel {
        let embedding = dense(4, 2, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]);
        let linear =
            Qwen38ReferenceLayer::linear(vec![0.0; 2], zero_delta_net(), vec![0.0; 2], zero_moe())
                .unwrap();
        let full = Qwen38ReferenceLayer::full(
            vec![0.0; 2],
            zero_full_attention(),
            vec![0.0; 2],
            zero_moe(),
        )
        .unwrap();
        let lm_head = if failing_lm_head {
            dense(4, 2, vec![f32::MAX, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        } else {
            // 0 -> 1, 1 -> 2, 2 -> 0 under greedy decoding.
            dense(4, 2, vec![-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        };
        Qwen38ReferenceModel::new(embedding, vec![linear, full], vec![0.0; 2], lm_head, EPS)
            .unwrap()
    }

    fn nonzero_tiny_model() -> Qwen38ReferenceModel {
        let embedding = dense(3, 2, vec![1.0, -0.5, -0.25, 0.75, 0.4, 0.9]);
        let linear = Qwen38ReferenceLayer::linear(
            vec![0.1, -0.05],
            nonzero_delta_net(),
            vec![0.02, -0.03],
            nonzero_moe(),
        )
        .unwrap();
        let full = Qwen38ReferenceLayer::full(
            vec![-0.04, 0.08],
            nonzero_full_attention(),
            vec![0.06, -0.02],
            nonzero_moe(),
        )
        .unwrap();
        Qwen38ReferenceModel::new(
            embedding,
            vec![linear, full],
            vec![0.05, -0.05],
            dense(3, 2, vec![0.8, -0.2, -0.4, 0.9, 0.3, 0.7]),
            EPS,
        )
        .unwrap()
    }

    fn nonzero_tiny_model_bf16() -> Qwen38ReferenceModel {
        let expert = |gate: [f32; 2], up: [f32; 2], down: [f32; 2]| {
            Qwen38Expert::new(
                bf16_dense(1, 2, gate.to_vec()),
                bf16_dense(1, 2, up.to_vec()),
                bf16_dense(2, 1, down.to_vec()),
            )
            .unwrap()
        };
        let moe = || {
            Qwen38Moe::new(
                bf16_dense(2, 2, vec![0.8, -0.3, -0.2, 0.7]),
                vec![
                    expert([0.5, 0.1], [0.3, -0.2], [0.4, -0.1]),
                    expert([-0.4, 0.6], [0.2, 0.5], [-0.3, 0.7]),
                ],
                expert([0.25, -0.15], [0.4, 0.2], [0.6, -0.2]),
                bf16_dense(1, 2, vec![0.3, -0.4]),
                Qwen38MoeGeometry { top_k: 1 },
            )
            .unwrap()
        };
        let delta = GatedDeltaNet::new_mixed(
            DeltaNetGeometry {
                hidden_size: 2,
                num_key_heads: 1,
                num_value_heads: 1,
                key_head_dim: 1,
                value_head_dim: 1,
                conv_kernel_size: 2,
                norm_eps: EPS,
            },
            bf16_dense(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.6, 0.4]),
            bf16_dense(1, 2, vec![0.5, -0.25]),
            bf16_dense(1, 2, vec![0.2, 0.1]),
            bf16_dense(1, 2, vec![-0.1, 0.3]),
            bf16_vec(&[0.25, 0.75, -0.2, 0.8, 0.1, 0.9]),
            bf16_vec(&[0.1]),
            bf16_vec(&[-0.2]),
            bf16_vec(&[1.1]),
            bf16_dense(2, 1, vec![0.7, -0.3]),
        )
        .unwrap();
        let full = FullAttention::new_mixed(
            FullAttentionGeometry {
                hidden_size: 2,
                num_query_heads: 1,
                num_key_value_heads: 1,
                head_dim: 2,
                rotary_dim: 2,
                norm_eps: EPS,
                rope_theta: 10_000.0,
            },
            bf16_dense(4, 2, vec![1.0, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, -0.25]),
            bf16_dense(2, 2, vec![0.8, 0.1, -0.2, 0.9]),
            bf16_dense(2, 2, vec![0.7, -0.3, 0.2, 0.6]),
            bf16_dense(2, 2, vec![0.5, 0.1, -0.2, 0.8]),
            vec![0.0; 2],
            vec![0.0; 2],
        )
        .unwrap();
        let layers = vec![
            Qwen38ReferenceLayer::linear(
                bf16_vec(&[0.1, -0.05]),
                delta,
                bf16_vec(&[0.02, -0.03]),
                moe(),
            )
            .unwrap(),
            Qwen38ReferenceLayer::full(
                bf16_vec(&[-0.04, 0.08]),
                full,
                bf16_vec(&[0.06, -0.02]),
                moe(),
            )
            .unwrap(),
        ];
        Qwen38ReferenceModel::new(
            bf16_dense(3, 2, vec![1.0, -0.5, -0.25, 0.75, 0.4, 0.9]),
            layers,
            bf16_vec(&[0.05, -0.05]),
            bf16_dense(3, 2, vec![0.8, -0.2, -0.4, 0.9, 0.3, 0.7]),
            EPS,
        )
        .unwrap()
    }

    #[test]
    fn two_layer_model_generates_deterministic_tokens_through_causal_decoder() {
        let model = tiny_model(false);
        let output = generate(
            &model,
            &[0],
            &GenerationConfig::greedy(3, Vec::new()),
            |_| {},
        )
        .unwrap();
        assert_eq!(output.generated_tokens, [1, 2, 0]);
        assert_eq!(output.stop_reason, StopReason::MaxNewTokens);
    }

    #[test]
    fn nonzero_two_layer_forward_matches_independent_transformers_oracle() {
        let model = nonzero_tiny_model();
        let mut state = model.new_state().unwrap();
        let step = model.forward_token(0, &mut state).unwrap();
        let oracle: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen3_8_transformers_tiny.json"
        )))
        .unwrap();
        assert_eq!(oracle["model_family"], "qwen3_8");
        assert_eq!(
            oracle["oracle"]["implementation"],
            "transformers.Qwen3_5MoeForCausalLM"
        );
        assert_eq!(state.position(), 1);
        let expected_layers = oracle["expected"]["layer_hidden_states"]
            .as_array()
            .unwrap();
        assert_eq!(step.layer_hidden_states.len(), expected_layers.len());
        for (actual, expected) in step.layer_hidden_states.iter().zip(expected_layers) {
            let expected = expected.as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (&actual, expected) in actual.iter().zip(expected) {
                let expected = expected.as_f64().unwrap() as f32;
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
        let expected_logits = oracle["expected"]["logits"].as_array().unwrap();
        assert_eq!(step.logits.len(), expected_logits.len());
        for (&actual, expected) in step.logits.iter().zip(expected_logits) {
            let expected = expected.as_f64().unwrap() as f32;
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        let expected_routes = oracle["expected"]["routes"].as_array().unwrap();
        for (actual, expected) in step.routes_by_layer.iter().zip(expected_routes) {
            assert_eq!(actual.len(), 1);
            assert_eq!(
                actual[0].expert,
                expected["experts"][0].as_u64().unwrap() as usize
            );
            let expected_weight = expected["weights"][0].as_f64().unwrap() as f32;
            assert!((actual[0].weight - expected_weight).abs() < 1e-6);
        }
        match &state.layers[0] {
            Qwen38ReferenceLayerState::Linear(delta) => {
                assert!(delta.conv_state().iter().any(|value| *value != 0.0));
                assert!(delta.recurrent_state().iter().any(|value| *value != 0.0));
            }
            Qwen38ReferenceLayerState::Full(_) => panic!("layer 0 must be recurrent"),
        }
        assert_eq!(state.layers[1].stored_f32_elements(), 4);
    }

    #[test]
    fn bf16_two_layer_forward_matches_transformers_dtype_oracle() {
        fn rank(value: f32) -> u16 {
            let bits = (value.to_bits() >> 16) as u16;
            if bits & 0x8000 == 0 {
                bits | 0x8000
            } else {
                !bits
            }
        }
        fn close(actual: f32, expected: f32) {
            let ulps = rank(actual).abs_diff(rank(expected));
            assert!(ulps <= 1, "{actual} != {expected} ({ulps} BF16 ULPs)");
        }
        let oracle: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen3_8_transformers_tiny.json"
        )))
        .unwrap();
        let expected = &oracle["expected_bf16"];
        let model = nonzero_tiny_model_bf16();
        let mut state = model.new_state().unwrap();
        let step = model.forward_token(0, &mut state).unwrap();
        for (actual, expected) in step
            .layer_hidden_states
            .iter()
            .zip(expected["layer_hidden_states"].as_array().unwrap())
        {
            for (&actual, expected) in actual.iter().zip(expected.as_array().unwrap()) {
                close(actual, expected.as_f64().unwrap() as f32);
            }
        }
        for (&actual, expected) in step
            .logits
            .iter()
            .zip(expected["logits"].as_array().unwrap())
        {
            close(actual, expected.as_f64().unwrap() as f32);
        }
    }

    #[test]
    fn every_token_advances_linear_and_full_layer_state_together() {
        let model = tiny_model(false);
        let mut state = model.new_state().unwrap();
        assert_eq!(
            state.layer_kind(0),
            Some(Qwen38ReferenceLayerKind::GatedDeltaNet)
        );
        assert_eq!(
            state.layer_kind(1),
            Some(Qwen38ReferenceLayerKind::FullAttention)
        );

        for (position, token) in [0, 1, 2].into_iter().enumerate() {
            let step = model.forward_token(token, &mut state).unwrap();
            assert_eq!(state.position(), position + 1);
            assert_eq!(state.layer_position(0), Some(position + 1));
            assert_eq!(state.layer_position(1), Some(position + 1));
            assert_eq!(step.layer_hidden_states.len(), 2);
            assert_eq!(step.routes_by_layer.len(), 2);
            assert!(step.routes_by_layer.iter().all(|routes| routes.len() == 1));
        }
    }

    #[test]
    fn late_lm_head_failure_rolls_back_every_layer_and_routes() {
        let model = tiny_model(true);
        let mut state = model.new_state().unwrap();

        // Token 1 is orthogonal to the overflowing LM-head row and succeeds.
        model.forward_token(1, &mut state).unwrap();
        let routes_before = state.routes_by_layer().to_vec();
        assert_eq!(state.position(), 1);

        // Token 0 reaches the LM head only after both staged layer states advance.
        let error = model.forward_token(0, &mut state).unwrap_err();
        assert!(matches!(
            error,
            Qwen38ReferenceError::NonFinite {
                component: "LM-head logits",
                ..
            }
        ));
        assert_eq!(state.position(), 1);
        assert_eq!(state.layer_position(0), Some(1));
        assert_eq!(state.layer_position(1), Some(1));
        assert_eq!(state.routes_by_layer(), routes_before);
    }

    #[test]
    fn constructor_rejects_lm_head_and_layer_shape_drift() {
        let model = tiny_model(false);
        assert_eq!(model.vocabulary_size(), 4);
        assert_eq!(model.hidden_size(), 2);
        assert_eq!(model.layer_count(), 2);

        let error = Qwen38ReferenceModel::new(
            zeros(4, 2),
            vec![Qwen38ReferenceLayer::linear(
                vec![0.0; 2],
                zero_delta_net(),
                vec![0.0; 2],
                zero_moe(),
            )
            .unwrap()],
            vec![0.0; 2],
            zeros(3, 2),
            EPS,
        )
        .unwrap_err();
        assert!(matches!(error, Qwen38ReferenceError::InvalidShape(_)));
    }
}
