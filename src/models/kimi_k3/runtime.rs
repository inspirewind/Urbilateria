//! Scalar, layer-streamed Kimi-K3 text correctness runtime.
//!
//! The runtime keeps routed experts in native MXFP4, streams one compact-BF16 decoder layer at a
//! time, stores fixed-size KDA recurrence, and keeps only the compressed 512+64 MLA representation
//! per cached token. It is an executable specification whose full-stack logits have been checked
//! against an independent PyTorch oracle; it is not a claim of production throughput.

use super::attention::{
    kda_step, mla_step, mla_two_pass_scratch, AttentionError, KdaParameters, KdaState, MlaCache,
    MlaParameters, Projection as AttentionProjection, Projector as AttentionProjector,
};
use super::expert::{KimiK3ExpertError, KimiK3ExpertStore, KIMI_K3_EXPERT_RESIDENT_BYTES};
use super::geometry::{
    runtime_geometry_from_config, AttentionLayerKind, GeometryError, KimiK3RuntimeGeometry,
};
use super::math::{fold_attn_res_weights, situ_glu, KimiK3MathError};
use super::moe::{latent_moe, KimiK3MoeError, RouteChoice};
use super::moe_runtime::KimiK3MoeProjector;
use super::residual::{AttnResidualError, AttnResidualState};
use super::schema::{self, KimiK3Requirements, SchemaError};
use super::weights::{
    KimiK3AttentionWeights, KimiK3FeedForwardWeights, KimiK3LayerWeightError, KimiK3LayerWeights,
};
use super::KimiK3Config;
use crate::config::ConfigError;
use crate::generation::CausalDecoder;
use crate::math::{rms_norm, MathError};
use crate::model::{WeightError, WeightMatrix};
use crate::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use crate::storage::{
    load_reference_matrix_row, load_reference_values, load_reference_vector,
    streamed_reference_matvec, SafetensorError, TensorIndex, TensorLoadError,
};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const ROOT_PREFIX: &str = "language_model.model";
const LM_HEAD: &str = "language_model.lm_head.weight";
const LM_HEAD_ROWS_PER_CHUNK: usize = 4_096;
const FIXED_PREFILL_SCRATCH_BYTES: u64 = 512 * 1024 * 1024;

static NEXT_KIMI_K3_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub enum KimiK3RuntimeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Schema(SchemaError),
    Geometry(GeometryError),
    LayerWeight(KimiK3LayerWeightError),
    Tensor(TensorLoadError),
    Weight(WeightError),
    Math(MathError),
    KimiMath(KimiK3MathError),
    Attention(AttentionError),
    Residual(AttnResidualError),
    Moe(KimiK3MoeError),
    Expert(Box<KimiK3ExpertError>),
    Invalid(String),
    Budget {
        component: &'static str,
        required: u64,
        maximum: u64,
    },
    TokenOutOfRange {
        token: u32,
        vocabulary: usize,
    },
    ContextExhausted {
        position: usize,
        limit: usize,
    },
    StateInstance,
    StatePoisoned,
}

impl fmt::Display for KimiK3RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Schema(error) => error.fmt(formatter),
            Self::Geometry(error) => error.fmt(formatter),
            Self::LayerWeight(error) => error.fmt(formatter),
            Self::Tensor(error) => error.fmt(formatter),
            Self::Weight(error) => error.fmt(formatter),
            Self::Math(error) => error.fmt(formatter),
            Self::KimiMath(error) => error.fmt(formatter),
            Self::Attention(error) => error.fmt(formatter),
            Self::Residual(error) => error.fmt(formatter),
            Self::Moe(error) => error.fmt(formatter),
            Self::Expert(error) => error.fmt(formatter),
            Self::Invalid(reason) => write!(formatter, "invalid Kimi-K3 runtime: {reason}"),
            Self::Budget {
                component,
                required,
                maximum,
            } => write!(
                formatter,
                "Kimi-K3 {component} needs {required} bytes, authorized maximum is {maximum}"
            ),
            Self::TokenOutOfRange { token, vocabulary } => write!(
                formatter,
                "Kimi-K3 token ID {token} is outside vocabulary 0..{vocabulary}"
            ),
            Self::ContextExhausted { position, limit } => write!(
                formatter,
                "Kimi-K3 runtime position {position} reaches context limit {limit}"
            ),
            Self::StateInstance => {
                formatter.write_str("Kimi-K3 state belongs to another runtime instance")
            }
            Self::StatePoisoned => formatter.write_str(
                "Kimi-K3 state is poisoned by a prior partial forward failure; create a new state",
            ),
        }
    }
}

impl std::error::Error for KimiK3RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Checkpoint(error) => Some(error),
            Self::Schema(error) => Some(error),
            Self::Geometry(error) => Some(error),
            Self::LayerWeight(error) => Some(error),
            Self::Tensor(error) => Some(error),
            Self::Weight(error) => Some(error),
            Self::Math(error) => Some(error),
            Self::KimiMath(error) => Some(error),
            Self::Attention(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::Moe(error) => Some(error),
            Self::Expert(error) => Some(error),
            Self::Invalid(_)
            | Self::Budget { .. }
            | Self::TokenOutOfRange { .. }
            | Self::ContextExhausted { .. }
            | Self::StateInstance
            | Self::StatePoisoned => None,
        }
    }
}

macro_rules! from_runtime_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for KimiK3RuntimeError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

from_runtime_error!(ConfigError, Config);
from_runtime_error!(SafetensorError, Checkpoint);
from_runtime_error!(SchemaError, Schema);
from_runtime_error!(GeometryError, Geometry);
from_runtime_error!(KimiK3LayerWeightError, LayerWeight);
from_runtime_error!(TensorLoadError, Tensor);
from_runtime_error!(WeightError, Weight);
from_runtime_error!(MathError, Math);
from_runtime_error!(KimiK3MathError, KimiMath);
from_runtime_error!(AttentionError, Attention);
from_runtime_error!(AttnResidualError, Residual);
from_runtime_error!(KimiK3MoeError, Moe);

impl From<KimiK3ExpertError> for KimiK3RuntimeError {
    fn from(value: KimiK3ExpertError) -> Self {
        Self::Expert(Box::new(value))
    }
}

/// Header-only memory contract for one complete official Kimi-K3 checkpoint.
#[derive(Debug, Clone)]
pub struct KimiK3RuntimeRequirements {
    pub schema: KimiK3Requirements,
    /// Final RMSNorm plus the pre-folded model-level AttnRes vector. This small allocation is
    /// covered by the fixed scratch reserve rather than added to `resident_core_bytes`.
    pub root_resident_bytes: u64,
    pub streamed_layer_bytes: u64,
    pub kda_state_bytes: u64,
    pub attn_res_working_bytes: u64,
    pub resident_core_bytes: u64,
    pub prefill_scratch_bytes_per_token: u64,
    /// Fixed index/allocator/transient reserve plus layer-wise
    /// `[tokens, hidden/snapshots/scores]` storage.
    pub prefill_scratch_bytes: u64,
    /// Diagnostic subset of prefill scratch for the longest compressed MLA attention call. Its
    /// context-sized score matrix is already included in `prefill_scratch_bytes_per_token`; only
    /// the fixed 98,304-byte expanded-KV vector draws from the fixed reserve.
    pub mla_two_pass_scratch_bytes: u64,
    pub mla_cache_bytes: u64,
    pub routed_expert_bytes: u64,
    pub expert_cache_bytes: u64,
    /// Value expected in `RuntimeLoadOptions::resident_budget_bytes`.
    pub resident_bytes: u64,
    pub context_limit: usize,
    pub expert_slots_per_layer: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KimiK3RuntimeStep {
    pub logits: Vec<f32>,
    pub routes_by_layer: Vec<Vec<RouteChoice>>,
}

#[derive(Debug)]
pub struct KimiK3RuntimeModel {
    instance_id: u64,
    config: KimiK3Config,
    geometry: KimiK3RuntimeGeometry,
    index: Arc<TensorIndex>,
    final_norm: Vec<f32>,
    folded_output_attn_res: Vec<f32>,
    requirements: KimiK3RuntimeRequirements,
    context_limit: usize,
    expert_slots_per_layer: usize,
    maximum_expert_bytes: u64,
}

impl KimiK3RuntimeModel {
    /// Strictly validates all 96 shards and computes execution memory without reading payloads.
    pub fn inspect_requirements(
        model_dir: impl AsRef<Path>,
        context_limit: usize,
        expert_slots_per_layer: usize,
    ) -> Result<KimiK3RuntimeRequirements, KimiK3RuntimeError> {
        let config = KimiK3Config::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        inspect_runtime_requirements(&config, &index, context_limit, expert_slots_per_layer)
    }

    pub fn load(
        model_dir: impl AsRef<Path>,
        options: RuntimeLoadOptions,
    ) -> Result<Self, KimiK3RuntimeError> {
        if options.context_limit == 0 || options.resident_budget_bytes == 0 {
            return Err(KimiK3RuntimeError::Invalid(
                "resident budget and context limit must be non-zero".to_owned(),
            ));
        }
        let config = KimiK3Config::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        let geometry = runtime_geometry_from_config(&config.text_config)?;
        let requirements = inspect_runtime_requirements(
            &config,
            &index,
            options.context_limit,
            options.expert_slots_per_layer,
        )?;
        for (component, required, maximum) in [
            (
                "resident core plus layer-wise prefill scratch",
                requirements.resident_bytes,
                options.resident_budget_bytes,
            ),
            (
                "compressed MLA cache",
                requirements.mla_cache_bytes,
                options.kv_cache_budget_bytes,
            ),
            (
                "largest routed expert",
                requirements.routed_expert_bytes,
                options.maximum_expert_bytes,
            ),
            (
                "expert cache",
                requirements.expert_cache_bytes,
                options.expert_cache_budget_bytes,
            ),
        ] {
            if required > maximum {
                return Err(KimiK3RuntimeError::Budget {
                    component,
                    required,
                    maximum,
                });
            }
        }

        // Strict schema validation above happens before any payload read.
        let final_norm = load_reference_vector(
            &index,
            &format!("{ROOT_PREFIX}.norm.weight"),
            config.text_config.hidden_size,
        )?;
        let output_norm = load_reference_vector(
            &index,
            &format!("{ROOT_PREFIX}.output_attn_res_norm.weight"),
            config.text_config.hidden_size,
        )?;
        let output_projection = load_reference_values(
            &index,
            &format!("{ROOT_PREFIX}.output_attn_res_proj.weight"),
        )?;
        if output_projection.len() != config.text_config.hidden_size {
            return Err(KimiK3RuntimeError::Invalid(format!(
                "model output AttnRes projection has {} values, expected {}",
                output_projection.len(),
                config.text_config.hidden_size
            )));
        }
        let folded_output_attn_res = fold_attn_res_weights(&output_norm, &output_projection)?;
        let index = Arc::new(index);

        Ok(Self {
            instance_id: NEXT_KIMI_K3_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            config,
            geometry,
            index,
            final_norm,
            folded_output_attn_res,
            requirements,
            context_limit: options.context_limit,
            expert_slots_per_layer: options.expert_slots_per_layer,
            maximum_expert_bytes: options.maximum_expert_bytes,
        })
    }

    pub fn new_state(&self) -> Result<KimiK3RuntimeState, KimiK3RuntimeError> {
        let mut attention = Vec::new();
        attention
            .try_reserve_exact(self.config.text_config.num_hidden_layers)
            .map_err(|error| {
                KimiK3RuntimeError::Invalid(format!("cannot reserve Kimi-K3 layer states: {error}"))
            })?;
        for &kind in &self.geometry.attention_layers {
            attention.push(match kind {
                AttentionLayerKind::Kda => {
                    KimiK3LayerState::Kda(KdaState::new(&self.geometry.kda)?)
                }
                AttentionLayerKind::Mla => {
                    KimiK3LayerState::Mla(MlaCache::new(self.context_limit, &self.geometry.mla)?)
                }
            });
        }
        let text = &self.config.text_config;
        let experts = KimiK3ExpertStore::new_shared(
            Arc::clone(&self.index),
            text.num_hidden_layers,
            text.num_experts,
            text.routed_expert_hidden_size,
            text.moe_intermediate_size,
            self.expert_slots_per_layer,
            self.maximum_expert_bytes,
        )?;
        Ok(KimiK3RuntimeState {
            instance_id: self.instance_id,
            position: 0,
            attention,
            experts,
            routes_by_layer: vec![Vec::new(); text.num_hidden_layers],
            poisoned: false,
        })
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut KimiK3RuntimeState,
    ) -> Result<KimiK3RuntimeStep, KimiK3RuntimeError> {
        self.validate_state(state)?;
        let token_index = self.validate_token(token)?;
        if state.position >= self.context_limit {
            return Err(KimiK3RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        let position = state.position;
        let result = self.forward_token_inner(token_index, position, state);
        let step = poison_on_error(&mut state.poisoned, result)?;
        state.position = position + 1;
        state.routes_by_layer = step.routes_by_layer.clone();
        Ok(step)
    }

    /// Executes a prompt layer by layer, loading each multi-gigabyte trunk layer only once.
    ///
    /// Every token calls the incremental KDA/MLA step and replaces its one retained hidden row in
    /// place. Calling the attention convenience prefill functions here would allocate a second
    /// full `[tokens, hidden]` result and violate the planner contract.
    pub fn prefill_tokens(
        &self,
        tokens: &[u32],
        state: &mut KimiK3RuntimeState,
    ) -> Result<KimiK3RuntimeStep, KimiK3RuntimeError> {
        self.validate_state(state)?;
        if tokens.is_empty() {
            return Err(KimiK3RuntimeError::Invalid(
                "prefill requires at least one token".to_owned(),
            ));
        }
        let end_position = state.position.checked_add(tokens.len()).ok_or_else(|| {
            KimiK3RuntimeError::Invalid("prefill position overflows usize".to_owned())
        })?;
        if end_position > self.context_limit {
            return Err(KimiK3RuntimeError::ContextExhausted {
                position: end_position - 1,
                limit: self.context_limit,
            });
        }
        // Reject unplanned input lengths before allocating in proportion to caller input.
        let mut token_indices = Vec::new();
        token_indices
            .try_reserve_exact(tokens.len())
            .map_err(|error| {
                KimiK3RuntimeError::Invalid(format!(
                    "cannot reserve prefill token indices: {error}"
                ))
            })?;
        for &token in tokens {
            token_indices.push(self.validate_token(token)?);
        }
        let start_position = state.position;
        let result = self.prefill_tokens_inner(&token_indices, start_position, state);
        let step = poison_on_error(&mut state.poisoned, result)?;
        state.position = end_position;
        state.routes_by_layer = step.routes_by_layer.clone();
        Ok(step)
    }

    pub fn config(&self) -> &KimiK3Config {
        &self.config
    }

    pub fn requirements(&self) -> &KimiK3RuntimeRequirements {
        &self.requirements
    }

    fn forward_token_inner(
        &self,
        token: usize,
        position: usize,
        state: &mut KimiK3RuntimeState,
    ) -> Result<KimiK3RuntimeStep, KimiK3RuntimeError> {
        let mut hidden = self.load_hidden(token)?;
        let mut residual = AttnResidualState::new(
            self.config.text_config.hidden_size,
            self.config.text_config.attn_res_block_size,
        )?;
        let mut routes_by_layer = Vec::with_capacity(self.config.text_config.num_hidden_layers);
        for layer_id in 0..self.config.text_config.num_hidden_layers {
            let weights = KimiK3LayerWeights::load(
                &self.config,
                &self.index,
                layer_id,
                self.requirements.streamed_layer_bytes,
            )?;
            let folds = LayerFolds::new(&weights)?;
            let (next, routes) = execute_loaded_layer(
                &self.config,
                &self.geometry,
                &weights,
                &folds,
                position,
                &hidden,
                &mut residual,
                &mut state.attention[layer_id],
                &mut state.experts,
            )?;
            hidden = next;
            routes_by_layer.push(routes);
        }
        self.finish_step(hidden, &residual, routes_by_layer)
    }

    fn prefill_tokens_inner(
        &self,
        tokens: &[usize],
        start_position: usize,
        state: &mut KimiK3RuntimeState,
    ) -> Result<KimiK3RuntimeStep, KimiK3RuntimeError> {
        let mut hidden_by_token = tokens
            .iter()
            .map(|&token| self.load_hidden(token))
            .collect::<Result<Vec<_>, _>>()?;
        let mut residual_by_token = (0..tokens.len())
            .map(|_| {
                AttnResidualState::new(
                    self.config.text_config.hidden_size,
                    self.config.text_config.attn_res_block_size,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_offset = tokens.len() - 1;
        let mut routes_by_layer = Vec::with_capacity(self.config.text_config.num_hidden_layers);

        for layer_id in 0..self.config.text_config.num_hidden_layers {
            let weights = KimiK3LayerWeights::load(
                &self.config,
                &self.index,
                layer_id,
                self.requirements.streamed_layer_bytes,
            )?;
            let folds = LayerFolds::new(&weights)?;
            for offset in 0..tokens.len() {
                let position = start_position.checked_add(offset).ok_or_else(|| {
                    KimiK3RuntimeError::Invalid("prefill token position overflows".to_owned())
                })?;
                let current = std::mem::take(&mut hidden_by_token[offset]);
                let (next, routes) = execute_loaded_layer(
                    &self.config,
                    &self.geometry,
                    &weights,
                    &folds,
                    position,
                    &current,
                    &mut residual_by_token[offset],
                    &mut state.attention[layer_id],
                    &mut state.experts,
                )?;
                hidden_by_token[offset] = next;
                if offset == final_offset {
                    routes_by_layer.push(routes);
                }
            }
        }

        let hidden = hidden_by_token
            .pop()
            .ok_or_else(|| KimiK3RuntimeError::Invalid("prefill lost final hidden".to_owned()))?;
        let residual = residual_by_token.pop().ok_or_else(|| {
            KimiK3RuntimeError::Invalid("prefill lost final AttnRes state".to_owned())
        })?;
        // The LM-head chunk may use most of the fixed transient reserve. Release every non-final
        // prompt row and snapshot before starting that streamed projection.
        drop(hidden_by_token);
        drop(residual_by_token);
        self.finish_step(hidden, &residual, routes_by_layer)
    }

    fn load_hidden(&self, token: usize) -> Result<Vec<f32>, KimiK3RuntimeError> {
        Ok(load_reference_matrix_row(
            &self.index,
            &format!("{ROOT_PREFIX}.embed_tokens.weight"),
            token,
            self.config.text_config.vocab_size,
            self.config.text_config.hidden_size,
        )?)
    }

    fn finish_step(
        &self,
        hidden: Vec<f32>,
        residual: &AttnResidualState,
        routes_by_layer: Vec<Vec<RouteChoice>>,
    ) -> Result<KimiK3RuntimeStep, KimiK3RuntimeError> {
        let hidden = residual.finalize(
            &hidden,
            &self.folded_output_attn_res,
            self.config.text_config.rms_norm_eps as f32,
        )?;
        let hidden = rms_norm(
            &hidden,
            &self.final_norm,
            self.config.text_config.rms_norm_eps as f32,
        )?;
        let logits = streamed_reference_matvec(
            &self.index,
            LM_HEAD,
            self.config.text_config.vocab_size,
            self.config.text_config.hidden_size,
            &hidden,
            LM_HEAD_ROWS_PER_CHUNK,
        )?;
        Ok(KimiK3RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    fn validate_token(&self, token: u32) -> Result<usize, KimiK3RuntimeError> {
        let token_index = usize::try_from(token).unwrap_or(usize::MAX);
        if token_index >= self.config.text_config.vocab_size {
            Err(KimiK3RuntimeError::TokenOutOfRange {
                token,
                vocabulary: self.config.text_config.vocab_size,
            })
        } else {
            Ok(token_index)
        }
    }

    fn validate_state(&self, state: &KimiK3RuntimeState) -> Result<(), KimiK3RuntimeError> {
        if state.instance_id != self.instance_id {
            return Err(KimiK3RuntimeError::StateInstance);
        }
        if state.poisoned {
            return Err(KimiK3RuntimeError::StatePoisoned);
        }
        if state.attention.len() != self.geometry.attention_layers.len() {
            return Err(KimiK3RuntimeError::Invalid(format!(
                "state has {} attention layers, expected {}",
                state.attention.len(),
                self.geometry.attention_layers.len()
            )));
        }
        for (layer, (&kind, layer_state)) in self
            .geometry
            .attention_layers
            .iter()
            .zip(&state.attention)
            .enumerate()
        {
            let position = match (kind, layer_state) {
                (AttentionLayerKind::Kda, KimiK3LayerState::Kda(value)) => value.position(),
                (AttentionLayerKind::Mla, KimiK3LayerState::Mla(value)) => value.len(),
                _ => {
                    return Err(KimiK3RuntimeError::Invalid(format!(
                        "state attention kind does not match configured layer {layer}"
                    )))
                }
            };
            if position != state.position {
                return Err(KimiK3RuntimeError::Invalid(format!(
                    "state layer {layer} is at position {position}, global position is {}",
                    state.position
                )));
            }
        }
        Ok(())
    }
}

impl CausalDecoder for KimiK3RuntimeModel {
    type State = KimiK3RuntimeState;
    type Error = KimiK3RuntimeError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        KimiK3RuntimeModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(KimiK3RuntimeModel::forward_token(self, token, state)?.logits)
    }

    fn prefill(&self, prompt: &[u32], state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(KimiK3RuntimeModel::prefill_tokens(self, prompt, state)?.logits)
    }
}

#[derive(Debug)]
enum KimiK3LayerState {
    Kda(KdaState),
    Mla(MlaCache),
}

#[derive(Debug)]
pub struct KimiK3RuntimeState {
    instance_id: u64,
    position: usize,
    attention: Vec<KimiK3LayerState>,
    experts: KimiK3ExpertStore,
    routes_by_layer: Vec<Vec<RouteChoice>>,
    poisoned: bool,
}

impl KimiK3RuntimeState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn routes_by_layer(&self) -> &[Vec<RouteChoice>] {
        &self.routes_by_layer
    }

    pub fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.experts.telemetry()
    }

    pub fn cached_f32_elements(&self) -> usize {
        self.attention
            .iter()
            .map(|state| match state {
                KimiK3LayerState::Kda(value) => value
                    .recurrent()
                    .len()
                    .saturating_add(value.query_history().len())
                    .saturating_add(value.key_history().len())
                    .saturating_add(value.value_history().len()),
                KimiK3LayerState::Mla(value) => value
                    .normalized_latents()
                    .len()
                    .saturating_add(value.shared_nope_slots().len()),
            })
            .fold(0usize, usize::saturating_add)
    }
}

struct LayerFolds {
    attention: Vec<f32>,
    mlp: Vec<f32>,
}

impl LayerFolds {
    fn new(weights: &KimiK3LayerWeights) -> Result<Self, KimiK3RuntimeError> {
        Ok(Self {
            attention: fold_attn_res_weights(
                &weights.common.self_attention_res_norm,
                &weights.common.self_attention_res_projection,
            )?,
            mlp: fold_attn_res_weights(
                &weights.common.mlp_res_norm,
                &weights.common.mlp_res_projection,
            )?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_loaded_layer(
    config: &KimiK3Config,
    geometry: &KimiK3RuntimeGeometry,
    weights: &KimiK3LayerWeights,
    folds: &LayerFolds,
    position: usize,
    hidden: &[f32],
    residual: &mut AttnResidualState,
    attention_state: &mut KimiK3LayerState,
    experts: &mut KimiK3ExpertStore,
) -> Result<(Vec<f32>, Vec<RouteChoice>), KimiK3RuntimeError> {
    let layer = weights.layer;
    let text = &config.text_config;
    let epsilon = text.rms_norm_eps as f32;
    let (attention_input, attention_continuation) =
        residual.begin_layer(layer, hidden, &folds.attention, epsilon)?;
    // AttnResidualState intentionally handles only aggregation; both decoder norms are explicit.
    let attention_input = rms_norm(&attention_input, &weights.common.input_layernorm, epsilon)?;

    let attention_output = match (&weights.attention, attention_state) {
        (KimiK3AttentionWeights::Kda(parameters), KimiK3LayerState::Kda(state)) => {
            let mut projector = WeightAttentionProjector::new(&weights.attention);
            kda_step(
                &geometry.kda,
                KdaParameters {
                    query_conv: &parameters.query_conv,
                    key_conv: &parameters.key_conv,
                    value_conv: &parameters.value_conv,
                    a_log: &parameters.a_log,
                    dt_bias: &parameters.dt_bias,
                    output_norm: &parameters.output_norm,
                },
                state,
                position,
                &attention_input,
                &mut projector,
            )?
        }
        (KimiK3AttentionWeights::Mla(parameters), KimiK3LayerState::Mla(cache)) => {
            let mut projector = WeightAttentionProjector::new(&weights.attention);
            mla_step(
                &geometry.mla,
                MlaParameters {
                    query_a_norm: &parameters.query_a_norm,
                    kv_a_norm: &parameters.kv_a_norm,
                },
                cache,
                position,
                &attention_input,
                &mut projector,
            )?
        }
        _ => {
            return Err(KimiK3RuntimeError::Invalid(format!(
                "loaded attention weights/state disagree at layer {layer}"
            )))
        }
    };
    let (mlp_input, mlp_continuation) = residual.after_attention(
        attention_continuation,
        &attention_output,
        &folds.mlp,
        epsilon,
    )?;
    let mlp_input = rms_norm(
        &mlp_input,
        &weights.common.post_attention_layernorm,
        epsilon,
    )?;

    let (mlp_output, routes) = match &weights.feed_forward {
        KimiK3FeedForwardWeights::Dense(dense) => {
            let gate = dense.gate.matvec(&mlp_input)?;
            let up = dense.up.matvec(&mlp_input)?;
            let activated = situ_glu(
                &gate,
                &up,
                text.activation_situ_beta as f32,
                text.activation_situ_linear_beta as f32,
            )?;
            (dense.down.matvec(&activated)?, Vec::new())
        }
        KimiK3FeedForwardWeights::Moe(moe) => {
            let mut projector = KimiK3MoeProjector::new(
                layer,
                &moe.router,
                &moe.routed_down,
                &moe.routed_up,
                &moe.shared_gate,
                &moe.shared_up,
                &moe.shared_down,
                experts,
            );
            let output = latent_moe(
                &mlp_input,
                &moe.correction_bias,
                &moe.routed_norm,
                geometry.latent_moe,
                &mut projector,
            )?;
            (output.hidden, output.routes)
        }
    };
    Ok((
        residual.finish_layer(mlp_continuation, &mlp_output)?,
        routes,
    ))
}

struct WeightAttentionProjector<'weights> {
    weights: &'weights KimiK3AttentionWeights,
}

impl<'weights> WeightAttentionProjector<'weights> {
    fn new(weights: &'weights KimiK3AttentionWeights) -> Self {
        Self { weights }
    }

    fn matrix(&self, projection: AttentionProjection) -> Result<&WeightMatrix, String> {
        match (self.weights, projection) {
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaQuery) => {
                Ok(&weights.query)
            }
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaKey) => Ok(&weights.key),
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaValue) => {
                Ok(&weights.value)
            }
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaBeta) => {
                Ok(&weights.beta)
            }
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaDecayA) => {
                Ok(&weights.decay_a)
            }
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaDecayB) => {
                Ok(&weights.decay_b)
            }
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaOutputGate) => {
                Ok(&weights.output_gate)
            }
            (KimiK3AttentionWeights::Kda(weights), AttentionProjection::KdaOutput) => {
                Ok(&weights.output)
            }
            (KimiK3AttentionWeights::Mla(weights), AttentionProjection::MlaQueryA) => {
                Ok(&weights.query_a)
            }
            (KimiK3AttentionWeights::Mla(weights), AttentionProjection::MlaQueryB) => {
                Ok(&weights.query_b)
            }
            (KimiK3AttentionWeights::Mla(weights), AttentionProjection::MlaKvA) => {
                Ok(&weights.kv_a)
            }
            (KimiK3AttentionWeights::Mla(weights), AttentionProjection::MlaKvB) => {
                Ok(&weights.kv_b)
            }
            (KimiK3AttentionWeights::Mla(weights), AttentionProjection::MlaOutputGate) => {
                Ok(&weights.output_gate)
            }
            (KimiK3AttentionWeights::Mla(weights), AttentionProjection::MlaOutput) => {
                Ok(&weights.output)
            }
            _ => Err(format!(
                "projection {projection} does not belong to this attention layer"
            )),
        }
    }
}

impl AttentionProjector for WeightAttentionProjector<'_> {
    type Error = String;

    fn project(
        &mut self,
        projection: AttentionProjection,
        input: &[f32],
        expected_output: usize,
    ) -> Result<Vec<f32>, Self::Error> {
        let matrix = self.matrix(projection)?;
        if matrix.rows() != expected_output {
            return Err(format!(
                "{projection} has {} rows, expected {expected_output}",
                matrix.rows()
            ));
        }
        matrix.matvec(input).map_err(|error| error.to_string())
    }
}

fn inspect_runtime_requirements(
    config: &KimiK3Config,
    index: &TensorIndex,
    context_limit: usize,
    expert_slots_per_layer: usize,
) -> Result<KimiK3RuntimeRequirements, KimiK3RuntimeError> {
    let text = &config.text_config;
    if context_limit == 0 || context_limit > text.max_position_embeddings {
        return Err(KimiK3RuntimeError::Invalid(format!(
            "context limit must be in 1..={}, got {context_limit}",
            text.max_position_embeddings
        )));
    }
    if expert_slots_per_layer > text.num_experts {
        return Err(KimiK3RuntimeError::Invalid(format!(
            "{expert_slots_per_layer} expert slots per layer exceed {} experts",
            text.num_experts
        )));
    }
    let schema = schema::inspect_requirements(config, index)?;
    let geometry = runtime_geometry_from_config(text)?;

    let mut streamed_layer_bytes = 0u64;
    for layer in 0..text.num_hidden_layers {
        streamed_layer_bytes = streamed_layer_bytes.max(
            KimiK3LayerWeights::inspect_resident_bytes(config, index, layer)?,
        );
    }
    let root_resident_bytes = checked_product_u64(
        &[2, usize_u64(text.hidden_size, "hidden_size")?, 4],
        "root resident bytes",
    )?;
    let kda_state_bytes = checked_product_u64(
        &[
            usize_u64(
                geometry.sequence_state.kda_state_f32_per_sequence,
                "KDA state elements",
            )?,
            4,
        ],
        "KDA state bytes",
    )?;
    let attn_res_working_bytes = checked_product_u64(
        &[
            2,
            usize_u64(
                geometry.sequence_state.attn_res_max_sources,
                "AttnRes source count",
            )?,
            usize_u64(text.hidden_size, "hidden_size")?,
            4,
        ],
        "AttnRes working bytes",
    )?;
    let resident_core_bytes = checked_sum_u64(
        &[
            streamed_layer_bytes,
            kda_state_bytes,
            attn_res_working_bytes,
        ],
        "resident core bytes",
    )?;

    let prefill_scratch_bytes_per_token = prefill_scratch_per_token(
        geometry.sequence_state.attn_res_max_sources,
        text.hidden_size,
        text.num_attention_heads,
    )?;
    let prefill_scratch_bytes = checked_sum_u64(
        &[
            FIXED_PREFILL_SCRATCH_BYTES,
            checked_product_u64(
                &[
                    prefill_scratch_bytes_per_token,
                    usize_u64(context_limit, "context_limit")?,
                ],
                "variable prefill scratch bytes",
            )?,
        ],
        "prefill scratch bytes",
    )?;
    let resident_bytes = checked_sum_u64(
        &[resident_core_bytes, prefill_scratch_bytes],
        "runtime resident bytes",
    )?;
    let mla_two_pass_scratch_bytes = usize_u64(
        mla_two_pass_scratch(&geometry.mla, context_limit)?.total_bytes()?,
        "MLA two-pass scratch bytes",
    )?;
    let mla_cache_bytes = checked_product_u64(
        &[
            usize_u64(
                geometry.sequence_state.mla_cache_f32_per_token,
                "MLA cache elements per token",
            )?,
            usize_u64(context_limit, "context_limit")?,
            4,
        ],
        "MLA cache bytes",
    )?;

    let logical_expert_values = checked_product_u64(
        &[
            usize_u64(text.routed_expert_hidden_size, "routed expert hidden size")?,
            usize_u64(text.moe_intermediate_size, "MoE intermediate size")?,
        ],
        "routed expert logical values",
    )?;
    let packed_bytes = checked_product_u64(&[logical_expert_values, 3], "expert packed")? / 2;
    let scale_bytes = checked_product_u64(&[logical_expert_values, 3], "expert scales")? / 32;
    let routed_expert_bytes =
        checked_sum_u64(&[packed_bytes, scale_bytes], "one routed expert bytes")?;
    if routed_expert_bytes != KIMI_K3_EXPERT_RESIDENT_BYTES {
        return Err(KimiK3RuntimeError::Invalid(format!(
            "official expert geometry produced {routed_expert_bytes} bytes, expected {KIMI_K3_EXPERT_RESIDENT_BYTES}"
        )));
    }
    let sparse_layers = text
        .num_hidden_layers
        .checked_sub(text.first_k_dense_replace)
        .ok_or_else(|| {
            KimiK3RuntimeError::Invalid("dense layer count exceeds decoder layers".to_owned())
        })?;
    let expert_cache_bytes = checked_product_u64(
        &[
            routed_expert_bytes,
            usize_u64(expert_slots_per_layer, "expert slots per layer")?,
            usize_u64(sparse_layers, "sparse layers")?,
        ],
        "expert cache bytes",
    )?;

    Ok(KimiK3RuntimeRequirements {
        schema,
        root_resident_bytes,
        streamed_layer_bytes,
        kda_state_bytes,
        attn_res_working_bytes,
        resident_core_bytes,
        prefill_scratch_bytes_per_token,
        prefill_scratch_bytes,
        mla_two_pass_scratch_bytes,
        mla_cache_bytes,
        routed_expert_bytes,
        expert_cache_bytes,
        resident_bytes,
        context_limit,
        expert_slots_per_layer,
    })
}

fn prefill_scratch_per_token(
    attn_res_sources: usize,
    hidden_size: usize,
    attention_heads: usize,
) -> Result<u64, KimiK3RuntimeError> {
    let values = checked_sum_u64(
        &[
            checked_product_u64(
                &[
                    usize_u64(attn_res_sources, "AttnRes source count")?,
                    usize_u64(hidden_size, "hidden_size")?,
                ],
                "prefill hidden and AttnRes values",
            )?,
            usize_u64(attention_heads, "attention heads")?,
        ],
        "prefill values per token",
    )?;
    checked_product_u64(&[values, 4], "prefill scratch bytes per token")
}

fn poison_on_error<T, E>(poisoned: &mut bool, result: Result<T, E>) -> Result<T, E> {
    if result.is_err() {
        *poisoned = true;
    }
    result
}

fn usize_u64(value: usize, field: &'static str) -> Result<u64, KimiK3RuntimeError> {
    u64::try_from(value).map_err(|_| {
        KimiK3RuntimeError::Invalid(format!("{field} does not fit runtime byte accounting"))
    })
}

fn checked_product_u64(
    factors: &[u64],
    expression: &'static str,
) -> Result<u64, KimiK3RuntimeError> {
    factors.iter().try_fold(1u64, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| KimiK3RuntimeError::Invalid(format!("{expression} overflows u64")))
    })
}

fn checked_sum_u64(terms: &[u64], expression: &'static str) -> Result<u64, KimiK3RuntimeError> {
    terms.iter().try_fold(0u64, |sum, &term| {
        sum.checked_add(term)
            .ok_or_else(|| KimiK3RuntimeError::Invalid(format!("{expression} overflows u64")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DenseMatrix;
    use crate::models::kimi_k3::weights::{KimiK3KdaWeights, KimiK3MlaWeights};

    fn tagged_matrix(tag: f32) -> WeightMatrix {
        WeightMatrix::F32(DenseMatrix::new(1, 1, vec![tag]).unwrap())
    }

    #[test]
    fn official_layerwise_prefill_contract_is_exact() {
        assert_eq!(prefill_scratch_per_token(9, 7_168, 96).unwrap(), 258_432);
    }

    #[test]
    fn official_native_expert_accounting_is_exact() {
        let logical = 3_584u64 * 3_072;
        assert_eq!(logical * 3 / 2 + logical * 3 / 32, 17_547_264);
    }

    #[test]
    fn partial_forward_error_poisoning_is_explicit() {
        let mut poisoned = false;
        assert_eq!(poison_on_error(&mut poisoned, Ok::<_, ()>(7)), Ok(7));
        assert!(!poisoned);
        assert_eq!(poison_on_error(&mut poisoned, Err::<(), _>(9)), Err(9));
        assert!(poisoned);
    }

    #[test]
    fn runtime_attention_projection_mapping_is_exhaustive() {
        let kda = KimiK3AttentionWeights::Kda(Box::new(KimiK3KdaWeights {
            query: tagged_matrix(1.0),
            key: tagged_matrix(2.0),
            value: tagged_matrix(3.0),
            output_gate: tagged_matrix(4.0),
            output: tagged_matrix(5.0),
            decay_a: tagged_matrix(6.0),
            decay_b: tagged_matrix(7.0),
            beta: tagged_matrix(8.0),
            query_conv: Vec::new(),
            key_conv: Vec::new(),
            value_conv: Vec::new(),
            a_log: Vec::new(),
            dt_bias: Vec::new(),
            output_norm: Vec::new(),
        }));
        let mut projector = WeightAttentionProjector::new(&kda);
        for (projection, tag) in [
            (AttentionProjection::KdaQuery, 1.0),
            (AttentionProjection::KdaKey, 2.0),
            (AttentionProjection::KdaValue, 3.0),
            (AttentionProjection::KdaOutputGate, 4.0),
            (AttentionProjection::KdaOutput, 5.0),
            (AttentionProjection::KdaDecayA, 6.0),
            (AttentionProjection::KdaDecayB, 7.0),
            (AttentionProjection::KdaBeta, 8.0),
        ] {
            assert_eq!(
                projector.project(projection, &[2.0], 1).unwrap(),
                [2.0 * tag]
            );
        }
        assert!(projector
            .project(AttentionProjection::MlaQueryA, &[2.0], 1)
            .is_err());

        let mla = KimiK3AttentionWeights::Mla(Box::new(KimiK3MlaWeights {
            query_a: tagged_matrix(9.0),
            query_a_norm: Vec::new(),
            query_b: tagged_matrix(10.0),
            kv_a: tagged_matrix(11.0),
            kv_a_norm: Vec::new(),
            kv_b: tagged_matrix(12.0),
            output: tagged_matrix(13.0),
            output_gate: tagged_matrix(14.0),
        }));
        let mut projector = WeightAttentionProjector::new(&mla);
        for (projection, tag) in [
            (AttentionProjection::MlaQueryA, 9.0),
            (AttentionProjection::MlaQueryB, 10.0),
            (AttentionProjection::MlaKvA, 11.0),
            (AttentionProjection::MlaKvB, 12.0),
            (AttentionProjection::MlaOutput, 13.0),
            (AttentionProjection::MlaOutputGate, 14.0),
        ] {
            assert_eq!(
                projector.project(projection, &[2.0], 1).unwrap(),
                [2.0 * tag]
            );
        }
        assert!(projector
            .project(AttentionProjection::KdaQuery, &[2.0], 1)
            .is_err());
    }
}
