//! Layer-streamed scalar correctness runtime for Qwen3.8-2.4T-A95B-FP8.
//!
//! This executes the complete base autoregressive forward. The official `mtp.*` namespace is
//! validated by the release schema but intentionally excluded: upstream `ForCausalLM.forward`
//! also instantiates only the text model and LM head and ignores MTP weights.

use super::attention::{AttentionError, FullAttention, FullAttentionCache, FullAttentionGeometry};
use super::expert::{Qwen38Expert, Qwen38ExpertError};
use super::expert_store::{Qwen38ExpertStore, Qwen38ExpertStoreError};
use super::linear_attention::{DeltaNetError, DeltaNetGeometry, DeltaNetState, GatedDeltaNet};
use super::math::{linear, round_to_bf16, round_to_bf16_in_place, uses_bf16_output};
use super::moe::{route_softmax_top_k, Qwen38MoeError, Qwen38Route};
use super::norm::{zero_centered_rms_norm, NormError};
use super::schema::{self, HfWeightIndex, Qwen38Requirements, SchemaError};
use super::weights::{Qwen38AttentionWeights, Qwen38LayerWeightError, Qwen38LayerWeights};
use super::Qwen38Config;
use crate::config::ConfigError;
use crate::generation::CausalDecoder;
use crate::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use crate::storage::{
    load_reference_matrix_row, load_reference_vector, streamed_reference_matvec, SafetensorError,
    TensorIndex, TensorLoadError,
};
use serde::Serialize;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const EMBEDDING: &str = "model.embed_tokens.weight";
const FINAL_NORM: &str = "model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const LM_HEAD_ROWS_PER_CHUNK: usize = 4_096;
const FIXED_SCRATCH_BYTES: u64 = 512 * 1024 * 1024;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub enum Qwen38RuntimeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Schema(SchemaError),
    Tensor(TensorLoadError),
    Layer(Qwen38LayerWeightError),
    ExpertStore(Qwen38ExpertStoreError),
    Expert(Qwen38ExpertError),
    Moe(Qwen38MoeError),
    Attention(AttentionError),
    DeltaNet(DeltaNetError),
    Norm(NormError),
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

impl fmt::Display for Qwen38RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Schema(error) => error.fmt(formatter),
            Self::Tensor(error) => error.fmt(formatter),
            Self::Layer(error) => error.fmt(formatter),
            Self::ExpertStore(error) => error.fmt(formatter),
            Self::Expert(error) => error.fmt(formatter),
            Self::Moe(error) => error.fmt(formatter),
            Self::Attention(error) => error.fmt(formatter),
            Self::DeltaNet(error) => error.fmt(formatter),
            Self::Norm(error) => error.fmt(formatter),
            Self::Invalid(reason) => write!(formatter, "invalid Qwen3.8 runtime: {reason}"),
            Self::Budget {
                component,
                required,
                maximum,
            } => write!(
                formatter,
                "Qwen3.8 {component} needs {required} bytes, authorized maximum is {maximum}"
            ),
            Self::TokenOutOfRange { token, vocabulary } => write!(
                formatter,
                "Qwen3.8 token ID {token} is outside vocabulary 0..{vocabulary}"
            ),
            Self::ContextExhausted { position, limit } => write!(
                formatter,
                "Qwen3.8 position {position} reaches context limit {limit}"
            ),
            Self::StateInstance => formatter.write_str("Qwen3.8 state belongs to another runtime"),
            Self::StatePoisoned => formatter.write_str(
                "Qwen3.8 state was poisoned by a partial forward failure; create a new state",
            ),
        }
    }
}

impl std::error::Error for Qwen38RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Checkpoint(error) => Some(error),
            Self::Schema(error) => Some(error),
            Self::Tensor(error) => Some(error),
            Self::Layer(error) => Some(error),
            Self::ExpertStore(error) => Some(error),
            Self::Expert(error) => Some(error),
            Self::Moe(error) => Some(error),
            Self::Attention(error) => Some(error),
            Self::DeltaNet(error) => Some(error),
            Self::Norm(error) => Some(error),
            Self::Invalid(_)
            | Self::Budget { .. }
            | Self::TokenOutOfRange { .. }
            | Self::ContextExhausted { .. }
            | Self::StateInstance
            | Self::StatePoisoned => None,
        }
    }
}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for Qwen38RuntimeError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

from_error!(ConfigError, Config);
from_error!(SafetensorError, Checkpoint);
from_error!(SchemaError, Schema);
from_error!(TensorLoadError, Tensor);
from_error!(Qwen38LayerWeightError, Layer);
from_error!(Qwen38ExpertStoreError, ExpertStore);
from_error!(Qwen38ExpertError, Expert);
from_error!(Qwen38MoeError, Moe);
from_error!(AttentionError, Attention);
from_error!(DeltaNetError, DeltaNet);
from_error!(NormError, Norm);

#[derive(Debug, Clone, Serialize)]
pub struct Qwen38RuntimeRequirements {
    pub schema: Qwen38Requirements,
    pub context_limit: usize,
    pub streamed_layer_bytes: u64,
    pub recurrent_state_bytes: u64,
    pub convolution_state_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub expert_bytes: u64,
    pub expert_cache_bytes: u64,
    /// Persistent layer/scratch/expert-cache residency, excluding causal state.
    pub resident_bytes: u64,
    /// Worst-case non-state residency while a cache miss loads before evicting an old expert.
    pub peak_resident_bytes: u64,
    pub expert_slots_per_layer: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen38RuntimeStep {
    pub logits: Vec<f32>,
    pub routes_by_layer: Vec<Vec<Qwen38Route>>,
}

/// Opt-in diagnostic result retaining each post-layer hidden state.
///
/// Normal generation uses [`Qwen38RuntimeStep`] and does not pay this allocation. The trace form
/// exists so release-checkpoint gates can compare every decoder boundary with an independent
/// oracle rather than treating final logits as the only observable.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen38RuntimeTraceStep {
    pub step: Qwen38RuntimeStep,
    pub layer_hidden_states: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct Qwen38RuntimeModel {
    instance_id: u64,
    config: Qwen38Config,
    index: Arc<TensorIndex>,
    final_norm: Vec<f32>,
    requirements: Qwen38RuntimeRequirements,
    context_limit: usize,
    expert_slots_per_layer: usize,
    maximum_expert_bytes: u64,
}

impl Qwen38RuntimeModel {
    pub fn inspect_requirements(
        model_dir: impl AsRef<Path>,
        context_limit: usize,
        expert_slots_per_layer: usize,
    ) -> Result<Qwen38RuntimeRequirements, Qwen38RuntimeError> {
        if context_limit == 0 {
            return Err(Qwen38RuntimeError::Invalid(
                "context limit must be non-zero".to_owned(),
            ));
        }
        let directory = model_dir.as_ref();
        let config = Qwen38Config::load(directory)?;
        let hf_index = HfWeightIndex::load(directory)?;
        let index = TensorIndex::open(directory)?;
        inspect_runtime_requirements(
            &config,
            &hf_index,
            &index,
            context_limit,
            expert_slots_per_layer,
        )
    }

    pub fn load(
        model_dir: impl AsRef<Path>,
        options: RuntimeLoadOptions,
    ) -> Result<Self, Qwen38RuntimeError> {
        if options.context_limit == 0
            || options.resident_budget_bytes == 0
            || options.kv_cache_budget_bytes == 0
            || options.maximum_expert_bytes == 0
        {
            return Err(Qwen38RuntimeError::Invalid(
                "resident/KV/per-expert budgets and context limit must be non-zero".to_owned(),
            ));
        }
        let directory = model_dir.as_ref();
        let config = Qwen38Config::load(directory)?;
        let hf_index = HfWeightIndex::load(directory)?;
        let index = TensorIndex::open(directory)?;
        let requirements = inspect_runtime_requirements(
            &config,
            &hf_index,
            &index,
            options.context_limit,
            options.expert_slots_per_layer,
        )?;
        for (component, required, maximum) in [
            (
                "peak resident layer/scratch/expert working set",
                requirements.peak_resident_bytes,
                options.resident_budget_bytes,
            ),
            (
                "attention/recurrent cache",
                requirements
                    .kv_cache_bytes
                    .saturating_add(requirements.recurrent_state_bytes)
                    .saturating_add(requirements.convolution_state_bytes),
                options.kv_cache_budget_bytes,
            ),
            (
                "one routed expert",
                requirements.expert_bytes,
                options.maximum_expert_bytes,
            ),
            (
                "expert cache",
                requirements.expert_cache_bytes,
                options.expert_cache_budget_bytes,
            ),
        ] {
            if required > maximum {
                return Err(Qwen38RuntimeError::Budget {
                    component,
                    required,
                    maximum,
                });
            }
        }
        let final_norm = load_reference_vector(&index, FINAL_NORM, config.hidden_size)?;
        Ok(Self {
            instance_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            config,
            index: Arc::new(index),
            final_norm,
            requirements,
            context_limit: options.context_limit,
            expert_slots_per_layer: options.expert_slots_per_layer,
            maximum_expert_bytes: options.maximum_expert_bytes,
        })
    }

    pub fn new_state(&self) -> Result<Qwen38RuntimeState, Qwen38RuntimeError> {
        let mut attention = Vec::with_capacity(self.config.num_hidden_layers);
        for layer in 0..self.config.num_hidden_layers {
            attention.push(
                if self.config.is_full_attention_layer(layer) == Some(true) {
                    Qwen38LayerState::Full(FullAttentionCache::with_capacity(
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                        self.context_limit,
                    )?)
                } else {
                    Qwen38LayerState::Linear(DeltaNetState::new(delta_geometry(&self.config))?)
                },
            );
        }
        let experts = Qwen38ExpertStore::new_shared(
            Arc::clone(&self.index),
            self.config.num_hidden_layers,
            self.config.num_experts,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            self.expert_slots_per_layer,
            self.maximum_expert_bytes,
        )?;
        Ok(Qwen38RuntimeState {
            instance_id: self.instance_id,
            position: 0,
            attention,
            experts,
            routes_by_layer: vec![Vec::new(); self.config.num_hidden_layers],
            poisoned: false,
        })
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut Qwen38RuntimeState,
    ) -> Result<Qwen38RuntimeStep, Qwen38RuntimeError> {
        self.validate_state(state)?;
        let token = self.validate_token(token)?;
        if state.position >= self.context_limit {
            return Err(Qwen38RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        let position = state.position;
        let result = self.forward_token_inner(token, position, state, None);
        match result {
            Ok(step) => {
                state.position = position + 1;
                state.routes_by_layer = step.routes_by_layer.clone();
                Ok(step)
            }
            Err(error) => {
                // Layer state commits are transactional, but earlier layers may already have
                // advanced when a later shard fails. Refuse reuse instead of silently diverging.
                state.poisoned = true;
                Err(error)
            }
        }
    }

    /// Executes one token while retaining all post-layer hidden states for correctness audits.
    pub fn forward_token_traced(
        &self,
        token: u32,
        state: &mut Qwen38RuntimeState,
    ) -> Result<Qwen38RuntimeTraceStep, Qwen38RuntimeError> {
        self.validate_state(state)?;
        let token = self.validate_token(token)?;
        if state.position >= self.context_limit {
            return Err(Qwen38RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        let position = state.position;
        let mut layer_hidden_states = Vec::with_capacity(self.config.num_hidden_layers);
        let result =
            self.forward_token_inner(token, position, state, Some(&mut layer_hidden_states));
        match result {
            Ok(step) => {
                state.position = position + 1;
                state.routes_by_layer = step.routes_by_layer.clone();
                Ok(Qwen38RuntimeTraceStep {
                    step,
                    layer_hidden_states,
                })
            }
            Err(error) => {
                state.poisoned = true;
                Err(error)
            }
        }
    }

    /// Prefills a prompt layer by layer, loading each large decoder trunk only once.
    pub fn prefill_tokens(
        &self,
        tokens: &[u32],
        state: &mut Qwen38RuntimeState,
    ) -> Result<Qwen38RuntimeStep, Qwen38RuntimeError> {
        self.validate_state(state)?;
        if tokens.is_empty() {
            return Err(Qwen38RuntimeError::Invalid(
                "prefill requires at least one token".to_owned(),
            ));
        }
        let end_position = state.position.checked_add(tokens.len()).ok_or_else(|| {
            Qwen38RuntimeError::Invalid("prefill position overflows usize".to_owned())
        })?;
        if end_position > self.context_limit {
            return Err(Qwen38RuntimeError::ContextExhausted {
                position: end_position - 1,
                limit: self.context_limit,
            });
        }
        let token_indices = tokens
            .iter()
            .map(|&token| self.validate_token(token))
            .collect::<Result<Vec<_>, _>>()?;
        let start_position = state.position;
        match self.prefill_tokens_inner(&token_indices, start_position, state) {
            Ok(step) => {
                state.position = end_position;
                state.routes_by_layer = step.routes_by_layer.clone();
                Ok(step)
            }
            Err(error) => {
                state.poisoned = true;
                Err(error)
            }
        }
    }

    pub fn config(&self) -> &Qwen38Config {
        &self.config
    }

    pub fn requirements(&self) -> &Qwen38RuntimeRequirements {
        &self.requirements
    }

    fn forward_token_inner(
        &self,
        token: usize,
        position: usize,
        state: &mut Qwen38RuntimeState,
        mut layer_hidden_states: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Qwen38RuntimeStep, Qwen38RuntimeError> {
        let mut hidden = load_reference_matrix_row(
            &self.index,
            EMBEDDING,
            token,
            self.config.vocab_size,
            self.config.hidden_size,
        )?;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        for layer in 0..self.config.num_hidden_layers {
            let weights = Qwen38LayerWeights::load(
                &self.config,
                &self.index,
                layer,
                self.requirements.streamed_layer_bytes,
            )?;
            let loaded = LoadedLayer::new(&self.config, weights)?;
            let (next, routes) = loaded.forward_token(
                layer,
                &hidden,
                position,
                &mut state.attention[layer],
                &mut state.experts,
            )?;
            hidden = next;
            if let Some(trace) = layer_hidden_states.as_mut() {
                trace.push(hidden.clone());
            }
            routes_by_layer.push(routes);
        }
        let logits = self.finish_hidden(&hidden)?;
        Ok(Qwen38RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    fn prefill_tokens_inner(
        &self,
        tokens: &[usize],
        start_position: usize,
        state: &mut Qwen38RuntimeState,
    ) -> Result<Qwen38RuntimeStep, Qwen38RuntimeError> {
        let mut hidden_by_token = tokens
            .iter()
            .map(|&token| {
                load_reference_matrix_row(
                    &self.index,
                    EMBEDDING,
                    token,
                    self.config.vocab_size,
                    self.config.hidden_size,
                )
                .map_err(Qwen38RuntimeError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_offset = tokens.len() - 1;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        for layer in 0..self.config.num_hidden_layers {
            let weights = Qwen38LayerWeights::load(
                &self.config,
                &self.index,
                layer,
                self.requirements.streamed_layer_bytes,
            )?;
            let loaded = LoadedLayer::new(&self.config, weights)?;
            for (offset, hidden) in hidden_by_token.iter_mut().enumerate() {
                let position = start_position.checked_add(offset).ok_or_else(|| {
                    Qwen38RuntimeError::Invalid("prefill token position overflows".to_owned())
                })?;
                let (next, routes) = loaded.forward_token(
                    layer,
                    hidden,
                    position,
                    &mut state.attention[layer],
                    &mut state.experts,
                )?;
                *hidden = next;
                if offset == final_offset {
                    routes_by_layer.push(routes);
                }
            }
        }
        let hidden = hidden_by_token.last().ok_or_else(|| {
            Qwen38RuntimeError::Invalid("prefill lost final hidden state".to_owned())
        })?;
        let logits = self.finish_hidden(hidden)?;
        Ok(Qwen38RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    fn finish_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>, Qwen38RuntimeError> {
        let mut hidden =
            zero_centered_rms_norm(hidden, &self.final_norm, self.config.rms_norm_eps as f32)?;
        round_runtime_bf16(&mut hidden, "final RMSNorm")?;
        let mut logits = streamed_reference_matvec(
            &self.index,
            LM_HEAD,
            self.config.vocab_size,
            self.config.hidden_size,
            &hidden,
            LM_HEAD_ROWS_PER_CHUNK,
        )?;
        round_runtime_bf16(&mut logits, "LM head")?;
        Ok(logits)
    }

    fn validate_token(&self, token: u32) -> Result<usize, Qwen38RuntimeError> {
        let index = usize::try_from(token).unwrap_or(usize::MAX);
        if index >= self.config.vocab_size {
            Err(Qwen38RuntimeError::TokenOutOfRange {
                token,
                vocabulary: self.config.vocab_size,
            })
        } else {
            Ok(index)
        }
    }

    fn validate_state(&self, state: &Qwen38RuntimeState) -> Result<(), Qwen38RuntimeError> {
        if state.instance_id != self.instance_id {
            return Err(Qwen38RuntimeError::StateInstance);
        }
        if state.poisoned {
            return Err(Qwen38RuntimeError::StatePoisoned);
        }
        if state.attention.len() != self.config.num_hidden_layers
            || state
                .attention
                .iter()
                .any(|layer| layer.len() != state.position)
        {
            return Err(Qwen38RuntimeError::Invalid(
                "per-layer cache positions disagree with global position".to_owned(),
            ));
        }
        Ok(())
    }
}

impl CausalDecoder for Qwen38RuntimeModel {
    type State = Qwen38RuntimeState;
    type Error = Qwen38RuntimeError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        Qwen38RuntimeModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(Qwen38RuntimeModel::forward_token(self, token, state)?.logits)
    }

    fn prefill(&self, prompt: &[u32], state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(Qwen38RuntimeModel::prefill_tokens(self, prompt, state)?.logits)
    }
}

#[derive(Debug)]
enum Qwen38LayerState {
    Full(FullAttentionCache),
    Linear(DeltaNetState),
}

impl Qwen38LayerState {
    fn len(&self) -> usize {
        match self {
            Self::Full(value) => value.len(),
            Self::Linear(value) => value.len(),
        }
    }
}

#[derive(Debug)]
pub struct Qwen38RuntimeState {
    instance_id: u64,
    position: usize,
    attention: Vec<Qwen38LayerState>,
    experts: Qwen38ExpertStore,
    routes_by_layer: Vec<Vec<Qwen38Route>>,
    poisoned: bool,
}

impl Qwen38RuntimeState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn routes_by_layer(&self) -> &[Vec<Qwen38Route>] {
        &self.routes_by_layer
    }

    pub fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.experts.telemetry()
    }

    pub fn cached_f32_elements(&self) -> usize {
        self.attention
            .iter()
            .map(|layer| match layer {
                Qwen38LayerState::Full(value) => value.stored_f32_elements(),
                Qwen38LayerState::Linear(value) => value.stored_f32_elements(),
            })
            .sum()
    }
}

struct LoadedLayer {
    input_norm: Vec<f32>,
    attention: LoadedAttention,
    post_attention_norm: Vec<f32>,
    moe: super::weights::Qwen38MoeTrunkWeights,
    norm_eps: f32,
    top_k: usize,
    bf16_activations: bool,
}

enum LoadedAttention {
    Full(FullAttention),
    Linear(GatedDeltaNet),
}

impl LoadedLayer {
    fn new(config: &Qwen38Config, weights: Qwen38LayerWeights) -> Result<Self, Qwen38RuntimeError> {
        let bf16_activations = uses_bf16_output(&weights.moe.router);
        let attention = match weights.attention {
            Qwen38AttentionWeights::Full(weights) => {
                LoadedAttention::Full(FullAttention::new_mixed(
                    full_geometry(config),
                    weights.query,
                    weights.key,
                    weights.value,
                    weights.output,
                    weights.query_norm,
                    weights.key_norm,
                )?)
            }
            Qwen38AttentionWeights::Linear(weights) => {
                LoadedAttention::Linear(GatedDeltaNet::new_mixed(
                    delta_geometry(config),
                    weights.qkv,
                    weights.z,
                    weights.beta,
                    weights.decay,
                    weights.convolution,
                    weights.dt_bias,
                    weights.a_log,
                    weights.output_norm,
                    weights.output,
                )?)
            }
        };
        Ok(Self {
            input_norm: weights.input_norm,
            attention,
            post_attention_norm: weights.post_attention_norm,
            moe: weights.moe,
            norm_eps: config.rms_norm_eps as f32,
            top_k: config.num_experts_per_tok,
            bf16_activations,
        })
    }

    fn forward_token(
        &self,
        layer: usize,
        hidden: &[f32],
        position: usize,
        state: &mut Qwen38LayerState,
        experts: &mut Qwen38ExpertStore,
    ) -> Result<(Vec<f32>, Vec<Qwen38Route>), Qwen38RuntimeError> {
        let mut normalized = zero_centered_rms_norm(hidden, &self.input_norm, self.norm_eps)?;
        if self.bf16_activations {
            round_runtime_bf16(&mut normalized, "layer input RMSNorm")?;
        }
        let mixed = match (&self.attention, state) {
            (LoadedAttention::Full(module), Qwen38LayerState::Full(cache)) => {
                module.forward_token(&normalized, position, cache)?
            }
            (LoadedAttention::Linear(module), Qwen38LayerState::Linear(state)) => {
                module.forward_token(&normalized, state)?
            }
            _ => {
                return Err(Qwen38RuntimeError::Invalid(
                    "attention weights and state variants disagree".to_owned(),
                ))
            }
        };
        let mut output = hidden.to_vec();
        add_residual(&mut output, &mixed, self.bf16_activations)?;
        let mut normalized =
            zero_centered_rms_norm(&output, &self.post_attention_norm, self.norm_eps)?;
        if self.bf16_activations {
            round_runtime_bf16(&mut normalized, "layer post-attention RMSNorm")?;
        }
        let (moe, routes) =
            execute_streamed_moe(layer, &normalized, &self.moe, self.top_k, experts)?;
        add_residual(&mut output, &moe, self.bf16_activations)?;
        Ok((output, routes))
    }
}

fn execute_streamed_moe(
    layer: usize,
    input: &[f32],
    weights: &super::weights::Qwen38MoeTrunkWeights,
    top_k: usize,
    experts: &mut Qwen38ExpertStore,
) -> Result<(Vec<f32>, Vec<Qwen38Route>), Qwen38RuntimeError> {
    let logits = linear(&weights.router, input).map_err(|source| Qwen38MoeError::Weight {
        projection: "router",
        source,
    })?;
    let mut routes = route_softmax_top_k(&logits, top_k)?;
    let bf16_activations = uses_bf16_output(&weights.router);
    if bf16_activations {
        for route in &mut routes {
            route.weight = round_runtime_bf16_scalar(route.weight, "router weight")?;
        }
    }
    let mut output = vec![0.0f32; input.len()];
    let mut ordered = routes.clone();
    ordered.sort_by_key(|route| route.expert);
    for route in ordered {
        let expert = experts.acquire(layer, route.expert)?;
        let values = expert.value.forward(input)?;
        for (accumulator, value) in output.iter_mut().zip(values) {
            let mut contribution = route.weight * value;
            if bf16_activations {
                contribution = round_runtime_bf16_scalar(contribution, "routed expert weighting")?;
            }
            *accumulator += contribution;
            if bf16_activations {
                *accumulator =
                    round_runtime_bf16_scalar(*accumulator, "routed expert accumulation")?;
            }
        }
    }
    let shared = Qwen38Expert::forward_projections(
        &weights.shared_gate,
        &weights.shared_expert_up,
        &weights.shared_expert_down,
        input,
    )?;
    let gate =
        linear(&weights.shared_expert_gate, input).map_err(|source| Qwen38MoeError::Weight {
            projection: "shared_expert_gate",
            source,
        })?;
    if gate.len() != 1 || !gate[0].is_finite() {
        return Err(Qwen38RuntimeError::Invalid(
            "shared expert gate must return one finite value".to_owned(),
        ));
    }
    let mut scale = if gate[0] >= 0.0 {
        1.0 / (1.0 + (-gate[0]).exp())
    } else {
        let exponential = gate[0].exp();
        exponential / (1.0 + exponential)
    };
    if bf16_activations {
        scale = round_runtime_bf16_scalar(scale, "shared expert gate")?;
    }
    for (accumulator, value) in output.iter_mut().zip(shared) {
        let mut contribution = scale * value;
        if bf16_activations {
            contribution = round_runtime_bf16_scalar(contribution, "shared expert weighting")?;
        }
        *accumulator += contribution;
        if bf16_activations {
            *accumulator = round_runtime_bf16_scalar(*accumulator, "MoE output")?;
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Qwen38RuntimeError::Invalid(
            "MoE output contains NaN or infinity".to_owned(),
        ));
    }
    Ok((output, routes))
}

fn add_residual(
    residual: &mut [f32],
    update: &[f32],
    bf16_activations: bool,
) -> Result<(), Qwen38RuntimeError> {
    if residual.len() != update.len() {
        return Err(Qwen38RuntimeError::Invalid(format!(
            "residual has {} values but update has {}",
            residual.len(),
            update.len()
        )));
    }
    for (residual, update) in residual.iter_mut().zip(update) {
        *residual += *update;
    }
    if bf16_activations {
        round_runtime_bf16(residual, "residual addition")?;
    }
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(Qwen38RuntimeError::Invalid(
            "residual addition produced NaN or infinity".to_owned(),
        ));
    }
    Ok(())
}

fn round_runtime_bf16(
    values: &mut [f32],
    operation: &'static str,
) -> Result<(), Qwen38RuntimeError> {
    round_to_bf16_in_place(values).map_err(|error| {
        Qwen38RuntimeError::Invalid(format!("{operation} BF16 cast failed: {error}"))
    })
}

fn round_runtime_bf16_scalar(
    value: f32,
    operation: &'static str,
) -> Result<f32, Qwen38RuntimeError> {
    round_to_bf16(value).map_err(|error| {
        Qwen38RuntimeError::Invalid(format!("{operation} BF16 cast failed: {error}"))
    })
}

fn full_geometry(config: &Qwen38Config) -> FullAttentionGeometry {
    FullAttentionGeometry {
        hidden_size: config.hidden_size,
        num_query_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        rotary_dim: (config.head_dim as f64 * config.partial_rotary_factor) as usize,
        norm_eps: config.rms_norm_eps as f32,
        rope_theta: config.rope_parameters.rope_theta as f32,
    }
}

fn delta_geometry(config: &Qwen38Config) -> DeltaNetGeometry {
    DeltaNetGeometry {
        hidden_size: config.hidden_size,
        num_key_heads: config.linear_num_key_heads,
        num_value_heads: config.linear_num_value_heads,
        key_head_dim: config.linear_key_head_dim,
        value_head_dim: config.linear_value_head_dim,
        conv_kernel_size: config.linear_conv_kernel_dim,
        norm_eps: config.rms_norm_eps as f32,
    }
}

fn inspect_runtime_requirements(
    config: &Qwen38Config,
    hf_index: &HfWeightIndex,
    index: &TensorIndex,
    context_limit: usize,
    expert_slots_per_layer: usize,
) -> Result<Qwen38RuntimeRequirements, Qwen38RuntimeError> {
    if context_limit == 0 || context_limit > config.max_position_embeddings {
        return Err(Qwen38RuntimeError::Invalid(format!(
            "context limit {context_limit} must be in 1..={}",
            config.max_position_embeddings
        )));
    }
    if expert_slots_per_layer > config.num_experts {
        return Err(Qwen38RuntimeError::Invalid(
            "expert slots per layer exceed expert count".to_owned(),
        ));
    }
    let schema = schema::inspect_requirements(config, hf_index, index)?;
    let streamed_layer_bytes = (0..config.num_hidden_layers).try_fold(0u64, |maximum, layer| {
        Ok::<_, Qwen38RuntimeError>(maximum.max(Qwen38LayerWeights::inspect_resident_bytes(
            config, index, layer,
        )?))
    })?;
    let linear_layers = config
        .layer_types
        .iter()
        .filter(|kind| kind.as_str() == "linear_attention")
        .count() as u64;
    let full_layers = config.num_hidden_layers as u64 - linear_layers;
    let recurrent_f32 = checked_product_u64(&[
        linear_layers,
        config.linear_num_value_heads as u64,
        config.linear_key_head_dim as u64,
        config.linear_value_head_dim as u64,
    ])?;
    let conv_channels = config
        .linear_num_key_heads
        .checked_mul(config.linear_key_head_dim)
        .and_then(|key| key.checked_mul(2))
        .and_then(|key| {
            key.checked_add(
                config
                    .linear_num_value_heads
                    .checked_mul(config.linear_value_head_dim)?,
            )
        })
        .ok_or_else(|| Qwen38RuntimeError::Invalid("conv channels overflow".to_owned()))?;
    let convolution_f32 = checked_product_u64(&[
        linear_layers,
        conv_channels as u64,
        config.linear_conv_kernel_dim as u64,
    ])?;
    let kv_f32 = checked_product_u64(&[
        full_layers,
        2,
        config.num_key_value_heads as u64,
        config.head_dim as u64,
        context_limit as u64,
    ])?;
    let recurrent_state_bytes = recurrent_f32.checked_mul(4).ok_or_else(|| {
        Qwen38RuntimeError::Invalid("recurrent state byte count overflows".to_owned())
    })?;
    let convolution_state_bytes = convolution_f32.checked_mul(4).ok_or_else(|| {
        Qwen38RuntimeError::Invalid("convolution state byte count overflows".to_owned())
    })?;
    let kv_cache_bytes = kv_f32
        .checked_mul(4)
        .ok_or_else(|| Qwen38RuntimeError::Invalid("KV cache byte count overflows".to_owned()))?;
    let expert_bytes = super::expert_store::Qwen38LoadedExpert::inspect_resident_bytes(
        index,
        config.hidden_size,
        config.moe_intermediate_size,
    )?;
    let expert_cache_bytes = checked_product_u64(&[
        expert_bytes,
        config.num_hidden_layers as u64,
        expert_slots_per_layer as u64,
    ])?;
    let prefill_hidden_bytes = checked_product_u64(&[
        context_limit as u64,
        config.hidden_size as u64,
        std::mem::size_of::<f32>() as u64,
    ])?;
    let scratch_bytes = FIXED_SCRATCH_BYTES
        .checked_add(prefill_hidden_bytes)
        .ok_or_else(|| Qwen38RuntimeError::Invalid("prefill scratch bytes overflow".to_owned()))?;
    let resident_bytes = streamed_layer_bytes
        .checked_add(scratch_bytes)
        .and_then(|value| value.checked_add(expert_cache_bytes))
        .ok_or_else(|| Qwen38RuntimeError::Invalid("resident byte count overflows".to_owned()))?;
    let peak_resident_bytes = resident_bytes.checked_add(expert_bytes).ok_or_else(|| {
        Qwen38RuntimeError::Invalid("peak resident byte count overflows".to_owned())
    })?;
    Ok(Qwen38RuntimeRequirements {
        schema,
        context_limit,
        streamed_layer_bytes,
        recurrent_state_bytes,
        convolution_state_bytes,
        kv_cache_bytes,
        scratch_bytes,
        expert_bytes,
        expert_cache_bytes,
        resident_bytes,
        peak_resident_bytes,
        expert_slots_per_layer,
    })
}

fn checked_product_u64(factors: &[u64]) -> Result<u64, Qwen38RuntimeError> {
    factors.iter().try_fold(1u64, |product, factor| {
        product.checked_mul(*factor).ok_or_else(|| {
            Qwen38RuntimeError::Invalid(format!("resource product overflows: {factors:?}"))
        })
    })
}
