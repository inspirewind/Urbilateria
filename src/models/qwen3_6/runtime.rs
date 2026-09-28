//! Budgeted resident/streamed BF16 runtime for Qwen3.6-35B-A3B.
//!
//! This executes the complete base autoregressive forward. The official `mtp.*` namespace is
//! validated by the release schema but intentionally excluded: upstream `ForCausalLM.forward`
//! also excludes MTP weights. Vision weights are validated but text requests do not execute them.

use super::expert_store::{Qwen36ExpertStore, Qwen36ExpertStoreError};
use super::schema::{self, HfWeightIndex, Qwen36Requirements, SchemaError};
use super::weights::{Qwen36AttentionWeights, Qwen36LayerWeightError, Qwen36LayerWeights};
use super::Qwen36Config;
use crate::config::ConfigError;
use crate::execution::{install, worker_threads};
use crate::generation::CausalDecoder;
use crate::model::WeightMatrix;
use crate::models::qwen3_8::attention::{
    AttentionError, FullAttention, FullAttentionCache, FullAttentionGeometry,
};
use crate::models::qwen3_8::expert::{
    Qwen38Expert as Qwen36Expert, Qwen38ExpertError as Qwen36ExpertError,
};
use crate::models::qwen3_8::linear_attention::{
    DeltaNetError, DeltaNetGeometry, DeltaNetState, DeltaNetWorkspace, GatedDeltaNet,
};
use crate::models::qwen3_8::math::{
    linear, linear_batch, round_to_bf16, round_to_bf16_in_place, uses_bf16_output,
};
use crate::models::qwen3_8::moe::{
    route_softmax_top_k, Qwen38MoeError as Qwen36MoeError, Qwen38Route as Qwen36Route,
};
use crate::models::qwen3_8::norm::{zero_centered_rms_norm, NormError};
use crate::profiling::{span, ProfileStage};
use crate::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use crate::storage::{
    load_compact_bf16_matrix, load_reference_matrix_row, load_reference_vector,
    streamed_reference_matvec_pipelined, SafetensorError, TensorIndex, TensorLoadError,
};
use rayon::prelude::*;
use serde::Serialize;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM: &str = "model.language_model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const LM_HEAD_ROWS_PER_CHUNK: usize = 4_096;
const FIXED_SCRATCH_BYTES: u64 = 512 * 1024 * 1024;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub enum Qwen36RuntimeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Schema(SchemaError),
    Tensor(TensorLoadError),
    Layer(Qwen36LayerWeightError),
    ExpertStore(Qwen36ExpertStoreError),
    Expert(Qwen36ExpertError),
    Moe(Qwen36MoeError),
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

impl fmt::Display for Qwen36RuntimeError {
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
            Self::Invalid(reason) => write!(formatter, "invalid Qwen3.6 runtime: {reason}"),
            Self::Budget {
                component,
                required,
                maximum,
            } => write!(
                formatter,
                "Qwen3.6 {component} needs {required} bytes, authorized maximum is {maximum}"
            ),
            Self::TokenOutOfRange { token, vocabulary } => write!(
                formatter,
                "Qwen3.6 token ID {token} is outside vocabulary 0..{vocabulary}"
            ),
            Self::ContextExhausted { position, limit } => write!(
                formatter,
                "Qwen3.6 position {position} reaches context limit {limit}"
            ),
            Self::StateInstance => formatter.write_str("Qwen3.6 state belongs to another runtime"),
            Self::StatePoisoned => formatter.write_str(
                "Qwen3.6 state was poisoned by a partial forward failure; create a new state",
            ),
        }
    }
}

impl std::error::Error for Qwen36RuntimeError {
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
        impl From<$source> for Qwen36RuntimeError {
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
from_error!(Qwen36LayerWeightError, Layer);
from_error!(Qwen36ExpertStoreError, ExpertStore);
from_error!(Qwen36ExpertError, Expert);
from_error!(Qwen36MoeError, Moe);
from_error!(AttentionError, Attention);
from_error!(DeltaNetError, DeltaNet);
from_error!(NormError, Norm);

#[derive(Debug, Clone, Serialize)]
pub struct Qwen36RuntimeRequirements {
    pub schema: Qwen36Requirements,
    pub context_limit: usize,
    pub streamed_layer_bytes: u64,
    pub layer_resident_bytes: Vec<u64>,
    pub lm_head_resident_bytes: u64,
    pub recurrent_state_bytes: u64,
    pub convolution_state_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub expert_bytes: u64,
    pub expert_cache_bytes: u64,
    /// Minimum streamed-layer/scratch/expert-cache residency, excluding causal state.
    pub resident_bytes: u64,
    /// Minimum non-state peak including one transient expert. Retained backbone
    /// weights are additional and selected by `backbone_residency` from the load budget.
    pub peak_resident_bytes: u64,
    pub expert_slots_per_layer: usize,
}

impl Qwen36RuntimeRequirements {
    /// Plan one runtime/state plus an optional recurrent session checkpoint. Prioritize
    /// weights touched on every token, then devote remaining RAM to a lazy expert LRU.
    pub fn plan_load_options(
        &self,
        total_ram: u64,
        session_checkpoint: bool,
    ) -> Result<RuntimeLoadOptions, Qwen36RuntimeError> {
        let state = self.kv_cache_bytes + self.recurrent_state_bytes + self.convolution_state_bytes;
        let checkpoint = if session_checkpoint {
            self.recurrent_state_bytes + self.convolution_state_bytes
        } else {
            0
        };
        let minimum = self.peak_resident_bytes - self.expert_cache_bytes;
        let fixed = minimum
            .checked_add(state)
            .and_then(|n| n.checked_add(checkpoint))
            .ok_or_else(|| Qwen36RuntimeError::Invalid("RAM plan overflows".into()))?;
        if total_ram < fixed {
            return Err(Qwen36RuntimeError::Budget {
                component: "runtime, causal state and session checkpoint",
                required: fixed,
                maximum: total_ram,
            });
        }
        let backbone = self.lm_head_resident_bytes + self.layer_resident_bytes.iter().sum::<u64>();
        let extra = total_ram - fixed;
        let cached_backbone = backbone.min(extra);
        let per_slot = self.expert_bytes * self.schema.base_layer_count as u64;
        let slots = ((extra - cached_backbone) / per_slot).min(self.schema.experts_per_layer as u64)
            as usize;
        let expert_cache = per_slot * slots as u64;
        Ok(RuntimeLoadOptions {
            resident_budget_bytes: minimum + cached_backbone + expert_cache,
            expert_cache_budget_bytes: expert_cache,
            kv_cache_budget_bytes: state,
            expert_slots_per_layer: slots,
            maximum_expert_bytes: self.expert_bytes,
            context_limit: self.context_limit,
        })
    }

    /// Returns the prefix of decoder layers retained and whether the LM head fits.
    /// The streamed-layer/miss reserve remains available even with a full cache.
    pub fn backbone_residency(&self, resident_budget: u64) -> (usize, bool) {
        let mut extra = resident_budget.saturating_sub(self.peak_resident_bytes);
        let head = extra >= self.lm_head_resident_bytes;
        if head {
            extra -= self.lm_head_resident_bytes;
        }
        let mut count = 0;
        for &bytes in &self.layer_resident_bytes {
            if extra < bytes {
                break;
            }
            extra -= bytes;
            count += 1;
        }
        (count, head)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36RuntimeStep {
    pub logits: Vec<f32>,
    pub routes_by_layer: Vec<Vec<Qwen36Route>>,
}

/// Opt-in diagnostic result retaining each post-layer hidden state.
///
/// Normal generation uses [`Qwen36RuntimeStep`] and does not pay this allocation. The trace form
/// exists so release-checkpoint gates can compare every decoder boundary with an independent
/// oracle rather than treating final logits as the only observable.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36RuntimeTraceStep {
    pub step: Qwen36RuntimeStep,
    pub layer_hidden_states: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct Qwen36RuntimeModel {
    instance_id: u64,
    config: Qwen36Config,
    index: Arc<TensorIndex>,
    final_norm: Vec<f32>,
    requirements: Qwen36RuntimeRequirements,
    context_limit: usize,
    expert_slots_per_layer: usize,
    maximum_expert_bytes: u64,
    cached_backbone_layers: usize,
    lm_head: Option<WeightMatrix>,
}

impl Qwen36RuntimeModel {
    pub fn inspect_requirements(
        model_dir: impl AsRef<Path>,
        context_limit: usize,
        expert_slots_per_layer: usize,
    ) -> Result<Qwen36RuntimeRequirements, Qwen36RuntimeError> {
        if context_limit == 0 {
            return Err(Qwen36RuntimeError::Invalid(
                "context limit must be non-zero".to_owned(),
            ));
        }
        let directory = model_dir.as_ref();
        let config = Qwen36Config::load(directory)?;
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
    ) -> Result<Self, Qwen36RuntimeError> {
        if options.context_limit == 0
            || options.resident_budget_bytes == 0
            || options.kv_cache_budget_bytes == 0
            || options.maximum_expert_bytes == 0
        {
            return Err(Qwen36RuntimeError::Invalid(
                "resident/KV/per-expert budgets and context limit must be non-zero".to_owned(),
            ));
        }
        let directory = model_dir.as_ref();
        let config = Qwen36Config::load(directory)?;
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
                return Err(Qwen36RuntimeError::Budget {
                    component,
                    required,
                    maximum,
                });
            }
        }
        let final_norm = load_reference_vector(&index, FINAL_NORM, config.hidden_size)?;
        let (cached_backbone_layers, cache_head) =
            requirements.backbone_residency(options.resident_budget_bytes);
        let lm_head = if cache_head {
            Some(
                load_compact_bf16_matrix(
                    &index,
                    LM_HEAD,
                    config.vocab_size,
                    config.hidden_size,
                    requirements.lm_head_resident_bytes,
                )
                .map_err(Qwen36LayerWeightError::from)?,
            )
        } else {
            None
        };
        Ok(Self {
            instance_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            config,
            index: Arc::new(index),
            final_norm,
            requirements,
            context_limit: options.context_limit,
            expert_slots_per_layer: options.expert_slots_per_layer,
            maximum_expert_bytes: options.maximum_expert_bytes,
            cached_backbone_layers,
            lm_head,
        })
    }

    pub fn new_state(&self) -> Result<Qwen36RuntimeState, Qwen36RuntimeError> {
        let mut attention = Vec::with_capacity(self.config.num_hidden_layers);
        for layer in 0..self.config.num_hidden_layers {
            attention.push(
                if self.config.is_full_attention_layer(layer) == Some(true) {
                    Qwen36LayerState::Full(FullAttentionCache::with_capacity(
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                        self.context_limit,
                    )?)
                } else {
                    Qwen36LayerState::Linear(DeltaNetState::new(delta_geometry(&self.config))?)
                },
            );
        }
        let experts = Qwen36ExpertStore::new_shared(
            Arc::clone(&self.index),
            self.config.num_hidden_layers,
            self.config.num_experts,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            self.expert_slots_per_layer,
            self.maximum_expert_bytes,
        )?;
        Ok(Qwen36RuntimeState {
            instance_id: self.instance_id,
            position: 0,
            attention,
            cached_layers: (0..self.config.num_hidden_layers).map(|_| None).collect(),
            experts,
            routes_by_layer: vec![Vec::new(); self.config.num_hidden_layers],
            delta_workspace: DeltaNetWorkspace::default(),
            poisoned: false,
        })
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut Qwen36RuntimeState,
    ) -> Result<Qwen36RuntimeStep, Qwen36RuntimeError> {
        self.validate_state(state)?;
        let token = self.validate_token(token)?;
        if state.position >= self.context_limit {
            return Err(Qwen36RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        let position = state.position;
        let result = with_worker_context(|| self.forward_token_inner(token, position, state, None));
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
        state: &mut Qwen36RuntimeState,
    ) -> Result<Qwen36RuntimeTraceStep, Qwen36RuntimeError> {
        self.validate_state(state)?;
        let token = self.validate_token(token)?;
        if state.position >= self.context_limit {
            return Err(Qwen36RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        let position = state.position;
        let mut layer_hidden_states = Vec::with_capacity(self.config.num_hidden_layers);
        let result = with_worker_context(|| {
            self.forward_token_inner(token, position, state, Some(&mut layer_hidden_states))
        });
        match result {
            Ok(step) => {
                state.position = position + 1;
                state.routes_by_layer = step.routes_by_layer.clone();
                Ok(Qwen36RuntimeTraceStep {
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
        state: &mut Qwen36RuntimeState,
    ) -> Result<Qwen36RuntimeStep, Qwen36RuntimeError> {
        self.prefill_tokens_with_batch_size(tokens, state, self.default_prefill_batch_size())
    }

    /// Maximum prefill chunk covered by the runtime's fixed scratch reservation.
    /// Native live activation buffers stay below 64 MiB before capacity headroom;
    /// KV scores and the DeltaNet transaction also fit the existing 512 MiB reserve.
    pub const MAX_PREFILL_BATCH: usize = 128;

    /// Larger chunks improve expert reuse when every routed expert fits in the cache.
    /// Partial caches retain 32-token grouping so a larger selection cannot turn a
    /// previously grouped chunk into tokenwise execution because of cache pressure.
    pub fn default_prefill_batch_size(&self) -> usize {
        if self.expert_slots_per_layer == self.config.num_experts {
            Self::MAX_PREFILL_BATCH
        } else {
            32
        }
    }

    /// Prefills with a bounded projection batch. A size of one retains tokenwise
    /// attention projections while still loading each decoder layer only once.
    pub fn prefill_tokens_with_batch_size(
        &self,
        tokens: &[u32],
        state: &mut Qwen36RuntimeState,
        batch_size: usize,
    ) -> Result<Qwen36RuntimeStep, Qwen36RuntimeError> {
        self.validate_state(state)?;
        if !(1..=Self::MAX_PREFILL_BATCH).contains(&batch_size) {
            return Err(Qwen36RuntimeError::Invalid(format!(
                "prefill batch size must be in 1..={}",
                Self::MAX_PREFILL_BATCH
            )));
        }
        if tokens.is_empty() {
            return Err(Qwen36RuntimeError::Invalid(
                "prefill requires at least one token".to_owned(),
            ));
        }
        let end_position = state.position.checked_add(tokens.len()).ok_or_else(|| {
            Qwen36RuntimeError::Invalid("prefill position overflows usize".to_owned())
        })?;
        if end_position > self.context_limit {
            return Err(Qwen36RuntimeError::ContextExhausted {
                position: end_position - 1,
                limit: self.context_limit,
            });
        }
        let token_indices = tokens
            .iter()
            .map(|&token| self.validate_token(token))
            .collect::<Result<Vec<_>, _>>()?;
        let start_position = state.position;
        match with_worker_context(|| {
            self.prefill_tokens_inner(&token_indices, start_position, state, batch_size)
        }) {
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

    pub fn config(&self) -> &Qwen36Config {
        &self.config
    }

    pub fn requirements(&self) -> &Qwen36RuntimeRequirements {
        &self.requirements
    }

    fn forward_token_inner(
        &self,
        token: usize,
        position: usize,
        state: &mut Qwen36RuntimeState,
        mut layer_hidden_states: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Qwen36RuntimeStep, Qwen36RuntimeError> {
        let mut hidden = load_reference_matrix_row(
            &self.index,
            EMBEDDING,
            token,
            self.config.vocab_size,
            self.config.hidden_size,
        )?;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        for layer in 0..self.config.num_hidden_layers {
            let loaded = self.cached_or_load_layer(layer, state)?;
            let (next, routes) = self.with_retained_layer_prefetch(layer + 1, state, |state| {
                loaded.forward_token(
                    layer,
                    &hidden,
                    position,
                    &mut state.attention[layer],
                    &mut state.experts,
                )
            })?;
            hidden = next;
            if let Some(trace) = layer_hidden_states.as_mut() {
                trace.push(hidden.clone());
            }
            routes_by_layer.push(routes);
        }
        let logits = self.finish_hidden(&hidden)?;
        Ok(Qwen36RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    fn prefill_tokens_inner(
        &self,
        tokens: &[usize],
        start_position: usize,
        state: &mut Qwen36RuntimeState,
        batch_size: usize,
    ) -> Result<Qwen36RuntimeStep, Qwen36RuntimeError> {
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
                .map_err(Qwen36RuntimeError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        // The full prompt's hidden vectors are budgeted separately. Projection buffers stay
        // within the fixed scratch reserve by processing at most MAX_PREFILL_BATCH tokens.
        for layer in 0..self.config.num_hidden_layers {
            let loaded = self.cached_or_load_layer(layer, state)?;
            let final_routes = self.with_retained_layer_prefetch(layer + 1, state, |state| {
                let mut final_routes = Vec::new();
                for (chunk, hidden) in hidden_by_token.chunks_mut(batch_size).enumerate() {
                    let position = start_position + chunk * batch_size;
                    final_routes = loaded.forward_batch(
                        layer,
                        hidden,
                        position,
                        &mut state.attention[layer],
                        &mut state.experts,
                        &mut state.delta_workspace,
                    )?;
                }
                Ok(final_routes)
            })?;
            routes_by_layer.push(final_routes);
        }
        let hidden = hidden_by_token.last().ok_or_else(|| {
            Qwen36RuntimeError::Invalid("prefill lost final hidden state".to_owned())
        })?;
        let logits = self.finish_hidden(hidden)?;
        Ok(Qwen36RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    fn cached_or_load_layer(
        &self,
        layer: usize,
        state: &mut Qwen36RuntimeState,
    ) -> Result<Arc<LoadedLayer>, Qwen36RuntimeError> {
        if let Some(cached) = &state.cached_layers[layer] {
            return Ok(Arc::clone(cached));
        }
        let loaded = Self::load_layer(
            &self.config,
            &self.index,
            layer,
            self.requirements.streamed_layer_bytes,
        )?;
        if layer < self.cached_backbone_layers {
            state.cached_layers[layer] = Some(Arc::clone(&loaded));
        }
        Ok(loaded)
    }

    fn load_layer(
        config: &Qwen36Config,
        index: &TensorIndex,
        layer: usize,
        maximum_bytes: u64,
    ) -> Result<Arc<LoadedLayer>, Qwen36RuntimeError> {
        let _profile = span(ProfileStage::Qwen36LayerLoad);
        let weights = Qwen36LayerWeights::load(config, index, layer, maximum_bytes)?;
        Ok(Arc::new(LoadedLayer::new(config, weights)?))
    }

    fn with_retained_layer_prefetch<T: Send>(
        &self,
        next: usize,
        state: &mut Qwen36RuntimeState,
        forward: impl FnOnce(&mut Qwen36RuntimeState) -> Result<T, Qwen36RuntimeError> + Send,
    ) -> Result<T, Qwen36RuntimeError> {
        // The next layer must already be charged to the persistent-cache reservation.
        // Cached hits, a single worker, and the streaming tail keep the synchronous path.
        if next >= self.cached_backbone_layers
            || state.cached_layers[next].is_some()
            || worker_threads() == 1
        {
            return forward(state);
        }
        let (output, loaded) = join_layer_load(
            || forward(state),
            || {
                Self::load_layer(
                    &self.config,
                    &self.index,
                    next,
                    self.requirements.streamed_layer_bytes,
                )
            },
        )?;
        state.cached_layers[next] = Some(loaded);
        Ok(output)
    }

    fn finish_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>, Qwen36RuntimeError> {
        let _profile = span(ProfileStage::Qwen36LmHead);
        let mut hidden =
            zero_centered_rms_norm(hidden, &self.final_norm, self.config.rms_norm_eps as f32)?;
        round_runtime_bf16(&mut hidden, "final RMSNorm")?;
        let mut logits = if let Some(head) = &self.lm_head {
            head.matvec(&hidden)
                .map_err(|error| Qwen36RuntimeError::Invalid(error.to_string()))?
        } else {
            streamed_reference_matvec_pipelined(
                Arc::clone(&self.index),
                LM_HEAD,
                self.config.vocab_size,
                self.config.hidden_size,
                &hidden,
                LM_HEAD_ROWS_PER_CHUNK,
            )?
        };
        round_runtime_bf16(&mut logits, "LM head")?;
        Ok(logits)
    }

    fn validate_token(&self, token: u32) -> Result<usize, Qwen36RuntimeError> {
        let index = usize::try_from(token).unwrap_or(usize::MAX);
        if index >= self.config.vocab_size {
            Err(Qwen36RuntimeError::TokenOutOfRange {
                token,
                vocabulary: self.config.vocab_size,
            })
        } else {
            Ok(index)
        }
    }

    fn validate_state(&self, state: &Qwen36RuntimeState) -> Result<(), Qwen36RuntimeError> {
        if state.instance_id != self.instance_id {
            return Err(Qwen36RuntimeError::StateInstance);
        }
        if state.poisoned {
            return Err(Qwen36RuntimeError::StatePoisoned);
        }
        if state.attention.len() != self.config.num_hidden_layers
            || state
                .attention
                .iter()
                .any(|layer| layer.len() != state.position)
        {
            return Err(Qwen36RuntimeError::Invalid(
                "per-layer cache positions disagree with global position".to_owned(),
            ));
        }
        Ok(())
    }
}

impl CausalDecoder for Qwen36RuntimeModel {
    type State = Qwen36RuntimeState;
    type Error = Qwen36RuntimeError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        Qwen36RuntimeModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(Qwen36RuntimeModel::forward_token(self, token, state)?.logits)
    }

    fn prefill(&self, prompt: &[u32], state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(Qwen36RuntimeModel::prefill_tokens(self, prompt, state)?.logits)
    }
}

#[derive(Debug)]
enum Qwen36LayerState {
    Full(FullAttentionCache),
    Linear(DeltaNetState),
}

impl Qwen36LayerState {
    fn len(&self) -> usize {
        match self {
            Self::Full(value) => value.len(),
            Self::Linear(value) => value.len(),
        }
    }
}

#[derive(Debug)]
pub struct Qwen36RuntimeState {
    instance_id: u64,
    position: usize,
    attention: Vec<Qwen36LayerState>,
    cached_layers: Vec<Option<Arc<LoadedLayer>>>,
    experts: Qwen36ExpertStore,
    routes_by_layer: Vec<Vec<Qwen36Route>>,
    // One native 2.125 MiB transaction buffer, shared by all 30 recurrent layers.
    // Covered by FIXED_SCRATCH_BYTES and deliberately excluded from checkpoints.
    delta_workspace: DeltaNetWorkspace,
    poisoned: bool,
}

impl Qwen36RuntimeState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn routes_by_layer(&self) -> &[Vec<Qwen36Route>] {
        &self.routes_by_layer
    }

    pub fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.experts.telemetry()
    }

    /// Logical causal-cache elements, excluding reusable execution scratch.
    pub fn cached_f32_elements(&self) -> usize {
        self.attention
            .iter()
            .map(|layer| match layer {
                Qwen36LayerState::Full(value) => value.stored_f32_elements(),
                Qwen36LayerState::Linear(value) => value.stored_f32_elements(),
            })
            .sum()
    }
}

/// Only recurrent layers need a copy; full-attention KV is rewound in place.
pub struct Qwen36SessionCheckpoint {
    position: usize,
    recurrent: Vec<(usize, DeltaNetState)>,
}

impl crate::runtime::session::SessionState for Qwen36RuntimeState {
    type Checkpoint = Qwen36SessionCheckpoint;

    fn position(&self) -> usize {
        self.position
    }
    fn checkpoint(&self) -> Self::Checkpoint {
        Qwen36SessionCheckpoint {
            position: self.position,
            recurrent: self
                .attention
                .iter()
                .enumerate()
                .filter_map(|(index, layer)| match layer {
                    Qwen36LayerState::Linear(value) => Some((index, value.clone())),
                    _ => None,
                })
                .collect(),
        }
    }
    fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<(), Box<dyn std::error::Error>> {
        for layer in &mut self.attention {
            if let Qwen36LayerState::Full(cache) = layer {
                cache.truncate(checkpoint.position)?;
            }
        }
        for (index, value) in checkpoint.recurrent {
            self.attention[index] = Qwen36LayerState::Linear(value);
        }
        self.position = checkpoint.position;
        self.routes_by_layer.clear();
        self.poisoned = false;
        Ok(())
    }
    fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.experts.telemetry()
    }
}

#[derive(Debug)]
struct LoadedLayer {
    input_norm: Vec<f32>,
    attention: LoadedAttention,
    post_attention_norm: Vec<f32>,
    moe: super::weights::Qwen36MoeTrunkWeights,
    norm_eps: f32,
    top_k: usize,
    bf16_activations: bool,
}

#[derive(Debug)]
enum LoadedAttention {
    Full(FullAttention),
    Linear(GatedDeltaNet),
}

impl LoadedLayer {
    fn new(config: &Qwen36Config, weights: Qwen36LayerWeights) -> Result<Self, Qwen36RuntimeError> {
        let bf16_activations = uses_bf16_output(&weights.moe.router);
        let attention = match weights.attention {
            Qwen36AttentionWeights::Full(weights) => {
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
            Qwen36AttentionWeights::Linear(weights) => {
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

    fn forward_batch(
        &self,
        layer: usize,
        hidden: &mut [Vec<f32>],
        position: usize,
        state: &mut Qwen36LayerState,
        experts: &mut Qwen36ExpertStore,
        delta_workspace: &mut DeltaNetWorkspace,
    ) -> Result<Vec<Qwen36Route>, Qwen36RuntimeError> {
        if hidden.len() == 1 {
            let (output, routes) =
                self.forward_token(layer, &hidden[0], position, state, experts)?;
            hidden[0] = output;
            return Ok(routes);
        }
        let mut normalized = Vec::new();
        for token in hidden.iter() {
            let mut values = zero_centered_rms_norm(token, &self.input_norm, self.norm_eps)?;
            if self.bf16_activations {
                round_runtime_bf16(&mut values, "layer input RMSNorm")?;
            }
            normalized.extend(values);
        }
        let attention_profile = span(ProfileStage::Qwen36Attention);
        let mixed = match (&self.attention, state) {
            (LoadedAttention::Full(module), Qwen36LayerState::Full(cache)) => {
                module.forward_tokens(&normalized, hidden.len(), position, cache)?
            }
            (LoadedAttention::Linear(module), Qwen36LayerState::Linear(state)) => module
                .forward_tokens_with_workspace(&normalized, hidden.len(), state, delta_workspace)?,
            _ => {
                return Err(Qwen36RuntimeError::Invalid(
                    "attention weights and state variants disagree".into(),
                ))
            }
        };
        drop(attention_profile);
        let width = self.input_norm.len();
        let mut moe_inputs = Vec::with_capacity(hidden.len() * width);
        for (output, mixed) in hidden.iter_mut().zip(mixed.chunks_exact(width)) {
            add_residual(output, mixed, self.bf16_activations)?;
            let mut normalized =
                zero_centered_rms_norm(output, &self.post_attention_norm, self.norm_eps)?;
            if self.bf16_activations {
                round_runtime_bf16(&mut normalized, "layer post-attention RMSNorm")?;
            }
            moe_inputs.extend(normalized);
        }
        let (moe, routes) = execute_batched_moe(
            layer,
            &moe_inputs,
            hidden.len(),
            &self.moe,
            self.top_k,
            experts,
        )?;
        for (output, moe) in hidden.iter_mut().zip(moe.chunks_exact(width)) {
            add_residual(output, moe, self.bf16_activations)?;
        }
        Ok(routes)
    }

    fn forward_token(
        &self,
        layer: usize,
        hidden: &[f32],
        position: usize,
        state: &mut Qwen36LayerState,
        experts: &mut Qwen36ExpertStore,
    ) -> Result<(Vec<f32>, Vec<Qwen36Route>), Qwen36RuntimeError> {
        let mut normalized = zero_centered_rms_norm(hidden, &self.input_norm, self.norm_eps)?;
        if self.bf16_activations {
            round_runtime_bf16(&mut normalized, "layer input RMSNorm")?;
        }
        let attention_profile = span(ProfileStage::Qwen36Attention);
        let mixed = match (&self.attention, state) {
            (LoadedAttention::Full(module), Qwen36LayerState::Full(cache)) => {
                module.forward_token(&normalized, position, cache)?
            }
            (LoadedAttention::Linear(module), Qwen36LayerState::Linear(state)) => {
                module.forward_token(&normalized, state)?
            }
            _ => {
                return Err(Qwen36RuntimeError::Invalid(
                    "attention weights and state variants disagree".to_owned(),
                ))
            }
        };
        drop(attention_profile);
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

/// Group independent token inputs only when all selected weights remain cache-charged.
/// Replaying the complete access sequence retains tokenwise hit/miss and LRU behavior.
fn execute_batched_moe(
    layer: usize,
    input: &[f32],
    batch: usize,
    weights: &super::weights::Qwen36MoeTrunkWeights,
    top_k: usize,
    experts: &mut Qwen36ExpertStore,
) -> Result<(Vec<f32>, Vec<Qwen36Route>), Qwen36RuntimeError> {
    let profile = span(ProfileStage::Qwen36Moe);
    let width = weights.router.cols();
    let logits =
        linear_batch(&weights.router, input, batch).map_err(|source| Qwen36MoeError::Weight {
            projection: "router",
            source,
        })?;
    let bf16_activations = uses_bf16_output(&weights.router);
    let mut accesses = Vec::with_capacity(batch * top_k);
    let mut final_routes = Vec::new();
    for (token, logits) in logits.chunks_exact(weights.router.rows()).enumerate() {
        let mut routes = route_softmax_top_k(logits, top_k)?;
        if bf16_activations {
            for route in &mut routes {
                route.weight = round_runtime_bf16_scalar(route.weight, "router weight")?;
            }
        }
        if token + 1 == batch {
            final_routes = routes.clone();
        }
        routes.sort_by_key(|route| route.expert);
        accesses.extend(routes.into_iter().map(|route| (token, route)));
    }
    let ids = accesses
        .iter()
        .map(|(_, route)| route.expert)
        .collect::<Vec<_>>();
    let Some(retained) = experts.acquire_retained(layer, &ids)? else {
        // No cache access has occurred. Keep the bounded one-token/one-transient fallback.
        // Close the routing span before recording the individual fallback MoE spans.
        drop(profile);
        let mut output = Vec::with_capacity(input.len());
        for token in input.chunks_exact(width) {
            let (values, routes) = execute_streamed_moe(layer, token, weights, top_k, experts)?;
            output.extend(values);
            final_routes = routes;
        }
        return Ok((output, final_routes));
    };
    let mut by_expert = std::collections::BTreeMap::new();
    for ((token, route), expert) in accesses.into_iter().zip(retained) {
        by_expert
            .entry(route.expert)
            .or_insert_with(|| (expert, Vec::new()))
            .1
            .push((token, route.weight));
    }
    // BTreeMap order is the same expert-ID order used by every scalar token reduction.
    let groups = by_expert.into_values().collect::<Vec<_>>();
    let context = crate::profiling::capture_context();
    let values = install(|| {
        groups
            .par_iter()
            .map(|(expert, requests)| {
                context.enter(|| {
                    let _compute = span(ProfileStage::Qwen36ExpertCompute);
                    let mut gathered = Vec::with_capacity(requests.len() * width);
                    for &(token, _) in requests {
                        gathered.extend_from_slice(&input[token * width..(token + 1) * width]);
                    }
                    expert.value.forward_batch(&gathered, requests.len())
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut output = vec![0.0f32; input.len()];
    for ((_, requests), values) in groups.iter().zip(values) {
        for (&(token, weight), values) in requests.iter().zip(values.chunks_exact(width)) {
            for (accumulator, &value) in output[token * width..(token + 1) * width]
                .iter_mut()
                .zip(values)
            {
                let mut contribution = weight * value;
                if bf16_activations {
                    contribution =
                        round_runtime_bf16_scalar(contribution, "routed expert weighting")?;
                }
                *accumulator += contribution;
                if bf16_activations {
                    *accumulator =
                        round_runtime_bf16_scalar(*accumulator, "routed expert accumulation")?;
                }
            }
        }
    }
    let shared = Qwen36Expert::forward_projections_batch(
        &weights.shared_gate,
        &weights.shared_expert_up,
        &weights.shared_expert_down,
        input,
        batch,
    )?;
    let gates = linear_batch(&weights.shared_expert_gate, input, batch).map_err(|source| {
        Qwen36MoeError::Weight {
            projection: "shared_expert_gate",
            source,
        }
    })?;
    if gates.len() != batch || gates.iter().any(|gate| !gate.is_finite()) {
        return Err(Qwen36RuntimeError::Invalid(
            "shared expert gate must return one finite value per token".into(),
        ));
    }
    for ((output, shared), gate) in output
        .chunks_exact_mut(width)
        .zip(shared.chunks_exact(width))
        .zip(gates)
    {
        let mut scale = if gate >= 0.0 {
            1.0 / (1.0 + (-gate).exp())
        } else {
            let exponential = gate.exp();
            exponential / (1.0 + exponential)
        };
        if bf16_activations {
            scale = round_runtime_bf16_scalar(scale, "shared expert gate")?;
        }
        for (accumulator, &value) in output.iter_mut().zip(shared) {
            let mut contribution = scale * value;
            if bf16_activations {
                contribution = round_runtime_bf16_scalar(contribution, "shared expert weighting")?;
            }
            *accumulator += contribution;
            if bf16_activations {
                *accumulator = round_runtime_bf16_scalar(*accumulator, "MoE output")?;
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Qwen36RuntimeError::Invalid(
            "MoE output contains NaN or infinity".into(),
        ));
    }
    Ok((output, final_routes))
}

fn execute_streamed_moe(
    layer: usize,
    input: &[f32],
    weights: &super::weights::Qwen36MoeTrunkWeights,
    top_k: usize,
    experts: &mut Qwen36ExpertStore,
) -> Result<(Vec<f32>, Vec<Qwen36Route>), Qwen36RuntimeError> {
    let _profile = span(ProfileStage::Qwen36Moe);
    let logits = linear(&weights.router, input).map_err(|source| Qwen36MoeError::Weight {
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
    let retained = if worker_threads() > 1 {
        experts.acquire_retained(layer, &ordered.iter().map(|r| r.expert).collect::<Vec<_>>())?
    } else {
        None
    };
    let shared_forward = || {
        Qwen36Expert::forward_projections(
            &weights.shared_gate,
            &weights.shared_expert_up,
            &weights.shared_expert_down,
            input,
        )
    };
    let (mut parallel_values, parallel_shared) = if let Some(retained) = retained {
        let context = crate::profiling::capture_context();
        // Shared and routed experts depend on the same input, so their projections can
        // share the worker pool. The shared contribution is still accumulated last.
        let (routed, shared) = install(|| {
            rayon::join(
                || {
                    retained
                        .par_iter()
                        .map(|expert| {
                            context.enter(|| {
                                let _profile = span(ProfileStage::Qwen36ExpertCompute);
                                expert.value.forward(input)
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                },
                || context.enter(shared_forward),
            )
        });
        (Some(routed?), Some(shared?))
    } else {
        (None, None)
    };
    // Parallel work preserves expert-ID accumulation order and every BF16 cast.
    for (index, route) in ordered.into_iter().enumerate() {
        let values = if let Some(outputs) = &mut parallel_values {
            std::mem::take(&mut outputs[index])
        } else {
            let expert = experts.acquire(layer, route.expert)?;
            let _compute = span(ProfileStage::Qwen36ExpertCompute);
            expert.value.forward(input)?
        };
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
    let shared = match parallel_shared {
        Some(values) => values,
        None => shared_forward()?,
    };
    let gate =
        linear(&weights.shared_expert_gate, input).map_err(|source| Qwen36MoeError::Weight {
            projection: "shared_expert_gate",
            source,
        })?;
    if gate.len() != 1 || !gate[0].is_finite() {
        return Err(Qwen36RuntimeError::Invalid(
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
        return Err(Qwen36RuntimeError::Invalid(
            "MoE output contains NaN or infinity".to_owned(),
        ));
    }
    Ok((output, routes))
}

fn add_residual(
    residual: &mut [f32],
    update: &[f32],
    bf16_activations: bool,
) -> Result<(), Qwen36RuntimeError> {
    if residual.len() != update.len() {
        return Err(Qwen36RuntimeError::Invalid(format!(
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
        return Err(Qwen36RuntimeError::Invalid(
            "residual addition produced NaN or infinity".to_owned(),
        ));
    }
    Ok(())
}

fn round_runtime_bf16(
    values: &mut [f32],
    operation: &'static str,
) -> Result<(), Qwen36RuntimeError> {
    round_to_bf16_in_place(values).map_err(|error| {
        Qwen36RuntimeError::Invalid(format!("{operation} BF16 cast failed: {error}"))
    })
}

fn round_runtime_bf16_scalar(
    value: f32,
    operation: &'static str,
) -> Result<f32, Qwen36RuntimeError> {
    round_to_bf16(value).map_err(|error| {
        Qwen36RuntimeError::Invalid(format!("{operation} BF16 cast failed: {error}"))
    })
}

fn full_geometry(config: &Qwen36Config) -> FullAttentionGeometry {
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

fn delta_geometry(config: &Qwen36Config) -> DeltaNetGeometry {
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
    config: &Qwen36Config,
    hf_index: &HfWeightIndex,
    index: &TensorIndex,
    context_limit: usize,
    expert_slots_per_layer: usize,
) -> Result<Qwen36RuntimeRequirements, Qwen36RuntimeError> {
    if context_limit == 0 || context_limit > config.max_position_embeddings {
        return Err(Qwen36RuntimeError::Invalid(format!(
            "context limit {context_limit} must be in 1..={}",
            config.max_position_embeddings
        )));
    }
    if expert_slots_per_layer > config.num_experts {
        return Err(Qwen36RuntimeError::Invalid(
            "expert slots per layer exceed expert count".to_owned(),
        ));
    }
    let schema = schema::inspect_requirements(config, hf_index, index)?;
    let layer_resident_bytes = (0..config.num_hidden_layers)
        .map(|layer| Qwen36LayerWeights::inspect_resident_bytes(config, index, layer))
        .collect::<Result<Vec<_>, _>>()?;
    let streamed_layer_bytes = layer_resident_bytes.iter().copied().max().unwrap_or(0);
    let lm_head_resident_bytes = (config.vocab_size * config.hidden_size * 2) as u64;
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
        .ok_or_else(|| Qwen36RuntimeError::Invalid("conv channels overflow".to_owned()))?;
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
        Qwen36RuntimeError::Invalid("recurrent state byte count overflows".to_owned())
    })?;
    let convolution_state_bytes = convolution_f32.checked_mul(4).ok_or_else(|| {
        Qwen36RuntimeError::Invalid("convolution state byte count overflows".to_owned())
    })?;
    let kv_cache_bytes = kv_f32
        .checked_mul(4)
        .ok_or_else(|| Qwen36RuntimeError::Invalid("KV cache byte count overflows".to_owned()))?;
    let expert_bytes = super::expert_store::Qwen36LoadedExpert::inspect_resident_bytes(
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
        .ok_or_else(|| Qwen36RuntimeError::Invalid("prefill scratch bytes overflow".to_owned()))?;
    let resident_bytes = streamed_layer_bytes
        .checked_add(scratch_bytes)
        .and_then(|value| value.checked_add(expert_cache_bytes))
        .ok_or_else(|| Qwen36RuntimeError::Invalid("resident byte count overflows".to_owned()))?;
    let peak_resident_bytes = resident_bytes.checked_add(expert_bytes).ok_or_else(|| {
        Qwen36RuntimeError::Invalid("peak resident byte count overflows".to_owned())
    })?;
    Ok(Qwen36RuntimeRequirements {
        schema,
        context_limit,
        streamed_layer_bytes,
        layer_resident_bytes,
        lm_head_resident_bytes,
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

/// Enter once per forward/prefill so nested kernels reuse the current worker without
/// repeated caller-to-pool dispatch. Profiling activation is local to each worker.
fn with_worker_context<T: Send>(forward: impl FnOnce() -> T + Send) -> T {
    if worker_threads() == 1 {
        return forward();
    }
    let context = crate::profiling::capture_context();
    install(|| context.enter(forward))
}

/// Both scoped jobs finish before an error can escape. Sharing the compute pool bounds
/// loading/computation concurrency, and the current layer's error takes precedence.
fn join_layer_load<T: Send>(
    forward: impl FnOnce() -> Result<T, Qwen36RuntimeError> + Send,
    load: impl FnOnce() -> Result<Arc<LoadedLayer>, Qwen36RuntimeError> + Send,
) -> Result<(T, Arc<LoadedLayer>), Qwen36RuntimeError> {
    let context = crate::profiling::capture_context();
    let (forward, loaded) =
        install(|| rayon::join(|| context.enter(forward), || context.enter(load)));
    Ok((forward?, loaded?))
}

fn checked_product_u64(factors: &[u64]) -> Result<u64, Qwen36RuntimeError> {
    factors.iter().try_fold(1u64, |product, factor| {
        product.checked_mul(*factor).ok_or_else(|| {
            Qwen36RuntimeError::Invalid(format!("resource product overflows: {factors:?}"))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn native_prefill_buffers_fit_fixed_scratch_reserve() {
        let config = Qwen36Config::from_json_str(include_str!("release_config.json")).unwrap();
        let hidden = config.hidden_size as u64;
        let intermediate = config.moe_intermediate_size as u64;
        let conv = (2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim)
            as u64;
        let widest = conv.max((2 * config.num_attention_heads * config.head_dim) as u64);
        // Overcount simultaneously live layer inputs/outputs, attention projections,
        // and shared-expert intermediates. Routed requests include gathered inputs,
        // both output layouts, and gate/up/activation arrays, even across worker jobs.
        let layer_values = 8 * hidden + 4 * widest;
        let routed_values = config.num_experts_per_tok as u64 * (3 * hidden + 3 * intermediate);
        let live_bytes =
            Qwen36RuntimeModel::MAX_PREFILL_BATCH as u64 * (layer_values + routed_values) * 4;
        assert!(live_bytes <= 64 * 1024 * 1024);
        let scores = config.num_attention_heads as u64 * config.max_position_embeddings as u64 * 4;
        let transaction = (config.linear_num_value_heads as u64
            * config.linear_key_head_dim as u64
            * config.linear_value_head_dim as u64
            + conv * config.linear_conv_kernel_dim as u64)
            * 4;
        // Double activation storage for capacity growth, then allow metadata/small scratch.
        // Weight payloads, prompt hidden snapshots, and committed causal state are budgeted apart.
        let conservative = live_bytes * 2 + scores + transaction + 16 * 1024 * 1024;
        assert!(conservative < FIXED_SCRATCH_BYTES);
    }

    #[test]
    fn forward_error_waits_for_prefetch_and_preserves_error_order() {
        let (release, wait) = mpsc::channel();
        let (started, ready) = mpsc::channel();
        let (done, completed) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started.send(()).unwrap();
            let result = join_layer_load::<()>(
                || Err(Qwen36RuntimeError::Invalid("current layer forward".into())),
                move || {
                    wait.recv().unwrap();
                    Err(Qwen36RuntimeError::Invalid("next layer read".into()))
                },
            );
            done.send(()).unwrap();
            result
        });
        ready.recv().unwrap();
        let premature = completed.recv_timeout(Duration::from_millis(30));
        // Always unblock the loader before asserting, including a failed regression.
        release.send(()).unwrap();
        let result = waiter.join().unwrap();
        assert!(matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)));
        assert!(matches!(result, Err(Qwen36RuntimeError::Invalid(message))
            if message == "current layer forward"));
    }

    #[test]
    fn successful_forward_propagates_prefetch_error() {
        let result = join_layer_load(
            || Ok(42),
            || Err(Qwen36RuntimeError::Invalid("next layer read".into())),
        );
        assert!(matches!(result, Err(Qwen36RuntimeError::Invalid(message))
            if message == "next layer read"));
    }
}
