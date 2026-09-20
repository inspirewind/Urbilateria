//! Scalar, layer-streamed DeepSeek-V4 base-model correctness runtime.
//!
//! This is intentionally not a throughput backend. It exists as the executable specification
//! against which later SIMD/GPU kernels can be checked.

use super::compressor::{
    CompressorCheckpoint, CompressorError, CompressorState, CompressorWeights, PreparedCompressor,
};
use super::math::{
    bounded_swiglu, hyper_connection_head, hyper_connection_post, hyper_connection_pre, linear,
    linear_with_mx_activation, linear_with_mx_activation_batch, paired_rope, prepare_mx_activation,
    round_to_bf16_in_place, route_sqrt_softplus, sparse_attention_with_sink,
    unit_rms_norm_in_place, DeepseekMathError, HyperConnectionMix,
};
use super::schema::{self, DeepseekV4Requirements, SchemaError};
use super::DeepseekV4Config;
use crate::config::ConfigError;
use crate::execution::{install, spawn_io, IoTask};
use crate::generation::CausalDecoder;
use crate::math::{
    normalized_hadamard, rms_norm, simulate_e2m1_activation, simulate_e4m3_activation, MathError,
    MxFp4Matrix, MxFp8Matrix, RouteChoice,
};
use crate::model::{WeightError, WeightMatrix};
use crate::profiling::{capture_context, span, ProfileSpan, ProfileStage};
use crate::runtime::cache::LayerLruCache;
use crate::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use crate::storage::{
    inspect_weight_matrix, load_compact_bf16_matrix, load_i64_matrix_row,
    load_reference_matrix_row, load_reference_vector, load_reference_vectors, load_weight_matrices,
    load_weight_matrix, streamed_reference_matvec_pipelined, DType, ReadBuffer, SafetensorError,
    TensorIndex, TensorLoadError, WeightFormat, WeightLoadError,
};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_DEEPSEEK_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);
const PREFILL_ROUTE_WINDOW: usize = 24;
type PendingDecoderLayer = (
    usize,
    IoTask<Result<Arc<DecoderLayer>, DeepseekRuntimeError>>,
);

#[derive(Debug, Default)]
struct DecoderLayerBuffers(Vec<(ReadBuffer, ReadBuffer)>);

impl DecoderLayerBuffers {
    fn push_weight(&mut self, weight: WeightMatrix) {
        let buffers = match weight {
            WeightMatrix::MxFp8(matrix) => matrix.into_e8m0_buffers(),
            WeightMatrix::MxFp4(matrix) => Some(matrix.into_e2m1_buffers()),
            _ => None,
        };
        if let Some(buffers) = buffers {
            self.0.push(buffers);
        }
    }

    fn take_matching(
        &mut self,
        values_len: usize,
        scales_len: usize,
    ) -> Option<(ReadBuffer, ReadBuffer)> {
        let index = self.0.iter().position(|(values, scales)| {
            values.len() == values_len && scales.len() == scales_len
        })?;
        Some(self.0.swap_remove(index))
    }
}

#[derive(Debug)]
pub enum DeepseekRuntimeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Tensor(TensorLoadError),
    WeightLoad(WeightLoadError),
    Weight(WeightError),
    Math(MathError),
    DeepseekMath(DeepseekMathError),
    Compressor(CompressorError),
    Schema(SchemaError),
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
}

impl fmt::Display for DeepseekRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(f),
            Self::Checkpoint(error) => error.fmt(f),
            Self::Tensor(error) => error.fmt(f),
            Self::WeightLoad(error) => error.fmt(f),
            Self::Weight(error) => error.fmt(f),
            Self::Math(error) => error.fmt(f),
            Self::DeepseekMath(error) => error.fmt(f),
            Self::Compressor(error) => error.fmt(f),
            Self::Schema(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "invalid DeepSeek-V4 runtime: {reason}"),
            Self::Budget {
                component,
                required,
                maximum,
            } => write!(
                f,
                "DeepSeek-V4 {component} needs {required} bytes, authorized maximum is {maximum}"
            ),
            Self::TokenOutOfRange { token, vocabulary } => {
                write!(f, "token ID {token} is outside vocabulary 0..{vocabulary}")
            }
            Self::ContextExhausted { position, limit } => {
                write!(
                    f,
                    "runtime position {position} reaches context limit {limit}"
                )
            }
        }
    }
}

impl std::error::Error for DeepseekRuntimeError {}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for DeepseekRuntimeError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

from_error!(ConfigError, Config);
from_error!(SafetensorError, Checkpoint);
from_error!(TensorLoadError, Tensor);
from_error!(WeightLoadError, WeightLoad);
from_error!(WeightError, Weight);
from_error!(MathError, Math);
from_error!(DeepseekMathError, DeepseekMath);
from_error!(CompressorError, Compressor);
from_error!(SchemaError, Schema);

#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekRuntimeStep {
    pub logits: Vec<f32>,
    pub routes_by_layer: Vec<Vec<RouteChoice>>,
}

#[derive(Debug)]
pub struct DeepseekRuntimeModel {
    instance_id: u64,
    config: DeepseekV4Config,
    index: Arc<TensorIndex>,
    decoder_vectors: Arc<[DecoderLayerVectors]>,
    final_norm: Vec<f32>,
    head_hc: HcHeadWeights,
    context_limit: usize,
    expert_slots_per_layer: usize,
    maximum_expert_bytes: u64,
    cached_decoder_layers: usize,
    decoder_layer_prefetch_depth: usize,
    lm_head: Option<WeightMatrix>,
}

impl DeepseekRuntimeModel {
    pub fn inspect_requirements(
        model_dir: impl AsRef<Path>,
        context_limit: usize,
        expert_slots_per_layer: usize,
    ) -> Result<DeepseekV4Requirements, DeepseekRuntimeError> {
        let config = DeepseekV4Config::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        Ok(schema::inspect_requirements(
            &config,
            &index,
            context_limit,
            expert_slots_per_layer,
        )?)
    }

    pub fn load(
        model_dir: impl AsRef<Path>,
        options: RuntimeLoadOptions,
    ) -> Result<Self, DeepseekRuntimeError> {
        if options.context_limit == 0
            || options.resident_budget_bytes == 0
            || options.kv_cache_budget_bytes == 0
            || options.maximum_expert_bytes == 0
        {
            return Err(DeepseekRuntimeError::Invalid(
                "resident/KV/per-expert budgets and context limit must be non-zero".to_owned(),
            ));
        }
        let config = DeepseekV4Config::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        let requirements = schema::inspect_requirements(
            &config,
            &index,
            options.context_limit,
            options.expert_slots_per_layer,
        )?;
        for (component, required, maximum) in [
            (
                "resident layer pipeline",
                requirements.resident_bytes,
                options.resident_budget_bytes,
            ),
            (
                "KV/state cache",
                requirements.kv_cache_bytes,
                options.kv_cache_budget_bytes,
            ),
            (
                "largest routed expert",
                requirements.maximum_expert_bytes,
                options.maximum_expert_bytes,
            ),
            (
                "expert working set",
                requirements.expert_cache_bytes,
                options.expert_cache_budget_bytes,
            ),
        ] {
            if required > maximum {
                return Err(DeepseekRuntimeError::Budget {
                    component,
                    required,
                    maximum,
                });
            }
        }
        if requirements.unexpected_tensor_count != 0 {
            return Err(DeepseekRuntimeError::Invalid(format!(
                "checkpoint has {} tensors outside the strict release schema",
                requirements.unexpected_tensor_count
            )));
        }
        let index = Arc::new(index);
        let (cached_decoder_layers, decoder_layer_prefetch_depth) =
            requirements.decoder_layer_pipeline_for_resident_budget(options.resident_budget_bytes);
        let decoder_vectors = (0..config.num_hidden_layers)
            .map(|layer| DecoderLayerVectors::load(&index, &config, layer))
            .collect::<Result<Vec<_>, _>>()?
            .into();
        let final_norm = load_reference_vector(&index, "norm.weight", config.hidden_size)?;
        let head_hc = HcHeadWeights::load(&index, &config, "")?;
        let lm_head = requirements
            .caches_lm_head_for_resident_budget(options.resident_budget_bytes)
            .then(|| {
                if index.require("head.weight")?.dtype == DType::Bf16 {
                    load_compact_bf16_matrix(
                        &index,
                        "head.weight",
                        config.vocab_size,
                        config.hidden_size,
                        requirements.lm_head_resident_bytes,
                    )
                } else {
                    load_weight_matrix(
                        &index,
                        "head.weight",
                        config.vocab_size,
                        config.hidden_size,
                        requirements.lm_head_resident_bytes,
                    )
                }
            })
            .transpose()?;
        Ok(Self {
            instance_id: NEXT_DEEPSEEK_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            config,
            index,
            decoder_vectors,
            final_norm,
            head_hc,
            context_limit: options.context_limit,
            expert_slots_per_layer: options.expert_slots_per_layer,
            maximum_expert_bytes: options.maximum_expert_bytes,
            cached_decoder_layers,
            decoder_layer_prefetch_depth,
            lm_head,
        })
    }

    pub fn new_state(&self) -> Result<DeepseekRuntimeState, DeepseekRuntimeError> {
        let mut attention = Vec::with_capacity(self.config.num_hidden_layers);
        for layer in 0..self.config.num_hidden_layers {
            attention.push(AttentionState::new(
                &self.config,
                self.config.base_compress_ratio(layer).unwrap_or(0),
            )?);
        }
        Ok(DeepseekRuntimeState {
            instance_id: self.instance_id,
            position: 0,
            attention,
            experts: DeepExpertStore::new(
                Arc::clone(&self.index),
                &self.config,
                self.expert_slots_per_layer,
                self.maximum_expert_bytes,
            )?,
            cached_layers: (0..self.config.num_hidden_layers).map(|_| None).collect(),
            validated_layers: vec![false; self.config.num_hidden_layers],
        })
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut DeepseekRuntimeState,
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        let mut profile = span(ProfileStage::DeepseekToken);
        if state.instance_id != self.instance_id {
            return Err(DeepseekRuntimeError::Invalid(
                "state belongs to another runtime instance".to_owned(),
            ));
        }
        let token_index = usize::try_from(token).unwrap_or(usize::MAX);
        if token_index >= self.config.vocab_size {
            return Err(DeepseekRuntimeError::TokenOutOfRange {
                token,
                vocabulary: self.config.vocab_size,
            });
        }
        if state.position >= self.context_limit {
            return Err(DeepseekRuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        let position = state.position;
        profile.set_token(position, token_index);
        let checkpoints = {
            let mut profile = span(ProfileStage::DeepseekStateCheckpoint);
            profile.set_token(position, token_index);
            state
                .attention
                .iter_mut()
                .map(AttentionState::checkpoint_next_token)
                .collect::<Vec<_>>()
        };
        let result = self.forward_token_inner(token_index, position, state);
        match result {
            Ok(step) => {
                state.position += 1;
                Ok(step)
            }
            Err(error) => {
                for (attention, checkpoint) in state.attention.iter_mut().zip(checkpoints) {
                    attention.restore(checkpoint);
                }
                Err(error)
            }
        }
    }

    /// Prefills a prompt layer by layer and returns the final token's logits.
    ///
    /// Unlike repeated [`Self::forward_token`] calls, this loads each streamed decoder layer once
    /// for the whole prompt and skips the vocabulary projection for intermediate prompt tokens.
    /// Token order within every attention cache and per-layer expert cache remains unchanged.
    pub fn prefill_tokens(
        &self,
        tokens: &[u32],
        state: &mut DeepseekRuntimeState,
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        if state.instance_id != self.instance_id {
            return Err(DeepseekRuntimeError::Invalid(
                "state belongs to another runtime instance".to_owned(),
            ));
        }
        if tokens.is_empty() {
            return Err(DeepseekRuntimeError::Invalid(
                "prefill requires at least one token".to_owned(),
            ));
        }
        let mut token_indices = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let token_index = usize::try_from(token).unwrap_or(usize::MAX);
            if token_index >= self.config.vocab_size {
                return Err(DeepseekRuntimeError::TokenOutOfRange {
                    token,
                    vocabulary: self.config.vocab_size,
                });
            }
            token_indices.push(token_index);
        }
        let end_position = state.position.checked_add(tokens.len()).ok_or_else(|| {
            DeepseekRuntimeError::Invalid("prefill position overflows".to_owned())
        })?;
        if end_position > self.context_limit {
            return Err(DeepseekRuntimeError::ContextExhausted {
                position: end_position - 1,
                limit: self.context_limit,
            });
        }
        let start_position = state.position;
        let checkpoint = {
            let mut profile = span(ProfileStage::DeepseekStateCheckpoint);
            profile.set_token_position(start_position);
            profile.set_batch_tokens(tokens.len());
            state.attention.clone()
        };
        let result = self.prefill_tokens_inner(&token_indices, start_position, state);
        match result {
            Ok(step) => {
                state.position = end_position;
                Ok(step)
            }
            Err(error) => {
                state.attention = checkpoint;
                Err(error)
            }
        }
    }

    fn forward_token_inner(
        &self,
        token: usize,
        position: usize,
        state: &mut DeepseekRuntimeState,
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        let mut hidden = self.load_hidden(token, position)?;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        let mut layer = self.cached_or_load_decoder_layer(
            state,
            LayerTraceContext {
                layer: 0,
                position,
                token,
                batch_tokens: 1,
            },
        )?;
        let mut pending_layers = VecDeque::new();
        let mut reusable_layer = None;
        for layer_id in 0..self.config.num_hidden_layers {
            let flow_id = layer_flow_id(position, layer_id);
            let next_layer_id = layer_id + 1;
            self.fill_decoder_layer_prefetch(
                state,
                layer_id,
                LayerTraceContext {
                    layer: layer_id,
                    position,
                    token,
                    batch_tokens: 1,
                },
                &mut pending_layers,
                &mut reusable_layer,
            );
            let forward = layer.forward(
                &hidden,
                token,
                position,
                &mut state.attention[layer_id],
                &mut state.experts,
                &self.config,
                flow_id,
            );
            let next_layer =
                match self.take_prefetched_decoder_layer(state, next_layer_id, &mut pending_layers)
                {
                    Ok(layer) => layer,
                    Err(error) => {
                        self.drain_decoder_layer_prefetch(&mut pending_layers);
                        return Err(error);
                    }
                };
            // Always drain every queued loader before propagating a compute error. A failed token
            // must not leave background reads or additional layer allocations alive during retry.
            let (next, routes) = match forward {
                Ok(result) => result,
                Err(error) => {
                    self.drain_decoder_layer_prefetch(&mut pending_layers);
                    return Err(error);
                }
            };
            hidden = next;
            routes_by_layer.push(routes);
            if let Some(next_layer) = next_layer {
                let retired = std::mem::replace(&mut layer, next_layer);
                if reusable_layer.is_none() {
                    reusable_layer = Arc::try_unwrap(retired)
                        .ok()
                        .map(DecoderLayer::into_native_buffers);
                }
            }
        }
        debug_assert!(pending_layers.is_empty());
        self.finish_step(hidden, routes_by_layer, position, token)
    }

    fn prefill_tokens_inner(
        &self,
        tokens: &[usize],
        start_position: usize,
        state: &mut DeepseekRuntimeState,
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        let mut hidden_by_token = tokens
            .iter()
            .enumerate()
            .map(|(offset, &token)| self.load_hidden(token, start_position + offset))
            .collect::<Result<Vec<_>, _>>()?;
        let final_offset = tokens.len() - 1;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        let mut layer = self.cached_or_load_decoder_layer(
            state,
            LayerTraceContext {
                layer: 0,
                position: start_position,
                token: tokens[0],
                batch_tokens: tokens.len(),
            },
        )?;
        let mut pending_layers = VecDeque::new();
        let mut reusable_layer = None;
        for layer_id in 0..self.config.num_hidden_layers {
            let steady_expert_capacity = state.experts.begin_prefill_layer(layer_id);
            let flow_id = layer_flow_id(start_position, layer_id);
            let next_layer_id = layer_id + 1;
            self.fill_decoder_layer_prefetch(
                state,
                layer_id,
                LayerTraceContext {
                    layer: layer_id,
                    position: start_position,
                    token: tokens[0],
                    batch_tokens: tokens.len(),
                },
                &mut pending_layers,
                &mut reusable_layer,
            );
            let attention = &mut state.attention[layer_id];
            let forward = (|| {
                let inputs = std::mem::take(&mut hidden_by_token);
                let prepared = layer.prepare_feed_forward_batch(
                    inputs,
                    tokens,
                    start_position,
                    attention,
                    &self.config,
                    flow_id,
                )?;
                let mut outputs = Vec::with_capacity(prepared.len());
                let mut final_routes = Vec::new();
                let mut prepared = prepared.into_iter();
                loop {
                    let batch = prepared
                        .by_ref()
                        .take(PREFILL_ROUTE_WINDOW)
                        .collect::<Vec<_>>();
                    if batch.is_empty() {
                        break;
                    }
                    for (next, routes) in layer.finish_feed_forward_batch(
                        batch,
                        &mut state.experts,
                        &self.config,
                        flow_id,
                    )? {
                        outputs.push(next);
                        final_routes = routes;
                    }
                }
                hidden_by_token = outputs;
                Ok::<_, DeepseekRuntimeError>(final_routes)
            })();
            let next_layer =
                self.take_prefetched_decoder_layer(state, next_layer_id, &mut pending_layers);
            state
                .experts
                .finish_prefill_layer(layer_id, steady_expert_capacity);
            let next_layer = match next_layer {
                Ok(layer) => layer,
                Err(error) => {
                    self.drain_decoder_layer_prefetch(&mut pending_layers);
                    return Err(error);
                }
            };
            let final_routes = match forward {
                Ok(routes) => routes,
                Err(error) => {
                    self.drain_decoder_layer_prefetch(&mut pending_layers);
                    return Err(error);
                }
            };
            routes_by_layer.push(final_routes);
            if let Some(next_layer) = next_layer {
                let retired = std::mem::replace(&mut layer, next_layer);
                if reusable_layer.is_none() {
                    reusable_layer = Arc::try_unwrap(retired)
                        .ok()
                        .map(DecoderLayer::into_native_buffers);
                }
            }
        }
        debug_assert!(pending_layers.is_empty());
        let hidden = hidden_by_token
            .pop()
            .expect("a non-empty prefill has a final hidden state");
        self.finish_step(
            hidden,
            routes_by_layer,
            start_position + final_offset,
            tokens[final_offset],
        )
    }

    fn fill_decoder_layer_prefetch(
        &self,
        state: &DeepseekRuntimeState,
        current_layer: usize,
        trace: LayerTraceContext,
        pending: &mut VecDeque<PendingDecoderLayer>,
        reusable_layer: &mut Option<DecoderLayerBuffers>,
    ) {
        let end = current_layer
            .saturating_add(self.decoder_layer_prefetch_depth)
            .min(self.config.num_hidden_layers.saturating_sub(1));
        for layer in current_layer.saturating_add(1)..=end {
            if state.cached_layers[layer].is_some()
                || pending
                    .iter()
                    .any(|(pending_layer, _)| *pending_layer == layer)
            {
                continue;
            }
            if let Some(task) = self.prefetch_decoder_layer(
                LayerTraceContext { layer, ..trace },
                reusable_layer.take(),
                state.validated_layers[layer],
            ) {
                pending.push_back((layer, task));
            }
        }
    }

    fn take_prefetched_decoder_layer(
        &self,
        state: &mut DeepseekRuntimeState,
        layer: usize,
        pending: &mut VecDeque<PendingDecoderLayer>,
    ) -> Result<Option<Arc<DecoderLayer>>, DeepseekRuntimeError> {
        if layer >= self.config.num_hidden_layers {
            return Ok(None);
        }
        if let Some(cached) = &state.cached_layers[layer] {
            return Ok(Some(Arc::clone(cached)));
        }
        let position = pending
            .iter()
            .position(|(pending_layer, _)| *pending_layer == layer)
            .ok_or_else(|| {
                DeepseekRuntimeError::Invalid(format!(
                    "decoder layer {layer} was neither cached nor queued"
                ))
            })?;
        debug_assert_eq!(position, 0);
        let (_, task) = pending
            .remove(position)
            .expect("a located decoder-layer task remains queued");
        let loaded = task.join()?;
        state.validated_layers[layer] = true;
        self.retain_decoder_layer(state, layer, &loaded);
        Ok(Some(loaded))
    }

    fn drain_decoder_layer_prefetch(&self, pending: &mut VecDeque<PendingDecoderLayer>) {
        while let Some((_, task)) = pending.pop_front() {
            let _ = task.join();
        }
    }

    fn prefetch_decoder_layer(
        &self,
        trace: LayerTraceContext,
        reusable_layer: Option<DecoderLayerBuffers>,
        prevalidated: bool,
    ) -> Option<IoTask<Result<Arc<DecoderLayer>, DeepseekRuntimeError>>> {
        if trace.layer >= self.config.num_hidden_layers {
            return None;
        }
        let index = Arc::clone(&self.index);
        let decoder_vectors = Arc::clone(&self.decoder_vectors);
        let config = self.config.clone();
        let profile_context = capture_context();
        Some(spawn_io(move || {
            profile_context.enter(|| {
                load_decoder_layer_reusing(
                    &index,
                    &config,
                    trace,
                    &decoder_vectors[trace.layer],
                    reusable_layer,
                    prevalidated,
                )
                .map(Arc::new)
            })
        }))
    }

    fn cached_or_load_decoder_layer(
        &self,
        state: &mut DeepseekRuntimeState,
        trace: LayerTraceContext,
    ) -> Result<Arc<DecoderLayer>, DeepseekRuntimeError> {
        if let Some(layer) = &state.cached_layers[trace.layer] {
            return Ok(Arc::clone(layer));
        }
        let loaded = Arc::new(load_decoder_layer_reusing(
            &self.index,
            &self.config,
            trace,
            &self.decoder_vectors[trace.layer],
            None,
            state.validated_layers[trace.layer],
        )?);
        state.validated_layers[trace.layer] = true;
        self.retain_decoder_layer(state, trace.layer, &loaded);
        Ok(loaded)
    }

    fn retain_decoder_layer(
        &self,
        state: &mut DeepseekRuntimeState,
        layer: usize,
        loaded: &Arc<DecoderLayer>,
    ) {
        if layer < self.cached_decoder_layers && state.cached_layers[layer].is_none() {
            state.cached_layers[layer] = Some(Arc::clone(loaded));
        }
    }

    fn load_hidden(&self, token: usize, position: usize) -> Result<Vec<f32>, DeepseekRuntimeError> {
        let embedding = {
            let mut profile = span(ProfileStage::DeepseekEmbeddingRead);
            profile.set_token(position, token);
            load_reference_matrix_row(
                &self.index,
                "embed.weight",
                token,
                self.config.vocab_size,
                self.config.hidden_size,
            )?
        };
        let mut hidden = Vec::with_capacity(self.config.hc_mult * self.config.hidden_size);
        for _ in 0..self.config.hc_mult {
            hidden.extend_from_slice(&embedding);
        }
        Ok(hidden)
    }

    fn finish_step(
        &self,
        hidden: Vec<f32>,
        routes_by_layer: Vec<Vec<RouteChoice>>,
        position: usize,
        token: usize,
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        let hidden = {
            let mut profile = span(ProfileStage::DeepseekFinalization);
            profile.set_token(position, token);
            let hidden = hyper_connection_head(
                &hidden,
                self.config.hidden_size,
                self.config.hc_mult,
                &self.head_hc.function,
                self.head_hc.scale,
                &self.head_hc.base,
                self.config.rms_norm_eps as f32,
                self.config.hc_eps as f32,
            )?;
            bf16_rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps as f32)?
        };
        let logits = {
            let mut profile = span(ProfileStage::DeepseekLmHead);
            profile.set_token(position, token);
            if let Some(head) = &self.lm_head {
                linear(head, &hidden)?
            } else {
                streamed_reference_matvec_pipelined(
                    Arc::clone(&self.index),
                    "head.weight",
                    self.config.vocab_size,
                    self.config.hidden_size,
                    &hidden,
                    4096,
                )?
            }
        };
        Ok(DeepseekRuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    pub fn config(&self) -> &DeepseekV4Config {
        &self.config
    }
}

impl CausalDecoder for DeepseekRuntimeModel {
    type State = DeepseekRuntimeState;
    type Error = DeepseekRuntimeError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        DeepseekRuntimeModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(DeepseekRuntimeModel::forward_token(self, token, state)?.logits)
    }

    fn prefill(&self, prompt: &[u32], state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        let _profile = span(ProfileStage::Prefill);
        Ok(DeepseekRuntimeModel::prefill_tokens(self, prompt, state)?.logits)
    }
}

#[derive(Debug)]
pub struct DeepseekRuntimeState {
    instance_id: u64,
    position: usize,
    attention: Vec<AttentionState>,
    experts: DeepExpertStore,
    cached_layers: Vec<Option<Arc<DecoderLayer>>>,
    /// Once a complete layer load succeeds, subsequent reads of the same immutable checkpoint
    /// ranges need shape checks but not another linear native-payload validation scan.
    validated_layers: Vec<bool>,
}

impl DeepseekRuntimeState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn expert_telemetry(&self) -> &ExpertTelemetry {
        &self.experts.telemetry
    }

    pub fn cached_f32_elements(&self) -> usize {
        self.attention
            .iter()
            .map(AttentionState::stored_f32_elements)
            .sum()
    }
}

#[derive(Debug)]
struct HcVectorWeights {
    base: Arc<[f32]>,
    scale: Arc<[f32]>,
}

#[derive(Debug)]
struct AttentionVectorWeights {
    sink: Arc<[f32]>,
    q_norm: Arc<[f32]>,
    kv_norm: Arc<[f32]>,
    compressor_norm: Option<Arc<[f32]>>,
    indexer_compressor_norm: Option<Arc<[f32]>>,
}

#[derive(Debug)]
struct DecoderLayerVectors {
    attn_hc: HcVectorWeights,
    ffn_hc: HcVectorWeights,
    attn_norm: Arc<[f32]>,
    ffn_norm: Arc<[f32]>,
    attention: AttentionVectorWeights,
    correction_bias: Option<Arc<[f32]>>,
}

impl DecoderLayerVectors {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}");
        let attn_prefix = format!("{prefix}.attn");
        let ffn_prefix = format!("{prefix}.ffn");
        let hc_count = (2 + config.hc_mult) * config.hc_mult;
        let ratio = config.base_compress_ratio(layer).unwrap_or(0);
        let mut specifications = vec![
            (format!("{prefix}.hc_attn_base"), hc_count),
            (format!("{prefix}.hc_attn_scale"), 3),
            (format!("{prefix}.hc_ffn_base"), hc_count),
            (format!("{prefix}.hc_ffn_scale"), 3),
            (format!("{prefix}.attn_norm.weight"), config.hidden_size),
            (format!("{prefix}.ffn_norm.weight"), config.hidden_size),
            (
                format!("{attn_prefix}.attn_sink"),
                config.num_attention_heads,
            ),
            (format!("{attn_prefix}.q_norm.weight"), config.q_lora_rank),
            (format!("{attn_prefix}.kv_norm.weight"), config.head_dim),
        ];
        if ratio > 0 {
            specifications.push((
                format!("{attn_prefix}.compressor.norm.weight"),
                config.head_dim,
            ));
        }
        if ratio == 4 {
            specifications.push((
                format!("{attn_prefix}.indexer.compressor.norm.weight"),
                config.index_head_dim,
            ));
        }
        if layer >= config.num_hash_layers {
            specifications.push((format!("{ffn_prefix}.gate.bias"), config.n_routed_experts));
        }
        let references = specifications
            .iter()
            .map(|(name, length)| (name.as_str(), *length))
            .collect::<Vec<_>>();
        let mut values = load_reference_vectors(index, &references)?.into_iter();
        let mut next = || {
            Arc::<[f32]>::from(
                values
                    .next()
                    .expect("each decoder vector specification has one payload"),
            )
        };
        let attn_hc = HcVectorWeights {
            base: next(),
            scale: next(),
        };
        let ffn_hc = HcVectorWeights {
            base: next(),
            scale: next(),
        };
        let attn_norm = next();
        let ffn_norm = next();
        let sink = next();
        let q_norm = next();
        let kv_norm = next();
        let compressor_norm = (ratio > 0).then(&mut next);
        let indexer_compressor_norm = (ratio == 4).then(&mut next);
        let correction_bias = (layer >= config.num_hash_layers).then(&mut next);
        debug_assert!(values.next().is_none());
        Ok(Self {
            attn_hc,
            ffn_hc,
            attn_norm,
            ffn_norm,
            attention: AttentionVectorWeights {
                sink,
                q_norm,
                kv_norm,
                compressor_norm,
                indexer_compressor_norm,
            },
            correction_bias,
        })
    }
}

#[derive(Debug, Clone)]
struct HcWeights {
    function: WeightMatrix,
    base: Arc<[f32]>,
    scale: Arc<[f32]>,
}

impl HcWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        prefix: &str,
        sublayer: &str,
        vectors: &HcVectorWeights,
    ) -> Result<Self, DeepseekRuntimeError> {
        let count = (2 + config.hc_mult) * config.hc_mult;
        Ok(Self {
            function: matrix(
                index,
                &format!("{prefix}.hc_{sublayer}_fn"),
                count,
                config.hc_mult * config.hidden_size,
            )?,
            base: Arc::clone(&vectors.base),
            scale: Arc::clone(&vectors.scale),
        })
    }

    fn load_reusing(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        prefix: &str,
        sublayer: &str,
        vectors: &HcVectorWeights,
        reuse: &mut DecoderLayerBuffers,
        prevalidated: bool,
    ) -> Result<Self, DeepseekRuntimeError> {
        let count = (2 + config.hc_mult) * config.hc_mult;
        Ok(Self {
            function: matrix_reusing(
                index,
                &format!("{prefix}.hc_{sublayer}_fn"),
                count,
                config.hc_mult * config.hidden_size,
                reuse,
                prevalidated,
            )?,
            base: Arc::clone(&vectors.base),
            scale: Arc::clone(&vectors.scale),
        })
    }

    fn pre(
        &self,
        hidden: &[f32],
        config: &DeepseekV4Config,
    ) -> Result<(Vec<f32>, HyperConnectionMix), DeepseekRuntimeError> {
        Ok(hyper_connection_pre(
            hidden,
            config.hidden_size,
            config.hc_mult,
            &self.function,
            &self.scale,
            &self.base,
            config.hc_sinkhorn_iters,
            config.rms_norm_eps as f32,
            config.hc_eps as f32,
        )?)
    }
}

#[derive(Debug)]
struct HcHeadWeights {
    function: WeightMatrix,
    base: Vec<f32>,
    scale: f32,
}

impl HcHeadWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        prefix: &str,
    ) -> Result<Self, DeepseekRuntimeError> {
        let separator = if prefix.is_empty() { "" } else { "." };
        let stem = format!("{prefix}{separator}hc_head");
        Ok(Self {
            function: matrix(
                index,
                &format!("{stem}_fn"),
                config.hc_mult,
                config.hc_mult * config.hidden_size,
            )?,
            base: load_reference_vector(index, &format!("{stem}_base"), config.hc_mult)?,
            scale: load_reference_vector(index, &format!("{stem}_scale"), 1)?[0],
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LayerTraceContext {
    layer: usize,
    position: usize,
    token: usize,
    batch_tokens: usize,
}

fn load_decoder_layer_reusing(
    index: &TensorIndex,
    config: &DeepseekV4Config,
    trace: LayerTraceContext,
    vectors: &DecoderLayerVectors,
    reusable_layer: Option<DecoderLayerBuffers>,
    prevalidated: bool,
) -> Result<DecoderLayer, DeepseekRuntimeError> {
    let mut profile = span(ProfileStage::DeepseekLayerLoad);
    if trace.batch_tokens == 1 {
        profile.set_token(trace.position, trace.token);
    } else {
        profile.set_token_position(trace.position);
    }
    profile.set_layer_id(trace.layer);
    profile.set_flow_id(layer_flow_id(trace.position, trace.layer));
    profile.set_batch_tokens(trace.batch_tokens);
    match reusable_layer {
        Some(mut buffers) => DecoderLayer::load_reusing(
            index,
            config,
            trace.layer,
            vectors,
            &mut buffers,
            prevalidated,
        ),
        None if prevalidated => DecoderLayer::load_reusing(
            index,
            config,
            trace.layer,
            vectors,
            &mut DecoderLayerBuffers::default(),
            true,
        ),
        None => DecoderLayer::load(index, config, trace.layer, vectors),
    }
}

#[derive(Debug)]
struct DecoderLayer {
    layer: usize,
    attn_hc: HcWeights,
    ffn_hc: HcWeights,
    attn_norm: Arc<[f32]>,
    ffn_norm: Arc<[f32]>,
    attention: AttentionWeights,
    moe: MoeWeights,
}

struct PreparedDeepFeedForward {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    mix: HyperConnectionMix,
    position: usize,
    token: usize,
    _layer_profile: ProfileSpan,
}

type DeepHiddenWithRoutes = (Vec<f32>, Vec<RouteChoice>);
type DeepHiddenBatchWithRoutes = Vec<DeepHiddenWithRoutes>;

impl DecoderLayer {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
        vectors: &DecoderLayerVectors,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}");
        Ok(Self {
            layer,
            attn_hc: HcWeights::load(index, config, &prefix, "attn", &vectors.attn_hc)?,
            ffn_hc: HcWeights::load(index, config, &prefix, "ffn", &vectors.ffn_hc)?,
            attn_norm: Arc::clone(&vectors.attn_norm),
            ffn_norm: Arc::clone(&vectors.ffn_norm),
            attention: AttentionWeights::load(index, config, layer, &vectors.attention)?,
            moe: MoeWeights::load(index, config, layer, vectors.correction_bias.as_ref())?,
        })
    }

    fn load_reusing(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
        vectors: &DecoderLayerVectors,
        reuse: &mut DecoderLayerBuffers,
        prevalidated: bool,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}");
        Ok(Self {
            layer,
            attn_hc: HcWeights::load_reusing(
                index,
                config,
                &prefix,
                "attn",
                &vectors.attn_hc,
                reuse,
                prevalidated,
            )?,
            ffn_hc: HcWeights::load_reusing(
                index,
                config,
                &prefix,
                "ffn",
                &vectors.ffn_hc,
                reuse,
                prevalidated,
            )?,
            attn_norm: Arc::clone(&vectors.attn_norm),
            ffn_norm: Arc::clone(&vectors.ffn_norm),
            attention: AttentionWeights::load_reusing(
                index,
                config,
                layer,
                &vectors.attention,
                reuse,
                prevalidated,
            )?,
            moe: MoeWeights::load_reusing(
                index,
                config,
                layer,
                vectors.correction_bias.as_ref(),
                reuse,
                prevalidated,
            )?,
        })
    }

    fn into_native_buffers(self) -> DecoderLayerBuffers {
        let Self {
            attn_hc,
            ffn_hc,
            attention,
            moe,
            ..
        } = self;
        let mut buffers = DecoderLayerBuffers::default();
        buffers.push_weight(attn_hc.function);
        buffers.push_weight(ffn_hc.function);
        attention.append_native_buffers(&mut buffers);
        moe.append_native_buffers(&mut buffers);
        buffers
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &[f32],
        token: usize,
        position: usize,
        attention_state: &mut AttentionState,
        experts: &mut DeepExpertStore,
        config: &DeepseekV4Config,
        flow_id: u64,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), DeepseekRuntimeError> {
        let mut layer_profile = span(ProfileStage::DeepseekLayer);
        layer_profile.set_token(position, token);
        layer_profile.set_layer_id(self.layer);
        layer_profile.set_flow_id(flow_id);
        let residual = hidden;
        let (collapsed, mix) = self.attn_hc.pre(hidden, config)?;
        let normalized = bf16_rms_norm(&collapsed, &self.attn_norm, config.rms_norm_eps as f32)?;
        let branch = {
            let mut profile = span(ProfileStage::DeepseekAttention);
            profile.set_token(position, token);
            profile.set_layer_id(self.layer);
            profile.set_flow_id(flow_id);
            self.attention
                .forward(&normalized, position, attention_state, config)?
        };
        let after_attention = hyper_connection_post(&branch, residual, config.hidden_size, &mix)?;

        let residual = after_attention;
        let (collapsed, mix) = self.ffn_hc.pre(&residual, config)?;
        let normalized = bf16_rms_norm(&collapsed, &self.ffn_norm, config.rms_norm_eps as f32)?;
        let (branch, routes) = {
            let mut profile = span(ProfileStage::DeepseekMoe);
            profile.set_token(position, token);
            profile.set_layer_id(self.layer);
            profile.set_flow_id(flow_id);
            self.moe
                .forward(&normalized, token, position, flow_id, experts, config)?
        };
        Ok((
            hyper_connection_post(&branch, &residual, config.hidden_size, &mix)?,
            routes,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_feed_forward(
        &self,
        hidden: Vec<f32>,
        token: usize,
        position: usize,
        attention_state: &mut AttentionState,
        config: &DeepseekV4Config,
        flow_id: u64,
    ) -> Result<PreparedDeepFeedForward, DeepseekRuntimeError> {
        let mut layer_profile = span(ProfileStage::DeepseekLayer);
        layer_profile.set_token(position, token);
        layer_profile.set_layer_id(self.layer);
        layer_profile.set_flow_id(flow_id);
        let (collapsed, attention_mix) = self.attn_hc.pre(&hidden, config)?;
        let normalized = bf16_rms_norm(&collapsed, &self.attn_norm, config.rms_norm_eps as f32)?;
        let branch = {
            let mut profile = span(ProfileStage::DeepseekAttention);
            profile.set_token(position, token);
            profile.set_layer_id(self.layer);
            profile.set_flow_id(flow_id);
            self.attention
                .forward(&normalized, position, attention_state, config)?
        };
        let residual = hyper_connection_post(&branch, &hidden, config.hidden_size, &attention_mix)?;
        let (collapsed, mix) = self.ffn_hc.pre(&residual, config)?;
        let normalized = bf16_rms_norm(&collapsed, &self.ffn_norm, config.rms_norm_eps as f32)?;
        Ok(PreparedDeepFeedForward {
            residual,
            normalized,
            mix,
            position,
            token,
            _layer_profile: layer_profile,
        })
    }

    fn prepare_feed_forward_batch(
        &self,
        inputs: Vec<Vec<f32>>,
        tokens: &[usize],
        start_position: usize,
        attention_state: &mut AttentionState,
        config: &DeepseekV4Config,
        flow_id: u64,
    ) -> Result<Vec<PreparedDeepFeedForward>, DeepseekRuntimeError> {
        if inputs.is_empty() || inputs.len() != tokens.len() {
            return Err(DeepseekRuntimeError::Invalid(
                "layer prefill batch has invalid token geometry".to_owned(),
            ));
        }
        if inputs.len() == 1 {
            return Ok(vec![self.prepare_feed_forward(
                inputs.into_iter().next().expect("one checked input"),
                tokens[0],
                start_position,
                attention_state,
                config,
                flow_id,
            )?]);
        }

        let mut intermediates = Vec::with_capacity(inputs.len());
        let mut normalized_inputs = Vec::with_capacity(inputs.len());
        for (offset, (hidden, &token)) in inputs.into_iter().zip(tokens).enumerate() {
            let position = start_position + offset;
            let mut layer_profile = span(ProfileStage::DeepseekLayer);
            layer_profile.set_token(position, token);
            layer_profile.set_layer_id(self.layer);
            layer_profile.set_flow_id(flow_id);
            let (collapsed, attention_mix) = self.attn_hc.pre(&hidden, config)?;
            normalized_inputs.push(bf16_rms_norm(
                &collapsed,
                &self.attn_norm,
                config.rms_norm_eps as f32,
            )?);
            intermediates.push((hidden, attention_mix, layer_profile));
        }

        let attention_inputs = normalized_inputs
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let branches = {
            let mut profile = span(ProfileStage::DeepseekAttention);
            profile.set_token(start_position, tokens[0]);
            profile.set_layer_id(self.layer);
            profile.set_flow_id(flow_id);
            profile.set_batch_tokens(tokens.len());
            let projected = self.attention.prepare_batch(&attention_inputs, config)?;
            let attention_output = projected
                .into_iter()
                .enumerate()
                .map(|(offset, projected)| {
                    self.attention.forward_prepared_attention(
                        projected,
                        start_position + offset,
                        attention_state,
                        config,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let low_rank = self
                .attention
                .project_low_rank_batch(&attention_output, config)?;
            self.attention.project_output_batch(&low_rank, config)?
        };

        intermediates
            .into_iter()
            .zip(branches)
            .zip(tokens)
            .enumerate()
            .map(
                |(offset, (((hidden, attention_mix, layer_profile), branch), &token))| {
                    let residual = hyper_connection_post(
                        &branch,
                        &hidden,
                        config.hidden_size,
                        &attention_mix,
                    )?;
                    let (collapsed, mix) = self.ffn_hc.pre(&residual, config)?;
                    let normalized =
                        bf16_rms_norm(&collapsed, &self.ffn_norm, config.rms_norm_eps as f32)?;
                    Ok(PreparedDeepFeedForward {
                        residual,
                        normalized,
                        mix,
                        position: start_position + offset,
                        token,
                        _layer_profile: layer_profile,
                    })
                },
            )
            .collect()
    }

    fn finish_feed_forward(
        &self,
        prepared: PreparedDeepFeedForward,
        experts: &mut DeepExpertStore,
        config: &DeepseekV4Config,
        flow_id: u64,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), DeepseekRuntimeError> {
        let branch = {
            let mut profile = span(ProfileStage::DeepseekMoe);
            profile.set_token(prepared.position, prepared.token);
            profile.set_layer_id(self.layer);
            profile.set_flow_id(flow_id);
            self.moe.forward(
                &prepared.normalized,
                prepared.token,
                prepared.position,
                flow_id,
                experts,
                config,
            )?
        };
        Ok((
            hyper_connection_post(
                &branch.0,
                &prepared.residual,
                config.hidden_size,
                &prepared.mix,
            )?,
            branch.1,
        ))
    }

    fn finish_feed_forward_batch(
        &self,
        prepared: Vec<PreparedDeepFeedForward>,
        experts: &mut DeepExpertStore,
        config: &DeepseekV4Config,
        flow_id: u64,
    ) -> Result<DeepHiddenBatchWithRoutes, DeepseekRuntimeError> {
        if prepared.len() < 2 {
            return prepared
                .into_iter()
                .map(|prepared| self.finish_feed_forward(prepared, experts, config, flow_id))
                .collect();
        }
        let inputs = prepared
            .iter()
            .map(|prepared| prepared.normalized.as_slice())
            .collect::<Vec<_>>();
        let tokens = prepared
            .iter()
            .map(|prepared| prepared.token)
            .collect::<Vec<_>>();
        let positions = prepared
            .iter()
            .map(|prepared| prepared.position)
            .collect::<Vec<_>>();
        let batched = {
            let mut profile = span(ProfileStage::DeepseekMoe);
            profile.set_token_position(positions[0]);
            profile.set_layer_id(self.layer);
            profile.set_flow_id(flow_id);
            profile.set_batch_tokens(prepared.len());
            self.moe
                .forward_batch(&inputs, &tokens, &positions, flow_id, experts, config)?
        };
        let branches = match batched {
            DeepBatchForward::Batched(branches) => branches,
            DeepBatchForward::Fallback(routes_by_token) => {
                if routes_by_token.len() != prepared.len() {
                    return Err(DeepseekRuntimeError::Invalid(
                        "MoE fallback route batch length changed".to_owned(),
                    ));
                }
                return prepared
                    .into_iter()
                    .zip(routes_by_token)
                    .map(|(prepared, routes)| {
                        let branch = {
                            let mut profile = span(ProfileStage::DeepseekMoe);
                            profile.set_token(prepared.position, prepared.token);
                            profile.set_layer_id(self.layer);
                            profile.set_flow_id(flow_id);
                            self.moe.forward_prepared(
                                &prepared.normalized,
                                prepared.token,
                                prepared.position,
                                flow_id,
                                routes,
                                experts,
                                config,
                            )?
                        };
                        Ok((
                            hyper_connection_post(
                                &branch.0,
                                &prepared.residual,
                                config.hidden_size,
                                &prepared.mix,
                            )?,
                            branch.1,
                        ))
                    })
                    .collect();
            }
        };
        prepared
            .into_iter()
            .zip(branches)
            .map(|(prepared, (branch, routes))| {
                Ok((
                    hyper_connection_post(
                        &branch,
                        &prepared.residual,
                        config.hidden_size,
                        &prepared.mix,
                    )?,
                    routes,
                ))
            })
            .collect()
    }
}

#[derive(Debug)]
struct AttentionWeights {
    sink: Arc<[f32]>,
    wq_a: WeightMatrix,
    q_norm: Arc<[f32]>,
    wq_b: WeightMatrix,
    wkv: WeightMatrix,
    kv_norm: Arc<[f32]>,
    wo_a: WeightMatrix,
    wo_b: WeightMatrix,
    compressor: Option<CompressorWeights>,
    indexer: Option<IndexerWeights>,
    ratio: usize,
}

struct PreparedAttention {
    query: Vec<f32>,
    local_kv: Vec<f32>,
    compressor: Option<PreparedCompressor>,
    indexer: Option<PreparedIndexer>,
}

impl AttentionWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
        vectors: &AttentionVectorWeights,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}.attn");
        let ratio = config.base_compress_ratio(layer).unwrap_or(0);
        let compressor = (ratio > 0)
            .then(|| {
                load_compressor(
                    index,
                    config,
                    &format!("{prefix}.compressor"),
                    ratio,
                    config.head_dim,
                    false,
                    vectors
                        .compressor_norm
                        .as_ref()
                        .expect("compressed attention has a resident norm"),
                )
            })
            .transpose()?;
        let indexer = (ratio == 4)
            .then(|| {
                IndexerWeights::load(
                    index,
                    config,
                    &format!("{prefix}.indexer"),
                    vectors
                        .indexer_compressor_norm
                        .as_ref()
                        .expect("ratio-4 attention has a resident indexer norm"),
                )
            })
            .transpose()?;
        Ok(Self {
            sink: Arc::clone(&vectors.sink),
            wq_a: matrix(
                index,
                &format!("{prefix}.wq_a.weight"),
                config.q_lora_rank,
                config.hidden_size,
            )?,
            q_norm: Arc::clone(&vectors.q_norm),
            wq_b: matrix(
                index,
                &format!("{prefix}.wq_b.weight"),
                config.num_attention_heads * config.head_dim,
                config.q_lora_rank,
            )?,
            wkv: matrix(
                index,
                &format!("{prefix}.wkv.weight"),
                config.head_dim,
                config.hidden_size,
            )?,
            kv_norm: Arc::clone(&vectors.kv_norm),
            wo_a: matrix(
                index,
                &format!("{prefix}.wo_a.weight"),
                config.o_groups * config.o_lora_rank,
                config.num_attention_heads * config.head_dim / config.o_groups,
            )?,
            wo_b: matrix(
                index,
                &format!("{prefix}.wo_b.weight"),
                config.hidden_size,
                config.o_groups * config.o_lora_rank,
            )?,
            compressor,
            indexer,
            ratio,
        })
    }

    fn load_reusing(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
        vectors: &AttentionVectorWeights,
        reuse: &mut DecoderLayerBuffers,
        prevalidated: bool,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}.attn");
        let ratio = config.base_compress_ratio(layer).unwrap_or(0);
        let compressor = (ratio > 0)
            .then(|| {
                load_compressor_reusing(
                    index,
                    config,
                    &format!("{prefix}.compressor"),
                    ratio,
                    config.head_dim,
                    false,
                    vectors
                        .compressor_norm
                        .as_ref()
                        .expect("compressed attention has a resident norm"),
                    reuse,
                    prevalidated,
                )
            })
            .transpose()?;
        let indexer = (ratio == 4)
            .then(|| {
                IndexerWeights::load_reusing(
                    index,
                    config,
                    &format!("{prefix}.indexer"),
                    vectors
                        .indexer_compressor_norm
                        .as_ref()
                        .expect("ratio-4 attention has a resident indexer norm"),
                    reuse,
                    prevalidated,
                )
            })
            .transpose()?;
        Ok(Self {
            sink: Arc::clone(&vectors.sink),
            wq_a: matrix_reusing(
                index,
                &format!("{prefix}.wq_a.weight"),
                config.q_lora_rank,
                config.hidden_size,
                reuse,
                prevalidated,
            )?,
            q_norm: Arc::clone(&vectors.q_norm),
            wq_b: matrix_reusing(
                index,
                &format!("{prefix}.wq_b.weight"),
                config.num_attention_heads * config.head_dim,
                config.q_lora_rank,
                reuse,
                prevalidated,
            )?,
            wkv: matrix_reusing(
                index,
                &format!("{prefix}.wkv.weight"),
                config.head_dim,
                config.hidden_size,
                reuse,
                prevalidated,
            )?,
            kv_norm: Arc::clone(&vectors.kv_norm),
            wo_a: matrix_reusing(
                index,
                &format!("{prefix}.wo_a.weight"),
                config.o_groups * config.o_lora_rank,
                config.num_attention_heads * config.head_dim / config.o_groups,
                reuse,
                prevalidated,
            )?,
            wo_b: matrix_reusing(
                index,
                &format!("{prefix}.wo_b.weight"),
                config.hidden_size,
                config.o_groups * config.o_lora_rank,
                reuse,
                prevalidated,
            )?,
            compressor,
            indexer,
            ratio,
        })
    }

    fn append_native_buffers(self, buffers: &mut DecoderLayerBuffers) {
        let Self {
            wq_a,
            wq_b,
            wkv,
            wo_a,
            wo_b,
            compressor,
            indexer,
            ..
        } = self;
        for weight in [wq_a, wq_b, wkv, wo_a, wo_b] {
            buffers.push_weight(weight);
        }
        if let Some(compressor) = compressor {
            for weight in compressor.into_weight_matrices() {
                buffers.push_weight(weight);
            }
        }
        if let Some(indexer) = indexer {
            indexer.append_native_buffers(buffers);
        }
    }

    fn forward(
        &self,
        input: &[f32],
        position: usize,
        state: &mut AttentionState,
        config: &DeepseekV4Config,
    ) -> Result<Vec<f32>, DeepseekRuntimeError> {
        let prepared = self
            .prepare_batch(&[input], config)?
            .pop()
            .expect("one attention input produces one prepared projection");
        let attention_output =
            self.forward_prepared_attention(prepared, position, state, config)?;
        let low_rank = self
            .project_low_rank_batch(&[attention_output], config)?
            .pop()
            .expect("one attention output produces one low-rank projection");
        Ok(linear(&self.wo_b, &low_rank)?)
    }

    fn prepare_batch(
        &self,
        inputs: &[&[f32]],
        config: &DeepseekV4Config,
    ) -> Result<Vec<PreparedAttention>, DeepseekRuntimeError> {
        if inputs.is_empty() || inputs.iter().any(|input| input.len() != config.hidden_size) {
            return Err(DeepseekRuntimeError::Invalid(
                "attention projection batch has invalid input geometry".to_owned(),
            ));
        }
        let batch = inputs.len();
        let mut input = Vec::with_capacity(batch.saturating_mul(config.hidden_size));
        let uses_input_mx = self.wq_a.uses_mx_activation_quantization()
            || self.wkv.uses_mx_activation_quantization();
        let mut mx_input = Vec::with_capacity(if uses_input_mx { input.capacity() } else { 0 });
        for &token in inputs {
            input.extend_from_slice(token);
            if uses_input_mx {
                mx_input.extend(prepare_mx_activation(token)?);
            }
        }

        let projected_qr = linear_with_mx_activation_batch(&self.wq_a, &input, &mx_input, batch)?;
        let projected_kv = linear_with_mx_activation_batch(&self.wkv, &input, &mx_input, batch)?;
        let mut normalized_qr = Vec::with_capacity(projected_qr.len());
        for qr in projected_qr.chunks_exact(config.q_lora_rank) {
            normalized_qr.extend(bf16_rms_norm(qr, &self.q_norm, config.rms_norm_eps as f32)?);
        }
        let uses_qr_mx = self.wq_b.uses_mx_activation_quantization();
        let mut mx_qr = Vec::with_capacity(if uses_qr_mx {
            normalized_qr.capacity()
        } else {
            0
        });
        if uses_qr_mx {
            for qr in normalized_qr.chunks_exact(config.q_lora_rank) {
                mx_qr.extend(prepare_mx_activation(qr)?);
            }
        }
        let mut projected_query =
            linear_with_mx_activation_batch(&self.wq_b, &normalized_qr, &mx_qr, batch)?;
        for query in projected_query.chunks_exact_mut(config.num_attention_heads * config.head_dim)
        {
            unit_rms_norm_in_place(query, config.head_dim, config.rms_norm_eps as f32)?;
        }

        let mut compressor = self
            .compressor
            .as_ref()
            .map(|compressor| compressor.prepare_batch(inputs))
            .transpose()?
            .map(Vec::into_iter);
        let qr_inputs = normalized_qr
            .chunks_exact(config.q_lora_rank)
            .collect::<Vec<_>>();
        let mut indexer = self
            .indexer
            .as_ref()
            .map(|indexer| indexer.prepare_batch(inputs, &qr_inputs, config))
            .transpose()?
            .map(Vec::into_iter);

        normalized_qr
            .chunks_exact(config.q_lora_rank)
            .zip(projected_query.chunks_exact(config.num_attention_heads * config.head_dim))
            .zip(projected_kv.chunks_exact(config.head_dim))
            .map(|((_qr, query), local_kv)| {
                Ok(PreparedAttention {
                    query: query.to_vec(),
                    local_kv: bf16_rms_norm(local_kv, &self.kv_norm, config.rms_norm_eps as f32)?,
                    compressor: compressor
                        .as_mut()
                        .map(|prepared| prepared.next().expect("aligned compressor batch")),
                    indexer: indexer
                        .as_mut()
                        .map(|prepared| prepared.next().expect("aligned indexer batch")),
                })
            })
            .collect()
    }

    fn forward_prepared_attention(
        &self,
        prepared: PreparedAttention,
        position: usize,
        state: &mut AttentionState,
        config: &DeepseekV4Config,
    ) -> Result<Vec<f32>, DeepseekRuntimeError> {
        if state.next_position != position || state.ratio != self.ratio {
            return Err(DeepseekRuntimeError::Invalid(
                "attention state position/ratio mismatch".to_owned(),
            ));
        }
        let PreparedAttention {
            mut query,
            mut local_kv,
            compressor,
            indexer,
        } = prepared;
        let rope_original =
            (self.ratio > 0).then_some(config.rope_scaling.original_max_position_embeddings);
        let rope_base = if self.ratio > 0 {
            config.compress_rope_theta
        } else {
            config.rope_theta
        } as f32;
        let factor = if self.ratio > 0 {
            config.rope_scaling.factor as f32
        } else {
            1.0
        };
        for head in query.chunks_mut(config.head_dim) {
            paired_rope(
                &mut head[config.head_dim - config.qk_rope_head_dim..],
                position,
                rope_base,
                rope_original,
                factor,
                config.rope_scaling.beta_fast,
                config.rope_scaling.beta_slow,
                false,
            )?;
        }
        paired_rope(
            &mut local_kv[config.head_dim - config.qk_rope_head_dim..],
            position,
            rope_base,
            rope_original,
            factor,
            config.rope_scaling.beta_fast,
            config.rope_scaling.beta_slow,
            false,
        )?;
        let non_rope = config.head_dim - config.qk_rope_head_dim;
        if non_rope > 0 {
            let quantized = simulate_e4m3_activation(&local_kv[..non_rope], 64)
                .map_err(|error| DeepseekRuntimeError::Invalid(error.to_string()))?;
            local_kv[..non_rope].copy_from_slice(&quantized);
        }
        state.local[position % config.sliding_window] = Some(local_kv);

        let compressed_selection = match (&self.indexer, &mut state.indexer, indexer) {
            (Some(indexer), Some(indexer_state), Some(prepared)) => {
                indexer.forward_prepared(prepared, position, indexer_state, config)?
            }
            (None, None, None) => Vec::new(),
            _ => {
                return Err(DeepseekRuntimeError::Invalid(
                    "indexer weights, state, and prepared projection disagree".to_owned(),
                ));
            }
        };
        match (&self.compressor, &mut state.compressor, compressor) {
            (Some(compressor), Some(compressor_state), Some(prepared)) => {
                let _ = compressor.forward_prepared(
                    prepared,
                    position,
                    compressor_state,
                    rope_base,
                    rope_original,
                    factor,
                    config.rope_scaling.beta_fast,
                    config.rope_scaling.beta_slow,
                )?;
            }
            (None, None, None) => {}
            _ => {
                return Err(DeepseekRuntimeError::Invalid(
                    "compressor weights, state, and prepared projection disagree".to_owned(),
                ));
            }
        }

        let mut selected = window_indices(position, config.sliding_window);
        if self.ratio == 4 {
            selected.extend(compressed_selection);
        } else if self.ratio > 0 {
            let count = (position + 1) / self.ratio;
            selected.extend((0..count).map(|index| config.sliding_window + index));
        }
        let zero = vec![0.0; config.head_dim];
        let mut cache = state
            .local
            .iter()
            .map(|entry| entry.as_deref().unwrap_or(&zero))
            .collect::<Vec<_>>();
        if let Some(compressor) = &state.compressor {
            cache.extend(compressor.compressed().iter().map(Vec::as_slice));
        }
        let mut output = sparse_attention_with_sink(
            &query,
            config.num_attention_heads,
            config.head_dim,
            &cache,
            &selected,
            &self.sink,
            (config.head_dim as f32).sqrt().recip(),
        )?;
        for head in output.chunks_mut(config.head_dim) {
            paired_rope(
                &mut head[config.head_dim - config.qk_rope_head_dim..],
                position,
                rope_base,
                rope_original,
                factor,
                config.rope_scaling.beta_fast,
                config.rope_scaling.beta_slow,
                true,
            )?;
        }
        state.next_position += 1;
        Ok(output)
    }

    fn project_low_rank_batch(
        &self,
        attention_output: &[Vec<f32>],
        config: &DeepseekV4Config,
    ) -> Result<Vec<Vec<f32>>, DeepseekRuntimeError> {
        let attention_width = config.num_attention_heads * config.head_dim;
        if attention_output.is_empty()
            || attention_output
                .iter()
                .any(|output| output.len() != attention_width)
        {
            return Err(DeepseekRuntimeError::Invalid(
                "attention output batch has invalid head geometry".to_owned(),
            ));
        }
        let batch = attention_output.len();
        let heads_per_group = config.num_attention_heads / config.o_groups;
        let group_width = heads_per_group * config.head_dim;
        let mut low_rank = (0..batch)
            .map(|_| Vec::with_capacity(config.o_groups * config.o_lora_rank))
            .collect::<Vec<_>>();
        for group in 0..config.o_groups {
            let mut input = Vec::with_capacity(batch.saturating_mul(group_width));
            for output in attention_output {
                input.extend_from_slice(&output[group * group_width..(group + 1) * group_width]);
            }
            // The official reference explicitly dequantizes `wo_a` and applies this grouped
            // einsum without activation-side FP8 simulation.
            let mut projected = self.wo_a.matmul_row_range(
                group * config.o_lora_rank,
                config.o_lora_rank,
                &input,
                batch,
            )?;
            round_to_bf16_in_place(&mut projected)?;
            for (token, values) in low_rank
                .iter_mut()
                .zip(projected.chunks_exact(config.o_lora_rank))
            {
                token.extend_from_slice(values);
            }
        }
        Ok(low_rank)
    }

    fn project_output_batch(
        &self,
        low_rank: &[Vec<f32>],
        config: &DeepseekV4Config,
    ) -> Result<Vec<Vec<f32>>, DeepseekRuntimeError> {
        if low_rank.is_empty() || low_rank.iter().any(|input| input.len() != self.wo_b.cols()) {
            return Err(DeepseekRuntimeError::Invalid(
                "attention output batch has invalid low-rank geometry".to_owned(),
            ));
        }
        let batch = low_rank.len();
        let mut input = Vec::with_capacity(batch.saturating_mul(self.wo_b.cols()));
        let uses_mx = self.wo_b.uses_mx_activation_quantization();
        let mut mx_input = Vec::with_capacity(if uses_mx { input.capacity() } else { 0 });
        for token in low_rank {
            input.extend_from_slice(token);
            if uses_mx {
                mx_input.extend(prepare_mx_activation(token)?);
            }
        }
        let output = linear_with_mx_activation_batch(&self.wo_b, &input, &mx_input, batch)?;
        if output.len() != batch.saturating_mul(config.hidden_size) {
            return Err(DeepseekRuntimeError::Invalid(
                "attention output batch returned invalid geometry".to_owned(),
            ));
        }
        Ok(output
            .chunks_exact(config.hidden_size)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

#[derive(Debug, Clone)]
struct AttentionState {
    ratio: usize,
    next_position: usize,
    local: Vec<Option<Vec<f32>>>,
    compressor: Option<CompressorState>,
    indexer: Option<CompressorState>,
}

#[derive(Debug)]
struct AttentionStateCheckpoint {
    next_position: usize,
    local_slot: usize,
    previous_local: Option<Vec<f32>>,
    compressor: Option<CompressorCheckpoint>,
    indexer: Option<CompressorCheckpoint>,
}

impl AttentionState {
    fn new(config: &DeepseekV4Config, ratio: usize) -> Result<Self, DeepseekRuntimeError> {
        Ok(Self {
            ratio,
            next_position: 0,
            local: vec![None; config.sliding_window],
            compressor: (ratio > 0)
                .then(|| CompressorState::new(ratio, config.head_dim))
                .transpose()?,
            indexer: (ratio == 4)
                .then(|| CompressorState::new(ratio, config.index_head_dim))
                .transpose()?,
        })
    }

    fn stored_f32_elements(&self) -> usize {
        self.local
            .iter()
            .filter_map(Option::as_ref)
            .map(Vec::len)
            .sum::<usize>()
            + self
                .compressor
                .as_ref()
                .map(CompressorState::stored_f32_elements)
                .unwrap_or(0)
            + self
                .indexer
                .as_ref()
                .map(CompressorState::stored_f32_elements)
                .unwrap_or(0)
    }

    fn checkpoint_next_token(&mut self) -> AttentionStateCheckpoint {
        let local_slot = self.next_position % self.local.len();
        AttentionStateCheckpoint {
            next_position: self.next_position,
            local_slot,
            previous_local: self.local[local_slot].take(),
            compressor: self.compressor.as_ref().map(CompressorState::checkpoint),
            indexer: self.indexer.as_ref().map(CompressorState::checkpoint),
        }
    }

    fn restore(&mut self, checkpoint: AttentionStateCheckpoint) {
        self.next_position = checkpoint.next_position;
        self.local[checkpoint.local_slot] = checkpoint.previous_local;
        if let (Some(state), Some(checkpoint)) = (&mut self.compressor, checkpoint.compressor) {
            state.restore(checkpoint);
        }
        if let (Some(state), Some(checkpoint)) = (&mut self.indexer, checkpoint.indexer) {
            state.restore(checkpoint);
        }
    }
}

#[derive(Debug)]
struct IndexerWeights {
    wq_b: WeightMatrix,
    weights_proj: WeightMatrix,
    compressor: CompressorWeights,
}

struct PreparedIndexer {
    query: Vec<f32>,
    weights: Vec<f32>,
    compressor: PreparedCompressor,
}

impl IndexerWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        prefix: &str,
        compressor_norm: &Arc<[f32]>,
    ) -> Result<Self, DeepseekRuntimeError> {
        Ok(Self {
            wq_b: matrix(
                index,
                &format!("{prefix}.wq_b.weight"),
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
            )?,
            weights_proj: matrix(
                index,
                &format!("{prefix}.weights_proj.weight"),
                config.index_n_heads,
                config.hidden_size,
            )?,
            compressor: load_compressor(
                index,
                config,
                &format!("{prefix}.compressor"),
                4,
                config.index_head_dim,
                true,
                compressor_norm,
            )?,
        })
    }

    fn load_reusing(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        prefix: &str,
        compressor_norm: &Arc<[f32]>,
        reuse: &mut DecoderLayerBuffers,
        prevalidated: bool,
    ) -> Result<Self, DeepseekRuntimeError> {
        Ok(Self {
            wq_b: matrix_reusing(
                index,
                &format!("{prefix}.wq_b.weight"),
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
                reuse,
                prevalidated,
            )?,
            weights_proj: matrix_reusing(
                index,
                &format!("{prefix}.weights_proj.weight"),
                config.index_n_heads,
                config.hidden_size,
                reuse,
                prevalidated,
            )?,
            compressor: load_compressor_reusing(
                index,
                config,
                &format!("{prefix}.compressor"),
                4,
                config.index_head_dim,
                true,
                compressor_norm,
                reuse,
                prevalidated,
            )?,
        })
    }

    fn append_native_buffers(self, buffers: &mut DecoderLayerBuffers) {
        buffers.push_weight(self.wq_b);
        buffers.push_weight(self.weights_proj);
        for weight in self.compressor.into_weight_matrices() {
            buffers.push_weight(weight);
        }
    }

    fn prepare_batch(
        &self,
        inputs: &[&[f32]],
        qrs: &[&[f32]],
        config: &DeepseekV4Config,
    ) -> Result<Vec<PreparedIndexer>, DeepseekRuntimeError> {
        if inputs.is_empty()
            || inputs.len() != qrs.len()
            || inputs.iter().any(|input| input.len() != config.hidden_size)
            || qrs.iter().any(|qr| qr.len() != config.q_lora_rank)
        {
            return Err(DeepseekRuntimeError::Invalid(
                "indexer projection batch has invalid input geometry".to_owned(),
            ));
        }
        let batch = inputs.len();
        let mut input = Vec::with_capacity(batch.saturating_mul(config.hidden_size));
        let input_uses_mx = self.weights_proj.uses_mx_activation_quantization();
        let mut mx_input = Vec::with_capacity(if input_uses_mx { input.capacity() } else { 0 });
        for &token in inputs {
            input.extend_from_slice(token);
            if input_uses_mx {
                mx_input.extend(prepare_mx_activation(token)?);
            }
        }
        let mut qr = Vec::with_capacity(batch.saturating_mul(config.q_lora_rank));
        let qr_uses_mx = self.wq_b.uses_mx_activation_quantization();
        let mut mx_qr = Vec::with_capacity(if qr_uses_mx { qr.capacity() } else { 0 });
        for &token in qrs {
            qr.extend_from_slice(token);
            if qr_uses_mx {
                mx_qr.extend(prepare_mx_activation(token)?);
            }
        }
        let query = linear_with_mx_activation_batch(&self.wq_b, &qr, &mx_qr, batch)?;
        let mut weights =
            linear_with_mx_activation_batch(&self.weights_proj, &input, &mx_input, batch)?;
        let weight_scale = (config.index_head_dim as f32).sqrt().recip()
            * (config.index_n_heads as f32).sqrt().recip();
        for token in weights.chunks_exact_mut(config.index_n_heads) {
            round_to_bf16_in_place(token)?;
            for value in token.iter_mut() {
                *value *= weight_scale;
            }
            round_to_bf16_in_place(token)?;
        }
        let compressor = self.compressor.prepare_batch(inputs)?;
        Ok(query
            .chunks_exact(config.index_n_heads * config.index_head_dim)
            .zip(weights.chunks_exact(config.index_n_heads))
            .zip(compressor)
            .map(|((query, weights), compressor)| PreparedIndexer {
                query: query.to_vec(),
                weights: weights.to_vec(),
                compressor,
            })
            .collect())
    }

    fn forward_prepared(
        &self,
        prepared: PreparedIndexer,
        position: usize,
        state: &mut CompressorState,
        config: &DeepseekV4Config,
    ) -> Result<Vec<usize>, DeepseekRuntimeError> {
        let PreparedIndexer {
            mut query,
            weights,
            compressor,
        } = prepared;
        for head in query.chunks_mut(config.index_head_dim) {
            paired_rope(
                &mut head[config.index_head_dim - config.qk_rope_head_dim..],
                position,
                config.compress_rope_theta as f32,
                Some(config.rope_scaling.original_max_position_embeddings),
                config.rope_scaling.factor as f32,
                config.rope_scaling.beta_fast,
                config.rope_scaling.beta_slow,
                false,
            )?;
            normalized_hadamard(head)
                .map_err(|error| DeepseekRuntimeError::Invalid(error.to_string()))?;
            round_to_bf16_in_place(head)?;
            let quantized = simulate_e2m1_activation(head, 32)
                .map_err(|error| DeepseekRuntimeError::Invalid(error.to_string()))?;
            head.copy_from_slice(&quantized);
        }
        let _ = self.compressor.forward_prepared(
            compressor,
            position,
            state,
            config.compress_rope_theta as f32,
            Some(config.rope_scaling.original_max_position_embeddings),
            config.rope_scaling.factor as f32,
            config.rope_scaling.beta_fast,
            config.rope_scaling.beta_slow,
        )?;
        let mut scores = Vec::with_capacity(state.compressed().len());
        for key in state.compressed() {
            let mut score = 0.0f32;
            for head in 0..config.index_n_heads {
                let query =
                    &query[head * config.index_head_dim..(head + 1) * config.index_head_dim];
                let dot = query.iter().zip(key).map(|(&q, &k)| q * k).sum::<f32>();
                score += dot.max(0.0) * weights[head];
            }
            scores.push(score);
        }
        let mut ranking = (0..scores.len()).collect::<Vec<_>>();
        ranking.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
        ranking.truncate(config.index_topk.min(ranking.len()));
        Ok(ranking
            .into_iter()
            .map(|index| config.sliding_window + index)
            .collect())
    }
}

#[derive(Debug)]
struct MoeWeights {
    layer: usize,
    router: WeightMatrix,
    correction_bias: Option<Arc<[f32]>>,
    shared: DeepExpert,
}

struct PreparedDeepRoutes {
    routes: Vec<RouteChoice>,
    execution_order: Vec<(usize, f32)>,
    expert_ids: Vec<usize>,
    mx_input: Vec<f32>,
}

enum DeepBatchForward {
    Batched(DeepHiddenBatchWithRoutes),
    Fallback(Vec<PreparedDeepRoutes>),
}

impl MoeWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
        correction_bias: Option<&Arc<[f32]>>,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}.ffn");
        Ok(Self {
            layer,
            router: matrix(
                index,
                &format!("{prefix}.gate.weight"),
                config.n_routed_experts,
                config.hidden_size,
            )?,
            correction_bias: correction_bias.map(Arc::clone),
            shared: DeepExpert::load(index, &format!("{prefix}.shared_experts"), config, u64::MAX)?
                .0,
        })
    }

    fn load_reusing(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
        correction_bias: Option<&Arc<[f32]>>,
        reuse: &mut DecoderLayerBuffers,
        prevalidated: bool,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}.ffn");
        Ok(Self {
            layer,
            router: matrix_reusing(
                index,
                &format!("{prefix}.gate.weight"),
                config.n_routed_experts,
                config.hidden_size,
                reuse,
                prevalidated,
            )?,
            correction_bias: correction_bias.map(Arc::clone),
            shared: DeepExpert::load_layer_reusing(
                index,
                &format!("{prefix}.shared_experts"),
                config,
                reuse,
                prevalidated,
            )?,
        })
    }

    fn append_native_buffers(self, buffers: &mut DecoderLayerBuffers) {
        buffers.push_weight(self.router);
        self.shared.append_native_buffers(buffers);
    }

    fn forward(
        &self,
        input: &[f32],
        token: usize,
        position: usize,
        flow_id: u64,
        experts: &mut DeepExpertStore,
        config: &DeepseekV4Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), DeepseekRuntimeError> {
        let prepared = self.prepare_routes(input, token, experts, config)?;
        self.forward_prepared(input, token, position, flow_id, prepared, experts, config)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prepared(
        &self,
        input: &[f32],
        token: usize,
        position: usize,
        flow_id: u64,
        prepared: PreparedDeepRoutes,
        experts: &mut DeepExpertStore,
        config: &DeepseekV4Config,
    ) -> Result<DeepHiddenWithRoutes, DeepseekRuntimeError> {
        let PreparedDeepRoutes {
            routes,
            execution_order,
            expert_ids,
            mx_input,
        } = prepared;
        let mut output = vec![0.0f32; config.hidden_size];
        let trace_context = ExpertTraceContext {
            position,
            token,
            flow_id,
        };
        let can_run_parallel = experts.can_acquire_batch(self.layer, &expert_ids)
            || experts.prepare_ordered_batch(self.layer, &expert_ids);
        let shared = if can_run_parallel {
            let cache_hits = expert_ids
                .iter()
                .map(|&expert| experts.contains(self.layer, expert))
                .collect::<Vec<_>>();
            // Shared weights are resident and independent of routed expert I/O. Compute them on
            // the caller while bounded I/O workers fetch/decode cache misses.
            let (loaded, shared) =
                experts.acquire_batch_while(self.layer, &expert_ids, trace_context, || {
                    self.shared.forward_with_mx_input(input, &mx_input, None)
                })?;
            // All handles fit simultaneously, so expose the independent routed experts to the
            // Rayon pool together. Their inner row-parallel matvecs may still steal work across
            // experts, avoiding six consecutive small synchronization waves. Indexed collection
            // retains ascending expert-ID order; accumulation below therefore stays bit exact.
            let profile_context = capture_context();
            let computed = install(|| {
                loaded
                    .par_iter()
                    .zip(execution_order.par_iter())
                    .zip(cache_hits.par_iter())
                    .map(|((expert, &(expert_id, route_weight)), &cache_hit)| {
                        profile_context.enter(|| {
                            compute_routed_expert(
                                expert,
                                input,
                                &mx_input,
                                route_weight,
                                expert_id,
                                cache_hit,
                                self.layer,
                                trace_context,
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?;
            for computed in computed {
                accumulate_routed_output(&mut output, computed);
            }
            shared
        } else if experts.slots_per_layer > 0 {
            // The planner reserves one transient expert beyond the cache. Use it as a one-stage
            // pipeline: shared compute hides the first miss, then expert N compute hides the read
            // for N+1. Installation remains in sorted route order, and the current Arc is dropped
            // before the next insertion can evict it, so physical residency stays bounded.
            let (mut current_id, mut current_weight) = execution_order[0];
            let mut current_hit = experts.contains(self.layer, current_id);
            let mut current_load =
                (!current_hit).then(|| experts.preload(self.layer, current_id, trace_context));
            let shared = match self.shared.forward_with_mx_input(input, &mx_input, None) {
                Ok(shared) => shared,
                Err(error) => {
                    if let Some(load) = current_load {
                        let _ = load.join();
                    }
                    return Err(error);
                }
            };

            for next_index in 1..=execution_order.len() {
                let expert = if current_hit {
                    experts.acquire(self.layer, current_id, trace_context)?
                } else {
                    let loaded = current_load
                        .take()
                        .expect("a cache miss has one pending load")
                        .join()?;
                    experts.install_preloaded(self.layer, current_id, loaded)?
                };
                let next = execution_order.get(next_index).copied();
                let mut next_hit = false;
                let mut next_load = None;
                if let Some((next_id, _)) = next {
                    next_hit = experts.contains(self.layer, next_id);
                    if !next_hit {
                        next_load = Some(experts.preload(self.layer, next_id, trace_context));
                    }
                }
                let computed = accumulate_routed_expert(
                    &mut output,
                    &expert,
                    input,
                    &mx_input,
                    current_weight,
                    current_id,
                    current_hit,
                    self.layer,
                    trace_context,
                );
                drop(expert);
                if let Err(error) = computed {
                    if let Some(load) = next_load {
                        let _ = load.join();
                    }
                    return Err(error);
                }
                if let Some((next_id, next_weight)) = next {
                    current_id = next_id;
                    current_weight = next_weight;
                    current_hit = next_hit;
                    current_load = next_load;
                }
            }
            shared
        } else {
            // A zero-slot test/runtime has no cache allocation beyond the single transient expert,
            // so it must load and release each route synchronously.
            for (expert_id, route_weight) in execution_order.iter().copied() {
                let cache_hit = experts.contains(self.layer, expert_id);
                let expert = experts.acquire(self.layer, expert_id, trace_context)?;
                accumulate_routed_expert(
                    &mut output,
                    &expert,
                    input,
                    &mx_input,
                    route_weight,
                    expert_id,
                    cache_hit,
                    self.layer,
                    trace_context,
                )?;
            }
            self.shared.forward_with_mx_input(input, &mx_input, None)?
        };
        // The official path accumulates routed experts in FP32, adds the shared expert last,
        // and only then casts the combined MoE output back to BF16.
        for (output, shared) in output.iter_mut().zip(shared) {
            *output += shared;
        }
        round_to_bf16_in_place(&mut output)?;
        Ok((output, routes))
    }

    fn prepare_routes(
        &self,
        input: &[f32],
        token: usize,
        experts: &DeepExpertStore,
        config: &DeepseekV4Config,
    ) -> Result<PreparedDeepRoutes, DeepseekRuntimeError> {
        let logits = linear(&self.router, input)?;
        let hash = if self.layer < config.num_hash_layers {
            Some(load_i64_matrix_row(
                &experts.index,
                &format!("layers.{}.ffn.gate.tid2eid", self.layer),
                token,
                config.vocab_size,
                config.num_experts_per_tok,
            )?)
        } else {
            None
        };
        let routes = route_sqrt_softplus(
            &logits,
            self.correction_bias.as_deref(),
            hash.as_deref(),
            config.num_experts_per_tok,
            config.routed_scaling_factor as f32,
        )?;
        // The standalone reference executes local experts in ascending expert-ID order after
        // routing. Keep the public route order intact, but match that FP32 accumulation order.
        let mut execution_order = routes
            .iter()
            .map(|route| (route.expert, route.weight))
            .collect::<Vec<_>>();
        execution_order.sort_by_key(|&(expert, _)| expert);
        let expert_ids = execution_order
            .iter()
            .map(|&(expert, _)| expert)
            .collect::<Vec<_>>();
        // Every release expert applies the same E4M3 activation transform to this source vector.
        // Share one materialization across six routed experts plus the resident shared expert.
        let mx_input = prepare_mx_activation(input)?;
        Ok(PreparedDeepRoutes {
            routes,
            execution_order,
            expert_ids,
            mx_input,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch(
        &self,
        inputs: &[&[f32]],
        tokens: &[usize],
        positions: &[usize],
        flow_id: u64,
        experts: &mut DeepExpertStore,
        config: &DeepseekV4Config,
    ) -> Result<DeepBatchForward, DeepseekRuntimeError> {
        let batch = inputs.len();
        if !(2..=PREFILL_ROUTE_WINDOW).contains(&batch)
            || tokens.len() != batch
            || positions.len() != batch
        {
            return Err(DeepseekRuntimeError::Invalid(format!(
                "MoE prefill batch must contain two through {PREFILL_ROUTE_WINDOW} aligned tokens"
            )));
        }
        let prepared = inputs
            .iter()
            .zip(tokens)
            .map(|(&input, &token)| self.prepare_routes(input, token, experts, config))
            .collect::<Result<Vec<_>, _>>()?;
        let flattened = prepared
            .iter()
            .flat_map(|prepared| prepared.expert_ids.iter().copied())
            .collect::<Vec<_>>();
        let union_fits = experts.can_acquire_batch(self.layer, &flattened);
        let stream_grouped = !union_fits && experts.can_stream_grouped_batch(self.layer);
        if !union_fits && !stream_grouped {
            return Ok(DeepBatchForward::Fallback(prepared));
        }

        let expert_ids = prepared
            .iter()
            .map(|prepared| prepared.expert_ids.clone())
            .collect::<Vec<_>>();
        let route_weights = prepared
            .iter()
            .map(|prepared| {
                prepared
                    .execution_order
                    .iter()
                    .map(|&(_, weight)| weight)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mx_inputs = prepared
            .iter()
            .map(|prepared| prepared.mx_input.as_slice())
            .collect::<Vec<_>>();
        let traces = positions
            .iter()
            .zip(tokens)
            .map(|(&position, &token)| ExpertTraceContext {
                position,
                token,
                flow_id,
            })
            .collect::<Vec<_>>();
        let mut shared_input = Vec::with_capacity(batch.saturating_mul(config.hidden_size));
        let mut shared_mx_input = Vec::with_capacity(shared_input.capacity());
        for (&input, &mx_input) in inputs.iter().zip(&mx_inputs) {
            shared_input.extend_from_slice(input);
            shared_mx_input.extend_from_slice(mx_input);
        }
        let profile_context = capture_context();
        let (computed, shared) = install(|| {
            rayon::join(
                || {
                    profile_context.enter(|| {
                        if stream_grouped {
                            experts.acquire_and_forward_token_batch_streamed(
                                self.layer,
                                &expert_ids,
                                &route_weights,
                                inputs,
                                &mx_inputs,
                                &traces,
                            )
                        } else {
                            experts.acquire_and_forward_token_batch(
                                self.layer,
                                &expert_ids,
                                &route_weights,
                                inputs,
                                &mx_inputs,
                                &traces,
                            )
                        }
                    })
                },
                || {
                    profile_context.enter(|| {
                        self.shared.forward_with_mx_input_batch(
                            &shared_input,
                            &shared_mx_input,
                            None,
                            batch,
                        )
                    })
                },
            )
        });
        let computed = computed?;
        let shared = shared?;
        prepared
            .into_iter()
            .zip(computed)
            .zip(shared)
            .map(|((prepared, computed), shared)| {
                let mut output = vec![0.0f32; config.hidden_size];
                for computed in computed {
                    accumulate_routed_output(&mut output, computed);
                }
                for (output, shared) in output.iter_mut().zip(shared) {
                    *output += shared;
                }
                round_to_bf16_in_place(&mut output)?;
                Ok((output, prepared.routes))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(DeepBatchForward::Batched)
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_routed_expert(
    output: &mut [f32],
    expert: &DeepExpert,
    input: &[f32],
    mx_input: &[f32],
    route_weight: f32,
    expert_id: usize,
    cache_hit: bool,
    layer: usize,
    trace: ExpertTraceContext,
) -> Result<(), DeepseekRuntimeError> {
    let computed = compute_routed_expert(
        expert,
        input,
        mx_input,
        route_weight,
        expert_id,
        cache_hit,
        layer,
        trace,
    )?;
    accumulate_routed_output(output, computed);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_routed_expert(
    expert: &DeepExpert,
    input: &[f32],
    mx_input: &[f32],
    route_weight: f32,
    expert_id: usize,
    cache_hit: bool,
    layer: usize,
    trace: ExpertTraceContext,
) -> Result<Vec<f32>, DeepseekRuntimeError> {
    let mut profile = span(ProfileStage::DeepseekExpertCompute);
    profile.set_token(trace.position, trace.token);
    profile.set_layer_id(layer);
    profile.set_expert_id(expert_id);
    profile.set_cache_hit(cache_hit);
    profile.set_flow_id(trace.flow_id);
    expert.forward_with_mx_input(input, mx_input, Some(route_weight))
}

fn accumulate_routed_output(output: &mut [f32], computed: Vec<f32>) {
    for (output, expert) in output.iter_mut().zip(computed) {
        *output += expert;
    }
}

#[derive(Debug)]
struct DeepExpert {
    w1: WeightMatrix,
    w2: WeightMatrix,
    w3: WeightMatrix,
    limit: f32,
}

#[derive(Debug)]
struct DeepExpertBuffers(Vec<(ReadBuffer, ReadBuffer)>);

impl DeepExpert {
    fn load(
        index: &TensorIndex,
        prefix: &str,
        config: &DeepseekV4Config,
        maximum_bytes: u64,
    ) -> Result<(Self, u64, u64), DeepseekRuntimeError> {
        let names = [
            (
                format!("{prefix}.w1.weight"),
                config.moe_intermediate_size,
                config.hidden_size,
            ),
            (
                format!("{prefix}.w2.weight"),
                config.hidden_size,
                config.moe_intermediate_size,
            ),
            (
                format!("{prefix}.w3.weight"),
                config.moe_intermediate_size,
                config.hidden_size,
            ),
        ];
        let bytes = names.iter().try_fold(0u64, |sum, (name, rows, cols)| {
            sum.checked_add(inspect_weight_matrix(index, name, *rows, *cols)?.resident_bytes)
                .ok_or_else(|| {
                    DeepseekRuntimeError::Invalid("expert byte count overflows".to_owned())
                })
        })?;
        let read_bytes = names.iter().try_fold(0u64, |sum, (name, _, _)| {
            sum.checked_add(weight_storage_bytes(index, name)?)
                .ok_or_else(|| {
                    DeepseekRuntimeError::Invalid("expert payload byte count overflows".to_owned())
                })
        })?;
        if bytes > maximum_bytes {
            return Err(DeepseekRuntimeError::Budget {
                component: "routed expert",
                required: bytes,
                maximum: maximum_bytes,
            });
        }
        let specifications = names
            .iter()
            .map(|(name, rows, cols)| (name.as_str(), *rows, *cols))
            .collect::<Vec<_>>();
        let mut matrices = load_weight_matrices(index, &specifications, maximum_bytes)?.into_iter();
        Ok((
            Self {
                w1: matrices.next().expect("three requested expert matrices"),
                w2: matrices.next().expect("three requested expert matrices"),
                w3: matrices.next().expect("three requested expert matrices"),
                limit: config.swiglu_limit as f32,
            },
            bytes,
            read_bytes,
        ))
    }

    fn load_reusing(
        index: &TensorIndex,
        prefix: &str,
        config: &DeepseekV4Config,
        maximum_bytes: u64,
        reuse: Option<DeepExpertBuffers>,
    ) -> Result<(Self, u64, u64), DeepseekRuntimeError> {
        let Some(DeepExpertBuffers(reuse)) = reuse else {
            return Self::load(index, prefix, config, maximum_bytes);
        };
        let specifications = [
            (
                format!("{prefix}.w1.weight"),
                config.moe_intermediate_size,
                config.hidden_size,
            ),
            (
                format!("{prefix}.w2.weight"),
                config.hidden_size,
                config.moe_intermediate_size,
            ),
            (
                format!("{prefix}.w3.weight"),
                config.moe_intermediate_size,
                config.hidden_size,
            ),
        ];
        if reuse.len() != specifications.len()
            || specifications.iter().any(|(name, rows, cols)| {
                !matches!(
                    inspect_weight_matrix(index, name, *rows, *cols),
                    Ok(layout)
                        if layout.format == (WeightFormat::MxFp4E2M1 { group_size: 32 })
                )
            })
        {
            return Self::load(index, prefix, config, maximum_bytes);
        }
        let bytes = specifications
            .iter()
            .try_fold(0u64, |sum, (name, rows, cols)| {
                sum.checked_add(inspect_weight_matrix(index, name, *rows, *cols)?.resident_bytes)
                    .ok_or_else(|| {
                        DeepseekRuntimeError::Invalid("expert byte count overflows".to_owned())
                    })
            })?;
        if bytes > maximum_bytes {
            return Err(DeepseekRuntimeError::Budget {
                component: "routed expert",
                required: bytes,
                maximum: maximum_bytes,
            });
        }
        let read_bytes = specifications.iter().try_fold(0u64, |sum, (name, _, _)| {
            sum.checked_add(weight_storage_bytes(index, name)?)
                .ok_or_else(|| {
                    DeepseekRuntimeError::Invalid("expert payload byte count overflows".to_owned())
                })
        })?;
        let mut matrices =
            load_mxfp4_matrices_reusing(index, &specifications, reuse, read_bytes)?.into_iter();
        Ok((
            Self {
                w1: matrices.next().expect("three reused expert matrices"),
                w2: matrices.next().expect("three reused expert matrices"),
                w3: matrices.next().expect("three reused expert matrices"),
                limit: config.swiglu_limit as f32,
            },
            bytes,
            read_bytes,
        ))
    }

    fn load_layer_reusing(
        index: &TensorIndex,
        prefix: &str,
        config: &DeepseekV4Config,
        reuse: &mut DecoderLayerBuffers,
        prevalidated: bool,
    ) -> Result<Self, DeepseekRuntimeError> {
        Ok(Self {
            w1: matrix_reusing(
                index,
                &format!("{prefix}.w1.weight"),
                config.moe_intermediate_size,
                config.hidden_size,
                reuse,
                prevalidated,
            )?,
            w2: matrix_reusing(
                index,
                &format!("{prefix}.w2.weight"),
                config.hidden_size,
                config.moe_intermediate_size,
                reuse,
                prevalidated,
            )?,
            w3: matrix_reusing(
                index,
                &format!("{prefix}.w3.weight"),
                config.moe_intermediate_size,
                config.hidden_size,
                reuse,
                prevalidated,
            )?,
            limit: config.swiglu_limit as f32,
        })
    }

    fn append_native_buffers(self, buffers: &mut DecoderLayerBuffers) {
        for weight in [self.w1, self.w2, self.w3] {
            buffers.push_weight(weight);
        }
    }

    fn into_mxfp4_buffers(self) -> Option<DeepExpertBuffers> {
        let Self { w1, w2, w3, .. } = self;
        [w1, w2, w3]
            .into_iter()
            .map(|weight| match weight {
                WeightMatrix::MxFp4(matrix) => Some(matrix.into_e2m1_buffers()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(DeepExpertBuffers)
    }

    fn forward_with_mx_input(
        &self,
        input: &[f32],
        mx_input: &[f32],
        route_weight: Option<f32>,
    ) -> Result<Vec<f32>, DeepseekRuntimeError> {
        let gate = linear_with_mx_activation(&self.w1, input, mx_input)?;
        let up = linear_with_mx_activation(&self.w3, input, mx_input)?;
        let mut activated = bounded_swiglu(&gate, &up, self.limit)?;
        if let Some(weight) = route_weight {
            if !weight.is_finite() || weight < 0.0 {
                return Err(DeepseekRuntimeError::Invalid(
                    "routed expert weight must be finite and non-negative".to_owned(),
                ));
            }
            for value in &mut activated {
                *value *= weight;
            }
        }
        // DeepSeek applies the route weight before converting the activation back to BF16 and
        // before the MX-quantized down projection. Moving it after `w2` changes the model.
        round_to_bf16_in_place(&mut activated)?;
        Ok(linear(&self.w2, &activated)?)
    }

    fn forward_with_mx_input_batch(
        &self,
        input: &[f32],
        mx_input: &[f32],
        route_weights: Option<&[f32]>,
        batch: usize,
    ) -> Result<Vec<Vec<f32>>, DeepseekRuntimeError> {
        if batch == 0
            || input.len() != batch.saturating_mul(self.w1.cols())
            || mx_input.len() != input.len()
            || route_weights.is_some_and(|weights| weights.len() != batch)
        {
            return Err(DeepseekRuntimeError::Invalid(
                "expert batch has invalid input or route-weight geometry".to_owned(),
            ));
        }
        let gate = linear_with_mx_activation_batch(&self.w1, input, mx_input, batch)?;
        let up = linear_with_mx_activation_batch(&self.w3, input, mx_input, batch)?;
        let intermediate = self.w1.rows();
        let mut activated = Vec::with_capacity(batch.saturating_mul(intermediate));
        for token in 0..batch {
            let start = token * intermediate;
            let end = start + intermediate;
            let mut token_activated =
                bounded_swiglu(&gate[start..end], &up[start..end], self.limit)?;
            if let Some(route_weights) = route_weights {
                let weight = route_weights[token];
                if !weight.is_finite() || weight < 0.0 {
                    return Err(DeepseekRuntimeError::Invalid(
                        "routed expert weight must be finite and non-negative".to_owned(),
                    ));
                }
                for value in &mut token_activated {
                    *value *= weight;
                }
            }
            round_to_bf16_in_place(&mut token_activated)?;
            activated.extend(token_activated);
        }
        let mut activated_mx = Vec::with_capacity(activated.len());
        for token in activated.chunks_exact(intermediate) {
            activated_mx.extend(prepare_mx_activation(token)?);
        }
        let output = linear_with_mx_activation_batch(&self.w2, &activated, &activated_mx, batch)?;
        Ok(output
            .chunks_exact(self.w2.rows())
            .map(<[f32]>::to_vec)
            .collect())
    }

    #[cfg(test)]
    fn forward(
        &self,
        input: &[f32],
        route_weight: Option<f32>,
    ) -> Result<Vec<f32>, DeepseekRuntimeError> {
        let mx_input = prepare_mx_activation(input)?;
        self.forward_with_mx_input(input, &mx_input, route_weight)
    }
}

#[derive(Debug)]
struct DeepExpertStore {
    index: Arc<TensorIndex>,
    config: DeepseekV4Config,
    maximum_bytes: u64,
    slots_per_layer: usize,
    cache: LayerLruCache<Arc<DeepExpert>>,
    // Layer-major prefill temporarily lends globally unused cache slots to the active layer.
    // Buffers trimmed when the steady-state capacity is restored remain inside that same global
    // slot budget and are consumed by later-layer misses before new allocations are made.
    recycled: Vec<DeepExpertBuffers>,
    telemetry: ExpertTelemetry,
}

#[derive(Debug, Clone, Copy)]
struct ExpertTraceContext {
    position: usize,
    token: usize,
    flow_id: u64,
}

type LoadedDeepExpert = (Arc<DeepExpert>, u64, u64);

impl DeepExpertStore {
    fn new(
        index: Arc<TensorIndex>,
        config: &DeepseekV4Config,
        slots_per_layer: usize,
        maximum_bytes: u64,
    ) -> Result<Self, DeepseekRuntimeError> {
        if slots_per_layer > config.n_routed_experts || maximum_bytes == 0 {
            return Err(DeepseekRuntimeError::Invalid(
                "expert cache geometry is invalid".to_owned(),
            ));
        }
        Ok(Self {
            index,
            config: config.clone(),
            maximum_bytes,
            slots_per_layer,
            cache: LayerLruCache::new(config.num_hidden_layers, slots_per_layer),
            recycled: Vec::new(),
            telemetry: ExpertTelemetry::default(),
        })
    }

    fn acquire(
        &mut self,
        layer: usize,
        expert: usize,
        trace_context: ExpertTraceContext,
    ) -> Result<Arc<DeepExpert>, DeepseekRuntimeError> {
        if layer >= self.config.num_hidden_layers || expert >= self.config.n_routed_experts {
            return Err(DeepseekRuntimeError::Invalid(
                "expert request is outside configured geometry".to_owned(),
            ));
        }
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let (expert, evicted) = self.cache.access_with_evicted(
            &mut self.telemetry,
            layer,
            expert,
            || {
                let mut profile = span(ProfileStage::DeepseekExpertLoad);
                profile.set_token(trace_context.position, trace_context.token);
                profile.set_layer_id(layer);
                profile.set_expert_id(expert);
                profile.set_cache_hit(false);
                profile.set_flow_id(trace_context.flow_id);
                let loaded = DeepExpert::load(
                    &index,
                    &format!("layers.{layer}.ffn.experts.{expert}"),
                    &config,
                    maximum,
                )?;
                profile.add_logical_bytes(loaded.2);
                Ok((Arc::new(loaded.0), loaded.1, loaded.2))
            },
            |expert| Ok::<_, DeepseekRuntimeError>(Arc::clone(expert)),
        )?;
        self.recycle_evicted(evicted);
        Ok(expert)
    }

    /// Starts one cache-miss load without mutating LRU state. The caller installs the completed
    /// value immediately before its logical access, preserving deterministic eviction order while
    /// one reserved transient expert overlaps the previous expert's compute.
    fn preload(
        &mut self,
        layer: usize,
        expert: usize,
        trace_context: ExpertTraceContext,
    ) -> IoTask<Result<LoadedDeepExpert, DeepseekRuntimeError>> {
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let reuse = self.recycled.pop();
        let profile_context = capture_context();
        spawn_io(move || {
            profile_context.enter(|| {
                let mut profile = span(ProfileStage::DeepseekExpertLoad);
                profile.set_token(trace_context.position, trace_context.token);
                profile.set_layer_id(layer);
                profile.set_expert_id(expert);
                profile.set_cache_hit(false);
                profile.set_flow_id(trace_context.flow_id);
                let loaded = DeepExpert::load_reusing(
                    &index,
                    &format!("layers.{layer}.ffn.experts.{expert}"),
                    &config,
                    maximum,
                    reuse,
                );
                if let Ok((_, _, read_bytes)) = &loaded {
                    profile.add_logical_bytes(*read_bytes);
                }
                loaded.map(|(value, bytes, read_bytes)| (Arc::new(value), bytes, read_bytes))
            })
        })
    }

    fn install_preloaded(
        &mut self,
        layer: usize,
        expert: usize,
        loaded: LoadedDeepExpert,
    ) -> Result<Arc<DeepExpert>, DeepseekRuntimeError> {
        let mut loaded = Some(loaded);
        let (expert, evicted) = self.cache.access_with_evicted(
            &mut self.telemetry,
            layer,
            expert,
            || {
                loaded.take().ok_or_else(|| {
                    DeepseekRuntimeError::Invalid(
                        "a prefetched expert was consumed before cache insertion".to_owned(),
                    )
                })
            },
            |expert| Ok::<_, DeepseekRuntimeError>(Arc::clone(expert)),
        )?;
        self.recycle_evicted(evicted);
        Ok(expert)
    }

    fn recycle_evicted(&mut self, evicted: Option<Arc<DeepExpert>>) {
        if let Some(evicted) = evicted {
            if let Ok(evicted) = Arc::try_unwrap(evicted) {
                if let Some(buffers) = evicted.into_mxfp4_buffers() {
                    self.recycled.push(buffers);
                }
            }
        }
    }

    fn begin_prefill_layer(&mut self, layer: usize) -> usize {
        self.cache
            .lend_unused_capacity(layer, self.config.n_routed_experts)
    }

    fn finish_prefill_layer(&mut self, layer: usize, steady_capacity: usize) {
        let evicted =
            self.cache
                .restore_layer_capacity(&mut self.telemetry, layer, steady_capacity);
        for expert in evicted {
            self.recycle_evicted(Some(expert));
        }
    }

    fn can_acquire_batch(&self, layer: usize, experts: &[usize]) -> bool {
        self.cache.can_insert_without_eviction(layer, experts)
    }

    /// Simulate the ordered route accesses and evict their inevitable victims up front, exposing
    /// every miss to the bounded I/O pool together. Logical accesses are still replayed in route
    /// order by `acquire_batch_while`, retaining exact telemetry and LRU state.
    fn prepare_ordered_batch(&mut self, layer: usize, experts: &[usize]) -> bool {
        let Some(evicted) = self
            .cache
            .prepare_ordered_batch(&mut self.telemetry, layer, experts)
        else {
            return false;
        };
        for expert in evicted {
            self.recycle_evicted(Some(expert));
        }
        true
    }

    fn acquire_batch_while<R>(
        &mut self,
        layer: usize,
        experts: &[usize],
        trace_context: ExpertTraceContext,
        while_loading: impl FnOnce() -> Result<R, DeepseekRuntimeError>,
    ) -> Result<(Vec<Arc<DeepExpert>>, R), DeepseekRuntimeError> {
        if layer >= self.config.num_hidden_layers
            || experts
                .iter()
                .any(|&expert| expert >= self.config.n_routed_experts)
        {
            return Err(DeepseekRuntimeError::Invalid(
                "expert batch request is outside configured geometry".to_owned(),
            ));
        }
        if !self.cache.can_insert_without_eviction(layer, experts) {
            return Err(DeepseekRuntimeError::Invalid(
                "expert batch requires free cache slots; use sequential execution".to_owned(),
            ));
        }

        let mut missing = Vec::new();
        for &expert in experts {
            if !self.cache.contains(layer, expert) && !missing.contains(&expert) {
                missing.push(expert);
            }
        }
        let maximum = self.maximum_bytes;
        let reuse = (0..missing.len())
            .map(|_| self.recycled.pop())
            .collect::<Vec<_>>();
        let profile_context = capture_context();
        let pending = missing
            .into_iter()
            .zip(reuse)
            .map(|(expert, reuse)| {
                let index = Arc::clone(&self.index);
                let config = self.config.clone();
                let profile_context = profile_context.clone();
                spawn_io(move || {
                    profile_context.enter(|| {
                        let mut profile = span(ProfileStage::DeepseekExpertLoad);
                        profile.set_token(trace_context.position, trace_context.token);
                        profile.set_layer_id(layer);
                        profile.set_expert_id(expert);
                        profile.set_cache_hit(false);
                        profile.set_flow_id(trace_context.flow_id);
                        let loaded = DeepExpert::load_reusing(
                            &index,
                            &format!("layers.{layer}.ffn.experts.{expert}"),
                            &config,
                            maximum,
                            reuse,
                        );
                        if let Ok((_, _, read_bytes)) = &loaded {
                            profile.add_logical_bytes(*read_bytes);
                        }
                        loaded.map(|(value, bytes, read_bytes)| {
                            (expert, (Arc::new(value), bytes, read_bytes))
                        })
                    })
                })
            })
            .collect::<Vec<_>>();
        let computed = while_loading();
        // Drain every I/O task before propagating either error so token rollback cannot race with
        // outstanding reads and allocations.
        let loaded = pending.into_iter().map(IoTask::join).collect::<Vec<_>>();
        let computed = computed?;
        let mut preloaded = HashMap::with_capacity(loaded.len());
        for loaded in loaded {
            let (expert, value) = loaded?;
            preloaded.insert(expert, value);
        }

        let experts = experts
            .iter()
            .map(|&expert| {
                self.cache.access(
                    &mut self.telemetry,
                    layer,
                    expert,
                    || {
                        preloaded.remove(&expert).ok_or_else(|| {
                            DeepseekRuntimeError::Invalid(
                                "parallel expert preload omitted a cache miss".to_owned(),
                            )
                        })
                    },
                    |expert| Ok(Arc::clone(expert)),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((experts, computed))
    }

    fn contains(&self, layer: usize, expert: usize) -> bool {
        self.cache.contains(layer, expert)
    }

    /// A fresh layer can group an oversized prefill window by expert without keeping the entire
    /// union resident. Processing bounded chunks in last-use order leaves the same final LRU hot
    /// set as token-major execution while repeatedly recycling the same cache-sized buffer set.
    fn can_stream_grouped_batch(&self, layer: usize) -> bool {
        self.slots_per_layer > 0 && self.cache.layer_is_empty(layer)
    }

    #[allow(clippy::too_many_arguments)]
    fn acquire_and_forward_token_batch_streamed(
        &mut self,
        layer: usize,
        expert_ids_by_token: &[Vec<usize>],
        route_weights_by_token: &[Vec<f32>],
        inputs: &[&[f32]],
        mx_inputs: &[&[f32]],
        traces: &[ExpertTraceContext],
    ) -> Result<Vec<Vec<Vec<f32>>>, DeepseekRuntimeError> {
        let batch = expert_ids_by_token.len();
        if !(2..=PREFILL_ROUTE_WINDOW).contains(&batch)
            || route_weights_by_token.len() != batch
            || inputs.len() != batch
            || mx_inputs.len() != batch
            || traces.len() != batch
            || layer >= self.config.num_hidden_layers
            || !self.can_stream_grouped_batch(layer)
            || expert_ids_by_token
                .iter()
                .zip(route_weights_by_token)
                .any(|(experts, weights)| experts.is_empty() || experts.len() != weights.len())
            || expert_ids_by_token
                .iter()
                .flatten()
                .any(|&expert| expert >= self.config.n_routed_experts)
            || inputs.iter().chain(mx_inputs).any(|input| {
                input.len() != self.config.hidden_size
                    || input.iter().any(|value| !value.is_finite())
            })
        {
            return Err(DeepseekRuntimeError::Invalid(
                "streamed expert token batch has invalid geometry, state, or activation values"
                    .to_owned(),
            ));
        }

        let mut group_indices = HashMap::new();
        let mut groups = Vec::<(usize, Vec<(usize, usize)>)>::new();
        for (token, experts) in expert_ids_by_token.iter().enumerate() {
            for (route, &expert) in experts.iter().enumerate() {
                let group = match group_indices.get(&expert) {
                    Some(&group) => group,
                    None => {
                        let group = groups.len();
                        groups.push((expert, Vec::new()));
                        group_indices.insert(expert, group);
                        group
                    }
                };
                groups[group].1.push((token, route));
            }
        }
        // Starting from an empty layer, one insertion per distinct expert in ascending last-use
        // order produces exactly the final hot-key set and recency order of token-major LRU. It
        // also lets every repeated route reuse the one loaded value even if the full union is
        // larger than the cache.
        groups.sort_by_key(|&(expert, ref positions)| {
            let &(token, route) = positions
                .last()
                .expect("every grouped expert has at least one route");
            (token, route, expert)
        });

        let mut computed = expert_ids_by_token
            .iter()
            .map(|experts| (0..experts.len()).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let capacity = self.cache.layer_capacity(layer);
        if capacity == 0 || groups.is_empty() {
            return Err(DeepseekRuntimeError::Invalid(
                "streamed expert batch requires a non-empty cache and route set".to_owned(),
            ));
        }
        // Make the final chunk exactly one cacheful (unless the entire union is smaller, which is
        // handled by the regular batch path). The cache therefore finishes with the last C
        // distinct experts in last-use order, exactly matching token-major LRU.
        let first_chunk = match groups.len() % capacity {
            0 => capacity,
            remainder => remainder,
        };
        let mut start = 0;
        let mut chunk_len = first_chunk;
        while start < groups.len() {
            let end = (start + chunk_len).min(groups.len());
            let chunk = &groups[start..end];
            let expert_ids = chunk.iter().map(|(expert, _)| *expert).collect::<Vec<_>>();
            let trace = traces[chunk[0].1[0].0];
            let (loaded, ()) = self.acquire_batch_while(layer, &expert_ids, trace, || Ok(()))?;
            let config = &self.config;
            let profile_context = capture_context();
            let completed = install(|| {
                loaded
                    .par_iter()
                    .zip(chunk.par_iter())
                    .map(|(expert, &(expert_id, ref positions))| {
                        let mut input =
                            Vec::with_capacity(positions.len().saturating_mul(config.hidden_size));
                        let mut mx_input = Vec::with_capacity(input.capacity());
                        let mut route_weights = Vec::with_capacity(positions.len());
                        for &(token, route) in positions {
                            input.extend_from_slice(inputs[token]);
                            mx_input.extend_from_slice(mx_inputs[token]);
                            route_weights.push(route_weights_by_token[token][route]);
                        }
                        let trace = traces[positions[0].0];
                        profile_context.enter(|| {
                            let mut profile = span(ProfileStage::DeepseekExpertCompute);
                            profile.set_token_position(trace.position);
                            profile.set_layer_id(layer);
                            profile.set_expert_id(expert_id);
                            profile.set_cache_hit(false);
                            profile.set_flow_id(trace.flow_id);
                            profile.set_batch_tokens(positions.len());
                            expert
                                .forward_with_mx_input_batch(
                                    &input,
                                    &mx_input,
                                    Some(&route_weights),
                                    positions.len(),
                                )
                                .map(|values| (positions.clone(), values))
                        })
                    })
                    .collect::<Vec<_>>()
            });
            drop(loaded);
            for result in completed {
                let (positions, values) = result?;
                if values.len() != positions.len() {
                    return Err(DeepseekRuntimeError::Invalid(
                        "streamed expert token batch returned the wrong number of rows".to_owned(),
                    ));
                }
                // The first occurrence caused the physical miss above. Remaining occurrences are
                // served by the grouped resident value and therefore count as real cache hits.
                self.telemetry.hits = self
                    .telemetry
                    .hits
                    .saturating_add(positions.len().saturating_sub(1) as u64);
                for ((token, route), value) in positions.into_iter().zip(values) {
                    computed[token][route] = Some(value);
                }
            }
            if end < groups.len() {
                let drained = self.cache.drain_layer(&mut self.telemetry, layer);
                for expert in drained {
                    self.recycle_evicted(Some(expert));
                }
            }
            start = end;
            chunk_len = capacity;
        }

        Ok(computed
            .into_iter()
            .map(|token| {
                token
                    .into_iter()
                    .map(|value| value.expect("every routed expert has one streamed batch result"))
                    .collect()
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn acquire_and_forward_token_batch(
        &mut self,
        layer: usize,
        expert_ids_by_token: &[Vec<usize>],
        route_weights_by_token: &[Vec<f32>],
        inputs: &[&[f32]],
        mx_inputs: &[&[f32]],
        traces: &[ExpertTraceContext],
    ) -> Result<Vec<Vec<Vec<f32>>>, DeepseekRuntimeError> {
        let batch = expert_ids_by_token.len();
        if !(2..=PREFILL_ROUTE_WINDOW).contains(&batch)
            || route_weights_by_token.len() != batch
            || inputs.len() != batch
            || mx_inputs.len() != batch
            || traces.len() != batch
            || layer >= self.config.num_hidden_layers
            || expert_ids_by_token
                .iter()
                .zip(route_weights_by_token)
                .any(|(experts, weights)| experts.len() != weights.len())
            || expert_ids_by_token
                .iter()
                .flatten()
                .any(|&expert| expert >= self.config.n_routed_experts)
            || inputs.iter().chain(mx_inputs).any(|input| {
                input.len() != self.config.hidden_size
                    || input.iter().any(|value| !value.is_finite())
            })
        {
            return Err(DeepseekRuntimeError::Invalid(
                "expert token batch has invalid geometry or activation values".to_owned(),
            ));
        }
        let flattened = expert_ids_by_token
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        if !self.cache.can_insert_without_eviction(layer, &flattened) {
            return Err(DeepseekRuntimeError::Invalid(
                "expert token batch would change cache eviction semantics".to_owned(),
            ));
        }

        let mut group_indices = HashMap::new();
        let mut groups = Vec::<(usize, Vec<(usize, usize)>)>::new();
        for (token, experts) in expert_ids_by_token.iter().enumerate() {
            for (route, &expert) in experts.iter().enumerate() {
                let group = match group_indices.get(&expert) {
                    Some(&group) => group,
                    None => {
                        let group = groups.len();
                        groups.push((expert, Vec::new()));
                        group_indices.insert(expert, group);
                        group
                    }
                };
                groups[group].1.push((token, route));
            }
        }
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let profile_context = capture_context();
        let work = groups
            .into_iter()
            .map(|(expert, positions)| {
                let cached = self.cache.peek(layer, expert).map(Arc::clone);
                let pending = if cached.is_none() {
                    let index = Arc::clone(&index);
                    let config = config.clone();
                    let reuse = self.recycled.pop();
                    let first_trace = traces[positions[0].0];
                    let batch_tokens = positions.len();
                    let profile_context = profile_context.clone();
                    Some(spawn_io(move || {
                        profile_context.enter(|| {
                            let mut profile = span(ProfileStage::DeepseekExpertLoad);
                            profile.set_token_position(first_trace.position);
                            profile.set_layer_id(layer);
                            profile.set_expert_id(expert);
                            profile.set_cache_hit(false);
                            profile.set_flow_id(first_trace.flow_id);
                            profile.set_batch_tokens(batch_tokens);
                            let loaded = DeepExpert::load_reusing(
                                &index,
                                &format!("layers.{layer}.ffn.experts.{expert}"),
                                &config,
                                maximum,
                                reuse,
                            );
                            if let Ok((_, _, read_bytes)) = &loaded {
                                profile.add_logical_bytes(*read_bytes);
                            }
                            loaded.map(|(value, resident, read_bytes)| {
                                (Arc::new(value), resident, read_bytes)
                            })
                        })
                    }))
                } else {
                    None
                };
                (expert, positions, cached, pending)
            })
            .collect::<Vec<_>>();
        let completed = install(|| {
            work.into_par_iter()
                .map(|(expert, positions, cached, pending)| {
                    let mut input =
                        Vec::with_capacity(positions.len().saturating_mul(config.hidden_size));
                    let mut mx_input = Vec::with_capacity(input.capacity());
                    let mut route_weights = Vec::with_capacity(positions.len());
                    for &(token, route) in &positions {
                        input.extend_from_slice(inputs[token]);
                        mx_input.extend_from_slice(mx_inputs[token]);
                        route_weights.push(route_weights_by_token[token][route]);
                    }
                    let first_trace = traces[positions[0].0];
                    match cached {
                        Some(value) => {
                            let computed = profile_context.enter(|| {
                                let mut profile = span(ProfileStage::DeepseekExpertCompute);
                                profile.set_token_position(first_trace.position);
                                profile.set_layer_id(layer);
                                profile.set_expert_id(expert);
                                profile.set_cache_hit(true);
                                profile.set_flow_id(first_trace.flow_id);
                                profile.set_batch_tokens(positions.len());
                                value.forward_with_mx_input_batch(
                                    &input,
                                    &mx_input,
                                    Some(&route_weights),
                                    positions.len(),
                                )
                            });
                            Ok::<_, DeepseekRuntimeError>((expert, positions, None, computed))
                        }
                        None => {
                            let (value, resident, read_bytes) = pending
                                .expect("each uncached expert has one pending load")
                                .join()?;
                            let computed = profile_context.enter(|| {
                                let mut compute_profile = span(ProfileStage::DeepseekExpertCompute);
                                compute_profile.set_token_position(first_trace.position);
                                compute_profile.set_layer_id(layer);
                                compute_profile.set_expert_id(expert);
                                compute_profile.set_cache_hit(false);
                                compute_profile.set_flow_id(first_trace.flow_id);
                                compute_profile.set_batch_tokens(positions.len());
                                value.forward_with_mx_input_batch(
                                    &input,
                                    &mx_input,
                                    Some(&route_weights),
                                    positions.len(),
                                )
                            });
                            Ok((
                                expert,
                                positions,
                                Some((value, resident, read_bytes)),
                                computed,
                            ))
                        }
                    }
                })
                .collect::<Vec<_>>()
        });

        let mut preloaded = HashMap::new();
        let mut computed_groups = Vec::with_capacity(completed.len());
        for completed in completed {
            let (expert, positions, loaded, computed) = completed?;
            if let Some(loaded) = loaded {
                preloaded.insert(expert, loaded);
            }
            computed_groups.push((positions, computed?));
        }

        // Replay token-major, expert-sorted logical accesses after all fallible work succeeds.
        // The borrowed capacity guarantees no eviction, preserving LRU and telemetry exactly.
        for expert_ids in expert_ids_by_token {
            for &expert in expert_ids {
                let (_, evicted) = self.cache.access_with_evicted(
                    &mut self.telemetry,
                    layer,
                    expert,
                    || {
                        preloaded.remove(&expert).ok_or_else(|| {
                            DeepseekRuntimeError::Invalid(
                                "batched expert preload omitted a cache miss".to_owned(),
                            )
                        })
                    },
                    |_| Ok::<_, DeepseekRuntimeError>(()),
                )?;
                debug_assert!(evicted.is_none());
                if let Some(evicted) = evicted {
                    self.recycle_evicted(Some(evicted));
                }
            }
        }

        let mut computed = expert_ids_by_token
            .iter()
            .map(|experts| (0..experts.len()).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for (positions, values) in computed_groups {
            if values.len() != positions.len() {
                return Err(DeepseekRuntimeError::Invalid(
                    "expert token batch returned the wrong number of rows".to_owned(),
                ));
            }
            for ((token, route), value) in positions.into_iter().zip(values) {
                computed[token][route] = Some(value);
            }
        }
        Ok(computed
            .into_iter()
            .map(|token| {
                token
                    .into_iter()
                    .map(|value| value.expect("every routed expert has one batch result"))
                    .collect()
            })
            .collect())
    }
}

fn layer_flow_id(position: usize, layer: usize) -> u64 {
    let position = u64::try_from(position)
        .unwrap_or(u64::MAX)
        .min(u32::MAX.into());
    let layer = u64::try_from(layer)
        .unwrap_or(u64::MAX)
        .min(u32::MAX.into());
    (position << 32) | layer
}

fn load_compressor(
    index: &TensorIndex,
    config: &DeepseekV4Config,
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    rotate: bool,
    norm: &Arc<[f32]>,
) -> Result<CompressorWeights, DeepseekRuntimeError> {
    let coefficient = if ratio == 4 { 2 } else { 1 };
    Ok(CompressorWeights::new(
        ratio,
        head_dim,
        config.qk_rope_head_dim,
        rotate,
        matrix(
            index,
            &format!("{prefix}.ape"),
            ratio,
            coefficient * head_dim,
        )?,
        matrix(
            index,
            &format!("{prefix}.wkv.weight"),
            coefficient * head_dim,
            config.hidden_size,
        )?,
        matrix(
            index,
            &format!("{prefix}.wgate.weight"),
            coefficient * head_dim,
            config.hidden_size,
        )?,
        Arc::clone(norm),
        config.rms_norm_eps as f32,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn load_compressor_reusing(
    index: &TensorIndex,
    config: &DeepseekV4Config,
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    rotate: bool,
    norm: &Arc<[f32]>,
    reuse: &mut DecoderLayerBuffers,
    prevalidated: bool,
) -> Result<CompressorWeights, DeepseekRuntimeError> {
    let coefficient = if ratio == 4 { 2 } else { 1 };
    Ok(CompressorWeights::new(
        ratio,
        head_dim,
        config.qk_rope_head_dim,
        rotate,
        matrix_reusing(
            index,
            &format!("{prefix}.ape"),
            ratio,
            coefficient * head_dim,
            reuse,
            prevalidated,
        )?,
        matrix_reusing(
            index,
            &format!("{prefix}.wkv.weight"),
            coefficient * head_dim,
            config.hidden_size,
            reuse,
            prevalidated,
        )?,
        matrix_reusing(
            index,
            &format!("{prefix}.wgate.weight"),
            coefficient * head_dim,
            config.hidden_size,
            reuse,
            prevalidated,
        )?,
        Arc::clone(norm),
        config.rms_norm_eps as f32,
    )?)
}

fn matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<WeightMatrix, DeepseekRuntimeError> {
    Ok(load_weight_matrix(index, name, rows, columns, u64::MAX)?)
}

fn matrix_reusing(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
    reuse: &mut DecoderLayerBuffers,
    prevalidated: bool,
) -> Result<WeightMatrix, DeepseekRuntimeError> {
    if reuse.0.is_empty() && !prevalidated {
        return matrix(index, name, rows, columns);
    }
    let layout = inspect_weight_matrix(index, name, rows, columns)?;
    let (block_rows, block_cols, group_size) = match layout.format {
        WeightFormat::MxFp8E4M3 {
            block_rows,
            block_cols,
        } => (Some(block_rows), Some(block_cols), None),
        WeightFormat::MxFp4E2M1 { group_size } => (None, None, Some(group_size)),
        _ => return matrix(index, name, rows, columns),
    };
    let scale_name = name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.scale"))
        .unwrap_or_else(|| format!("{name}.scale"));
    let values_len = usize::try_from(index.require(name)?.data_len).map_err(|_| {
        DeepseekRuntimeError::Invalid(format!("weight tensor {name:?} exceeds addressable memory"))
    })?;
    let scales_len = usize::try_from(index.require(&scale_name)?.data_len).map_err(|_| {
        DeepseekRuntimeError::Invalid(format!(
            "weight scale tensor {scale_name:?} exceeds addressable memory"
        ))
    })?;
    let reusable = reuse.take_matching(values_len, scales_len);
    if reusable.is_none() && !prevalidated {
        return matrix(index, name, rows, columns);
    }

    // Safetensor sidecars are often far from their payloads. Preserve the normal loader's
    // physical ordering while overwriting the two exact-size allocations in place.
    let names = [name.to_owned(), scale_name];
    let tensors = [index.require(&names[0])?, index.require(&names[1])?];
    let mut inputs = match reusable {
        Some((values, scales)) => [Some(values.into_vec()), Some(scales.into_vec())],
        None => [None, None],
    };
    let mut outputs = [None, None];
    let mut order = [0usize, 1usize];
    order.sort_by(|&left, &right| {
        tensors[left]
            .shard
            .cmp(&tensors[right].shard)
            .then_with(|| tensors[left].data_offset.cmp(&tensors[right].data_offset))
    });
    for index_in_pair in order {
        let len = usize::try_from(tensors[index_in_pair].data_len).map_err(|_| {
            DeepseekRuntimeError::Invalid(format!(
                "weight tensor {:?} exceeds addressable memory",
                names[index_in_pair]
            ))
        })?;
        outputs[index_in_pair] = Some(match inputs[index_in_pair].take() {
            Some(reuse) => index.read_range_reusing(&names[index_in_pair], 0, len, reuse)?,
            None => index.read_range(&names[index_in_pair], 0, len)?,
        });
    }
    let values: ReadBuffer = outputs[0].take().expect("weight payload was read").into();
    let scales: ReadBuffer = outputs[1].take().expect("weight scale was read").into();
    match (block_rows, block_cols, group_size) {
        (Some(block_rows), Some(block_cols), None) => {
            let matrix = if prevalidated {
                MxFp8Matrix::from_read_buffers_prevalidated(
                    rows, columns, block_rows, block_cols, values, scales,
                )
            } else {
                MxFp8Matrix::from_read_buffers(
                    rows, columns, block_rows, block_cols, values, scales,
                )
            };
            matrix
                .map(WeightMatrix::MxFp8)
                .map_err(WeightError::from)
                .map_err(DeepseekRuntimeError::from)
        }
        (None, None, Some(group_size)) => {
            let matrix = if prevalidated {
                MxFp4Matrix::from_read_buffers_prevalidated(
                    rows, columns, group_size, values, scales,
                )
            } else {
                MxFp4Matrix::from_read_buffers(rows, columns, group_size, values, scales)
            };
            matrix
                .map(WeightMatrix::MxFp4)
                .map_err(WeightError::from)
                .map_err(DeepseekRuntimeError::from)
        }
        _ => unreachable!("native MX layout has one recognized geometry"),
    }
}

fn load_mxfp4_matrices_reusing(
    index: &TensorIndex,
    specifications: &[(String, usize, usize)],
    reuse: Vec<(ReadBuffer, ReadBuffer)>,
    maximum_bytes: u64,
) -> Result<Vec<WeightMatrix>, DeepseekRuntimeError> {
    let mut names = Vec::with_capacity(specifications.len() * 2);
    let mut buffers = Vec::with_capacity(specifications.len() * 2);
    for ((name, rows, cols), (values, scales)) in specifications.iter().zip(reuse) {
        let layout = inspect_weight_matrix(index, name, *rows, *cols)?;
        if layout.format != (WeightFormat::MxFp4E2M1 { group_size: 32 }) {
            return Err(DeepseekRuntimeError::Invalid(format!(
                "reusable expert tensor {name:?} is not native group-32 MXFP4"
            )));
        }
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .ok_or_else(|| {
                DeepseekRuntimeError::Invalid(format!(
                    "reusable MXFP4 tensor {name:?} has no .weight suffix"
                ))
            })?;
        names.push(name.clone());
        buffers.push(values.into_vec());
        names.push(scale_name);
        buffers.push(scales.into_vec());
    }

    // V4 stores all three scale tensors together, followed much later by all three payloads.
    // Reading this batch in physical order avoids repeatedly seeking between those two regions.
    // Existing experts originate as Vec-backed reads, so keeping that allocation class also avoids
    // glibc retaining a displaced generation of large heap buffers under tight RAM.
    let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
    let payloads = index.read_tensors_bounded_reusing(&name_refs, buffers, maximum_bytes)?;
    let mut payloads = payloads.into_iter();
    specifications
        .iter()
        .map(|(_, rows, cols)| {
            let values = payloads
                .next()
                .expect("each reusable MXFP4 matrix has a value payload")
                .into();
            let scales = payloads
                .next()
                .expect("each reusable MXFP4 matrix has a scale payload")
                .into();
            MxFp4Matrix::from_read_buffers(*rows, *cols, 32, values, scales)
                .map(WeightMatrix::MxFp4)
                .map_err(WeightError::from)
                .map_err(DeepseekRuntimeError::from)
        })
        .collect()
}

fn bf16_rms_norm(
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, DeepseekRuntimeError> {
    let mut output = rms_norm(input, weight, eps)?;
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

fn weight_storage_bytes(index: &TensorIndex, name: &str) -> Result<u64, DeepseekRuntimeError> {
    let mut bytes = index.require(name)?.data_len;
    let native_scale = name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.scale"));
    for sidecar in [Some(format!("{name}.qs")), native_scale]
        .into_iter()
        .flatten()
    {
        if let Some(tensor) = index.get(&sidecar) {
            bytes = bytes.checked_add(tensor.data_len).ok_or_else(|| {
                DeepseekRuntimeError::Invalid("weight payload bytes overflow".to_owned())
            })?;
        }
    }
    Ok(bytes)
}

fn window_indices(position: usize, window: usize) -> Vec<usize> {
    if position >= window - 1 {
        let current = position % window;
        ((current + 1)..window).chain(0..=current).collect()
    } else {
        (0..=position).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{generate, GenerationConfig};
    use crate::model::DenseMatrix;
    use crate::profiling::{ProfileSession, ProfileStage};
    use serde::Deserialize;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[derive(Debug, Deserialize)]
    struct TinyOracle {
        format_version: usize,
        model_family: String,
        token: usize,
        geometry: OracleGeometry,
        weights: BTreeMap<String, OracleTensor>,
        expected: serde_json::Value,
    }

    #[derive(Debug, Deserialize)]
    struct OracleGeometry {
        hidden_size: usize,
        hc_mult: usize,
        mix_count: usize,
        vocab_size: usize,
        num_attention_heads: usize,
        head_dim: usize,
        q_lora_rank: usize,
        o_groups: usize,
        o_lora_rank: usize,
        moe_intermediate_size: usize,
        n_routed_experts: usize,
        num_experts_per_tok: usize,
        routed_scaling_factor: f64,
        rms_norm_eps: f64,
        hc_eps: f64,
        hc_sinkhorn_iters: usize,
        swiglu_limit: f64,
    }

    #[derive(Debug, Deserialize)]
    struct OracleTensor {
        dtype: String,
        shape: Vec<usize>,
        values: Vec<serde_json::Value>,
    }

    fn tiny_oracle() -> TinyOracle {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/deepseek_v4_tiny_oracle.json"
        ))
        .unwrap()
    }

    fn oracle_vector(oracle: &TinyOracle, name: &str) -> Vec<f32> {
        oracle.expected[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect()
    }

    fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 2.0e-5_f32.max(expected.abs() * 2.0e-5);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name}[{index}]: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn write_oracle_checkpoint(path: &Path, oracle: &TinyOracle) {
        fs::create_dir_all(path).unwrap();
        let geometry = &oracle.geometry;
        assert_eq!(
            geometry.mix_count,
            (2 + geometry.hc_mult) * geometry.hc_mult
        );
        let config = DeepseekV4Config {
            model_type: "deepseek_v4".to_owned(),
            architectures: vec!["DeepseekV4ForCausalLM".to_owned()],
            hidden_size: geometry.hidden_size,
            num_hidden_layers: 1,
            num_attention_heads: geometry.num_attention_heads,
            num_key_value_heads: 1,
            head_dim: geometry.head_dim,
            q_lora_rank: geometry.q_lora_rank,
            qk_rope_head_dim: geometry.head_dim,
            o_groups: geometry.o_groups,
            o_lora_rank: geometry.o_lora_rank,
            sliding_window: 4,
            compress_ratios: vec![0],
            compress_rope_theta: 160000.0,
            index_n_heads: 1,
            index_head_dim: geometry.head_dim,
            index_topk: 4,
            vocab_size: geometry.vocab_size,
            bos_token_id: 0,
            eos_token_id: (geometry.vocab_size - 1) as u32,
            max_position_embeddings: 16,
            rms_norm_eps: geometry.rms_norm_eps,
            rope_theta: 10000.0,
            rope_scaling: super::super::config::RopeScaling {
                beta_fast: 32,
                beta_slow: 1,
                factor: 40.0,
                original_max_position_embeddings: 4,
                rope_type: "yarn".to_owned(),
            },
            hc_mult: geometry.hc_mult,
            hc_sinkhorn_iters: geometry.hc_sinkhorn_iters,
            hc_eps: geometry.hc_eps,
            moe_intermediate_size: geometry.moe_intermediate_size,
            n_routed_experts: geometry.n_routed_experts,
            n_shared_experts: 1,
            num_experts_per_tok: geometry.num_experts_per_tok,
            num_hash_layers: 1,
            norm_topk_prob: true,
            routed_scaling_factor: geometry.routed_scaling_factor,
            scoring_func: "sqrtsoftplus".to_owned(),
            topk_method: "noaux_tc".to_owned(),
            swiglu_limit: geometry.swiglu_limit,
            expert_dtype: "fp4".to_owned(),
            quantization_config: super::super::config::QuantizationConfig {
                activation_scheme: "dynamic".to_owned(),
                fmt: "e4m3".to_owned(),
                quant_method: "fp8".to_owned(),
                scale_fmt: "ue8m0".to_owned(),
                weight_block_size: [128, 128],
            },
            num_nextn_predict_layers: 0,
            dspark_block_size: 0,
            dspark_noise_token_id: 1,
            dspark_target_layer_ids: vec![],
            dspark_markov_rank: 2,
            tie_word_embeddings: false,
            attention_bias: false,
            hidden_act: "silu".to_owned(),
        };
        fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        let mut descriptors = BTreeMap::new();
        let mut payload = Vec::new();
        for (name, tensor) in &oracle.weights {
            assert_eq!(tensor.shape.iter().product::<usize>(), tensor.values.len());
            let start = payload.len();
            match tensor.dtype.as_str() {
                "F32" => payload.extend(
                    tensor
                        .values
                        .iter()
                        .flat_map(|value| (value.as_f64().unwrap() as f32).to_le_bytes()),
                ),
                "BF16" => payload.extend(tensor.values.iter().flat_map(|value| {
                    (((value.as_f64().unwrap() as f32).to_bits() >> 16) as u16).to_le_bytes()
                })),
                "I64" => payload.extend(
                    tensor
                        .values
                        .iter()
                        .flat_map(|value| value.as_i64().unwrap().to_le_bytes()),
                ),
                dtype => panic!("unsupported oracle dtype {dtype:?}"),
            }
            descriptors.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": tensor.dtype, "shape": tensor.shape,
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let mut header = serde_json::to_vec(&descriptors).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        fs::write(path.join("model.safetensors"), file).unwrap();
    }

    fn fixture_dir() -> PathBuf {
        crate::test_support::temp_dir("urbilateria_deepseek_runtime")
    }

    fn add_f32(
        tensors: &mut BTreeMap<String, serde_json::Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let start = payload.len();
        payload.extend(values.into_iter().flat_map(f32::to_le_bytes));
        tensors.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]
            }),
        );
    }

    fn add_bf16(
        tensors: &mut BTreeMap<String, serde_json::Value>,
        payload: &mut Vec<u8>,
        name: &str,
        length: usize,
        value: f32,
    ) {
        let start = payload.len();
        let bits = (value.to_bits() >> 16) as u16;
        payload.extend((0..length).flat_map(|_| bits.to_le_bytes()));
        tensors.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": "BF16", "shape": [length], "data_offsets": [start, payload.len()]
            }),
        );
    }

    fn write_tiny_checkpoint(path: &Path) {
        fs::create_dir_all(path).unwrap();
        let config = r#"{
            "model_type":"deepseek_v4", "architectures":["DeepseekV4ForCausalLM"],
            "hidden_size":2, "num_hidden_layers":1, "num_attention_heads":1,
            "num_key_value_heads":1, "head_dim":2, "q_lora_rank":2,
            "qk_rope_head_dim":2, "o_groups":1, "o_lora_rank":2,
            "sliding_window":4, "compress_ratios":[0], "compress_rope_theta":160000.0,
            "index_n_heads":1, "index_head_dim":2, "index_topk":4,
            "vocab_size":8, "bos_token_id":0, "eos_token_id":7,
            "max_position_embeddings":8, "rms_norm_eps":0.000001,
            "rope_theta":10000.0,
            "rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":40.0,
                "original_max_position_embeddings":4,"type":"yarn"},
            "hc_mult":1, "hc_sinkhorn_iters":2, "hc_eps":0.000001,
            "moe_intermediate_size":2, "n_routed_experts":1,
            "n_shared_experts":1, "num_experts_per_tok":1, "num_hash_layers":0,
            "norm_topk_prob":true, "routed_scaling_factor":1.0,
            "scoring_func":"sqrtsoftplus", "topk_method":"noaux_tc",
            "swiglu_limit":10.0, "expert_dtype":"fp4",
            "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
                "quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},
            "num_nextn_predict_layers":0, "dspark_block_size":0,
            "dspark_noise_token_id":2, "dspark_target_layer_ids":[],
            "dspark_markov_rank":2, "tie_word_embeddings":false,
            "attention_bias":false, "hidden_act":"silu"
        }"#;
        fs::write(path.join("config.json"), config).unwrap();

        let mut tensors = BTreeMap::new();
        let mut payload = Vec::new();
        add_f32(
            &mut tensors,
            &mut payload,
            "embed.weight",
            &[8, 2],
            (0..16).map(|value| value as f32 * 0.01).collect(),
        );
        for sublayer in ["attn", "ffn"] {
            add_f32(
                &mut tensors,
                &mut payload,
                &format!("layers.0.hc_{sublayer}_fn"),
                &[3, 2],
                vec![0.0; 6],
            );
            add_f32(
                &mut tensors,
                &mut payload,
                &format!("layers.0.hc_{sublayer}_base"),
                &[3],
                vec![0.0; 3],
            );
            add_f32(
                &mut tensors,
                &mut payload,
                &format!("layers.0.hc_{sublayer}_scale"),
                &[3],
                vec![0.0; 3],
            );
        }
        for name in [
            "layers.0.attn_norm.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.attn.q_norm.weight",
            "layers.0.attn.kv_norm.weight",
            "norm.weight",
        ] {
            add_bf16(&mut tensors, &mut payload, name, 2, 1.0);
        }
        add_f32(
            &mut tensors,
            &mut payload,
            "layers.0.attn.attn_sink",
            &[1],
            vec![0.0],
        );
        for (name, shape) in [
            ("layers.0.attn.wq_a.weight", [2, 2]),
            ("layers.0.attn.wq_b.weight", [2, 2]),
            ("layers.0.attn.wkv.weight", [2, 2]),
            ("layers.0.attn.wo_a.weight", [2, 2]),
            ("layers.0.attn.wo_b.weight", [2, 2]),
            ("layers.0.ffn.gate.weight", [1, 2]),
        ] {
            add_f32(
                &mut tensors,
                &mut payload,
                name,
                &shape,
                vec![0.0; shape.iter().product()],
            );
        }
        add_f32(
            &mut tensors,
            &mut payload,
            "layers.0.ffn.gate.bias",
            &[1],
            vec![0.0],
        );
        for expert in ["shared_experts", "experts.0"] {
            for (projection, shape) in [("w1", [2, 2]), ("w2", [2, 2]), ("w3", [2, 2])] {
                add_f32(
                    &mut tensors,
                    &mut payload,
                    &format!("layers.0.ffn.{expert}.{projection}.weight"),
                    &shape,
                    vec![0.0; 4],
                );
            }
        }
        add_f32(
            &mut tensors,
            &mut payload,
            "hc_head_fn",
            &[1, 2],
            vec![0.0; 2],
        );
        add_f32(&mut tensors, &mut payload, "hc_head_base", &[1], vec![0.0]);
        add_f32(&mut tensors, &mut payload, "hc_head_scale", &[1], vec![0.0]);
        add_f32(
            &mut tensors,
            &mut payload,
            "head.weight",
            &[8, 2],
            vec![0.0; 16],
        );
        let mut header = serde_json::to_vec(&tensors).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        fs::write(path.join("model.safetensors"), file).unwrap();
    }

    fn options() -> RuntimeLoadOptions {
        RuntimeLoadOptions {
            resident_budget_bytes: 1 << 20,
            expert_cache_budget_bytes: 1 << 20,
            kv_cache_budget_bytes: 1 << 20,
            expert_slots_per_layer: 1,
            maximum_expert_bytes: 1 << 20,
            context_limit: 4,
        }
    }

    #[test]
    fn window_indices_match_the_official_ring_order() {
        assert_eq!(window_indices(0, 4), vec![0]);
        assert_eq!(window_indices(2, 4), vec![0, 1, 2]);
        assert_eq!(window_indices(3, 4), vec![0, 1, 2, 3]);
        assert_eq!(window_indices(4, 4), vec![1, 2, 3, 0]);
    }

    #[test]
    fn routed_weight_is_applied_before_the_down_projection_boundary() {
        let identity =
            || WeightMatrix::F32(DenseMatrix::new(1, 1, vec![1.0]).expect("valid scalar matrix"));
        let expert = DeepExpert {
            w1: identity(),
            w2: identity(),
            w3: identity(),
            limit: 10.0,
        };
        let weight = 0.3;
        let weighted = expert.forward(&[1.0], Some(weight)).unwrap();

        let mut expected = bounded_swiglu(&[1.0], &[1.0], 10.0).unwrap();
        expected[0] *= weight;
        round_to_bf16_in_place(&mut expected).unwrap();
        assert_eq!(weighted, expected);

        let unweighted = expert.forward(&[1.0], None).unwrap();
        let mut incorrectly_weighted_after_w2 = vec![unweighted[0] * weight];
        round_to_bf16_in_place(&mut incorrectly_weighted_after_w2).unwrap();
        assert_ne!(weighted, incorrectly_weighted_after_w2);
    }

    #[test]
    #[ignore = "requires the 156 GiB release checkpoint and executes one real base layer"]
    fn real_checkpoint_layer_zero_executes_a_complete_forward() {
        let directory = std::env::var_os("URB_DEEPSEEK_V4_DIR")
            .map(PathBuf::from)
            .expect("set URB_DEEPSEEK_V4_DIR to run the real layer forward");
        let config = DeepseekV4Config::load(&directory).unwrap();
        let index = Arc::new(TensorIndex::open(&directory).unwrap());
        let embedding = load_reference_matrix_row(
            &index,
            "embed.weight",
            0,
            config.vocab_size,
            config.hidden_size,
        )
        .unwrap();
        let hidden = embedding.repeat(config.hc_mult);
        let vectors = DecoderLayerVectors::load(&index, &config, 0).unwrap();
        let layer = DecoderLayer::load(&index, &config, 0, &vectors).unwrap();
        let mut attention = AttentionState::new(&config, 0).unwrap();
        let mut experts = DeepExpertStore::new(Arc::clone(&index), &config, 0, u64::MAX).unwrap();

        let (output, routes) = layer
            .forward(&hidden, 0, 0, &mut attention, &mut experts, &config, 0)
            .unwrap();

        assert_eq!(output.len(), config.hc_mult * config.hidden_size);
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            [254, 222, 245, 200, 53, 35]
        );
        assert!((routes.iter().map(|route| route.weight).sum::<f32>() - 1.5).abs() < 1e-5);
        assert_eq!(attention.next_position, 1);
    }

    #[test]
    fn independent_tiny_oracle_matches_one_hash_moe_layer() {
        let oracle = tiny_oracle();
        assert_eq!(oracle.format_version, 1);
        assert_eq!(oracle.model_family, "deepseek_v4");
        let directory = fixture_dir();
        write_oracle_checkpoint(&directory, &oracle);

        let requirements = DeepseekRuntimeModel::inspect_requirements(&directory, 4, 1).unwrap();
        assert_eq!(requirements.required_tensor_count, 36);
        assert_eq!(requirements.unexpected_tensor_count, 0);
        let model = DeepseekRuntimeModel::load(&directory, options()).unwrap();
        let mut state = model.new_state().unwrap();
        let step = model
            .forward_token(oracle.token as u32, &mut state)
            .unwrap();

        assert_close("logits", &step.logits, &oracle_vector(&oracle, "logits"));
        let expected_routes = oracle.expected["routes"].as_array().unwrap();
        assert_eq!(step.routes_by_layer.len(), 1);
        assert_eq!(step.routes_by_layer[0].len(), expected_routes.len());
        for (actual, expected) in step.routes_by_layer[0].iter().zip(expected_routes) {
            assert_eq!(actual.expert, expected["expert"].as_u64().unwrap() as usize);
            assert_close(
                "route weight",
                &[actual.weight],
                &[expected["weight"].as_f64().unwrap() as f32],
            );
            assert_close(
                "route selection score",
                &[actual.selection_score],
                &[expected["selection_score"].as_f64().unwrap() as f32],
            );
        }
        assert_eq!(state.position(), 1);
        assert_eq!(state.cached_f32_elements(), oracle.geometry.head_dim);
        assert_eq!(
            step.logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0,
            oracle.expected["argmax"].as_u64().unwrap() as usize
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn layerwise_prefill_matches_sequential_tokens_and_loads_each_layer_once() {
        let oracle = tiny_oracle();
        let directory = fixture_dir();
        write_oracle_checkpoint(&directory, &oracle);
        const PROMPT_TOKENS: usize = 13;
        let mut load_options = options();
        load_options.context_limit = PROMPT_TOKENS + 1;
        let model = DeepseekRuntimeModel::load(&directory, load_options).unwrap();
        let prompt = [oracle.token as u32; PROMPT_TOKENS];

        let mut sequential_state = model.new_state().unwrap();
        let mut sequential = None;
        for &token in &prompt {
            sequential = Some(model.forward_token(token, &mut sequential_state).unwrap());
        }
        let sequential = sequential.unwrap();

        let mut batched_state = model.new_state().unwrap();
        let batched = model.prefill_tokens(&prompt, &mut batched_state).unwrap();
        assert_eq!(batched.logits, sequential.logits);
        assert_eq!(batched.routes_by_layer, sequential.routes_by_layer);
        assert_eq!(batched_state.position(), sequential_state.position());
        assert_eq!(
            batched_state.cached_f32_elements(),
            sequential_state.cached_f32_elements()
        );
        let batched_telemetry = batched_state.expert_telemetry().clone();
        let sequential_telemetry = sequential_state.expert_telemetry().clone();
        assert_eq!(
            batched_telemetry.accesses(),
            sequential_telemetry.accesses()
        );
        assert!(batched_telemetry.hits > sequential_telemetry.hits);
        assert!(batched_telemetry.misses < sequential_telemetry.misses);
        assert!(batched_telemetry.evictions < sequential_telemetry.evictions);
        assert!(batched_telemetry.bytes_read < sequential_telemetry.bytes_read);
        assert_eq!(
            batched_telemetry.resident_experts,
            sequential_telemetry.resident_experts
        );
        assert_eq!(
            batched_telemetry.resident_bytes,
            sequential_telemetry.resident_bytes
        );

        let next_sequential = model
            .forward_token(oracle.token as u32, &mut sequential_state)
            .unwrap();
        let next_batched = model
            .forward_token(oracle.token as u32, &mut batched_state)
            .unwrap();
        assert_eq!(next_batched, next_sequential);
        let next_sequential_telemetry = sequential_state.expert_telemetry();
        let next_batched_telemetry = batched_state.expert_telemetry();
        assert_eq!(
            next_batched_telemetry.hits - batched_telemetry.hits,
            next_sequential_telemetry.hits - sequential_telemetry.hits
        );
        assert_eq!(
            next_batched_telemetry.misses - batched_telemetry.misses,
            next_sequential_telemetry.misses - sequential_telemetry.misses
        );
        assert_eq!(
            next_batched_telemetry.bytes_read - batched_telemetry.bytes_read,
            next_sequential_telemetry.bytes_read - sequential_telemetry.bytes_read
        );

        let mut profiled_state = model.new_state().unwrap();
        let profile = ProfileSession::start_with_threads_and_trace(None, true);
        let logits = CausalDecoder::prefill(&model, &prompt, &mut profiled_state).unwrap();
        let _decode = model
            .forward_token(oracle.token as u32, &mut profiled_state)
            .unwrap();
        let report = profile.finish();
        assert_eq!(logits, sequential.logits);
        assert_eq!(report.stage(ProfileStage::Prefill).unwrap().calls, 1);
        assert_eq!(
            report.stage(ProfileStage::DeepseekLayerLoad).unwrap().calls,
            model.config().num_hidden_layers as u64
        );
        assert_eq!(
            report.stage(ProfileStage::DeepseekLayer).unwrap().calls,
            (model.config().num_hidden_layers * (prompt.len() + 1)) as u64
        );
        assert_eq!(report.stage(ProfileStage::DeepseekLmHead).unwrap().calls, 2);
        assert!(report
            .stage(ProfileStage::StreamedMatrixReadDecode)
            .is_none());
        let trace: serde_json::Value = serde_json::from_slice(
            &report
                .chrome_trace_json_pretty()
                .unwrap()
                .expect("trace collection was enabled"),
        )
        .unwrap();
        let events = trace["traceEvents"].as_array().unwrap();
        let layer_load = events
            .iter()
            .find(|event| event["name"] == "deepseek.layer.load")
            .unwrap();
        assert_eq!(layer_load["args"]["token_position"], 0);
        assert_eq!(layer_load["args"]["layer_id"], 0);
        assert_eq!(layer_load["args"]["flow_id"], 0);
        assert_eq!(layer_load["args"]["batch_tokens"], prompt.len());
        let expert_compute = events
            .iter()
            .find(|event| event["name"] == "deepseek.expert.compute")
            .unwrap();
        assert_eq!(expert_compute["args"]["token_position"], 0);
        assert_eq!(expert_compute["args"]["layer_id"], 0);
        assert!(expert_compute["args"]["expert_id"].is_u64());
        assert!(expert_compute["args"]["cache_hit"].is_boolean());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn complete_tiny_checkpoint_runs_through_the_shared_generator() {
        let directory = fixture_dir();
        write_tiny_checkpoint(&directory);
        let requirements = DeepseekRuntimeModel::inspect_requirements(&directory, 4, 1).unwrap();
        assert_eq!(requirements.required_tensor_count, 30);
        let model = DeepseekRuntimeModel::load(&directory, options()).unwrap();
        let mut state = model.new_state().unwrap();
        let step = model.forward_token(1, &mut state).unwrap();
        assert_eq!(step.logits, vec![0.0; 8]);
        assert_eq!(state.position(), 1);
        assert_eq!(state.cached_f32_elements(), 2);
        let output = generate(&model, &[1], &GenerationConfig::greedy(2, vec![7]), |_| {}).unwrap();
        assert_eq!(output.generated_tokens, vec![0, 0]);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn decoder_reloads_share_the_model_resident_vector_payloads() {
        let directory = fixture_dir();
        write_tiny_checkpoint(&directory);
        let model = DeepseekRuntimeModel::load(&directory, options()).unwrap();
        let vectors = &model.decoder_vectors[0];
        let first = DecoderLayer::load(&model.index, &model.config, 0, vectors).unwrap();
        let second = DecoderLayer::load(&model.index, &model.config, 0, vectors).unwrap();

        assert!(Arc::ptr_eq(&first.attn_hc.base, &vectors.attn_hc.base));
        assert!(Arc::ptr_eq(&first.attn_hc.scale, &vectors.attn_hc.scale));
        assert!(Arc::ptr_eq(&first.ffn_hc.base, &vectors.ffn_hc.base));
        assert!(Arc::ptr_eq(&first.ffn_hc.scale, &vectors.ffn_hc.scale));
        assert!(Arc::ptr_eq(&first.attn_norm, &vectors.attn_norm));
        assert!(Arc::ptr_eq(&first.ffn_norm, &vectors.ffn_norm));
        assert!(Arc::ptr_eq(&first.attention.sink, &vectors.attention.sink));
        assert!(Arc::ptr_eq(
            &first.attention.q_norm,
            &vectors.attention.q_norm
        ));
        assert!(Arc::ptr_eq(
            &first.attention.kv_norm,
            &vectors.attention.kv_norm
        ));
        assert!(Arc::ptr_eq(
            first.moe.correction_bias.as_ref().unwrap(),
            vectors.correction_bias.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(&first.attn_norm, &second.attn_norm));
        assert!(Arc::ptr_eq(
            first.moe.correction_bias.as_ref().unwrap(),
            second.moe.correction_bias.as_ref().unwrap()
        ));

        fs::remove_dir_all(directory).unwrap();
    }
}
