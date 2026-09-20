//! Layer-streamed scalar correctness runtime for Tencent Hy4-preview-FP8.
//!
//! Hy4's DSA indexer selects `index_topk=2048` cached positions.  For contexts no longer than
//! that, every causal position is selected, so the exact sparse attention is the dense latent-MLA
//! equation used here.  This keeps the first executable implementation small while preserving the
//! trained attention sink and all Gated-MLA semantics.

use super::expert::{Hy4Expert, Hy4ExpertBuffers, Hy4ExpertError, MX_BLOCK};
use super::math::{
    identity_hyper_connection_head, identity_hyper_connection_post, identity_hyper_connection_pre,
    Hy4MathError, IdentityHyperConnectionMix,
};
use super::schema::{self, Hy4Requirements, SchemaError};
use super::Hy4Config;
use crate::config::ConfigError;
use crate::execution::{install, spawn_io};
use crate::generation::CausalDecoder;
use crate::math::{
    rms_norm, route_noaux_tc, silu, simulate_e4m3_activation, MathError, RouteChoice, RouteError,
};
use crate::model::{WeightError, WeightMatrix};
use crate::models::deepseek_v4::math::{paired_rope, round_to_bf16_in_place, DeepseekMathError};
use crate::profiling::{capture_context, span, ProfileSpan, ProfileStage};
use crate::runtime::cache::LayerLruCache;
use crate::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use crate::storage::{
    load_compact_bf16_matrix, load_reference_matrix_row, load_reference_vector, load_weight_matrix,
    streamed_reference_matvec_pipelined, SafetensorError, TensorIndex, TensorLoadError,
    WeightLoadError,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_HY4_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub enum Hy4RuntimeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Tensor(TensorLoadError),
    WeightLoad(WeightLoadError),
    Weight(WeightError),
    Math(MathError),
    Route(RouteError),
    DeepseekMath(DeepseekMathError),
    Hy4Math(Hy4MathError),
    Expert(Hy4ExpertError),
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

impl fmt::Display for Hy4RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Tensor(error) => error.fmt(formatter),
            Self::WeightLoad(error) => error.fmt(formatter),
            Self::Weight(error) => error.fmt(formatter),
            Self::Math(error) => error.fmt(formatter),
            Self::Route(error) => error.fmt(formatter),
            Self::DeepseekMath(error) => error.fmt(formatter),
            Self::Hy4Math(error) => error.fmt(formatter),
            Self::Expert(error) => error.fmt(formatter),
            Self::Schema(error) => error.fmt(formatter),
            Self::Invalid(reason) => write!(formatter, "invalid Hy4 runtime: {reason}"),
            Self::Budget {
                component,
                required,
                maximum,
            } => write!(
                formatter,
                "Hy4 {component} needs {required} bytes, authorized maximum is {maximum}"
            ),
            Self::TokenOutOfRange { token, vocabulary } => {
                write!(
                    formatter,
                    "token ID {token} is outside vocabulary 0..{vocabulary}"
                )
            }
            Self::ContextExhausted { position, limit } => {
                write!(
                    formatter,
                    "runtime position {position} reaches context limit {limit}"
                )
            }
        }
    }
}

impl std::error::Error for Hy4RuntimeError {}

type HiddenWithRoutes = (Vec<f32>, Vec<RouteChoice>);
type HiddenBatchWithRoutes = Vec<HiddenWithRoutes>;

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for Hy4RuntimeError {
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
from_error!(RouteError, Route);
from_error!(DeepseekMathError, DeepseekMath);
from_error!(Hy4MathError, Hy4Math);
from_error!(Hy4ExpertError, Expert);
from_error!(SchemaError, Schema);

#[derive(Debug, Clone, PartialEq)]
pub struct Hy4RuntimeStep {
    pub logits: Vec<f32>,
    pub routes_by_layer: Vec<Vec<RouteChoice>>,
}

#[derive(Debug)]
pub struct Hy4RuntimeModel {
    instance_id: u64,
    config: Hy4Config,
    index: Arc<TensorIndex>,
    final_norm: Vec<f32>,
    head_hc: HcHeadWeights,
    lm_head: Option<WeightMatrix>,
    context_limit: usize,
    expert_slots_per_layer: usize,
    expert_cache_budget_bytes: u64,
    maximum_expert_bytes: u64,
    cached_decoder_layers: usize,
}

impl Hy4RuntimeModel {
    pub fn inspect_requirements(
        model_dir: impl AsRef<Path>,
        context_limit: usize,
        expert_slots_per_layer: usize,
    ) -> Result<Hy4Requirements, Hy4RuntimeError> {
        let config = Hy4Config::load(model_dir.as_ref())?;
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
    ) -> Result<Self, Hy4RuntimeError> {
        if options.context_limit == 0
            || options.resident_budget_bytes == 0
            || options.kv_cache_budget_bytes == 0
            || options.maximum_expert_bytes == 0
        {
            return Err(Hy4RuntimeError::Invalid(
                "resident/KV/per-expert budgets and context limit must be non-zero".to_owned(),
            ));
        }
        let config = Hy4Config::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        let requirements = schema::inspect_requirements(
            &config,
            &index,
            options.context_limit,
            options.expert_slots_per_layer,
        )?;
        for (component, required, maximum) in [
            (
                "streamed resident layer peak",
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
                return Err(Hy4RuntimeError::Budget {
                    component,
                    required,
                    maximum,
                });
            }
        }
        let index = Arc::new(index);
        let cached_decoder_layers =
            requirements.cached_decoder_layers_for_resident_budget(options.resident_budget_bytes);
        let final_norm = load_reference_vector(&index, "model.norm.weight", config.hidden_size)?;
        let head_hc = HcHeadWeights::load(&index, &config)?;
        let lm_head = requirements
            .caches_lm_head_for_resident_budget(options.resident_budget_bytes)
            .then(|| {
                load_compact_bf16_matrix(
                    &index,
                    "lm_head.weight",
                    config.vocab_size,
                    config.hidden_size,
                    requirements.lm_head_bytes,
                )
            })
            .transpose()?;
        Ok(Self {
            instance_id: NEXT_HY4_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            config,
            index,
            final_norm,
            head_hc,
            lm_head,
            context_limit: options.context_limit,
            expert_slots_per_layer: options.expert_slots_per_layer,
            expert_cache_budget_bytes: options.expert_cache_budget_bytes,
            maximum_expert_bytes: options.maximum_expert_bytes,
            cached_decoder_layers,
        })
    }

    pub fn new_state(&self) -> Result<Hy4RuntimeState, Hy4RuntimeError> {
        Ok(Hy4RuntimeState {
            instance_id: self.instance_id,
            position: 0,
            attention: (0..self.config.num_hidden_layers)
                .map(|_| AttentionState::default())
                .collect(),
            experts: Hy4ExpertStore::new(
                Arc::clone(&self.index),
                &self.config,
                self.expert_slots_per_layer,
                self.expert_cache_budget_bytes,
                self.maximum_expert_bytes,
            )?,
            cached_layers: (0..self.config.num_hidden_layers).map(|_| None).collect(),
        })
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut Hy4RuntimeState,
    ) -> Result<Hy4RuntimeStep, Hy4RuntimeError> {
        self.validate_state_and_tokens(&[token], state)?;
        let position = state.position;
        state.experts.direct_io = state.cached_layers.iter().all(Option::is_some);
        let checkpoint = state.attention.clone();
        match self.forward_token_inner(token as usize, position, state) {
            Ok(step) => {
                state.position += 1;
                Ok(step)
            }
            Err(error) => {
                state.attention = checkpoint;
                Err(error)
            }
        }
    }

    /// Loads each 770B decoder layer once for the entire prompt.
    pub fn prefill_tokens(
        &self,
        tokens: &[u32],
        state: &mut Hy4RuntimeState,
    ) -> Result<Hy4RuntimeStep, Hy4RuntimeError> {
        if tokens.is_empty() {
            return Err(Hy4RuntimeError::Invalid(
                "prefill requires at least one token".to_owned(),
            ));
        }
        self.validate_state_and_tokens(tokens, state)?;
        let end = state
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| Hy4RuntimeError::Invalid("prefill position overflows".to_owned()))?;
        if end > self.context_limit {
            return Err(Hy4RuntimeError::ContextExhausted {
                position: end - 1,
                limit: self.context_limit,
            });
        }
        let start = state.position;
        let checkpoint = state.attention.clone();
        let indices = tokens
            .iter()
            .map(|&token| token as usize)
            .collect::<Vec<_>>();
        match self.prefill_tokens_inner(&indices, start, state) {
            Ok(step) => {
                state.position = end;
                Ok(step)
            }
            Err(error) => {
                state.attention = checkpoint;
                Err(error)
            }
        }
    }

    fn validate_state_and_tokens(
        &self,
        tokens: &[u32],
        state: &Hy4RuntimeState,
    ) -> Result<(), Hy4RuntimeError> {
        if state.instance_id != self.instance_id {
            return Err(Hy4RuntimeError::Invalid(
                "state belongs to another runtime instance".to_owned(),
            ));
        }
        if let Some(&token) = tokens
            .iter()
            .find(|&&token| token as usize >= self.config.vocab_size)
        {
            return Err(Hy4RuntimeError::TokenOutOfRange {
                token,
                vocabulary: self.config.vocab_size,
            });
        }
        if state.position >= self.context_limit {
            return Err(Hy4RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        Ok(())
    }

    fn forward_token_inner(
        &self,
        token: usize,
        position: usize,
        state: &mut Hy4RuntimeState,
    ) -> Result<Hy4RuntimeStep, Hy4RuntimeError> {
        let mut hidden = self.load_hidden(token)?;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        let mut layer = self.cached_or_load_decoder_layer(state, 0)?;
        for layer_id in 0..self.config.num_hidden_layers {
            let next_layer_id = layer_id + 1;
            let cached_next = (next_layer_id < self.config.num_hidden_layers)
                .then(|| state.cached_layers[next_layer_id].clone())
                .flatten();
            let profile_context = capture_context();
            let ((next, routes), prefetched) = std::thread::scope(|scope| {
                let loader = (next_layer_id < self.config.num_hidden_layers
                    && cached_next.is_none())
                .then(|| {
                    scope.spawn(|| {
                        profile_context
                            .enter(|| load_decoder_layer(&self.index, &self.config, next_layer_id))
                    })
                });
                let forward = layer.forward(
                    &hidden,
                    position,
                    &mut state.attention[layer_id],
                    &mut state.experts,
                    &self.config,
                );
                let prefetched = match cached_next {
                    Some(layer) => Some(layer),
                    None => join_layer_loader(loader)?,
                };
                Ok::<_, Hy4RuntimeError>((forward?, prefetched))
            })?;
            hidden = next;
            routes_by_layer.push(routes);
            if let Some(next_layer) = prefetched {
                self.retain_decoder_layer(state, next_layer_id, &next_layer);
                layer = next_layer;
            }
        }
        self.finish_step(hidden, routes_by_layer)
    }

    fn prefill_tokens_inner(
        &self,
        tokens: &[usize],
        start_position: usize,
        state: &mut Hy4RuntimeState,
    ) -> Result<Hy4RuntimeStep, Hy4RuntimeError> {
        let mut hidden_by_token = tokens
            .iter()
            .map(|&token| self.load_hidden(token))
            .collect::<Result<Vec<_>, _>>()?;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        let mut layer = self.cached_or_load_decoder_layer(state, 0)?;
        for layer_id in 0..self.config.num_hidden_layers {
            let steady_expert_capacity = state.experts.begin_prefill_layer(layer_id);
            let next_layer_id = layer_id + 1;
            let cached_next = (next_layer_id < self.config.num_hidden_layers)
                .then(|| state.cached_layers[next_layer_id].clone())
                .flatten();
            let profile_context = capture_context();
            let layer_result = std::thread::scope(|scope| {
                let loader = (next_layer_id < self.config.num_hidden_layers
                    && cached_next.is_none())
                .then(|| {
                    scope.spawn(|| {
                        profile_context
                            .enter(|| load_decoder_layer(&self.index, &self.config, next_layer_id))
                    })
                });
                let forward = (|| {
                    let inputs = std::mem::take(&mut hidden_by_token);
                    let token_count = inputs.len();
                    let attention = &mut state.attention[layer_id];
                    let experts = &mut state.experts;
                    let (sender, receiver) = std::sync::mpsc::channel();
                    let layer_ref = layer.as_ref();
                    let config = &self.config;
                    let producer_context = capture_context();
                    let consumer_context = producer_context.clone();
                    let (produced, consumed) = install(|| {
                        rayon::join(
                            move || {
                                producer_context.enter(|| {
                                    for (offset, input) in inputs.into_iter().enumerate() {
                                        let prepared = layer_ref.prepare_feed_forward(
                                            &input,
                                            start_position + offset,
                                            attention,
                                            config,
                                            offset != 0,
                                        )?;
                                        if sender.send(prepared).is_err() {
                                            return Ok::<_, Hy4RuntimeError>(());
                                        }
                                    }
                                    Ok(())
                                })
                            },
                            move || {
                                consumer_context.enter(|| {
                                    let mut outputs = Vec::with_capacity(token_count);
                                    let mut final_routes = Vec::new();
                                    if token_count != 0 {
                                        let prepared = receiver.recv().map_err(|_| {
                                            Hy4RuntimeError::Invalid(
                                                "prefill feed-forward producer stopped early"
                                                    .to_owned(),
                                            )
                                        })?;
                                        let (next, routes) = layer_ref
                                            .finish_feed_forward(prepared, experts, config)?;
                                        outputs.push(next);
                                        final_routes = routes;
                                    }
                                    let mut remaining = token_count.saturating_sub(1);
                                    while remaining != 0 {
                                        let batch = remaining.min(18);
                                        let mut prepared = Vec::with_capacity(batch);
                                        for _ in 0..batch {
                                            prepared.push(receiver.recv().map_err(|_| {
                                                Hy4RuntimeError::Invalid(
                                                    "prefill feed-forward producer stopped early"
                                                        .to_owned(),
                                                )
                                            })?);
                                        }
                                        for (next, routes) in layer_ref
                                            .finish_feed_forward_batch(prepared, experts, config)?
                                        {
                                            outputs.push(next);
                                            final_routes = routes;
                                        }
                                        remaining -= batch;
                                    }
                                    Ok::<_, Hy4RuntimeError>((outputs, final_routes))
                                })
                            },
                        )
                    });
                    produced?;
                    let (outputs, final_routes) = consumed?;
                    hidden_by_token = outputs;
                    Ok::<_, Hy4RuntimeError>(final_routes)
                })();
                let prefetched = match cached_next {
                    Some(layer) => Some(layer),
                    None => join_layer_loader(loader)?,
                };
                Ok::<_, Hy4RuntimeError>((forward?, prefetched))
            });
            state
                .experts
                .finish_prefill_layer(layer_id, steady_expert_capacity);
            let (final_routes, prefetched) = layer_result?;
            routes_by_layer.push(final_routes);
            if let Some(next_layer) = prefetched {
                self.retain_decoder_layer(state, next_layer_id, &next_layer);
                layer = next_layer;
            }
        }
        self.finish_step(
            hidden_by_token
                .pop()
                .expect("non-empty prefill has a final token"),
            routes_by_layer,
        )
    }

    fn load_hidden(&self, token: usize) -> Result<Vec<f32>, Hy4RuntimeError> {
        let embedding = load_reference_matrix_row(
            &self.index,
            "model.embed_tokens.weight",
            token,
            self.config.vocab_size,
            self.config.hidden_size,
        )?;
        let mut hidden = Vec::with_capacity(self.config.hc_mult * self.config.hidden_size);
        for _ in 0..self.config.hc_mult {
            hidden.extend_from_slice(&embedding);
        }
        Ok(hidden)
    }

    fn cached_or_load_decoder_layer(
        &self,
        state: &mut Hy4RuntimeState,
        layer: usize,
    ) -> Result<Arc<DecoderLayer>, Hy4RuntimeError> {
        if let Some(cached) = &state.cached_layers[layer] {
            return Ok(Arc::clone(cached));
        }
        let loaded = load_decoder_layer(&self.index, &self.config, layer)?;
        self.retain_decoder_layer(state, layer, &loaded);
        Ok(loaded)
    }

    fn retain_decoder_layer(
        &self,
        state: &mut Hy4RuntimeState,
        layer: usize,
        loaded: &Arc<DecoderLayer>,
    ) {
        if layer < self.cached_decoder_layers && state.cached_layers[layer].is_none() {
            state.cached_layers[layer] = Some(Arc::clone(loaded));
        }
    }

    fn finish_step(
        &self,
        hidden: Vec<f32>,
        routes_by_layer: Vec<Vec<RouteChoice>>,
    ) -> Result<Hy4RuntimeStep, Hy4RuntimeError> {
        let hidden = identity_hyper_connection_head(
            &hidden,
            self.config.hidden_size,
            self.config.hc_mult,
            &self.head_hc.function,
            self.head_hc.scale,
            &self.head_hc.base,
            self.config.rms_norm_eps as f32,
            self.config.hc_eps as f32,
        )?;
        let hidden = bf16_rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps as f32)?;
        let logits = {
            let _profile = span(ProfileStage::Hy4LmHead);
            if let Some(lm_head) = &self.lm_head {
                lm_head.matvec(&hidden)?
            } else {
                streamed_reference_matvec_pipelined(
                    Arc::clone(&self.index),
                    "lm_head.weight",
                    self.config.vocab_size,
                    self.config.hidden_size,
                    &hidden,
                    4096,
                )?
            }
        };
        Ok(Hy4RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    pub fn config(&self) -> &Hy4Config {
        &self.config
    }
}

fn load_decoder_layer(
    index: &TensorIndex,
    config: &Hy4Config,
    layer: usize,
) -> Result<Arc<DecoderLayer>, Hy4RuntimeError> {
    let _profile = span(ProfileStage::Hy4LayerLoad);
    Ok(Arc::new(DecoderLayer::load(index, config, layer)?))
}

fn join_layer_loader(
    loader: Option<std::thread::ScopedJoinHandle<'_, Result<Arc<DecoderLayer>, Hy4RuntimeError>>>,
) -> Result<Option<Arc<DecoderLayer>>, Hy4RuntimeError> {
    loader
        .map(|loader| {
            loader.join().map_err(|_| {
                Hy4RuntimeError::Invalid("Hy4 layer prefetch thread panicked".to_owned())
            })?
        })
        .transpose()
}

impl CausalDecoder for Hy4RuntimeModel {
    type State = Hy4RuntimeState;
    type Error = Hy4RuntimeError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        Hy4RuntimeModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(Hy4RuntimeModel::forward_token(self, token, state)?.logits)
    }

    fn prefill(&self, prompt: &[u32], state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(Hy4RuntimeModel::prefill_tokens(self, prompt, state)?.logits)
    }
}

#[derive(Debug)]
pub struct Hy4RuntimeState {
    instance_id: u64,
    position: usize,
    attention: Vec<AttentionState>,
    experts: Hy4ExpertStore,
    cached_layers: Vec<Option<Arc<DecoderLayer>>>,
}

impl Hy4RuntimeState {
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

    pub fn cached_decoder_layers(&self) -> usize {
        self.cached_layers
            .iter()
            .filter(|layer| layer.is_some())
            .count()
    }
}

#[derive(Debug, Clone)]
struct HcWeights {
    function: WeightMatrix,
    base: Vec<f32>,
    scale: Vec<f32>,
}

impl HcWeights {
    fn load(
        index: &TensorIndex,
        config: &Hy4Config,
        prefix: &str,
    ) -> Result<Self, Hy4RuntimeError> {
        Ok(Self {
            function: matrix(
                index,
                &format!("{prefix}.hc_fn"),
                2 * config.hc_mult,
                config.hc_mult * config.hidden_size,
            )?,
            base: load_reference_vector(index, &format!("{prefix}.hc_base"), 2 * config.hc_mult)?,
            scale: load_reference_vector(index, &format!("{prefix}.hc_scale"), 2)?,
        })
    }

    fn pre(
        &self,
        hidden: &[f32],
        config: &Hy4Config,
    ) -> Result<(Vec<f32>, IdentityHyperConnectionMix), Hy4RuntimeError> {
        Ok(identity_hyper_connection_pre(
            hidden,
            config.hidden_size,
            config.hc_mult,
            &self.function,
            &self.scale,
            &self.base,
            config.rms_norm_eps as f32,
            config.hc_eps as f32,
            config.hc_magnitude as f32,
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
    fn load(index: &TensorIndex, config: &Hy4Config) -> Result<Self, Hy4RuntimeError> {
        Ok(Self {
            function: matrix(
                index,
                "model.hc_head.hc_head_fn",
                config.hc_mult,
                config.hc_mult * config.hidden_size,
            )?,
            base: load_reference_vector(index, "model.hc_head.hc_head_base", config.hc_mult)?,
            scale: load_reference_vector(index, "model.hc_head.hc_head_scale", 1)?[0],
        })
    }
}

#[derive(Debug)]
struct DecoderLayer {
    attn_hc: HcWeights,
    mlp_hc: HcWeights,
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    attention: AttentionWeights,
    feed_forward: FeedForward,
}

struct PreparedFeedForward {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    mix: IdentityHyperConnectionMix,
    sparse: Option<PreparedMoe>,
    _layer_profile: ProfileSpan,
}

impl DecoderLayer {
    fn load(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
    ) -> Result<Self, Hy4RuntimeError> {
        let prefix = format!("model.layers.{layer}");
        Ok(Self {
            attn_hc: HcWeights::load(index, config, &format!("{prefix}.hc_attn_layer.hc_pre"))?,
            mlp_hc: HcWeights::load(index, config, &format!("{prefix}.hc_mlp_layer.hc_pre"))?,
            input_norm: load_reference_vector(
                index,
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden_size,
            )?,
            post_attention_norm: load_reference_vector(
                index,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?,
            attention: AttentionWeights::load(index, config, layer)?,
            feed_forward: FeedForward::load(index, config, layer)?,
        })
    }

    fn forward(
        &self,
        hidden: &[f32],
        position: usize,
        attention_state: &mut AttentionState,
        experts: &mut Hy4ExpertStore,
        config: &Hy4Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), Hy4RuntimeError> {
        let prepared =
            self.prepare_feed_forward(hidden, position, attention_state, config, false)?;
        self.finish_feed_forward(prepared, experts, config)
    }

    /// Finishes the ordered attention-state mutation, then retains only the bounded tensors
    /// needed by this token's feed-forward branch. Prefill can prepare the following token while
    /// this branch waits for routed-expert I/O.
    fn prepare_feed_forward(
        &self,
        hidden: &[f32],
        position: usize,
        attention_state: &mut AttentionState,
        config: &Hy4Config,
        eager_sparse: bool,
    ) -> Result<PreparedFeedForward, Hy4RuntimeError> {
        let layer_profile = span(ProfileStage::Hy4Layer);
        let residual = hidden;
        let (collapsed, mix) = self.attn_hc.pre(hidden, config)?;
        let normalized = bf16_rms_norm(&collapsed, &self.input_norm, config.rms_norm_eps as f32)?;
        let branch = {
            let _profile = span(ProfileStage::Hy4Attention);
            self.attention
                .forward(&normalized, position, attention_state, config)?
        };
        let after_attention =
            identity_hyper_connection_post(&branch, residual, config.hidden_size, &mix)?;

        let residual = after_attention;
        let (collapsed, mix) = self.mlp_hc.pre(&residual, config)?;
        let normalized = bf16_rms_norm(
            &collapsed,
            &self.post_attention_norm,
            config.rms_norm_eps as f32,
        )?;
        let sparse = if eager_sparse {
            self.feed_forward.prepare_sparse(&normalized, config)?
        } else {
            None
        };
        Ok(PreparedFeedForward {
            residual,
            normalized,
            mix,
            sparse,
            _layer_profile: layer_profile,
        })
    }

    fn finish_feed_forward(
        &self,
        prepared: PreparedFeedForward,
        experts: &mut Hy4ExpertStore,
        config: &Hy4Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), Hy4RuntimeError> {
        let (branch, routes) = match (&self.feed_forward, prepared.sparse) {
            (FeedForward::Sparse(moe), Some(prepared)) => {
                moe.finish_prepared(prepared, experts, config)?
            }
            (_, None) => {
                let _profile = span(ProfileStage::Hy4Moe);
                self.feed_forward
                    .forward(&prepared.normalized, experts, config)?
            }
            (FeedForward::Dense(_), Some(_)) => {
                unreachable!("only sparse feed-forward branches can be prepared eagerly")
            }
        };
        Ok((
            identity_hyper_connection_post(
                &branch,
                &prepared.residual,
                config.hidden_size,
                &prepared.mix,
            )?,
            routes,
        ))
    }

    /// Finishes up to eighteen prepared sparse tokens together when their union fits in the current
    /// cache without eviction. The fallback retains the ordinary per-token path for dense layers
    /// and for the small tail where borrowed cache capacity is exhausted.
    fn finish_feed_forward_batch(
        &self,
        prepared: Vec<PreparedFeedForward>,
        experts: &mut Hy4ExpertStore,
        config: &Hy4Config,
    ) -> Result<HiddenBatchWithRoutes, Hy4RuntimeError> {
        let FeedForward::Sparse(moe) = &self.feed_forward else {
            return prepared
                .into_iter()
                .map(|prepared| self.finish_feed_forward(prepared, experts, config))
                .collect();
        };
        let can_batch = (2..=18).contains(&prepared.len())
            && prepared.iter().all(|prepared| prepared.sparse.is_some())
            && moe.prepared_batch_fits(
                prepared
                    .iter()
                    .map(|prepared| prepared.sparse.as_ref().expect("checked above")),
                experts,
            );
        if !can_batch {
            return prepared
                .into_iter()
                .map(|prepared| self.finish_feed_forward(prepared, experts, config))
                .collect();
        }

        let mut posts = Vec::with_capacity(prepared.len());
        let mut sparse = Vec::with_capacity(prepared.len());
        for prepared in prepared {
            let PreparedFeedForward {
                residual,
                normalized: _,
                mix,
                sparse: prepared_sparse,
                _layer_profile,
            } = prepared;
            posts.push((residual, mix, _layer_profile));
            sparse.push(prepared_sparse.expect("batch eligibility requires sparse preparation"));
        }
        let branches = moe.finish_prepared_batch(sparse, experts, config)?;
        posts
            .into_iter()
            .zip(branches)
            .map(|((residual, mix, _layer_profile), (branch, routes))| {
                Ok((
                    identity_hyper_connection_post(&branch, &residual, config.hidden_size, &mix)?,
                    routes,
                ))
            })
            .collect()
    }
}

#[derive(Debug)]
struct AttentionWeights {
    sink: Vec<f32>,
    q_a: WeightMatrix,
    q_norm: Vec<f32>,
    q_b: WeightMatrix,
    kv_a: WeightMatrix,
    kv_norm: Vec<f32>,
    kv_b: WeightMatrix,
    output_gate: WeightMatrix,
    output: WeightMatrix,
}

impl AttentionWeights {
    fn load(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
    ) -> Result<Self, Hy4RuntimeError> {
        let prefix = format!("model.layers.{layer}.self_attn");
        Ok(Self {
            sink: load_reference_vector(
                index,
                &format!("{prefix}.learnable_sink_param"),
                config.num_attention_heads,
            )?,
            q_a: mx_matrix(
                index,
                &format!("{prefix}.q_a_proj"),
                config.q_lora_rank,
                config.hidden_size,
            )?,
            q_norm: load_reference_vector(
                index,
                &format!("{prefix}.q_a_layernorm.weight"),
                config.q_lora_rank,
            )?,
            q_b: mx_matrix(
                index,
                &format!("{prefix}.q_b_proj"),
                config.num_attention_heads * config.qk_head_dim,
                config.q_lora_rank,
            )?,
            kv_a: mx_matrix(
                index,
                &format!("{prefix}.kv_a_proj_with_mqa"),
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden_size,
            )?,
            kv_norm: load_reference_vector(
                index,
                &format!("{prefix}.kv_a_layernorm.weight"),
                config.kv_lora_rank,
            )?,
            kv_b: mx_matrix(
                index,
                &format!("{prefix}.kv_b_proj"),
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )?,
            output_gate: bf16_matrix(
                index,
                &format!("{prefix}.linear_gate.weight"),
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
            )?,
            output: mx_matrix(
                index,
                &format!("{prefix}.o_proj"),
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        position: usize,
        state: &mut AttentionState,
        config: &Hy4Config,
    ) -> Result<Vec<f32>, Hy4RuntimeError> {
        if state.next_position != position {
            return Err(Hy4RuntimeError::Invalid(
                "attention state position mismatch".to_owned(),
            ));
        }
        let quantized_input = mx_activation(input)?;
        let profile_context = capture_context();
        // Q and KV consume the same immutable activation and do not interact until both low-rank
        // branches have finished. Keep cache mutation after the join so state remains ordered.
        let (query, kv_state) = install(|| {
            rayon::join(
                || {
                    profile_context.enter(|| {
                        let mut q_lora = mx_linear_quantized(&self.q_a, &quantized_input)?;
                        q_lora = bf16_rms_norm(&q_lora, &self.q_norm, config.rms_norm_eps as f32)?;
                        let mut query = mx_linear(&self.q_b, &q_lora)?;
                        for head in query.chunks_mut(config.qk_head_dim) {
                            paired_rope(
                                &mut head[config.qk_nope_head_dim..],
                                position,
                                config.rope_parameters.rope_theta as f32,
                                None,
                                1.0,
                                0,
                                0,
                                false,
                            )?;
                        }
                        Ok::<_, Hy4RuntimeError>(query)
                    })
                },
                || {
                    profile_context.enter(|| {
                        let kv = mx_linear_quantized(&self.kv_a, &quantized_input)?;
                        let latent = bf16_rms_norm(
                            &kv[..config.kv_lora_rank],
                            &self.kv_norm,
                            config.rms_norm_eps as f32,
                        )?;
                        let mut rope = kv[config.kv_lora_rank..].to_vec();
                        paired_rope(
                            &mut rope,
                            position,
                            config.rope_parameters.rope_theta as f32,
                            None,
                            1.0,
                            0,
                            0,
                            false,
                        )?;
                        Ok::<_, Hy4RuntimeError>((latent, rope))
                    })
                },
            )
        });
        let query = query?;
        let (mut latent, rope) = kv_state?;
        round_to_bf16_in_place(&mut latent)?;
        state.entries.push(KvEntry { latent, rope });

        let block = config.qk_nope_head_dim + config.v_head_dim;
        let scale = (config.qk_head_dim as f32).sqrt().recip();
        let profile_context = capture_context();
        // Heads and the output gate share only immutable inputs. The ordered head collect and the
        // later elementwise gate application preserve checkpoint output order.
        let (head_outputs, gate) = install(|| {
            rayon::join(
                || {
                    (0..config.num_attention_heads)
                        .into_par_iter()
                        .map(|head| {
                            profile_context.enter(|| {
                                let q = &query
                                    [head * config.qk_head_dim..(head + 1) * config.qk_head_dim];
                                let mut absorbed = self.kv_b.transpose_rows_matvec(
                                    head * block,
                                    &q[..config.qk_nope_head_dim],
                                )?;
                                round_to_bf16_in_place(&mut absorbed)?;
                                let probabilities = latent_attention_probabilities(
                                    &absorbed,
                                    &q[config.qk_nope_head_dim..],
                                    &state.entries,
                                    self.sink[head],
                                    scale,
                                )?;
                                let mut weighted_latent = vec![0.0f32; config.kv_lora_rank];
                                for (probability, entry) in probabilities.iter().zip(&state.entries)
                                {
                                    for (output, &value) in
                                        weighted_latent.iter_mut().zip(&entry.latent)
                                    {
                                        *output += *probability * value;
                                    }
                                }
                                round_to_bf16_in_place(&mut weighted_latent)?;
                                let mut value = self.kv_b.matvec_rows(
                                    head * block + config.qk_nope_head_dim,
                                    config.v_head_dim,
                                    &weighted_latent,
                                )?;
                                round_to_bf16_in_place(&mut value)?;
                                Ok::<_, Hy4RuntimeError>(value)
                            })
                        })
                        .collect::<Vec<_>>()
                },
                || profile_context.enter(|| bf16_linear(&self.output_gate, input)),
            )
        });
        let gate = gate?;
        let mut attention_output =
            Vec::with_capacity(config.num_attention_heads * config.v_head_dim);
        for head_output in head_outputs {
            attention_output.extend(head_output?);
        }
        for (value, gate) in attention_output.iter_mut().zip(gate) {
            *value *= stable_sigmoid(gate);
        }
        round_to_bf16_in_place(&mut attention_output)?;
        state.next_position += 1;
        mx_linear(&self.output, &attention_output)
    }
}

#[derive(Debug, Clone)]
struct KvEntry {
    latent: Vec<f32>,
    rope: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
struct AttentionState {
    next_position: usize,
    entries: Vec<KvEntry>,
}

impl AttentionState {
    fn stored_f32_elements(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.latent.len() + entry.rope.len())
            .sum()
    }
}

fn latent_attention_probabilities(
    query_latent: &[f32],
    query_rope: &[f32],
    cache: &[KvEntry],
    sink: f32,
    scale: f32,
) -> Result<Vec<f32>, Hy4RuntimeError> {
    if cache.is_empty() || !sink.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err(Hy4RuntimeError::Invalid(
            "latent attention cache/sink/scale is invalid".to_owned(),
        ));
    }
    let mut scores = Vec::with_capacity(cache.len());
    for entry in cache {
        if entry.latent.len() != query_latent.len() || entry.rope.len() != query_rope.len() {
            return Err(Hy4RuntimeError::Invalid(
                "latent attention cache geometry changed".to_owned(),
            ));
        }
        let latent = query_latent
            .iter()
            .zip(&entry.latent)
            .map(|(&left, &right)| left * right)
            .sum::<f32>();
        let rope = query_rope
            .iter()
            .zip(&entry.rope)
            .map(|(&left, &right)| left * right)
            .sum::<f32>();
        scores.push((latent + rope) * scale);
    }
    let maximum = scores.iter().copied().fold(sink, f32::max);
    let mut denominator = (sink - maximum).exp();
    for score in &mut scores {
        *score = (*score - maximum).exp();
        denominator += *score;
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Hy4RuntimeError::Invalid(
            "latent attention softmax is non-finite".to_owned(),
        ));
    }
    for score in &mut scores {
        *score /= denominator;
    }
    Ok(scores)
}

#[derive(Debug)]
enum FeedForward {
    Dense(DenseMlp),
    Sparse(MoeWeights),
}

impl FeedForward {
    fn load(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
    ) -> Result<Self, Hy4RuntimeError> {
        if config.layer_is_sparse(layer) {
            Ok(Self::Sparse(MoeWeights::load(index, config, layer)?))
        } else {
            Ok(Self::Dense(DenseMlp::load(
                index,
                &format!("model.layers.{layer}.mlp"),
                config.hidden_size,
                config.intermediate_size,
            )?))
        }
    }

    fn forward(
        &self,
        input: &[f32],
        experts: &mut Hy4ExpertStore,
        config: &Hy4Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), Hy4RuntimeError> {
        match self {
            Self::Dense(mlp) => Ok((mlp.forward(input)?, Vec::new())),
            Self::Sparse(moe) => moe.forward(input, experts, config),
        }
    }

    fn prepare_sparse(
        &self,
        input: &[f32],
        config: &Hy4Config,
    ) -> Result<Option<PreparedMoe>, Hy4RuntimeError> {
        match self {
            Self::Dense(_) => Ok(None),
            Self::Sparse(moe) => moe.prepare(input, config).map(Some),
        }
    }
}

#[derive(Debug)]
struct DenseMlp {
    gate: WeightMatrix,
    up: WeightMatrix,
    down: WeightMatrix,
}

impl DenseMlp {
    fn load(
        index: &TensorIndex,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Hy4RuntimeError> {
        Ok(Self {
            gate: mx_matrix(index, &format!("{prefix}.gate_proj"), intermediate, hidden)?,
            up: mx_matrix(index, &format!("{prefix}.up_proj"), intermediate, hidden)?,
            down: mx_matrix(index, &format!("{prefix}.down_proj"), hidden, intermediate)?,
        })
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Hy4RuntimeError> {
        let quantized = mx_activation(input)?;
        let gate = mx_linear_quantized(&self.gate, &quantized)?;
        let up = mx_linear_quantized(&self.up, &quantized)?;
        let mut activated = gate
            .iter()
            .zip(up)
            .map(|(&gate, up)| silu(gate) * up)
            .collect::<Vec<_>>();
        round_to_bf16_in_place(&mut activated)?;
        mx_linear(&self.down, &activated)
    }
}

#[derive(Debug)]
struct MoeWeights {
    layer: usize,
    router: WeightMatrix,
    correction_bias: Vec<f32>,
    shared: DenseMlp,
}

struct PreparedMoe {
    routes: Vec<RouteChoice>,
    execution_order: Vec<(usize, f32)>,
    expert_ids: Vec<usize>,
    quantized_input: Vec<f32>,
    shared: Vec<f32>,
    _profile: ProfileSpan,
}

impl MoeWeights {
    fn load(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
    ) -> Result<Self, Hy4RuntimeError> {
        let prefix = format!("model.layers.{layer}.mlp");
        Ok(Self {
            layer,
            router: bf16_matrix(
                index,
                &format!("{prefix}.gate.weight"),
                config.n_routed_experts,
                config.hidden_size,
            )?,
            correction_bias: load_reference_vector(
                index,
                &format!("{prefix}.gate.e_score_correction_bias"),
                config.n_routed_experts,
            )?,
            shared: DenseMlp::load(
                index,
                &format!("{prefix}.shared_experts"),
                config.hidden_size,
                config.n_shared_experts * config.moe_intermediate_size,
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        experts: &mut Hy4ExpertStore,
        config: &Hy4Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), Hy4RuntimeError> {
        // The release loads BF16 checkpoint router weights into an FP32 GateLinear and emits FP32.
        let logits = self.router.matvec(input)?;
        let routes = route_noaux_tc(
            &logits,
            &self.correction_bias,
            config.num_experts_per_tok,
            config.n_group,
            config.topk_group,
            config.norm_topk_prob,
            config.routed_scaling_factor as f32,
        )?;
        let mut output = vec![0.0f32; config.hidden_size];
        let mut execution_order = routes
            .iter()
            .map(|route| (route.expert, route.weight))
            .collect::<Vec<_>>();
        execution_order.sort_by_key(|&(expert, _)| expert);
        let expert_ids = execution_order
            .iter()
            .map(|&(expert, _)| expert)
            .collect::<Vec<_>>();
        let quantized_input = mx_activation(input)?;
        let (computed, shared) = if experts.batch_has_missing(self.layer, &expert_ids) {
            let profile_context = capture_context();
            let (computed, shared) = install(|| {
                rayon::join(
                    || {
                        profile_context.enter(|| {
                            experts.acquire_and_forward_batch(
                                self.layer,
                                &expert_ids,
                                &quantized_input,
                                config.swiglu_limit as f32,
                            )
                        })
                    },
                    || profile_context.enter(|| self.shared.forward(input)),
                )
            });
            (computed?, shared?)
        } else {
            let loaded = experts.acquire_batch(self.layer, &expert_ids)?;
            let shared = self.shared.forward(input)?;
            let profile_context = capture_context();
            let computed = install(|| {
                loaded
                    .par_iter()
                    .map(|expert| {
                        profile_context.enter(|| {
                            let _profile = span(ProfileStage::Hy4ExpertCompute);
                            expert
                                .forward_quantized(&quantized_input, config.swiglu_limit as f32)
                                .map_err(Hy4RuntimeError::from)
                        })
                    })
                    .collect::<Vec<_>>()
            });
            (computed, shared)
        };
        for ((_, route_weight), computed) in execution_order.into_iter().zip(computed) {
            let computed = computed?;
            for (output, value) in output.iter_mut().zip(computed) {
                *output += route_weight * value;
            }
        }
        for (output, shared) in output.iter_mut().zip(shared) {
            *output += shared;
        }
        round_to_bf16_in_place(&mut output)?;
        Ok((output, routes))
    }

    /// Prepares the cache-independent half of a sparse feed-forward. In layer-major prefill this
    /// runs for token N+1 while token N waits for routed-expert storage, exposing the router,
    /// activation quantizer, and shared expert as useful CPU work without speculatively touching
    /// the expert cache.
    fn prepare(&self, input: &[f32], config: &Hy4Config) -> Result<PreparedMoe, Hy4RuntimeError> {
        let profile = span(ProfileStage::Hy4Moe);
        let profile_context = capture_context();
        let (routed, shared) = install(|| {
            rayon::join(
                || {
                    profile_context.enter(|| {
                        let logits = self.router.matvec(input)?;
                        let routes = route_noaux_tc(
                            &logits,
                            &self.correction_bias,
                            config.num_experts_per_tok,
                            config.n_group,
                            config.topk_group,
                            config.norm_topk_prob,
                            config.routed_scaling_factor as f32,
                        )?;
                        let mut execution_order = routes
                            .iter()
                            .map(|route| (route.expert, route.weight))
                            .collect::<Vec<_>>();
                        execution_order.sort_by_key(|&(expert, _)| expert);
                        let expert_ids = execution_order
                            .iter()
                            .map(|&(expert, _)| expert)
                            .collect::<Vec<_>>();
                        let quantized_input = mx_activation(input)?;
                        Ok::<_, Hy4RuntimeError>((
                            routes,
                            execution_order,
                            expert_ids,
                            quantized_input,
                        ))
                    })
                },
                || profile_context.enter(|| self.shared.forward(input)),
            )
        });
        let (routes, execution_order, expert_ids, quantized_input) = routed?;
        Ok(PreparedMoe {
            routes,
            execution_order,
            expert_ids,
            quantized_input,
            shared: shared?,
            _profile: profile,
        })
    }

    fn finish_prepared(
        &self,
        prepared: PreparedMoe,
        experts: &mut Hy4ExpertStore,
        config: &Hy4Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), Hy4RuntimeError> {
        let PreparedMoe {
            routes,
            execution_order,
            expert_ids,
            quantized_input,
            shared,
            _profile,
        } = prepared;
        let computed = if experts.batch_has_missing(self.layer, &expert_ids) {
            experts.acquire_and_forward_batch(
                self.layer,
                &expert_ids,
                &quantized_input,
                config.swiglu_limit as f32,
            )?
        } else {
            let loaded = experts.acquire_batch(self.layer, &expert_ids)?;
            let profile_context = capture_context();
            install(|| {
                loaded
                    .par_iter()
                    .map(|expert| {
                        profile_context.enter(|| {
                            let _profile = span(ProfileStage::Hy4ExpertCompute);
                            expert
                                .forward_quantized(&quantized_input, config.swiglu_limit as f32)
                                .map_err(Hy4RuntimeError::from)
                        })
                    })
                    .collect::<Vec<_>>()
            })
        };
        let mut output = vec![0.0f32; config.hidden_size];
        for ((_, route_weight), computed) in execution_order.into_iter().zip(computed) {
            let computed = computed?;
            for (output, value) in output.iter_mut().zip(computed) {
                *output += route_weight * value;
            }
        }
        for (output, shared) in output.iter_mut().zip(shared) {
            *output += shared;
        }
        round_to_bf16_in_place(&mut output)?;
        Ok((output, routes))
    }

    fn prepared_batch_fits<'a>(
        &self,
        prepared: impl Iterator<Item = &'a PreparedMoe>,
        experts: &Hy4ExpertStore,
    ) -> bool {
        let expert_ids = prepared
            .flat_map(|prepared| prepared.expert_ids.iter().copied())
            .collect::<Vec<_>>();
        experts.batch_fits_without_eviction(self.layer, &expert_ids)
    }

    /// Computes repeated expert routes across several tokens as compact MXFP8 batches. Routed
    /// accumulation remains token-local and follows the same sorted expert order as the scalar
    /// path, so only weight decoding/traversal is shared.
    fn finish_prepared_batch(
        &self,
        prepared: Vec<PreparedMoe>,
        experts: &mut Hy4ExpertStore,
        config: &Hy4Config,
    ) -> Result<HiddenBatchWithRoutes, Hy4RuntimeError> {
        let expert_ids = prepared
            .iter()
            .map(|prepared| prepared.expert_ids.clone())
            .collect::<Vec<_>>();
        let quantized = prepared
            .iter()
            .map(|prepared| prepared.quantized_input.as_slice())
            .collect::<Vec<_>>();
        let computed = experts.acquire_and_forward_token_batch(
            self.layer,
            &expert_ids,
            &quantized,
            config.swiglu_limit as f32,
        )?;
        prepared
            .into_iter()
            .zip(computed)
            .map(|(prepared, computed)| {
                let PreparedMoe {
                    routes,
                    execution_order,
                    expert_ids: _,
                    quantized_input: _,
                    shared,
                    _profile,
                } = prepared;
                let mut output = vec![0.0f32; config.hidden_size];
                for ((_, route_weight), computed) in execution_order.into_iter().zip(computed) {
                    for (output, value) in output.iter_mut().zip(computed) {
                        *output += route_weight * value;
                    }
                }
                for (output, shared) in output.iter_mut().zip(shared) {
                    *output += shared;
                }
                round_to_bf16_in_place(&mut output)?;
                Ok((output, routes))
            })
            .collect()
    }
}

#[derive(Debug)]
struct Hy4ExpertStore {
    index: Arc<TensorIndex>,
    config: Hy4Config,
    maximum_bytes: u64,
    direct_io: bool,
    cache: LayerLruCache<Arc<Hy4Expert>>,
    // Layer-major prefill can trim many borrowed cache slots at once. Their same-shaped backing
    // buffers remain within the global slot budget and are consumed before later layers allocate.
    recycled: Vec<Hy4ExpertBuffers>,
    // Expert payloads live behind retained read-only checkpoint handles. Once all four slices for
    // one layer/expert pair pass the FP8 validation scans, later cache reloads can retain that
    // result and avoid rescanning the same 37.1 MiB immutable payload.
    payload_validated: Arc<[AtomicBool]>,
    telemetry: ExpertTelemetry,
}

#[derive(Clone, Debug)]
struct ExpertPayloadValidation {
    flags: Arc<[AtomicBool]>,
    slot: usize,
}

impl ExpertPayloadValidation {
    fn required(&self) -> bool {
        !self.flags[self.slot].load(Ordering::Acquire)
    }

    fn complete(&self) {
        self.flags[self.slot].store(true, Ordering::Release);
    }
}

impl Hy4ExpertStore {
    fn new(
        index: Arc<TensorIndex>,
        config: &Hy4Config,
        slots_per_layer: usize,
        cache_budget_bytes: u64,
        maximum_bytes: u64,
    ) -> Result<Self, Hy4RuntimeError> {
        if slots_per_layer > config.n_routed_experts
            || cache_budget_bytes == 0
            || maximum_bytes == 0
        {
            return Err(Hy4RuntimeError::Invalid(
                "expert cache geometry is invalid".to_owned(),
            ));
        }
        let sparse_layers = (0..config.num_hidden_layers)
            .filter(|&layer| config.layer_is_sparse(layer))
            .collect::<Vec<_>>();
        let base_slots = slots_per_layer
            .checked_mul(sparse_layers.len())
            .ok_or_else(|| Hy4RuntimeError::Invalid("expert cache slots overflow".to_owned()))?;
        let maximum_slots = cache_budget_bytes
            .checked_div(maximum_bytes)
            .and_then(|slots| usize::try_from(slots).ok())
            .unwrap_or(usize::MAX)
            .min(sparse_layers.len().saturating_mul(config.n_routed_experts));
        if maximum_slots < base_slots {
            return Err(Hy4RuntimeError::Invalid(
                "expert cache budget is smaller than its per-layer capacity".to_owned(),
            ));
        }
        let extras = maximum_slots - base_slots;
        let bonus_per_layer = extras.checked_div(sparse_layers.len()).unwrap_or(0);
        let remainder = extras.checked_rem(sparse_layers.len()).unwrap_or(0);
        let mut capacities = vec![0usize; config.num_hidden_layers];
        for (rank, &layer) in sparse_layers.iter().enumerate() {
            // Spread the remainder across depth rather than assigning it to one contiguous prefix.
            let receives_remainder = (rank + 1).saturating_mul(remainder) / sparse_layers.len()
                > rank.saturating_mul(remainder) / sparse_layers.len();
            capacities[layer] = slots_per_layer
                .saturating_add(bonus_per_layer)
                .saturating_add(usize::from(receives_remainder))
                .min(config.n_routed_experts);
        }
        let validation_slots = config
            .num_hidden_layers
            .checked_mul(config.n_routed_experts)
            .ok_or_else(|| {
                Hy4RuntimeError::Invalid("expert validation bitmap size overflows".to_owned())
            })?;
        Ok(Self {
            index,
            config: config.clone(),
            maximum_bytes,
            direct_io: false,
            cache: LayerLruCache::with_layer_capacities(capacities),
            recycled: Vec::with_capacity(config.num_experts_per_tok),
            payload_validated: (0..validation_slots)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>()
                .into(),
            telemetry: ExpertTelemetry::default(),
        })
    }

    fn begin_prefill_layer(&mut self, layer: usize) -> usize {
        if self.config.layer_is_sparse(layer) {
            self.cache
                .lend_unused_capacity(layer, self.config.n_routed_experts)
        } else {
            0
        }
    }

    fn finish_prefill_layer(&mut self, layer: usize, steady_capacity: usize) {
        if !self.config.layer_is_sparse(layer) {
            return;
        }
        let evicted =
            self.cache
                .restore_layer_capacity(&mut self.telemetry, layer, steady_capacity);
        for expert in evicted {
            self.recycle_evicted(expert);
        }
    }

    fn recycle_evicted(&mut self, expert: Arc<Hy4Expert>) {
        if let Ok(expert) = Arc::try_unwrap(expert) {
            if let Some(buffers) = expert.into_reusable_buffers() {
                self.recycled.push(buffers);
            }
        }
    }

    fn payload_validation(&self, layer: usize, expert: usize) -> ExpertPayloadValidation {
        ExpertPayloadValidation {
            flags: Arc::clone(&self.payload_validated),
            slot: layer * self.config.n_routed_experts + expert,
        }
    }

    fn acquire(&mut self, layer: usize, expert: usize) -> Result<Arc<Hy4Expert>, Hy4RuntimeError> {
        if layer >= self.config.num_hidden_layers || expert >= self.config.n_routed_experts {
            return Err(Hy4RuntimeError::Invalid(
                "expert request is outside configured geometry".to_owned(),
            ));
        }
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let direct_io = self.direct_io;
        let validation = self.payload_validation(layer, expert);
        self.cache.access(
            &mut self.telemetry,
            layer,
            expert,
            || {
                let mut profile = span(ProfileStage::Hy4ExpertLoad);
                let loaded = Hy4Expert::load_with_payload_validation(
                    &index,
                    &config,
                    layer,
                    expert,
                    maximum,
                    validation.required(),
                    direct_io,
                )?;
                validation.complete();
                let resident = loaded.resident_bytes() as u64;
                let physical = expert_physical_bytes(&config)?;
                profile.add_logical_bytes(physical);
                Ok((Arc::new(loaded), resident, physical))
            },
            |expert| Ok(Arc::clone(expert)),
        )
    }

    fn batch_has_missing(&self, layer: usize, experts: &[usize]) -> bool {
        experts
            .iter()
            .any(|&expert| !self.cache.contains(layer, expert))
    }

    fn batch_fits_without_eviction(&self, layer: usize, experts: &[usize]) -> bool {
        layer < self.config.num_hidden_layers
            && experts
                .iter()
                .all(|&expert| expert < self.config.n_routed_experts)
            && self.cache.can_insert_without_eviction(layer, experts)
    }

    /// Fetches all experts selected for one token with enough independent reads to keep NVMe
    /// busy, then replays cache accesses and expert execution in the original deterministic order.
    /// A cached entry can be evicted by an earlier replayed miss; that uncommon case falls back to
    /// the normal single load and therefore retains the exact LayerLruCache semantics.
    fn acquire_batch(
        &mut self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<Arc<Hy4Expert>>, Hy4RuntimeError> {
        if layer >= self.config.num_hidden_layers
            || experts
                .iter()
                .any(|&expert| expert >= self.config.n_routed_experts)
        {
            return Err(Hy4RuntimeError::Invalid(
                "expert batch request is outside configured geometry".to_owned(),
            ));
        }

        let mut missing = Vec::new();
        for &expert in experts {
            if !self.cache.contains(layer, expert) && !missing.contains(&expert) {
                missing.push(expert);
            }
        }
        if missing.is_empty() {
            return experts
                .iter()
                .map(|&expert| self.acquire(layer, expert))
                .collect();
        }
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let direct_io = self.direct_io;
        let payload_validated = Arc::clone(&self.payload_validated);
        let profile_context = capture_context();
        let loaded = install(|| {
            missing
                .par_iter()
                .map(|&expert| {
                    profile_context.enter(|| {
                        let validation = ExpertPayloadValidation {
                            flags: Arc::clone(&payload_validated),
                            slot: layer * config.n_routed_experts + expert,
                        };
                        let mut profile = span(ProfileStage::Hy4ExpertLoad);
                        let value = Hy4Expert::load_with_payload_validation(
                            &index,
                            &config,
                            layer,
                            expert,
                            maximum,
                            validation.required(),
                            direct_io,
                        )?;
                        validation.complete();
                        let resident = value.resident_bytes() as u64;
                        let physical = expert_physical_bytes(&config)?;
                        profile.add_logical_bytes(physical);
                        Ok::<_, Hy4RuntimeError>((expert, (Arc::new(value), resident, physical)))
                    })
                })
                .collect::<Vec<_>>()
        });
        let mut preloaded = HashMap::with_capacity(loaded.len());
        for loaded in loaded {
            let (expert, value) = loaded?;
            preloaded.insert(expert, value);
        }

        experts
            .iter()
            .map(|&expert| {
                let index = Arc::clone(&self.index);
                let config = self.config.clone();
                let maximum = self.maximum_bytes;
                let direct_io = self.direct_io;
                let validation = self.payload_validation(layer, expert);
                self.cache.access(
                    &mut self.telemetry,
                    layer,
                    expert,
                    || {
                        if let Some(value) = preloaded.remove(&expert) {
                            return Ok(value);
                        }
                        let mut profile = span(ProfileStage::Hy4ExpertLoad);
                        let value = Hy4Expert::load_with_payload_validation(
                            &index,
                            &config,
                            layer,
                            expert,
                            maximum,
                            validation.required(),
                            direct_io,
                        )?;
                        validation.complete();
                        let resident = value.resident_bytes() as u64;
                        let physical = expert_physical_bytes(&config)?;
                        profile.add_logical_bytes(physical);
                        Ok((Arc::new(value), resident, physical))
                    },
                    |expert| Ok(Arc::clone(expert)),
                )
            })
            .collect()
    }

    /// Loads each missing expert and starts its MXFP8 projections immediately, while sibling
    /// workers continue their outstanding reads and already-resident experts compute in parallel.
    /// Cache accesses are replayed only after every load succeeds, retaining deterministic
    /// hit/miss/LRU accounting and routed accumulation order.
    fn acquire_and_forward_batch(
        &mut self,
        layer: usize,
        experts: &[usize],
        quantized_input: &[f32],
        swiglu_limit: f32,
    ) -> Result<Vec<Result<Vec<f32>, Hy4RuntimeError>>, Hy4RuntimeError> {
        if layer >= self.config.num_hidden_layers
            || experts
                .iter()
                .any(|&expert| expert >= self.config.n_routed_experts)
        {
            return Err(Hy4RuntimeError::Invalid(
                "expert batch request is outside configured geometry".to_owned(),
            ));
        }

        let mut missing = Vec::new();
        for &expert in experts {
            if !self.cache.contains(layer, expert) && !missing.contains(&expert) {
                missing.push(expert);
            }
        }
        let cached = experts
            .iter()
            .enumerate()
            .filter(|(_, expert)| !missing.contains(expert))
            .map(|(position, &expert)| {
                let value = self
                    .cache
                    .peek(layer, expert)
                    .expect("an expert classified as cached remains resident before replay");
                (position, expert, Arc::clone(value))
            })
            .collect::<Vec<_>>();
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let direct_io = self.direct_io;
        let reuse = (0..missing.len())
            .map(|_| self.recycled.pop())
            .collect::<Vec<_>>();
        let payload_validated = Arc::clone(&self.payload_validated);
        let profile_context = capture_context();
        let compute_cached = || {
            cached
                .par_iter()
                .map(|(position, _, expert)| {
                    let result = profile_context.enter(|| {
                        let _profile = span(ProfileStage::Hy4ExpertCompute);
                        expert
                            .forward_quantized(quantized_input, swiglu_limit)
                            .map_err(Hy4RuntimeError::from)
                    });
                    (*position, result)
                })
                .collect::<Vec<_>>()
        };
        let (loaded, cached_computed) = if direct_io {
            let quantized_input: Arc<[f32]> = quantized_input.to_vec().into();
            let tasks = missing
                .iter()
                .copied()
                .zip(reuse)
                .map(|(expert, reuse)| {
                    let index = Arc::clone(&index);
                    let config = config.clone();
                    let payload_validated = Arc::clone(&payload_validated);
                    let profile_context = profile_context.clone();
                    let quantized_input = Arc::clone(&quantized_input);
                    spawn_io(move || {
                        profile_context.enter(|| {
                            let validation = ExpertPayloadValidation {
                                flags: payload_validated,
                                slot: layer * config.n_routed_experts + expert,
                            };
                            let mut profile = span(ProfileStage::Hy4ExpertLoad);
                            let physical = expert_physical_bytes(&config)?;
                            profile.add_logical_bytes(physical);
                            let (value, computed) = Hy4Expert::load_and_forward_quantized(
                                &index,
                                &config,
                                layer,
                                expert,
                                maximum,
                                &quantized_input,
                                swiglu_limit,
                                reuse,
                                profile,
                                validation.required(),
                                true,
                            )?;
                            validation.complete();
                            let resident = value.resident_bytes() as u64;
                            let value = Arc::new(value);
                            let computed = computed.map_err(Hy4RuntimeError::from);
                            Ok::<_, Hy4RuntimeError>((
                                expert,
                                (Arc::clone(&value), resident, physical),
                                computed,
                            ))
                        })
                    })
                })
                .collect::<Vec<_>>();
            let cached_computed = install(compute_cached);
            let loaded = tasks.into_iter().map(|task| task.join()).collect();
            (loaded, cached_computed)
        } else {
            install(|| {
                rayon::join(
                    || {
                        missing
                            .par_iter()
                            .zip(reuse.into_par_iter())
                            .map(|(&expert, reuse)| {
                                profile_context.enter(|| {
                                    let validation = ExpertPayloadValidation {
                                        flags: Arc::clone(&payload_validated),
                                        slot: layer * config.n_routed_experts + expert,
                                    };
                                    let mut profile = span(ProfileStage::Hy4ExpertLoad);
                                    let physical = expert_physical_bytes(&config)?;
                                    profile.add_logical_bytes(physical);
                                    let (value, computed) = Hy4Expert::load_and_forward_quantized(
                                        &index,
                                        &config,
                                        layer,
                                        expert,
                                        maximum,
                                        quantized_input,
                                        swiglu_limit,
                                        reuse,
                                        profile,
                                        validation.required(),
                                        false,
                                    )?;
                                    validation.complete();
                                    let resident = value.resident_bytes() as u64;
                                    let value = Arc::new(value);
                                    let computed = computed.map_err(Hy4RuntimeError::from);
                                    Ok::<_, Hy4RuntimeError>((
                                        expert,
                                        (Arc::clone(&value), resident, physical),
                                        computed,
                                    ))
                                })
                            })
                            .collect::<Vec<_>>()
                    },
                    compute_cached,
                )
            })
        };
        let mut preloaded = HashMap::with_capacity(loaded.len());
        for loaded in loaded {
            let (expert, value, computed) = loaded?;
            preloaded.insert(expert, (value, computed));
        }

        let mut computed = (0..experts.len()).map(|_| None).collect::<Vec<_>>();
        for (position, result) in cached_computed {
            computed[position] = Some(result);
        }

        // Refresh every expert that was resident at batch start before inserting misses. This
        // prevents a selected cached expert from being evicted by an earlier replayed insertion.
        for &(_, expert, _) in &cached {
            let index = Arc::clone(&self.index);
            let config = self.config.clone();
            let maximum = self.maximum_bytes;
            let direct_io = self.direct_io;
            let validation = self.payload_validation(layer, expert);
            self.cache.access(
                &mut self.telemetry,
                layer,
                expert,
                || {
                    let mut profile = span(ProfileStage::Hy4ExpertLoad);
                    let value = Hy4Expert::load_with_payload_validation(
                        &index,
                        &config,
                        layer,
                        expert,
                        maximum,
                        validation.required(),
                        direct_io,
                    )?;
                    validation.complete();
                    let resident = value.resident_bytes() as u64;
                    let physical = expert_physical_bytes(&config)?;
                    profile.add_logical_bytes(physical);
                    Ok::<_, Hy4RuntimeError>((Arc::new(value), resident, physical))
                },
                |_| Ok::<_, Hy4RuntimeError>(()),
            )?;
        }

        for (position, &expert) in experts.iter().enumerate() {
            if !missing.contains(&expert) {
                continue;
            }
            let index = Arc::clone(&self.index);
            let config = self.config.clone();
            let maximum = self.maximum_bytes;
            let direct_io = self.direct_io;
            let validation = self.payload_validation(layer, expert);
            let mut precomputed = None;
            let (_, evicted) = self.cache.access_with_evicted(
                &mut self.telemetry,
                layer,
                expert,
                || {
                    if let Some((value, result)) = preloaded.remove(&expert) {
                        precomputed = Some(result);
                        return Ok::<_, Hy4RuntimeError>(value);
                    }
                    let mut profile = span(ProfileStage::Hy4ExpertLoad);
                    let value = Hy4Expert::load_with_payload_validation(
                        &index,
                        &config,
                        layer,
                        expert,
                        maximum,
                        validation.required(),
                        direct_io,
                    )?;
                    validation.complete();
                    let resident = value.resident_bytes() as u64;
                    let physical = expert_physical_bytes(&config)?;
                    profile.add_logical_bytes(physical);
                    Ok::<_, Hy4RuntimeError>((Arc::new(value), resident, physical))
                },
                |_| Ok::<_, Hy4RuntimeError>(()),
            )?;
            if let Some(evicted) = evicted {
                self.recycle_evicted(evicted);
            }
            computed[position] = precomputed;
        }
        Ok(computed
            .into_iter()
            .map(|result| result.expect("every requested expert is computed exactly once"))
            .collect())
    }

    /// Loads and executes the union of several consecutive tokens' routes. This is used only when
    /// every new expert fits without eviction: all physical reads still occur exactly once, while
    /// repeated routes reuse decoded weights in compact MXFP8 row batches. Cache accesses are
    /// then replayed token by token using the ordinary cached-before-missing batch order.
    fn acquire_and_forward_token_batch(
        &mut self,
        layer: usize,
        expert_ids_by_token: &[Vec<usize>],
        quantized_inputs: &[&[f32]],
        swiglu_limit: f32,
    ) -> Result<Vec<Vec<Vec<f32>>>, Hy4RuntimeError> {
        if !(2..=18).contains(&expert_ids_by_token.len())
            || quantized_inputs.len() != expert_ids_by_token.len()
            || layer >= self.config.num_hidden_layers
            || expert_ids_by_token
                .iter()
                .flatten()
                .any(|&expert| expert >= self.config.n_routed_experts)
            || quantized_inputs.iter().any(|input| {
                input.len() != self.config.hidden_size
                    || input.iter().any(|value| !value.is_finite())
            })
        {
            return Err(Hy4RuntimeError::Invalid(
                "expert token batch has invalid geometry or activation values".to_owned(),
            ));
        }
        let flattened = expert_ids_by_token
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        if !self.cache.can_insert_without_eviction(layer, &flattened) {
            return Err(Hy4RuntimeError::Invalid(
                "expert token batch would change cache eviction semantics".to_owned(),
            ));
        }
        let mut group_indices = HashMap::new();
        let mut groups = Vec::<(usize, Vec<(usize, usize)>)>::new();
        for (token, expert_ids) in expert_ids_by_token.iter().enumerate() {
            for (route, &expert) in expert_ids.iter().enumerate() {
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
        let work = groups
            .into_iter()
            .map(|(expert, positions)| {
                let cached = self.cache.peek(layer, expert).map(Arc::clone);
                let reuse = cached.is_none().then(|| self.recycled.pop()).flatten();
                (expert, positions, cached, reuse)
            })
            .collect::<Vec<_>>();
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let direct_io = self.direct_io;
        let payload_validated = Arc::clone(&self.payload_validated);
        let profile_context = capture_context();
        let completed = install(|| {
            work.into_par_iter()
                .map(|(expert, positions, cached, reuse)| {
                    let mut input =
                        Vec::with_capacity(positions.len().saturating_mul(config.hidden_size));
                    for &(token, _) in &positions {
                        input.extend_from_slice(quantized_inputs[token]);
                    }
                    match cached {
                        Some(expert_value) => {
                            let computed = profile_context.enter(|| {
                                let _profile = span(ProfileStage::Hy4ExpertCompute);
                                expert_value
                                    .forward_quantized_batch(&input, positions.len(), swiglu_limit)
                                    .map_err(Hy4RuntimeError::from)
                            });
                            Ok::<_, Hy4RuntimeError>((expert, positions, None, computed))
                        }
                        None => profile_context.enter(|| {
                            let validation = ExpertPayloadValidation {
                                flags: Arc::clone(&payload_validated),
                                slot: layer * config.n_routed_experts + expert,
                            };
                            let mut profile = span(ProfileStage::Hy4ExpertLoad);
                            let physical = expert_physical_bytes(&config)?;
                            profile.add_logical_bytes(physical);
                            let (value, computed) = Hy4Expert::load_and_forward_quantized_batch(
                                &index,
                                &config,
                                layer,
                                expert,
                                maximum,
                                &input,
                                positions.len(),
                                swiglu_limit,
                                reuse,
                                profile,
                                validation.required(),
                                direct_io,
                            )?;
                            validation.complete();
                            let resident = value.resident_bytes() as u64;
                            Ok((
                                expert,
                                positions,
                                Some((Arc::new(value), resident, physical)),
                                computed.map_err(Hy4RuntimeError::from),
                            ))
                        }),
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
            computed_groups.push((positions, computed));
        }

        // `acquire_and_forward_batch` refreshes all routes resident at each token's start before
        // inserting that token's misses. Reproduce that order exactly so later LRU outcomes and
        // telemetry remain unchanged even though computation was grouped across tokens.
        for expert_ids in expert_ids_by_token {
            let missing = expert_ids
                .iter()
                .copied()
                .filter(|&expert| !self.cache.contains(layer, expert))
                .collect::<Vec<_>>();
            for expert in expert_ids
                .iter()
                .copied()
                .filter(|expert| !missing.contains(expert))
                .chain(
                    expert_ids
                        .iter()
                        .copied()
                        .filter(|expert| missing.contains(expert)),
                )
            {
                let index = Arc::clone(&self.index);
                let config = self.config.clone();
                let maximum = self.maximum_bytes;
                let direct_io = self.direct_io;
                let validation = self.payload_validation(layer, expert);
                let (_, evicted) = self.cache.access_with_evicted(
                    &mut self.telemetry,
                    layer,
                    expert,
                    || {
                        if let Some(value) = preloaded.remove(&expert) {
                            return Ok::<_, Hy4RuntimeError>(value);
                        }
                        let mut profile = span(ProfileStage::Hy4ExpertLoad);
                        let value = Hy4Expert::load_with_payload_validation(
                            &index,
                            &config,
                            layer,
                            expert,
                            maximum,
                            validation.required(),
                            direct_io,
                        )?;
                        validation.complete();
                        let resident = value.resident_bytes() as u64;
                        let physical = expert_physical_bytes(&config)?;
                        profile.add_logical_bytes(physical);
                        Ok((Arc::new(value), resident, physical))
                    },
                    |_| Ok::<_, Hy4RuntimeError>(()),
                )?;
                debug_assert!(evicted.is_none());
                if let Some(evicted) = evicted {
                    self.recycle_evicted(evicted);
                }
            }
        }

        let mut computed = expert_ids_by_token
            .iter()
            .map(|expert_ids| {
                (0..expert_ids.len())
                    .map(|_| None)
                    .collect::<Vec<Option<Vec<f32>>>>()
            })
            .collect::<Vec<_>>();
        for (positions, values) in computed_groups {
            let values = values?;
            if values.len() != positions.len() {
                return Err(Hy4RuntimeError::Invalid(
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

fn expert_physical_bytes(config: &Hy4Config) -> Result<u64, Hy4RuntimeError> {
    let hidden = config.hidden_size as u64;
    let intermediate = config.moe_intermediate_size as u64;
    let gate_rows = 2 * intermediate;
    gate_rows
        .checked_mul(hidden)
        .and_then(|value| value.checked_add(gate_rows * hidden.div_ceil(MX_BLOCK as u64)))
        .and_then(|value| value.checked_add(hidden * intermediate))
        .and_then(|value| value.checked_add(hidden * intermediate.div_ceil(MX_BLOCK as u64)))
        .ok_or_else(|| Hy4RuntimeError::Invalid("expert physical bytes overflow".to_owned()))
}

fn matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<WeightMatrix, Hy4RuntimeError> {
    Ok(load_weight_matrix(index, name, rows, columns, u64::MAX)?)
}

fn mx_matrix(
    index: &TensorIndex,
    stem: &str,
    rows: usize,
    columns: usize,
) -> Result<WeightMatrix, Hy4RuntimeError> {
    matrix(index, &format!("{stem}.weight"), rows, columns)
}

fn bf16_matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<WeightMatrix, Hy4RuntimeError> {
    Ok(load_compact_bf16_matrix(
        index,
        name,
        rows,
        columns,
        u64::MAX,
    )?)
}

fn mx_linear(weight: &WeightMatrix, input: &[f32]) -> Result<Vec<f32>, Hy4RuntimeError> {
    let quantized = mx_activation(input)?;
    mx_linear_quantized(weight, &quantized)
}

fn mx_activation(input: &[f32]) -> Result<Vec<f32>, Hy4RuntimeError> {
    let _profile = span(ProfileStage::Hy4ActivationQuantization);
    simulate_e4m3_activation(input, MX_BLOCK)
        .map_err(|error| Hy4RuntimeError::Invalid(error.to_string()))
}

fn mx_linear_quantized(
    weight: &WeightMatrix,
    quantized: &[f32],
) -> Result<Vec<f32>, Hy4RuntimeError> {
    let mut output = weight.matvec(quantized)?;
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

fn bf16_linear(weight: &WeightMatrix, input: &[f32]) -> Result<Vec<f32>, Hy4RuntimeError> {
    let mut output = weight.matvec_bf16_fp32(input)?;
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

fn bf16_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, Hy4RuntimeError> {
    let mut output = rms_norm(input, weight, eps)?;
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_payload_validation_is_shared_after_completion() {
        let flags: Arc<[AtomicBool]> = vec![AtomicBool::new(false)].into();
        let validation = ExpertPayloadValidation {
            flags: Arc::clone(&flags),
            slot: 0,
        };
        assert!(validation.required());
        validation.complete();
        assert!(!validation.clone().required());
    }

    #[test]
    fn sink_is_a_zero_value_softmax_slot() {
        let cache = vec![KvEntry {
            latent: vec![2.0],
            rope: vec![0.0, 0.0],
        }];
        let probabilities =
            latent_attention_probabilities(&[1.0], &[0.0, 0.0], &cache, 2.0, 1.0).unwrap();
        assert_eq!(probabilities, [0.5]);
    }

    #[test]
    fn sink_stabilizes_when_larger_than_all_token_scores() {
        let cache = vec![KvEntry {
            latent: vec![-1000.0],
            rope: vec![0.0, 0.0],
        }];
        let probability =
            latent_attention_probabilities(&[1.0], &[0.0, 0.0], &cache, 1000.0, 1.0).unwrap()[0];
        assert_eq!(probability, 0.0);
    }
}
