//! Scalar, layer-streamed DeepSeek-V4 base-model correctness runtime.
//!
//! This is intentionally not a throughput backend. It exists as the executable specification
//! against which later SIMD/GPU kernels can be checked.

use super::compressor::{CompressorError, CompressorState, CompressorWeights};
use super::math::{
    bounded_swiglu, hyper_connection_head, hyper_connection_post, hyper_connection_pre, linear,
    paired_rope, round_to_bf16_in_place, route_sqrt_softplus, sparse_attention_with_sink,
    unit_rms_norm_in_place, DeepseekMathError, HyperConnectionMix,
};
use super::schema::{self, DeepseekV4Requirements, SchemaError};
use super::DeepseekV4Config;
use crate::config::ConfigError;
use crate::execution::install;
use crate::generation::CausalDecoder;
use crate::math::{
    normalized_hadamard, rms_norm, simulate_e2m1_activation, simulate_e4m3_activation, MathError,
    RouteChoice,
};
use crate::model::{WeightError, WeightMatrix};
use crate::profiling::{capture_context, span, ProfileStage};
use crate::runtime::cache::LayerLruCache;
use crate::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use crate::storage::{
    inspect_weight_matrix, load_i64_matrix_row, load_reference_matrix_row, load_reference_vector,
    load_weight_matrices, load_weight_matrix, streamed_reference_matvec, SafetensorError,
    TensorIndex, TensorLoadError, WeightLoadError,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_DEEPSEEK_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

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
    final_norm: Vec<f32>,
    head_hc: HcHeadWeights,
    context_limit: usize,
    expert_slots_per_layer: usize,
    maximum_expert_bytes: u64,
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
        let final_norm = load_reference_vector(&index, "norm.weight", config.hidden_size)?;
        let head_hc = HcHeadWeights::load(&index, &config, "")?;
        Ok(Self {
            instance_id: NEXT_DEEPSEEK_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            config,
            index,
            final_norm,
            head_hc,
            context_limit: options.context_limit,
            expert_slots_per_layer: options.expert_slots_per_layer,
            maximum_expert_bytes: options.maximum_expert_bytes,
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
        })
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut DeepseekRuntimeState,
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        let _profile = span(ProfileStage::DeepseekToken);
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
        let checkpoint = {
            let _profile = span(ProfileStage::DeepseekStateCheckpoint);
            state.attention.clone()
        };
        let result = self.forward_token_inner(token_index, position, state);
        match result {
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
            let _profile = span(ProfileStage::DeepseekStateCheckpoint);
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
        let mut hidden = self.load_hidden(token)?;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        for layer_id in 0..self.config.num_hidden_layers {
            let layer = {
                let _profile = span(ProfileStage::DeepseekLayerLoad);
                DecoderLayer::load(&self.index, &self.config, layer_id)?
            };
            let (next, routes) = layer.forward(
                &hidden,
                token,
                position,
                &mut state.attention[layer_id],
                &mut state.experts,
                &self.config,
            )?;
            hidden = next;
            routes_by_layer.push(routes);
        }
        self.finish_step(hidden, routes_by_layer)
    }

    fn prefill_tokens_inner(
        &self,
        tokens: &[usize],
        start_position: usize,
        state: &mut DeepseekRuntimeState,
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        let mut hidden_by_token = tokens
            .iter()
            .map(|&token| self.load_hidden(token))
            .collect::<Result<Vec<_>, _>>()?;
        let final_offset = tokens.len() - 1;
        let mut routes_by_layer = Vec::with_capacity(self.config.num_hidden_layers);
        for layer_id in 0..self.config.num_hidden_layers {
            let layer = {
                let _profile = span(ProfileStage::DeepseekLayerLoad);
                DecoderLayer::load(&self.index, &self.config, layer_id)?
            };
            let attention = &mut state.attention[layer_id];
            for (offset, hidden) in hidden_by_token.iter_mut().enumerate() {
                let current = std::mem::take(hidden);
                let (next, routes) = layer.forward(
                    &current,
                    tokens[offset],
                    start_position + offset,
                    attention,
                    &mut state.experts,
                    &self.config,
                )?;
                *hidden = next;
                if offset == final_offset {
                    routes_by_layer.push(routes);
                }
            }
        }
        let hidden = hidden_by_token
            .pop()
            .expect("a non-empty prefill has a final hidden state");
        self.finish_step(hidden, routes_by_layer)
    }

    fn load_hidden(&self, token: usize) -> Result<Vec<f32>, DeepseekRuntimeError> {
        let embedding = {
            let _profile = span(ProfileStage::DeepseekEmbeddingRead);
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
    ) -> Result<DeepseekRuntimeStep, DeepseekRuntimeError> {
        let hidden = {
            let _profile = span(ProfileStage::DeepseekFinalization);
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
            let _profile = span(ProfileStage::DeepseekLmHead);
            streamed_reference_matvec(
                &self.index,
                "head.weight",
                self.config.vocab_size,
                self.config.hidden_size,
                &hidden,
                4096,
            )?
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

#[derive(Debug, Clone)]
struct HcWeights {
    function: WeightMatrix,
    base: Vec<f32>,
    scale: Vec<f32>,
}

impl HcWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        prefix: &str,
        sublayer: &str,
    ) -> Result<Self, DeepseekRuntimeError> {
        let count = (2 + config.hc_mult) * config.hc_mult;
        Ok(Self {
            function: matrix(
                index,
                &format!("{prefix}.hc_{sublayer}_fn"),
                count,
                config.hc_mult * config.hidden_size,
            )?,
            base: load_reference_vector(index, &format!("{prefix}.hc_{sublayer}_base"), count)?,
            scale: load_reference_vector(index, &format!("{prefix}.hc_{sublayer}_scale"), 3)?,
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

#[derive(Debug)]
struct DecoderLayer {
    attn_hc: HcWeights,
    ffn_hc: HcWeights,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attention: AttentionWeights,
    moe: MoeWeights,
}

impl DecoderLayer {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
    ) -> Result<Self, DeepseekRuntimeError> {
        let prefix = format!("layers.{layer}");
        Ok(Self {
            attn_hc: HcWeights::load(index, config, &prefix, "attn")?,
            ffn_hc: HcWeights::load(index, config, &prefix, "ffn")?,
            attn_norm: load_reference_vector(
                index,
                &format!("{prefix}.attn_norm.weight"),
                config.hidden_size,
            )?,
            ffn_norm: load_reference_vector(
                index,
                &format!("{prefix}.ffn_norm.weight"),
                config.hidden_size,
            )?,
            attention: AttentionWeights::load(index, config, layer)?,
            moe: MoeWeights::load(index, config, layer)?,
        })
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
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), DeepseekRuntimeError> {
        let _layer_profile = span(ProfileStage::DeepseekLayer);
        let residual = hidden;
        let (collapsed, mix) = self.attn_hc.pre(hidden, config)?;
        let normalized = bf16_rms_norm(&collapsed, &self.attn_norm, config.rms_norm_eps as f32)?;
        let branch = {
            let _profile = span(ProfileStage::DeepseekAttention);
            self.attention
                .forward(&normalized, position, attention_state, config)?
        };
        let after_attention = hyper_connection_post(&branch, residual, config.hidden_size, &mix)?;

        let residual = after_attention;
        let (collapsed, mix) = self.ffn_hc.pre(&residual, config)?;
        let normalized = bf16_rms_norm(&collapsed, &self.ffn_norm, config.rms_norm_eps as f32)?;
        let (branch, routes) = {
            let _profile = span(ProfileStage::DeepseekMoe);
            self.moe.forward(&normalized, token, experts, config)?
        };
        Ok((
            hyper_connection_post(&branch, &residual, config.hidden_size, &mix)?,
            routes,
        ))
    }
}

#[derive(Debug)]
struct AttentionWeights {
    sink: Vec<f32>,
    wq_a: WeightMatrix,
    q_norm: Vec<f32>,
    wq_b: WeightMatrix,
    wkv: WeightMatrix,
    kv_norm: Vec<f32>,
    wo_a: WeightMatrix,
    wo_b: WeightMatrix,
    compressor: Option<CompressorWeights>,
    indexer: Option<IndexerWeights>,
    ratio: usize,
}

impl AttentionWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
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
                )
            })
            .transpose()?;
        let indexer = (ratio == 4)
            .then(|| IndexerWeights::load(index, config, &format!("{prefix}.indexer")))
            .transpose()?;
        Ok(Self {
            sink: load_reference_vector(
                index,
                &format!("{prefix}.attn_sink"),
                config.num_attention_heads,
            )?,
            wq_a: matrix(
                index,
                &format!("{prefix}.wq_a.weight"),
                config.q_lora_rank,
                config.hidden_size,
            )?,
            q_norm: load_reference_vector(
                index,
                &format!("{prefix}.q_norm.weight"),
                config.q_lora_rank,
            )?,
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
            kv_norm: load_reference_vector(
                index,
                &format!("{prefix}.kv_norm.weight"),
                config.head_dim,
            )?,
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

    fn forward(
        &self,
        input: &[f32],
        position: usize,
        state: &mut AttentionState,
        config: &DeepseekV4Config,
    ) -> Result<Vec<f32>, DeepseekRuntimeError> {
        if state.next_position != position || state.ratio != self.ratio {
            return Err(DeepseekRuntimeError::Invalid(
                "attention state position/ratio mismatch".to_owned(),
            ));
        }
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
        let mut qr = linear(&self.wq_a, input)?;
        qr = bf16_rms_norm(&qr, &self.q_norm, config.rms_norm_eps as f32)?;
        let mut query = linear(&self.wq_b, &qr)?;
        unit_rms_norm_in_place(&mut query, config.head_dim, config.rms_norm_eps as f32)?;
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

        let mut local_kv = linear(&self.wkv, input)?;
        local_kv = bf16_rms_norm(&local_kv, &self.kv_norm, config.rms_norm_eps as f32)?;
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

        let compressed_selection =
            if let (Some(indexer), Some(indexer_state)) = (&self.indexer, &mut state.indexer) {
                indexer.forward(input, &qr, position, indexer_state, config)?
            } else {
                Vec::new()
            };
        if let (Some(compressor), Some(compressor_state)) =
            (&self.compressor, &mut state.compressor)
        {
            let _ = compressor.forward_token(
                input,
                position,
                compressor_state,
                rope_base,
                rope_original,
                factor,
                config.rope_scaling.beta_fast,
                config.rope_scaling.beta_slow,
            )?;
        }

        let mut selected = window_indices(position, config.sliding_window);
        if self.ratio == 4 {
            selected.extend(compressed_selection);
        } else if self.ratio > 0 {
            let count = (position + 1) / self.ratio;
            selected.extend((0..count).map(|index| config.sliding_window + index));
        }
        let mut cache = state
            .local
            .iter()
            .map(|entry| entry.clone().unwrap_or_else(|| vec![0.0; config.head_dim]))
            .collect::<Vec<_>>();
        if let Some(compressor) = &state.compressor {
            cache.extend(compressor.compressed().iter().cloned());
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
        let heads_per_group = config.num_attention_heads / config.o_groups;
        let group_width = heads_per_group * config.head_dim;
        let mut low_rank = Vec::with_capacity(config.o_groups * config.o_lora_rank);
        for group in 0..config.o_groups {
            // The official reference explicitly dequantizes `wo_a` and applies this grouped
            // einsum without activation-side FP8 simulation.
            let mut group_output = self.wo_a.matvec_rows(
                group * config.o_lora_rank,
                config.o_lora_rank,
                &output[group * group_width..(group + 1) * group_width],
            )?;
            round_to_bf16_in_place(&mut group_output)?;
            low_rank.extend(group_output);
        }
        state.next_position += 1;
        Ok(linear(&self.wo_b, &low_rank)?)
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
}

#[derive(Debug)]
struct IndexerWeights {
    wq_b: WeightMatrix,
    weights_proj: WeightMatrix,
    compressor: CompressorWeights,
}

impl IndexerWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        prefix: &str,
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
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        qr: &[f32],
        position: usize,
        state: &mut CompressorState,
        config: &DeepseekV4Config,
    ) -> Result<Vec<usize>, DeepseekRuntimeError> {
        let mut query = linear(&self.wq_b, qr)?;
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
        let _ = self.compressor.forward_token(
            input,
            position,
            state,
            config.compress_rope_theta as f32,
            Some(config.rope_scaling.original_max_position_embeddings),
            config.rope_scaling.factor as f32,
            config.rope_scaling.beta_fast,
            config.rope_scaling.beta_slow,
        )?;
        let weight_scale = (config.index_head_dim as f32).sqrt().recip()
            * (config.index_n_heads as f32).sqrt().recip();
        let mut weights = linear(&self.weights_proj, input)?;
        round_to_bf16_in_place(&mut weights)?;
        for value in &mut weights {
            *value *= weight_scale;
        }
        round_to_bf16_in_place(&mut weights)?;
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
    correction_bias: Option<Vec<f32>>,
    shared: DeepExpert,
}

impl MoeWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV4Config,
        layer: usize,
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
            correction_bias: (layer >= config.num_hash_layers)
                .then(|| {
                    load_reference_vector(
                        index,
                        &format!("{prefix}.gate.bias"),
                        config.n_routed_experts,
                    )
                })
                .transpose()?,
            shared: DeepExpert::load(index, &format!("{prefix}.shared_experts"), config, u64::MAX)?
                .0,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        token: usize,
        experts: &mut DeepExpertStore,
        config: &DeepseekV4Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), DeepseekRuntimeError> {
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
        let mut output = vec![0.0f32; config.hidden_size];
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
        let loaded = experts.acquire_batch(self.layer, &expert_ids)?;
        for ((_, route_weight), expert) in execution_order.into_iter().zip(loaded) {
            let computed = {
                let _profile = span(ProfileStage::DeepseekExpertCompute);
                expert.forward(input, Some(route_weight))?
            };
            for (output, expert) in output.iter_mut().zip(computed) {
                *output += expert;
            }
        }
        // The official path accumulates routed experts in FP32, adds the shared expert last,
        // and only then casts the combined MoE output back to BF16.
        let shared = self.shared.forward(input, None)?;
        for (output, shared) in output.iter_mut().zip(shared) {
            *output += shared;
        }
        round_to_bf16_in_place(&mut output)?;
        Ok((output, routes))
    }
}

#[derive(Debug)]
struct DeepExpert {
    w1: WeightMatrix,
    w2: WeightMatrix,
    w3: WeightMatrix,
    limit: f32,
}

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

    fn forward(
        &self,
        input: &[f32],
        route_weight: Option<f32>,
    ) -> Result<Vec<f32>, DeepseekRuntimeError> {
        let gate = linear(&self.w1, input)?;
        let up = linear(&self.w3, input)?;
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
}

#[derive(Debug)]
struct DeepExpertStore {
    index: Arc<TensorIndex>,
    config: DeepseekV4Config,
    maximum_bytes: u64,
    cache: LayerLruCache<Arc<DeepExpert>>,
    telemetry: ExpertTelemetry,
}

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
            cache: LayerLruCache::new(config.num_hidden_layers, slots_per_layer),
            telemetry: ExpertTelemetry::default(),
        })
    }

    fn acquire(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<Arc<DeepExpert>, DeepseekRuntimeError> {
        if layer >= self.config.num_hidden_layers || expert >= self.config.n_routed_experts {
            return Err(DeepseekRuntimeError::Invalid(
                "expert request is outside configured geometry".to_owned(),
            ));
        }
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        self.cache.access(
            &mut self.telemetry,
            layer,
            expert,
            || {
                let mut profile = span(ProfileStage::DeepseekExpertLoad);
                let loaded = DeepExpert::load(
                    &index,
                    &format!("layers.{layer}.ffn.experts.{expert}"),
                    &config,
                    maximum,
                )?;
                profile.add_logical_bytes(loaded.2);
                Ok((Arc::new(loaded.0), loaded.1, loaded.2))
            },
            |expert| Ok(Arc::clone(expert)),
        )
    }

    fn acquire_batch(
        &mut self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<Arc<DeepExpert>>, DeepseekRuntimeError> {
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
            return experts
                .iter()
                .map(|&expert| self.acquire(layer, expert))
                .collect();
        }

        let mut missing = Vec::new();
        for &expert in experts {
            if !self.cache.contains(layer, expert) && !missing.contains(&expert) {
                missing.push(expert);
            }
        }
        let index = Arc::clone(&self.index);
        let config = self.config.clone();
        let maximum = self.maximum_bytes;
        let profile_context = capture_context();
        let loaded = install(|| {
            missing
                .par_iter()
                .map(|&expert| {
                    profile_context.enter(|| {
                        let mut profile = span(ProfileStage::DeepseekExpertLoad);
                        let loaded = DeepExpert::load(
                            &index,
                            &format!("layers.{layer}.ffn.experts.{expert}"),
                            &config,
                            maximum,
                        );
                        if let Ok((_, _, read_bytes)) = &loaded {
                            profile.add_logical_bytes(*read_bytes);
                        }
                        loaded.map(|(value, bytes, read_bytes)| {
                            (expert, (Arc::new(value), bytes, read_bytes))
                        })
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
            .collect()
    }
}

fn load_compressor(
    index: &TensorIndex,
    config: &DeepseekV4Config,
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    rotate: bool,
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
        load_reference_vector(index, &format!("{prefix}.norm.weight"), head_dim)?,
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
    use std::time::{SystemTime, UNIX_EPOCH};

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
            max_position_embeddings: 8,
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
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "urbilateria_deepseek_runtime_{}_{}",
            std::process::id(),
            nonce
        ))
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
        let layer = DecoderLayer::load(&index, &config, 0).unwrap();
        let mut attention = AttentionState::new(&config, 0).unwrap();
        let mut experts = DeepExpertStore::new(Arc::clone(&index), &config, 0, u64::MAX).unwrap();

        let (output, routes) = layer
            .forward(&hidden, 0, 0, &mut attention, &mut experts, &config)
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
        let model = DeepseekRuntimeModel::load(&directory, options()).unwrap();
        let prompt = [oracle.token as u32, oracle.token as u32];

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
        assert_eq!(
            batched_state.expert_telemetry(),
            sequential_state.expert_telemetry()
        );

        let next_sequential = model
            .forward_token(oracle.token as u32, &mut sequential_state)
            .unwrap();
        let next_batched = model
            .forward_token(oracle.token as u32, &mut batched_state)
            .unwrap();
        assert_eq!(next_batched, next_sequential);

        let mut profiled_state = model.new_state().unwrap();
        let profile = ProfileSession::start();
        let logits = CausalDecoder::prefill(&model, &prompt, &mut profiled_state).unwrap();
        let report = profile.finish();
        assert_eq!(logits, sequential.logits);
        assert_eq!(report.stage(ProfileStage::Prefill).unwrap().calls, 1);
        assert_eq!(
            report.stage(ProfileStage::DeepseekLayerLoad).unwrap().calls,
            model.config().num_hidden_layers as u64
        );
        assert_eq!(
            report.stage(ProfileStage::DeepseekLayer).unwrap().calls,
            (model.config().num_hidden_layers * prompt.len()) as u64
        );
        assert_eq!(report.stage(ProfileStage::DeepseekLmHead).unwrap().calls, 1);
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
}
