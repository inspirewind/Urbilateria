//! Scalar, layer-streamed DeepSeek-V4.1 base text runtime.
//!
//! DSpark and vision are intentionally outside this base autoregressive path. Every native dtype
//! boundary, Single-Pass mHC shift, Engram lookup, and CSA2 source handoff remains explicit.

use super::compressor::{CompressorError, CompressorState};
use super::engram::{EngramError, EngramTable, NgramHashState};
use super::indexer::{select_candidate_blocks, select_positions, IndexerError};
use super::kv::{IndexKeyCache, IndexKeyRow, KvCacheError, MainKvCache, WindowKvCache};
use super::math::{
    bf16_rms_norm, bounded_swiglu_reference as bounded_swiglu, collapse_bf16, engram_inject,
    linear, paired_rope_reference as paired_rope, post_bf16,
    route_sqrt_softplus_reference as route_sqrt_softplus, sparse_attention, V41MathError,
};
use super::mhc::{identity_pre_mix, single_pass_pre};
use super::schema::{self, DeepseekV41Requirements, SchemaError};
use super::DeepseekV41Config;
use crate::config::ConfigError;
use crate::execution::spawn_io;
use crate::generation::CausalDecoder;
use crate::math::RouteChoice;
use crate::model::{WeightError, WeightMatrix};
use crate::models::deepseek_v4::math::{
    round_to_bf16_in_place, DeepseekMathError, HyperConnectionMix,
};
use crate::profiling::{capture_context, span, ProfileStage};
use crate::runtime::cache::LayerLruCache;
use crate::runtime::{ExpertTelemetry, RuntimeLoadOptions};
use crate::storage::{
    inspect_weight_matrix, load_reference_matrix_row, load_reference_values, load_reference_vector,
    load_weight_matrices, load_weight_matrix, streamed_reference_matvec_pipelined, SafetensorError,
    TensorIndex, TensorLoadError, WeightLoadError,
};
use crate::tokenizer::{ByteBpeTokenizer, TokenizerError};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub enum DeepseekV41RuntimeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Tensor(TensorLoadError),
    WeightLoad(WeightLoadError),
    Weight(WeightError),
    Math(V41MathError),
    DeepseekMath(DeepseekMathError),
    Compressor(CompressorError),
    Engram(EngramError),
    Indexer(IndexerError),
    Kv(KvCacheError),
    Schema(SchemaError),
    Tokenizer(TokenizerError),
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

impl fmt::Display for DeepseekV41RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Tensor(error) => error.fmt(formatter),
            Self::WeightLoad(error) => error.fmt(formatter),
            Self::Weight(error) => error.fmt(formatter),
            Self::Math(error) => error.fmt(formatter),
            Self::DeepseekMath(error) => error.fmt(formatter),
            Self::Compressor(error) => error.fmt(formatter),
            Self::Engram(error) => error.fmt(formatter),
            Self::Indexer(error) => error.fmt(formatter),
            Self::Kv(error) => error.fmt(formatter),
            Self::Schema(error) => error.fmt(formatter),
            Self::Tokenizer(error) => error.fmt(formatter),
            Self::Invalid(reason) => write!(formatter, "invalid DeepSeek-V4.1 runtime: {reason}"),
            Self::Budget {
                component,
                required,
                maximum,
            } => write!(
                formatter,
                "DeepSeek-V4.1 {component} needs {required} bytes, authorized maximum is {maximum}"
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

impl std::error::Error for DeepseekV41RuntimeError {}

type LayerForward = (Vec<f32>, Vec<f32>, Vec<RouteChoice>);

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for DeepseekV41RuntimeError {
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
from_error!(V41MathError, Math);
from_error!(DeepseekMathError, DeepseekMath);
from_error!(CompressorError, Compressor);
from_error!(EngramError, Engram);
from_error!(IndexerError, Indexer);
from_error!(KvCacheError, Kv);
from_error!(SchemaError, Schema);
from_error!(TokenizerError, Tokenizer);

#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekV41RuntimeStep {
    pub logits: Vec<f32>,
    pub routes_by_layer: Vec<Vec<RouteChoice>>,
}

#[derive(Debug)]
pub struct DeepseekV41RuntimeModel {
    instance_id: u64,
    config: DeepseekV41Config,
    index: Arc<TensorIndex>,
    final_norm: Vec<f32>,
    engram_hash_prototype: NgramHashState,
    context_limit: usize,
    expert_slots_per_layer: usize,
    maximum_expert_bytes: u64,
}

impl DeepseekV41RuntimeModel {
    pub fn inspect_requirements(
        model_dir: impl AsRef<Path>,
        context_limit: usize,
        expert_slots_per_layer: usize,
    ) -> Result<DeepseekV41Requirements, DeepseekV41RuntimeError> {
        let config = DeepseekV41Config::load(model_dir.as_ref())?;
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
    ) -> Result<Self, DeepseekV41RuntimeError> {
        if options.context_limit == 0
            || options.resident_budget_bytes == 0
            || options.kv_cache_budget_bytes == 0
            || options.maximum_expert_bytes == 0
        {
            return Err(DeepseekV41RuntimeError::Invalid(
                "resident/KV/per-expert budgets and context limit must be non-zero".to_owned(),
            ));
        }
        let config = DeepseekV41Config::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        let requirements = schema::inspect_requirements(
            &config,
            &index,
            options.context_limit,
            options.expert_slots_per_layer,
        )?;
        for (component, required, maximum) in [
            (
                "streamed layer peak",
                requirements.streamed_layer_bytes,
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
                return Err(DeepseekV41RuntimeError::Budget {
                    component,
                    required,
                    maximum,
                });
            }
        }
        let tokenizer = ByteBpeTokenizer::load(model_dir.as_ref())?;
        let engram_hash_prototype = NgramHashState::new(&config, &tokenizer)?;
        let final_norm =
            load_reference_vector(&index, "norm.weight", config.text_config.hidden_size)?;
        Ok(Self {
            instance_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            config,
            index: Arc::new(index),
            final_norm,
            engram_hash_prototype,
            context_limit: options.context_limit,
            expert_slots_per_layer: options.expert_slots_per_layer,
            maximum_expert_bytes: options.maximum_expert_bytes,
        })
    }

    pub fn new_state(&self) -> Result<DeepseekV41RuntimeState, DeepseekV41RuntimeError> {
        let text = &self.config.text_config;
        let attention = (0..text.num_hidden_layers)
            .map(|_| AttentionState::new(text.head_dim, text.sliding_window))
            .collect::<Result<Vec<_>, _>>()?;
        let mut sources = HashMap::new();
        for &layer in &text.kv_source_layer_ids {
            sources.insert(
                layer,
                SourceState::new(
                    text.compress_ratios[layer],
                    text.head_dim,
                    text.index_head_dim,
                )?,
            );
        }
        Ok(DeepseekV41RuntimeState {
            instance_id: self.instance_id,
            position: 0,
            attention,
            sources,
            engram_hash: self.engram_hash_prototype.clone(),
            experts: ExpertStore::new(
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
        state: &mut DeepseekV41RuntimeState,
    ) -> Result<DeepseekV41RuntimeStep, DeepseekV41RuntimeError> {
        let _profile = span(ProfileStage::DeepseekToken);
        self.validate_step(token, state)?;
        let checkpoint = {
            let _profile = span(ProfileStage::DeepseekStateCheckpoint);
            StateCheckpoint::capture(state)
        };
        let result = self.forward_token_inner(token, state);
        match result {
            Ok(step) => {
                state.position += 1;
                Ok(step)
            }
            Err(error) => {
                checkpoint.restore(state).map_err(|restore_error| {
                    DeepseekV41RuntimeError::Invalid(format!(
                        "token execution failed ({error}); state rollback also failed ({restore_error})"
                    ))
                })?;
                Err(error)
            }
        }
    }

    fn validate_step(
        &self,
        token: u32,
        state: &DeepseekV41RuntimeState,
    ) -> Result<(), DeepseekV41RuntimeError> {
        if state.instance_id != self.instance_id {
            return Err(DeepseekV41RuntimeError::Invalid(
                "state belongs to another runtime instance".to_owned(),
            ));
        }
        if token as usize >= self.config.text_config.vocab_size {
            return Err(DeepseekV41RuntimeError::TokenOutOfRange {
                token,
                vocabulary: self.config.text_config.vocab_size,
            });
        }
        if state.position >= self.context_limit {
            return Err(DeepseekV41RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        Ok(())
    }

    fn forward_token_inner(
        &self,
        token: u32,
        state: &mut DeepseekV41RuntimeState,
    ) -> Result<DeepseekV41RuntimeStep, DeepseekV41RuntimeError> {
        let text = &self.config.text_config;
        let position = state.position;
        let hashes = state.engram_hash.push(position, &[token], None)?;
        let hashes = &hashes[0];
        let embedding = load_reference_matrix_row(
            &self.index,
            "embed.weight",
            token as usize,
            text.vocab_size,
            text.hidden_size,
        )?;
        let mut hidden = Vec::with_capacity(text.hc_mult * text.hidden_size);
        for _ in 0..text.hc_mult {
            hidden.extend_from_slice(&embedding);
        }
        let mut incoming_pre = identity_pre_mix(text.hc_mult)?;
        let mut shared_step = SharedAttentionStep::default();
        let mut routes_by_layer = Vec::with_capacity(text.num_hidden_layers);
        for layer_id in 0..text.num_hidden_layers {
            let layer = {
                let _profile = span(ProfileStage::DeepseekLayerLoad);
                DecoderLayer::load(&self.index, &self.config, layer_id)?
            };
            if let Some(engram) = &layer.engram {
                let hash_index = text
                    .engram_layer_ids
                    .iter()
                    .position(|&id| id == layer_id)
                    .expect("Engram layer has a hash plane");
                hidden = engram.forward(&self.index, &hidden, &hashes[hash_index], &self.config)?;
            }
            let (next, next_pre, routes) = layer.forward(
                &hidden,
                &incoming_pre,
                position,
                &mut state.attention[layer_id],
                &mut state.sources,
                &mut shared_step,
                &mut state.experts,
                &self.config,
            )?;
            hidden = next;
            incoming_pre = next_pre;
            routes_by_layer.push(routes);
        }
        let hidden = collapse_bf16(&hidden, &incoming_pre, text.hidden_size)?;
        let hidden = bf16_rms_norm(&hidden, &self.final_norm, text.rms_norm_eps as f32)?;
        let _head_profile = span(ProfileStage::DeepseekLmHead);
        let logits = streamed_reference_matvec_pipelined(
            Arc::clone(&self.index),
            "head.weight",
            text.vocab_size,
            text.hidden_size,
            &hidden,
            4096,
        )?;
        Ok(DeepseekV41RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    pub fn config(&self) -> &DeepseekV41Config {
        &self.config
    }
}

impl CausalDecoder for DeepseekV41RuntimeModel {
    type State = DeepseekV41RuntimeState;
    type Error = DeepseekV41RuntimeError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        DeepseekV41RuntimeModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(DeepseekV41RuntimeModel::forward_token(self, token, state)?.logits)
    }
}

#[derive(Debug)]
pub struct DeepseekV41RuntimeState {
    instance_id: u64,
    position: usize,
    attention: Vec<AttentionState>,
    sources: HashMap<usize, SourceState>,
    engram_hash: NgramHashState,
    experts: ExpertStore,
}

impl DeepseekV41RuntimeState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn expert_telemetry(&self) -> &ExpertTelemetry {
        &self.experts.telemetry
    }

    pub fn cache_bytes(&self) -> usize {
        self.attention
            .iter()
            .map(AttentionState::storage_bytes)
            .sum::<usize>()
            + self
                .sources
                .values()
                .map(SourceState::storage_bytes)
                .sum::<usize>()
    }
}

#[derive(Debug)]
struct StateCheckpoint {
    attention: Vec<AttentionState>,
    sources: HashMap<usize, SourceCheckpoint>,
    engram_hash_len: usize,
}

#[derive(Debug, Clone)]
struct SourceCheckpoint {
    compressor: CompressorState,
    main_len: usize,
    index_len: usize,
}

impl StateCheckpoint {
    fn capture(state: &DeepseekV41RuntimeState) -> Self {
        Self {
            attention: state.attention.clone(),
            sources: state
                .sources
                .iter()
                .map(|(&owner, source)| {
                    (
                        owner,
                        SourceCheckpoint {
                            compressor: source.compressor.clone(),
                            main_len: source.main.len(),
                            index_len: source.index.len(),
                        },
                    )
                })
                .collect(),
            engram_hash_len: state.engram_hash.checkpoint(),
        }
    }

    fn restore(self, state: &mut DeepseekV41RuntimeState) -> Result<(), DeepseekV41RuntimeError> {
        state.attention = self.attention;
        for (owner, checkpoint) in self.sources {
            let source = state.sources.get_mut(&owner).ok_or_else(|| {
                DeepseekV41RuntimeError::Invalid(format!(
                    "KV source {owner} disappeared while restoring a failed token"
                ))
            })?;
            source.compressor = checkpoint.compressor;
            source.main.rollback(checkpoint.main_len)?;
            source.index.rollback(checkpoint.index_len)?;
        }
        state.engram_hash.rollback(self.engram_hash_len)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct AttentionState {
    window: WindowKvCache,
}

impl AttentionState {
    fn new(width: usize, window: usize) -> Result<Self, KvCacheError> {
        Ok(Self {
            window: WindowKvCache::new(width, window)?,
        })
    }

    fn storage_bytes(&self) -> usize {
        self.window.storage_bytes()
    }
}

#[derive(Debug, Clone)]
struct SourceState {
    ratio: usize,
    compressor: CompressorState,
    main: MainKvCache,
    index: IndexKeyCache,
}

impl SourceState {
    fn new(
        ratio: usize,
        head_dim: usize,
        index_dim: usize,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        Ok(Self {
            ratio,
            compressor: CompressorState::new(ratio, head_dim)?,
            main: MainKvCache::new(head_dim)?,
            index: IndexKeyCache::new(index_dim)?,
        })
    }

    fn storage_bytes(&self) -> usize {
        self.compressor.state_bytes() + self.main.storage_bytes() + self.index.storage_bytes()
    }
}

#[derive(Debug, Default)]
struct SharedAttentionStep {
    selected: Vec<usize>,
    candidates: Option<Vec<bool>>,
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
        config: &DeepseekV41Config,
        prefix: &str,
        sublayer: &str,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let count = (2 + text.hc_mult) * text.hc_mult;
        Ok(Self {
            function: matrix(
                index,
                &format!("{prefix}.hc_{sublayer}_fn"),
                count,
                text.hc_mult * text.hidden_size,
            )?,
            base: load_reference_vector(index, &format!("{prefix}.hc_{sublayer}_base"), count)?,
            scale: load_reference_vector(index, &format!("{prefix}.hc_{sublayer}_scale"), 3)?,
        })
    }

    fn predict(
        &self,
        hidden: &[f32],
        incoming_pre: &[f32],
        config: &DeepseekV41Config,
    ) -> Result<(Vec<f32>, HyperConnectionMix), DeepseekV41RuntimeError> {
        let text = &config.text_config;
        Ok(single_pass_pre(
            hidden,
            incoming_pre,
            text.hidden_size,
            text.hc_mult,
            &self.function,
            &self.scale,
            &self.base,
            text.hc_sinkhorn_iters,
            text.rms_norm_eps as f32,
            text.hc_eps as f32,
        )?)
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
    engram: Option<EngramWeights>,
}

impl DecoderLayer {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV41Config,
        layer: usize,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let prefix = format!("layers.{layer}");
        Ok(Self {
            attn_hc: HcWeights::load(index, config, &prefix, "attn")?,
            ffn_hc: HcWeights::load(index, config, &prefix, "ffn")?,
            attn_norm: load_reference_vector(
                index,
                &format!("{prefix}.attn_norm.weight"),
                text.hidden_size,
            )?,
            ffn_norm: load_reference_vector(
                index,
                &format!("{prefix}.ffn_norm.weight"),
                text.hidden_size,
            )?,
            attention: AttentionWeights::load(index, config, layer)?,
            moe: MoeWeights::load(index, config, layer)?,
            engram: text
                .engram_layer_ids
                .contains(&layer)
                .then(|| EngramWeights::load(index, config, layer))
                .transpose()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &[f32],
        incoming_pre: &[f32],
        position: usize,
        attention_state: &mut AttentionState,
        sources: &mut HashMap<usize, SourceState>,
        shared_step: &mut SharedAttentionStep,
        experts: &mut ExpertStore,
        config: &DeepseekV41Config,
    ) -> Result<LayerForward, DeepseekV41RuntimeError> {
        let _layer_profile = span(ProfileStage::DeepseekLayer);
        let (collapsed, attn_mix) = self.attn_hc.predict(hidden, incoming_pre, config)?;
        let normalized = bf16_rms_norm(
            &collapsed,
            &self.attn_norm,
            config.text_config.rms_norm_eps as f32,
        )?;
        let attention_branch = self.attention.forward(
            &normalized,
            position,
            attention_state,
            sources,
            shared_step,
            config,
        )?;
        let after_attention = post_bf16(
            &attention_branch,
            hidden,
            config.text_config.hidden_size,
            &attn_mix,
        )?;

        // The attention predictor supplies the FFN's pre-mix; the FFN predictor supplies the next
        // layer's attention pre-mix.
        let (collapsed, ffn_mix) = self
            .ffn_hc
            .predict(&after_attention, &attn_mix.pre, config)?;
        let normalized = bf16_rms_norm(
            &collapsed,
            &self.ffn_norm,
            config.text_config.rms_norm_eps as f32,
        )?;
        let (ffn_branch, routes) = self.moe.forward(&normalized, experts, config)?;
        let output = post_bf16(
            &ffn_branch,
            &after_attention,
            config.text_config.hidden_size,
            &ffn_mix,
        )?;
        Ok((output, ffn_mix.pre, routes))
    }
}

#[derive(Debug)]
struct EngramWeights {
    layer: usize,
    rows: usize,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    wkv: WeightMatrix,
}

impl EngramWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV41Config,
        layer: usize,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let position = text
            .engram_layer_ids
            .iter()
            .position(|&id| id == layer)
            .ok_or_else(|| DeepseekV41RuntimeError::Invalid("unknown Engram layer".to_owned()))?;
        let prefix = format!("layers.{layer}.engram");
        let gate_width = text.hc_mult * text.hidden_size;
        let hash_width =
            (text.engram_max_ngram_size - 1) * text.engram_n_heads * text.engram_head_dim;
        Ok(Self {
            layer,
            rows: text.engram_num_embeddings[position],
            q_weight: load_reference_values_exact(
                index,
                &format!("{prefix}.q_weight"),
                gate_width,
            )?,
            k_weight: load_reference_values_exact(
                index,
                &format!("{prefix}.k_weight"),
                gate_width,
            )?,
            wkv: matrix(
                index,
                &format!("{prefix}.wkv.weight"),
                text.hidden_size * (text.hc_mult + 1),
                hash_width,
            )?,
        })
    }

    fn forward(
        &self,
        index: &TensorIndex,
        hidden: &[f32],
        hashes: &[u64],
        config: &DeepseekV41Config,
    ) -> Result<Vec<f32>, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let table = EngramTable::open(index, self.layer, self.rows, text.engram_head_dim)?;
        let row_ids = hashes
            .iter()
            .map(|&row| {
                usize::try_from(row).map_err(|_| {
                    DeepseekV41RuntimeError::Invalid("Engram row ID does not fit usize".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rows = table.read_rows(&row_ids)?;
        let mut flattened = rows.into_iter().flatten().collect::<Vec<_>>();
        round_to_bf16_in_place(&mut flattened)?;
        let key_value = linear(&self.wkv, &flattened)?;
        Ok(engram_inject(
            hidden,
            &key_value,
            &self.q_weight,
            &self.k_weight,
            text.hidden_size,
            text.hc_mult,
            text.rms_norm_eps as f32,
        )?)
    }
}

#[derive(Debug)]
struct CompressorWeights {
    ratio: usize,
    wkv: WeightMatrix,
    wgate: Option<WeightMatrix>,
    norm: Vec<f32>,
}

impl CompressorWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV41Config,
        layer: usize,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let ratio = text.compress_ratios[layer];
        let prefix = format!("layers.{layer}.attn.compressor");
        Ok(Self {
            ratio,
            wkv: matrix(
                index,
                &format!("{prefix}.wkv.weight"),
                text.head_dim,
                text.hidden_size,
            )?,
            wgate: (ratio > 1)
                .then(|| {
                    matrix(
                        index,
                        &format!("{prefix}.wgate.weight"),
                        text.head_dim,
                        text.hidden_size,
                    )
                })
                .transpose()?,
            norm: load_reference_vector(index, &format!("{prefix}.norm.weight"), text.head_dim)?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        position: usize,
        state: &mut CompressorState,
        config: &DeepseekV41Config,
    ) -> Result<Option<Vec<f32>>, DeepseekV41RuntimeError> {
        let mut projected = self.wkv.matvec_fp32_accum(input)?;
        if self.ratio == 1 {
            round_to_bf16_in_place(&mut projected)?;
        }
        let scores = self
            .wgate
            .as_ref()
            .map(|weight| weight.matvec_fp32_accum(input))
            .transpose()?;
        let Some(mut latent) = state.push(position, &projected, scores.as_deref())? else {
            return Ok(None);
        };
        round_to_bf16_in_place(&mut latent)?;
        Ok(Some(bf16_rms_norm(
            &latent,
            &self.norm,
            config.text_config.rms_norm_eps as f32,
        )?))
    }
}

#[derive(Debug)]
struct AttentionWeights {
    layer: usize,
    ratio: usize,
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
}

impl AttentionWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV41Config,
        layer: usize,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let prefix = format!("layers.{layer}.attn");
        let ratio = text.compress_ratios[layer];
        Ok(Self {
            layer,
            ratio,
            sink: load_reference_vector(
                index,
                &format!("{prefix}.attn_sink"),
                text.num_attention_heads,
            )?,
            wq_a: matrix(
                index,
                &format!("{prefix}.wq_a.weight"),
                text.q_lora_rank,
                text.hidden_size,
            )?,
            q_norm: load_reference_vector(
                index,
                &format!("{prefix}.q_norm.weight"),
                text.q_lora_rank,
            )?,
            wq_b: matrix(
                index,
                &format!("{prefix}.wq_b.weight"),
                text.num_attention_heads * text.head_dim,
                text.q_lora_rank,
            )?,
            wkv: matrix(
                index,
                &format!("{prefix}.wkv.weight"),
                text.head_dim,
                text.hidden_size,
            )?,
            kv_norm: load_reference_vector(
                index,
                &format!("{prefix}.kv_norm.weight"),
                text.head_dim,
            )?,
            wo_a: matrix(
                index,
                &format!("{prefix}.wo_a.weight"),
                text.o_groups * text.o_lora_rank,
                text.num_attention_heads * text.head_dim / text.o_groups,
            )?,
            wo_b: matrix(
                index,
                &format!("{prefix}.wo_b.weight"),
                text.hidden_size,
                text.o_groups * text.o_lora_rank,
            )?,
            compressor: text
                .kv_source_layer_ids
                .contains(&layer)
                .then(|| CompressorWeights::load(index, config, layer))
                .transpose()?,
            indexer: text
                .index_source_layer_ids
                .contains(&layer)
                .then(|| IndexerWeights::load(index, config, layer))
                .transpose()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input: &[f32],
        position: usize,
        state: &mut AttentionState,
        sources: &mut HashMap<usize, SourceState>,
        shared: &mut SharedAttentionStep,
        config: &DeepseekV41Config,
    ) -> Result<Vec<f32>, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        if state.window.next_position() != position {
            return Err(DeepseekV41RuntimeError::Invalid(format!(
                "layer {} window expected position {}, got {position}",
                self.layer,
                state.window.next_position()
            )));
        }
        let mut qr = linear(&self.wq_a, input)?;
        qr = bf16_rms_norm(&qr, &self.q_norm, text.rms_norm_eps as f32)?;
        let mut query = linear(&self.wq_b, &qr)?;
        let (rope_base, rope_original, rope_factor) = rope_parameters(self.ratio, text);
        for head in query.chunks_exact_mut(text.head_dim) {
            paired_rope(
                &mut head[text.head_dim - text.qk_rope_head_dim..],
                position,
                rope_base,
                rope_original,
                rope_factor,
                text.rope_scaling.beta_fast,
                text.rope_scaling.beta_slow,
                false,
            )?;
        }

        let mut local = linear(&self.wkv, input)?;
        local = bf16_rms_norm(&local, &self.kv_norm, text.rms_norm_eps as f32)?;
        paired_rope(
            &mut local[text.head_dim - text.qk_rope_head_dim..],
            position,
            rope_base,
            rope_original,
            rope_factor,
            text.rope_scaling.beta_fast,
            text.rope_scaling.beta_slow,
            false,
        )?;
        state.window.append(position, &local)?;

        if self.ratio > 0 {
            let owner = config.kv_source_for(self.layer).ok_or_else(|| {
                DeepseekV41RuntimeError::Invalid(format!(
                    "layer {} has compression ratio {} but no preceding KV source",
                    self.layer, self.ratio
                ))
            })?;
            let source = sources.get_mut(&owner).ok_or_else(|| {
                DeepseekV41RuntimeError::Invalid(format!("missing state for KV source {owner}"))
            })?;
            if source.ratio != self.ratio {
                return Err(DeepseekV41RuntimeError::Invalid(
                    "consumer compression ratio differs from its KV source".to_owned(),
                ));
            }
            let latent = if let Some(compressor) = &self.compressor {
                compressor.forward(input, position, &mut source.compressor, config)?
            } else {
                None
            };
            if let Some(indexer) = &self.indexer {
                indexer.forward(
                    input,
                    &qr,
                    latent.as_deref(),
                    position,
                    source,
                    shared,
                    config,
                )?;
            }
            if let Some(mut latent) = latent {
                let compressed_position = source.main.len() * self.ratio;
                paired_rope(
                    &mut latent[text.head_dim - text.qk_rope_head_dim..],
                    compressed_position,
                    text.compress_rope_theta as f32,
                    Some(text.rope_scaling.original_max_position_embeddings),
                    text.rope_scaling.factor as f32,
                    text.rope_scaling.beta_fast,
                    text.rope_scaling.beta_slow,
                    false,
                )?;
                source.main.push(&latent)?;
            }
        }

        let mut cache = Vec::<Vec<f32>>::new();
        for absolute in window_positions(position, text.sliding_window) {
            let row = state.window.get(absolute).ok_or_else(|| {
                DeepseekV41RuntimeError::Invalid(format!(
                    "window slot for absolute position {absolute} is absent"
                ))
            })?;
            cache.push(row.decode()?);
        }
        if self.ratio > 0 {
            let owner = config.kv_source_for(self.layer).expect("validated source");
            let source = sources.get(&owner).expect("validated source state");
            for &compressed in &shared.selected {
                let row = source.main.row(compressed).ok_or_else(|| {
                    DeepseekV41RuntimeError::Invalid(format!(
                        "selected compressed row {compressed} is unavailable from source {owner}"
                    ))
                })?;
                cache.push(row.decode());
            }
        }
        let selected = (0..cache.len()).collect::<Vec<_>>();
        let mut output = sparse_attention(
            &query,
            text.num_attention_heads,
            text.head_dim,
            &cache,
            &selected,
            &self.sink,
            (text.head_dim as f32).sqrt().recip(),
        )?;
        for head in output.chunks_exact_mut(text.head_dim) {
            paired_rope(
                &mut head[text.head_dim - text.qk_rope_head_dim..],
                position,
                rope_base,
                rope_original,
                rope_factor,
                text.rope_scaling.beta_fast,
                text.rope_scaling.beta_slow,
                true,
            )?;
        }
        let heads_per_group = text.num_attention_heads / text.o_groups;
        let group_width = heads_per_group * text.head_dim;
        let mut low_rank = Vec::with_capacity(text.o_groups * text.o_lora_rank);
        for group in 0..text.o_groups {
            let mut projected = self.wo_a.matvec_rows_fp32(
                group * text.o_lora_rank,
                text.o_lora_rank,
                &output[group * group_width..(group + 1) * group_width],
            )?;
            round_to_bf16_in_place(&mut projected)?;
            low_rank.extend(projected);
        }
        Ok(linear(&self.wo_b, &low_rank)?)
    }
}

#[derive(Debug)]
struct IndexerWeights {
    layer: usize,
    wq_b: WeightMatrix,
    weights_proj: WeightMatrix,
    wk: Option<WeightMatrix>,
    k_norm: Option<Vec<f32>>,
}

impl IndexerWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV41Config,
        layer: usize,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let prefix = format!("layers.{layer}.attn.indexer");
        let owns_k = text.kv_source_layer_ids.contains(&layer);
        Ok(Self {
            layer,
            wq_b: matrix(
                index,
                &format!("{prefix}.wq_b.weight"),
                text.index_n_heads * text.index_head_dim,
                text.q_lora_rank,
            )?,
            weights_proj: matrix(
                index,
                &format!("{prefix}.weights_proj.weight"),
                text.index_n_heads,
                text.hidden_size,
            )?,
            wk: owns_k
                .then(|| {
                    matrix(
                        index,
                        &format!("{prefix}.wk.weight"),
                        text.index_head_dim,
                        text.head_dim,
                    )
                })
                .transpose()?,
            k_norm: owns_k
                .then(|| {
                    load_reference_vector(
                        index,
                        &format!("{prefix}.k_norm.weight"),
                        text.index_head_dim,
                    )
                })
                .transpose()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input: &[f32],
        qr: &[f32],
        latent: Option<&[f32]>,
        position: usize,
        source: &mut SourceState,
        shared: &mut SharedAttentionStep,
        config: &DeepseekV41Config,
    ) -> Result<(), DeepseekV41RuntimeError> {
        let text = &config.text_config;
        if let (Some(wk), Some(k_norm), Some(latent)) = (&self.wk, &self.k_norm, latent) {
            let mut key = wk.matvec_fp32_accum(latent)?;
            round_to_bf16_in_place(&mut key)?;
            key = bf16_rms_norm(&key, k_norm, text.rms_norm_eps as f32)?;
            let compressed_position = source.index.len() * source.ratio;
            paired_rope(
                &mut key[text.index_head_dim - text.qk_rope_head_dim..],
                compressed_position,
                text.compress_rope_theta as f32,
                Some(text.rope_scaling.original_max_position_embeddings),
                text.rope_scaling.factor as f32,
                text.rope_scaling.beta_fast,
                text.rope_scaling.beta_slow,
                false,
            )?;
            source.index.push(&key)?;
        } else if self.wk.is_some() != latent.is_some() && latent.is_some() {
            return Err(DeepseekV41RuntimeError::Invalid(
                "index owner has incomplete key projection weights".to_owned(),
            ));
        }

        let expected = (position + 1) / source.ratio;
        if source.index.len() != expected {
            return Err(DeepseekV41RuntimeError::Invalid(format!(
                "index source {} has {} keys at position {position}, expected {expected}",
                self.layer,
                source.index.len()
            )));
        }
        if expected == 0 {
            shared.selected.clear();
            return Ok(());
        }

        let mut query = linear(&self.wq_b, qr)?;
        for head in query.chunks_exact_mut(text.index_head_dim) {
            paired_rope(
                &mut head[text.index_head_dim - text.qk_rope_head_dim..],
                position,
                text.compress_rope_theta as f32,
                Some(text.rope_scaling.original_max_position_embeddings),
                text.rope_scaling.factor as f32,
                text.rope_scaling.beta_fast,
                text.rope_scaling.beta_slow,
                false,
            )?;
        }
        let query = IndexKeyRow::quantize(&query)?.decode()?;
        let mut weights = self.weights_proj.matvec_fp32_accum(input)?;
        round_to_bf16_in_place(&mut weights)?;
        let scale = (text.index_head_dim as f32).sqrt().recip()
            * (text.index_n_heads as f32).sqrt().recip();
        for weight in &mut weights {
            *weight *= scale;
        }
        round_to_bf16_in_place(&mut weights)?;

        let mut scores = Vec::with_capacity(source.index.len());
        for row in 0..source.index.len() {
            let key = source
                .index
                .row(row)
                .expect("bounded index row exists")
                .decode()?;
            let mut score = 0.0f32;
            for head in 0..text.index_n_heads {
                let q = &query[head * text.index_head_dim..(head + 1) * text.index_head_dim];
                let dot = q
                    .iter()
                    .zip(&key)
                    .map(|(&left, &right)| left * right)
                    .sum::<f32>();
                score += dot.max(0.0) * weights[head];
            }
            scores.push(score);
        }

        if self.layer == text.candidate_source_layer_id {
            shared.candidates = Some(select_candidate_blocks(
                &scores,
                expected,
                text.candidate_topk_blocks,
                text.candidate_block_size,
            )?);
        }
        let candidates = (self.layer > text.candidate_source_layer_id)
            .then_some(shared.candidates.as_deref())
            .flatten()
            .ok_or_else(|| {
                DeepseekV41RuntimeError::Invalid(
                    "decoder index source ran before candidate source publication".to_owned(),
                )
            });
        let candidates = if self.layer > text.candidate_source_layer_id {
            Some(candidates?)
        } else {
            None
        };
        shared.selected = select_positions(&scores, expected, text.index_topk, 0, candidates)?
            .into_iter()
            .map(|position| {
                position.ok_or_else(|| {
                    DeepseekV41RuntimeError::Invalid(
                        "tokenwise index selection produced an unreachable slot".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
}

#[derive(Debug)]
struct MoeWeights {
    layer: usize,
    router: WeightMatrix,
    correction_bias: Vec<f32>,
    shared: Expert,
}

impl MoeWeights {
    fn load(
        index: &TensorIndex,
        config: &DeepseekV41Config,
        layer: usize,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let prefix = format!("layers.{layer}.ffn");
        Ok(Self {
            layer,
            router: matrix(
                index,
                &format!("{prefix}.gate.weight"),
                text.n_routed_experts,
                text.hidden_size,
            )?,
            correction_bias: load_reference_vector(
                index,
                &format!("{prefix}.gate.bias"),
                text.n_routed_experts,
            )?,
            shared: Expert::load(index, &format!("{prefix}.shared_experts"), config, u64::MAX)?.0,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        experts: &mut ExpertStore,
        config: &DeepseekV41Config,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), DeepseekV41RuntimeError> {
        let _profile = span(ProfileStage::DeepseekMoe);
        let text = &config.text_config;
        let logits = self.router.matvec_fp32_accum(input)?;
        let routes = route_sqrt_softplus(
            &logits,
            Some(&self.correction_bias),
            None,
            text.num_experts_per_tok,
            text.routed_scaling_factor as f32,
        )?;
        let mut execution = routes
            .iter()
            .map(|route| (route.expert, route.weight))
            .collect::<Vec<_>>();
        execution.sort_unstable_by_key(|&(expert, _)| expert);
        let ids = execution
            .iter()
            .map(|&(expert, _)| expert)
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; text.hidden_size];
        let shared = if experts.cache.can_insert_without_eviction(self.layer, &ids) {
            // The shared expert is already resident. Compute it while routed weights are fetched
            // on the existing bounded I/O pool, then retain the original routed/shared sum order.
            let (loaded, shared) = experts
                .acquire_batch_while(self.layer, &ids, || self.shared.forward(input, None))?;
            for ((_, weight), expert) in execution.into_iter().zip(loaded) {
                let computed = expert.forward(input, Some(weight))?;
                for (output, value) in output.iter_mut().zip(computed) {
                    *output += value;
                }
            }
            shared
        } else {
            // A full or small cache authorizes only one transient expert. Execute and drop each
            // handle before acquiring the next; collecting all routed Arcs would pin evictions.
            for (id, weight) in execution {
                let expert = experts.acquire(self.layer, id)?;
                let computed = expert.forward(input, Some(weight))?;
                for (output, value) in output.iter_mut().zip(computed) {
                    *output += value;
                }
            }
            self.shared.forward(input, None)?
        };
        for (output, value) in output.iter_mut().zip(shared) {
            *output += value;
        }
        round_to_bf16_in_place(&mut output)?;
        Ok((output, routes))
    }
}

#[derive(Debug)]
struct Expert {
    w1: WeightMatrix,
    w2: WeightMatrix,
    w3: WeightMatrix,
    limit: f32,
}

impl Expert {
    fn load(
        index: &TensorIndex,
        prefix: &str,
        config: &DeepseekV41Config,
        maximum_bytes: u64,
    ) -> Result<(Self, u64, u64), DeepseekV41RuntimeError> {
        let text = &config.text_config;
        let names = [
            (
                format!("{prefix}.w1.weight"),
                text.moe_intermediate_size,
                text.hidden_size,
            ),
            (
                format!("{prefix}.w2.weight"),
                text.hidden_size,
                text.moe_intermediate_size,
            ),
            (
                format!("{prefix}.w3.weight"),
                text.moe_intermediate_size,
                text.hidden_size,
            ),
        ];
        let resident = names.iter().try_fold(0u64, |sum, (name, rows, columns)| {
            sum.checked_add(inspect_weight_matrix(index, name, *rows, *columns)?.resident_bytes)
                .ok_or_else(|| {
                    DeepseekV41RuntimeError::Invalid("expert resident bytes overflow".to_owned())
                })
        })?;
        let payload = names.iter().try_fold(0u64, |sum, (name, _, _)| {
            sum.checked_add(weight_storage_bytes(index, name)?)
                .ok_or_else(|| {
                    DeepseekV41RuntimeError::Invalid("expert payload bytes overflow".to_owned())
                })
        })?;
        if resident > maximum_bytes {
            return Err(DeepseekV41RuntimeError::Budget {
                component: "routed expert",
                required: resident,
                maximum: maximum_bytes,
            });
        }
        let specs = names
            .iter()
            .map(|(name, rows, columns)| (name.as_str(), *rows, *columns))
            .collect::<Vec<_>>();
        let mut matrices = load_weight_matrices(index, &specs, maximum_bytes)?.into_iter();
        Ok((
            Self {
                w1: matrices.next().expect("three expert matrices requested"),
                w2: matrices.next().expect("three expert matrices requested"),
                w3: matrices.next().expect("three expert matrices requested"),
                limit: text.swiglu_limit as f32,
            },
            resident,
            payload,
        ))
    }

    fn forward(
        &self,
        input: &[f32],
        route_weight: Option<f32>,
    ) -> Result<Vec<f32>, DeepseekV41RuntimeError> {
        let _profile = span(ProfileStage::DeepseekExpertCompute);
        let gate = linear(&self.w1, input)?;
        let up = linear(&self.w3, input)?;
        let mut activated = bounded_swiglu(&gate, &up, self.limit)?;
        if let Some(weight) = route_weight {
            for value in &mut activated {
                *value *= weight;
            }
        }
        round_to_bf16_in_place(&mut activated)?;
        Ok(linear(&self.w2, &activated)?)
    }
}

#[derive(Debug)]
struct ExpertStore {
    index: Arc<TensorIndex>,
    config: DeepseekV41Config,
    maximum_bytes: u64,
    cache: LayerLruCache<Arc<Expert>>,
    telemetry: ExpertTelemetry,
}

impl ExpertStore {
    fn new(
        index: Arc<TensorIndex>,
        config: &DeepseekV41Config,
        slots_per_layer: usize,
        maximum_bytes: u64,
    ) -> Result<Self, DeepseekV41RuntimeError> {
        let text = &config.text_config;
        if slots_per_layer > text.n_routed_experts || maximum_bytes == 0 {
            return Err(DeepseekV41RuntimeError::Invalid(
                "expert cache geometry is invalid".to_owned(),
            ));
        }
        Ok(Self {
            index,
            config: config.clone(),
            maximum_bytes,
            cache: LayerLruCache::new(text.num_hidden_layers, slots_per_layer),
            telemetry: ExpertTelemetry::default(),
        })
    }

    fn acquire(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<Arc<Expert>, DeepseekV41RuntimeError> {
        let text = &self.config.text_config;
        if layer >= text.num_hidden_layers || expert >= text.n_routed_experts {
            return Err(DeepseekV41RuntimeError::Invalid(
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
                let loaded = Expert::load(
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

    fn acquire_batch_while<R>(
        &mut self,
        layer: usize,
        experts: &[usize],
        while_loading: impl FnOnce() -> Result<R, DeepseekV41RuntimeError>,
    ) -> Result<(Vec<Arc<Expert>>, R), DeepseekV41RuntimeError> {
        if layer >= self.config.text_config.num_hidden_layers
            || experts
                .iter()
                .any(|&id| id >= self.config.text_config.n_routed_experts)
        {
            return Err(DeepseekV41RuntimeError::Invalid(
                "expert batch request is outside configured geometry".to_owned(),
            ));
        }
        if !self.cache.can_insert_without_eviction(layer, experts) {
            return Err(DeepseekV41RuntimeError::Invalid(
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
        let context = capture_context();
        let pending = missing
            .into_iter()
            .map(|expert| {
                let index = Arc::clone(&self.index);
                let config = self.config.clone();
                let context = context.clone();
                spawn_io(move || {
                    context.enter(|| {
                        let mut profile = span(ProfileStage::DeepseekExpertLoad);
                        let loaded = Expert::load(
                            &index,
                            &format!("layers.{layer}.ffn.experts.{expert}"),
                            &config,
                            maximum,
                        );
                        if let Ok((_, _, payload)) = &loaded {
                            profile.add_logical_bytes(*payload);
                        }
                        loaded.map(|(value, resident, payload)| {
                            (expert, (Arc::new(value), resident, payload))
                        })
                    })
                })
            })
            .collect::<Vec<_>>();
        let computed = while_loading();
        // Drain all tasks before propagating any error so a failed token leaves no outstanding
        // reads or expert allocations that can overlap its retry.
        let loaded = pending
            .into_iter()
            .map(|task| task.join())
            .collect::<Vec<_>>();
        let computed = computed?;
        let mut preloaded = HashMap::new();
        for loaded in loaded {
            let (expert, value) = loaded?;
            preloaded.insert(expert, value);
        }
        let loaded = experts
            .iter()
            .map(|&expert| {
                self.cache.access(
                    &mut self.telemetry,
                    layer,
                    expert,
                    || {
                        preloaded.remove(&expert).ok_or_else(|| {
                            DeepseekV41RuntimeError::Invalid(
                                "parallel expert preload omitted a cache miss".to_owned(),
                            )
                        })
                    },
                    |expert| Ok(Arc::clone(expert)),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((loaded, computed))
    }
}

fn matrix(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<WeightMatrix, DeepseekV41RuntimeError> {
    Ok(load_weight_matrix(index, name, rows, columns, u64::MAX)?)
}

fn load_reference_values_exact(
    index: &TensorIndex,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>, DeepseekV41RuntimeError> {
    let tensor = index.require(name)?;
    if tensor.declared_elements != elements as u64 {
        return Err(DeepseekV41RuntimeError::Invalid(format!(
            "tensor {name:?} must contain {elements} elements"
        )));
    }
    Ok(load_reference_values(index, name)?)
}

fn weight_storage_bytes(index: &TensorIndex, name: &str) -> Result<u64, DeepseekV41RuntimeError> {
    let mut bytes = index.require(name)?.data_len;
    if let Some(prefix) = name.strip_suffix(".weight") {
        if let Some(scale) = index.get(&format!("{prefix}.scale")) {
            bytes = bytes.checked_add(scale.data_len).ok_or_else(|| {
                DeepseekV41RuntimeError::Invalid("weight payload bytes overflow".to_owned())
            })?;
        }
    }
    Ok(bytes)
}

fn rope_parameters(ratio: usize, text: &super::DeepseekV41TextConfig) -> (f32, Option<usize>, f32) {
    if ratio == 0 {
        (text.rope_theta as f32, None, 1.0)
    } else {
        (
            text.compress_rope_theta as f32,
            Some(text.rope_scaling.original_max_position_embeddings),
            text.rope_scaling.factor as f32,
        )
    }
}

fn window_positions(position: usize, window: usize) -> Vec<usize> {
    let start = position.saturating_add(1).saturating_sub(window);
    (start..=position).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct ExpertFixture {
        directory: PathBuf,
        config: DeepseekV41Config,
        index: Arc<TensorIndex>,
    }

    impl ExpertFixture {
        fn new() -> Self {
            let mut config = DeepseekV41Config::from_json_str(include_str!(
                "../../../tests/fixtures/deepseek_v4_1_flash_config.json"
            ))
            .unwrap();
            // Exercise only ExpertStore; the full runtime intentionally rejects toy geometry.
            let text = &mut config.text_config;
            text.hidden_size = 2;
            text.moe_intermediate_size = 3;
            text.num_hidden_layers = 2;
            text.n_routed_experts = 4;
            text.num_experts_per_tok = 2;
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = std::env::temp_dir().join(format!(
                "urb_v41_expert_batch_{}_{}",
                std::process::id(),
                nonce
            ));
            fs::create_dir_all(&directory).unwrap();
            let mut header = serde_json::Map::new();
            let mut payload = Vec::new();
            for layer in 0..text.num_hidden_layers {
                for expert in 0..text.n_routed_experts {
                    for (matrix, rows, columns) in [(1, 3, 2), (2, 2, 3), (3, 3, 2)] {
                        let start = payload.len();
                        let marker = ((layer + 1) * 100 + expert * 10 + matrix) as f32;
                        for element in 0..rows * columns {
                            payload.extend((marker + element as f32 / 8.0).to_le_bytes());
                        }
                        header.insert(
                            format!("layers.{layer}.ffn.experts.{expert}.w{matrix}.weight"),
                            serde_json::json!({
                                "dtype": "F32", "shape": [rows, columns],
                                "data_offsets": [start, payload.len()]
                            }),
                        );
                    }
                }
            }
            let mut header = serde_json::to_vec(&header).unwrap();
            while header.len() % 8 != 0 {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend(payload);
            fs::write(directory.join("experts.safetensors"), bytes).unwrap();
            let index = Arc::new(TensorIndex::open(&directory).unwrap());
            Self {
                directory,
                config,
                index,
            }
        }

        fn store(&self, slots: usize) -> ExpertStore {
            ExpertStore::new(Arc::clone(&self.index), &self.config, slots, 72).unwrap()
        }
    }

    impl Drop for ExpertFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    fn assert_expert_ids(experts: &[Arc<Expert>], layer: usize, ids: &[usize]) {
        assert_eq!(experts.len(), ids.len());
        for (loaded, &id) in experts.iter().zip(ids) {
            for (matrix, weight, columns) in
                [(1, &loaded.w1, 2), (2, &loaded.w2, 3), (3, &loaded.w3, 2)]
            {
                let marker = ((layer + 1) * 100 + id * 10 + matrix) as f32;
                assert_eq!(
                    weight.row(0).unwrap(),
                    (0..columns)
                        .map(|column| marker + column as f32 / 8.0)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    fn serial_acquire(store: &mut ExpertStore, layer: usize, ids: &[usize]) {
        for &id in ids {
            store.acquire(layer, id).unwrap();
        }
    }

    #[test]
    fn overlapped_expert_batch_preserves_order_duplicates_and_serial_telemetry() {
        let fixture = ExpertFixture::new();
        let mut batch = fixture.store(3);
        let mut serial = fixture.store(3);
        let cached = batch.acquire(0, 1).unwrap();
        serial.acquire(0, 1).unwrap();

        let ids = [2, 1, 2, 0];
        let mut computations = 0;
        let (experts, computed) = batch
            .acquire_batch_while(0, &ids, || {
                computations += 1;
                Ok(vec![7.0f32, 9.0])
            })
            .unwrap();
        assert_eq!(computations, 1);
        assert_eq!(computed, [7.0, 9.0]);
        assert_expert_ids(&experts, 0, &ids);
        assert!(Arc::ptr_eq(&experts[0], &experts[2]));
        assert!(Arc::ptr_eq(&experts[1], &cached));
        serial_acquire(&mut serial, 0, &ids);
        assert_eq!(batch.telemetry, serial.telemetry);
        assert_eq!(batch.telemetry.hits, 2);
        assert_eq!(batch.telemetry.misses, 3);
        assert_eq!(batch.telemetry.bytes_read, 216);
        assert_eq!(batch.telemetry.resident_bytes, 216);

        for (layer, ids) in [(0, &[0, 2, 1, 0][..]), (1, &[1, 0, 1][..])] {
            let (experts, ()) = batch.acquire_batch_while(layer, ids, || Ok(())).unwrap();
            assert_expert_ids(&experts, layer, ids);
            serial_acquire(&mut serial, layer, ids);
            assert_eq!(batch.telemetry, serial.telemetry);
        }
    }

    #[test]
    fn overlapped_expert_batch_rejects_insufficient_slots_before_computing() {
        let fixture = ExpertFixture::new();
        for slots in [0, 1] {
            let mut batch = fixture.store(slots);
            batch.acquire(0, 1).unwrap();
            let before = batch.telemetry.clone();
            let error = batch
                .acquire_batch_while::<()>(0, &[2, 1, 2], || {
                    panic!("insufficient cache capacity must fail before the callback")
                })
                .unwrap_err();
            assert!(error.to_string().contains("free cache slots"));
            assert_eq!(batch.telemetry, before);
            assert_eq!(batch.cache.contains(0, 1), slots == 1);
            assert!(!batch.cache.contains(0, 2));
        }
    }

    #[test]
    fn moe_streaming_and_overlapped_paths_match_outputs_and_serial_cache_policy() {
        let fixture = ExpertFixture::new();
        let moe = MoeWeights {
            layer: 0,
            router: crate::model::DenseMatrix::new(
                4,
                2,
                vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            )
            .unwrap()
            .into(),
            correction_bias: vec![0.0; 4],
            shared: Expert::load(
                &fixture.index,
                "layers.0.ffn.experts.3",
                &fixture.config,
                72,
            )
            .unwrap()
            .0,
        };
        for slots in [0, 1] {
            let mut streaming = fixture.store(slots);
            let mut overlapped = fixture.store(4);
            let mut serial = fixture.store(slots);
            for input in [[1.0, 2.0], [1.0, 2.0], [-2.0, 1.0], [1.0, -2.0]] {
                let actual = moe
                    .forward(&input, &mut streaming, &fixture.config)
                    .unwrap();
                let expected = moe
                    .forward(&input, &mut overlapped, &fixture.config)
                    .unwrap();
                assert_eq!(actual, expected);
                let mut ids = actual
                    .1
                    .iter()
                    .map(|route| route.expert)
                    .collect::<Vec<_>>();
                ids.sort_unstable();
                serial_acquire(&mut serial, 0, &ids);
                assert_eq!(streaming.telemetry, serial.telemetry);
                assert_eq!(streaming.telemetry.resident_experts, slots);
                assert_eq!(streaming.telemetry.resident_bytes, slots as u64 * 72);
                for id in 0..4 {
                    assert_eq!(
                        streaming.cache.contains(0, id),
                        serial.cache.contains(0, id)
                    );
                    if let Some(expert) = streaming.cache.peek(0, id) {
                        assert_eq!(Arc::strong_count(expert), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn failed_shared_computation_leaves_expert_cache_unchanged_and_retry_succeeds() {
        let fixture = ExpertFixture::new();
        let mut batch = fixture.store(3);
        let mut serial = fixture.store(3);
        let cached = batch.acquire(0, 1).unwrap();
        serial.acquire(0, 1).unwrap();
        let before = batch.telemetry.clone();
        let ids = [1, 2, 0, 2];
        let error = batch
            .acquire_batch_while::<()>(0, &ids, || {
                Err(DeepseekV41RuntimeError::Invalid(
                    "shared compute failed".to_owned(),
                ))
            })
            .unwrap_err();
        assert!(error.to_string().contains("shared compute failed"));
        assert_eq!(batch.telemetry, before);
        assert!(Arc::ptr_eq(batch.cache.peek(0, 1).unwrap(), &cached));
        for id in [0, 2, 3] {
            assert!(!batch.cache.contains(0, id));
        }
        let (experts, ()) = batch.acquire_batch_while(0, &ids, || Ok(())).unwrap();
        assert_expert_ids(&experts, 0, &ids);
        serial_acquire(&mut serial, 0, &ids);
        assert_eq!(batch.telemetry, serial.telemetry);
        assert!(Arc::ptr_eq(&experts[0], &cached));
        assert!(Arc::ptr_eq(&experts[1], &experts[3]));
    }
}
