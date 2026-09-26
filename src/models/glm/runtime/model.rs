use super::{ExpertStore, ExpertTelemetry, RuntimeMoeError, RuntimeMoeLayer};
use crate::config::{ConfigError, GlmConfig, MLA_LATENT_NORM_EPS};
use crate::generation::CausalDecoder;
use crate::math::{rms_norm, MathError, RouteChoice};
use crate::model::{GatedMlp, MlpError, WeightError, WeightMatrix};
use crate::models::glm::{
    AttentionError, AttentionMode, MlaAttention, MlaCache, MlaGeometry, MoeGeometry,
};
use crate::profiling::{span, ProfileStage};
use crate::runtime::RuntimeLoadOptions;
use crate::storage::{
    inspect_weight_matrix, load_reference_vector, load_weight_matrix, DType, SafetensorError,
    TensorIndex, TensorLoadError, WeightLoadError,
};
use serde::Serialize;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_RUNTIME_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RuntimeRequirements {
    pub resident_bytes: u64,
    pub maximum_expert_bytes: u64,
    /// Minimum temporary working set needed even when persistent cache capacity is zero.
    pub transient_expert_bytes: u64,
    /// Global expert working-set bound, including the transient minimum above.
    pub expert_cache_bytes: u64,
    pub kv_cache_bytes: u64,
    pub expert_slots_per_layer: usize,
    pub context_limit: usize,
    pub exact_context_ceiling: usize,
}

#[derive(Debug)]
pub enum RuntimeError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Tensor(TensorLoadError),
    WeightLoad(WeightLoadError),
    Weight(WeightError),
    Attention(AttentionError),
    Mlp(MlpError),
    Moe(RuntimeMoeError),
    Math(MathError),
    Invalid(String),
    ResidentBudget {
        name: String,
        required: u64,
        remaining: u64,
    },
    ExpertCacheBudget {
        required: u64,
        maximum: u64,
    },
    KvCacheBudget {
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

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(f),
            Self::Checkpoint(error) => error.fmt(f),
            Self::Tensor(error) => error.fmt(f),
            Self::WeightLoad(error) => error.fmt(f),
            Self::Weight(error) => error.fmt(f),
            Self::Attention(error) => error.fmt(f),
            Self::Mlp(error) => error.fmt(f),
            Self::Moe(error) => error.fmt(f),
            Self::Math(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "invalid runtime model: {reason}"),
            Self::ResidentBudget {
                name,
                required,
                remaining,
            } => write!(
                f,
                "resident tensor {name:?} needs {required} bytes, only {remaining} budget bytes remain"
            ),
            Self::ExpertCacheBudget { required, maximum } => write!(
                f,
                "expert working set needs at most {required} bytes, budget is {maximum} bytes"
            ),
            Self::KvCacheBudget { required, maximum } => write!(
                f,
                "compressed KV cache needs {required} bytes, budget is {maximum} bytes"
            ),
            Self::TokenOutOfRange { token, vocabulary } => {
                write!(f, "token ID {token} is outside vocabulary 0..{vocabulary}")
            }
            Self::ContextExhausted { position, limit } => write!(
                f,
                "runtime position {position} reaches exact dense-MLA context limit {limit}"
            ),
        }
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Checkpoint(error) => Some(error),
            Self::Tensor(error) => Some(error),
            Self::WeightLoad(error) => Some(error),
            Self::Weight(error) => Some(error),
            Self::Attention(error) => Some(error),
            Self::Mlp(error) => Some(error),
            Self::Moe(error) => Some(error),
            Self::Math(error) => Some(error),
            _ => None,
        }
    }
}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for RuntimeError {
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
from_error!(AttentionError, Attention);
from_error!(MlpError, Mlp);
from_error!(RuntimeMoeError, Moe);
from_error!(MathError, Math);

#[derive(Debug, Clone)]
enum RuntimeFeedForward {
    Dense(GatedMlp),
    Sparse(RuntimeMoeLayer),
}

#[derive(Debug, Clone)]
struct RuntimeDecoderLayer {
    input_norm: Vec<f32>,
    attention: MlaAttention,
    post_attention_norm: Vec<f32>,
    feed_forward: RuntimeFeedForward,
    norm_eps: f32,
}

impl RuntimeDecoderLayer {
    fn forward(
        &self,
        hidden: &[f32],
        position: usize,
        cache: &mut MlaCache,
        experts: &mut ExpertStore,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), RuntimeError> {
        let _layer_profile = span(ProfileStage::GlmLayer);
        let normalized = rms_norm(hidden, &self.input_norm, self.norm_eps)?;
        let attention = {
            let _profile = span(ProfileStage::GlmAttention);
            self.attention
                .forward_token(&normalized, position, cache, AttentionMode::Absorbed)?
        };
        let after_attention = residual_add(hidden, &attention)?;
        let normalized = rms_norm(&after_attention, &self.post_attention_norm, self.norm_eps)?;
        let (feed_forward, routes) = {
            let _profile = span(ProfileStage::GlmFeedForward);
            match &self.feed_forward {
                RuntimeFeedForward::Dense(mlp) => (mlp.forward(&normalized)?, Vec::new()),
                RuntimeFeedForward::Sparse(moe) => moe.forward(&normalized, experts)?,
            }
        };
        Ok((residual_add(&after_attention, &feed_forward)?, routes))
    }
}

#[derive(Debug)]
pub struct RuntimeState {
    instance_id: u64,
    position: usize,
    layer_caches: Vec<MlaCache>,
    experts: ExpertStore,
}

impl RuntimeState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.experts.telemetry()
    }

    pub fn cached_f32_elements(&self) -> usize {
        self.layer_caches
            .iter()
            .map(MlaCache::stored_f32_elements)
            .sum()
    }
}

impl crate::runtime::session::SessionState for RuntimeState {
    type Checkpoint = usize;

    fn position(&self) -> usize {
        self.position
    }
    fn checkpoint(&self) -> usize {
        self.position
    }
    fn restore(&mut self, position: usize) -> Result<(), Box<dyn std::error::Error>> {
        for cache in &mut self.layer_caches {
            cache.truncate(position)?;
        }
        self.position = position;
        Ok(())
    }
    fn expert_telemetry(&self) -> &ExpertTelemetry {
        self.experts.telemetry()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeStep {
    pub logits: Vec<f32>,
    pub routes_by_layer: Vec<Vec<RouteChoice>>,
}

/// Exact scalar execution of the mixed-quantized base model with synchronous expert streaming.
#[derive(Debug)]
pub struct RuntimeModel {
    instance_id: u64,
    config: GlmConfig,
    embedding: WeightMatrix,
    layers: Vec<RuntimeDecoderLayer>,
    final_norm: Vec<f32>,
    lm_head: WeightMatrix,
    index: Arc<TensorIndex>,
    resident_bytes: u64,
    expert_slots_per_layer: usize,
    maximum_expert_bytes: u64,
    context_limit: usize,
}

impl RuntimeModel {
    /// Performs the complete runtime-specific checkpoint and memory inspection without reading
    /// tensor payloads.
    pub fn inspect_requirements(
        model_dir: impl AsRef<Path>,
        context_limit: usize,
        expert_slots_per_layer: usize,
    ) -> Result<RuntimeRequirements, RuntimeError> {
        let config = GlmConfig::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        inspect_runtime_requirements(&config, &index, context_limit, expert_slots_per_layer)
    }

    pub fn load(
        model_dir: impl AsRef<Path>,
        options: RuntimeLoadOptions,
    ) -> Result<Self, RuntimeError> {
        if options.resident_budget_bytes == 0
            || options.maximum_expert_bytes == 0
            || options.kv_cache_budget_bytes == 0
            || options.context_limit == 0
        {
            return Err(RuntimeError::Invalid(
                "resident, per-expert, and KV budgets plus context limit must be non-zero"
                    .to_owned(),
            ));
        }
        let config = GlmConfig::load(model_dir.as_ref())?;
        let index = TensorIndex::open(model_dir.as_ref())?;
        let requirements = inspect_runtime_requirements(
            &config,
            &index,
            options.context_limit,
            options.expert_slots_per_layer,
        )?;
        if requirements.resident_bytes > options.resident_budget_bytes {
            return Err(RuntimeError::ResidentBudget {
                name: "complete resident core".to_owned(),
                required: requirements.resident_bytes,
                remaining: options.resident_budget_bytes,
            });
        }
        if requirements.maximum_expert_bytes > options.maximum_expert_bytes {
            return Err(RuntimeError::Invalid(format!(
                "largest routed expert needs {} bytes, per-expert limit is {}",
                requirements.maximum_expert_bytes, options.maximum_expert_bytes
            )));
        }
        if requirements.expert_cache_bytes > options.expert_cache_budget_bytes {
            return Err(RuntimeError::ExpertCacheBudget {
                required: requirements.expert_cache_bytes,
                maximum: options.expert_cache_budget_bytes,
            });
        }
        if requirements.kv_cache_bytes > options.kv_cache_budget_bytes {
            return Err(RuntimeError::KvCacheBudget {
                required: requirements.kv_cache_bytes,
                maximum: options.kv_cache_budget_bytes,
            });
        }
        let instance_id = NEXT_RUNTIME_INSTANCE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| RuntimeError::Invalid("runtime instance ID exhausted".to_owned()))?;

        let mut loader = ResidentLoader::new(&index, options.resident_budget_bytes);
        let embedding = loader.matrix(
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(load_layer(&config, layer, &mut loader)?);
        }
        let final_norm = loader.vector("model.norm.weight", config.hidden_size)?;
        let lm_head = loader.matrix("lm_head.weight", config.vocab_size, config.hidden_size)?;
        let resident_bytes = loader.used;
        if resident_bytes != requirements.resident_bytes {
            return Err(RuntimeError::Invalid(format!(
                "preflight resident bytes {} disagree with loaded bytes {resident_bytes}",
                requirements.resident_bytes
            )));
        }

        Ok(Self {
            instance_id,
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            index: Arc::new(index),
            resident_bytes,
            expert_slots_per_layer: options.expert_slots_per_layer,
            maximum_expert_bytes: options.maximum_expert_bytes,
            context_limit: options.context_limit,
        })
    }

    pub fn config(&self) -> &GlmConfig {
        &self.config
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn context_limit(&self) -> usize {
        self.context_limit
    }

    pub fn new_state(&self) -> Result<RuntimeState, RuntimeError> {
        let layer_caches = self
            .layers
            .iter()
            .map(|layer| layer.attention.new_cache_with_capacity(self.context_limit))
            .collect::<Result<Vec<_>, _>>()?;
        let experts = ExpertStore::new_shared(
            Arc::clone(&self.index),
            self.config.num_hidden_layers,
            self.config.n_routed_experts,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            self.expert_slots_per_layer,
            self.maximum_expert_bytes,
        )
        .map_err(RuntimeMoeError::from)?;
        Ok(RuntimeState {
            instance_id: self.instance_id,
            position: 0,
            layer_caches,
            experts,
        })
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut RuntimeState,
    ) -> Result<RuntimeStep, RuntimeError> {
        let _profile = span(ProfileStage::GlmToken);
        self.validate_state(state)?;
        if token as usize >= self.config.vocab_size {
            return Err(RuntimeError::TokenOutOfRange {
                token,
                vocabulary: self.config.vocab_size,
            });
        }
        if state.position >= self.context_limit {
            return Err(RuntimeError::ContextExhausted {
                position: state.position,
                limit: self.context_limit,
            });
        }
        let start = state.position;
        match self.forward_token_inner(token, state) {
            Ok(step) => Ok(step),
            Err(error) => {
                let mut rollback_failure = None;
                for (layer, cache) in state.layer_caches.iter_mut().enumerate() {
                    if let Err(cause) = cache.truncate(start) {
                        rollback_failure = Some(format!("layer {layer}: {cause}"));
                    }
                }
                state.position = start;
                if let Some(cause) = rollback_failure {
                    Err(RuntimeError::Invalid(format!(
                        "state rollback failed after {error}: {cause}"
                    )))
                } else {
                    Err(error)
                }
            }
        }
    }

    fn forward_token_inner(
        &self,
        token: u32,
        state: &mut RuntimeState,
    ) -> Result<RuntimeStep, RuntimeError> {
        let mut hidden = {
            let _profile = span(ProfileStage::GlmEmbedding);
            self.embedding.row(token as usize)?
        };
        let mut routes_by_layer = Vec::with_capacity(self.layers.len());
        for (layer, cache) in self.layers.iter().zip(&mut state.layer_caches) {
            let (next, routes) =
                layer.forward(&hidden, state.position, cache, &mut state.experts)?;
            hidden = next;
            routes_by_layer.push(routes);
        }
        let normalized = {
            let _profile = span(ProfileStage::GlmFinalNorm);
            rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps as f32)?
        };
        let logits = {
            let _profile = span(ProfileStage::GlmLmHead);
            self.lm_head.matvec(&normalized)?
        };
        state.position += 1;
        Ok(RuntimeStep {
            logits,
            routes_by_layer,
        })
    }

    fn validate_state(&self, state: &RuntimeState) -> Result<(), RuntimeError> {
        if state.instance_id != self.instance_id {
            return Err(RuntimeError::Invalid(
                "runtime state belongs to a different model instance".to_owned(),
            ));
        }
        if state.layer_caches.len() != self.layers.len()
            || state
                .layer_caches
                .iter()
                .any(|cache| cache.len() != state.position)
        {
            return Err(RuntimeError::Invalid(
                "runtime cache lengths disagree with model/state position".to_owned(),
            ));
        }
        Ok(())
    }
}

impl CausalDecoder for RuntimeModel {
    type State = RuntimeState;
    type Error = RuntimeError;

    fn new_state(&self) -> Result<Self::State, Self::Error> {
        RuntimeModel::new_state(self)
    }

    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        Ok(RuntimeModel::forward_token(self, token, state)?.logits)
    }
}

struct ResidentLoader<'a> {
    index: &'a TensorIndex,
    maximum: u64,
    used: u64,
}

impl<'a> ResidentLoader<'a> {
    fn new(index: &'a TensorIndex, maximum: u64) -> Self {
        Self {
            index,
            maximum,
            used: 0,
        }
    }

    fn remaining(&self) -> u64 {
        self.maximum.saturating_sub(self.used)
    }

    fn matrix(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<WeightMatrix, RuntimeError> {
        let matrix = load_weight_matrix(self.index, name, rows, cols, self.remaining())?;
        let bytes = matrix.resident_bytes() as u64;
        self.charge(name, bytes)?;
        Ok(matrix)
    }

    fn vector(&mut self, name: &str, length: usize) -> Result<Vec<f32>, RuntimeError> {
        let bytes = (length as u64).checked_mul(4).ok_or_else(|| {
            RuntimeError::Invalid(format!("resident vector {name:?} byte count overflows"))
        })?;
        if bytes > self.remaining() {
            return Err(RuntimeError::ResidentBudget {
                name: name.to_owned(),
                required: bytes,
                remaining: self.remaining(),
            });
        }
        let vector = load_reference_vector(self.index, name, length)?;
        self.charge(name, bytes)?;
        Ok(vector)
    }

    fn charge(&mut self, name: &str, bytes: u64) -> Result<(), RuntimeError> {
        if bytes > self.remaining() {
            return Err(RuntimeError::ResidentBudget {
                name: name.to_owned(),
                required: bytes,
                remaining: self.remaining(),
            });
        }
        self.used += bytes;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct RuntimeManifest {
    resident_bytes: u64,
    maximum_expert_bytes: u64,
    /// Sum of the largest expert in every sparse layer: the worst-case bytes for one slot/layer.
    expert_slot_bytes_all_layers: u64,
}

struct ManifestBuilder<'a> {
    index: &'a TensorIndex,
    resident_bytes: u64,
    maximum_expert_bytes: u64,
    expert_slot_bytes_all_layers: u64,
}

impl<'a> ManifestBuilder<'a> {
    fn new(index: &'a TensorIndex) -> Self {
        Self {
            index,
            resident_bytes: 0,
            maximum_expert_bytes: 0,
            expert_slot_bytes_all_layers: 0,
        }
    }

    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<(), RuntimeError> {
        let bytes = inspect_weight_matrix(self.index, name, rows, cols)?.resident_bytes;
        self.resident_bytes = checked_add_bytes(self.resident_bytes, bytes, "resident core")?;
        Ok(())
    }

    fn vector(&mut self, name: &str, length: usize) -> Result<(), RuntimeError> {
        let tensor = self.index.require(name)?;
        if tensor.shape != [length as u64] {
            return Err(RuntimeError::Invalid(format!(
                "resident vector {name:?} must declare [{length}], got {:?}",
                tensor.shape
            )));
        }
        if !matches!(tensor.dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Err(RuntimeError::Invalid(format!(
                "resident vector {name:?} has unsupported dtype {}",
                tensor.dtype
            )));
        }
        if self.index.get(&format!("{name}.qs")).is_some() {
            return Err(RuntimeError::Invalid(format!(
                "resident vector {name:?} unexpectedly has a .qs sidecar"
            )));
        }
        let bytes = (length as u64)
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::Invalid("resident vector bytes overflow".to_owned()))?;
        self.resident_bytes = checked_add_bytes(self.resident_bytes, bytes, "resident core")?;
        Ok(())
    }

    fn expert(
        &self,
        layer: usize,
        expert: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<u64, RuntimeError> {
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let mut bytes = 0u64;
        for (projection, rows, cols) in [
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ] {
            let layout = inspect_weight_matrix(
                self.index,
                &format!("{prefix}.{projection}.weight"),
                rows,
                cols,
            )?;
            bytes = checked_add_bytes(bytes, layout.resident_bytes, "routed expert")?;
        }
        Ok(bytes)
    }

    fn finish(self) -> RuntimeManifest {
        RuntimeManifest {
            resident_bytes: self.resident_bytes,
            maximum_expert_bytes: self.maximum_expert_bytes,
            expert_slot_bytes_all_layers: self.expert_slot_bytes_all_layers,
        }
    }
}

fn inspect_runtime_requirements(
    config: &GlmConfig,
    index: &TensorIndex,
    context_limit: usize,
    expert_slots_per_layer: usize,
) -> Result<RuntimeRequirements, RuntimeError> {
    if context_limit == 0 {
        return Err(RuntimeError::Invalid(
            "context limit must be non-zero".to_owned(),
        ));
    }
    if expert_slots_per_layer > config.n_routed_experts {
        return Err(RuntimeError::Invalid(format!(
            "expert_slots_per_layer={expert_slots_per_layer} exceeds n_routed_experts={}",
            config.n_routed_experts
        )));
    }
    if index.names().any(|name| {
        name.contains(".self_attn.indexer.") || name.contains(".self_attn.indexers_proj")
    }) {
        return Err(RuntimeError::Invalid(
            "DSA indexer execution is not implemented; refusing a checkpoint that contains indexer weights"
                .to_owned(),
        ));
    }
    let exact_context_ceiling = if config.index_topk > 0 {
        config.index_topk.min(config.max_position_embeddings)
    } else {
        config.max_position_embeddings
    };
    if context_limit > exact_context_ceiling {
        return Err(RuntimeError::Invalid(format!(
            "requested context {context_limit} exceeds exact dense-MLA fallback limit {exact_context_ceiling}"
        )));
    }
    let manifest = preflight_runtime(config, index)?;
    let persistent_expert_cache_bytes = manifest
        .expert_slot_bytes_all_layers
        .checked_mul(expert_slots_per_layer as u64)
        .ok_or_else(|| RuntimeError::Invalid("expert cache byte count overflows".to_owned()))?;
    let transient_expert_bytes = manifest.maximum_expert_bytes;
    let expert_cache_bytes = persistent_expert_cache_bytes.max(transient_expert_bytes);
    Ok(RuntimeRequirements {
        resident_bytes: manifest.resident_bytes,
        maximum_expert_bytes: manifest.maximum_expert_bytes,
        transient_expert_bytes,
        expert_cache_bytes,
        kv_cache_bytes: kv_cache_bytes(config, context_limit)?,
        expert_slots_per_layer,
        context_limit,
        exact_context_ceiling,
    })
}

fn preflight_runtime(
    config: &GlmConfig,
    index: &TensorIndex,
) -> Result<RuntimeManifest, RuntimeError> {
    let mut manifest = ManifestBuilder::new(index);
    manifest.matrix(
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
    )?;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        manifest.vector(
            &format!("{prefix}.input_layernorm.weight"),
            config.hidden_size,
        )?;
        manifest.vector(
            &format!("{prefix}.post_attention_layernorm.weight"),
            config.hidden_size,
        )?;
        let attention = format!("{prefix}.self_attn");
        manifest.matrix(
            &format!("{attention}.q_a_proj.weight"),
            config.q_lora_rank_value(),
            config.hidden_size,
        )?;
        manifest.vector(
            &format!("{attention}.q_a_layernorm.weight"),
            config.q_lora_rank_value(),
        )?;
        manifest.matrix(
            &format!("{attention}.q_b_proj.weight"),
            config.num_attention_heads * config.qk_head_dim,
            config.q_lora_rank_value(),
        )?;
        manifest.matrix(
            &format!("{attention}.kv_a_proj_with_mqa.weight"),
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
        )?;
        manifest.vector(
            &format!("{attention}.kv_a_layernorm.weight"),
            config.kv_lora_rank,
        )?;
        manifest.matrix(
            &format!("{attention}.kv_b_proj.weight"),
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        )?;
        manifest.matrix(
            &format!("{attention}.o_proj.weight"),
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
        )?;

        if config.layer_is_sparse(layer) {
            manifest.matrix(
                &format!("{prefix}.mlp.gate.weight"),
                config.n_routed_experts,
                config.hidden_size,
            )?;
            manifest.vector(
                &format!("{prefix}.mlp.gate.e_score_correction_bias"),
                config.n_routed_experts,
            )?;
            let shared = config
                .moe_intermediate_size
                .checked_mul(config.n_shared_experts)
                .ok_or_else(|| RuntimeError::Invalid("shared expert size overflows".to_owned()))?;
            preflight_mlp(
                &mut manifest,
                &format!("{prefix}.mlp.shared_experts"),
                config.hidden_size,
                shared,
            )?;
            let mut layer_maximum = 0u64;
            for expert in 0..config.n_routed_experts {
                let bytes = manifest.expert(
                    layer,
                    expert,
                    config.hidden_size,
                    config.moe_intermediate_size,
                )?;
                layer_maximum = layer_maximum.max(bytes);
                manifest.maximum_expert_bytes = manifest.maximum_expert_bytes.max(bytes);
            }
            manifest.expert_slot_bytes_all_layers = checked_add_bytes(
                manifest.expert_slot_bytes_all_layers,
                layer_maximum,
                "expert cache slot",
            )?;
        } else {
            preflight_mlp(
                &mut manifest,
                &format!("{prefix}.mlp"),
                config.hidden_size,
                config.intermediate_size,
            )?;
        }
    }
    manifest.vector("model.norm.weight", config.hidden_size)?;
    manifest.matrix("lm_head.weight", config.vocab_size, config.hidden_size)?;
    Ok(manifest.finish())
}

fn preflight_mlp(
    manifest: &mut ManifestBuilder<'_>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<(), RuntimeError> {
    manifest.matrix(&format!("{prefix}.gate_proj.weight"), intermediate, hidden)?;
    manifest.matrix(&format!("{prefix}.up_proj.weight"), intermediate, hidden)?;
    manifest.matrix(&format!("{prefix}.down_proj.weight"), hidden, intermediate)?;
    Ok(())
}

fn kv_cache_bytes(config: &GlmConfig, context: usize) -> Result<u64, RuntimeError> {
    let values = config
        .kv_lora_rank
        .checked_add(config.qk_rope_head_dim)
        .and_then(|per_layer| per_layer.checked_mul(config.num_hidden_layers))
        .and_then(|per_token| per_token.checked_mul(context))
        .ok_or_else(|| RuntimeError::Invalid("compressed KV size overflows usize".to_owned()))?;
    (values as u64)
        .checked_mul(4)
        .ok_or_else(|| RuntimeError::Invalid("compressed KV bytes overflow u64".to_owned()))
}

fn checked_add_bytes(left: u64, right: u64, label: &str) -> Result<u64, RuntimeError> {
    left.checked_add(right)
        .ok_or_else(|| RuntimeError::Invalid(format!("{label} byte count overflows")))
}

fn load_layer(
    config: &GlmConfig,
    layer: usize,
    loader: &mut ResidentLoader<'_>,
) -> Result<RuntimeDecoderLayer, RuntimeError> {
    let prefix = format!("model.layers.{layer}");
    let input_norm = loader.vector(
        &format!("{prefix}.input_layernorm.weight"),
        config.hidden_size,
    )?;
    let post_attention_norm = loader.vector(
        &format!("{prefix}.post_attention_layernorm.weight"),
        config.hidden_size,
    )?;
    let attention_prefix = format!("{prefix}.self_attn");
    let geometry = MlaGeometry {
        hidden_size: config.hidden_size,
        num_heads: config.num_attention_heads,
        q_lora_rank: config.q_lora_rank_value(),
        kv_lora_rank: config.kv_lora_rank,
        qk_nope_head_dim: config.qk_nope_head_dim,
        qk_rope_head_dim: config.qk_rope_head_dim,
        v_head_dim: config.v_head_dim,
        latent_norm_eps: MLA_LATENT_NORM_EPS,
        rope_theta: config.rope_parameters.rope_theta as f32,
    };
    let attention = MlaAttention::new_mixed(
        geometry,
        loader.matrix(
            &format!("{attention_prefix}.q_a_proj.weight"),
            config.q_lora_rank_value(),
            config.hidden_size,
        )?,
        loader.vector(
            &format!("{attention_prefix}.q_a_layernorm.weight"),
            config.q_lora_rank_value(),
        )?,
        loader.matrix(
            &format!("{attention_prefix}.q_b_proj.weight"),
            config.num_attention_heads * config.qk_head_dim,
            config.q_lora_rank_value(),
        )?,
        loader.matrix(
            &format!("{attention_prefix}.kv_a_proj_with_mqa.weight"),
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
        )?,
        loader.vector(
            &format!("{attention_prefix}.kv_a_layernorm.weight"),
            config.kv_lora_rank,
        )?,
        loader.matrix(
            &format!("{attention_prefix}.kv_b_proj.weight"),
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        )?,
        loader.matrix(
            &format!("{attention_prefix}.o_proj.weight"),
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
        )?,
    )?;

    let feed_forward = if config.layer_is_sparse(layer) {
        let router = loader.matrix(
            &format!("{prefix}.mlp.gate.weight"),
            config.n_routed_experts,
            config.hidden_size,
        )?;
        let correction_bias = loader.vector(
            &format!("{prefix}.mlp.gate.e_score_correction_bias"),
            config.n_routed_experts,
        )?;
        let shared_intermediate = config
            .moe_intermediate_size
            .checked_mul(config.n_shared_experts)
            .ok_or_else(|| RuntimeError::Invalid("shared expert size overflows".to_owned()))?;
        let shared = load_mlp(
            loader,
            &format!("{prefix}.mlp.shared_experts"),
            config.hidden_size,
            shared_intermediate,
        )?;
        RuntimeFeedForward::Sparse(RuntimeMoeLayer::new(
            layer,
            router,
            correction_bias,
            shared,
            MoeGeometry {
                top_k: config.num_experts_per_tok,
                n_group: config.n_group,
                topk_group: config.topk_group,
                normalize: config.norm_topk_prob,
                routed_scaling_factor: config.routed_scaling_factor as f32,
            },
        )?)
    } else {
        RuntimeFeedForward::Dense(load_mlp(
            loader,
            &format!("{prefix}.mlp"),
            config.hidden_size,
            config.intermediate_size,
        )?)
    };
    Ok(RuntimeDecoderLayer {
        input_norm,
        attention,
        post_attention_norm,
        feed_forward,
        norm_eps: config.rms_norm_eps as f32,
    })
}

fn load_mlp(
    loader: &mut ResidentLoader<'_>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<GatedMlp, RuntimeError> {
    Ok(GatedMlp::new_mixed(
        loader.matrix(&format!("{prefix}.gate_proj.weight"), intermediate, hidden)?,
        loader.matrix(&format!("{prefix}.up_proj.weight"), intermediate, hidden)?,
        loader.matrix(&format!("{prefix}.down_proj.weight"), hidden, intermediate)?,
    )?)
}

fn residual_add(left: &[f32], right: &[f32]) -> Result<Vec<f32>, RuntimeError> {
    if left.len() != right.len() {
        return Err(RuntimeError::Invalid(format!(
            "residual lengths {} and {} differ",
            left.len(),
            right.len()
        )));
    }
    let result: Vec<f32> = left.iter().zip(right).map(|(&a, &b)| a + b).collect();
    if result.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::Invalid(
            "residual produced NaN or infinity".to_owned(),
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{generate, GenerationConfig, StopReason};
    use serde::Deserialize;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn fixture_dir(label: &str) -> PathBuf {
        crate::test_support::temp_dir(&format!("urbilateria_runtime_{label}"))
    }

    fn add_f32_tensor(
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
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }

    fn add_weight_tensor(
        tensors: &mut BTreeMap<String, serde_json::Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
        values: Vec<f32>,
        mixed_quantized: bool,
    ) {
        if !mixed_quantized {
            add_f32_tensor(tensors, payload, name, shape, values);
            return;
        }
        assert_eq!(shape.len(), 2);
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let rows = shape[0];
        let cols = shape[1];
        let int8 = matches!(name, "model.embed_tokens.weight" | "lm_head.weight");
        let (packed, scales) = if int8 {
            let mut packed = Vec::with_capacity(rows * cols);
            let mut scales = Vec::with_capacity(rows);
            for row in values.chunks_exact(cols) {
                let maximum = row
                    .iter()
                    .fold(0.0f32, |current, value| current.max(value.abs()));
                let scale = (maximum / 127.0).max(1e-8);
                scales.push(scale);
                packed.extend(
                    row.iter()
                        .map(|value| (value / scale).round().clamp(-128.0, 127.0) as i8 as u8),
                );
            }
            (packed, scales)
        } else {
            let matrix = crate::math::Int4Matrix::quantize_per_row(rows, cols, &values).unwrap();
            (matrix.packed().to_vec(), matrix.scales().to_vec())
        };
        let start = payload.len();
        payload.extend(packed);
        tensors.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": "U8",
                "shape": [payload.len() - start],
                "data_offsets": [start, payload.len()],
            }),
        );
        add_f32_tensor(
            tensors,
            payload,
            &format!("{name}.qs"),
            &[scales.len()],
            scales,
        );
    }

    fn write_tiny_checkpoint(
        path: &Path,
        sparse: bool,
        mixed_quantized: bool,
        corrupt_experts: bool,
    ) {
        fs::create_dir_all(path).unwrap();
        let config = serde_json::json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "attention_bias": false,
            "mlp_bias": false,
            "rope_interleave": true,
            "indexer_rope_interleave": true,
            "tie_word_embeddings": false,
            "vocab_size": 8,
            "intermediate_size": 4,
            "moe_intermediate_size": 2,
            "first_k_dense_replace": if sparse { 0 } else { 1 },
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "q_lora_rank": 2,
            "kv_lora_rank": 2,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "qk_head_dim": 4,
            "v_head_dim": 2,
            "mlp_layer_types": [if sparse { "sparse" } else { "dense" }],
            "rms_norm_eps": 0.00001,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "max_position_embeddings": 8,
            "eos_token_id": [6],
            "pad_token_id": 6,
            "hidden_act": "silu",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc"
        });
        fs::write(
            path.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();

        let mut tensors = BTreeMap::new();
        let mut payload = Vec::new();
        let values = |count: usize, scale: f32| {
            (0..count)
                .map(|index| ((index % 7) as f32 - 3.0) * scale)
                .collect::<Vec<_>>()
        };
        add_weight_tensor(
            &mut tensors,
            &mut payload,
            "model.embed_tokens.weight",
            &[8, 4],
            values(32, 0.03),
            mixed_quantized,
        );
        for name in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
        ] {
            add_f32_tensor(&mut tensors, &mut payload, name, &[4], vec![1.0; 4]);
        }
        add_weight_tensor(
            &mut tensors,
            &mut payload,
            "model.layers.0.self_attn.q_a_proj.weight",
            &[2, 4],
            values(8, 0.02),
            mixed_quantized,
        );
        add_f32_tensor(
            &mut tensors,
            &mut payload,
            "model.layers.0.self_attn.q_a_layernorm.weight",
            &[2],
            vec![1.0; 2],
        );
        add_weight_tensor(
            &mut tensors,
            &mut payload,
            "model.layers.0.self_attn.q_b_proj.weight",
            &[4, 2],
            values(8, 0.02),
            mixed_quantized,
        );
        add_weight_tensor(
            &mut tensors,
            &mut payload,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            &[4, 4],
            values(16, 0.015),
            mixed_quantized,
        );
        add_f32_tensor(
            &mut tensors,
            &mut payload,
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            &[2],
            vec![1.0; 2],
        );
        add_weight_tensor(
            &mut tensors,
            &mut payload,
            "model.layers.0.self_attn.kv_b_proj.weight",
            &[4, 2],
            values(8, 0.02),
            mixed_quantized,
        );
        add_weight_tensor(
            &mut tensors,
            &mut payload,
            "model.layers.0.self_attn.o_proj.weight",
            &[4, 2],
            values(8, 0.02),
            mixed_quantized,
        );

        if sparse {
            add_f32_tensor(
                &mut tensors,
                &mut payload,
                "model.layers.0.mlp.gate.weight",
                &[2, 4],
                vec![0.0; 8],
            );
            add_f32_tensor(
                &mut tensors,
                &mut payload,
                "model.layers.0.mlp.gate.e_score_correction_bias",
                &[2],
                vec![0.0; 2],
            );
            for projection in ["gate_proj", "up_proj"] {
                add_weight_tensor(
                    &mut tensors,
                    &mut payload,
                    &format!("model.layers.0.mlp.shared_experts.{projection}.weight"),
                    &[2, 4],
                    values(8, 0.01),
                    mixed_quantized,
                );
            }
            add_weight_tensor(
                &mut tensors,
                &mut payload,
                "model.layers.0.mlp.shared_experts.down_proj.weight",
                &[4, 2],
                values(8, 0.01),
                mixed_quantized,
            );
            // Header-valid but non-finite lazy expert payloads force an execution-time failure,
            // after attention has appended KV, so transactional rollback is exercised.
            for expert in 0..2 {
                for (projection, shape) in [
                    ("gate_proj", [2, 4]),
                    ("up_proj", [2, 4]),
                    ("down_proj", [4, 2]),
                ] {
                    let mut expert_values = values(8, 0.01 * (expert + 1) as f32);
                    if corrupt_experts {
                        expert_values[0] = f32::NAN;
                        add_f32_tensor(
                            &mut tensors,
                            &mut payload,
                            &format!("model.layers.0.mlp.experts.{expert}.{projection}.weight"),
                            &shape,
                            expert_values,
                        );
                    } else {
                        add_weight_tensor(
                            &mut tensors,
                            &mut payload,
                            &format!("model.layers.0.mlp.experts.{expert}.{projection}.weight"),
                            &shape,
                            expert_values,
                            mixed_quantized,
                        );
                    }
                }
            }
        } else {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                add_weight_tensor(
                    &mut tensors,
                    &mut payload,
                    &format!("model.layers.0.mlp.{projection}.weight"),
                    &[4, 4],
                    values(16, 0.01),
                    mixed_quantized,
                );
            }
        }
        // The F32 fixture keeps zero logits for an exact tie-breaking assertion. The mixed
        // fixture uses non-zero INT8 rows so its integration test exercises numerical data flow.
        add_weight_tensor(
            &mut tensors,
            &mut payload,
            "lm_head.weight",
            &[8, 4],
            if mixed_quantized {
                values(32, 0.025)
            } else {
                vec![0.0; 32]
            },
            mixed_quantized,
        );

        let mut header = serde_json::to_vec(&tensors).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(payload);
        fs::write(path.join("model.safetensors"), output).unwrap();
    }

    fn options(context_limit: usize) -> RuntimeLoadOptions {
        RuntimeLoadOptions {
            resident_budget_bytes: 1 << 20,
            expert_cache_budget_bytes: 1 << 20,
            kv_cache_budget_bytes: 1 << 20,
            expert_slots_per_layer: 1,
            maximum_expert_bytes: 1 << 20,
            context_limit,
        }
    }

    #[derive(Debug, Deserialize)]
    struct OfficialOracle {
        transformers: String,
        torch: String,
        tokens: Vec<u32>,
        full_logits: Vec<Vec<f32>>,
        incremental_logits: Vec<Vec<f32>>,
        argmax: Vec<u32>,
        indexer_topk_covers_all: bool,
        max_layer_parity_error: f64,
        max_logits_parity_error: f64,
    }

    fn oracle_values(count: usize, seed: i32) -> Vec<f32> {
        (0..count)
            .map(|index| ((index as i32 * 17 + seed * 13) % 29 - 14) as f32 / 100.0)
            .collect()
    }

    fn oracle_norm(length: usize, seed: i32) -> Vec<f32> {
        oracle_values(length, seed)
            .into_iter()
            .map(|value| 1.0 + value / 10.0)
            .collect()
    }

    /// Materializes the deterministic weights used by the offline oracle generators.
    fn write_oracle_checkpoint(path: &Path) {
        fs::create_dir_all(path).unwrap();
        let config = serde_json::json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": 6,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "attention_bias": false,
            "mlp_bias": false,
            "rope_interleave": true,
            "indexer_rope_interleave": true,
            "tie_word_embeddings": false,
            "vocab_size": 7,
            "intermediate_size": 5,
            "moe_intermediate_size": 2,
            "first_k_dense_replace": 1,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "q_lora_rank": 5,
            "kv_lora_rank": 3,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 4,
            "qk_head_dim": 6,
            "v_head_dim": 3,
            "mlp_layer_types": ["dense"],
            "rms_norm_eps": 0.00001,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "max_position_embeddings": 8,
            "eos_token_id": [6],
            "pad_token_id": 6,
            "hidden_act": "silu",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc"
        });
        fs::write(
            path.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();

        let mut tensors = BTreeMap::new();
        let mut payload = Vec::new();
        for (name, shape, values) in [
            (
                "model.embed_tokens.weight",
                &[7, 6][..],
                oracle_values(42, 1),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                &[6][..],
                oracle_norm(6, 2),
            ),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                &[5, 6][..],
                oracle_values(30, 3),
            ),
            (
                "model.layers.0.self_attn.q_a_layernorm.weight",
                &[5][..],
                oracle_norm(5, 4),
            ),
            (
                "model.layers.0.self_attn.q_b_proj.weight",
                &[12, 5][..],
                oracle_values(60, 5),
            ),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                &[7, 6][..],
                oracle_values(42, 6),
            ),
            (
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                &[3][..],
                oracle_norm(3, 7),
            ),
            (
                "model.layers.0.self_attn.kv_b_proj.weight",
                &[10, 3][..],
                oracle_values(30, 8),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                &[6, 6][..],
                oracle_values(36, 9),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                &[6][..],
                oracle_norm(6, 10),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                &[5, 6][..],
                oracle_values(30, 11),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                &[5, 6][..],
                oracle_values(30, 12),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                &[6, 5][..],
                oracle_values(30, 13),
            ),
            ("model.norm.weight", &[6][..], oracle_norm(6, 14)),
            ("lm_head.weight", &[7, 6][..], oracle_values(42, 15)),
        ] {
            add_f32_tensor(&mut tensors, &mut payload, name, shape, values);
        }

        let mut header = serde_json::to_vec(&tensors).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(payload);
        fs::write(path.join("model.safetensors"), output).unwrap();
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right) {
            assert!((a - b).abs() < tolerance, "{a} versus {b}");
        }
    }

    #[test]
    fn session_checkpoint_replays_a_different_suffix_with_identical_logits() {
        use crate::runtime::session::SessionState;
        let dir = fixture_dir("session_rewind");
        write_oracle_checkpoint(&dir);
        let model = RuntimeModel::load(&dir, options(8)).unwrap();
        let mut state = model.new_state().unwrap();
        model.forward_token(1, &mut state).unwrap();
        model.forward_token(2, &mut state).unwrap();
        let saved = state.checkpoint();
        for token in [3, 4, 1, 2] {
            model.forward_token(token, &mut state).unwrap();
        }
        state.restore(saved).unwrap();
        let resumed = model.forward_token(0, &mut state).unwrap().logits;
        let mut fresh = model.new_state().unwrap();
        let expected =
            crate::generation::CausalDecoder::prefill(&model, &[1, 2, 0], &mut fresh).unwrap();
        assert_eq!(resumed, expected);
        assert_eq!(state.cached_f32_elements(), fresh.cached_f32_elements());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bounded_runtime_matches_independent_transformers_golden() {
        // Regenerate this frozen output with tools/generate_transformers_oracle.py. That
        // process uses Transformers/PyTorch and does not import this crate.
        let official: OfficialOracle = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/glm_moe_dsa_transformers_5_12_tiny.json"
        ))
        .unwrap();
        assert_eq!(official.transformers, "5.12.0");
        assert_eq!(official.torch, "2.9.0+cu126");
        assert!(official.indexer_topk_covers_all);
        assert!(official.max_layer_parity_error < 3e-8);
        assert!(official.max_logits_parity_error < 3e-8);

        let dir = fixture_dir("official_oracle");
        write_oracle_checkpoint(&dir);
        let model = RuntimeModel::load(&dir, options(official.tokens.len())).unwrap();
        let mut state = model.new_state().unwrap();
        for (position, &token) in official.tokens.iter().enumerate() {
            let step = model.forward_token(token, &mut state).unwrap();
            assert_close(&step.logits, &official.full_logits[position], 3e-6);
            assert_close(&step.logits, &official.incremental_logits[position], 3e-6);
            let argmax = step
                .logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(token, _)| token as u32)
                .unwrap();
            assert_eq!(argmax, official.argmax[position]);
        }
        assert_eq!(state.position(), official.tokens.len());
        assert_eq!(state.cached_f32_elements(), official.tokens.len() * 7);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn complete_tiny_checkpoint_loads_and_generates() {
        let dir = fixture_dir("dense");
        write_tiny_checkpoint(&dir, false, false, false);
        let requirements = RuntimeModel::inspect_requirements(&dir, 4, 1).unwrap();
        assert!(requirements.resident_bytes > 0);
        assert_eq!(requirements.kv_cache_bytes, 64);
        assert_eq!(requirements.maximum_expert_bytes, 0);
        let model = RuntimeModel::load(&dir, options(4)).unwrap();
        assert!(model.resident_bytes() > 0);
        assert_eq!(model.context_limit(), 4);

        let output = generate(
            &model,
            &[1, 2],
            &GenerationConfig::greedy(2, vec![6]),
            |_| {},
        )
        .unwrap();
        assert_eq!(output.generated_tokens, vec![0, 0]);
        assert_eq!(output.stop_reason, StopReason::MaxNewTokens);

        let mut state = model.new_state().unwrap();
        model.forward_token(1, &mut state).unwrap();
        assert_eq!(state.position(), 1);
        assert_eq!(state.cached_f32_elements(), 4);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn complete_tiny_mixed_checkpoint_executes_int8_and_int4() {
        let dir = fixture_dir("mixed");
        write_tiny_checkpoint(&dir, false, true, false);
        let index = TensorIndex::open(&dir).unwrap();
        assert_eq!(
            inspect_weight_matrix(&index, "model.embed_tokens.weight", 8, 4)
                .unwrap()
                .format,
            crate::storage::WeightFormat::Int8PerRow
        );
        assert_eq!(
            inspect_weight_matrix(&index, "model.layers.0.self_attn.q_a_proj.weight", 2, 4,)
                .unwrap()
                .format,
            crate::storage::WeightFormat::Int4 { group_size: 4 }
        );
        let model = RuntimeModel::load(&dir, options(3)).unwrap();
        let mut state = model.new_state().unwrap();
        let step = model.forward_token(1, &mut state).unwrap();
        assert!(step.logits.iter().any(|value| value.abs() > 1e-8));
        let output = generate(
            &model,
            &[1],
            &GenerationConfig::greedy(2, Vec::new()),
            |_| {},
        )
        .unwrap();
        assert_eq!(output.generated_tokens.len(), 2);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn complete_sparse_mixed_checkpoint_streams_and_reuses_expert() {
        let dir = fixture_dir("sparse_mixed");
        write_tiny_checkpoint(&dir, true, true, false);
        let zero_slot_requirements = RuntimeModel::inspect_requirements(&dir, 3, 0).unwrap();
        assert_eq!(zero_slot_requirements.transient_expert_bytes, 44);
        assert_eq!(zero_slot_requirements.expert_cache_bytes, 44);
        let requirements = RuntimeModel::inspect_requirements(&dir, 3, 1).unwrap();
        assert_eq!(requirements.maximum_expert_bytes, 44);
        assert_eq!(requirements.transient_expert_bytes, 44);
        assert_eq!(requirements.expert_cache_bytes, 44);
        let model = RuntimeModel::load(&dir, options(3)).unwrap();
        let mut state = model.new_state().unwrap();

        let first = model.forward_token(1, &mut state).unwrap();
        assert_eq!(first.routes_by_layer[0][0].expert, 0);
        assert!(first.logits.iter().any(|value| value.abs() > 1e-8));
        assert_eq!(state.expert_telemetry().misses, 1);
        assert_eq!(state.expert_telemetry().hits, 0);
        model.forward_token(2, &mut state).unwrap();
        assert_eq!(state.expert_telemetry().misses, 1);
        assert_eq!(state.expert_telemetry().hits, 1);
        assert_eq!(state.expert_telemetry().resident_experts, 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn lazy_expert_failure_rolls_back_every_layer_cache() {
        let dir = fixture_dir("rollback");
        write_tiny_checkpoint(&dir, true, false, true);
        let requirements = RuntimeModel::inspect_requirements(&dir, 4, 1).unwrap();
        assert_eq!(requirements.maximum_expert_bytes, 96);
        assert_eq!(requirements.expert_cache_bytes, 96);
        let model = RuntimeModel::load(&dir, options(4)).unwrap();
        let mut state = model.new_state().unwrap();
        assert!(model.forward_token(1, &mut state).is_err());
        assert_eq!(state.position(), 0);
        assert_eq!(state.cached_f32_elements(), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn context_limit_is_enforced_before_state_mutation() {
        let dir = fixture_dir("context");
        write_tiny_checkpoint(&dir, false, false, false);
        let model = RuntimeModel::load(&dir, options(1)).unwrap();
        let mut state = model.new_state().unwrap();
        model.forward_token(1, &mut state).unwrap();
        assert!(matches!(
            model.forward_token(2, &mut state),
            Err(RuntimeError::ContextExhausted {
                position: 1,
                limit: 1
            })
        ));
        assert_eq!(state.position(), 1);
        assert_eq!(state.cached_f32_elements(), 4);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_token_is_rejected_before_state_mutation() {
        let dir = fixture_dir("invalid_token");
        write_tiny_checkpoint(&dir, false, false, false);
        let model = RuntimeModel::load(&dir, options(2)).unwrap();
        let mut state = model.new_state().unwrap();
        model.forward_token(1, &mut state).unwrap();
        assert!(matches!(
            model.forward_token(99, &mut state),
            Err(RuntimeError::TokenOutOfRange {
                token: 99,
                vocabulary: 8
            })
        ));
        assert_eq!(state.position(), 1);
        assert_eq!(state.cached_f32_elements(), 4);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn state_cannot_be_mixed_between_runtime_instances() {
        let dir = fixture_dir("provenance");
        write_tiny_checkpoint(&dir, false, false, false);
        let first = RuntimeModel::load(&dir, options(2)).unwrap();
        let second = RuntimeModel::load(&dir, options(2)).unwrap();
        let mut state = first.new_state().unwrap();
        assert!(matches!(
            second.forward_token(1, &mut state),
            Err(RuntimeError::Invalid(reason))
                if reason.contains("different model instance")
        ));
        assert_eq!(state.position(), 0);
        assert_eq!(state.cached_f32_elements(), 0);
        fs::remove_dir_all(dir).unwrap();
    }
}
