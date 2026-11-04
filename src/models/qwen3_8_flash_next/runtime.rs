//! Bounded BF16 text execution. PLE rows and selected packed experts are read on demand.
use super::components::{HyperConnection, Ple, PleState, Qsa, QsaState, Weights};
use super::{schema, Qwen38FlashNextConfig};
use crate::generation::CausalDecoder;
use crate::model::{Bf16Matrix, WeightMatrix};
use crate::models::qwen3_8::{
    expert::Qwen38Expert,
    linear_attention::{DeltaNetGeometry, DeltaNetState, GatedDeltaNet},
    math::{linear, round_to_bf16 as bf},
    moe::route_softmax_top_k,
};
use crate::runtime::telemetry::{self, Event as RuntimeEvent, Weights as WeightResidency};
use crate::runtime::{
    cache::LayerLruCache, session::SessionState, ExpertTelemetry, RuntimeLoadOptions,
};
use crate::storage::{load_compact_bf16_matrix, load_reference_matrix_row, TensorIndex};
use serde::Serialize;
use std::{
    fmt,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

#[derive(Debug)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Qwen3.8-Flash-Next: {}", self.0)
    }
}
impl std::error::Error for Error {}
pub(super) type Result<T> = std::result::Result<T, Error>;
macro_rules! convert { ($($ty:ty),* $(,)?) => { $(impl From<$ty> for Error { fn from(e:$ty)->Self { Self(e.to_string()) } })* }; }
convert!(
    crate::config::ConfigError,
    schema::SchemaError,
    crate::storage::SafetensorError,
    crate::storage::TensorLoadError,
    crate::storage::WeightLoadError,
    crate::model::WeightError,
    crate::model::MatrixError,
    crate::math::MathError,
    crate::models::qwen3_8::norm::NormError,
    crate::models::qwen3_8::linear_attention::DeltaNetError,
    crate::models::qwen3_8::expert::Qwen38ExpertError,
    crate::models::qwen3_8::moe::Qwen38MoeError
);

const ROOT: &str = "model.language_model.";
const SCRATCH: u64 = 512 * 1024 * 1024;
const BATCH: usize = 32;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize)]
pub struct Qwen38FlashNextRuntimeRequirements {
    pub schema: schema::Qwen38FlashNextRequirements,
    pub context_limit: usize,
    pub layer_resident_bytes: Vec<u64>,
    pub streamed_layer_bytes: u64,
    pub lm_head_resident_bytes: u64,
    pub recurrent_state_bytes: u64,
    pub convolution_state_bytes: u64,
    pub kv_cache_bytes: u64,
    pub indexer_cache_bytes: u64,
    pub ple_state_bytes: u64,
    pub scratch_bytes: u64,
    pub expert_bytes: u64,
    pub expert_cache_bytes: u64,
    pub peak_resident_bytes: u64,
    pub expert_slots_per_layer: usize,
}
impl Qwen38FlashNextRuntimeRequirements {
    pub fn state_bytes(&self) -> u64 {
        self.recurrent_state_bytes
            + self.convolution_state_bytes
            + self.kv_cache_bytes
            + self.indexer_cache_bytes
            + self.ple_state_bytes
    }
    pub fn checkpoint_bytes(&self) -> u64 {
        self.recurrent_state_bytes
            + self.convolution_state_bytes
            + self.ple_state_bytes
            + 12 * 3 * 128 * 4
    }
    pub fn plan_load_options(
        &self,
        total_ram: u64,
        session_checkpoint: bool,
    ) -> Result<RuntimeLoadOptions> {
        let minimum = self.peak_resident_bytes - self.expert_cache_bytes;
        let fixed = minimum
            + self.state_bytes()
            + if session_checkpoint {
                self.checkpoint_bytes()
            } else {
                0
            };
        if total_ram < fixed {
            return Err(Error(format!(
                "runtime and causal state need {fixed} bytes, RAM budget is {total_ram}"
            )));
        }
        let extra = total_ram - fixed;
        let backbone = (self.lm_head_resident_bytes
            + self.layer_resident_bytes.iter().sum::<u64>())
        .min(extra);
        let per_slot = self.expert_bytes * 48;
        let slots = ((extra - backbone) / per_slot).min(512) as usize;
        Ok(RuntimeLoadOptions {
            resident_budget_bytes: minimum + backbone + per_slot * slots as u64,
            expert_cache_budget_bytes: per_slot * slots as u64,
            kv_cache_budget_bytes: self.state_bytes(),
            expert_slots_per_layer: slots,
            maximum_expert_bytes: self.expert_bytes,
            context_limit: self.context_limit,
        })
    }
    pub fn backbone_residency(&self, budget: u64) -> (usize, bool) {
        let mut extra = budget.saturating_sub(self.peak_resident_bytes - self.expert_cache_bytes);
        let head = extra >= self.lm_head_resident_bytes;
        if head {
            extra -= self.lm_head_resident_bytes;
        }
        let mut layers = 0;
        for &bytes in &self.layer_resident_bytes {
            if extra < bytes {
                break;
            }
            extra -= bytes;
            layers += 1;
        }
        (layers, head)
    }
}

pub struct Qwen38FlashNextRuntimeModel {
    id: u64,
    config: Qwen38FlashNextConfig,
    index: Arc<TensorIndex>,
    requirements: Qwen38FlashNextRuntimeRequirements,
    layers: Vec<Layer>,
    mixer: HyperConnection,
    head: Option<WeightMatrix>,
    slots: usize,
    residency: WeightResidency,
}
enum Attention {
    Linear(Box<GatedDeltaNet>),
    Indexed(Box<Qsa>),
}
struct Layer {
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    ple: Option<Ple>,
    attention: Attention,
    router: WeightMatrix,
    shared: Qwen38Expert,
    shared_gate: WeightMatrix,
}
enum AttentionState {
    Linear(DeltaNetState),
    Indexed(QsaState),
}

pub struct Qwen38FlashNextRuntimeState {
    model_id: u64,
    id: u64,
    position: usize,
    poisoned: bool,
    attention: Vec<AttentionState>,
    ple: PleState,
    experts: LayerLruCache<Qwen38Expert>,
    telemetry: ExpertTelemetry,
    residency: WeightResidency,
}
pub struct SessionCheckpoint {
    model_id: u64,
    state_id: u64,
    position: usize,
    recurrent: Vec<(usize, DeltaNetState)>,
    indexed: Vec<(usize, usize, Vec<Vec<f32>>)>,
    ple: PleState,
}
impl SessionState for Qwen38FlashNextRuntimeState {
    type Checkpoint = SessionCheckpoint;
    fn position(&self) -> usize {
        self.position
    }
    fn checkpoint(&self) -> SessionCheckpoint {
        let mut recurrent = Vec::new();
        let mut indexed = Vec::new();
        for (i, a) in self.attention.iter().enumerate() {
            match a {
                AttentionState::Linear(s) => recurrent.push((i, s.clone())),
                AttentionState::Indexed(s) => indexed.push((i, s.blocks.len(), s.tail.clone())),
            }
        }
        SessionCheckpoint {
            model_id: self.model_id,
            state_id: self.id,
            position: self.position,
            recurrent,
            indexed,
            ple: self.ple.clone(),
        }
    }
    fn restore(
        &mut self,
        c: SessionCheckpoint,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        if c.model_id != self.model_id || c.state_id != self.id || c.position > self.position {
            return Err(Box::new(Error(
                "checkpoint belongs to another sequence or future position".into(),
            )));
        }
        for (i, s) in c.recurrent {
            self.attention[i] = AttentionState::Linear(s);
        }
        for (i, blocks, tail) in c.indexed {
            if let AttentionState::Indexed(s) = &mut self.attention[i] {
                s.truncate(c.position, blocks, tail);
            }
        }
        self.ple = c.ple;
        self.position = c.position;
        self.poisoned = false;
        Ok(())
    }
    fn expert_telemetry(&self) -> &ExpertTelemetry {
        &self.telemetry
    }
    fn expert_layers(&self) -> Vec<telemetry::LayerCache> {
        self.experts.residency()
    }
    fn weight_residency(&self) -> Option<WeightResidency> {
        Some(self.residency.clone())
    }
}

fn geometry(c: &Qwen38FlashNextConfig) -> DeltaNetGeometry {
    DeltaNetGeometry {
        hidden_size: c.hidden_size,
        num_key_heads: c.linear_num_key_heads,
        num_value_heads: c.linear_num_value_heads,
        key_head_dim: c.linear_key_head_dim,
        value_head_dim: c.linear_value_head_dim,
        conv_kernel_size: c.linear_conv_kernel_dim,
        norm_eps: c.rms_norm_eps as f32,
    }
}
fn resident_bytes(index: &TensorIndex, prefix: &str) -> u64 {
    index
        .tensors()
        .filter(|t| {
            t.name.starts_with(prefix)
                && !t.name.contains("mlp.experts.")
                && !t.name.contains("ple_embedding.")
        })
        .map(|t| {
            if t.shape.len() == 2 {
                t.data_len
            } else {
                t.declared_elements * 4
            }
        })
        .sum()
}
impl Qwen38FlashNextRuntimeModel {
    pub fn inspect_requirements(
        directory: &Path,
        context: usize,
        slots: usize,
    ) -> Result<Qwen38FlashNextRuntimeRequirements> {
        let c = Qwen38FlashNextConfig::load(directory)?;
        if context == 0 || context > c.max_position_embeddings || slots > c.num_experts {
            return Err(Error("context/expert slots outside release limits".into()));
        }
        let index = TensorIndex::open(directory)?;
        let manifest = schema::inspect_manifest(&c, directory)?;
        schema::inspect_requirements(&c, &schema::HfWeightIndex::load(directory)?, &index)?;
        let layer_resident_bytes = (0..c.num_hidden_layers)
            .map(|l| resident_bytes(&index, &format!("{ROOT}layers.{l}.")))
            .collect::<Vec<_>>();
        let streamed_layer_bytes = *layer_resident_bytes.iter().max().unwrap();
        let expert_bytes = (c.hidden_size * c.moe_intermediate_size * 6) as u64;
        let expert_cache_bytes = expert_bytes * slots as u64 * c.num_hidden_layers as u64;
        let root_bytes = resident_bytes(&index, &format!("{ROOT}hyper_connection_mixer."));
        let kv_cache_bytes = context as u64 * 12 * (2 * 2 * 256 * 4 + 2 * 24); // KV values plus row headers
        let indexer_cache_bytes =
            (context.div_ceil(4) as u64 * 12 * (128 * 4 + 24)) * 2 + 12 * 4 * 128 * 4;
        Ok(Qwen38FlashNextRuntimeRequirements {
            schema: manifest,
            context_limit: context,
            layer_resident_bytes,
            streamed_layer_bytes,
            lm_head_resident_bytes: index.require("lm_head.weight")?.data_len,
            recurrent_state_bytes: 36 * 48 * 128 * 128 * 4,
            convolution_state_bytes: 36 * 10240 * 4 * 4,
            kv_cache_bytes: kv_cache_bytes * 2, // Vec growth reserve, bounded by capacity checks
            indexer_cache_bytes,
            ple_state_bytes: 10 * 10240 * 4 + 3 * 4,
            scratch_bytes: SCRATCH,
            expert_bytes,
            expert_cache_bytes,
            expert_slots_per_layer: slots,
            peak_resident_bytes: root_bytes
                + streamed_layer_bytes
                + SCRATCH
                + expert_bytes
                + expert_cache_bytes,
        })
    }
    pub fn load(directory: &Path, options: RuntimeLoadOptions) -> Result<Self> {
        let requirements = Self::inspect_requirements(
            directory,
            options.context_limit,
            options.expert_slots_per_layer,
        )?;
        if options.resident_budget_bytes < requirements.peak_resident_bytes
            || options.kv_cache_budget_bytes < requirements.state_bytes()
            || options.expert_cache_budget_bytes < requirements.expert_cache_bytes
            || options.maximum_expert_bytes < requirements.expert_bytes
        {
            return Err(Error(
                "load budget is below validated resident/state/expert requirements".into(),
            ));
        }
        let config = Qwen38FlashNextConfig::load(directory)?;
        let index = Arc::new(TensorIndex::open(directory)?);
        let (count, keep_head) = requirements
            .backbone_residency(options.resident_budget_bytes - requirements.expert_cache_bytes);
        let mut residency = WeightResidency {
            total_layers: config.num_hidden_layers,
            loaded_layers: 0,
            planned_layers: count,
            resident_bytes: 0,
            lm_head_loaded: false,
            lm_head_planned: keep_head,
            loading: true,
            expert_budget_bytes: requirements.expert_cache_bytes,
            experts_per_layer: config.num_experts,
            row_streamed_embeddings: true,
        };
        telemetry::emit(true, || RuntimeEvent::Weights(residency.clone()));
        let mut w = Weights::load(&index, &format!("{ROOT}hyper_connection_mixer."))?;
        let mixer = HyperConnection::take(&mut w, "", &config, false)?;
        w.finish()?;
        residency.resident_bytes =
            resident_bytes(&index, &format!("{ROOT}hyper_connection_mixer."));
        telemetry::emit(true, || RuntimeEvent::Weights(residency.clone()));
        let head = if keep_head {
            Some(load_compact_bf16_matrix(
                &index,
                "lm_head.weight",
                config.vocab_size,
                config.hidden_size,
                requirements.lm_head_resident_bytes,
            )?)
        } else {
            None
        };
        if head.is_some() {
            residency.lm_head_loaded = true;
            residency.resident_bytes += requirements.lm_head_resident_bytes;
            telemetry::emit(true, || RuntimeEvent::Weights(residency.clone()));
        }
        let mut layers = Vec::with_capacity(count);
        for layer in 0..count {
            layers.push(Layer::load(&index, &config, layer)?);
            residency.loaded_layers = layers.len();
            residency.resident_bytes += requirements.layer_resident_bytes[layer];
            telemetry::emit(true, || RuntimeEvent::Weights(residency.clone()));
        }
        residency.loading = false;
        telemetry::emit(true, || RuntimeEvent::Weights(residency.clone()));
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            config,
            index,
            requirements,
            layers,
            mixer,
            head,
            slots: options.expert_slots_per_layer,
            residency,
        })
    }
    pub fn requirements(&self) -> &Qwen38FlashNextRuntimeRequirements {
        &self.requirements
    }
    /// Diagnostic forward retaining post-layer HC activations for independent numerical checks.
    pub fn forward_token_traced(
        &self,
        token: u32,
        state: &mut Qwen38FlashNextRuntimeState,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let mut layers = Vec::new();
        let logits = self.forward_batch_inner(&[token], state, Some(&mut layers))?;
        Ok((logits, layers))
    }
    fn forward_batch(
        &self,
        tokens: &[u32],
        state: &mut Qwen38FlashNextRuntimeState,
    ) -> Result<Vec<f32>> {
        self.forward_batch_inner(tokens, state, None)
    }
    fn forward_batch_inner(
        &self,
        tokens: &[u32],
        state: &mut Qwen38FlashNextRuntimeState,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>> {
        if state.model_id != self.id || state.poisoned {
            return Err(Error(
                "foreign or poisoned runtime state; create a new state or restore a checkpoint"
                    .into(),
            ));
        }
        if tokens.is_empty()
            || tokens.len() > BATCH
            || tokens.iter().any(|&t| t as usize >= self.config.vocab_size)
            || state
                .position
                .checked_add(tokens.len())
                .is_none_or(|p| p > self.requirements.context_limit)
        {
            return Err(Error("invalid tokens or exhausted context".into()));
        }
        state.poisoned = true;
        let mut hidden = tokens
            .iter()
            .map(|&t| {
                Ok(load_reference_matrix_row(
                    &self.index,
                    &format!("{ROOT}embed_tokens.weight"),
                    t as usize,
                    self.config.vocab_size,
                    self.config.hidden_size,
                )?
                .repeat(self.config.hc_count))
            })
            .collect::<Result<Vec<_>>>()?;
        for layer in 0..self.config.num_hidden_layers {
            let streamed;
            let weights = if let Some(w) = self.layers.get(layer) {
                w
            } else {
                streamed = Layer::load(&self.index, &self.config, layer)?;
                &streamed
            };
            for (i, (h, &token)) in hidden.iter_mut().zip(tokens).enumerate() {
                if let Some(ple) = &weights.ple {
                    let output = ple.forward(&self.index, h, token, &mut state.ple)?;
                    for (x, y) in h.iter_mut().zip(output) {
                        *x = bf(*x + y)?;
                    }
                }
                let (input, injection) = weights.attn_hc.mix(h)?;
                let output = match (&weights.attention, &mut state.attention[layer]) {
                    (Attention::Linear(m), AttentionState::Linear(s)) => {
                        m.forward_token(&input, s)?
                    }
                    (Attention::Indexed(m), AttentionState::Indexed(s)) => {
                        m.forward(&input, state.position + i, s)?
                    }
                    _ => return Err(Error("attention/state variant mismatch".into())),
                };
                weights.attn_hc.combine(h, &output, &injection)?;
                let (input, injection) = weights.mlp_hc.mix(h)?;
                let mut routes = route_softmax_top_k(
                    &linear(&weights.router, &input)?,
                    self.config.num_experts_per_tok,
                )?;
                routes.sort_by_key(|r| r.expert);
                let mut output = vec![0.0; self.config.hidden_size];
                for route in routes {
                    let values = state.experts.access(
                        &mut state.telemetry,
                        layer,
                        route.expert,
                        || {
                            let e = load_expert(&self.index, &self.config, layer, route.expert)?;
                            Ok::<_, Error>((
                                e,
                                self.requirements.expert_bytes,
                                self.requirements.expert_bytes,
                            ))
                        },
                        |e| Ok::<_, Error>(e.forward(&input)?),
                    )?;
                    let weight = bf(route.weight)?;
                    for (o, v) in output.iter_mut().zip(values) {
                        *o = bf(*o + bf(weight * v)?)?;
                    }
                }
                let shared = weights.shared.forward(&input)?;
                let gate = linear(&weights.shared_gate, &input)?[0];
                let gate = bf(1.0 / (1.0 + (-gate).exp()))?;
                for (o, v) in output.iter_mut().zip(shared) {
                    *o = bf(*o + bf(gate * v)?)?;
                }
                weights.mlp_hc.combine(h, &output, &injection)?;
            }
            if let Some(trace) = trace.as_mut() {
                trace.push(hidden.last().unwrap().clone());
            }
            crate::runtime::session::observe_experts(state, false);
        }
        let (last, _) = self.mixer.mix(hidden.last().unwrap())?;
        let logits = if let Some(head) = &self.head {
            linear(head, &last)?
        } else {
            // Use the same FP32-accumulating BF16 projection as the resident head.
            // The generic streamed reference uses F64 and can change BF16 rounding.
            let mut logits = Vec::with_capacity(self.config.vocab_size);
            let width = self.config.hidden_size;
            for start in (0..self.config.vocab_size).step_by(4096) {
                let count = 4096.min(self.config.vocab_size - start);
                let bytes = self.index.read_range(
                    "lm_head.weight",
                    (start * width * 2) as u64,
                    count * width * 2,
                )?;
                let matrix = WeightMatrix::Bf16(Bf16Matrix::from_le_bytes(count, width, bytes)?);
                logits.extend(linear(&matrix, &last)?);
            }
            logits
        };
        state.position += tokens.len();
        state.poisoned = false;
        telemetry::sequence(state.position, self.requirements.context_limit, false);
        Ok(logits)
    }
}
impl CausalDecoder for Qwen38FlashNextRuntimeModel {
    type State = Qwen38FlashNextRuntimeState;
    type Error = Error;
    fn new_state(&self) -> Result<Self::State> {
        let attention = (0..self.config.num_hidden_layers)
            .map(|l| {
                Ok(if self.config.is_full_attention_layer(l) == Some(true) {
                    AttentionState::Indexed(QsaState::new())
                } else {
                    AttentionState::Linear(DeltaNetState::new(geometry(&self.config))?)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Qwen38FlashNextRuntimeState {
            model_id: self.id,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            position: 0,
            poisoned: false,
            attention,
            ple: PleState::new(),
            experts: LayerLruCache::new(self.config.num_hidden_layers, self.slots),
            telemetry: ExpertTelemetry::default(),
            residency: self.residency.clone(),
        })
    }
    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>> {
        self.forward_batch(&[token], state)
    }
    fn prefill(&self, tokens: &[u32], state: &mut Self::State) -> Result<Vec<f32>> {
        if tokens.is_empty()
            || state
                .position
                .checked_add(tokens.len())
                .is_none_or(|p| p > self.requirements.context_limit)
            || tokens.iter().any(|&t| t as usize >= self.config.vocab_size)
        {
            return Err(Error(
                "invalid prefill token sequence or context limit".into(),
            ));
        }
        let mut result = Vec::new();
        for chunk in tokens.chunks(BATCH) {
            result = self.forward_batch(chunk, state)?;
        }
        Ok(result)
    }
}
impl Layer {
    fn load(index: &TensorIndex, c: &Qwen38FlashNextConfig, layer: usize) -> Result<Self> {
        let mut w = Weights::load(index, &format!("{ROOT}layers.{layer}."))?;
        let attn_hc = HyperConnection::take(&mut w, "attn_hyper_connection.", c, true)?;
        let mlp_hc = HyperConnection::take(&mut w, "mlp_hyper_connection.", c, true)?;
        let ple = if c.ple_layer_ids.contains(&(layer + 1)) {
            Some(Ple::take(&mut w, index, layer, c)?)
        } else {
            None
        };
        let attention = if c.is_full_attention_layer(layer) == Some(true) {
            Attention::Indexed(Box::new(Qsa::take(&mut w, c)?))
        } else {
            Attention::Linear(Box::new(
                GatedDeltaNet::new_mixed(
                    geometry(c),
                    w.matrix("linear_attn.in_proj_qkv.weight")?,
                    w.matrix("linear_attn.in_proj_z.weight")?,
                    w.matrix("linear_attn.in_proj_b.weight")?,
                    w.matrix("linear_attn.in_proj_a.weight")?,
                    w.vector("linear_attn.conv1d.weight")?,
                    w.vector("linear_attn.dt_bias")?,
                    w.vector("linear_attn.A_log")?,
                    w.vector("linear_attn.norm.weight")?,
                    w.matrix("linear_attn.out_proj.weight")?,
                )?
                .with_flash_next_semantics(),
            ))
        };
        let router = w.matrix("mlp.gate.weight")?;
        let shared_gate = w.matrix("mlp.shared_expert_gate.weight")?;
        let shared = Qwen38Expert::new(
            w.matrix("mlp.shared_expert.gate_proj.weight")?,
            w.matrix("mlp.shared_expert.up_proj.weight")?,
            w.matrix("mlp.shared_expert.down_proj.weight")?,
        )?;
        w.finish()?;
        Ok(Self {
            attn_hc,
            mlp_hc,
            ple,
            attention,
            router,
            shared,
            shared_gate,
        })
    }
}
fn load_expert(
    index: &TensorIndex,
    c: &Qwen38FlashNextConfig,
    layer: usize,
    expert: usize,
) -> Result<Qwen38Expert> {
    if layer >= c.num_hidden_layers || expert >= c.num_experts {
        return Err(Error("expert index out of bounds".into()));
    }
    let bytes = c.hidden_size * c.moe_intermediate_size * 2;
    let prefix = format!("{ROOT}layers.{layer}.mlp.experts");
    let matrix = |name: &str, offset: usize, rows, cols| -> Result<WeightMatrix> {
        Ok(WeightMatrix::Bf16(Bf16Matrix::from_le_bytes(
            rows,
            cols,
            index.read_range(&format!("{prefix}.{name}"), offset as u64, bytes)?,
        )?))
    };
    Ok(Qwen38Expert::new(
        matrix(
            "gate_up_proj",
            expert * 2 * bytes,
            c.moe_intermediate_size,
            c.hidden_size,
        )?,
        matrix(
            "gate_up_proj",
            (expert * 2 + 1) * bytes,
            c.moe_intermediate_size,
            c.hidden_size,
        )?,
        matrix(
            "down_proj",
            expert * bytes,
            c.hidden_size,
            c.moe_intermediate_size,
        )?,
    )?)
}
