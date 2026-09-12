//! Header-only validation and memory planning for the native Hy4-preview-FP8 checkpoint.

use super::Hy4Config;
use crate::config::ConfigError;
use crate::storage::{DType, SafetensorError, TensorIndex, TensorInfo};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::PathBuf;

pub const RELEASE_TENSOR_COUNT: usize = 2_838;
pub const RELEASE_BASE_TENSOR_COUNT: usize = 2_799;
pub const RELEASE_MTP_TENSOR_COUNT: usize = 39;
pub const RELEASE_SHARD_COUNT: usize = 130;
pub const MODEL_OPT_MX_BLOCK: usize = 32;
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug)]
pub enum SchemaError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    Invalid(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Read { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::Json { path, source } => {
                write!(formatter, "invalid JSON in {}: {source}", path.display())
            }
            Self::Invalid(reason) => write!(formatter, "invalid Hy4 checkpoint: {reason}"),
        }
    }
}

impl std::error::Error for SchemaError {}

impl From<ConfigError> for SchemaError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl From<SafetensorError> for SchemaError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl TensorSpec {
    fn new(name: impl Into<String>, dtype: DType, shape: &[u64]) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn elements(&self) -> Result<u64, SchemaError> {
        self.shape.iter().try_fold(1u64, |product, &dimension| {
            product.checked_mul(dimension).ok_or_else(|| {
                SchemaError::Invalid(format!("tensor {:?} element count overflows", self.name))
            })
        })
    }

    fn payload_bytes(&self) -> Result<u64, SchemaError> {
        self.elements()?
            .checked_mul(self.dtype.element_bytes().ok_or_else(|| {
                SchemaError::Invalid(format!("tensor {:?} has unknown dtype", self.name))
            })?)
            .ok_or_else(|| {
                SchemaError::Invalid(format!("tensor {:?} byte count overflows", self.name))
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Hy4Requirements {
    pub required_tensor_count: usize,
    pub checkpoint_tensor_count: usize,
    pub checkpoint_shard_count: usize,
    pub checkpoint_file_bytes: u64,
    pub checkpoint_payload_bytes: u64,
    pub logical_parameter_count: u64,
    pub fp8_matrix_count: usize,
    pub scale_tensor_count: usize,
    pub base_tensor_count: usize,
    pub mtp_tensor_count: usize,
    pub dense_layer_count: usize,
    pub moe_layer_count: usize,
    pub full_indexer_layer_count: usize,
    pub routed_experts_per_layer: usize,
    pub selected_experts_per_token: usize,
    pub context_limit: usize,
    pub exact_context_ceiling: usize,
    pub streamed_layer_bytes: u64,
    pub decoder_layer_bytes: u64,
    pub lm_head_bytes: u64,
    pub resident_bytes: u64,
    pub kv_cache_bytes: u64,
    pub maximum_expert_bytes: u64,
    pub transient_expert_bytes: u64,
    pub expert_cache_bytes: u64,
    pub expert_slots_per_layer: usize,
    pub mtp_payload_bytes: u64,
}

impl Hy4Requirements {
    /// Number of complete decoder layers that can be pinned above the conservative two-layer
    /// streaming allowance represented by [`Self::resident_bytes`].
    pub fn cached_decoder_layers_for_resident_budget(&self, resident_budget: u64) -> usize {
        let layer_count = self.dense_layer_count.saturating_add(self.moe_layer_count);
        let cache_budget = resident_budget.saturating_sub(self.resident_bytes);
        if cache_budget >= self.decoder_layer_bytes {
            layer_count
        } else {
            cache_budget
                .checked_div(self.streamed_layer_bytes)
                .unwrap_or(0)
                .min(layer_count as u64) as usize
        }
    }

    pub fn caches_lm_head_for_resident_budget(&self, resident_budget: u64) -> bool {
        resident_budget
            >= self
                .resident_bytes
                .saturating_add(self.decoder_layer_bytes)
                .saturating_add(self.lm_head_bytes)
    }
}

#[derive(Debug, Deserialize)]
struct HfIndex {
    weight_map: BTreeMap<String, String>,
}

pub fn inspect_requirements(
    config: &Hy4Config,
    index: &TensorIndex,
    context_limit: usize,
    expert_slots_per_layer: usize,
) -> Result<Hy4Requirements, SchemaError> {
    config.validate_release()?;
    if context_limit == 0 || context_limit > config.exact_dense_context_ceiling() {
        return invalid(format!(
            "runtime context {context_limit} is outside 1..={}; the checkpoint supports 1M positions, but exact CPU execution currently uses dense Gated-MLA until IndexCache execution lands",
            config.exact_dense_context_ceiling()
        ));
    }
    if expert_slots_per_layer > config.n_routed_experts {
        return invalid(format!(
            "expert slots/layer {expert_slots_per_layer} exceeds {} experts",
            config.n_routed_experts
        ));
    }
    let specs = official_tensor_specs(config)?;
    validate_specs(index, &specs)?;
    validate_hf_assignment(index)?;

    let logical_parameter_count = specs.iter().try_fold(0u64, |total, spec| {
        if spec.dtype == DType::U8 {
            Ok(total)
        } else {
            total
                .checked_add(spec.elements()?)
                .ok_or_else(|| SchemaError::Invalid("logical parameter count overflows".to_owned()))
        }
    })?;
    let maximum_expert_bytes = expert_resident_bytes(config)?;
    let sparse_layers = config.sparse_layer_count() as u64;
    let expert_cache_bytes = maximum_expert_bytes
        .checked_mul(expert_slots_per_layer as u64)
        .and_then(|bytes| bytes.checked_mul(sparse_layers))
        .ok_or_else(|| SchemaError::Invalid("expert cache bytes overflow".to_owned()))?;
    let kv_values = (config.kv_lora_rank + config.qk_rope_head_dim) as u64;
    let kv_cache_bytes = (config.num_hidden_layers as u64)
        .checked_mul(context_limit as u64)
        .and_then(|values| values.checked_mul(kv_values))
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| SchemaError::Invalid("KV cache bytes overflow".to_owned()))?;
    let decoder_layer_sizes = (0..config.num_hidden_layers)
        .map(|layer| layer_resident_bytes(index, layer))
        .collect::<Result<Vec<_>, _>>()?;
    let streamed_layer_bytes = decoder_layer_sizes.iter().copied().max().unwrap_or(0);
    let decoder_layer_bytes = decoder_layer_sizes.iter().try_fold(0u64, |total, &bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| SchemaError::Invalid("decoder layer byte count overflows".to_owned()))
    })?;
    let lm_head_bytes = index.require("lm_head.weight")?.data_len;
    let root_resident_bytes = root_resident_bytes(config)?;
    // Compute of layer N overlaps loading of layer N+1, so two complete decoder layers can be
    // resident at the same time. The release geometry always contains multiple layers.
    let resident_bytes = streamed_layer_bytes
        .checked_mul(2)
        .ok_or_else(|| SchemaError::Invalid("pipelined layer resident bytes overflow".to_owned()))?
        .checked_add(root_resident_bytes)
        .ok_or_else(|| SchemaError::Invalid("resident byte count overflows".to_owned()))?;
    let mtp_payload_bytes = index
        .tensors()
        .filter(|tensor| tensor.name.starts_with("model.mtp_layers."))
        .try_fold(0u64, |total, tensor| {
            total
                .checked_add(tensor.data_len)
                .ok_or_else(|| SchemaError::Invalid("MTP payload bytes overflow".to_owned()))
        })?;
    Ok(Hy4Requirements {
        required_tensor_count: specs.len(),
        checkpoint_tensor_count: index.names().count(),
        checkpoint_shard_count: index.shards().len(),
        checkpoint_file_bytes: index.total_file_bytes(),
        checkpoint_payload_bytes: index.total_payload_bytes(),
        logical_parameter_count,
        fp8_matrix_count: specs
            .iter()
            .filter(|spec| spec.dtype == DType::F8E4M3)
            .count(),
        scale_tensor_count: specs.iter().filter(|spec| spec.dtype == DType::U8).count(),
        base_tensor_count: specs
            .iter()
            .filter(|spec| !spec.name.starts_with("model.mtp_layers."))
            .count(),
        mtp_tensor_count: specs
            .iter()
            .filter(|spec| spec.name.starts_with("model.mtp_layers."))
            .count(),
        dense_layer_count: config.num_hidden_layers - config.sparse_layer_count(),
        moe_layer_count: config.sparse_layer_count(),
        full_indexer_layer_count: config.full_indexer_layer_count(),
        routed_experts_per_layer: config.n_routed_experts,
        selected_experts_per_token: config.num_experts_per_tok,
        context_limit,
        exact_context_ceiling: config.exact_dense_context_ceiling(),
        streamed_layer_bytes,
        decoder_layer_bytes,
        lm_head_bytes,
        resident_bytes,
        kv_cache_bytes,
        maximum_expert_bytes,
        transient_expert_bytes: maximum_expert_bytes,
        expert_cache_bytes,
        expert_slots_per_layer,
        mtp_payload_bytes,
    })
}

pub fn official_tensor_specs(config: &Hy4Config) -> Result<Vec<TensorSpec>, SchemaError> {
    config.validate_release()?;
    let h = config.hidden_size as u64;
    let vocab = config.vocab_size as u64;
    let mut specs = vec![
        TensorSpec::new("model.embed_tokens.weight", DType::Bf16, &[vocab, h]),
        TensorSpec::new("model.norm.weight", DType::Bf16, &[h]),
        TensorSpec::new(
            "model.hc_head.hc_head_fn",
            DType::F32,
            &[
                config.hc_mult as u64,
                (config.hc_mult * config.hidden_size) as u64,
            ],
        ),
        TensorSpec::new(
            "model.hc_head.hc_head_base",
            DType::F32,
            &[config.hc_mult as u64],
        ),
        TensorSpec::new("model.hc_head.hc_head_scale", DType::F32, &[1]),
        TensorSpec::new("lm_head.weight", DType::Bf16, &[vocab, h]),
    ];
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        append_hc(&mut specs, config, &prefix);
        specs.push(TensorSpec::new(
            format!("{prefix}.input_layernorm.weight"),
            DType::Bf16,
            &[h],
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.post_attention_layernorm.weight"),
            DType::Bf16,
            &[h],
        ));
        append_attention(&mut specs, config, &prefix, false);
        if config.layer_has_full_indexer(layer) {
            append_indexer(&mut specs, config, &prefix);
        }
        if config.layer_is_sparse(layer) {
            append_moe(&mut specs, config, &prefix, false);
        } else {
            append_dense_mlp(&mut specs, config, &prefix);
        }
    }
    for mtp in 0..config.num_nextn_predict_layers {
        let prefix = format!("model.mtp_layers.{mtp}");
        specs.extend([
            TensorSpec::new(format!("{prefix}.eh_proj.weight"), DType::Bf16, &[h, 2 * h]),
            TensorSpec::new(format!("{prefix}.enorm.weight"), DType::Bf16, &[h]),
            TensorSpec::new(format!("{prefix}.hnorm.weight"), DType::Bf16, &[h]),
            TensorSpec::new(
                format!("{prefix}.input_layernorm.weight"),
                DType::Bf16,
                &[h],
            ),
            TensorSpec::new(
                format!("{prefix}.post_attention_layernorm.weight"),
                DType::Bf16,
                &[h],
            ),
            TensorSpec::new(
                format!("{prefix}.final_layernorm.weight"),
                DType::Bf16,
                &[h],
            ),
        ]);
        append_attention(&mut specs, config, &prefix, true);
        append_indexer(&mut specs, config, &prefix);
        append_moe(&mut specs, config, &prefix, true);
    }
    specs.sort_by(|left, right| left.name.cmp(&right.name));
    if specs.len() != RELEASE_TENSOR_COUNT {
        return invalid(format!(
            "generated schema has {} tensors; expected {RELEASE_TENSOR_COUNT}",
            specs.len()
        ));
    }
    let unique = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() != specs.len() {
        return invalid("generated Hy4 schema contains duplicate tensor names");
    }
    Ok(specs)
}

fn append_hc(specs: &mut Vec<TensorSpec>, config: &Hy4Config, prefix: &str) {
    let width = (config.hc_mult * config.hidden_size) as u64;
    let gates = (2 * config.hc_mult) as u64;
    for sublayer in ["attn", "mlp"] {
        let stem = format!("{prefix}.hc_{sublayer}_layer.hc_pre");
        specs.extend([
            TensorSpec::new(format!("{stem}.hc_fn"), DType::F32, &[gates, width]),
            TensorSpec::new(format!("{stem}.hc_base"), DType::F32, &[gates]),
            TensorSpec::new(format!("{stem}.hc_scale"), DType::F32, &[2]),
        ]);
    }
}

fn append_attention(specs: &mut Vec<TensorSpec>, config: &Hy4Config, prefix: &str, mtp: bool) {
    let attention = format!("{prefix}.self_attn");
    let h = config.hidden_size as u64;
    append_mxfp8(
        specs,
        &format!("{attention}.q_a_proj"),
        config.q_lora_rank as u64,
        h,
    );
    specs.push(TensorSpec::new(
        format!("{attention}.q_a_layernorm.weight"),
        DType::Bf16,
        &[config.q_lora_rank as u64],
    ));
    append_mxfp8(
        specs,
        &format!("{attention}.q_b_proj"),
        (config.num_attention_heads * config.qk_head_dim) as u64,
        config.q_lora_rank as u64,
    );
    append_mxfp8(
        specs,
        &format!("{attention}.kv_a_proj_with_mqa"),
        (config.kv_lora_rank + config.qk_rope_head_dim) as u64,
        h,
    );
    specs.push(TensorSpec::new(
        format!("{attention}.kv_a_layernorm.weight"),
        DType::Bf16,
        &[config.kv_lora_rank as u64],
    ));
    append_mxfp8(
        specs,
        &format!("{attention}.kv_b_proj"),
        (config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)) as u64,
        config.kv_lora_rank as u64,
    );
    append_mxfp8(
        specs,
        &format!("{attention}.o_proj"),
        h,
        (config.num_attention_heads * config.v_head_dim) as u64,
    );
    specs.push(TensorSpec::new(
        format!("{attention}.linear_gate.weight"),
        DType::Bf16,
        &[(config.num_attention_heads * config.v_head_dim) as u64, h],
    ));
    specs.push(TensorSpec::new(
        format!("{attention}.learnable_sink_param"),
        if mtp { DType::Bf16 } else { DType::F32 },
        &[config.num_attention_heads as u64],
    ));
}

fn append_indexer(specs: &mut Vec<TensorSpec>, config: &Hy4Config, prefix: &str) {
    let indexer = format!("{prefix}.self_attn.indexer");
    append_mxfp8(
        specs,
        &format!("{indexer}.wk"),
        config.index_head_dim as u64,
        config.hidden_size as u64,
    );
    append_mxfp8(
        specs,
        &format!("{indexer}.wq_b"),
        (config.index_n_heads * config.index_head_dim) as u64,
        config.q_lora_rank as u64,
    );
    specs.push(TensorSpec::new(
        format!("{indexer}.weights_proj.weight"),
        DType::Bf16,
        &[config.index_n_heads as u64, config.hidden_size as u64],
    ));
    specs.push(TensorSpec::new(
        format!("{indexer}.k_norm.weight"),
        DType::Bf16,
        &[config.index_head_dim as u64],
    ));
    specs.push(TensorSpec::new(
        format!("{indexer}.k_norm.bias"),
        DType::Bf16,
        &[config.index_head_dim as u64],
    ));
}

fn append_dense_mlp(specs: &mut Vec<TensorSpec>, config: &Hy4Config, prefix: &str) {
    for projection in ["gate_proj", "up_proj"] {
        append_mxfp8(
            specs,
            &format!("{prefix}.mlp.{projection}"),
            config.intermediate_size as u64,
            config.hidden_size as u64,
        );
    }
    append_mxfp8(
        specs,
        &format!("{prefix}.mlp.down_proj"),
        config.hidden_size as u64,
        config.intermediate_size as u64,
    );
}

fn append_moe(specs: &mut Vec<TensorSpec>, config: &Hy4Config, prefix: &str, mtp: bool) {
    let moe = format!("{prefix}.mlp");
    let experts = config.n_routed_experts as u64;
    let h = config.hidden_size as u64;
    let intermediate = config.moe_intermediate_size as u64;
    specs.extend([
        TensorSpec::new(format!("{moe}.gate.weight"), DType::Bf16, &[experts, h]),
        TensorSpec::new(
            format!("{moe}.gate.e_score_correction_bias"),
            if mtp { DType::Bf16 } else { DType::F32 },
            &[experts],
        ),
        TensorSpec::new(
            format!("{moe}.experts.gate_up_proj"),
            DType::F8E4M3,
            &[experts, 2 * intermediate, h],
        ),
        TensorSpec::new(
            format!("{moe}.experts.gate_up_proj_scale"),
            DType::U8,
            &[
                experts,
                2 * intermediate,
                h.div_ceil(MODEL_OPT_MX_BLOCK as u64),
            ],
        ),
        TensorSpec::new(
            format!("{moe}.experts.down_proj"),
            DType::F8E4M3,
            &[experts, h, intermediate],
        ),
        TensorSpec::new(
            format!("{moe}.experts.down_proj_scale"),
            DType::U8,
            &[experts, h, intermediate.div_ceil(MODEL_OPT_MX_BLOCK as u64)],
        ),
    ]);
    let shared = config.n_shared_experts as u64 * intermediate;
    for projection in ["gate_proj", "up_proj"] {
        append_mxfp8(
            specs,
            &format!("{moe}.shared_experts.{projection}"),
            shared,
            h,
        );
    }
    append_mxfp8(specs, &format!("{moe}.shared_experts.down_proj"), h, shared);
}

fn append_mxfp8(specs: &mut Vec<TensorSpec>, stem: &str, rows: u64, columns: u64) {
    specs.push(TensorSpec::new(
        format!("{stem}.weight"),
        DType::F8E4M3,
        &[rows, columns],
    ));
    specs.push(TensorSpec::new(
        format!("{stem}.weight_scale"),
        DType::U8,
        &[rows, columns.div_ceil(MODEL_OPT_MX_BLOCK as u64)],
    ));
}

fn validate_specs(index: &TensorIndex, specs: &[TensorSpec]) -> Result<(), SchemaError> {
    if index.shards().len() != RELEASE_SHARD_COUNT {
        return invalid(format!(
            "checkpoint has {} shards; expected {RELEASE_SHARD_COUNT}",
            index.shards().len()
        ));
    }
    if index.names().count() != specs.len() {
        return invalid(format!(
            "checkpoint has {} tensors; schema requires {}",
            index.names().count(),
            specs.len()
        ));
    }
    let expected = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    let actual = index.names().collect::<BTreeSet<_>>();
    if expected != actual {
        let missing = expected.difference(&actual).next();
        let unexpected = actual.difference(&expected).next();
        return invalid(format!(
            "tensor names differ (first missing={missing:?}, first unexpected={unexpected:?})"
        ));
    }
    for spec in specs {
        let tensor = index.require(&spec.name)?;
        let expected_bytes = spec.payload_bytes()?;
        if tensor.dtype != spec.dtype
            || tensor.shape != spec.shape
            || tensor.data_len != expected_bytes
            || tensor.declared_elements != spec.elements()?
        {
            return invalid(format!(
                "tensor {:?} is {} {:?} / {} bytes; expected {} {:?} / {} bytes",
                spec.name,
                tensor.dtype,
                tensor.shape,
                tensor.data_len,
                spec.dtype,
                spec.shape,
                expected_bytes
            ));
        }
    }
    Ok(())
}

fn validate_hf_assignment(index: &TensorIndex) -> Result<(), SchemaError> {
    let path = index.model_dir().join("model.safetensors.index.json");
    let metadata = fs::metadata(&path).map_err(|source| SchemaError::Read {
        path: path.clone(),
        source,
    })?;
    if metadata.len() > MAX_INDEX_BYTES {
        return invalid(format!("HF index is larger than {MAX_INDEX_BYTES} bytes"));
    }
    let json = fs::read_to_string(&path).map_err(|source| SchemaError::Read {
        path: path.clone(),
        source,
    })?;
    let declared: HfIndex = serde_json::from_str(&json).map_err(|source| SchemaError::Json {
        path: path.clone(),
        source,
    })?;
    if declared.weight_map.len() != index.names().count() {
        return invalid("HF weight map count differs from safetensors headers");
    }
    for tensor in index.tensors() {
        let actual = tensor
            .shard
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                SchemaError::Invalid(format!("invalid shard path for {:?}", tensor.name))
            })?;
        let expected = declared.weight_map.get(&tensor.name).ok_or_else(|| {
            SchemaError::Invalid(format!("HF index omits tensor {:?}", tensor.name))
        })?;
        if actual != expected {
            return invalid(format!(
                "tensor {:?} is in {actual:?}; HF index assigns {expected:?}",
                tensor.name
            ));
        }
    }
    Ok(())
}

fn layer_resident_bytes(index: &TensorIndex, layer: usize) -> Result<u64, SchemaError> {
    let prefix = format!("model.layers.{layer}.");
    resident_subset(index, |tensor| {
        tensor.name.starts_with(&prefix) && !tensor.name.contains(".mlp.experts.")
    })
}

fn resident_subset(
    index: &TensorIndex,
    include: impl Fn(&TensorInfo) -> bool,
) -> Result<u64, SchemaError> {
    index
        .tensors()
        .filter(|tensor| include(tensor))
        .try_fold(0u64, |total, tensor| {
            if tensor.dtype == DType::U8 {
                return Ok(total);
            }
            let bytes = if tensor.dtype == DType::F8E4M3 {
                let scale_name = mxfp8_scale_name(&tensor.name);
                let scale = index.require(&scale_name)?;
                tensor.data_len.checked_add(scale.data_len).ok_or_else(|| {
                    SchemaError::Invalid("MXFP8 resident bytes overflow".to_owned())
                })?
            } else {
                tensor.data_len
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| SchemaError::Invalid("layer resident bytes overflow".to_owned()))
        })
}

fn mxfp8_scale_name(name: &str) -> String {
    if let Some(stem) = name.strip_suffix(".weight") {
        format!("{stem}.weight_scale")
    } else {
        format!("{name}_scale")
    }
}

fn root_resident_bytes(config: &Hy4Config) -> Result<u64, SchemaError> {
    let hc_width = config
        .hc_mult
        .checked_mul(config.hidden_size)
        .ok_or_else(|| SchemaError::Invalid("HC head width overflows".to_owned()))?;
    let hc_parameters = config
        .hc_mult
        .checked_mul(hc_width)
        .and_then(|value| value.checked_add(config.hc_mult + 1))
        .ok_or_else(|| SchemaError::Invalid("HC head parameters overflow".to_owned()))?;
    let values = hc_parameters
        .checked_add(config.hidden_size)
        .ok_or_else(|| SchemaError::Invalid("root resident values overflow".to_owned()))?;
    u64::try_from(values)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| SchemaError::Invalid("root resident bytes overflow".to_owned()))
}

fn expert_resident_bytes(config: &Hy4Config) -> Result<u64, SchemaError> {
    let h = config.hidden_size as u64;
    let intermediate = config.moe_intermediate_size as u64;
    let gate_up_values = (2 * intermediate)
        .checked_mul(h)
        .ok_or_else(|| SchemaError::Invalid("gate-up expert values overflow".to_owned()))?;
    let gate_up_scales = (2 * intermediate)
        .checked_mul(h.div_ceil(MODEL_OPT_MX_BLOCK as u64))
        .ok_or_else(|| SchemaError::Invalid("gate-up expert scales overflow".to_owned()))?;
    let down_values = h
        .checked_mul(intermediate)
        .ok_or_else(|| SchemaError::Invalid("down expert values overflow".to_owned()))?;
    let down_scales = h
        .checked_mul(intermediate.div_ceil(MODEL_OPT_MX_BLOCK as u64))
        .ok_or_else(|| SchemaError::Invalid("down expert scales overflow".to_owned()))?;
    gate_up_values
        .checked_add(gate_up_scales)
        .and_then(|value| value.checked_add(down_values))
        .and_then(|value| value.checked_add(down_scales))
        .ok_or_else(|| SchemaError::Invalid("expert resident bytes overflow".to_owned()))
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, SchemaError> {
    Err(SchemaError::Invalid(reason.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_schema_has_the_expected_partition() {
        let config = crate::models::hy4::config::release_test_config();
        let specs = official_tensor_specs(&config).unwrap();
        assert_eq!(specs.len(), RELEASE_TENSOR_COUNT);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.name.starts_with("model.mtp_layers."))
                .count(),
            RELEASE_MTP_TENSOR_COUNT
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| !spec.name.starts_with("model.mtp_layers."))
                .count(),
            RELEASE_BASE_TENSOR_COUNT
        );
    }
}
