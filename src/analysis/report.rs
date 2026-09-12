use super::classify::{
    classify_tensor, expected_logical_elements, expected_matrix_shape, supported_group_size,
    ParameterCategory,
};
use crate::config::{ConfigError, DeepseekV4Config, GlmConfig, ModelConfig};
use crate::models::deepseek_v4::schema as deepseek_schema;
use crate::models::kimi_k3::KimiK3Config;
use crate::storage::{
    infer_unique_packed_bits, validate_quantized_scale_layout, DType, SafetensorError, TensorIndex,
    TensorInfo,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum AnalysisError {
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Arithmetic(String),
    Schema(String),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(f),
            Self::Checkpoint(error) => error.fmt(f),
            Self::Arithmetic(reason) => write!(f, "analysis arithmetic error: {reason}"),
            Self::Schema(reason) => write!(f, "checkpoint schema error: {reason}"),
        }
    }
}

impl std::error::Error for AnalysisError {}

impl From<ConfigError> for AnalysisError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl From<SafetensorError> for AnalysisError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub model_type: String,
    pub hidden_size: usize,
    pub layers: usize,
    pub dense_layers: usize,
    pub sparse_layers: usize,
    pub routed_experts_per_layer: usize,
    pub active_experts_per_token: usize,
    pub attention_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub full_indexer_layers: usize,
    pub index_topk: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CategoryStats {
    pub category: ParameterCategory,
    pub tensor_count: u64,
    pub logical_parameters: u64,
    pub payload_bytes: u64,
    pub scale_bytes: u64,
}

impl CategoryStats {
    fn new(category: ParameterCategory) -> Self {
        Self {
            category,
            tensor_count: 0,
            logical_parameters: 0,
            payload_bytes: 0,
            scale_bytes: 0,
        }
    }

    pub fn total_storage_bytes(&self) -> u64 {
        self.payload_bytes.saturating_add(self.scale_bytes)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizationStats {
    pub bits_per_weight: u8,
    pub matrix_count: u64,
    pub logical_parameters: u64,
    pub payload_bytes: u64,
    pub scale_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KvEstimate {
    pub state_bytes: u8,
    pub mla_values_per_token_per_layer: u64,
    pub mla_compressed_bytes_per_token: u64,
    pub configured_indexer_bytes_per_token: u64,
    pub compressed_bytes_per_token: u64,
    pub conventional_mha_bytes_per_token: u64,
    pub compression_ratio: f64,
    pub bytes_at_4k_context: u64,
    pub bytes_at_32k_context: u64,
    pub bytes_at_max_context: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckpointReport {
    pub model_path: PathBuf,
    pub model: ModelSummary,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub file_bytes: u64,
    pub tensor_payload_bytes: u64,
    pub container_overhead_bytes: u64,
    pub known_logical_parameters: u64,
    pub estimated_active_parameters_per_token: u64,
    pub unknown_packed_bytes: u64,
    pub categories: Vec<CategoryStats>,
    pub quantization: Vec<QuantizationStats>,
    pub kv_cache_f32: KvEstimate,
    pub routed_expert_count_found: usize,
    pub minimum_routed_expert_bytes: u64,
    pub maximum_routed_expert_bytes: u64,
    pub required_base_tensor_count: usize,
    pub missing_base_tensor_count: usize,
    pub sidecars_without_weight: usize,
    pub indexer_weights_complete: bool,
    pub format_notes: Vec<String>,
    pub warnings: Vec<String>,
}

impl CheckpointReport {
    pub fn category(&self, category: ParameterCategory) -> Option<&CategoryStats> {
        self.categories
            .iter()
            .find(|stats| stats.category == category)
    }
}

#[derive(Default)]
struct MutableQuant {
    matrices: u64,
    params: u64,
    payload: u64,
    scales: u64,
}

pub fn analyze_checkpoint(model_dir: &Path) -> Result<CheckpointReport, AnalysisError> {
    let index = TensorIndex::open(model_dir)?;
    match ModelConfig::load(model_dir)? {
        ModelConfig::Glm52(config) => analyze_index(model_dir, &config, &index),
        ModelConfig::DeepseekV4(config) => analyze_deepseek_index(model_dir, &config, &index),
        ModelConfig::DeepseekV41(_) => Err(AnalysisError::Schema(
            "DeepSeek-V4.1 uses its CED/CSA2/Engram release schema; use the V4.1 inspect/preflight adapter"
                .to_owned(),
        )),
        ModelConfig::Hy4(_) => Err(AnalysisError::Schema(
            "Hy4 uses its native release-specific 2,838-tensor schema report; use the Hy4 inspect/preflight adapter"
                .to_owned(),
        )),
        ModelConfig::KimiK3(config) => analyze_kimi_k3_index(model_dir, &config, &index),
        ModelConfig::Qwen38(_) => Err(AnalysisError::Schema(
            "Qwen3.8 uses its release-specific 287,119-tensor manifest/schema report; the generic MLA-oriented CheckpointReport would mislabel hybrid recurrent/GQA state"
                .to_owned(),
        )),
    }
}

fn analyze_kimi_k3_index(
    model_dir: &Path,
    config: &KimiK3Config,
    index: &TensorIndex,
) -> Result<CheckpointReport, AnalysisError> {
    let text = &config.text_config;
    let expected_tensor_count = kimi_k3_expected_tensor_count(config)?;
    let mut categories: BTreeMap<ParameterCategory, CategoryStats> = BTreeMap::new();
    let mut quantization: BTreeMap<u8, MutableQuant> = BTreeMap::new();
    let mut expert_parts: BTreeMap<(usize, usize), (u8, u64)> = BTreeMap::new();
    let mut sidecars_without_weight = 0usize;
    let mut unknown_packed_bytes = 0u64;
    let mut known_parameters = 0u64;

    for tensor in index.tensors() {
        if tensor.name.ends_with(".weight_scale") {
            let weight_name = tensor
                .name
                .strip_suffix(".weight_scale")
                .map(|prefix| format!("{prefix}.weight_packed"))
                .expect("suffix checked above");
            if index.get(&weight_name).is_none() {
                sidecars_without_weight += 1;
            }
            let category = classify_kimi_k3_tensor(&weight_name);
            let stats = categories
                .entry(category)
                .or_insert_with(|| CategoryStats::new(category));
            stats.scale_bytes = add(stats.scale_bytes, tensor.data_len, "Kimi-K3 scale bytes")?;
            let quant = quantization.entry(4).or_default();
            quant.scales = add(quant.scales, tensor.data_len, "Kimi-K3 MXFP4 scales")?;
            if let Some((layer, expert, bit)) = kimi_k3_expert_part(&tensor.name) {
                let entry = expert_parts.entry((layer, expert)).or_default();
                entry.0 |= bit;
                entry.1 = add(entry.1, tensor.data_len, "Kimi-K3 expert bytes")?;
            }
            continue;
        }

        let category = classify_kimi_k3_tensor(&tensor.name);
        let logical = if tensor.name.ends_with(".weight_packed") {
            if tensor.dtype != DType::U8 || tensor.shape.len() != 2 {
                unknown_packed_bytes = add(
                    unknown_packed_bytes,
                    tensor.data_len,
                    "Kimi-K3 unknown packed bytes",
                )?;
                0
            } else {
                tensor.declared_elements.checked_mul(2).ok_or_else(|| {
                    AnalysisError::Arithmetic(
                        "Kimi-K3 MXFP4 logical element count overflows".to_owned(),
                    )
                })?
            }
        } else {
            tensor.declared_elements
        };
        let stats = categories
            .entry(category)
            .or_insert_with(|| CategoryStats::new(category));
        stats.tensor_count = add(stats.tensor_count, 1, "Kimi-K3 category tensors")?;
        stats.payload_bytes = add(stats.payload_bytes, tensor.data_len, "Kimi-K3 payload")?;
        stats.logical_parameters = add(
            stats.logical_parameters,
            logical,
            "Kimi-K3 logical parameters",
        )?;
        known_parameters = add(
            known_parameters,
            logical,
            "Kimi-K3 known logical parameters",
        )?;

        let bits = if tensor.name.ends_with(".weight_packed") {
            4
        } else {
            match tensor.dtype {
                DType::F64 | DType::U64 | DType::I64 => 64,
                DType::F32 | DType::U32 | DType::I32 => 32,
                DType::F16 | DType::Bf16 | DType::U16 | DType::I16 => 16,
                DType::F8E4M3 | DType::F8E5M2 | DType::F8E8M0 | DType::U8 | DType::I8 => 8,
                _ => 0,
            }
        };
        if bits > 0 {
            let quant = quantization.entry(bits).or_default();
            quant.matrices = add(quant.matrices, 1, "Kimi-K3 dtype tensors")?;
            quant.params = add(quant.params, logical, "Kimi-K3 dtype parameters")?;
            quant.payload = add(quant.payload, tensor.data_len, "Kimi-K3 dtype payload")?;
        }
        if let Some((layer, expert, bit)) = kimi_k3_expert_part(&tensor.name) {
            let entry = expert_parts.entry((layer, expert)).or_default();
            entry.0 |= bit;
            entry.1 = add(entry.1, tensor.data_len, "Kimi-K3 expert bytes")?;
        }
    }

    let complete_experts = expert_parts
        .values()
        .filter(|(mask, _)| *mask == 0b11_1111)
        .map(|(_, bytes)| *bytes)
        .collect::<Vec<_>>();
    let routed_parameters = categories
        .get(&ParameterCategory::RoutedExpert)
        .map(|stats| stats.logical_parameters)
        .unwrap_or(0);
    let always_active = known_parameters.saturating_sub(routed_parameters);
    let active_routed = (u128::from(routed_parameters) * text.num_experts_per_token as u128
        / text.num_experts as u128) as u64;
    let checkpoint_tensor_count = index.names().count();
    let missing_count = expected_tensor_count.saturating_sub(checkpoint_tensor_count);
    let mut warnings = Vec::new();
    if missing_count > 0 {
        warnings.push(format!(
            "checkpoint transfer is incomplete: {checkpoint_tensor_count}/{expected_tensor_count} expected tensor entries are currently visible"
        ));
    }
    if checkpoint_tensor_count > expected_tensor_count {
        warnings.push(format!(
            "checkpoint contains {} entries beyond the official Kimi-K3 manifest count; strict preflight must reject or explain them",
            checkpoint_tensor_count - expected_tensor_count
        ));
    }
    if sidecars_without_weight > 0 {
        warnings.push(format!(
            "{sidecars_without_weight} Kimi-K3 .weight_scale tensors have no matching .weight_packed tensor"
        ));
    }
    if !has_kimi_k3_tokenizer(model_dir) {
        warnings.push(
            "tiktoken.model/tokenizer_config.json are not both present; Kimi-K3 text prompting is not ready in this directory"
                .to_owned(),
        );
    }
    if !shard_numbers_are_contiguous(index) {
        warnings.push("numbered safetensors shard filenames contain a gap".to_owned());
    }
    warnings.push(
        "vision weights are included in totals, but the first Kimi-K3 execution milestone is text-only"
            .to_owned(),
    );

    let kda_state_bytes = text
        .linear_attn_config
        .kda_layers
        .len()
        .saturating_mul(text.linear_attn_config.num_heads)
        .saturating_mul(text.linear_attn_config.head_dim)
        .saturating_mul(text.linear_attn_config.head_dim)
        .saturating_mul(4);
    let conv_history_bytes = text
        .linear_attn_config
        .kda_layers
        .len()
        .saturating_mul(3)
        .saturating_mul(text.linear_attn_config.num_heads)
        .saturating_mul(text.linear_attn_config.head_dim)
        .saturating_mul(
            text.linear_attn_config
                .short_conv_kernel_size
                .saturating_sub(1),
        )
        .saturating_mul(4);
    let file_bytes = index.total_file_bytes();
    let payload_bytes = index.total_payload_bytes();
    Ok(CheckpointReport {
        model_path: model_dir.to_path_buf(),
        model: ModelSummary {
            model_type: config.model_type.clone(),
            hidden_size: text.hidden_size,
            layers: text.num_hidden_layers,
            dense_layers: text.first_k_dense_replace,
            sparse_layers: text.num_hidden_layers.saturating_sub(text.first_k_dense_replace),
            routed_experts_per_layer: text.num_experts,
            active_experts_per_token: text.num_experts_per_token,
            attention_heads: text.num_attention_heads,
            vocab_size: text.vocab_size,
            max_context: text.max_position_embeddings,
            full_indexer_layers: text.full_attention_layer_count(),
            index_topk: 0,
        },
        shard_count: index.shards().len(),
        tensor_count: checkpoint_tensor_count,
        file_bytes,
        tensor_payload_bytes: payload_bytes,
        container_overhead_bytes: file_bytes.saturating_sub(payload_bytes),
        known_logical_parameters: known_parameters,
        estimated_active_parameters_per_token: always_active.saturating_add(active_routed),
        unknown_packed_bytes,
        categories: categories.into_values().collect(),
        quantization: quantization
            .into_iter()
            .map(|(bits, stats)| QuantizationStats {
                bits_per_weight: bits,
                matrix_count: stats.matrices,
                logical_parameters: stats.params,
                payload_bytes: stats.payload,
                scale_bytes: stats.scales,
            })
            .collect(),
        kv_cache_f32: kimi_k3_kv_estimate(config),
        routed_expert_count_found: complete_experts.len(),
        minimum_routed_expert_bytes: complete_experts.iter().copied().min().unwrap_or(0),
        maximum_routed_expert_bytes: complete_experts.iter().copied().max().unwrap_or(0),
        required_base_tensor_count: expected_tensor_count,
        missing_base_tensor_count: missing_count,
        sidecars_without_weight,
        indexer_weights_complete: true,
        format_notes: vec![
            "routed experts use native low-nibble-first MXFP4 E2M1 with one U8 E8M0 scale per 32 logical values"
                .to_owned(),
            format!(
                "hybrid decoder has {} recurrent KDA layers and {} gated NoPE-MLA layers",
                text.kda_layer_count(),
                text.full_attention_layer_count()
            ),
            format!(
                "F32 KDA recurrent state is {kda_state_bytes} bytes plus {conv_history_bytes} bytes of causal convolution history"
            ),
        ],
        warnings,
    })
}

fn kimi_k3_expected_tensor_count(config: &KimiK3Config) -> Result<usize, AnalysisError> {
    let text = &config.text_config;
    // Five language roots, three projector tensors, then six tensors per vision block plus the
    // patch projection, position embedding, and final normalization.
    let vision = config
        .vision_config
        .vt_num_hidden_layers
        .checked_mul(6)
        .and_then(|count| count.checked_add(3))
        .ok_or_else(|| {
            AnalysisError::Arithmetic("Kimi-K3 vision tensor count overflows".to_owned())
        })?;
    let mut total = 5usize
        .checked_add(3)
        .and_then(|count| count.checked_add(vision))
        .ok_or_else(|| {
            AnalysisError::Arithmetic("Kimi-K3 root tensor count overflows".to_owned())
        })?;
    for layer in 0..text.num_hidden_layers {
        let attention = if text.is_kda_layer(layer) { 14 } else { 8 };
        let feed_forward = if layer < text.first_k_dense_replace {
            3
        } else {
            text.num_experts
                .checked_mul(6)
                .and_then(|count| count.checked_add(8))
                .ok_or_else(|| {
                    AnalysisError::Arithmetic(
                        "Kimi-K3 sparse-layer tensor count overflows".to_owned(),
                    )
                })?
        };
        total = total
            .checked_add(6)
            .and_then(|count| count.checked_add(attention))
            .and_then(|count| count.checked_add(feed_forward))
            .ok_or_else(|| {
                AnalysisError::Arithmetic("Kimi-K3 layer tensor count overflows".to_owned())
            })?;
    }
    Ok(total)
}

fn classify_kimi_k3_tensor(name: &str) -> ParameterCategory {
    let name = name.strip_suffix(".weight_scale").unwrap_or(name);
    if name == "language_model.model.embed_tokens.weight" {
        ParameterCategory::Embedding
    } else if name == "language_model.lm_head.weight" {
        ParameterCategory::LmHead
    } else if name.contains(".block_sparse_moe.experts.") {
        ParameterCategory::RoutedExpert
    } else if name.contains(".block_sparse_moe.shared_experts.") {
        ParameterCategory::SharedExpert
    } else if name.contains(".block_sparse_moe.gate.") {
        ParameterCategory::Router
    } else if name.contains(".block_sparse_moe.routed_expert_")
        || (name.contains(".mlp.") && name.starts_with("language_model."))
    {
        ParameterCategory::DenseMlp
    } else if name.contains(".self_attn.") {
        ParameterCategory::Attention
    } else if name.contains("norm") || name.contains("layernorm") {
        ParameterCategory::Normalization
    } else {
        ParameterCategory::Other
    }
}

fn kimi_k3_expert_part(name: &str) -> Option<(usize, usize, u8)> {
    let tail = name.strip_prefix("language_model.model.layers.")?;
    let (layer, rest) = tail.split_once('.')?;
    let tail = rest.strip_prefix("block_sparse_moe.experts.")?;
    let (expert, projection) = tail.split_once('.')?;
    let bit = match projection {
        "w1.weight_packed" => 0b00_0001,
        "w1.weight_scale" => 0b00_0010,
        "w2.weight_packed" => 0b00_0100,
        "w2.weight_scale" => 0b00_1000,
        "w3.weight_packed" => 0b01_0000,
        "w3.weight_scale" => 0b10_0000,
        _ => return None,
    };
    Some((layer.parse().ok()?, expert.parse().ok()?, bit))
}

fn kimi_k3_kv_estimate(config: &KimiK3Config) -> KvEstimate {
    let text = &config.text_config;
    let mla_values = (text.kv_lora_rank + text.qk_rope_head_dim) as u64;
    let mla_layers = text.full_attention_layer_count() as u64;
    let compressed = mla_values.saturating_mul(mla_layers).saturating_mul(4);
    let conventional = text.num_hidden_layers as u64
        * text.num_attention_heads as u64
        * (text.qk_nope_head_dim + text.qk_rope_head_dim + text.v_head_dim) as u64
        * 4;
    KvEstimate {
        state_bytes: 4,
        mla_values_per_token_per_layer: mla_values,
        mla_compressed_bytes_per_token: compressed,
        configured_indexer_bytes_per_token: 0,
        compressed_bytes_per_token: compressed,
        conventional_mha_bytes_per_token: conventional,
        compression_ratio: conventional as f64 / compressed.max(1) as f64,
        bytes_at_4k_context: compressed.saturating_mul(4096),
        bytes_at_32k_context: compressed.saturating_mul(32_768),
        bytes_at_max_context: compressed.saturating_mul(text.max_position_embeddings as u64),
    }
}

fn analyze_deepseek_index(
    model_dir: &Path,
    config: &DeepseekV4Config,
    index: &TensorIndex,
) -> Result<CheckpointReport, AnalysisError> {
    let requirements = deepseek_schema::inspect_requirements(config, index, 1, 0)
        .map_err(|error| AnalysisError::Schema(error.to_string()))?;
    let mut categories: BTreeMap<ParameterCategory, CategoryStats> = BTreeMap::new();
    let mut quantization: BTreeMap<u8, MutableQuant> = BTreeMap::new();
    let mut expert_bytes: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut experts = BTreeSet::new();

    for tensor in index.tensors() {
        if tensor.name.ends_with(".scale") {
            let weight_name = format!("{}.weight", tensor.name.trim_end_matches(".scale"));
            let category = classify_deepseek_tensor(&weight_name, config.num_hidden_layers);
            let stats = categories
                .entry(category)
                .or_insert_with(|| CategoryStats::new(category));
            stats.scale_bytes = add(stats.scale_bytes, tensor.data_len, "DeepSeek scale bytes")?;
            let bits = index
                .get(&weight_name)
                .map(|weight| {
                    if weight.dtype == DType::I8 && weight.name.contains(".experts.") {
                        4
                    } else {
                        8
                    }
                })
                .unwrap_or(8);
            let quant = quantization.entry(bits).or_default();
            quant.scales = add(quant.scales, tensor.data_len, "DeepSeek quant scale bytes")?;
            continue;
        }
        let category = classify_deepseek_tensor(&tensor.name, config.num_hidden_layers);
        let logical = if tensor.dtype == DType::I8 && tensor.name.contains(".experts.") {
            tensor.declared_elements.checked_mul(2).ok_or_else(|| {
                AnalysisError::Arithmetic("DeepSeek FP4 logical elements overflow".to_owned())
            })?
        } else {
            tensor.declared_elements
        };
        let stats = categories
            .entry(category)
            .or_insert_with(|| CategoryStats::new(category));
        stats.tensor_count = add(stats.tensor_count, 1, "DeepSeek category tensors")?;
        stats.payload_bytes = add(stats.payload_bytes, tensor.data_len, "DeepSeek payload")?;
        stats.logical_parameters = add(
            stats.logical_parameters,
            logical,
            "DeepSeek category parameters",
        )?;
        let bits = match tensor.dtype {
            DType::I8 if tensor.name.contains(".experts.") => 4,
            DType::F8E4M3 | DType::F8E8M0 | DType::U8 | DType::I8 => 8,
            DType::F16 | DType::Bf16 | DType::U16 | DType::I16 => 16,
            DType::F32 | DType::U32 | DType::I32 => 32,
            DType::F64 | DType::U64 | DType::I64 => 64,
            _ => 0,
        };
        if bits > 0 {
            let quant = quantization.entry(bits).or_default();
            quant.matrices = add(quant.matrices, 1, "DeepSeek dtype tensors")?;
            quant.params = add(quant.params, logical, "DeepSeek dtype parameters")?;
            quant.payload = add(quant.payload, tensor.data_len, "DeepSeek dtype payload")?;
        }
        if let Some((layer, expert)) = deepseek_expert_key(&tensor.name) {
            experts.insert((layer, expert));
            let bytes = tensor.data_len.saturating_add(
                index
                    .get(&format!(
                        "{}.scale",
                        tensor.name.trim_end_matches(".weight")
                    ))
                    .map(|scale| scale.data_len)
                    .unwrap_or(0),
            );
            let total = expert_bytes.entry((layer, expert)).or_default();
            *total = add(*total, bytes, "DeepSeek expert bytes")?;
        }
    }
    let known_parameters = requirements
        .base_logical_parameters
        .saturating_add(requirements.dspark_logical_parameters);
    let routed_parameters = categories
        .get(&ParameterCategory::RoutedExpert)
        .map(|stats| stats.logical_parameters)
        .unwrap_or(0);
    let always_active = requirements
        .base_logical_parameters
        .saturating_sub(routed_parameters);
    let active_routed = (u128::from(routed_parameters) * config.num_experts_per_tok as u128
        / config.n_routed_experts as u128) as u64;
    let file_bytes = index.total_file_bytes();
    let payload_bytes = index.total_payload_bytes();
    let expert_sizes = expert_bytes.values().copied().collect::<Vec<_>>();
    let quantization = quantization
        .into_iter()
        .map(|(bits, stats)| QuantizationStats {
            bits_per_weight: bits,
            matrix_count: stats.matrices,
            logical_parameters: stats.params,
            payload_bytes: stats.payload,
            scale_bytes: stats.scales,
        })
        .collect();
    Ok(CheckpointReport {
        model_path: model_dir.to_path_buf(),
        model: ModelSummary {
            model_type: config.model_type.clone(),
            hidden_size: config.hidden_size,
            layers: config.num_hidden_layers,
            dense_layers: 0,
            sparse_layers: config.num_hidden_layers,
            routed_experts_per_layer: config.n_routed_experts,
            active_experts_per_token: config.num_experts_per_tok,
            attention_heads: config.num_attention_heads,
            vocab_size: config.vocab_size,
            max_context: config.max_position_embeddings,
            full_indexer_layers: config.indexed_layer_count(),
            index_topk: config.index_topk,
        },
        shard_count: index.shards().len(),
        tensor_count: index.names().count(),
        file_bytes,
        tensor_payload_bytes: payload_bytes,
        container_overhead_bytes: file_bytes.saturating_sub(payload_bytes),
        known_logical_parameters: known_parameters,
        estimated_active_parameters_per_token: always_active.saturating_add(active_routed),
        unknown_packed_bytes: 0,
        categories: categories.into_values().collect(),
        quantization,
        kv_cache_f32: deepseek_kv_estimate(config),
        routed_expert_count_found: experts.len(),
        minimum_routed_expert_bytes: expert_sizes.iter().copied().min().unwrap_or(0),
        maximum_routed_expert_bytes: expert_sizes.iter().copied().max().unwrap_or(0),
        required_base_tensor_count: requirements.required_tensor_count,
        missing_base_tensor_count: 0,
        sidecars_without_weight: 0,
        indexer_weights_complete: true,
        format_notes: vec![
            "DeepSeek-V4 native MXFP8 E4M3/E8M0 block-128 matrices".to_owned(),
            "routed experts use low-nibble-first MXFP4 E2M1 with 32-value groups".to_owned(),
            format!(
                "{} attached DSpark stage(s) are reported separately from the 43-layer base",
                requirements.dspark_stage_count
            ),
        ],
        warnings: vec![
            "scalar base-model execution is a correctness path; DSpark speculative decoding is not yet enabled"
                .to_owned(),
        ],
    })
}

fn classify_deepseek_tensor(name: &str, base_layers: usize) -> ParameterCategory {
    if name.starts_with("mtp.") {
        return ParameterCategory::Mtp;
    }
    if name == "embed.weight" {
        ParameterCategory::Embedding
    } else if name == "head.weight" {
        ParameterCategory::LmHead
    } else if name.contains(".attn.indexer.") {
        ParameterCategory::Indexer
    } else if name.contains(".ffn.experts.") {
        ParameterCategory::RoutedExpert
    } else if name.contains(".ffn.shared_experts.") {
        ParameterCategory::SharedExpert
    } else if name.contains(".ffn.gate.") {
        ParameterCategory::Router
    } else if name.contains(".attn.") {
        ParameterCategory::Attention
    } else if name.contains("norm") {
        ParameterCategory::Normalization
    } else if name.starts_with("layers.")
        && name
            .split('.')
            .nth(1)
            .and_then(|layer| layer.parse::<usize>().ok())
            .is_some_and(|layer| layer >= base_layers)
    {
        ParameterCategory::Mtp
    } else {
        ParameterCategory::Other
    }
}

fn deepseek_expert_key(name: &str) -> Option<(usize, usize)> {
    let tail = name.strip_prefix("layers.")?;
    let (layer, rest) = tail.split_once('.')?;
    let tail = rest.strip_prefix("ffn.experts.")?;
    let (expert, projection) = tail.split_once('.')?;
    if !matches!(projection, "w1.weight" | "w2.weight" | "w3.weight") {
        return None;
    }
    Some((layer.parse().ok()?, expert.parse().ok()?))
}

fn deepseek_kv_estimate(config: &DeepseekV4Config) -> KvEstimate {
    let bytes = |context: usize| {
        deepseek_schema::kv_cache_bytes_for_context(config, context).unwrap_or(u64::MAX)
    };
    let at_4k = bytes(4096.min(config.max_position_embeddings));
    let at_32k = bytes(32_768.min(config.max_position_embeddings));
    let at_max = bytes(config.max_position_embeddings);
    let conventional = config.num_hidden_layers as u64
        * config.num_attention_heads as u64
        * config.head_dim as u64
        * 2
        * 4;
    let compressed_per_token = at_max / config.max_position_embeddings as u64;
    KvEstimate {
        state_bytes: 4,
        mla_values_per_token_per_layer: config.head_dim as u64,
        mla_compressed_bytes_per_token: compressed_per_token,
        configured_indexer_bytes_per_token: config.indexed_layer_count() as u64
            * config.index_head_dim as u64
            * 4,
        compressed_bytes_per_token: compressed_per_token,
        conventional_mha_bytes_per_token: conventional,
        compression_ratio: conventional as f64 / compressed_per_token.max(1) as f64,
        bytes_at_4k_context: at_4k,
        bytes_at_32k_context: at_32k,
        bytes_at_max_context: at_max,
    }
}

fn analyze_index(
    model_dir: &Path,
    config: &GlmConfig,
    index: &TensorIndex,
) -> Result<CheckpointReport, AnalysisError> {
    let mut category_map: BTreeMap<ParameterCategory, CategoryStats> = BTreeMap::new();
    let mut quant_map: BTreeMap<u8, MutableQuant> = BTreeMap::new();
    let mut known_params = 0u64;
    let mut unknown_packed_bytes = 0u64;
    let mut sidecars_without_weight = 0usize;
    let mut per_row_int4 = 0u64;
    let mut grouped_int4 = BTreeSet::new();
    let mut routed_experts = BTreeSet::new();
    let mut routed_expert_projections: BTreeMap<(usize, usize), (u8, u64)> = BTreeMap::new();

    let mut schema_errors = Vec::new();
    for tensor in index
        .tensors()
        .filter(|tensor| !tensor.name.ends_with(".qs"))
    {
        let sidecar = index.get(&format!("{}.qs", tensor.name));
        if let Err(reason) = validate_known_tensor_schema(tensor, sidecar, config) {
            schema_errors.push(format!("{}: {reason}", tensor.name));
        }
    }
    if !schema_errors.is_empty() {
        let examples = schema_errors
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ");
        return Err(AnalysisError::Schema(format!(
            "{} known tensor(s) violate the supported dtype/shape/sidecar contract; {examples}",
            schema_errors.len()
        )));
    }

    for name in index.names().filter(|name| name.ends_with(".qs")) {
        let base = name.trim_end_matches(".qs");
        if index.get(base).is_none() {
            sidecars_without_weight += 1;
        }
    }

    for tensor in index.tensors() {
        if tensor.name.ends_with(".qs") {
            continue;
        }
        let category = classify_tensor(&tensor.name, config);
        let sidecar = index.get(&format!("{}.qs", tensor.name));
        let logical = expected_logical_elements(&tensor.name, config).or_else(|| {
            if tensor.dtype == DType::U8 && sidecar.is_some() {
                None
            } else {
                Some(tensor.declared_elements)
            }
        });
        let stats = category_map
            .entry(category)
            .or_insert_with(|| CategoryStats::new(category));
        stats.tensor_count = add(stats.tensor_count, 1, "category tensor count")?;
        stats.payload_bytes = add(stats.payload_bytes, tensor.data_len, "category payload")?;
        if let Some(scale) = sidecar {
            stats.scale_bytes = add(stats.scale_bytes, scale.data_len, "category scales")?;
        }
        if let Some(logical) = logical {
            stats.logical_parameters = add(stats.logical_parameters, logical, "category params")?;
            known_params = add(known_params, logical, "known parameter count")?;
            if let Some(bits) = infer_bits(tensor, sidecar, logical, config) {
                let quant = quant_map.entry(bits).or_default();
                quant.matrices = add(quant.matrices, 1, "quant matrix count")?;
                quant.params = add(quant.params, logical, "quant params")?;
                quant.payload = add(quant.payload, tensor.data_len, "quant payload")?;
                if let Some(scale) = sidecar {
                    quant.scales = add(quant.scales, scale.data_len, "quant scales")?;
                }
                if bits == 4 {
                    if let (Some(scale), Some((rows, cols))) =
                        (sidecar, expected_matrix_shape(&tensor.name, config))
                    {
                        if scale.declared_elements == rows {
                            per_row_int4 += 1;
                        } else if rows > 0 && scale.declared_elements % rows == 0 {
                            let groups = scale.declared_elements / rows;
                            if groups > 0 {
                                if let Some(group_size) =
                                    supported_group_size(rows, cols, scale.declared_elements)
                                {
                                    grouped_int4.insert(group_size);
                                }
                            }
                        }
                    }
                }
            }
        } else if tensor.dtype == DType::U8 {
            unknown_packed_bytes = add(
                unknown_packed_bytes,
                tensor.data_len,
                "unknown packed bytes",
            )?;
        }
        if category == ParameterCategory::RoutedExpert {
            if let Some((key, projection)) = expert_projection_key(&tensor.name) {
                let entry = routed_expert_projections.entry(key).or_default();
                entry.0 |= projection;
                entry.1 = add(
                    entry.1,
                    tensor
                        .data_len
                        .saturating_add(sidecar.map(|scale| scale.data_len).unwrap_or(0)),
                    "routed expert storage",
                )?;
            }
        }
    }
    let mut complete_expert_bytes = Vec::new();
    for (key, (mask, bytes)) in routed_expert_projections {
        if mask == 0b111 {
            routed_experts.insert(key);
            complete_expert_bytes.push(bytes);
        }
    }
    let minimum_routed_expert_bytes = complete_expert_bytes.iter().copied().min().unwrap_or(0);
    let maximum_routed_expert_bytes = complete_expert_bytes.iter().copied().max().unwrap_or(0);

    let routed_params = category_map
        .get(&ParameterCategory::RoutedExpert)
        .map(|stats| stats.logical_parameters)
        .unwrap_or(0);
    let mtp_params = category_map
        .get(&ParameterCategory::Mtp)
        .map(|stats| stats.logical_parameters)
        .unwrap_or(0);
    let always_active = known_params
        .saturating_sub(routed_params)
        .saturating_sub(mtp_params);
    let active_routed = (u128::from(routed_params) * config.num_experts_per_tok as u128
        / config.n_routed_experts as u128) as u64;
    let active_params = add(always_active, active_routed, "active parameter estimate")?;

    let required = required_base_tensors(config);
    let missing: Vec<String> = required
        .iter()
        .filter(|name| index.get(name).is_none())
        .take(8)
        .cloned()
        .collect();
    let missing_count = required
        .iter()
        .filter(|name| index.get(name).is_none())
        .count();

    let mut warnings = Vec::new();
    if missing_count > 0 {
        warnings.push(format!(
            "checkpoint is missing {missing_count}/{} required base tensors; examples: {}",
            required.len(),
            missing.join(", ")
        ));
    }
    if sidecars_without_weight > 0 {
        warnings.push(format!(
            "{sidecars_without_weight} .qs scale tensors have no matching weight tensor"
        ));
    }
    if unknown_packed_bytes > 0 {
        warnings.push(format!(
            "{unknown_packed_bytes} packed U8 bytes have no known GLM logical shape and are excluded from parameter totals"
        ));
    }
    if !has_tokenizer(model_dir) {
        warnings.push(
            "no tokenizer.json/tokenizer.model was found; chat inference is not ready in this directory"
                .to_owned(),
        );
    }
    let indexer_weights_complete = indexer_weights_present(config, index);
    if !indexer_weights_complete {
        warnings.push(
            "DSA/IndexShare is configured but full indexer weights are absent; long-context sparse attention cannot be claimed"
                .to_owned(),
        );
    }
    if !shard_numbers_are_contiguous(index) {
        warnings.push("numbered safetensors shard filenames contain a gap".to_owned());
    }

    let mut format_notes = Vec::new();
    if per_row_int4 > 0 {
        format_notes.push(format!(
            "detected {per_row_int4} per-row INT4 matrices (legacy Colibrì layout)"
        ));
        warnings.push(
            "legacy per-row INT4 is inspectable, but it is not the quality baseline for future Urbilateria inference; prefer validated grouped-scale INT4"
                .to_owned(),
        );
    }
    for group_size in grouped_int4 {
        format_notes.push(format!(
            "detected grouped INT4 with inferred group size {group_size}"
        ));
    }
    if config.dtype.as_deref() == Some("bfloat16") && quant_map.contains_key(&4) {
        format_notes.push(
            "config dtype describes the source model; shard payloads are a converted INT4 container"
                .to_owned(),
        );
    }

    let categories = category_map.into_values().collect();
    let quantization = quant_map
        .into_iter()
        .map(|(bits, stats)| QuantizationStats {
            bits_per_weight: bits,
            matrix_count: stats.matrices,
            logical_parameters: stats.params,
            payload_bytes: stats.payload,
            scale_bytes: stats.scales,
        })
        .collect();
    let file_bytes = index.total_file_bytes();
    let payload_bytes = index.total_payload_bytes();
    let model = ModelSummary {
        model_type: config.model_type.clone(),
        hidden_size: config.hidden_size,
        layers: config.num_hidden_layers,
        dense_layers: config.num_hidden_layers - config.sparse_layer_count(),
        sparse_layers: config.sparse_layer_count(),
        routed_experts_per_layer: config.n_routed_experts,
        active_experts_per_token: config.num_experts_per_tok,
        attention_heads: config.num_attention_heads,
        vocab_size: config.vocab_size,
        max_context: config.max_position_embeddings,
        full_indexer_layers: config.full_indexer_layer_count(),
        index_topk: config.index_topk,
    };
    Ok(CheckpointReport {
        model_path: model_dir.to_path_buf(),
        model,
        shard_count: index.shards().len(),
        tensor_count: index.names().count(),
        file_bytes,
        tensor_payload_bytes: payload_bytes,
        container_overhead_bytes: file_bytes.saturating_sub(payload_bytes),
        known_logical_parameters: known_params,
        estimated_active_parameters_per_token: active_params,
        unknown_packed_bytes,
        categories,
        quantization,
        kv_cache_f32: kv_estimate(config, 4),
        routed_expert_count_found: routed_experts.len(),
        minimum_routed_expert_bytes,
        maximum_routed_expert_bytes,
        required_base_tensor_count: required.len(),
        missing_base_tensor_count: missing_count,
        sidecars_without_weight,
        indexer_weights_complete,
        format_notes,
        warnings,
    })
}

pub(crate) fn kv_estimate(config: &GlmConfig, state_bytes: u8) -> KvEstimate {
    let mla_values = (config.kv_lora_rank + config.qk_rope_head_dim) as u64;
    let layer_values = mla_values * config.num_hidden_layers as u64;
    let index_values = (config.full_indexer_layer_count() * config.index_head_dim) as u64;
    let mla_compressed = layer_values * u64::from(state_bytes);
    let indexer_bytes = index_values * u64::from(state_bytes);
    let compressed = mla_compressed + indexer_bytes;
    let conventional_values = config.num_hidden_layers as u64
        * config.num_attention_heads as u64
        * (config.qk_head_dim + config.v_head_dim) as u64;
    let conventional = conventional_values * u64::from(state_bytes);
    KvEstimate {
        state_bytes,
        mla_values_per_token_per_layer: mla_values,
        mla_compressed_bytes_per_token: mla_compressed,
        configured_indexer_bytes_per_token: indexer_bytes,
        compressed_bytes_per_token: compressed,
        conventional_mha_bytes_per_token: conventional,
        compression_ratio: conventional as f64 / compressed.max(1) as f64,
        bytes_at_4k_context: compressed.saturating_mul(4096),
        bytes_at_32k_context: compressed.saturating_mul(32_768),
        bytes_at_max_context: compressed.saturating_mul(config.max_position_embeddings as u64),
    }
}

fn infer_bits(
    tensor: &TensorInfo,
    sidecar: Option<&TensorInfo>,
    logical: u64,
    config: &GlmConfig,
) -> Option<u8> {
    if sidecar.is_some() {
        if let Some((rows, cols)) = expected_matrix_shape(&tensor.name, config) {
            if tensor.data_len == rows.checked_mul(cols.div_ceil(4))? {
                return Some(2);
            }
            if tensor.data_len == rows.checked_mul(cols.div_ceil(2))? {
                return Some(4);
            }
            if tensor.data_len == rows.checked_mul(cols)? {
                return Some(8);
            }
        }
    }
    match tensor.dtype {
        DType::F64 | DType::U64 | DType::I64 => Some(64),
        DType::F32 | DType::U32 | DType::I32 => Some(32),
        DType::F16 | DType::Bf16 | DType::U16 | DType::I16 => Some(16),
        DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E5M2 if tensor.data_len == logical => {
            Some(8)
        }
        _ => None,
    }
}

fn required_base_tensors(config: &GlmConfig) -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".to_owned(),
        "model.norm.weight".to_owned(),
        "lm_head.weight".to_owned(),
    ];
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
        ] {
            names.push(format!("{prefix}.{suffix}"));
        }
        if config.layer_is_sparse(layer) {
            for suffix in [
                "mlp.gate.weight",
                "mlp.gate.e_score_correction_bias",
                "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.up_proj.weight",
                "mlp.shared_experts.down_proj.weight",
            ] {
                names.push(format!("{prefix}.{suffix}"));
            }
            for expert in 0..config.n_routed_experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!("{prefix}.mlp.experts.{expert}.{projection}.weight"));
                }
            }
        } else {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                names.push(format!("{prefix}.mlp.{projection}.weight"));
            }
        }
    }
    names
}

fn expert_projection_key(name: &str) -> Option<((usize, usize), u8)> {
    let layer_tail = name.strip_prefix("model.layers.")?;
    let (layer, rest) = layer_tail.split_once('.')?;
    let expert_tail = rest.strip_prefix("mlp.experts.")?;
    let (expert, projection) = expert_tail.split_once('.')?;
    let bit = match projection {
        "gate_proj.weight" => 0b001,
        "up_proj.weight" => 0b010,
        "down_proj.weight" => 0b100,
        _ => return None,
    };
    Some(((layer.parse().ok()?, expert.parse().ok()?), bit))
}

fn has_tokenizer(model_dir: &Path) -> bool {
    [
        "tokenizer.json",
        "tokenizer.model",
        "tiktoken.model",
        "tokenizer_config.json",
    ]
    .iter()
    .any(|name| model_dir.join(name).is_file())
}

fn has_kimi_k3_tokenizer(model_dir: &Path) -> bool {
    model_dir.join("tiktoken.model").is_file() && model_dir.join("tokenizer_config.json").is_file()
}

fn indexer_weights_present(config: &GlmConfig, index: &TensorIndex) -> bool {
    if config.full_indexer_layer_count() == 0 {
        return true;
    }
    (0..config.num_hidden_layers)
        .filter(|&layer| config.layer_has_full_indexer(layer))
        .all(|layer| {
            [
                "wq_b.weight",
                "wk.weight",
                "weights_proj.weight",
                "k_norm.weight",
                "k_norm.bias",
            ]
            .iter()
            .all(|suffix| {
                index
                    .get(&format!("model.layers.{layer}.self_attn.indexer.{suffix}"))
                    .is_some()
            })
        })
}

fn validate_known_tensor_schema(
    tensor: &TensorInfo,
    sidecar: Option<&TensorInfo>,
    config: &GlmConfig,
) -> Result<(), String> {
    let Some(logical_elements) = expected_logical_elements(&tensor.name, config) else {
        return Ok(());
    };
    if let Some((rows, cols)) = expected_matrix_shape(&tensor.name, config) {
        if tensor.dtype == DType::U8 {
            let sidecar = sidecar.ok_or_else(|| {
                "packed U8 matrix is missing its required F32 .qs sidecar".to_owned()
            })?;
            if sidecar.dtype != DType::F32 {
                return Err(format!(".qs sidecar must be F32, got {}", sidecar.dtype));
            }
            let bits = infer_unique_packed_bits(rows, cols, tensor.data_len, &[2, 4, 8])?;
            validate_quantized_scale_layout(bits, rows, cols, sidecar)?;
            return Ok(());
        }

        if sidecar.is_some() {
            return Err("non-U8 matrix unexpectedly has a .qs sidecar".to_owned());
        }
        if !matches!(
            tensor.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64 | DType::F8E4M3 | DType::F8E5M2
        ) {
            return Err(format!(
                "matrix dtype {} is not a supported floating format",
                tensor.dtype
            ));
        }
        if tensor.shape != [rows, cols] {
            return Err(format!(
                "floating matrix must declare shape [{rows}, {cols}], got {:?}",
                tensor.shape
            ));
        }
        return Ok(());
    }

    if sidecar.is_some() {
        return Err("non-matrix tensor unexpectedly has a .qs sidecar".to_owned());
    }
    if !matches!(tensor.dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(format!(
            "vector dtype {} is not supported; expected F16/BF16/F32",
            tensor.dtype
        ));
    }
    if tensor.shape != [logical_elements] {
        return Err(format!(
            "vector must declare shape [{logical_elements}], got {:?}",
            tensor.shape
        ));
    }
    Ok(())
}

fn shard_numbers_are_contiguous(index: &TensorIndex) -> bool {
    let mut numbers = Vec::new();
    for shard in index.shards() {
        let Some(filename) = shard.path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let number = filename
            .strip_prefix("model-")
            .and_then(|value| value.split_once("-of-").map(|(number, _)| number))
            .and_then(|value| value.parse::<usize>().ok())
            .or_else(|| {
                shard
                    .path
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .and_then(|stem| stem.strip_prefix("out-"))
                    .and_then(|value| value.parse::<usize>().ok())
            });
        let Some(number) = number else {
            continue;
        };
        numbers.push(number);
    }
    if numbers.is_empty() {
        return true;
    }
    numbers.sort_unstable();
    numbers.windows(2).all(|pair| pair[1] == pair[0] + 1)
}

fn add(left: u64, right: u64, what: &str) -> Result<u64, AnalysisError> {
    left.checked_add(right)
        .ok_or_else(|| AnalysisError::Arithmetic(format!("{what} overflows u64")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GlmConfig {
        GlmConfig::from_json_str(
            &serde_json::json!({
                "model_type":"glm_moe_dsa", "hidden_size":8, "num_hidden_layers":2,
                "num_attention_heads":2, "vocab_size":16, "intermediate_size":12,
                "moe_intermediate_size":4, "first_k_dense_replace":1,
                "n_routed_experts":8, "n_shared_experts":1, "num_experts_per_tok":2,
                "q_lora_rank":4, "kv_lora_rank":3, "qk_nope_head_dim":2,
                "qk_rope_head_dim":2, "qk_head_dim":4, "v_head_dim":3,
                "max_position_embeddings":64, "hidden_act":"silu",
                "scoring_func":"sigmoid", "topk_method":"noaux_tc"
            })
            .to_string(),
        )
        .unwrap()
    }

    fn tensor(name: &str, dtype: DType, shape: Vec<u64>, data_len: u64) -> TensorInfo {
        let declared_elements = shape.iter().product();
        TensorInfo {
            name: name.to_owned(),
            dtype,
            shape,
            shard: PathBuf::from("fixture.safetensors"),
            data_offset: 0,
            data_len,
            declared_elements,
        }
    }

    #[test]
    fn packed_known_matrix_requires_a_valid_scale_sidecar() {
        let cfg = config();
        let weight = tensor(
            "model.layers.0.self_attn.q_a_proj.weight",
            DType::U8,
            vec![16],
            16,
        );
        assert!(validate_known_tensor_schema(&weight, None, &cfg)
            .unwrap_err()
            .contains("missing"));

        let valid_scale = tensor("scale", DType::F32, vec![4], 16);
        validate_known_tensor_schema(&weight, Some(&valid_scale), &cfg).unwrap();

        let invalid_scale = tensor("scale", DType::F32, vec![5], 20);
        assert!(
            validate_known_tensor_schema(&weight, Some(&invalid_scale), &cfg)
                .unwrap_err()
                .contains("scale count")
        );

        let transposed_scale = tensor("scale", DType::F32, vec![1, 4], 16);
        assert!(
            validate_known_tensor_schema(&weight, Some(&transposed_scale), &cfg)
                .unwrap_err()
                .contains("canonical")
        );
    }

    #[test]
    fn expert_count_key_requires_one_of_the_three_projection_names() {
        assert_eq!(
            expert_projection_key("model.layers.1.mlp.experts.7.up_proj.weight"),
            Some(((1, 7), 0b010))
        );
        assert_eq!(
            expert_projection_key("model.layers.1.mlp.experts.7.unknown.weight"),
            None
        );
    }
}
