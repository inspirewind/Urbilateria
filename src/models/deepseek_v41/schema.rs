//! Header-only native checkpoint schema and memory accounting for DeepSeek-V4.1-Flash.

use super::DeepseekV41Config;
use crate::storage::{DType, SafetensorError, TensorIndex};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug)]
pub enum SchemaError {
    Checkpoint(SafetensorError),
    Invalid(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "invalid DeepSeek-V4.1 checkpoint: {reason}"),
        }
    }
}

impl std::error::Error for SchemaError {}

impl From<SafetensorError> for SchemaError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DeepseekV41Requirements {
    pub required_tensor_count: usize,
    pub checkpoint_tensor_count: usize,
    pub checkpoint_shard_count: usize,
    pub checkpoint_file_bytes: u64,
    pub checkpoint_payload_bytes: u64,
    pub backbone_logical_parameters: u64,
    pub engram_logical_parameters: u64,
    pub vision_logical_parameters: u64,
    pub dspark_logical_parameters: u64,
    /// Largest native non-expert layer payload used by the streaming fallback.
    pub streamed_layer_bytes: u64,
    /// Exact native bytes required to retain all 40 CED backbone layers.
    pub backbone_layer_bytes: u64,
    pub backbone_layer_count: usize,
    pub streamed_embedding_bytes: u64,
    pub streamed_lm_head_bytes: u64,
    /// Resident BF16 representation used when the vocabulary projection is pinned.
    pub lm_head_resident_bytes: u64,
    pub engram_table_bytes: u64,
    pub engram_lookup_bytes_per_token: u64,
    pub maximum_expert_bytes: u64,
    pub transient_expert_bytes: u64,
    pub expert_cache_bytes: u64,
    pub expert_slots_per_layer: usize,
    pub global_kv_cache_bytes: u64,
    pub sliding_window_cache_bytes: u64,
    pub kv_cache_bytes: u64,
    pub context_limit: usize,
    pub exact_context_ceiling: usize,
    pub encoder_layer_count: usize,
    pub decoder_layer_count: usize,
    pub kv_source_layer_count: usize,
    pub index_source_layer_count: usize,
    pub approximate_replay_window: usize,
}

impl DeepseekV41Requirements {
    /// Number of leading CED layers that fit above the mandatory one-layer streaming allowance.
    /// Partial residency is conservatively rounded by the largest layer; complete residency uses
    /// the exact sum because Engram and CSA2 ownership make individual layer sizes non-uniform.
    pub fn cached_backbone_layers_for_resident_budget(&self, resident_budget: u64) -> usize {
        let cache_budget = resident_budget.saturating_sub(self.streamed_layer_bytes);
        if cache_budget >= self.backbone_layer_bytes {
            self.backbone_layer_count
        } else {
            cache_budget
                .checked_div(self.streamed_layer_bytes)
                .unwrap_or(0)
                .min(self.backbone_layer_count as u64) as usize
        }
    }

    pub fn caches_lm_head_for_resident_budget(&self, resident_budget: u64) -> bool {
        resident_budget
            >= self
                .streamed_layer_bytes
                .saturating_add(self.backbone_layer_bytes)
                .saturating_add(self.lm_head_resident_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    TokenRows,
    BackboneRoot,
    BackboneLayer(usize),
    Expert(usize, usize),
    EngramTable,
    EngramOther(usize),
    Vision,
    DSpark,
}

#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    dtype: DType,
    shape: Vec<u64>,
    category: Category,
    logical_elements: u64,
}

impl TensorSpec {
    fn new(
        name: impl Into<String>,
        dtype: DType,
        shape: &[usize],
        category: Category,
    ) -> Result<Self, SchemaError> {
        let logical_elements = shape.iter().try_fold(1u64, |total, &dimension| {
            total.checked_mul(dimension as u64).ok_or_else(|| {
                SchemaError::Invalid("tensor logical element count overflows".to_owned())
            })
        })?;
        Ok(Self {
            name: name.into(),
            dtype,
            shape: shape.iter().map(|&value| value as u64).collect(),
            category,
            logical_elements,
        })
    }

    fn packed_fp4(
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        category: Category,
    ) -> Result<Self, SchemaError> {
        let mut spec = Self::new(name, DType::I8, &[rows, cols.div_ceil(2)], category)?;
        spec.logical_elements = (rows as u64).checked_mul(cols as u64).ok_or_else(|| {
            SchemaError::Invalid("FP4 logical element count overflows".to_owned())
        })?;
        Ok(spec)
    }
}

pub fn inspect_requirements(
    config: &DeepseekV41Config,
    index: &TensorIndex,
    context_limit: usize,
    expert_slots_per_layer: usize,
) -> Result<DeepseekV41Requirements, SchemaError> {
    let text = &config.text_config;
    if context_limit == 0 || context_limit > text.max_position_embeddings {
        return Err(SchemaError::Invalid(format!(
            "context_limit={context_limit} is outside 1..={}",
            text.max_position_embeddings
        )));
    }
    if expert_slots_per_layer > text.n_routed_experts {
        return Err(SchemaError::Invalid(format!(
            "expert_slots_per_layer={expert_slots_per_layer} exceeds {}",
            text.n_routed_experts
        )));
    }
    let specs = official_tensor_specs(config)?;
    validate_specs(index, &specs)?;

    let mut layer_bytes = vec![0u64; text.num_hidden_layers];
    let mut expert_bytes = vec![vec![0u64; text.n_routed_experts]; text.num_hidden_layers];
    let mut roots = 0u64;
    let mut backbone_params = 0u64;
    let mut engram_params = 0u64;
    let mut vision_params = 0u64;
    let mut dspark_params = 0u64;
    let mut engram_table_bytes = 0u64;
    for spec in &specs {
        let bytes = index.require(&spec.name)?.data_len;
        match spec.category {
            Category::TokenRows => {
                backbone_params = add(backbone_params, spec.logical_elements, "backbone params")?;
            }
            Category::BackboneRoot => {
                roots = add(roots, bytes, "root bytes")?;
                backbone_params = add(backbone_params, spec.logical_elements, "backbone params")?;
            }
            Category::BackboneLayer(layer) => {
                layer_bytes[layer] = add(layer_bytes[layer], bytes, "layer bytes")?;
                backbone_params = add(backbone_params, spec.logical_elements, "backbone params")?;
            }
            Category::Expert(layer, expert) => {
                expert_bytes[layer][expert] =
                    add(expert_bytes[layer][expert], bytes, "expert bytes")?;
                backbone_params = add(backbone_params, spec.logical_elements, "backbone params")?;
            }
            Category::EngramTable => {
                engram_table_bytes = add(engram_table_bytes, bytes, "Engram table bytes")?;
                engram_params = add(engram_params, spec.logical_elements, "Engram params")?;
            }
            Category::EngramOther(layer) => {
                layer_bytes[layer] = add(layer_bytes[layer], bytes, "Engram projection bytes")?;
                engram_params = add(engram_params, spec.logical_elements, "Engram params")?;
            }
            Category::Vision => {
                vision_params = add(vision_params, spec.logical_elements, "vision params")?;
            }
            Category::DSpark => {
                dspark_params = add(dspark_params, spec.logical_elements, "DSpark params")?;
            }
        }
    }
    let maximum_expert_bytes = expert_bytes.iter().flatten().copied().max().unwrap_or(0);
    if maximum_expert_bytes == 0 {
        return Err(SchemaError::Invalid(
            "routed experts have no payload".to_owned(),
        ));
    }
    let expert_cache_bytes = maximum_expert_bytes
        .checked_mul(expert_slots_per_layer as u64)
        .and_then(|bytes| bytes.checked_mul(text.num_hidden_layers as u64))
        .ok_or_else(|| SchemaError::Invalid("expert cache bytes overflow".to_owned()))?;
    let streamed_layer_bytes = layer_bytes.iter().copied().max().unwrap_or(0);
    let backbone_layer_bytes = layer_bytes.into_iter().try_fold(0u64, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| SchemaError::Invalid("backbone layer bytes overflow".to_owned()))
    })?;
    let global_kv_cache_bytes = global_kv_cache_bytes(config, context_limit)?;
    let sliding_window_cache_bytes = sliding_window_cache_bytes(config)?;
    let kv_cache_bytes = add(
        global_kv_cache_bytes,
        sliding_window_cache_bytes,
        "total KV bytes",
    )?;
    // Each Engram module fetches (max_ngram-1)*heads rows. A row holds one E4M3 byte per
    // channel and one E8M0 byte per group of 32 channels.
    let rows_per_module = (text.engram_max_ngram_size - 1)
        .checked_mul(text.engram_n_heads)
        .ok_or_else(|| SchemaError::Invalid("Engram rows/token overflow".to_owned()))?;
    let bytes_per_row = text
        .engram_head_dim
        .checked_add(text.engram_head_dim.div_ceil(32))
        .ok_or_else(|| SchemaError::Invalid("Engram row bytes overflow".to_owned()))?;
    let engram_lookup_bytes_per_token = (text.engram_layer_ids.len() as u64)
        .checked_mul(rows_per_module as u64)
        .and_then(|rows| rows.checked_mul(bytes_per_row as u64))
        .ok_or_else(|| SchemaError::Invalid("Engram lookup bytes/token overflow".to_owned()))?;

    Ok(DeepseekV41Requirements {
        required_tensor_count: specs.len(),
        checkpoint_tensor_count: index.names().count(),
        checkpoint_shard_count: index.shards().len(),
        checkpoint_file_bytes: index.total_file_bytes(),
        checkpoint_payload_bytes: index.total_payload_bytes(),
        backbone_logical_parameters: backbone_params,
        engram_logical_parameters: engram_params,
        vision_logical_parameters: vision_params,
        dspark_logical_parameters: dspark_params,
        streamed_layer_bytes: streamed_layer_bytes.max(roots),
        backbone_layer_bytes,
        backbone_layer_count: text.num_hidden_layers,
        streamed_embedding_bytes: index.require("embed.weight")?.data_len,
        streamed_lm_head_bytes: index.require("head.weight")?.data_len,
        lm_head_resident_bytes: index.require("head.weight")?.data_len,
        engram_table_bytes,
        engram_lookup_bytes_per_token,
        maximum_expert_bytes,
        transient_expert_bytes: maximum_expert_bytes,
        expert_cache_bytes: expert_cache_bytes.max(maximum_expert_bytes),
        expert_slots_per_layer,
        global_kv_cache_bytes,
        sliding_window_cache_bytes,
        kv_cache_bytes,
        context_limit,
        exact_context_ceiling: text.max_position_embeddings,
        encoder_layer_count: config.encoder_layer_count(),
        decoder_layer_count: config.decoder_layer_count(),
        kv_source_layer_count: text.kv_source_layer_ids.len(),
        index_source_layer_count: text.index_source_layer_ids.len(),
        approximate_replay_window: text.sliding_window,
    })
}

pub fn global_kv_cache_bytes(
    config: &DeepseekV41Config,
    context: usize,
) -> Result<u64, SchemaError> {
    let text = &config.text_config;
    let main_bytes = text.head_dim.div_ceil(2) + text.head_dim.div_ceil(16);
    let index_bytes = text.index_head_dim.div_ceil(2) + text.index_head_dim.div_ceil(32);
    let bytes_per_entry = main_bytes
        .checked_add(index_bytes)
        .ok_or_else(|| SchemaError::Invalid("global KV entry bytes overflow".to_owned()))?;
    text.kv_source_layer_ids
        .iter()
        .try_fold(0u64, |total, &layer| {
            let ratio = text.compress_ratios[layer];
            let entries = context.div_ceil(ratio);
            let bytes = (entries as u64)
                .checked_mul(bytes_per_entry as u64)
                .ok_or_else(|| SchemaError::Invalid("global KV bytes overflow".to_owned()))?;
            add(total, bytes, "global KV bytes")
        })
}

pub fn sliding_window_cache_bytes(config: &DeepseekV41Config) -> Result<u64, SchemaError> {
    let text = &config.text_config;
    let bytes_per_entry = text.head_dim + text.head_dim.div_ceil(32);
    (text.num_hidden_layers as u64)
        .checked_mul(text.sliding_window as u64)
        .and_then(|entries| entries.checked_mul(bytes_per_entry as u64))
        .ok_or_else(|| SchemaError::Invalid("sliding-window cache bytes overflow".to_owned()))
}

fn official_tensor_specs(config: &DeepseekV41Config) -> Result<Vec<TensorSpec>, SchemaError> {
    let text = &config.text_config;
    let vision = &config.vision_config;
    let mut specs = Vec::with_capacity(96_085);
    push(
        &mut specs,
        "embed.weight",
        DType::Bf16,
        &[text.vocab_size, text.hidden_size],
        Category::TokenRows,
    )?;
    push(
        &mut specs,
        "head.weight",
        DType::Bf16,
        &[text.vocab_size, text.hidden_size],
        Category::TokenRows,
    )?;
    push(
        &mut specs,
        "norm.weight",
        DType::Bf16,
        &[text.hidden_size],
        Category::BackboneRoot,
    )?;
    for name in ["image_start", "image_end", "image_newline"] {
        push(
            &mut specs,
            name,
            DType::Bf16,
            &[text.hidden_size],
            Category::Vision,
        )?;
    }
    let unshuffle = vision
        .hidden_size
        .checked_mul(vision.downsample_ratio * vision.downsample_ratio)
        .ok_or_else(|| SchemaError::Invalid("vision unshuffle width overflows".to_owned()))?;
    for (name, shape) in [
        ("aligner.w1.weight", vec![text.hidden_size, unshuffle]),
        ("aligner.w1.bias", vec![text.hidden_size]),
        (
            "aligner.w2.weight",
            vec![text.hidden_size, text.hidden_size],
        ),
        ("aligner.w2.bias", vec![text.hidden_size]),
    ] {
        push(&mut specs, name, DType::Bf16, &shape, Category::Vision)?;
    }

    for layer in 0..text.num_hidden_layers {
        append_block(
            &mut specs,
            config,
            &format!("layers.{layer}"),
            Category::BackboneLayer(layer),
            Some((layer, text.n_routed_experts)),
        )?;
        if text.kv_source_layer_ids.contains(&layer) {
            let prefix = format!("layers.{layer}.attn.compressor");
            push(
                &mut specs,
                format!("{prefix}.norm.weight"),
                DType::Bf16,
                &[text.head_dim],
                Category::BackboneLayer(layer),
            )?;
            push(
                &mut specs,
                format!("{prefix}.wkv.weight"),
                DType::Bf16,
                &[text.head_dim, text.hidden_size],
                Category::BackboneLayer(layer),
            )?;
            if text.compress_ratios[layer] > 1 {
                push(
                    &mut specs,
                    format!("{prefix}.wgate.weight"),
                    DType::Bf16,
                    &[text.head_dim, text.hidden_size],
                    Category::BackboneLayer(layer),
                )?;
            }
        }
        if text.index_source_layer_ids.contains(&layer) {
            let prefix = format!("layers.{layer}.attn.indexer");
            append_mx_matrix(
                &mut specs,
                format!("{prefix}.wq_b"),
                text.index_n_heads * text.index_head_dim,
                text.q_lora_rank,
                Category::BackboneLayer(layer),
            )?;
            push(
                &mut specs,
                format!("{prefix}.weights_proj.weight"),
                DType::Bf16,
                &[text.index_n_heads, text.hidden_size],
                Category::BackboneLayer(layer),
            )?;
            if text.kv_source_layer_ids.contains(&layer) {
                push(
                    &mut specs,
                    format!("{prefix}.wk.weight"),
                    DType::Bf16,
                    &[text.index_head_dim, text.head_dim],
                    Category::BackboneLayer(layer),
                )?;
                push(
                    &mut specs,
                    format!("{prefix}.k_norm.weight"),
                    DType::Bf16,
                    &[text.index_head_dim],
                    Category::BackboneLayer(layer),
                )?;
            }
        }
        if let Some(position) = text.engram_layer_ids.iter().position(|&id| id == layer) {
            append_engram(
                &mut specs,
                config,
                layer,
                text.engram_num_embeddings[position],
            )?;
        }
    }
    append_vision(&mut specs, config)?;

    for stage in 0..text.num_nextn_predict_layers {
        append_block(
            &mut specs,
            config,
            &format!("mtp.{stage}"),
            Category::DSpark,
            Some((usize::MAX, text.dspark_n_routed_experts)),
        )?;
    }
    append_mx_matrix(
        &mut specs,
        "mtp.0.main_proj",
        text.hidden_size,
        text.hidden_size * text.dspark_target_layer_ids.len(),
        Category::DSpark,
    )?;
    push(
        &mut specs,
        "mtp.0.main_norm.weight",
        DType::Bf16,
        &[text.hidden_size],
        Category::DSpark,
    )?;
    let last = text.num_nextn_predict_layers - 1;
    push(
        &mut specs,
        format!("mtp.{last}.norm.weight"),
        DType::Bf16,
        &[text.hidden_size],
        Category::DSpark,
    )?;
    for name in ["embed", "head"] {
        push(
            &mut specs,
            format!("mtp.{last}.markov_head.{name}.weight"),
            DType::Bf16,
            &[text.vocab_size, text.dspark_markov_rank],
            Category::DSpark,
        )?;
    }
    push(
        &mut specs,
        format!("mtp.{last}.confidence_head.proj.weight"),
        DType::Bf16,
        &[1, text.hidden_size + text.dspark_markov_rank],
        Category::DSpark,
    )?;
    Ok(specs)
}

fn append_block(
    specs: &mut Vec<TensorSpec>,
    config: &DeepseekV41Config,
    prefix: &str,
    category: Category,
    routed: Option<(usize, usize)>,
) -> Result<(), SchemaError> {
    let text = &config.text_config;
    let hc_mix = (2 + text.hc_mult) * text.hc_mult;
    let hc_width = text.hc_mult * text.hidden_size;
    for sublayer in ["attn", "ffn"] {
        push(
            specs,
            format!("{prefix}.hc_{sublayer}_fn"),
            DType::F32,
            &[hc_mix, hc_width],
            category,
        )?;
        push(
            specs,
            format!("{prefix}.hc_{sublayer}_base"),
            DType::F32,
            &[hc_mix],
            category,
        )?;
        push(
            specs,
            format!("{prefix}.hc_{sublayer}_scale"),
            DType::F32,
            &[3],
            category,
        )?;
    }
    for name in ["attn_norm", "ffn_norm"] {
        push(
            specs,
            format!("{prefix}.{name}.weight"),
            DType::Bf16,
            &[text.hidden_size],
            category,
        )?;
    }
    push(
        specs,
        format!("{prefix}.attn.q_norm.weight"),
        DType::Bf16,
        &[text.q_lora_rank],
        category,
    )?;
    push(
        specs,
        format!("{prefix}.attn.kv_norm.weight"),
        DType::Bf16,
        &[text.head_dim],
        category,
    )?;
    push(
        specs,
        format!("{prefix}.attn.attn_sink"),
        DType::F32,
        &[text.num_attention_heads],
        category,
    )?;
    append_mx_matrix(
        specs,
        format!("{prefix}.attn.wq_a"),
        text.q_lora_rank,
        text.hidden_size,
        category,
    )?;
    append_mx_matrix(
        specs,
        format!("{prefix}.attn.wq_b"),
        text.num_attention_heads * text.head_dim,
        text.q_lora_rank,
        category,
    )?;
    append_mx_matrix(
        specs,
        format!("{prefix}.attn.wkv"),
        text.head_dim,
        text.hidden_size,
        category,
    )?;
    let group_input = text.num_attention_heads * text.head_dim / text.o_groups;
    append_mx_matrix(
        specs,
        format!("{prefix}.attn.wo_a"),
        text.o_groups * text.o_lora_rank,
        group_input,
        category,
    )?;
    append_mx_matrix(
        specs,
        format!("{prefix}.attn.wo_b"),
        text.hidden_size,
        text.o_groups * text.o_lora_rank,
        category,
    )?;
    let routed_experts = routed.expect("all V4.1 blocks are MoE").1;
    push(
        specs,
        format!("{prefix}.ffn.gate.weight"),
        DType::Bf16,
        &[routed_experts, text.hidden_size],
        category,
    )?;
    for name in ["bias", "bias_vl"] {
        push(
            specs,
            format!("{prefix}.ffn.gate.{name}"),
            DType::F32,
            &[routed_experts],
            category,
        )?;
    }
    for projection in ["w1", "w3"] {
        append_mx_matrix(
            specs,
            format!("{prefix}.ffn.shared_experts.{projection}"),
            text.moe_intermediate_size,
            text.hidden_size,
            category,
        )?;
    }
    append_mx_matrix(
        specs,
        format!("{prefix}.ffn.shared_experts.w2"),
        text.hidden_size,
        text.moe_intermediate_size,
        category,
    )?;
    for expert in 0..routed_experts {
        let expert_category =
            if let Some((layer, _)) = routed.filter(|(layer, _)| *layer != usize::MAX) {
                Category::Expert(layer, expert)
            } else {
                Category::DSpark
            };
        for (projection, rows, cols) in [
            ("w1", text.moe_intermediate_size, text.hidden_size),
            ("w2", text.hidden_size, text.moe_intermediate_size),
            ("w3", text.moe_intermediate_size, text.hidden_size),
        ] {
            let base = format!("{prefix}.ffn.experts.{expert}.{projection}");
            specs.push(TensorSpec::packed_fp4(
                format!("{base}.weight"),
                rows,
                cols,
                expert_category,
            )?);
            push(
                specs,
                format!("{base}.scale"),
                DType::F8E8M0,
                &[rows, cols.div_ceil(32)],
                expert_category,
            )?;
            specs
                .last_mut()
                .expect("just pushed FP4 scale")
                .logical_elements = 0;
        }
    }
    Ok(())
}

fn append_mx_matrix(
    specs: &mut Vec<TensorSpec>,
    base: impl AsRef<str>,
    rows: usize,
    cols: usize,
    category: Category,
) -> Result<(), SchemaError> {
    let base = base.as_ref();
    push(
        specs,
        format!("{base}.weight"),
        DType::F8E4M3,
        &[rows, cols],
        category,
    )?;
    push(
        specs,
        format!("{base}.scale"),
        DType::F8E8M0,
        &[rows.div_ceil(32), cols.div_ceil(32)],
        category,
    )?;
    specs
        .last_mut()
        .expect("just pushed MX scale")
        .logical_elements = 0;
    Ok(())
}

fn append_engram(
    specs: &mut Vec<TensorSpec>,
    config: &DeepseekV41Config,
    layer: usize,
    rows: usize,
) -> Result<(), SchemaError> {
    let text = &config.text_config;
    let prefix = format!("layers.{layer}.engram");
    push(
        specs,
        format!("{prefix}.embed.weight"),
        DType::F8E4M3,
        &[rows, text.engram_head_dim],
        Category::EngramTable,
    )?;
    push(
        specs,
        format!("{prefix}.embed.scale"),
        DType::F8E8M0,
        &[rows, text.engram_head_dim.div_ceil(32)],
        Category::EngramTable,
    )?;
    specs
        .last_mut()
        .expect("just pushed Engram scale")
        .logical_elements = 0;
    for name in ["q_weight", "k_weight"] {
        push(
            specs,
            format!("{prefix}.{name}"),
            DType::Bf16,
            &[text.hc_mult, text.hidden_size],
            Category::EngramOther(layer),
        )?;
    }
    let hash_columns = (text.engram_max_ngram_size - 1) * text.engram_n_heads;
    append_mx_matrix(
        specs,
        format!("{prefix}.wkv"),
        text.hidden_size * (text.hc_mult + 1),
        hash_columns * text.engram_head_dim,
        Category::EngramOther(layer),
    )
}

fn append_vision(
    specs: &mut Vec<TensorSpec>,
    config: &DeepseekV41Config,
) -> Result<(), SchemaError> {
    let vision = &config.vision_config;
    let patch_input = 3 * vision.patch_size * vision.patch_size;
    push(
        specs,
        "vision.patch_embed.proj.weight",
        DType::Bf16,
        &[vision.hidden_size, patch_input],
        Category::Vision,
    )?;
    push(
        specs,
        "vision.patch_embed.proj.bias",
        DType::Bf16,
        &[vision.hidden_size],
        Category::Vision,
    )?;
    for layer in 0..vision.num_hidden_layers {
        let prefix = format!("vision.blocks.{layer}");
        for name in ["norm1", "norm2"] {
            push(
                specs,
                format!("{prefix}.{name}.weight"),
                DType::Bf16,
                &[vision.hidden_size],
                Category::Vision,
            )?;
        }
        push(
            specs,
            format!("{prefix}.attn.wqkv.weight"),
            DType::Bf16,
            &[3 * vision.hidden_size, vision.hidden_size],
            Category::Vision,
        )?;
        push(
            specs,
            format!("{prefix}.attn.wqkv.bias"),
            DType::Bf16,
            &[3 * vision.hidden_size],
            Category::Vision,
        )?;
        push(
            specs,
            format!("{prefix}.attn.wo.weight"),
            DType::Bf16,
            &[vision.hidden_size, vision.hidden_size],
            Category::Vision,
        )?;
        push(
            specs,
            format!("{prefix}.attn.wo.bias"),
            DType::Bf16,
            &[vision.hidden_size],
            Category::Vision,
        )?;
        push(
            specs,
            format!("{prefix}.mlp.w1.weight"),
            DType::Bf16,
            &[2 * vision.intermediate_size, vision.hidden_size],
            Category::Vision,
        )?;
        push(
            specs,
            format!("{prefix}.mlp.w2.weight"),
            DType::Bf16,
            &[vision.hidden_size, vision.intermediate_size],
            Category::Vision,
        )?;
    }
    push(
        specs,
        "vision.norm.weight",
        DType::Bf16,
        &[vision.hidden_size],
        Category::Vision,
    )
}

fn push(
    specs: &mut Vec<TensorSpec>,
    name: impl Into<String>,
    dtype: DType,
    shape: &[usize],
    category: Category,
) -> Result<(), SchemaError> {
    specs.push(TensorSpec::new(name, dtype, shape, category)?);
    Ok(())
}

fn validate_specs(index: &TensorIndex, specs: &[TensorSpec]) -> Result<(), SchemaError> {
    let mut expected = BTreeSet::new();
    for spec in specs {
        if !expected.insert(spec.name.as_str()) {
            return Err(SchemaError::Invalid(format!(
                "internal schema generated tensor {:?} twice",
                spec.name
            )));
        }
        let actual = index.require(&spec.name)?;
        if actual.dtype != spec.dtype || actual.shape != spec.shape {
            return Err(SchemaError::Invalid(format!(
                "tensor {:?} must be {} {:?}, got {} {:?}",
                spec.name, spec.dtype, spec.shape, actual.dtype, actual.shape
            )));
        }
    }
    let unexpected = index
        .names()
        .filter(|name| !expected.contains(*name))
        .count();
    if index.names().count() != specs.len() || unexpected != 0 {
        return Err(SchemaError::Invalid(format!(
            "schema requires {} exact tensors, checkpoint has {} with {unexpected} unexpected",
            specs.len(),
            index.names().count()
        )));
    }
    Ok(())
}

fn add(left: u64, right: u64, label: &str) -> Result<u64, SchemaError> {
    left.checked_add(right)
        .ok_or_else(|| SchemaError::Invalid(format!("{label} overflow")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_manifest_has_exact_tensor_count_and_cache_geometry() {
        let config = DeepseekV41Config::from_json_str(include_str!(
            "../../../tests/fixtures/deepseek_v4_1_flash_config.json"
        ))
        .unwrap();
        let specs = official_tensor_specs(&config).unwrap();
        assert_eq!(specs.len(), 96_085);
        assert_eq!(
            global_kv_cache_bytes(&config, 1_048_576).unwrap(),
            933_232_640
        );
        assert_eq!(sliding_window_cache_bytes(&config).unwrap(), 2_703_360);
    }
}
