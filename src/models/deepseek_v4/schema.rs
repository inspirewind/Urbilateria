//! Header-only tensor schema and memory accounting for DeepSeek-V4.

use super::DeepseekV4Config;
use crate::storage::{inspect_weight_matrix, DType, TensorIndex, WeightFormat, WeightLoadError};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug)]
pub enum SchemaError {
    Weight(WeightLoadError),
    Invalid(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "invalid DeepSeek-V4 checkpoint: {reason}"),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<WeightLoadError> for SchemaError {
    fn from(value: WeightLoadError) -> Self {
        Self::Weight(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DeepseekV4Requirements {
    /// Peak decoded non-expert weights for the adjacent-layer double buffer.
    pub resident_bytes: u64,
    /// Largest decoded non-expert decoder layer.
    pub streamed_layer_bytes: u64,
    /// Exact decoded bytes required to retain every non-expert decoder layer.
    pub decoder_layer_bytes: u64,
    /// Decoded small vectors pinned once for all decoder layers.
    pub decoder_vector_cache_bytes: u64,
    pub decoder_layer_count: usize,
    /// Resident representation used when the vocabulary projection is pinned.
    pub lm_head_resident_bytes: u64,
    pub maximum_expert_bytes: u64,
    pub transient_expert_bytes: u64,
    pub expert_cache_bytes: u64,
    pub kv_cache_bytes: u64,
    pub expert_slots_per_layer: usize,
    pub context_limit: usize,
    pub exact_context_ceiling: usize,
    pub required_tensor_count: usize,
    pub checkpoint_tensor_count: usize,
    pub unexpected_tensor_count: usize,
    pub base_logical_parameters: u64,
    pub dspark_logical_parameters: u64,
    pub base_expert_count: usize,
    pub dspark_stage_count: usize,
    pub indexed_layer_count: usize,
    pub streamed_embedding_bytes: u64,
    pub streamed_lm_head_bytes: u64,
}

impl DeepseekV4Requirements {
    /// Number of leading decoder layers that can be retained above the two-layer streaming
    /// allowance represented by [`Self::resident_bytes`].
    pub fn cached_decoder_layers_for_resident_budget(&self, resident_budget: u64) -> usize {
        let cache_budget = resident_budget.saturating_sub(self.resident_bytes);
        if cache_budget >= self.decoder_layer_bytes {
            self.decoder_layer_count
        } else {
            cache_budget
                .checked_div(self.streamed_layer_bytes)
                .unwrap_or(0)
                .min(self.decoder_layer_count as u64) as usize
        }
    }

    /// Splits the decoded-layer residency allowance between persistent layers and bounded
    /// lookahead. A partially resident model trades up to three leading cached layers for additional
    /// pending layers, keeping the same conservative count of simultaneously resident largest
    /// layers while allowing up to four storage reads to run ahead of decode compute.
    pub fn decoder_layer_pipeline_for_resident_budget(
        &self,
        resident_budget: u64,
    ) -> (usize, usize) {
        let capacity = self.cached_decoder_layers_for_resident_budget(resident_budget);
        decoder_layer_pipeline(capacity, self.decoder_layer_count)
    }

    pub fn caches_lm_head_for_resident_budget(&self, resident_budget: u64) -> bool {
        resident_budget
            >= self
                .resident_bytes
                .saturating_add(self.decoder_layer_bytes)
                .saturating_add(self.lm_head_resident_bytes)
    }
}

fn decoder_layer_pipeline(capacity: usize, layer_count: usize) -> (usize, usize) {
    if capacity < layer_count {
        let prefetch_depth = capacity.saturating_add(1).min(4);
        (capacity + 1 - prefetch_depth, prefetch_depth)
    } else {
        (capacity, 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Residency {
    TokenRows,
    BaseLayer(usize),
    BaseExpert(usize),
    BaseFinal,
    DSpark,
}

struct ManifestBuilder<'a> {
    index: &'a TensorIndex,
    expected: BTreeSet<String>,
    base_layer_bytes: Vec<u64>,
    base_layer_vector_bytes: u64,
    base_layer_expert_slot_bytes: Vec<u64>,
    current_expert_bytes: u64,
    maximum_expert_bytes: u64,
    base_final_bytes: u64,
    base_parameters: u64,
    dspark_parameters: u64,
    streamed_embedding_bytes: u64,
    streamed_lm_head_bytes: u64,
}

impl<'a> ManifestBuilder<'a> {
    fn new(index: &'a TensorIndex, layers: usize) -> Self {
        Self {
            index,
            expected: BTreeSet::new(),
            base_layer_bytes: vec![0; layers],
            base_layer_vector_bytes: 0,
            base_layer_expert_slot_bytes: vec![0; layers],
            current_expert_bytes: 0,
            maximum_expert_bytes: 0,
            base_final_bytes: 0,
            base_parameters: 0,
            dspark_parameters: 0,
            streamed_embedding_bytes: 0,
            streamed_lm_head_bytes: 0,
        }
    }

    fn matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        residency: Residency,
    ) -> Result<u64, SchemaError> {
        let name = name.into();
        let layout = inspect_weight_matrix(self.index, &name, rows, cols)?;
        self.expect(&name)?;
        if matches!(
            layout.format,
            WeightFormat::MxFp8E4M3 { .. } | WeightFormat::MxFp4E2M1 { .. }
        ) {
            self.expect(&native_scale_name(&name))?;
        }
        let parameters = (rows as u64)
            .checked_mul(cols as u64)
            .ok_or_else(|| SchemaError::Invalid(format!("{name:?} parameters overflow")))?;
        self.charge(residency, layout.resident_bytes, parameters)?;
        Ok(layout.resident_bytes)
    }

    fn vector(
        &mut self,
        name: impl Into<String>,
        length: usize,
        dtype: DType,
        residency: Residency,
    ) -> Result<(), SchemaError> {
        let name = name.into();
        let tensor = self.index.require(&name).map_err(WeightLoadError::from)?;
        if tensor.shape != [length as u64] || tensor.dtype != dtype {
            return Err(SchemaError::Invalid(format!(
                "tensor {name:?} must be {dtype} [{length}], got {} {:?}",
                tensor.dtype, tensor.shape
            )));
        }
        self.expect(&name)?;
        let resident = (length as u64)
            .checked_mul(4)
            .ok_or_else(|| SchemaError::Invalid(format!("{name:?} bytes overflow")))?;
        if matches!(residency, Residency::BaseLayer(_)) {
            self.base_layer_vector_bytes = self
                .base_layer_vector_bytes
                .checked_add(resident)
                .ok_or_else(|| {
                    SchemaError::Invalid("decoder vector cache bytes overflow".to_owned())
                })?;
        }
        self.charge(residency, resident, length as u64)
    }

    fn integer_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        residency: Residency,
    ) -> Result<(), SchemaError> {
        let name = name.into();
        let tensor = self.index.require(&name).map_err(WeightLoadError::from)?;
        if tensor.shape != [rows as u64, cols as u64] || tensor.dtype != DType::I64 {
            return Err(SchemaError::Invalid(format!(
                "tensor {name:?} must be I64 [{rows},{cols}], got {} {:?}",
                tensor.dtype, tensor.shape
            )));
        }
        self.expect(&name)?;
        let parameters = (rows as u64)
            .checked_mul(cols as u64)
            .ok_or_else(|| SchemaError::Invalid(format!("{name:?} elements overflow")))?;
        self.charge(residency, 0, parameters)
    }

    fn expect(&mut self, name: &str) -> Result<(), SchemaError> {
        if !self.expected.insert(name.to_owned()) {
            return Err(SchemaError::Invalid(format!(
                "schema generated tensor {name:?} twice"
            )));
        }
        Ok(())
    }

    fn charge(
        &mut self,
        residency: Residency,
        resident: u64,
        parameters: u64,
    ) -> Result<(), SchemaError> {
        let add = |left: &mut u64, right: u64, label: &str| -> Result<(), SchemaError> {
            *left = left
                .checked_add(right)
                .ok_or_else(|| SchemaError::Invalid(format!("{label} overflows u64")))?;
            Ok(())
        };
        match residency {
            Residency::TokenRows => {}
            Residency::BaseLayer(layer) => {
                add(
                    &mut self.base_layer_bytes[layer],
                    resident,
                    "base layer bytes",
                )?;
            }
            Residency::BaseExpert(_) => {
                add(&mut self.current_expert_bytes, resident, "expert bytes")?;
            }
            Residency::BaseFinal => {
                add(&mut self.base_final_bytes, resident, "final weights")?;
            }
            Residency::DSpark => {}
        }
        match residency {
            Residency::DSpark => add(&mut self.dspark_parameters, parameters, "DSpark parameters"),
            _ => add(&mut self.base_parameters, parameters, "base parameters"),
        }
    }

    fn finish_expert(&mut self, layer: usize) -> Result<(), SchemaError> {
        if self.current_expert_bytes == 0 {
            return Err(SchemaError::Invalid(format!(
                "layer {layer} expert has no weights"
            )));
        }
        self.base_layer_expert_slot_bytes[layer] =
            self.base_layer_expert_slot_bytes[layer].max(self.current_expert_bytes);
        self.maximum_expert_bytes = self.maximum_expert_bytes.max(self.current_expert_bytes);
        self.current_expert_bytes = 0;
        Ok(())
    }
}

pub fn inspect_requirements(
    config: &DeepseekV4Config,
    index: &TensorIndex,
    context_limit: usize,
    expert_slots_per_layer: usize,
) -> Result<DeepseekV4Requirements, SchemaError> {
    if context_limit == 0 || context_limit > config.max_position_embeddings {
        return Err(SchemaError::Invalid(format!(
            "context_limit={context_limit} is outside 1..={} ",
            config.max_position_embeddings
        )));
    }
    if expert_slots_per_layer > config.n_routed_experts {
        return Err(SchemaError::Invalid(format!(
            "expert_slots_per_layer={expert_slots_per_layer} exceeds {}",
            config.n_routed_experts
        )));
    }
    let mut manifest = ManifestBuilder::new(index, config.num_hidden_layers);
    manifest.matrix(
        "embed.weight",
        config.vocab_size,
        config.hidden_size,
        Residency::TokenRows,
    )?;
    manifest.streamed_embedding_bytes = index
        .require("embed.weight")
        .map_err(WeightLoadError::from)?
        .data_len;

    for layer in 0..config.num_hidden_layers {
        inspect_block(
            &mut manifest,
            config,
            &format!("layers.{layer}"),
            config.base_compress_ratio(layer).unwrap_or(0),
            layer < config.num_hash_layers,
            Some(layer),
        )?;
    }

    manifest.vector(
        "norm.weight",
        config.hidden_size,
        DType::Bf16,
        Residency::BaseFinal,
    )?;
    let lm_head_layout_bytes = manifest.matrix(
        "head.weight",
        config.vocab_size,
        config.hidden_size,
        Residency::TokenRows,
    )?;
    let lm_head = index
        .require("head.weight")
        .map_err(WeightLoadError::from)?;
    manifest.streamed_lm_head_bytes = lm_head.data_len;
    let lm_head_resident_bytes = if lm_head.dtype == DType::Bf16 {
        lm_head.data_len
    } else {
        lm_head_layout_bytes
    };
    inspect_hc_head(&mut manifest, config, "", Residency::BaseFinal)?;

    let dspark_stages = config.declared_dspark_stage_count();
    for stage in 0..dspark_stages {
        inspect_block(
            &mut manifest,
            config,
            &format!("mtp.{stage}"),
            0,
            false,
            None,
        )?;
    }
    if dspark_stages > 0 {
        manifest.matrix(
            "mtp.0.main_proj.weight",
            config.hidden_size,
            config.hidden_size * config.dspark_target_layer_ids.len(),
            Residency::DSpark,
        )?;
        manifest.vector(
            "mtp.0.main_norm.weight",
            config.hidden_size,
            DType::Bf16,
            Residency::DSpark,
        )?;
        let last = dspark_stages - 1;
        let prefix = format!("mtp.{last}");
        inspect_hc_head(&mut manifest, config, &prefix, Residency::DSpark)?;
        manifest.vector(
            format!("{prefix}.norm.weight"),
            config.hidden_size,
            DType::Bf16,
            Residency::DSpark,
        )?;
        manifest.matrix(
            format!("{prefix}.markov_head.markov_w1.weight"),
            config.vocab_size,
            config.dspark_markov_rank,
            Residency::DSpark,
        )?;
        manifest.matrix(
            format!("{prefix}.markov_head.markov_w2.weight"),
            config.vocab_size,
            config.dspark_markov_rank,
            Residency::DSpark,
        )?;
        manifest.matrix(
            format!("{prefix}.confidence_head.proj.weight"),
            1,
            config.hidden_size + config.dspark_markov_rank,
            Residency::DSpark,
        )?;
    }

    let unexpected_tensor_count = index
        .names()
        .filter(|name| !manifest.expected.contains(*name))
        .count();
    let persistent_expert_bytes = manifest
        .base_layer_expert_slot_bytes
        .iter()
        .try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
        .and_then(|bytes| bytes.checked_mul(expert_slots_per_layer as u64))
        .ok_or_else(|| SchemaError::Invalid("expert cache bytes overflow".to_owned()))?;
    // Decode layer N+1 while layer N executes. Charge the largest adjacent pair rather than two
    // copies of the largest layer: compression ratios make layer sizes non-uniform, and only
    // adjacent layers can coexist in the pipeline.
    let streamed_layer_bytes = manifest.base_layer_bytes.iter().copied().max().unwrap_or(0);
    let decoder_layer_bytes = manifest
        .base_layer_bytes
        .iter()
        .try_fold(0u64, |sum, &bytes| {
            sum.checked_add(bytes)
                .ok_or_else(|| SchemaError::Invalid("decoder layer bytes overflow".to_owned()))
        })?;
    let peak_streamed_layers = match manifest.base_layer_bytes.as_slice() {
        [] => 0,
        [only] => *only,
        layers => layers
            .windows(2)
            .map(|pair| pair[0].saturating_add(pair[1]))
            .max()
            .unwrap_or(0),
    };
    // Runtime decoder layers share one immutable Arc-backed copy of every small vector. Keep the
    // existing per-layer accounting as a conservative pipeline/cache allowance and add the exact
    // all-layer cache explicitly; the double charge is under 2 MiB for the release and prevents
    // this optimization from weakening any previous RAM guarantee.
    let decoder_vector_cache_bytes = manifest.base_layer_vector_bytes;
    let resident_bytes = peak_streamed_layers
        .checked_add(manifest.base_final_bytes)
        .and_then(|bytes| bytes.checked_add(decoder_vector_cache_bytes))
        .ok_or_else(|| {
            SchemaError::Invalid("resident layer pipeline + root weights overflow".to_owned())
        })?;
    let maximum_expert_bytes = manifest.maximum_expert_bytes;
    Ok(DeepseekV4Requirements {
        resident_bytes,
        streamed_layer_bytes,
        decoder_layer_bytes,
        decoder_vector_cache_bytes,
        decoder_layer_count: config.num_hidden_layers,
        lm_head_resident_bytes,
        maximum_expert_bytes,
        transient_expert_bytes: maximum_expert_bytes,
        expert_cache_bytes: persistent_expert_bytes.max(maximum_expert_bytes),
        kv_cache_bytes: kv_cache_bytes_for_context(config, context_limit)?,
        expert_slots_per_layer,
        context_limit,
        exact_context_ceiling: config.max_position_embeddings,
        required_tensor_count: manifest.expected.len(),
        checkpoint_tensor_count: index.names().count(),
        unexpected_tensor_count,
        base_logical_parameters: manifest.base_parameters,
        dspark_logical_parameters: manifest.dspark_parameters,
        base_expert_count: config.num_hidden_layers * config.n_routed_experts,
        dspark_stage_count: dspark_stages,
        indexed_layer_count: config.indexed_layer_count(),
        streamed_embedding_bytes: manifest.streamed_embedding_bytes,
        streamed_lm_head_bytes: manifest.streamed_lm_head_bytes,
    })
}

fn inspect_block(
    manifest: &mut ManifestBuilder<'_>,
    config: &DeepseekV4Config,
    prefix: &str,
    compress_ratio: usize,
    hash_router: bool,
    base_layer: Option<usize>,
) -> Result<(), SchemaError> {
    let residency = base_layer
        .map(Residency::BaseLayer)
        .unwrap_or(Residency::DSpark);
    let h = config.hidden_size;
    let hc = config.hc_mult;
    let mix_hc = (2 + hc) * hc;
    for sublayer in ["attn", "ffn"] {
        manifest.matrix(
            format!("{prefix}.hc_{sublayer}_fn"),
            mix_hc,
            hc * h,
            residency,
        )?;
        manifest.vector(
            format!("{prefix}.hc_{sublayer}_base"),
            mix_hc,
            DType::F32,
            residency,
        )?;
        manifest.vector(
            format!("{prefix}.hc_{sublayer}_scale"),
            3,
            DType::F32,
            residency,
        )?;
    }
    manifest.vector(
        format!("{prefix}.attn_norm.weight"),
        h,
        DType::Bf16,
        residency,
    )?;
    manifest.vector(
        format!("{prefix}.ffn_norm.weight"),
        h,
        DType::Bf16,
        residency,
    )?;
    let attention = format!("{prefix}.attn");
    manifest.vector(
        format!("{attention}.attn_sink"),
        config.num_attention_heads,
        DType::F32,
        residency,
    )?;
    manifest.matrix(
        format!("{attention}.wq_a.weight"),
        config.q_lora_rank,
        h,
        residency,
    )?;
    manifest.vector(
        format!("{attention}.q_norm.weight"),
        config.q_lora_rank,
        DType::Bf16,
        residency,
    )?;
    manifest.matrix(
        format!("{attention}.wq_b.weight"),
        config.num_attention_heads * config.head_dim,
        config.q_lora_rank,
        residency,
    )?;
    manifest.matrix(
        format!("{attention}.wkv.weight"),
        config.head_dim,
        h,
        residency,
    )?;
    manifest.vector(
        format!("{attention}.kv_norm.weight"),
        config.head_dim,
        DType::Bf16,
        residency,
    )?;
    manifest.matrix(
        format!("{attention}.wo_a.weight"),
        config.o_groups * config.o_lora_rank,
        config.num_attention_heads * config.head_dim / config.o_groups,
        residency,
    )?;
    manifest.matrix(
        format!("{attention}.wo_b.weight"),
        h,
        config.o_groups * config.o_lora_rank,
        residency,
    )?;
    if compress_ratio > 0 {
        inspect_compressor(
            manifest,
            config,
            &format!("{attention}.compressor"),
            compress_ratio,
            config.head_dim,
            residency,
        )?;
        if compress_ratio == 4 {
            let indexer = format!("{attention}.indexer");
            manifest.matrix(
                format!("{indexer}.wq_b.weight"),
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
                residency,
            )?;
            manifest.matrix(
                format!("{indexer}.weights_proj.weight"),
                config.index_n_heads,
                h,
                residency,
            )?;
            inspect_compressor(
                manifest,
                config,
                &format!("{indexer}.compressor"),
                compress_ratio,
                config.index_head_dim,
                residency,
            )?;
        }
    }

    let ffn = format!("{prefix}.ffn");
    manifest.matrix(
        format!("{ffn}.gate.weight"),
        config.n_routed_experts,
        h,
        residency,
    )?;
    if hash_router {
        manifest.integer_matrix(
            format!("{ffn}.gate.tid2eid"),
            config.vocab_size,
            config.num_experts_per_tok,
            residency,
        )?;
    } else {
        manifest.vector(
            format!("{ffn}.gate.bias"),
            config.n_routed_experts,
            DType::F32,
            residency,
        )?;
    }
    inspect_mlp(
        manifest,
        &format!("{ffn}.shared_experts"),
        h,
        config.moe_intermediate_size,
        residency,
    )?;
    for expert in 0..config.n_routed_experts {
        let expert_residency = base_layer
            .map(Residency::BaseExpert)
            .unwrap_or(Residency::DSpark);
        inspect_mlp(
            manifest,
            &format!("{ffn}.experts.{expert}"),
            h,
            config.moe_intermediate_size,
            expert_residency,
        )?;
        if let Some(layer) = base_layer {
            manifest.finish_expert(layer)?;
        }
    }
    Ok(())
}

fn inspect_mlp(
    manifest: &mut ManifestBuilder<'_>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    residency: Residency,
) -> Result<(), SchemaError> {
    manifest.matrix(
        format!("{prefix}.w1.weight"),
        intermediate,
        hidden,
        residency,
    )?;
    manifest.matrix(
        format!("{prefix}.w3.weight"),
        intermediate,
        hidden,
        residency,
    )?;
    manifest.matrix(
        format!("{prefix}.w2.weight"),
        hidden,
        intermediate,
        residency,
    )?;
    Ok(())
}

fn inspect_compressor(
    manifest: &mut ManifestBuilder<'_>,
    config: &DeepseekV4Config,
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    residency: Residency,
) -> Result<(), SchemaError> {
    let coefficient = if ratio == 4 { 2 } else { 1 };
    manifest.matrix(
        format!("{prefix}.ape"),
        ratio,
        coefficient * head_dim,
        residency,
    )?;
    for projection in ["wkv", "wgate"] {
        manifest.matrix(
            format!("{prefix}.{projection}.weight"),
            coefficient * head_dim,
            config.hidden_size,
            residency,
        )?;
    }
    manifest.vector(
        format!("{prefix}.norm.weight"),
        head_dim,
        DType::Bf16,
        residency,
    )?;
    Ok(())
}

fn inspect_hc_head(
    manifest: &mut ManifestBuilder<'_>,
    config: &DeepseekV4Config,
    prefix: &str,
    residency: Residency,
) -> Result<(), SchemaError> {
    let separator = if prefix.is_empty() { "" } else { "." };
    manifest.matrix(
        format!("{prefix}{separator}hc_head_fn"),
        config.hc_mult,
        config.hc_mult * config.hidden_size,
        residency,
    )?;
    manifest.vector(
        format!("{prefix}{separator}hc_head_base"),
        config.hc_mult,
        DType::F32,
        residency,
    )?;
    manifest.vector(
        format!("{prefix}{separator}hc_head_scale"),
        1,
        DType::F32,
        residency,
    )?;
    Ok(())
}

pub(crate) fn kv_cache_bytes_for_context(
    config: &DeepseekV4Config,
    context: usize,
) -> Result<u64, SchemaError> {
    let mut values = 0u64;
    for layer in 0..config.num_hidden_layers {
        let ratio = config.base_compress_ratio(layer).unwrap_or(0);
        let window = context.min(config.sliding_window) as u64;
        values = values
            .checked_add(window.saturating_mul(config.head_dim as u64))
            .ok_or_else(|| SchemaError::Invalid("KV cache values overflow".to_owned()))?;
        if ratio > 0 {
            values = values
                .checked_add((context / ratio) as u64 * config.head_dim as u64)
                .ok_or_else(|| SchemaError::Invalid("compressed KV values overflow".to_owned()))?;
            let coefficient = if ratio == 4 { 2 } else { 1 };
            let compressor_state =
                2u64 * (coefficient * ratio) as u64 * (coefficient * config.head_dim) as u64;
            values = values
                .checked_add(compressor_state)
                .ok_or_else(|| SchemaError::Invalid("compressor state overflows".to_owned()))?;
            if ratio == 4 {
                values = values
                    .checked_add((context / ratio) as u64 * config.index_head_dim as u64)
                    .and_then(|value| {
                        value.checked_add(
                            2u64 * (2 * ratio) as u64 * (2 * config.index_head_dim) as u64,
                        )
                    })
                    .ok_or_else(|| SchemaError::Invalid("indexer state overflows".to_owned()))?;
            }
        }
    }
    let hidden = (config.hc_mult * config.hidden_size) as u64;
    values = values
        .checked_add(hidden)
        .ok_or_else(|| SchemaError::Invalid("hidden state values overflow".to_owned()))?;
    values
        .checked_mul(4)
        .ok_or_else(|| SchemaError::Invalid("KV cache bytes overflow".to_owned()))
}

fn native_scale_name(name: &str) -> String {
    name.strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.scale"))
        .unwrap_or_else(|| format!("{name}.scale"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_release_tensor_count_is_stable() {
        let config = DeepseekV4Config::from_json_str(include_str!(
            "../../../tests/fixtures/deepseek_v4_flash_0731_config.json"
        ))
        .unwrap();
        let base_without_compressors = 6 + config.num_hidden_layers * 1565;
        let compressors = 21 * 11 + 20 * 4;
        let dspark = 3 * 1565 + 10;
        assert_eq!(base_without_compressors + compressors + dspark, 72_317);
    }

    #[test]
    fn partial_layer_residency_trades_cache_slots_for_four_layer_lookahead() {
        assert_eq!(decoder_layer_pipeline(0, 43), (0, 1));
        assert_eq!(decoder_layer_pipeline(1, 43), (0, 2));
        assert_eq!(decoder_layer_pipeline(2, 43), (0, 3));
        assert_eq!(decoder_layer_pipeline(3, 43), (0, 4));
        assert_eq!(decoder_layer_pipeline(17, 43), (14, 4));
        assert_eq!(decoder_layer_pipeline(43, 43), (43, 1));
    }
}
