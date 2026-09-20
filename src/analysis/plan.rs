use super::classify::ParameterCategory;
use super::report::{kv_estimate, CheckpointReport};
use crate::config::{DeepseekV41Config, DeepseekV4Config, GlmConfig, Hy4Config, KimiK3Config};
use crate::models::deepseek_v4::schema::{self, DeepseekV4Requirements};
use crate::models::deepseek_v41::schema::{self as deepseek_v41_schema, DeepseekV41Requirements};
use crate::models::hy4::schema::Hy4Requirements;
use serde::Serialize;

const MIB: u64 = 1024 * 1024;

/// Plans the V4.1 layer-streamed runtime using its trained mixed-precision cache layout. The
/// `kv_state_bytes` field is zero because main KV, indexer K, and SWA use distinct packed formats.
pub fn build_deepseek_v41_resource_plan(
    config: &DeepseekV41Config,
    requirements: &DeepseekV41Requirements,
    ram_budget_bytes: u64,
    context_tokens: u64,
) -> ResourcePlan {
    let text = &config.text_config;
    let sparse_layers = text.num_hidden_layers as u64;
    let expert_count = sparse_layers.saturating_mul(text.n_routed_experts as u64);
    let maximum_expert = requirements.maximum_expert_bytes;
    let all_experts = maximum_expert.saturating_mul(expert_count);
    let resident_core = requirements.streamed_layer_bytes;
    let safety = (ram_budget_bytes / 10).max(512 * MIB);
    // This is the initial bounded chunk arena. Runtime construction will derive its prompt chunk
    // size from this allowance rather than retaining four F32 hidden streams for the full prompt.
    let scratch = 512 * MIB;
    let context_usize = usize::try_from(context_tokens).unwrap_or(usize::MAX);
    let kv_bytes = if context_usize == usize::MAX {
        u64::MAX
    } else {
        deepseek_v41_schema::global_kv_cache_bytes(config, context_usize)
            .and_then(|global| {
                deepseek_v41_schema::sliding_window_cache_bytes(config).and_then(|swa| {
                    global.checked_add(swa).ok_or_else(|| {
                        deepseek_v41_schema::SchemaError::Invalid(
                            "V4.1 total KV bytes overflow".to_owned(),
                        )
                    })
                })
            })
            .unwrap_or(u64::MAX)
    };
    let fixed = safety
        .saturating_add(resident_core)
        .saturating_add(kv_bytes)
        .saturating_add(scratch);
    let raw_expert_budget = ram_budget_bytes.saturating_sub(fixed);
    let expert_budget = raw_expert_budget.min(all_experts);
    let slots_total = expert_budget
        .checked_div(maximum_expert)
        .unwrap_or(0)
        .min(expert_count);
    let slots_per_layer = slots_total.checked_div(sparse_layers).unwrap_or(0);
    let unused_after_full = raw_expert_budget.saturating_sub(expert_budget);
    let before_kv = ram_budget_bytes
        .saturating_sub(safety)
        .saturating_sub(resident_core)
        .saturating_sub(scratch)
        .saturating_sub(maximum_expert);
    let mut low = 0usize;
    let mut high = text.max_position_embeddings;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let fits = deepseek_v41_schema::global_kv_cache_bytes(config, middle)
            .and_then(|global| {
                deepseek_v41_schema::sliding_window_cache_bytes(config).map(|swa| global + swa)
            })
            .is_ok_and(|bytes| bytes <= before_kv);
        if fits {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let maximum_context_under_budget = low as u64;
    let within_context = context_tokens > 0
        && context_tokens <= text.max_position_embeddings as u64
        && context_usize != usize::MAX;
    let transient_fits = raw_expert_budget >= maximum_expert;
    let feasible = ram_budget_bytes >= fixed
        && transient_fits
        && within_context
        && context_tokens <= maximum_context_under_budget;
    let mut notes = vec![
        "V4.1 keeps native 32x32 MXFP8 trunk weights, packed E2M1 experts, E2M1/E4M3 global KV, and E4M3/E8M0 SWA KV".to_owned(),
        format!(
            "Engram's {} table payload remains on storage; the base transient lookup is {} bytes/token before I/O alignment or row caching",
            requirements.engram_table_bytes, requirements.engram_lookup_bytes_per_token
        ),
        "prefill scratch is a bounded chunk arena; the runtime must not retain four F32 residual streams for the full prompt".to_owned(),
        "the exact path executes all 40 layers; the 128-token decoder SWA bounded replay remains a separate approximate mode".to_owned(),
        "vision and DSpark weights are schema-validated but excluded from the initial text base-runtime working set".to_owned(),
    ];
    if ram_budget_bytes < fixed {
        notes.push(
            "RAM budget is smaller than streamed layer + packed KV + scratch + safety reserve"
                .to_owned(),
        );
    }
    if !transient_fits {
        notes.push(format!(
            "expert working set must fit one transient expert ({maximum_expert} bytes)"
        ));
    }
    if slots_per_layer < text.num_experts_per_tok as u64 {
        notes.push("expert cache holds fewer entries per layer than one token activates; storage reads will dominate decode".to_owned());
    }
    if context_tokens > maximum_context_under_budget {
        notes.push(format!(
            "requested context {context_tokens} exceeds conservative packed-cache limit {maximum_context_under_budget}"
        ));
    }
    ResourcePlan {
        ram_budget_bytes,
        safety_reserve_bytes: safety,
        resident_core_bytes: resident_core,
        kv_cache_bytes: kv_bytes,
        scratch_bytes: scratch,
        expert_cache_budget_bytes: expert_budget,
        unused_ram_after_full_expert_residency: unused_after_full,
        context_tokens,
        kv_state_bytes: 0,
        kv_includes_configured_indexer: true,
        maximum_context_under_budget,
        average_expert_bytes: maximum_expert,
        maximum_expert_bytes: maximum_expert,
        transient_expert_bytes: maximum_expert,
        expert_slots_total: slots_total,
        expert_slots_per_sparse_layer: slots_per_layer,
        expert_capacity_fraction: (slots_per_layer as f64 / text.n_routed_experts as f64).min(1.0),
        cold_routed_bytes_per_token: maximum_expert
            .saturating_mul(text.num_experts_per_tok as u64)
            .saturating_mul(sparse_layers),
        feasible,
        notes,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourcePlan {
    pub ram_budget_bytes: u64,
    pub safety_reserve_bytes: u64,
    pub resident_core_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub expert_cache_budget_bytes: u64,
    pub unused_ram_after_full_expert_residency: u64,
    pub context_tokens: u64,
    pub kv_state_bytes: u8,
    pub kv_includes_configured_indexer: bool,
    pub maximum_context_under_budget: u64,
    pub average_expert_bytes: u64,
    pub maximum_expert_bytes: u64,
    pub transient_expert_bytes: u64,
    pub expert_slots_total: u64,
    pub expert_slots_per_sparse_layer: u64,
    pub expert_capacity_fraction: f64,
    pub cold_routed_bytes_per_token: u64,
    pub feasible: bool,
    pub notes: Vec<String>,
}

/// Plans Hy4's layer-cached ModelOpt-MXFP8 runtime.
pub fn build_hy4_resource_plan(
    config: &Hy4Config,
    requirements: &Hy4Requirements,
    ram_budget_bytes: u64,
    context_tokens: u64,
    state_bytes: u8,
) -> ResourcePlan {
    let sparse_layers = config.sparse_layer_count() as u64;
    let expert_count = sparse_layers.saturating_mul(config.n_routed_experts as u64);
    let maximum_expert = requirements.maximum_expert_bytes;
    let all_experts = maximum_expert.saturating_mul(expert_count);
    let transient_expert = maximum_expert;
    let base_resident = requirements.resident_bytes;
    let kv_per_token = config.num_hidden_layers as u64
        * (config.kv_lora_rank + config.qk_rope_head_dim) as u64
        * u64::from(state_bytes);
    let kv_bytes = kv_per_token.saturating_mul(context_tokens);
    // Hy4's layer-cached runtime has bounded expert, KV, and scratch allocations. A 9% reserve
    // keeps several GiB of allocator/OS headroom at practical budgets while allowing spare expert
    // slots below one complete per-layer stripe to reduce storage traffic at 96 GiB.
    // GNU/Linux inference caps glibc at two arenas before worker creation, eliminating the
    // multi-GiB allocator high-water mark measured with routed-expert churn. Keep 1% there while
    // retaining the prior 9% margin on allocators where that high-water bound is unavailable.
    let safety_per_mille = if cfg!(all(target_os = "linux", target_env = "gnu")) {
        10
    } else {
        90
    };
    let safety = (ram_budget_bytes.saturating_mul(safety_per_mille) / 1_000).max(512 * MIB);
    let scratch = 512 * MIB;
    let fixed = safety
        .saturating_add(base_resident)
        .saturating_add(kv_bytes)
        .saturating_add(scratch);
    let working = ram_budget_bytes.saturating_sub(fixed);
    let discretionary = working.saturating_sub(transient_expert);
    let expert_bytes_per_slot = maximum_expert.saturating_mul(sparse_layers);
    let minimum_expert_slots =
        (config.num_experts_per_tok as u64).min(config.n_routed_experts as u64);
    let minimum_expert_cache = expert_bytes_per_slot.saturating_mul(minimum_expert_slots);
    // Exact all-layer residency avoids multiplying the unusually large dense layer by the number
    // of sparse layers. Partial caching remains rounded down by the largest layer so it is safe
    // without carrying a per-layer allocation table into the runtime.
    let available_layer_cache = discretionary.saturating_sub(minimum_expert_cache);
    let layer_cache_bytes = if available_layer_cache >= requirements.decoder_layer_bytes {
        requirements.decoder_layer_bytes
    } else {
        available_layer_cache
            .checked_div(requirements.streamed_layer_bytes)
            .unwrap_or(0)
            .saturating_mul(requirements.streamed_layer_bytes)
    };
    let remaining_after_layers = discretionary.saturating_sub(layer_cache_bytes);
    let lm_head_cache_bytes = if layer_cache_bytes == requirements.decoder_layer_bytes
        && remaining_after_layers >= minimum_expert_cache.saturating_add(requirements.lm_head_bytes)
    {
        requirements.lm_head_bytes
    } else {
        0
    };
    let resident_core = base_resident
        .saturating_add(layer_cache_bytes)
        .saturating_add(lm_head_cache_bytes);
    let expert_working = remaining_after_layers.saturating_sub(lm_head_cache_bytes);
    let all_fit = expert_working >= all_experts;
    let expert_budget = expert_working.min(all_experts);
    let unused_after_full_expert_residency = if all_fit {
        expert_working.saturating_sub(all_experts)
    } else {
        0
    };
    let slot_capacity = expert_budget
        .checked_div(maximum_expert)
        .unwrap_or(0)
        .min(expert_count);
    let slots_per_layer = slot_capacity
        .checked_div(sparse_layers)
        .unwrap_or(0)
        .min(config.n_routed_experts as u64);
    // Keep the complete byte-budget capacity: the runtime distributes any remainder below one
    // full per-layer stripe across sparse layers instead of silently stranding those slots.
    let slots_total = slot_capacity;
    let expert_capacity_fraction = slots_per_layer as f64 / config.n_routed_experts as f64;
    let before_kv = ram_budget_bytes
        .saturating_sub(safety)
        .saturating_sub(base_resident)
        .saturating_sub(scratch)
        .saturating_sub(transient_expert);
    let maximum_context_under_budget = before_kv
        .checked_div(kv_per_token)
        .unwrap_or(0)
        .min(config.exact_dense_context_ceiling() as u64);
    let within_context =
        context_tokens > 0 && context_tokens <= config.exact_dense_context_ceiling() as u64;
    let transient_fits = working >= transient_expert;
    let feasible = ram_budget_bytes >= fixed
        && transient_fits
        && within_context
        && context_tokens <= maximum_context_under_budget;
    let mut notes = vec![
        format!(
            "Hy4 pins {} of {} decoder layers, {} the compact BF16 LM head, and retains a two-layer load/compute allowance; embedding rows remain streamed",
            requirements.cached_decoder_layers_for_resident_budget(resident_core),
            config.num_hidden_layers,
            if requirements.caches_lm_head_for_resident_budget(resident_core) { "caches" } else { "streams" }
        ),
        "ModelOpt MXFP8 remains native with 1x32 E8M0 scale groups; MTP is schema-validated but excluded from the base forward".to_owned(),
        format!(
            "the checkpoint advertises {} positions; exact CPU execution is currently capped at {} tokens by dense Gated-MLA until DSA IndexCache execution is available",
            config.max_position_embeddings,
            config.exact_dense_context_ceiling()
        ),
        "plan excludes OS page cache and allocator fragmentation; keep the safety reserve enabled".to_owned(),
    ];
    if ram_budget_bytes < fixed {
        notes.push("RAM budget is smaller than streamed layer + KV + scratch + safety".to_owned());
    }
    if !transient_fits {
        notes.push(format!(
            "expert working set must fit one transient expert ({transient_expert} bytes)"
        ));
    }
    if context_tokens > maximum_context_under_budget {
        notes.push(format!(
            "requested context {context_tokens} exceeds conservative limit {maximum_context_under_budget}"
        ));
    }
    ResourcePlan {
        ram_budget_bytes,
        safety_reserve_bytes: safety,
        resident_core_bytes: resident_core,
        kv_cache_bytes: kv_bytes,
        scratch_bytes: scratch,
        expert_cache_budget_bytes: expert_budget,
        unused_ram_after_full_expert_residency: unused_after_full_expert_residency,
        context_tokens,
        kv_state_bytes: state_bytes,
        kv_includes_configured_indexer: false,
        maximum_context_under_budget,
        average_expert_bytes: maximum_expert,
        maximum_expert_bytes: maximum_expert,
        transient_expert_bytes: transient_expert,
        expert_slots_total: slots_total,
        expert_slots_per_sparse_layer: slots_per_layer,
        expert_capacity_fraction,
        cold_routed_bytes_per_token: maximum_expert
            .saturating_mul(config.num_experts_per_tok as u64)
            .saturating_mul(sparse_layers),
        feasible,
        notes,
    }
}

pub fn build_resource_plan(
    config: &GlmConfig,
    report: &CheckpointReport,
    ram_budget_bytes: u64,
    context_tokens: u64,
    kv_state_bytes: u8,
) -> ResourcePlan {
    let routed_storage = report
        .category(ParameterCategory::RoutedExpert)
        .map(|stats| stats.total_storage_bytes())
        .unwrap_or(0);
    let mtp_storage = report
        .category(ParameterCategory::Mtp)
        .map(|stats| stats.total_storage_bytes())
        .unwrap_or(0);
    let resident_core = report
        .tensor_payload_bytes
        .saturating_sub(routed_storage)
        .saturating_sub(mtp_storage);
    let sparse_layers = config.sparse_layer_count() as u64;
    let expert_count = sparse_layers.saturating_mul(config.n_routed_experts as u64);
    let average_expert = routed_storage.checked_div(expert_count).unwrap_or(0);
    let maximum_expert = report.maximum_routed_expert_bytes.max(average_expert);
    let transient_expert = if expert_count > 0 { maximum_expert } else { 0 };
    let kv = kv_estimate(config, kv_state_bytes);
    let kv_includes_configured_indexer = report.indexer_weights_complete;
    let kv_bytes_per_token = if kv_includes_configured_indexer {
        kv.compressed_bytes_per_token
    } else {
        kv.mla_compressed_bytes_per_token
    };
    let kv_bytes = kv_bytes_per_token.saturating_mul(context_tokens);
    let safety = (ram_budget_bytes / 10).max(512 * MIB);
    let scratch = 512 * MIB;
    let fixed = safety
        .saturating_add(resident_core)
        .saturating_add(kv_bytes)
        .saturating_add(scratch);
    let raw_expert_budget = ram_budget_bytes.saturating_sub(fixed);
    let expert_budget = raw_expert_budget.min(routed_storage);
    let unused_after_full_residency = raw_expert_budget.saturating_sub(expert_budget);
    let slots_total = raw_expert_budget
        .checked_div(maximum_expert)
        .unwrap_or(0)
        .min(expert_count);
    let slots_per_layer = slots_total.checked_div(sparse_layers).unwrap_or(0);
    let capacity_fraction = if config.n_routed_experts == 0 {
        0.0
    } else {
        (slots_per_layer as f64 / config.n_routed_experts as f64).min(1.0)
    };
    let before_kv = ram_budget_bytes
        .saturating_sub(safety)
        .saturating_sub(resident_core)
        .saturating_sub(scratch)
        .saturating_sub(transient_expert);
    let exact_execution_limit = if !kv_includes_configured_indexer
        && config.full_indexer_layer_count() > 0
        && config.index_topk > 0
    {
        config.index_topk.min(config.max_position_embeddings) as u64
    } else {
        config.max_position_embeddings as u64
    };
    let maximum_context = before_kv
        .checked_div(kv_bytes_per_token)
        .unwrap_or(0)
        .min(exact_execution_limit);
    let cold_routed = maximum_expert
        .saturating_mul(config.num_experts_per_tok as u64)
        .saturating_mul(sparse_layers);
    let within_model_context = context_tokens > 0 && context_tokens <= exact_execution_limit;
    let transient_expert_fits = expert_count == 0 || raw_expert_budget >= transient_expert;
    let feasible = ram_budget_bytes >= fixed
        && transient_expert_fits
        && context_tokens <= maximum_context
        && within_model_context;
    let mut notes = Vec::new();
    if ram_budget_bytes < fixed {
        notes.push(
            "RAM budget is smaller than core + requested KV + scratch + safety reserve".to_owned(),
        );
    }
    if context_tokens > maximum_context {
        notes.push(format!(
            "requested context {context_tokens} exceeds the conservative budget limit {maximum_context}"
        ));
    }
    if context_tokens == 0 {
        notes.push("requested context must contain at least one token".to_owned());
    } else if context_tokens > config.max_position_embeddings as u64 {
        notes.push(format!(
            "requested context {context_tokens} exceeds model max_position_embeddings={}",
            config.max_position_embeddings
        ));
    } else if context_tokens > exact_execution_limit {
        notes.push(format!(
            "requested context {context_tokens} exceeds exact dense-MLA fallback limit {exact_execution_limit}; DSA indexer weights are unavailable"
        ));
    }
    if !transient_expert_fits {
        notes.push(format!(
            "expert working-set budget must fit at least one transient expert ({transient_expert} bytes), even with zero persistent cache slots"
        ));
    }
    if slots_per_layer < config.num_experts_per_tok as u64 {
        notes.push(
            "expert cache holds fewer slots per layer than one token activates; synchronous streaming will dominate"
                .to_owned(),
        );
    }
    notes.push(
        "expert_capacity_fraction is storage coverage, not a predicted cache hit rate; real hit rates require a routing trace"
            .to_owned(),
    );
    notes.push(
        "plan excludes OS page cache and allocator fragmentation; keep the safety reserve enabled"
            .to_owned(),
    );
    if !kv_includes_configured_indexer && config.full_indexer_layer_count() > 0 {
        notes.push(
            "indexer weights are incomplete, so KV capacity uses the runnable dense-MLA fallback and excludes configured indexer-key cache"
                .to_owned(),
        );
    }
    if unused_after_full_residency > 0 {
        notes.push(format!(
            "all routed experts fit; {unused_after_full_residency} bytes remain outside the bounded expert cache"
        ));
    }

    ResourcePlan {
        ram_budget_bytes,
        safety_reserve_bytes: safety,
        resident_core_bytes: resident_core,
        kv_cache_bytes: kv_bytes,
        scratch_bytes: scratch,
        expert_cache_budget_bytes: expert_budget,
        unused_ram_after_full_expert_residency: unused_after_full_residency,
        context_tokens,
        kv_state_bytes,
        kv_includes_configured_indexer,
        maximum_context_under_budget: maximum_context,
        average_expert_bytes: average_expert,
        maximum_expert_bytes: maximum_expert,
        transient_expert_bytes: transient_expert,
        expert_slots_total: slots_total,
        expert_slots_per_sparse_layer: slots_per_layer,
        expert_capacity_fraction: capacity_fraction,
        cold_routed_bytes_per_token: cold_routed,
        feasible,
        notes,
    }
}

/// Plans the memory-adaptive DeepSeek-V4 correctness runtime. Embedding rows remain streamed;
/// decoder layers and the LM head are retained when the requested RAM budget can still preserve
/// one complete routed-expert stripe, otherwise they fall back to bounded streaming.
pub fn build_deepseek_resource_plan(
    config: &DeepseekV4Config,
    report: &CheckpointReport,
    requirements: &DeepseekV4Requirements,
    ram_budget_bytes: u64,
    context_tokens: u64,
    kv_state_bytes: u8,
) -> ResourcePlan {
    let sparse_layers = config.num_hidden_layers as u64;
    let expert_count = sparse_layers.saturating_mul(config.n_routed_experts as u64);
    let routed_storage = report
        .category(ParameterCategory::RoutedExpert)
        .map(|stats| stats.total_storage_bytes())
        .unwrap_or(0);
    let average_expert = routed_storage.checked_div(expert_count).unwrap_or(0);
    let maximum_expert = requirements.maximum_expert_bytes.max(average_expert);
    let routed_resident_upper_bound = maximum_expert.saturating_mul(expert_count);
    let transient_expert = if expert_count > 0 { maximum_expert } else { 0 };
    let base_resident = requirements.resident_bytes;
    let safety = (ram_budget_bytes / 10).max(512 * MIB);
    let scratch = 512 * MIB;
    let context_usize = usize::try_from(context_tokens).unwrap_or(usize::MAX);
    let kv_bytes = deepseek_kv_bytes(config, context_usize, kv_state_bytes).unwrap_or(u64::MAX);
    let fixed = safety
        .saturating_add(base_resident)
        .saturating_add(kv_bytes)
        .saturating_add(scratch);
    let working = ram_budget_bytes.saturating_sub(fixed);
    let discretionary = working.saturating_sub(transient_expert);
    let expert_bytes_per_slot = maximum_expert.saturating_mul(sparse_layers);
    let minimum_expert_slots =
        (config.num_experts_per_tok as u64).min(config.n_routed_experts as u64);
    let minimum_expert_cache = expert_bytes_per_slot.saturating_mul(minimum_expert_slots);
    let available_layer_cache = discretionary.saturating_sub(minimum_expert_cache);
    let layer_cache_bytes = if available_layer_cache >= requirements.decoder_layer_bytes {
        requirements.decoder_layer_bytes
    } else {
        available_layer_cache
            .checked_div(requirements.streamed_layer_bytes)
            .unwrap_or(0)
            .saturating_mul(requirements.streamed_layer_bytes)
    };
    let remaining_after_layers = discretionary.saturating_sub(layer_cache_bytes);
    let lm_head_cache_bytes = if layer_cache_bytes == requirements.decoder_layer_bytes
        && remaining_after_layers
            >= minimum_expert_cache.saturating_add(requirements.lm_head_resident_bytes)
    {
        requirements.lm_head_resident_bytes
    } else {
        0
    };
    let resident_core = base_resident
        .saturating_add(layer_cache_bytes)
        .saturating_add(lm_head_cache_bytes);
    let expert_working = remaining_after_layers.saturating_sub(lm_head_cache_bytes);
    let expert_budget = expert_working.min(routed_resident_upper_bound);
    let unused_after_full_residency = expert_working.saturating_sub(expert_budget);
    let slots_total = expert_budget
        .checked_div(maximum_expert)
        .unwrap_or(0)
        .min(expert_count);
    let slots_per_layer = slots_total.checked_div(sparse_layers).unwrap_or(0);
    let capacity_fraction = (slots_per_layer as f64 / config.n_routed_experts as f64).min(1.0);
    let before_kv = ram_budget_bytes
        .saturating_sub(safety)
        .saturating_sub(base_resident)
        .saturating_sub(scratch)
        .saturating_sub(transient_expert);
    let maximum_context = deepseek_context_under_budget(config, before_kv, kv_state_bytes);
    let cold_routed = maximum_expert
        .saturating_mul(config.num_experts_per_tok as u64)
        .saturating_mul(sparse_layers);
    let within_context = context_tokens > 0
        && context_tokens <= config.max_position_embeddings as u64
        && context_usize != usize::MAX;
    let transient_expert_fits = expert_count == 0 || working >= transient_expert;
    let feasible = ram_budget_bytes >= fixed
        && transient_expert_fits
        && context_tokens <= maximum_context
        && within_context;
    let mut notes = Vec::new();
    if ram_budget_bytes < fixed {
        notes.push(
            "RAM budget is smaller than streamed layer peak + requested KV + scratch + safety reserve"
                .to_owned(),
        );
    }
    if context_tokens == 0 {
        notes.push("requested context must contain at least one token".to_owned());
    } else if context_tokens > config.max_position_embeddings as u64 {
        notes.push(format!(
            "requested context {context_tokens} exceeds model max_position_embeddings={}",
            config.max_position_embeddings
        ));
    }
    if context_tokens > maximum_context {
        notes.push(format!(
            "requested context {context_tokens} exceeds the conservative budget limit {maximum_context}"
        ));
    }
    if !transient_expert_fits {
        notes.push(format!(
            "expert working-set budget must fit at least one transient expert ({transient_expert} bytes)"
        ));
    }
    if slots_per_layer < config.num_experts_per_tok as u64 {
        notes.push(
            "expert cache holds fewer slots per layer than one token activates; synchronous streaming will dominate"
                .to_owned(),
        );
    }
    let (cached_layers, layer_prefetch_depth) =
        requirements.decoder_layer_pipeline_for_resident_budget(resident_core);
    notes.push(format!(
        "DeepSeek-V4 pins {cached_layers} of {} decoder layers, uses {layer_prefetch_depth}-layer read lookahead, {} the LM head, and row-streams embedding rows",
        config.num_hidden_layers,
        if requirements.caches_lm_head_for_resident_budget(resident_core) {
            "caches"
        } else {
            "streams"
        }
    ));
    notes.push(
        "DSpark weights are validated but excluded from the base-model correctness runtime working set"
            .to_owned(),
    );
    notes.push(
        "plan excludes OS page cache and allocator fragmentation; keep the safety reserve enabled"
            .to_owned(),
    );
    if unused_after_full_residency > 0 {
        notes.push(format!(
            "all routed experts fit; {unused_after_full_residency} bytes remain outside the bounded expert cache"
        ));
    }

    ResourcePlan {
        ram_budget_bytes,
        safety_reserve_bytes: safety,
        resident_core_bytes: resident_core,
        kv_cache_bytes: kv_bytes,
        scratch_bytes: scratch,
        expert_cache_budget_bytes: expert_budget,
        unused_ram_after_full_expert_residency: unused_after_full_residency,
        context_tokens,
        kv_state_bytes,
        kv_includes_configured_indexer: true,
        maximum_context_under_budget: maximum_context,
        average_expert_bytes: average_expert,
        maximum_expert_bytes: maximum_expert,
        transient_expert_bytes: transient_expert,
        expert_slots_total: slots_total,
        expert_slots_per_sparse_layer: slots_per_layer,
        expert_capacity_fraction: capacity_fraction,
        cold_routed_bytes_per_token: cold_routed,
        feasible,
        notes,
    }
}

/// Plans a text-only, layer-streamed Kimi-K3 correctness runtime.
///
/// Routed experts remain in native MXFP4 and embedding/LM-head rows are streamed. The resident
/// core is therefore the largest non-expert decoder layer plus all recurrent KDA state, rather
/// than the 1.56 TB checkpoint payload. Vision/projector execution is intentionally excluded.
pub fn build_kimi_k3_resource_plan(
    config: &KimiK3Config,
    report: &CheckpointReport,
    ram_budget_bytes: u64,
    context_tokens: u64,
    state_bytes: u8,
) -> ResourcePlan {
    let text = &config.text_config;
    let sparse_layers = text
        .num_hidden_layers
        .saturating_sub(text.first_k_dense_replace) as u64;
    let expert_count = sparse_layers.saturating_mul(text.num_experts as u64);
    let geometry_expert_bytes = kimi_k3_expert_bytes(
        text.routed_expert_hidden_size as u64,
        text.moe_intermediate_size as u64,
    );
    let maximum_expert = report
        .maximum_routed_expert_bytes
        .max(geometry_expert_bytes);
    let average_expert = geometry_expert_bytes;
    let routed_resident_upper_bound = maximum_expert.saturating_mul(expert_count);
    let transient_expert = if expert_count > 0 { maximum_expert } else { 0 };

    let peak_layer = kimi_k3_peak_layer_bytes(config);
    let kda_state = text.kda_layer_count() as u64
        * text.linear_attn_config.num_heads as u64
        * text.linear_attn_config.head_dim as u64
        * text.linear_attn_config.head_dim as u64
        * u64::from(state_bytes);
    let convolution_history = text.kda_layer_count() as u64
        * 3
        * text.linear_attn_config.num_heads as u64
        * text.linear_attn_config.head_dim as u64
        * text
            .linear_attn_config
            .short_conv_kernel_size
            .saturating_sub(1) as u64
        * u64::from(state_bytes);
    let attn_res_working_set = kimi_k3_attn_res_working_set(
        text.num_hidden_layers,
        text.attn_res_block_size,
        text.hidden_size,
    );
    let resident_core = peak_layer
        .saturating_add(kda_state)
        .saturating_add(convolution_history)
        .saturating_add(attn_res_working_set);

    let kv_bytes_per_token = text.full_attention_layer_count() as u64
        * (text.kv_lora_rank + text.qk_rope_head_dim) as u64
        * u64::from(state_bytes);
    let kv_bytes = kv_bytes_per_token.saturating_mul(context_tokens);
    let safety = (ram_budget_bytes / 10).max(512 * MIB);
    let scratch_base = 512 * MIB;
    // Layer-wise prefill retains one hidden vector and every AttnRes block snapshot for each
    // prompt token. Gated MLA additionally keeps one score/probability per head and source. The
    // fixed base covers the retained safetensors index, transient projections, source stacks,
    // and allocator headroom.
    let (attn_res_snapshots, prefill_scratch_bytes_per_token) = kimi_k3_prefill_scratch_per_token(
        text.num_hidden_layers,
        text.attn_res_block_size,
        text.hidden_size,
        text.num_attention_heads,
    );
    let scratch =
        scratch_base.saturating_add(prefill_scratch_bytes_per_token.saturating_mul(context_tokens));
    let fixed = safety
        .saturating_add(resident_core)
        .saturating_add(kv_bytes)
        .saturating_add(scratch);
    let expert_working_set_budget = ram_budget_bytes.saturating_sub(fixed);
    let all_experts_fit = expert_working_set_budget >= routed_resident_upper_bound;
    // The K3 cache loads a miss completely before evicting its LRU entry, so a partially
    // resident cache needs one expert-sized transient slot outside persistent residency. When all
    // experts fit there are no post-fill misses and the full upper bound needs no extra slot.
    let expert_budget = if all_experts_fit {
        routed_resident_upper_bound
    } else {
        expert_working_set_budget.saturating_sub(transient_expert)
    };
    let unused_after_full_residency = if all_experts_fit {
        expert_working_set_budget.saturating_sub(expert_budget)
    } else {
        0
    };
    let slot_capacity_total = expert_budget
        .checked_div(maximum_expert)
        .unwrap_or(0)
        .min(expert_count);
    let slots_per_layer = slot_capacity_total.checked_div(sparse_layers).unwrap_or(0);
    let slots_total = slots_per_layer.saturating_mul(sparse_layers);
    let unused_slot_capacity = slot_capacity_total.saturating_sub(slots_total);
    let capacity_fraction = if text.num_experts == 0 {
        0.0
    } else {
        (slots_per_layer as f64 / text.num_experts as f64).min(1.0)
    };
    let before_context = ram_budget_bytes
        .saturating_sub(safety)
        .saturating_sub(resident_core)
        .saturating_sub(scratch_base)
        .saturating_sub(transient_expert);
    let context_bytes_per_token =
        kv_bytes_per_token.saturating_add(prefill_scratch_bytes_per_token);
    let maximum_context = before_context
        .checked_div(context_bytes_per_token)
        .unwrap_or(0)
        .min(text.max_position_embeddings as u64);
    let cold_routed = maximum_expert
        .saturating_mul(text.num_experts_per_token as u64)
        .saturating_mul(sparse_layers);
    let within_context =
        context_tokens > 0 && context_tokens <= text.max_position_embeddings as u64;
    let transient_expert_fits =
        expert_count == 0 || all_experts_fit || expert_working_set_budget >= transient_expert;
    let feasible = matches!(state_bytes, 2 | 4)
        && ram_budget_bytes >= fixed
        && transient_expert_fits
        && context_tokens <= maximum_context
        && within_context;

    let mut notes = Vec::new();
    if ram_budget_bytes < fixed {
        notes.push(
            "RAM budget is smaller than streamed layer peak + recurrent KDA state + requested MLA KV + scratch + safety reserve"
                .to_owned(),
        );
    }
    if context_tokens == 0 {
        notes.push("requested context must contain at least one token".to_owned());
    } else if context_tokens > text.max_position_embeddings as u64 {
        notes.push(format!(
            "requested context {context_tokens} exceeds model max_position_embeddings={}",
            text.max_position_embeddings
        ));
    }
    if context_tokens > maximum_context {
        notes.push(format!(
            "requested context {context_tokens} exceeds the conservative budget limit {maximum_context}"
        ));
    }
    if !transient_expert_fits {
        notes.push(format!(
            "expert working-set budget must fit at least one transient expert ({transient_expert} bytes)"
        ));
    } else if !all_experts_fit && expert_count > 0 {
        notes.push(format!(
            "expert cache budget reserves {transient_expert} bytes for transactional load-before-evict misses"
        ));
    }
    if slots_per_layer < text.num_experts_per_token as u64 {
        notes.push(
            "expert cache holds fewer slots per layer than one token activates; synchronous NVMe streaming will dominate"
                .to_owned(),
        );
    }
    if unused_slot_capacity > 0 {
        notes.push(format!(
            "uniform per-layer expert caches leave capacity for {unused_slot_capacity} additional expert slot(s) unassigned"
        ));
    }
    if state_bytes == 2 {
        notes.push(
            "2-byte KDA/MLA state is a capacity estimate only; the scalar correctness path uses F32 state"
                .to_owned(),
        );
    }
    notes.push(format!(
        "resident core includes a {peak_layer}-byte peak compact-BF16 layer, {kda_state} bytes of KDA matrix state, and {convolution_history} bytes of convolution history"
    ));
    notes.push(format!(
        "layer-wise prefill scratch includes {prefill_scratch_bytes_per_token} bytes per requested context token for hidden activations, {attn_res_snapshots} AttnRes snapshots, and MLA head scores"
    ));
    notes.push(
        "the prefill bound assumes each token calls the incremental KDA/MLA step and overwrites its single hidden row in place; allocating a second full [tokens, hidden] attention output is outside this plan"
            .to_owned(),
    );
    notes.push(
        "Kimi-K3 planning assumes layer-streamed non-expert weights, row-streamed embedding/LM head, and native-MXFP4 routed experts"
            .to_owned(),
    );
    notes.push(
        "the plan is text-only: MoonViT and multimodal projector weights are schema-validated but not resident"
            .to_owned(),
    );
    notes.push(
        "plan excludes OS page cache and allocator fragmentation; keep the safety reserve enabled"
            .to_owned(),
    );
    if report.missing_base_tensor_count > 0 {
        notes.push(format!(
            "checkpoint transfer is incomplete ({} tensors missing); geometry-based planning remains available but execution is blocked",
            report.missing_base_tensor_count
        ));
    }
    if unused_after_full_residency > 0 {
        notes.push(format!(
            "all routed experts fit; {unused_after_full_residency} bytes remain outside the bounded expert cache"
        ));
    }

    ResourcePlan {
        ram_budget_bytes,
        safety_reserve_bytes: safety,
        resident_core_bytes: resident_core,
        kv_cache_bytes: kv_bytes,
        scratch_bytes: scratch,
        expert_cache_budget_bytes: expert_budget,
        unused_ram_after_full_expert_residency: unused_after_full_residency,
        context_tokens,
        kv_state_bytes: state_bytes,
        kv_includes_configured_indexer: true,
        maximum_context_under_budget: maximum_context,
        average_expert_bytes: average_expert,
        maximum_expert_bytes: maximum_expert,
        transient_expert_bytes: transient_expert,
        expert_slots_total: slots_total,
        expert_slots_per_sparse_layer: slots_per_layer,
        expert_capacity_fraction: capacity_fraction,
        cold_routed_bytes_per_token: cold_routed,
        feasible,
        notes,
    }
}

fn kimi_k3_expert_bytes(latent: u64, intermediate: u64) -> u64 {
    let logical = latent.saturating_mul(intermediate);
    // Three matrices, each one packed nibble plus one E8M0 byte per 32 values.
    logical
        .saturating_mul(3)
        .saturating_div(2)
        .saturating_add(logical.saturating_mul(3).saturating_div(32))
}

fn kimi_k3_peak_layer_bytes(config: &KimiK3Config) -> u64 {
    let text = &config.text_config;
    let h = text.hidden_size as u64;
    let heads = text.num_attention_heads as u64;
    let kda_heads = text.linear_attn_config.num_heads as u64;
    let d = text.linear_attn_config.head_dim as u64;
    let p = kda_heads.saturating_mul(d);
    // Six BF16 checkpoint vectors are widened to F32 because norms and AttnRes read them
    // elementwise. Their resident cost is therefore 6 * hidden * 4, not their BF16 payload.
    let common = kimi_k3_common_vector_bytes(h);
    let kda = 5u64
        .saturating_mul(p)
        .saturating_mul(h)
        .saturating_mul(2)
        .saturating_add(
            3u64.saturating_mul(p)
                .saturating_mul(text.linear_attn_config.short_conv_kernel_size as u64)
                .saturating_mul(4),
        )
        .saturating_add(d.saturating_mul(h).saturating_mul(2))
        .saturating_add(p.saturating_mul(d).saturating_mul(2))
        .saturating_add(kda_heads.saturating_mul(h).saturating_mul(2))
        .saturating_add(d.saturating_mul(4))
        .saturating_add(p.saturating_mul(4))
        .saturating_add(d.saturating_mul(4));
    let q_width = heads.saturating_mul((text.qk_nope_head_dim + text.qk_rope_head_dim) as u64);
    let kv_width = heads.saturating_mul((text.qk_nope_head_dim + text.v_head_dim) as u64);
    let output_width = heads.saturating_mul(text.v_head_dim as u64);
    let mla = (text.q_lora_rank as u64)
        .saturating_mul(h)
        .saturating_mul(2)
        // LayerNorm vectors are widened from checkpoint BF16 to resident F32.
        .saturating_add((text.q_lora_rank as u64).saturating_mul(4))
        .saturating_add(
            q_width
                .saturating_mul(text.q_lora_rank as u64)
                .saturating_mul(2),
        )
        .saturating_add((text.kv_lora_rank + text.qk_rope_head_dim) as u64 * h * 2)
        .saturating_add((text.kv_lora_rank as u64).saturating_mul(4))
        .saturating_add(
            kv_width
                .saturating_mul(text.kv_lora_rank as u64)
                .saturating_mul(2),
        )
        .saturating_add(output_width.saturating_mul(h).saturating_mul(4));
    let dense = 6u64
        .saturating_mul(text.intermediate_size as u64)
        .saturating_mul(h);
    let shared_width = text
        .moe_intermediate_size
        .saturating_mul(text.num_shared_experts) as u64;
    let sparse = (text.num_experts as u64)
        .saturating_mul(h)
        .saturating_mul(2)
        .saturating_add((text.num_experts as u64).saturating_mul(4))
        .saturating_add(
            4u64.saturating_mul(text.routed_expert_hidden_size as u64)
                .saturating_mul(h),
        )
        // The routed RMSNorm vector is also widened to resident F32.
        .saturating_add((text.routed_expert_hidden_size as u64).saturating_mul(4))
        .saturating_add(6u64.saturating_mul(shared_width).saturating_mul(h));
    common
        .saturating_add(kda)
        .saturating_add(dense)
        .max(common.saturating_add(kda).saturating_add(sparse))
        .max(common.saturating_add(mla).saturating_add(sparse))
}

fn kimi_k3_common_vector_bytes(hidden: u64) -> u64 {
    6u64.saturating_mul(hidden).saturating_mul(4)
}

fn kimi_k3_attn_res_working_set(layers: usize, block: usize, hidden: usize) -> u64 {
    let snapshots = layers
        .saturating_sub(1)
        .checked_div(block)
        .unwrap_or(0)
        .saturating_add(1);
    // Retain the committed snapshots while assembling one temporary source stack.
    2u64.saturating_mul(snapshots.saturating_add(1) as u64)
        .saturating_mul(hidden as u64)
        .saturating_mul(4)
}

fn kimi_k3_prefill_scratch_per_token(
    layers: usize,
    attn_res_block: usize,
    hidden: usize,
    attention_heads: usize,
) -> (usize, u64) {
    let snapshots = layers
        .saturating_sub(1)
        .checked_div(attn_res_block)
        .unwrap_or(0)
        .saturating_add(1);
    let bytes = (snapshots as u64)
        .saturating_add(1)
        .saturating_mul(hidden as u64)
        .saturating_add(attention_heads as u64)
        .saturating_mul(4);
    (snapshots, bytes)
}

fn deepseek_kv_bytes(config: &DeepseekV4Config, context: usize, state_bytes: u8) -> Option<u64> {
    if context == 0 || context > config.max_position_embeddings {
        return None;
    }
    let f32_bytes = schema::kv_cache_bytes_for_context(config, context).ok()?;
    f32_bytes
        .checked_div(4)?
        .checked_mul(u64::from(state_bytes))
}

fn deepseek_context_under_budget(
    config: &DeepseekV4Config,
    kv_budget: u64,
    state_bytes: u8,
) -> u64 {
    let mut low = 0usize;
    let mut high = config.max_position_embeddings;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let fits =
            deepseek_kv_bytes(config, middle, state_bytes).is_some_and(|bytes| bytes <= kv_budget);
        if fits {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::report::{CategoryStats, KvEstimate, ModelSummary};
    use std::path::PathBuf;

    fn config() -> GlmConfig {
        GlmConfig::from_json_str(
            &serde_json::json!({
                "model_type":"glm_moe_dsa", "hidden_size":8, "num_hidden_layers":2,
                "num_attention_heads":2, "vocab_size":16, "intermediate_size":12,
                "moe_intermediate_size":4, "first_k_dense_replace":1,
                "n_routed_experts":8, "n_shared_experts":1, "num_experts_per_tok":2,
                "q_lora_rank":4, "kv_lora_rank":3, "qk_nope_head_dim":2,
                "qk_rope_head_dim":2, "qk_head_dim":4, "v_head_dim":3,
                "index_n_heads":2, "index_head_dim":2, "index_topk":8,
                "indexer_types":["full", "shared"],
                "max_position_embeddings":64, "hidden_act":"silu",
                "scoring_func":"sigmoid", "topk_method":"noaux_tc"
            })
            .to_string(),
        )
        .unwrap()
    }

    fn report(config: &GlmConfig, indexer_weights_complete: bool) -> CheckpointReport {
        let kv = kv_estimate(config, 4);
        CheckpointReport {
            model_path: PathBuf::from("fixture"),
            model: ModelSummary {
                model_type: "glm_moe_dsa".to_owned(),
                hidden_size: 8,
                layers: 2,
                dense_layers: 1,
                sparse_layers: 1,
                routed_experts_per_layer: 8,
                active_experts_per_token: 2,
                attention_heads: 2,
                vocab_size: 16,
                max_context: 64,
                full_indexer_layers: 1,
                index_topk: 8,
            },
            shard_count: 1,
            tensor_count: 1,
            file_bytes: 1_800,
            tensor_payload_bytes: 1_800,
            container_overhead_bytes: 0,
            known_logical_parameters: 0,
            estimated_active_parameters_per_token: 0,
            unknown_packed_bytes: 0,
            categories: vec![CategoryStats {
                category: ParameterCategory::RoutedExpert,
                tensor_count: 24,
                logical_parameters: 800,
                payload_bytes: 800,
                scale_bytes: 0,
            }],
            quantization: Vec::new(),
            kv_cache_f32: KvEstimate { ..kv },
            routed_expert_count_found: 8,
            minimum_routed_expert_bytes: 100,
            maximum_routed_expert_bytes: 100,
            required_base_tensor_count: 0,
            missing_base_tensor_count: 0,
            sidecars_without_weight: 0,
            indexer_weights_complete,
            format_notes: Vec::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn context_limit_is_infeasible_and_expert_slots_are_physically_capped() {
        let config = config();
        let report = report(&config, false);
        let plan = build_resource_plan(&config, &report, 4 * 1024 * MIB, 65, 4);
        assert!(!plan.feasible);
        assert_eq!(plan.expert_slots_total, 8);
        assert_eq!(plan.expert_slots_per_sparse_layer, 8);
        assert_eq!(plan.expert_cache_budget_bytes, 800);
        assert_eq!(plan.maximum_context_under_budget, 8);
        assert!(plan.unused_ram_after_full_expert_residency > 0);
        assert!(!plan.kv_includes_configured_indexer);
        assert_eq!(
            plan.kv_cache_bytes,
            report.kv_cache_f32.mla_compressed_bytes_per_token * 65
        );
    }

    #[test]
    fn complete_indexer_uses_the_configured_extra_key_cache() {
        let config = config();
        let report = report(&config, true);
        let plan = build_resource_plan(&config, &report, 4 * 1024 * MIB, 32, 4);
        assert!(plan.feasible);
        assert!(plan.kv_includes_configured_indexer);
        assert_eq!(
            plan.kv_cache_bytes,
            report.kv_cache_f32.compressed_bytes_per_token * 32
        );
    }

    #[test]
    fn worst_case_expert_size_drives_slots_and_cold_io() {
        let config = config();
        let mut report = report(&config, false);
        report.maximum_routed_expert_bytes = 200;
        let plan = build_resource_plan(&config, &report, 4 * 1024 * MIB, 32, 4);
        assert_eq!(plan.average_expert_bytes, 100);
        assert_eq!(plan.maximum_expert_bytes, 200);
        assert_eq!(plan.cold_routed_bytes_per_token, 400);
    }

    #[test]
    fn zero_persistent_slots_still_require_one_transient_expert() {
        let config = config();
        let report = report(&config, false);
        let resident = report.tensor_payload_bytes - 800;
        let kv = report.kv_cache_f32.mla_compressed_bytes_per_token;
        let budget = 1_024 * MIB + resident + kv + 99;
        let plan = build_resource_plan(&config, &report, budget, 1, 4);
        assert_eq!(plan.maximum_expert_bytes, 100);
        assert_eq!(plan.transient_expert_bytes, 100);
        assert_eq!(plan.expert_slots_per_sparse_layer, 0);
        assert!(!plan.feasible);
        assert!(plan
            .notes
            .iter()
            .any(|note| note.contains("transient expert")));
    }

    #[test]
    fn kimi_k3_native_expert_storage_matches_the_release_abi() {
        assert_eq!(kimi_k3_expert_bytes(3_584, 3_072), 17_547_264);
        assert_eq!(
            kimi_k3_expert_bytes(3_584, 3_072)
                .saturating_mul(92)
                .saturating_mul(896),
            1_446_456_066_048
        );
    }

    #[test]
    fn kimi_k3_layer_common_vectors_are_planned_at_expanded_f32_residency() {
        assert_eq!(kimi_k3_common_vector_bytes(7_168), 172_032);
    }

    #[test]
    fn kimi_k3_layerwise_prefill_accounts_for_every_token_snapshot() {
        assert_eq!(
            kimi_k3_prefill_scratch_per_token(93, 12, 7_168, 96),
            (8, 258_432)
        );
    }

    #[test]
    fn kimi_k3_attn_res_residency_uses_snapshot_count_not_block_width() {
        assert_eq!(kimi_k3_attn_res_working_set(93, 12, 7_168), 516_096);
    }
}
