//! Model architecture explanations shared by terminal and plain-text frontends.

use crate::models::qwen3_8::Qwen38Config;
use crate::{
    CommonModelConfig, DeepseekV41Config, DeepseekV4Config, GlmConfig, Hy4Config, KimiK3Config,
    ModelConfig,
};
use std::error::Error;
use std::path::{Path, PathBuf};

pub struct Explanation {
    pub model_path: PathBuf,
    pub model: CommonModelConfig,
    pub lines: Vec<String>,
}

pub fn explain_checkpoint(model_dir: &Path) -> Result<Explanation, Box<dyn Error + Send + Sync>> {
    let config = ModelConfig::load(model_dir)?;
    let model = config.common();
    let lines = match &config {
        ModelConfig::Glm52(config) => glm_explanation(config),
        ModelConfig::DeepseekV4(config) => deepseek_explanation(config),
        ModelConfig::DeepseekV41(config) => deepseek_v41_explanation(config),
        ModelConfig::Hy4(config) => hy4_explanation(config),
        ModelConfig::KimiK3(config) => kimi_k3_explanation(config),
        ModelConfig::Qwen38(config) => qwen38_explanation(config),
    };
    Ok(Explanation {
        model_path: model_dir.to_owned(),
        model,
        lines,
    })
}

fn glm_explanation(config: &GlmConfig) -> Vec<String> {
    let q_lora = config.q_lora_rank_value();
    vec![
        "GLM-5.2 token path (row-major weights use Y = X · Wᵀ)".to_owned(),
        format!("  token → embedding [{}]", config.hidden_size),
        format!(
            "  × {} decoder blocks: RMSNorm → MLA/IndexShare → residual → RMSNorm → FFN → residual",
            config.num_hidden_layers
        ),
        format!(
            "  MLA query: {} → {} → {} heads × {} dims ({} non-RoPE + {} RoPE)",
            config.hidden_size,
            q_lora,
            config.num_attention_heads,
            config.qk_head_dim,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim
        ),
        format!(
            "  compressed KV/token/layer: {} latent + {} shared RoPE values",
            config.kv_lora_rank, config.qk_rope_head_dim
        ),
        format!(
            "  FFN: {} dense layers, then {} MoE layers; each token selects {}/{} routed experts plus {} shared",
            config.num_hidden_layers - config.sparse_layer_count(),
            config.sparse_layer_count(),
            config.num_experts_per_tok,
            config.n_routed_experts,
            config.n_shared_experts
        ),
        format!(
            "  router: choose by sigmoid(logit)+bias; mix by unbiased sigmoid, normalize={}, scale={}",
            config.norm_topk_prob, config.routed_scaling_factor
        ),
        format!(
            "  final RMSNorm → LM head [{} logits] → sampler",
            config.vocab_size
        ),
    ]
}

fn deepseek_explanation(config: &DeepseekV4Config) -> Vec<String> {
    vec![
        "DeepSeek-V4 token path (row-major weights use Y = X · Wᵀ)".to_owned(),
        format!(
            "  token → streamed embedding [{}] → {} Hyper-Connection streams",
            config.hidden_size, config.hc_mult
        ),
        format!(
            "  × {} decoder blocks: HC/Sinkhorn → RMSNorm → MLA → HC merge → HC/Sinkhorn → RMSNorm → MoE → HC merge",
            config.num_hidden_layers
        ),
        format!(
            "  MLA query: {} → Q-LoRA {} → {} heads × {} dims ({} paired-RoPE dims)",
            config.hidden_size,
            config.q_lora_rank,
            config.num_attention_heads,
            config.head_dim,
            config.qk_rope_head_dim
        ),
        format!(
            "  attention memory: {}-token local window; compression schedule has {} indexed ratio-4 layers",
            config.sliding_window,
            config.indexed_layer_count()
        ),
        format!(
            "  MoE: every base layer selects {}/{} routed FP4 experts plus {} shared FP8 expert; first {} layers use token-hash IDs",
            config.num_experts_per_tok,
            config.n_routed_experts,
            config.n_shared_experts,
            config.num_hash_layers
        ),
        format!(
            "  router: sqrt(softplus(logit)); correction bias selects only; normalized × {}",
            config.routed_scaling_factor
        ),
        format!(
            "  HC head → final RMSNorm → streamed LM head [{} logits] → sampler",
            config.vocab_size
        ),
        format!(
            "  checkpoint also contains {} DSpark stage(s), validated independently of base decoding",
            config.declared_dspark_stage_count()
        ),
    ]
}

fn deepseek_v41_explanation(config: &DeepseekV41Config) -> Vec<String> {
    let text = &config.text_config;
    vec![
        "DeepSeek-V4.1-Flash execution model".to_owned(),
        format!(
            "  CED stack: {} causal-encoder + {} decoder layers; hidden {} x {} Single-Pass mHC streams",
            config.encoder_layer_count(),
            config.decoder_layer_count(),
            text.hidden_size,
            text.hc_mult
        ),
        format!(
            "  CSA2: KV sources {:?}; index sources {:?}; top-{}; decoder candidates {} blocks x {} positions",
            text.kv_source_layer_ids,
            text.index_source_layer_ids,
            text.index_topk,
            text.candidate_topk_blocks,
            text.candidate_block_size
        ),
        format!(
            "  MoE: {} routed experts, top-{}, plus {} shared expert; intermediate {}",
            text.n_routed_experts,
            text.num_experts_per_tok,
            text.n_shared_experts,
            text.moe_intermediate_size
        ),
        format!(
            "  Engram: layers {:?}, table rows {:?}, 2-gram through {}-gram with {} heads",
            text.engram_layer_ids,
            text.engram_num_embeddings,
            text.engram_max_ngram_size,
            text.engram_n_heads
        ),
        format!(
            "  checkpoint: 32x32 E4M3/E8M0 trunk, packed E2M1 experts; context {}",
            text.max_position_embeddings
        ),
        "  execution status: scalar base-text CED/CSA2/Engram generation is enabled; vision and DSpark remain schema-only".to_owned(),
    ]
}

fn hy4_explanation(config: &Hy4Config) -> Vec<String> {
    vec![
        "Hy4-preview token path (row-major weights use Y = X · Wᵀ)".to_owned(),
        format!(
            "  token → streamed BF16 embedding [{}] → {} identity-HC streams",
            config.hidden_size, config.hc_mult
        ),
        format!(
            "  × {} blocks: iHC → RMSNorm → Gated DSA/MLA → iHC merge → iHC → RMSNorm → FFN/MoE → iHC merge",
            config.num_hidden_layers
        ),
        format!(
            "  MLA: Q-LoRA {} · KV-LoRA {} · {} heads × ({} NoPE + {} RoPE) → {}-dim values",
            config.q_lora_rank,
            config.kv_lora_rank,
            config.num_attention_heads,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.v_head_dim
        ),
        format!(
            "  Gated attention: elementwise {}-wide gate + {} learnable per-head sinks",
            config.num_attention_heads * config.v_head_dim,
            config.num_attention_heads
        ),
        format!(
            "  DSA IndexCache: {} heads × {} dims · top-{} positions · {} full indexer layers",
            config.index_n_heads,
            config.index_head_dim,
            config.index_topk,
            config.full_indexer_layer_count()
        ),
        format!(
            "  FFN: {} dense + {} MoE layers; each token selects {}/{} routed experts plus {} shared",
            config.num_hidden_layers - config.sparse_layer_count(),
            config.sparse_layer_count(),
            config.num_experts_per_tok,
            config.n_routed_experts,
            config.n_shared_experts
        ),
        "  storage: ModelOpt MXFP8 E4M3 matrices with U8 E8M0 scales per 1×32 input block".to_owned(),
        format!(
            "  final iHC head → RMSNorm → streamed BF16 LM head [{} logits]; {} native MTP layer is validation-only",
            config.vocab_size, config.num_nextn_predict_layers
        ),
        format!(
            "  context: checkpoint advertises {}; exact dense fallback covers ≤{} tokens until IndexCache execution lands",
            config.max_position_embeddings,
            config.exact_dense_context_ceiling()
        ),
    ]
}

fn kimi_k3_explanation(config: &KimiK3Config) -> Vec<String> {
    let text = &config.text_config;
    vec![
        "Kimi-K3 text token path (row-major weights use Y = X · Wᵀ)".to_owned(),
        format!(
            "  XTML/tiktoken token → embedding [{}] → {} decoder blocks",
            text.hidden_size, text.num_hidden_layers
        ),
        format!(
            "  hybrid attention: {} KDA layers ({} heads × {} state dims, causal conv {}) + {} gated NoPE-MLA layers",
            text.kda_layer_count(),
            text.linear_attn_config.num_heads,
            text.linear_attn_config.head_dim,
            text.linear_attn_config.short_conv_kernel_size,
            text.full_attention_layer_count()
        ),
        format!(
            "  MLA: Q-LoRA {} · KV-LoRA {} · {} heads × ({} NoPE + {} reserved RoPE) → gated {}-dim values",
            text.q_lora_rank,
            text.kv_lora_rank,
            text.num_attention_heads,
            text.qk_nope_head_dim,
            text.qk_rope_head_dim,
            text.v_head_dim
        ),
        format!(
            "  AttnRes block size {} mixes normalized keys with raw residual values",
            text.attn_res_block_size
        ),
        format!(
            "  LatentMoE: {} dense layer, then {} sparse layers; each token selects {}/{} MXFP4 routed experts plus {} shared experts",
            text.first_k_dense_replace,
            text.num_hidden_layers.saturating_sub(text.first_k_dense_replace),
            text.num_experts_per_token,
            text.num_experts,
            text.num_shared_experts
        ),
        format!(
            "  routed expert: {} → latent {} → SiTU hidden {} → latent → {}, with normalized aggregate",
            text.hidden_size,
            text.routed_expert_hidden_size,
            text.moe_intermediate_size,
            text.hidden_size
        ),
        format!(
            "  final RMSNorm + output AttnRes → LM head [{} logits]; generation stop token is {}",
            text.vocab_size, config.eos_token_id
        ),
        format!(
            "  MoonViT-V2 has {} vision blocks; vision execution is a separate milestone from text-only generation",
            config.vision_config.vt_num_hidden_layers
        ),
    ]
}

fn qwen38_explanation(config: &Qwen38Config) -> Vec<String> {
    let linear = config
        .layer_types
        .iter()
        .filter(|kind| kind.as_str() == "linear_attention")
        .count();
    let full = config.num_hidden_layers.saturating_sub(linear);
    vec![
        "Qwen3.8-2.4T-A95B-FP8 token path (Y = X · Wᵀ)".to_owned(),
        format!(
            "  byte-BPE/ChatML token → embedding [{}] → {} hybrid decoder blocks",
            config.hidden_size, config.num_hidden_layers
        ),
        format!(
            "  hybrid attention: {linear} Gated DeltaNet layers ({} key heads, {} value heads, dim {}, conv {}) + {full} gated GQA layers",
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_conv_kernel_dim
        ),
        format!(
            "  full GQA: {} query heads / {} KV heads × {} dims; RoPE covers {:.0}% of each head",
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.partial_rotary_factor * 100.0
        ),
        format!(
            "  MoE: every layer selects {}/{} routed E4M3 experts plus one BF16 shared expert (intermediate {})",
            config.num_experts_per_tok,
            config.num_experts,
            config.moe_intermediate_size
        ),
        "  routed storage: 128×128 E4M3 blocks with BF16 weight_scale_inv; trunk/router/shared expert stay BF16".to_owned(),
        format!(
            "  final RMSNorm → independent LM head [{} logits]; native context {}",
            config.vocab_size, config.max_position_embeddings
        ),
        "  runtime boundary: full scalar forward, state, streaming, and preflight are implemented; public generate awaits real-weight and independent-logits gates".to_owned(),
    ]
}
