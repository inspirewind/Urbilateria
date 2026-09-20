//! Urbilateria is a deliberately small pure-Rust multi-model inference research framework.
//!
//! It combines trustworthy model inspection and resource planning with a readable scalar
//! tokenizer, family-specific mixed-quantized runtimes, streamed-expert cache, and one shared
//! autoregressive token loop. GLM-5.2 and DeepSeek-V4 deliberately meet only at small stable
//! boundaries; their blocks and tensor schemas remain model-private.
//! Formal large-checkpoint execution remains behind explicit correctness gates; see
//! `docs/ROADMAP.md`.

pub mod analysis;
pub mod config;
pub mod execution;
pub mod generation;
pub mod math;
pub mod model;
pub mod models;
pub mod profiling;
pub mod runtime;
pub mod storage;
pub mod tokenizer;

#[cfg(test)]
mod test_support;

pub use config::{
    CommonModelConfig, DeepseekV41Config, DeepseekV4Config, GlmConfig, Hy4Config, KimiK3Config,
    ModelConfig, ModelFamily, Qwen38Config,
};
