//! Native BF16 text adapter for Qwen3.8-Flash-Next (`qwen4_exp`).
mod components;
mod config;
pub mod prompt;
mod runtime;
pub mod schema;
pub use config::{
    Qwen38FlashNextConfig, Qwen38FlashNextGenerationConfig, BOS_TOKEN_ID, IM_END_TOKEN_ID,
};
pub use runtime::{
    Error, Qwen38FlashNextRuntimeModel, Qwen38FlashNextRuntimeRequirements,
    Qwen38FlashNextRuntimeState,
};
