//! Qwen3.6-35B-A3B: BF16 text inference over the native multimodal checkpoint.
mod config;
mod expert_store;
pub mod prompt;
mod runtime;
pub mod schema;
mod weights;
pub use config::*;
pub use runtime::*;
