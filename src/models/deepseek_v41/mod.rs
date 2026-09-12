//! DeepSeek-V4.1 model-family support.
//!
//! V4.1 deliberately has its own family boundary. Its CED execution graph, CSA2 cache
//! ownership, Single-Pass mHC state, Engram tables, and 32x32 MXFP8 checkpoint ABI are not
//! compatible with the earlier DeepSeek-V4 adapter.

pub mod compressor;
mod config;
pub mod engram;
pub mod indexer;
pub mod kv;
pub mod math;
pub mod mhc;
pub mod prompt;
pub mod runtime;
pub mod schema;

pub use config::{
    AttentionMode, DeepseekV41Config, DeepseekV41QuantizationConfig, DeepseekV41TextConfig,
    DeepseekV41VisionConfig, RopeScaling,
};
