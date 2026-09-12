//! Kimi-K3 model-family support.

pub mod attention;
pub mod config;
pub mod expert;
pub mod geometry;
pub mod math;
pub mod moe;
pub mod moe_runtime;
pub mod prompt;
pub mod residual;
pub mod runtime;
pub mod schema;
pub mod tokenizer;
pub mod weights;

pub use config::{
    KimiK3AutoMap, KimiK3Config, KimiK3LinearAttentionConfig, KimiK3QuantizationConfig,
    KimiK3QuantizationGroup, KimiK3QuantizedWeights, KimiK3TextConfig, KimiK3VisionConfig,
};
