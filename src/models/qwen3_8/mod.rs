//! Qwen3.8-2.4T-A95B-FP8 release adapter.

pub mod attention;
mod config;
pub mod expert;
mod expert_store;
pub mod linear_attention;
mod math;
pub mod moe;
pub mod norm;
pub mod prompt;
pub mod reference;
mod runtime;
pub mod schema;
mod weights;

pub use config::{
    QuantizationConfig, Qwen38Config, Qwen38GenerationConfig, Qwen38ReleaseConfig, RopeParameters,
    ARCHITECTURE, BOS_TOKEN_ID, IM_END_TOKEN_ID, MODEL_TYPE,
};
pub use runtime::{
    Qwen38RuntimeError, Qwen38RuntimeModel, Qwen38RuntimeRequirements, Qwen38RuntimeState,
    Qwen38RuntimeStep, Qwen38RuntimeTraceStep,
};
pub use schema::{
    inspect_manifest, inspect_requirements, official_tensor_specs, validate_checkpoint,
    HfWeightIndex, Qwen38ManifestReport, Qwen38Requirements, SchemaError, TensorSpec,
    BASE_ROUTED_EXPERT_COUNT, BASE_TENSOR_COUNT, MTP_TENSOR_COUNT, RELEASE_LOGICAL_PARAMETER_COUNT,
    RELEASE_SHARD_COUNT, RELEASE_TENSOR_COUNT, RELEASE_TOTAL_SIZE,
};
