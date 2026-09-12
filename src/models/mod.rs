//! Model-family implementations.
//!
//! Cross-model infrastructure lives in `storage`, `math`, `generation`, and `runtime`.
//! Architecture math, tensor schemas, and prompt protocols stay in their family module.

pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod glm;
pub mod hy4;
pub mod kimi_k3;
pub mod qwen3_8;
