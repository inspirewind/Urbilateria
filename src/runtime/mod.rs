//! Model-neutral constrained-memory contracts and cache services.
//!
//! Model implementations and their state types live under `models::<family>::runtime`.

pub(crate) mod cache;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeLoadOptions {
    pub resident_budget_bytes: u64,
    pub expert_cache_budget_bytes: u64,
    pub kv_cache_budget_bytes: u64,
    pub expert_slots_per_layer: usize,
    pub maximum_expert_bytes: u64,
    pub context_limit: usize,
}

pub use cache::ExpertTelemetry;

// Transitional aliases for callers that used the old GLM-specific root runtime API.
#[deprecated(note = "use models::glm::runtime::GlmRuntimeError")]
pub type RuntimeError = crate::models::glm::runtime::GlmRuntimeError;
#[deprecated(note = "use models::glm::runtime::GlmRuntimeModel")]
pub type RuntimeModel = crate::models::glm::runtime::GlmRuntimeModel;
#[deprecated(note = "use models::glm::runtime::GlmRuntimeRequirements")]
pub type RuntimeRequirements = crate::models::glm::runtime::GlmRuntimeRequirements;
#[deprecated(note = "use models::glm::runtime::GlmRuntimeState")]
pub type RuntimeState = crate::models::glm::runtime::GlmRuntimeState;
#[deprecated(note = "use models::glm::runtime::GlmRuntimeStep")]
pub type RuntimeStep = crate::models::glm::runtime::GlmRuntimeStep;
