//! GLM-5.2 constrained-memory runtime.

mod expert;
mod model;
mod moe;

pub use crate::runtime::ExpertTelemetry;
pub use expert::{ExpertStore, ExpertStoreError};
pub use model::{
    RuntimeError as GlmRuntimeError, RuntimeModel as GlmRuntimeModel,
    RuntimeRequirements as GlmRuntimeRequirements, RuntimeState as GlmRuntimeState,
    RuntimeStep as GlmRuntimeStep,
};
pub use moe::{RuntimeMoeError, RuntimeMoeLayer};

#[deprecated(note = "use GlmRuntimeError")]
pub type RuntimeError = GlmRuntimeError;
#[deprecated(note = "use GlmRuntimeModel")]
pub type RuntimeModel = GlmRuntimeModel;
#[deprecated(note = "use GlmRuntimeRequirements")]
pub type RuntimeRequirements = GlmRuntimeRequirements;
#[deprecated(note = "use GlmRuntimeState")]
pub type RuntimeState = GlmRuntimeState;
#[deprecated(note = "use GlmRuntimeStep")]
pub type RuntimeStep = GlmRuntimeStep;
