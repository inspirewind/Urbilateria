//! Static checkpoint analysis, tensor probing, and constrained-hardware planning.

mod classify;
mod plan;
mod probe;
mod report;
mod trace;

pub use classify::{classify_tensor, expected_matrix_shape, ParameterCategory};
pub use plan::{
    build_deepseek_resource_plan, build_deepseek_v41_resource_plan, build_hy4_resource_plan,
    build_kimi_k3_resource_plan, build_resource_plan, ResourcePlan,
};
pub use probe::{probe_tensor, NumericStats, ProbeError, ProbeReport, QuantProbe};
pub use report::{analyze_checkpoint, AnalysisError, CheckpointReport};
pub use trace::{
    CacheSimulation, ExpertFrequency, ExpertPairFrequency, LayerCacheSimulation, LayerRouteSummary,
    RouteEvent, RouteSample, RouteTrace, RouteTraceSummary, TraceError,
};
