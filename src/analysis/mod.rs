//! Static checkpoint analysis, tensor probing, and constrained-hardware planning.

mod classify;
mod explain;
mod inspect;
mod list;
mod plan;
mod planning;
mod preflight;
mod probe;
mod report;
mod text;
mod trace;

pub use classify::{classify_tensor, expected_matrix_shape, ParameterCategory};
pub use explain::{explain_checkpoint, Explanation};
pub use inspect::{inspect_checkpoint, Inspection, InspectionReport};
pub use list::{list_tensors, ListOptions, TensorListing};
pub use plan::{
    build_deepseek_resource_plan, build_deepseek_v41_resource_plan, build_hy4_resource_plan,
    build_kimi_k3_resource_plan, build_resource_plan, ResourcePlan,
};
pub use planning::{detect_available_ram, plan_checkpoint, PlanOptions, Planning};
pub use preflight::{preflight_checkpoint, Preflight, PreflightOptions, PreflightReport};
pub use probe::{probe_tensor, NumericStats, ProbeError, ProbeReport, QuantProbe};
pub use report::{analyze_checkpoint, AnalysisError, CheckpointReport};
pub use text::{
    decode_tokens, parse_token_ids, tokenize_text, DecodeReport, Decoding, Tokenization,
    TokenizeOptions, TokenizeReport,
};
pub use trace::{
    CacheSimulation, ExpertFrequency, ExpertPairFrequency, LayerCacheSimulation, LayerRouteSummary,
    RouteEvent, RouteSample, RouteTrace, RouteTraceSummary, TraceError,
};
