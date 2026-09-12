/// Routing geometry used by GLM's bounded-memory MoE executor.
#[derive(Debug, Clone, Copy)]
pub struct MoeGeometry {
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub normalize: bool,
    pub routed_scaling_factor: f32,
}
