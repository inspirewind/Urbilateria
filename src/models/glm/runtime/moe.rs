use super::{ExpertStore, ExpertStoreError};
use crate::math::{route_noaux_tc, RouteChoice, RouteError};
use crate::model::{GatedMlp, MlpError, WeightError, WeightMatrix};
use crate::models::glm::MoeGeometry;
use std::fmt;

#[derive(Debug)]
pub enum RuntimeMoeError {
    Weight(WeightError),
    Mlp(MlpError),
    Route(RouteError),
    Expert(ExpertStoreError),
    InvalidShape(String),
    InvalidGeometry(String),
}

impl fmt::Display for RuntimeMoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight(error) => error.fmt(f),
            Self::Mlp(error) => error.fmt(f),
            Self::Route(error) => error.fmt(f),
            Self::Expert(error) => error.fmt(f),
            Self::InvalidShape(reason) => write!(f, "invalid streamed MoE shape: {reason}"),
            Self::InvalidGeometry(reason) => write!(f, "invalid streamed MoE geometry: {reason}"),
        }
    }
}

impl std::error::Error for RuntimeMoeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight(error) => Some(error),
            Self::Mlp(error) => Some(error),
            Self::Route(error) => Some(error),
            Self::Expert(error) => Some(error),
            Self::InvalidShape(_) | Self::InvalidGeometry(_) => None,
        }
    }
}

impl From<WeightError> for RuntimeMoeError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

impl From<MlpError> for RuntimeMoeError {
    fn from(value: MlpError) -> Self {
        Self::Mlp(value)
    }
}

impl From<RouteError> for RuntimeMoeError {
    fn from(value: RouteError) -> Self {
        Self::Route(value)
    }
}

impl From<ExpertStoreError> for RuntimeMoeError {
    fn from(value: ExpertStoreError) -> Self {
        Self::Expert(value)
    }
}

/// A sparse GLM feed-forward layer whose routed experts come from [`ExpertStore`].
#[derive(Debug, Clone)]
pub struct RuntimeMoeLayer {
    layer: usize,
    router: WeightMatrix,
    correction_bias: Vec<f32>,
    shared: GatedMlp,
    geometry: MoeGeometry,
}

impl RuntimeMoeLayer {
    pub fn new(
        layer: usize,
        router: WeightMatrix,
        correction_bias: Vec<f32>,
        shared: GatedMlp,
        geometry: MoeGeometry,
    ) -> Result<Self, RuntimeMoeError> {
        if router.rows() == 0 || router.cols() == 0 {
            return Err(RuntimeMoeError::InvalidGeometry(
                "router dimensions must be non-zero".to_owned(),
            ));
        }
        if correction_bias.len() != router.rows() {
            return Err(RuntimeMoeError::InvalidShape(format!(
                "router has {} experts but correction bias has {} values",
                router.rows(),
                correction_bias.len()
            )));
        }
        if correction_bias.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeMoeError::InvalidGeometry(
                "correction bias contains NaN or infinity".to_owned(),
            ));
        }
        if shared.hidden_size() != router.cols() {
            return Err(RuntimeMoeError::InvalidShape(format!(
                "router hidden size {} differs from shared expert hidden size {}",
                router.cols(),
                shared.hidden_size()
            )));
        }
        route_noaux_tc(
            &vec![0.0; router.rows()],
            &correction_bias,
            geometry.top_k,
            geometry.n_group,
            geometry.topk_group,
            geometry.normalize,
            geometry.routed_scaling_factor,
        )?;
        Ok(Self {
            layer,
            router,
            correction_bias,
            shared,
            geometry,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.router.cols()
    }

    pub fn expert_count(&self) -> usize {
        self.router.rows()
    }

    pub fn layer_index(&self) -> usize {
        self.layer
    }

    pub fn forward(
        &self,
        input: &[f32],
        experts: &mut ExpertStore,
    ) -> Result<(Vec<f32>, Vec<RouteChoice>), RuntimeMoeError> {
        if experts.expert_count() != self.expert_count()
            || experts.hidden_size() != self.hidden_size()
        {
            return Err(RuntimeMoeError::InvalidShape(format!(
                "expert store is hidden={} experts={}, layer needs hidden={} experts={}",
                experts.hidden_size(),
                experts.expert_count(),
                self.hidden_size(),
                self.expert_count()
            )));
        }
        let logits = self.router.matvec(input)?;
        let routes = route_noaux_tc(
            &logits,
            &self.correction_bias,
            self.geometry.top_k,
            self.geometry.n_group,
            self.geometry.topk_group,
            self.geometry.normalize,
            self.geometry.routed_scaling_factor,
        )?;
        let shared = self.shared.forward(input)?;
        let mut output: Vec<f64> = shared.into_iter().map(f64::from).collect();
        for route in &routes {
            let expert = experts.execute(self.layer, route.expert, input)?;
            for (accumulator, value) in output.iter_mut().zip(expert) {
                *accumulator += f64::from(route.weight) * f64::from(value);
            }
        }
        Ok((
            output.into_iter().map(|value| value as f32).collect(),
            routes,
        ))
    }
}
