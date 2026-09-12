//! Scalar, storage-independent reference implementation of Kimi-K3 LatentMoE.
//!
//! The released checkpoint routes on the full 7,168-wide hidden state, compresses the
//! routed branch to 3,584 values, evaluates 16 of 896 latent experts, normalizes their
//! weighted aggregate, and projects it back to the hidden width.  Its two shared experts
//! form one unweighted full-width SiTU MLP.  This module expresses that data flow without
//! assuming how any matrix is stored; an optimized runtime can stream MXFP4 experts through
//! [`LatentMoeProjector`] while tests can inject tiny dense matrices.

use super::math::{situ_glu, KimiK3MathError};
use std::fmt;

/// The released checkpoint's SiTU gate cap.
pub const KIMI_K3_SITU_BETA: f32 = 4.0;
/// The released checkpoint's SiTU linear-branch cap.
pub const KIMI_K3_SITU_LINEAR_BETA: f32 = 25.0;

const ROUTE_NORMALIZATION_EPSILON: f64 = 1e-20;

/// Geometry for K3's `sigmoid + noaux_tc` router.
///
/// The released K3 model has 896 experts in one group (`num_expert_group=1`), selects
/// 16, renormalizes their unbiased sigmoid scores, and uses a scaling factor of one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoAuxTcConfig {
    pub expert_count: usize,
    pub top_k: usize,
    pub expert_group_count: usize,
    pub selected_group_count: usize,
    pub routed_scaling_factor: f32,
}

impl NoAuxTcConfig {
    pub const KIMI_K3: Self = Self {
        expert_count: 896,
        top_k: 16,
        expert_group_count: 1,
        selected_group_count: 1,
        routed_scaling_factor: 1.0,
    };

    pub fn validate(self) -> Result<(), KimiK3MoeError> {
        if self.expert_count == 0 {
            return Err(invalid_geometry("expert_count", "must be non-zero"));
        }
        if self.top_k == 0 || self.top_k > self.expert_count {
            return Err(invalid_geometry(
                "top_k",
                format!("must be in 1..={}", self.expert_count),
            ));
        }
        if self.expert_group_count == 0 || self.expert_count % self.expert_group_count != 0 {
            return Err(invalid_geometry(
                "expert_group_count",
                format!(
                    "{} experts must divide into a non-zero number of equal groups",
                    self.expert_count
                ),
            ));
        }
        if self.selected_group_count == 0 || self.selected_group_count > self.expert_group_count {
            return Err(invalid_geometry(
                "selected_group_count",
                format!("must be in 1..={}", self.expert_group_count),
            ));
        }
        if !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 {
            return Err(invalid_geometry(
                "routed_scaling_factor",
                "must be finite and strictly positive",
            ));
        }

        let experts_per_group = self.expert_count / self.expert_group_count;
        let group_filtering =
            self.expert_group_count > 1 && self.expert_group_count > self.selected_group_count;
        if group_filtering && experts_per_group < 2 {
            return Err(invalid_geometry(
                "expert_group_count",
                "noaux_tc group scores need at least two experts per filtered group",
            ));
        }
        let exposed = self
            .selected_group_count
            .checked_mul(experts_per_group)
            .ok_or(KimiK3MoeError::Overflow {
                expression: "selected_group_count * experts_per_group",
            })?;
        if group_filtering && exposed < self.top_k {
            return Err(invalid_geometry(
                "selected_group_count",
                format!("selected groups expose {exposed} experts, fewer than top_k"),
            ));
        }
        Ok(())
    }
}

/// Full scalar LatentMoE geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentMoeGeometry {
    pub hidden_size: usize,
    pub latent_size: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_count: usize,
    pub routing: NoAuxTcConfig,
    pub rms_norm_epsilon: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
}

impl LatentMoeGeometry {
    pub const KIMI_K3: Self = Self {
        hidden_size: 7_168,
        latent_size: 3_584,
        expert_intermediate_size: 3_072,
        shared_expert_count: 2,
        routing: NoAuxTcConfig::KIMI_K3,
        rms_norm_epsilon: 1e-5,
        situ_beta: KIMI_K3_SITU_BETA,
        situ_linear_beta: KIMI_K3_SITU_LINEAR_BETA,
    };

    pub fn shared_intermediate_size(self) -> Result<usize, KimiK3MoeError> {
        self.expert_intermediate_size
            .checked_mul(self.shared_expert_count)
            .ok_or(KimiK3MoeError::Overflow {
                expression: "expert_intermediate_size * shared_expert_count",
            })
    }

    pub fn validate(self) -> Result<(), KimiK3MoeError> {
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("latent_size", self.latent_size),
            ("expert_intermediate_size", self.expert_intermediate_size),
            ("shared_expert_count", self.shared_expert_count),
        ] {
            if value == 0 {
                return Err(invalid_geometry(field, "must be non-zero"));
            }
        }
        self.routing.validate()?;
        self.shared_intermediate_size()?;
        for (field, value) in [
            ("rms_norm_epsilon", self.rms_norm_epsilon),
            ("situ_beta", self.situ_beta),
            ("situ_linear_beta", self.situ_linear_beta),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid_geometry(
                    field,
                    "must be finite and strictly positive",
                ));
            }
        }
        Ok(())
    }
}

/// One selected routed expert, in deterministic mixture order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteChoice {
    pub expert: usize,
    /// `sigmoid(logit)`, before correction bias or normalization.
    pub unbiased_score: f32,
    /// `unbiased_score + correction_bias`, used only for expert selection.
    pub selection_score: f32,
    /// Unbiased score after top-k normalization and routed scaling.
    pub weight: f32,
}

/// Every matrix-vector projection used by [`latent_moe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeProjection {
    Router,
    RoutedDown,
    RoutedExpertW1 { expert: usize },
    RoutedExpertW3 { expert: usize },
    RoutedExpertW2 { expert: usize },
    RoutedUp,
    SharedW1,
    SharedW3,
    SharedW2,
}

impl fmt::Display for MoeProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Router => f.write_str("router"),
            Self::RoutedDown => f.write_str("routed latent down projection"),
            Self::RoutedExpertW1 { expert } => write!(f, "routed expert {expert} w1"),
            Self::RoutedExpertW3 { expert } => write!(f, "routed expert {expert} w3"),
            Self::RoutedExpertW2 { expert } => write!(f, "routed expert {expert} w2"),
            Self::RoutedUp => f.write_str("routed latent up projection"),
            Self::SharedW1 => f.write_str("shared expert w1"),
            Self::SharedW3 => f.write_str("shared expert w3"),
            Self::SharedW2 => f.write_str("shared expert w2"),
        }
    }
}

/// Projection injection point for dense test matrices or streamed checkpoint weights.
///
/// `output` is already sized for the requested checkpoint matrix and is initialized to
/// NaN.  Leaving any element unwritten is therefore detected before it can contaminate a
/// later stage.  Implementations may reuse internal caches, but this API never lends them a
/// caller-owned result buffer.
pub trait LatentMoeProjector {
    type Error: fmt::Display;

    fn project(
        &mut self,
        projection: MoeProjection,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum KimiK3MoeError {
    Shape {
        value: String,
        expected: usize,
        got: usize,
    },
    InvalidGeometry {
        field: &'static str,
        reason: String,
    },
    NonFinite {
        value: String,
        index: usize,
    },
    Overflow {
        expression: &'static str,
    },
    Allocation {
        value: &'static str,
        elements: usize,
    },
    Projection {
        projection: MoeProjection,
        reason: String,
    },
    Math(KimiK3MathError),
}

impl fmt::Display for KimiK3MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                value,
                expected,
                got,
            } => write!(
                f,
                "Kimi K3 LatentMoE: {value} has length {got}, expected {expected}"
            ),
            Self::InvalidGeometry { field, reason } => {
                write!(f, "Kimi K3 LatentMoE: invalid {field}: {reason}")
            }
            Self::NonFinite { value, index } => {
                write!(f, "Kimi K3 LatentMoE: {value}[{index}] is NaN or infinity")
            }
            Self::Overflow { expression } => write!(
                f,
                "Kimi K3 LatentMoE: integer overflow while computing {expression}"
            ),
            Self::Allocation { value, elements } => write!(
                f,
                "Kimi K3 LatentMoE: cannot allocate {elements} elements for {value}"
            ),
            Self::Projection { projection, reason } => {
                write!(f, "Kimi K3 LatentMoE: {projection} failed: {reason}")
            }
            Self::Math(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for KimiK3MoeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KimiK3MathError> for KimiK3MoeError {
    fn from(value: KimiK3MathError) -> Self {
        Self::Math(value)
    }
}

/// Complete single-token scalar result.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentMoeOutput {
    /// Routed and shared branches added together.
    pub hidden: Vec<f32>,
    /// The normalized latent aggregate after its H <- 3,584 up projection.
    pub routed_hidden: Vec<f32>,
    /// The unweighted full-width shared-expert branch.
    pub shared_hidden: Vec<f32>,
    pub routes: Vec<RouteChoice>,
}

/// Strict K3 `noaux_tc` routing from already-computed full-width router logits.
///
/// Correction bias participates only in ranking.  Returned mixture weights always use
/// the unbiased sigmoid scores, normalized with an additive `1e-20`, then multiplied by
/// `routed_scaling_factor`.  Ties are resolved by lower group/expert ID, and choices are
/// returned from highest selection score to lowest so latent accumulation is reproducible.
pub fn route_noaux_tc(
    logits: &[f32],
    correction_bias: &[f32],
    config: NoAuxTcConfig,
) -> Result<Vec<RouteChoice>, KimiK3MoeError> {
    config.validate()?;
    expect_len("router logits", config.expert_count, logits.len())?;
    expect_len(
        "router correction bias",
        config.expert_count,
        correction_bias.len(),
    )?;
    validate_finite("router logits", logits)?;
    validate_finite("router correction bias", correction_bias)?;

    let mut unbiased = allocate_f32("unbiased router scores", config.expert_count, 0.0)?;
    let mut selection = allocate_f32("biased router scores", config.expert_count, 0.0)?;
    for expert in 0..config.expert_count {
        // This is the float32 sigmoid used by the released eager router and the C oracle.
        // Overflow in exp(-logit) is intentional and yields a score of exactly zero.
        let score = 1.0f32 / (1.0f32 + (-logits[expert]).exp());
        let choice = score + correction_bias[expert];
        if !score.is_finite() {
            return Err(nonfinite("unbiased router scores", expert));
        }
        if !choice.is_finite() {
            return Err(nonfinite("biased router scores", expert));
        }
        unbiased[expert] = score;
        selection[expert] = choice;
    }

    let experts_per_group = config.expert_count / config.expert_group_count;
    let group_filtering =
        config.expert_group_count > 1 && config.expert_group_count > config.selected_group_count;
    let mut allowed = allocate_bool("expert group mask", config.expert_count, !group_filtering)?;

    if group_filtering {
        let mut groups = allocate_pairs("group ranking", config.expert_group_count)?;
        for group in 0..config.expert_group_count {
            let start = group
                .checked_mul(experts_per_group)
                .ok_or(KimiK3MoeError::Overflow {
                    expression: "group * experts_per_group",
                })?;
            let end = start
                .checked_add(experts_per_group)
                .ok_or(KimiK3MoeError::Overflow {
                    expression: "group start + experts_per_group",
                })?;
            let mut first = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for &score in &selection[start..end] {
                if score > first {
                    second = first;
                    first = score;
                } else if score > second {
                    second = score;
                }
            }
            let score = first + second;
            if !score.is_finite() {
                return Err(nonfinite("noaux_tc group scores", group));
            }
            groups.push((group, score));
        }
        groups.sort_by(|(left_id, left), (right_id, right)| {
            right.total_cmp(left).then_with(|| left_id.cmp(right_id))
        });
        for &(group, _) in groups.iter().take(config.selected_group_count) {
            let start = group
                .checked_mul(experts_per_group)
                .ok_or(KimiK3MoeError::Overflow {
                    expression: "selected group * experts_per_group",
                })?;
            let end = start
                .checked_add(experts_per_group)
                .ok_or(KimiK3MoeError::Overflow {
                    expression: "selected group start + experts_per_group",
                })?;
            allowed[start..end].fill(true);
        }
    }

    let allowed_count = allowed.iter().filter(|&&value| value).count();
    let mut ranking = allocate_indices("expert ranking", allowed_count)?;
    ranking.extend((0..config.expert_count).filter(|&expert| allowed[expert]));
    ranking.sort_by(|&left, &right| {
        selection[right]
            .total_cmp(&selection[left])
            .then_with(|| left.cmp(&right))
    });
    if ranking.len() < config.top_k {
        return Err(invalid_geometry(
            "selected_group_count",
            format!(
                "selected groups expose {} experts, fewer than top_k={}",
                ranking.len(),
                config.top_k
            ),
        ));
    }
    ranking.truncate(config.top_k);

    // Match the scalar oracle's stable reduction: accumulate selected f32 scores in f64,
    // cast the reciprocal once, then apply it in deterministic route order.
    let denominator = ranking.iter().try_fold(0.0f64, |sum, &expert| {
        let sum = sum + f64::from(unbiased[expert]);
        if sum.is_finite() {
            Ok(sum)
        } else {
            Err(nonfinite("top-k router score sum", expert))
        }
    })? + ROUTE_NORMALIZATION_EPSILON;
    let inverse = (1.0 / denominator) as f32;
    if !inverse.is_finite() {
        return Err(nonfinite("top-k router normalization", 0));
    }

    let mut routes = allocate_routes(config.top_k)?;
    for expert in ranking {
        let weight = unbiased[expert] * inverse * config.routed_scaling_factor;
        if !weight.is_finite() {
            return Err(nonfinite("normalized route weights", expert));
        }
        routes.push(RouteChoice {
            expert,
            unbiased_score: unbiased[expert],
            selection_score: selection[expert],
            weight,
        });
    }
    Ok(routes)
}

/// Applies the released `routed_expert_norm` to a routed latent aggregate.
///
/// The sum of squares uses an f64 accumulator, while the reciprocal RMS and output are
/// f32, matching the scalar oracle.  Both inputs are immutable, so validation failures
/// cannot leave a partially normalized caller buffer.
pub fn routed_expert_rms_norm(
    aggregate: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, KimiK3MoeError> {
    if aggregate.is_empty() {
        return Err(invalid_geometry("latent_size", "must be non-zero"));
    }
    expect_len("routed expert norm weight", aggregate.len(), weight.len())?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(invalid_geometry(
            "rms_norm_epsilon",
            "must be finite and strictly positive",
        ));
    }
    validate_finite("routed expert aggregate", aggregate)?;
    validate_finite("routed expert norm weight", weight)?;

    let square_sum = aggregate
        .iter()
        .enumerate()
        .try_fold(0.0f64, |sum, (index, &value)| {
            let value = f64::from(value);
            let sum = sum + value * value;
            if sum.is_finite() {
                Ok(sum)
            } else {
                Err(nonfinite("routed expert sum of squares", index))
            }
        })?;
    let inverse = (1.0 / (square_sum / aggregate.len() as f64 + f64::from(epsilon)).sqrt()) as f32;
    if !inverse.is_finite() {
        return Err(nonfinite("routed expert inverse RMS", 0));
    }

    let mut output = allocate_f32("routed expert normalized output", aggregate.len(), 0.0)?;
    for index in 0..aggregate.len() {
        let value = weight[index] * aggregate[index] * inverse;
        if !value.is_finite() {
            return Err(nonfinite("routed expert normalized output", index));
        }
        output[index] = value;
    }
    Ok(output)
}

/// Executes one token of K3 LatentMoE using injected matrix-vector projections.
///
/// All dimensions and caller-owned inputs are validated before the first projection.  Each
/// intermediate is private and checked before use; therefore an error never exposes a partial
/// model output.  Routed experts are evaluated and accumulated in the deterministic order
/// returned by [`route_noaux_tc`].
pub fn latent_moe<P: LatentMoeProjector>(
    input: &[f32],
    correction_bias: &[f32],
    routed_norm_weight: &[f32],
    geometry: LatentMoeGeometry,
    projector: &mut P,
) -> Result<LatentMoeOutput, KimiK3MoeError> {
    geometry.validate()?;
    expect_len("input hidden state", geometry.hidden_size, input.len())?;
    expect_len(
        "router correction bias",
        geometry.routing.expert_count,
        correction_bias.len(),
    )?;
    expect_len(
        "routed expert norm weight",
        geometry.latent_size,
        routed_norm_weight.len(),
    )?;
    validate_finite("input hidden state", input)?;
    validate_finite("router correction bias", correction_bias)?;
    validate_finite("routed expert norm weight", routed_norm_weight)?;

    // The router deliberately observes the original H-wide input before latent projection.
    let logits = project(
        projector,
        MoeProjection::Router,
        input,
        geometry.hidden_size,
        geometry.routing.expert_count,
    )?;
    let routes = route_noaux_tc(&logits, correction_bias, geometry.routing)?;
    let latent = project(
        projector,
        MoeProjection::RoutedDown,
        input,
        geometry.hidden_size,
        geometry.latent_size,
    )?;

    let mut aggregate = allocate_f32("routed latent aggregate", geometry.latent_size, 0.0)?;
    for route in &routes {
        let gate = project(
            projector,
            MoeProjection::RoutedExpertW1 {
                expert: route.expert,
            },
            &latent,
            geometry.latent_size,
            geometry.expert_intermediate_size,
        )?;
        let up = project(
            projector,
            MoeProjection::RoutedExpertW3 {
                expert: route.expert,
            },
            &latent,
            geometry.latent_size,
            geometry.expert_intermediate_size,
        )?;
        let activated = situ_glu(&gate, &up, geometry.situ_beta, geometry.situ_linear_beta)?;
        let expert_output = project(
            projector,
            MoeProjection::RoutedExpertW2 {
                expert: route.expert,
            },
            &activated,
            geometry.expert_intermediate_size,
            geometry.latent_size,
        )?;
        for index in 0..geometry.latent_size {
            let term = route.weight * expert_output[index];
            let value = aggregate[index] + term;
            if !term.is_finite() || !value.is_finite() {
                return Err(nonfinite("routed latent aggregate", index));
            }
            aggregate[index] = value;
        }
    }

    let normalized =
        routed_expert_rms_norm(&aggregate, routed_norm_weight, geometry.rms_norm_epsilon)?;
    let routed_hidden = project(
        projector,
        MoeProjection::RoutedUp,
        &normalized,
        geometry.latent_size,
        geometry.hidden_size,
    )?;

    // Two K3 shared experts are represented by one H -> (2 * I) -> H MLP and have no
    // router weight or routed scaling factor.
    let shared_intermediate = geometry.shared_intermediate_size()?;
    let shared_gate = project(
        projector,
        MoeProjection::SharedW1,
        input,
        geometry.hidden_size,
        shared_intermediate,
    )?;
    let shared_up = project(
        projector,
        MoeProjection::SharedW3,
        input,
        geometry.hidden_size,
        shared_intermediate,
    )?;
    let shared_activated = situ_glu(
        &shared_gate,
        &shared_up,
        geometry.situ_beta,
        geometry.situ_linear_beta,
    )?;
    let shared_hidden = project(
        projector,
        MoeProjection::SharedW2,
        &shared_activated,
        shared_intermediate,
        geometry.hidden_size,
    )?;

    let mut hidden = allocate_f32("LatentMoE output", geometry.hidden_size, 0.0)?;
    for index in 0..geometry.hidden_size {
        let value = routed_hidden[index] + shared_hidden[index];
        if !value.is_finite() {
            return Err(nonfinite("LatentMoE output", index));
        }
        hidden[index] = value;
    }

    Ok(LatentMoeOutput {
        hidden,
        routed_hidden,
        shared_hidden,
        routes,
    })
}

fn project<P: LatentMoeProjector>(
    projector: &mut P,
    projection: MoeProjection,
    input: &[f32],
    expected_input: usize,
    expected_output: usize,
) -> Result<Vec<f32>, KimiK3MoeError> {
    expect_len(format!("{projection} input"), expected_input, input.len())?;
    validate_finite(format!("{projection} input"), input)?;
    let mut output = allocate_f32("projection output", expected_output, f32::NAN)?;
    projector
        .project(projection, input, &mut output)
        .map_err(|error| KimiK3MoeError::Projection {
            projection,
            reason: error.to_string(),
        })?;
    validate_finite(format!("{projection} output"), &output)?;
    Ok(output)
}

fn expect_len(value: impl Into<String>, expected: usize, got: usize) -> Result<(), KimiK3MoeError> {
    if got == expected {
        Ok(())
    } else {
        Err(KimiK3MoeError::Shape {
            value: value.into(),
            expected,
            got,
        })
    }
}

fn validate_finite(value: impl Into<String>, values: &[f32]) -> Result<(), KimiK3MoeError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(nonfinite(value, index))
    } else {
        Ok(())
    }
}

fn invalid_geometry(field: &'static str, reason: impl Into<String>) -> KimiK3MoeError {
    KimiK3MoeError::InvalidGeometry {
        field,
        reason: reason.into(),
    }
}

fn nonfinite(value: impl Into<String>, index: usize) -> KimiK3MoeError {
    KimiK3MoeError::NonFinite {
        value: value.into(),
        index,
    }
}

fn allocate_f32(
    value: &'static str,
    elements: usize,
    fill: f32,
) -> Result<Vec<f32>, KimiK3MoeError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| KimiK3MoeError::Allocation { value, elements })?;
    output.resize(elements, fill);
    Ok(output)
}

fn allocate_bool(
    value: &'static str,
    elements: usize,
    fill: bool,
) -> Result<Vec<bool>, KimiK3MoeError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| KimiK3MoeError::Allocation { value, elements })?;
    output.resize(elements, fill);
    Ok(output)
}

fn allocate_indices(value: &'static str, elements: usize) -> Result<Vec<usize>, KimiK3MoeError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| KimiK3MoeError::Allocation { value, elements })?;
    Ok(output)
}

fn allocate_pairs(
    value: &'static str,
    elements: usize,
) -> Result<Vec<(usize, f32)>, KimiK3MoeError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| KimiK3MoeError::Allocation { value, elements })?;
    Ok(output)
}

fn allocate_routes(elements: usize) -> Result<Vec<RouteChoice>, KimiK3MoeError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| KimiK3MoeError::Allocation {
            value: "route choices",
            elements,
        })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_geometry() -> LatentMoeGeometry {
        LatentMoeGeometry {
            hidden_size: 2,
            latent_size: 2,
            expert_intermediate_size: 1,
            shared_expert_count: 2,
            routing: NoAuxTcConfig {
                expert_count: 3,
                top_k: 2,
                expert_group_count: 1,
                selected_group_count: 1,
                routed_scaling_factor: 1.0,
            },
            rms_norm_epsilon: 1e-5,
            situ_beta: KIMI_K3_SITU_BETA,
            situ_linear_beta: KIMI_K3_SITU_LINEAR_BETA,
        }
    }

    #[test]
    fn official_router_covers_896_experts_and_has_stable_ties() {
        let logits = vec![0.0; 896];
        let mut bias = vec![0.0; 896];
        bias[895] = 1.0;

        let routes = route_noaux_tc(&logits, &bias, NoAuxTcConfig::KIMI_K3).unwrap();
        assert_eq!(routes.len(), 16);
        assert_eq!(routes[0].expert, 895);
        assert_eq!(
            routes[1..]
                .iter()
                .map(|route| route.expert)
                .collect::<Vec<_>>(),
            (0..15).collect::<Vec<_>>()
        );
        for route in &routes {
            assert_eq!(route.unbiased_score, 0.5);
            assert!((route.weight - 1.0 / 16.0).abs() < 1e-7);
        }
        assert_eq!(routes[0].selection_score, 1.5);
    }

    #[test]
    fn grouped_noaux_tc_uses_top_two_biased_scores_but_unbiased_mixture() {
        let config = NoAuxTcConfig {
            expert_count: 8,
            top_k: 2,
            expert_group_count: 2,
            selected_group_count: 1,
            routed_scaling_factor: 2.0,
        };
        // Bias makes group 1 win even though its unbiased logits are lower.
        let logits = [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0];
        let bias = [0.0, 0.0, 0.0, 0.0, 10.0, 9.0, 0.0, 0.0];
        let routes = route_noaux_tc(&logits, &bias, config).unwrap();
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            [4, 5]
        );
        let score4 = 0.5f32;
        let score5 = 1.0 / (1.0 + 1.0f32.exp());
        let inverse = (1.0 / (f64::from(score4 + score5) + 1e-20)) as f32;
        assert!((routes[0].weight - score4 * inverse * 2.0).abs() < 1e-7);
        assert!((routes[1].weight - score5 * inverse * 2.0).abs() < 1e-7);
    }

    #[test]
    fn normalization_epsilon_keeps_underflowed_routes_finite() {
        let config = NoAuxTcConfig {
            expert_count: 4,
            top_k: 2,
            expert_group_count: 1,
            selected_group_count: 1,
            routed_scaling_factor: 1.0,
        };
        let routes = route_noaux_tc(&[-1_000.0; 4], &[0.0; 4], config).unwrap();
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            [0, 1]
        );
        assert!(routes.iter().all(|route| route.weight == 0.0));
    }

    #[derive(Default)]
    struct TinyProjector {
        calls: Vec<MoeProjection>,
        poison: Option<MoeProjection>,
    }

    impl LatentMoeProjector for TinyProjector {
        type Error = &'static str;

        fn project(
            &mut self,
            projection: MoeProjection,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), Self::Error> {
            self.calls.push(projection);
            if self.poison == Some(projection) {
                output.fill(f32::NAN);
                return Ok(());
            }
            match projection {
                MoeProjection::Router => output.copy_from_slice(&[0.0, input[0], input[1]]),
                MoeProjection::RoutedDown => output.copy_from_slice(input),
                MoeProjection::RoutedExpertW1 { expert: 0 } => output[0] = input[0],
                MoeProjection::RoutedExpertW3 { expert: 0 } => output[0] = input[1],
                MoeProjection::RoutedExpertW2 { expert: 0 } => {
                    output.copy_from_slice(&[input[0], 2.0 * input[0]]);
                }
                MoeProjection::RoutedExpertW1 { expert: 2 } => output[0] = input[1],
                MoeProjection::RoutedExpertW3 { expert: 2 } => output[0] = input[0],
                MoeProjection::RoutedExpertW2 { expert: 2 } => {
                    output.copy_from_slice(&[-input[0], input[0]]);
                }
                MoeProjection::RoutedExpertW1 { .. }
                | MoeProjection::RoutedExpertW3 { .. }
                | MoeProjection::RoutedExpertW2 { .. } => return Err("unexpected expert"),
                MoeProjection::RoutedUp => {
                    output.copy_from_slice(&[input[0] + input[1], 2.0 * input[0] - input[1]]);
                }
                MoeProjection::SharedW1 => output.copy_from_slice(input),
                MoeProjection::SharedW3 => output.copy_from_slice(&[input[1], input[0]]),
                MoeProjection::SharedW2 => {
                    output.copy_from_slice(&[input[0], 2.0 * input[1]]);
                }
            }
            Ok(())
        }
    }

    #[test]
    fn tiny_latent_moe_matches_hand_computed_golden_and_stage_order() {
        assert_eq!(KIMI_K3_SITU_BETA, 4.0);
        assert_eq!(KIMI_K3_SITU_LINEAR_BETA, 25.0);

        let mut projector = TinyProjector::default();
        let output = latent_moe(
            &[1.0, 2.0],
            &[3.0, 0.0, 0.0],
            &[2.0, 0.5],
            tiny_geometry(),
            &mut projector,
        )
        .unwrap();

        assert_eq!(
            output
                .routes
                .iter()
                .map(|route| route.expert)
                .collect::<Vec<_>>(),
            [0, 2]
        );
        // Independent scalar evaluation of the official equations (beta=4, linear=25).
        let routed_golden = [-0.00282675, -2.0631323];
        let shared_golden = [1.4293511, 3.254_516];
        let output_golden = [1.4265244, 1.1913837];
        for (got, expected) in output.routed_hidden.iter().zip(routed_golden) {
            assert!((got - expected).abs() < 2e-5, "{got} != {expected}");
        }
        for (got, expected) in output.shared_hidden.iter().zip(shared_golden) {
            assert!((got - expected).abs() < 2e-5, "{got} != {expected}");
        }
        for (got, expected) in output.hidden.iter().zip(output_golden) {
            assert!((got - expected).abs() < 2e-5, "{got} != {expected}");
        }
        assert_eq!(
            projector.calls,
            [
                MoeProjection::Router,
                MoeProjection::RoutedDown,
                MoeProjection::RoutedExpertW1 { expert: 0 },
                MoeProjection::RoutedExpertW3 { expert: 0 },
                MoeProjection::RoutedExpertW2 { expert: 0 },
                MoeProjection::RoutedExpertW1 { expert: 2 },
                MoeProjection::RoutedExpertW3 { expert: 2 },
                MoeProjection::RoutedExpertW2 { expert: 2 },
                MoeProjection::RoutedUp,
                MoeProjection::SharedW1,
                MoeProjection::SharedW3,
                MoeProjection::SharedW2,
            ]
        );
    }

    #[test]
    fn invalid_inputs_fail_before_projector_and_nan_output_stops_transaction() {
        let mut projector = TinyProjector::default();
        let error = latent_moe(
            &[f32::NAN, 2.0],
            &[0.0; 3],
            &[1.0; 2],
            tiny_geometry(),
            &mut projector,
        )
        .unwrap_err();
        assert!(matches!(error, KimiK3MoeError::NonFinite { .. }));
        assert!(projector.calls.is_empty());

        projector.poison = Some(MoeProjection::RoutedExpertW2 { expert: 0 });
        let error = latent_moe(
            &[1.0, 2.0],
            &[3.0, 0.0, 0.0],
            &[1.0; 2],
            tiny_geometry(),
            &mut projector,
        )
        .unwrap_err();
        assert!(matches!(error, KimiK3MoeError::NonFinite { .. }));
        assert_eq!(
            projector.calls.last(),
            Some(&MoeProjection::RoutedExpertW2 { expert: 0 })
        );
    }

    #[test]
    fn dimensions_and_shared_width_overflow_are_rejected() {
        let mut geometry = tiny_geometry();
        geometry.expert_intermediate_size = usize::MAX;
        geometry.shared_expert_count = 2;
        assert!(matches!(
            geometry.validate(),
            Err(KimiK3MoeError::Overflow { .. })
        ));

        let error = route_noaux_tc(
            &[0.0; 2],
            &[0.0; 3],
            NoAuxTcConfig {
                expert_count: 3,
                top_k: 2,
                expert_group_count: 1,
                selected_group_count: 1,
                routed_scaling_factor: 1.0,
            },
        )
        .unwrap_err();
        assert!(matches!(error, KimiK3MoeError::Shape { .. }));
    }
}
