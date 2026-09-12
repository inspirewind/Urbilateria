//! Qwen3.8 sparse-MoE routing and single-token reference forward.
//!
//! The released router applies FP32 softmax over every expert, chooses top-k,
//! and renormalizes only the chosen probabilities. Routed experts are combined
//! with those normalized weights. The independently computed shared SwiGLU
//! expert is gated by `sigmoid(shared_expert_gate(x))` and added last.

use super::expert::{Qwen38Expert, Qwen38ExpertError};
use super::math::{linear, round_to_bf16, uses_bf16_output};
use crate::model::{WeightError, WeightMatrix};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen38MoeGeometry {
    pub top_k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen38Route {
    pub expert: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen38MoeOutput {
    pub hidden: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub routes: Vec<Qwen38Route>,
}

#[derive(Debug)]
pub enum Qwen38MoeError {
    Weight {
        projection: &'static str,
        source: WeightError,
    },
    Expert {
        expert: Option<usize>,
        source: Qwen38ExpertError,
    },
    InvalidGeometry(String),
    InvalidShape(String),
    NonFinite {
        operation: &'static str,
        index: usize,
    },
}

impl fmt::Display for Qwen38MoeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight { projection, source } => {
                write!(formatter, "Qwen3.8 MoE {projection}: {source}")
            }
            Self::Expert {
                expert: Some(expert),
                source,
            } => write!(formatter, "Qwen3.8 routed expert {expert}: {source}"),
            Self::Expert {
                expert: None,
                source,
            } => write!(formatter, "Qwen3.8 shared expert: {source}"),
            Self::InvalidGeometry(reason) => {
                write!(formatter, "invalid Qwen3.8 MoE geometry: {reason}")
            }
            Self::InvalidShape(reason) => {
                write!(formatter, "invalid Qwen3.8 MoE shape: {reason}")
            }
            Self::NonFinite { operation, index } => write!(
                formatter,
                "Qwen3.8 MoE {operation} produced NaN or infinity at index {index}"
            ),
        }
    }
}

impl std::error::Error for Qwen38MoeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight { source, .. } => Some(source),
            Self::Expert { source, .. } => Some(source),
            Self::InvalidGeometry(_) | Self::InvalidShape(_) | Self::NonFinite { .. } => None,
        }
    }
}

/// Applies the released router equation to already-computed FP32 logits.
///
/// Equal probabilities are resolved by ascending expert ID. Real checkpoints
/// almost never tie, but this rule makes tiny fixtures and CPU runs reproducible.
pub fn route_softmax_top_k(
    logits: &[f32],
    top_k: usize,
) -> Result<Vec<Qwen38Route>, Qwen38MoeError> {
    if logits.is_empty() {
        return Err(Qwen38MoeError::InvalidGeometry(
            "at least one routed expert is required".to_owned(),
        ));
    }
    if top_k == 0 || top_k > logits.len() {
        return Err(Qwen38MoeError::InvalidGeometry(format!(
            "top_k={top_k} is outside 1..={} experts",
            logits.len()
        )));
    }
    validate_finite("router logits", logits)?;

    // Keep every intermediate in f32, matching `softmax(..., dtype=torch.float)`.
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = Vec::with_capacity(logits.len());
    let mut denominator = 0.0f32;
    for &logit in logits {
        let probability = (logit - maximum).exp();
        denominator += probability;
        probabilities.push(probability);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Qwen38MoeError::NonFinite {
            operation: "router softmax normalization",
            index: 0,
        });
    }
    for probability in &mut probabilities {
        *probability /= denominator;
    }
    validate_finite("router probabilities", &probabilities)?;

    let mut ranking = (0..probabilities.len()).collect::<Vec<_>>();
    ranking.sort_by(|&left, &right| {
        probabilities[right]
            .total_cmp(&probabilities[left])
            .then_with(|| left.cmp(&right))
    });
    ranking.truncate(top_k);

    let mut selected_sum = 0.0f32;
    for &expert in &ranking {
        selected_sum += probabilities[expert];
    }
    if !selected_sum.is_finite() || selected_sum <= 0.0 {
        return Err(Qwen38MoeError::NonFinite {
            operation: "top-k probability normalization",
            index: 0,
        });
    }

    let mut routes = Vec::with_capacity(top_k);
    for expert in ranking {
        let weight = probabilities[expert] / selected_sum;
        if !weight.is_finite() {
            return Err(Qwen38MoeError::NonFinite {
                operation: "top-k route weight",
                index: expert,
            });
        }
        routes.push(Qwen38Route { expert, weight });
    }
    Ok(routes)
}

/// A resident Qwen3.8 sparse-MoE layer suitable for tiny/reference execution.
///
/// Production runtimes can use [`route_softmax_top_k`] and [`Qwen38Expert`]
/// independently when routed experts are loaded through an LRU instead.
#[derive(Debug, Clone)]
pub struct Qwen38Moe {
    router: WeightMatrix,
    routed_experts: Vec<Qwen38Expert>,
    shared_expert: Qwen38Expert,
    shared_expert_gate: WeightMatrix,
    geometry: Qwen38MoeGeometry,
}

/// Descriptive alias used by decoder blocks that store one MoE per layer.
pub type Qwen38MoeLayer = Qwen38Moe;

impl Qwen38Moe {
    pub fn new(
        router: WeightMatrix,
        routed_experts: Vec<Qwen38Expert>,
        shared_expert: Qwen38Expert,
        shared_expert_gate: WeightMatrix,
        geometry: Qwen38MoeGeometry,
    ) -> Result<Self, Qwen38MoeError> {
        if routed_experts.is_empty() {
            return Err(Qwen38MoeError::InvalidGeometry(
                "at least one routed expert is required".to_owned(),
            ));
        }
        if geometry.top_k == 0 || geometry.top_k > routed_experts.len() {
            return Err(Qwen38MoeError::InvalidGeometry(format!(
                "top_k={} is outside 1..={} experts",
                geometry.top_k,
                routed_experts.len()
            )));
        }

        let hidden_size = router.cols();
        if hidden_size == 0 || router.rows() != routed_experts.len() {
            return Err(Qwen38MoeError::InvalidShape(format!(
                "router is [{}, {}], but there are {} routed experts",
                router.rows(),
                hidden_size,
                routed_experts.len()
            )));
        }
        if routed_experts
            .iter()
            .any(|expert| expert.hidden_size() != hidden_size)
        {
            return Err(Qwen38MoeError::InvalidShape(format!(
                "router hidden size {hidden_size} does not match every routed expert"
            )));
        }
        if shared_expert.hidden_size() != hidden_size {
            return Err(Qwen38MoeError::InvalidShape(format!(
                "shared expert hidden size {} differs from router hidden size {hidden_size}",
                shared_expert.hidden_size()
            )));
        }
        if shared_expert_gate.rows() != 1 || shared_expert_gate.cols() != hidden_size {
            return Err(Qwen38MoeError::InvalidShape(format!(
                "shared_expert_gate is [{}, {}], expected [1, {hidden_size}]",
                shared_expert_gate.rows(),
                shared_expert_gate.cols()
            )));
        }

        Ok(Self {
            router,
            routed_experts,
            shared_expert,
            shared_expert_gate,
            geometry,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.router.cols()
    }

    pub fn expert_count(&self) -> usize {
        self.routed_experts.len()
    }

    pub fn top_k(&self) -> usize {
        self.geometry.top_k
    }

    pub fn resident_bytes(&self) -> usize {
        self.routed_experts.iter().fold(
            self.router
                .resident_bytes()
                .saturating_add(self.shared_expert.resident_bytes())
                .saturating_add(self.shared_expert_gate.resident_bytes()),
            |total, expert| total.saturating_add(expert.resident_bytes()),
        )
    }

    /// Returns the full-width router logits and normalized selected routes.
    pub fn route(&self, input: &[f32]) -> Result<(Vec<f32>, Vec<Qwen38Route>), Qwen38MoeError> {
        self.validate_input(input)?;
        let logits = linear(&self.router, input).map_err(|source| Qwen38MoeError::Weight {
            projection: "router",
            source,
        })?;
        if logits.len() != self.expert_count() {
            return Err(Qwen38MoeError::InvalidShape(format!(
                "router returned {} logits, expected {}",
                logits.len(),
                self.expert_count()
            )));
        }
        validate_finite("router logits", &logits)?;
        let mut routes = route_softmax_top_k(&logits, self.geometry.top_k)?;
        if uses_bf16_output(&self.router) {
            for route in &mut routes {
                route.weight =
                    round_to_bf16(route.weight).map_err(|source| Qwen38MoeError::Weight {
                        projection: "router weight cast",
                        source,
                    })?;
            }
        }
        Ok((logits, routes))
    }

    pub fn forward(&self, input: &[f32]) -> Result<Qwen38MoeOutput, Qwen38MoeError> {
        let (router_logits, routes) = self.route(input)?;
        let mut hidden = vec![0.0f32; self.hidden_size()];

        // The eager reference iterates hit experts by ascending expert ID. Preserve
        // the public route ranking while matching that accumulation order.
        let mut execution_order = routes.clone();
        execution_order.sort_by_key(|route| route.expert);
        for route in execution_order {
            let expert_output =
                self.routed_experts[route.expert]
                    .forward(input)
                    .map_err(|source| Qwen38MoeError::Expert {
                        expert: Some(route.expert),
                        source,
                    })?;
            for (index, (accumulator, value)) in hidden.iter_mut().zip(expert_output).enumerate() {
                let contribution = route.weight * value;
                let contribution = if uses_bf16_output(&self.router) {
                    round_to_bf16(contribution).map_err(|source| Qwen38MoeError::Weight {
                        projection: "routed expert weighting cast",
                        source,
                    })?
                } else {
                    contribution
                };
                *accumulator += contribution;
                if uses_bf16_output(&self.router) {
                    *accumulator =
                        round_to_bf16(*accumulator).map_err(|source| Qwen38MoeError::Weight {
                            projection: "routed expert accumulation cast",
                            source,
                        })?;
                }
                if !contribution.is_finite() || !accumulator.is_finite() {
                    return Err(Qwen38MoeError::NonFinite {
                        operation: "routed expert accumulation",
                        index,
                    });
                }
            }
        }
        let shared =
            self.shared_expert
                .forward(input)
                .map_err(|source| Qwen38MoeError::Expert {
                    expert: None,
                    source,
                })?;
        let gate =
            linear(&self.shared_expert_gate, input).map_err(|source| Qwen38MoeError::Weight {
                projection: "shared_expert_gate",
                source,
            })?;
        if gate.len() != 1 {
            return Err(Qwen38MoeError::InvalidShape(format!(
                "shared_expert_gate returned {} values, expected 1",
                gate.len()
            )));
        }
        validate_finite("shared expert gate", &gate)?;
        let shared_scale = sigmoid(gate[0]);
        let shared_scale = if uses_bf16_output(&self.shared_expert_gate) {
            round_to_bf16(shared_scale).map_err(|source| Qwen38MoeError::Weight {
                projection: "shared expert sigmoid cast",
                source,
            })?
        } else {
            shared_scale
        };
        if !shared_scale.is_finite() {
            return Err(Qwen38MoeError::NonFinite {
                operation: "shared expert sigmoid gate",
                index: 0,
            });
        }
        for (index, (accumulator, value)) in hidden.iter_mut().zip(shared).enumerate() {
            let contribution = shared_scale * value;
            let contribution = if uses_bf16_output(&self.shared_expert_gate) {
                round_to_bf16(contribution).map_err(|source| Qwen38MoeError::Weight {
                    projection: "shared expert weighting cast",
                    source,
                })?
            } else {
                contribution
            };
            *accumulator += contribution;
            if uses_bf16_output(&self.shared_expert_gate) {
                *accumulator =
                    round_to_bf16(*accumulator).map_err(|source| Qwen38MoeError::Weight {
                        projection: "MoE output cast",
                        source,
                    })?;
            }
            if !contribution.is_finite() || !accumulator.is_finite() {
                return Err(Qwen38MoeError::NonFinite {
                    operation: "shared expert accumulation",
                    index,
                });
            }
        }

        Ok(Qwen38MoeOutput {
            hidden,
            router_logits,
            routes,
        })
    }

    fn validate_input(&self, input: &[f32]) -> Result<(), Qwen38MoeError> {
        if input.len() != self.hidden_size() {
            return Err(Qwen38MoeError::InvalidShape(format!(
                "input has length {}, expected hidden size {}",
                input.len(),
                self.hidden_size()
            )));
        }
        validate_finite("input", input)
    }
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn validate_finite(operation: &'static str, values: &[f32]) -> Result<(), Qwen38MoeError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(Qwen38MoeError::NonFinite { operation, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::silu;
    use crate::model::DenseMatrix;

    fn dense(rows: usize, columns: usize, values: Vec<f32>) -> WeightMatrix {
        WeightMatrix::F32(DenseMatrix::new(rows, columns, values).unwrap())
    }

    fn scalar_expert(scale: f32) -> Qwen38Expert {
        Qwen38Expert::new(
            dense(1, 1, vec![1.0]),
            dense(1, 1, vec![scale]),
            dense(1, 1, vec![1.0]),
        )
        .unwrap()
    }

    #[test]
    fn router_has_stable_tie_break_by_expert_id() {
        let routes = route_softmax_top_k(&[2.0, 2.0, 2.0, 1.0], 2).unwrap();
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            [0, 1]
        );
        assert!((routes[0].weight - 0.5).abs() < 1e-7);
        assert!((routes[1].weight - 0.5).abs() < 1e-7);
    }

    #[test]
    fn topk_probabilities_are_renormalized_after_global_softmax() {
        let routes = route_softmax_top_k(&[4.0f32.ln(), 2.0f32.ln(), 1.0f32.ln()], 2).unwrap();
        assert_eq!(routes[0].expert, 0);
        assert_eq!(routes[1].expert, 1);
        assert!((routes[0].weight - 2.0 / 3.0).abs() < 1e-6);
        assert!((routes[1].weight - 1.0 / 3.0).abs() < 1e-6);
        assert!((routes.iter().map(|route| route.weight).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn forward_combines_routed_experts_then_sigmoid_gated_shared_expert() {
        let layer = Qwen38Moe::new(
            dense(3, 1, vec![2.0, 1.0, 0.0]),
            vec![scalar_expert(1.0), scalar_expert(3.0), scalar_expert(8.0)],
            scalar_expert(4.0),
            dense(1, 1, vec![3.0f32.ln()]),
            Qwen38MoeGeometry { top_k: 2 },
        )
        .unwrap();

        let output = layer.forward(&[1.0]).unwrap();
        let routed_scale = output.routes[0].weight + 3.0 * output.routes[1].weight;
        let expected = silu(1.0) * (routed_scale + 0.75 * 4.0);
        assert!((output.hidden[0] - expected).abs() < 1e-6);
        assert_eq!(output.router_logits, [2.0, 1.0, 0.0]);
        assert_eq!(
            output
                .routes
                .iter()
                .map(|route| route.expert)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn rejects_invalid_geometry_shapes_and_values() {
        let build = |top_k| {
            Qwen38Moe::new(
                dense(2, 1, vec![1.0, 0.0]),
                vec![scalar_expert(1.0), scalar_expert(2.0)],
                scalar_expert(1.0),
                dense(1, 1, vec![0.0]),
                Qwen38MoeGeometry { top_k },
            )
        };
        assert!(matches!(build(0), Err(Qwen38MoeError::InvalidGeometry(_))));

        let layer = build(1).unwrap();
        assert!(matches!(
            layer.forward(&[f32::INFINITY]),
            Err(Qwen38MoeError::NonFinite {
                operation: "input",
                index: 0
            })
        ));
        assert!(matches!(
            layer.forward(&[]),
            Err(Qwen38MoeError::InvalidShape(_))
        ));

        assert!(matches!(
            Qwen38Moe::new(
                dense(2, 1, vec![1.0, 0.0]),
                vec![scalar_expert(1.0), scalar_expert(2.0)],
                scalar_expert(1.0),
                dense(2, 1, vec![0.0, 0.0]),
                Qwen38MoeGeometry { top_k: 1 },
            ),
            Err(Qwen38MoeError::InvalidShape(_))
        ));
    }
}
