//! Faithful `sigmoid + noaux_tc` expert routing used by GLM-5.2.

const NORMALIZATION_EPSILON: f32 = 1e-20;

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct RouteChoice {
    pub expert: usize,
    /// The unbiased sigmoid probability, normalized if requested and then scaled.
    pub weight: f32,
    /// `sigmoid(logit) + correction_bias`, used only to choose expert IDs.
    pub selection_score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    Shape(String),
    InvalidGeometry(String),
    NonFinite,
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(f, "router shape error: {reason}"),
            Self::InvalidGeometry(reason) => write!(f, "router geometry error: {reason}"),
            Self::NonFinite => f.write_str("router logits, bias, or scaling contain NaN/infinity"),
        }
    }
}

impl std::error::Error for RouteError {}

/// Routes one token. The equations match the official eager router. PyTorch leaves `topk`
/// tie ordering unspecified, so this reference deliberately chooses the lower group/expert ID.
#[allow(clippy::too_many_arguments)]
pub fn route_noaux_tc(
    logits: &[f32],
    correction_bias: &[f32],
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    normalize: bool,
    routed_scaling_factor: f32,
) -> Result<Vec<RouteChoice>, RouteError> {
    if logits.len() != correction_bias.len() {
        return Err(RouteError::Shape(format!(
            "{} logits but {} bias values",
            logits.len(),
            correction_bias.len()
        )));
    }
    if logits.is_empty() {
        return Err(RouteError::InvalidGeometry("no experts".to_owned()));
    }
    if logits
        .iter()
        .chain(correction_bias)
        .any(|value| !value.is_finite())
        || !routed_scaling_factor.is_finite()
    {
        return Err(RouteError::NonFinite);
    }
    if routed_scaling_factor <= 0.0 {
        return Err(RouteError::InvalidGeometry(
            "routed_scaling_factor must be positive".to_owned(),
        ));
    }
    if n_group == 0 || logits.len() % n_group != 0 {
        return Err(RouteError::InvalidGeometry(format!(
            "{} experts cannot be divided into {n_group} equal groups",
            logits.len()
        )));
    }
    if top_k == 0 || top_k > logits.len() {
        return Err(RouteError::InvalidGeometry(format!(
            "top_k={top_k} is outside 1..={} ",
            logits.len()
        )));
    }
    if topk_group == 0 || topk_group > n_group {
        return Err(RouteError::InvalidGeometry(format!(
            "topk_group={topk_group} is outside 1..={n_group}"
        )));
    }
    let experts_per_group = logits.len() / n_group;
    if experts_per_group < 2 {
        return Err(RouteError::InvalidGeometry(
            "noaux_tc group scoring requires at least two experts per group".to_owned(),
        ));
    }

    let gates: Vec<f32> = logits
        .iter()
        .map(|&value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let selection: Vec<f32> = gates
        .iter()
        .zip(correction_bias)
        .map(|(&gate, &bias)| gate + bias)
        .collect();

    let mut group_ranking = Vec::with_capacity(n_group);
    for group in 0..n_group {
        let start = group * experts_per_group;
        let end = start + experts_per_group;
        let mut scores = selection[start..end].to_vec();
        scores.sort_by(|left, right| right.total_cmp(left));
        group_ranking.push((group, scores[0] + scores[1]));
    }
    group_ranking.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    let mut allowed = vec![false; logits.len()];
    for &(group, _) in &group_ranking[..topk_group] {
        let start = group * experts_per_group;
        allowed[start..start + experts_per_group].fill(true);
    }

    let mut expert_ranking: Vec<usize> = (0..logits.len()).filter(|&id| allowed[id]).collect();
    expert_ranking.sort_by(|&left, &right| {
        selection[right]
            .total_cmp(&selection[left])
            .then_with(|| left.cmp(&right))
    });
    if expert_ranking.len() < top_k {
        return Err(RouteError::InvalidGeometry(format!(
            "selected groups expose {} experts, fewer than top_k={top_k}",
            expert_ranking.len()
        )));
    }
    expert_ranking.truncate(top_k);

    let denominator: f32 =
        expert_ranking.iter().map(|&id| gates[id]).sum::<f32>() + NORMALIZATION_EPSILON;
    if normalize && !denominator.is_finite() {
        return Err(RouteError::NonFinite);
    }
    Ok(expert_ranking
        .into_iter()
        .map(|expert| {
            let weight = if normalize {
                gates[expert] / denominator
            } else {
                gates[expert]
            } * routed_scaling_factor;
            RouteChoice {
                expert,
                weight,
                selection_score: selection[expert],
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_bias_changes_selection_but_not_gate_weight() {
        let logits = [2.0, 1.0, 0.0, -1.0];
        let unbiased = route_noaux_tc(&logits, &[0.0; 4], 2, 1, 1, false, 1.0).unwrap();
        assert_eq!(
            unbiased.iter().map(|c| c.expert).collect::<Vec<_>>(),
            [0, 1]
        );

        let biased = route_noaux_tc(&logits, &[0.0, 0.0, 2.0, 0.0], 2, 1, 1, false, 1.0).unwrap();
        assert_eq!(biased.iter().map(|c| c.expert).collect::<Vec<_>>(), [2, 0]);
        assert!((biased[0].weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalized_weights_sum_to_routed_scale() {
        let choices = route_noaux_tc(&[3.0, 2.0, 1.0, 0.0], &[0.0; 4], 3, 1, 1, true, 2.5).unwrap();
        let sum: f32 = choices.iter().map(|choice| choice.weight).sum();
        assert!((sum - 2.5).abs() < 1e-6);
    }

    #[test]
    fn grouped_router_masks_losing_groups() {
        let choices = route_noaux_tc(
            &[3.0, 2.0, -3.0, -4.0, 1.0, 0.5, 0.0, -1.0],
            &[0.0; 8],
            2,
            2,
            1,
            true,
            1.0,
        )
        .unwrap();
        assert!(choices.iter().all(|choice| choice.expert < 4));
    }

    #[test]
    fn normalization_epsilon_matches_underflowed_official_router_behavior() {
        let choices = route_noaux_tc(&[-1_000.0; 4], &[0.0; 4], 2, 1, 1, true, 2.5).unwrap();
        assert_eq!(choices.len(), 2);
        assert!(choices
            .iter()
            .all(|choice| choice.weight == 0.0 && choice.weight.is_finite()));
    }
}
