use crate::math::RouteChoice;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteSample {
    pub expert: usize,
    pub weight: f32,
}

impl From<&RouteChoice> for RouteSample {
    fn from(choice: &RouteChoice) -> Self {
        Self {
            expert: choice.expert,
            weight: choice.weight,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteEvent {
    pub token_position: usize,
    pub layer: usize,
    pub routes: Vec<RouteSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTrace {
    layer_count: usize,
    expert_count: usize,
    events: Vec<RouteEvent>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ExpertFrequency {
    pub expert: usize,
    pub selections: u64,
    pub selection_fraction: f64,
    pub accumulated_weight: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ExpertPairFrequency {
    pub first: usize,
    pub second: usize,
    pub coactivations: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LayerRouteSummary {
    pub layer: usize,
    pub event_count: u64,
    pub selection_count: u64,
    pub unique_experts: usize,
    pub selection_entropy_nats: f64,
    pub normalized_selection_entropy: f64,
    pub mean_within_token_gate_entropy_nats: f64,
    pub top_experts: Vec<ExpertFrequency>,
    pub top_coactivations: Vec<ExpertPairFrequency>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RouteTraceSummary {
    pub event_count: usize,
    pub token_count: usize,
    pub layers: Vec<LayerRouteSummary>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LayerCacheSimulation {
    pub layer: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub bytes_loaded: u64,
    pub final_cached_experts: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CacheSimulation {
    pub slots_per_layer: usize,
    pub expert_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub bytes_loaded: u64,
    pub layers: Vec<LayerCacheSimulation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    InvalidGeometry(String),
    InvalidEvent(String),
    Arithmetic(String),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(reason) => write!(f, "invalid route trace geometry: {reason}"),
            Self::InvalidEvent(reason) => write!(f, "invalid route event: {reason}"),
            Self::Arithmetic(reason) => write!(f, "route trace arithmetic error: {reason}"),
        }
    }
}

impl std::error::Error for TraceError {}

impl RouteTrace {
    pub fn new(layer_count: usize, expert_count: usize) -> Result<Self, TraceError> {
        if layer_count == 0 || expert_count == 0 {
            return Err(TraceError::InvalidGeometry(
                "layer and expert counts must be non-zero".to_owned(),
            ));
        }
        Ok(Self {
            layer_count,
            expert_count,
            events: Vec::new(),
        })
    }

    pub fn events(&self) -> &[RouteEvent] {
        &self.events
    }

    pub fn record_choices(
        &mut self,
        token_position: usize,
        layer: usize,
        choices: &[RouteChoice],
    ) -> Result<(), TraceError> {
        let routes = choices.iter().map(RouteSample::from).collect();
        self.record(RouteEvent {
            token_position,
            layer,
            routes,
        })
    }

    /// Records the route vectors emitted by one decoder step. Dense layers are represented by
    /// empty vectors and are skipped.
    pub fn record_model_step(
        &mut self,
        token_position: usize,
        routes_by_layer: &[Vec<RouteChoice>],
    ) -> Result<(), TraceError> {
        if routes_by_layer.len() != self.layer_count {
            return Err(TraceError::InvalidEvent(format!(
                "model step has {} layers, trace expects {}",
                routes_by_layer.len(),
                self.layer_count
            )));
        }
        let mut candidate = self.clone();
        for (layer, choices) in routes_by_layer.iter().enumerate() {
            if !choices.is_empty() {
                candidate.record_choices(token_position, layer, choices)?;
            }
        }
        *self = candidate;
        Ok(())
    }

    pub fn record(&mut self, event: RouteEvent) -> Result<(), TraceError> {
        if event.layer >= self.layer_count {
            return Err(TraceError::InvalidEvent(format!(
                "layer {} is outside 0..{}",
                event.layer, self.layer_count
            )));
        }
        if event.routes.is_empty() {
            return Err(TraceError::InvalidEvent(
                "a routed layer event must contain at least one expert".to_owned(),
            ));
        }
        let mut unique = BTreeSet::new();
        for route in &event.routes {
            if route.expert >= self.expert_count {
                return Err(TraceError::InvalidEvent(format!(
                    "expert {} is outside 0..{}",
                    route.expert, self.expert_count
                )));
            }
            if !route.weight.is_finite() || route.weight <= 0.0 {
                return Err(TraceError::InvalidEvent(
                    "route weights must be finite and positive".to_owned(),
                ));
            }
            if !unique.insert(route.expert) {
                return Err(TraceError::InvalidEvent(format!(
                    "expert {} occurs twice in one event",
                    route.expert
                )));
            }
        }
        if let Some(previous) = self.events.last() {
            let previous_key = (previous.token_position, previous.layer);
            let key = (event.token_position, event.layer);
            if key <= previous_key {
                return Err(TraceError::InvalidEvent(format!(
                    "events must be strictly ordered by (token, layer); {key:?} follows {previous_key:?}"
                )));
            }
        }
        self.events.push(event);
        Ok(())
    }

    pub fn summarize(
        &self,
        top_experts: usize,
        top_pairs: usize,
    ) -> Result<RouteTraceSummary, TraceError> {
        let mut layers = Vec::new();
        for layer in 0..self.layer_count {
            let events: Vec<&RouteEvent> = self
                .events
                .iter()
                .filter(|event| event.layer == layer)
                .collect();
            if events.is_empty() {
                continue;
            }
            let mut selections = vec![0u64; self.expert_count];
            let mut masses = vec![0.0f64; self.expert_count];
            let mut pairs: BTreeMap<(usize, usize), u64> = BTreeMap::new();
            let mut within_entropy = 0.0f64;
            let mut selection_count = 0u64;

            for event in &events {
                let total_weight: f64 = event
                    .routes
                    .iter()
                    .map(|route| f64::from(route.weight))
                    .sum();
                if !total_weight.is_finite() || total_weight <= 0.0 {
                    return Err(TraceError::InvalidEvent(
                        "event weight sum is non-finite or non-positive".to_owned(),
                    ));
                }
                for route in &event.routes {
                    selections[route.expert] =
                        selections[route.expert].checked_add(1).ok_or_else(|| {
                            TraceError::Arithmetic("expert selection count overflows".to_owned())
                        })?;
                    masses[route.expert] += f64::from(route.weight);
                    selection_count = selection_count.checked_add(1).ok_or_else(|| {
                        TraceError::Arithmetic("total selection count overflows".to_owned())
                    })?;
                    let probability = f64::from(route.weight) / total_weight;
                    within_entropy -= probability * probability.ln();
                }
                for first in 0..event.routes.len() {
                    for second in first + 1..event.routes.len() {
                        let a = event.routes[first].expert;
                        let b = event.routes[second].expert;
                        let pair = if a < b { (a, b) } else { (b, a) };
                        let count = pairs.entry(pair).or_default();
                        *count = count.checked_add(1).ok_or_else(|| {
                            TraceError::Arithmetic("coactivation count overflows".to_owned())
                        })?;
                    }
                }
            }

            let selection_entropy = if selection_count == 0 {
                0.0
            } else {
                selections
                    .iter()
                    .filter(|&&count| count > 0)
                    .map(|&count| {
                        let probability = count as f64 / selection_count as f64;
                        -probability * probability.ln()
                    })
                    .sum()
            };
            let normalized_entropy = if self.expert_count <= 1 {
                0.0
            } else {
                selection_entropy / (self.expert_count as f64).ln()
            };

            let mut frequencies: Vec<ExpertFrequency> = selections
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .map(|(expert, &count)| ExpertFrequency {
                    expert,
                    selections: count,
                    selection_fraction: count as f64 / selection_count.max(1) as f64,
                    accumulated_weight: masses[expert],
                })
                .collect();
            frequencies.sort_by(|left, right| {
                right
                    .selections
                    .cmp(&left.selections)
                    .then_with(|| left.expert.cmp(&right.expert))
            });
            frequencies.truncate(top_experts);

            let mut pair_frequencies: Vec<ExpertPairFrequency> = pairs
                .into_iter()
                .map(|((first, second), coactivations)| ExpertPairFrequency {
                    first,
                    second,
                    coactivations,
                })
                .collect();
            pair_frequencies.sort_by(|left, right| {
                right
                    .coactivations
                    .cmp(&left.coactivations)
                    .then_with(|| (left.first, left.second).cmp(&(right.first, right.second)))
            });
            pair_frequencies.truncate(top_pairs);

            layers.push(LayerRouteSummary {
                layer,
                event_count: events.len() as u64,
                selection_count,
                unique_experts: selections.iter().filter(|&&count| count > 0).count(),
                selection_entropy_nats: selection_entropy,
                normalized_selection_entropy: normalized_entropy,
                mean_within_token_gate_entropy_nats: within_entropy / events.len() as f64,
                top_experts: frequencies,
                top_coactivations: pair_frequencies,
            });
        }

        let token_count = self
            .events
            .last()
            .map(|event| event.token_position.saturating_add(1))
            .unwrap_or(0);
        Ok(RouteTraceSummary {
            event_count: self.events.len(),
            token_count,
            layers,
        })
    }

    /// Replays exact route IDs through an independent per-layer deterministic LRU.
    /// `bytes_loaded` is a storage-only counter; it deliberately makes no latency claim.
    pub fn simulate_lru(
        &self,
        slots_per_layer: usize,
        expert_bytes: u64,
    ) -> Result<CacheSimulation, TraceError> {
        let mut caches = vec![Vec::<usize>::new(); self.layer_count];
        let mut layer_hits = vec![0u64; self.layer_count];
        let mut layer_misses = vec![0u64; self.layer_count];

        for event in &self.events {
            let cache = &mut caches[event.layer];
            for route in &event.routes {
                if let Some(position) = cache.iter().position(|&expert| expert == route.expert) {
                    layer_hits[event.layer] =
                        layer_hits[event.layer].checked_add(1).ok_or_else(|| {
                            TraceError::Arithmetic("cache hit count overflows".to_owned())
                        })?;
                    let expert = cache.remove(position);
                    cache.push(expert);
                } else {
                    layer_misses[event.layer] =
                        layer_misses[event.layer].checked_add(1).ok_or_else(|| {
                            TraceError::Arithmetic("cache miss count overflows".to_owned())
                        })?;
                    if slots_per_layer > 0 {
                        if cache.len() == slots_per_layer {
                            cache.remove(0);
                        }
                        cache.push(route.expert);
                    }
                }
            }
        }

        let mut layers = Vec::with_capacity(self.layer_count);
        let mut hits = 0u64;
        let mut misses = 0u64;
        for layer in 0..self.layer_count {
            let layer_total = layer_hits[layer].saturating_add(layer_misses[layer]);
            let bytes_loaded = layer_misses[layer]
                .checked_mul(expert_bytes)
                .ok_or_else(|| {
                    TraceError::Arithmetic("per-layer loaded bytes overflows".to_owned())
                })?;
            layers.push(LayerCacheSimulation {
                layer,
                hits: layer_hits[layer],
                misses: layer_misses[layer],
                hit_rate: ratio(layer_hits[layer], layer_total),
                bytes_loaded,
                final_cached_experts: caches[layer].len(),
            });
            hits = hits.checked_add(layer_hits[layer]).ok_or_else(|| {
                TraceError::Arithmetic("total cache hit count overflows".to_owned())
            })?;
            misses = misses.checked_add(layer_misses[layer]).ok_or_else(|| {
                TraceError::Arithmetic("total cache miss count overflows".to_owned())
            })?;
        }
        let bytes_loaded = misses
            .checked_mul(expert_bytes)
            .ok_or_else(|| TraceError::Arithmetic("loaded bytes overflows".to_owned()))?;
        Ok(CacheSimulation {
            slots_per_layer,
            expert_bytes,
            hits,
            misses,
            hit_rate: ratio(hits, hits.saturating_add(misses)),
            bytes_loaded,
            layers,
        })
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(token: usize, layer: usize, experts: &[usize]) -> RouteEvent {
        RouteEvent {
            token_position: token,
            layer,
            routes: experts
                .iter()
                .map(|&expert| RouteSample {
                    expert,
                    weight: 1.0,
                })
                .collect(),
        }
    }

    #[test]
    fn summary_reports_frequency_entropy_and_coactivation() {
        let mut trace = RouteTrace::new(2, 4).unwrap();
        trace.record(event(0, 0, &[0, 1])).unwrap();
        trace.record(event(0, 1, &[2, 3])).unwrap();
        trace.record(event(1, 0, &[0, 2])).unwrap();
        let summary = trace.summarize(4, 4).unwrap();
        assert_eq!(summary.token_count, 2);
        assert_eq!(summary.layers[0].selection_count, 4);
        assert_eq!(summary.layers[0].top_experts[0].expert, 0);
        assert_eq!(summary.layers[0].top_experts[0].selections, 2);
        assert_eq!(summary.layers[0].top_coactivations.len(), 2);
        assert!(summary.layers[0].normalized_selection_entropy > 0.0);
    }

    #[test]
    fn lru_capacity_curve_uses_the_same_exact_trace() {
        let mut trace = RouteTrace::new(1, 4).unwrap();
        trace.record(event(0, 0, &[0, 1])).unwrap();
        trace.record(event(1, 0, &[0, 1])).unwrap();

        let zero = trace.simulate_lru(0, 100).unwrap();
        assert_eq!((zero.hits, zero.misses, zero.bytes_loaded), (0, 4, 400));
        let one = trace.simulate_lru(1, 100).unwrap();
        assert_eq!((one.hits, one.misses), (0, 4));
        let two = trace.simulate_lru(2, 100).unwrap();
        assert_eq!((two.hits, two.misses, two.bytes_loaded), (2, 2, 200));
    }

    #[test]
    fn rejects_duplicate_experts_and_out_of_order_events() {
        let mut trace = RouteTrace::new(2, 4).unwrap();
        assert!(trace.record(event(0, 0, &[1, 1])).is_err());
        trace.record(event(1, 0, &[1])).unwrap();
        assert!(trace.record(event(0, 1, &[2])).is_err());
    }
}
