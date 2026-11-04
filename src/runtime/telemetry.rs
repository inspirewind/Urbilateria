//! Opt-in snapshots from the inference thread. These describe owned weight payloads, not RSS
//! or the OS page cache. Observers never affect model execution or the expert eviction policy.

use super::ExpertTelemetry;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

pub const JSON_PREFIX: &str = "URB_RUNTIME ";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Weights {
    pub total_layers: usize,
    /// Current runtimes retain a contiguous prefix, with zero-based layer IDs.
    pub loaded_layers: usize,
    pub planned_layers: usize,
    /// Estimated payload of successfully loaded backbone tensors, including the LM head.
    /// Excludes routed experts, sequence state, temporary buffers and allocator overhead.
    pub resident_bytes: u64,
    pub lm_head_loaded: bool,
    pub lm_head_planned: bool,
    pub loading: bool,
    pub expert_budget_bytes: u64,
    pub experts_per_layer: usize,
    pub row_streamed_embeddings: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerCache {
    pub layer: usize,
    pub experts: usize,
    pub capacity: usize,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Context {
        history: usize,
        dropped: usize,
        prompt: usize,
        limit: usize,
    },
    Kv {
        reused: usize,
        prefill: usize,
        capacity: usize,
    },
    /// Positions whose forward pass has finished, including all layers of a hybrid model.
    /// This is sequence occupancy, not a byte count or a count of sampled token IDs.
    KvState {
        position: usize,
        capacity: usize,
    },
    Weights(Weights),
    Experts {
        totals: ExpertTelemetry,
        layers: Vec<LayerCache>,
    },
    Stop {
        reason: String,
    },
}

impl Event {
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Weights(w) => {
                w.loaded_layers <= w.planned_layers
                    && w.planned_layers <= w.total_layers
                    && w.total_layers <= 4096
                    && (!w.lm_head_loaded || w.lm_head_planned)
            }
            Self::Experts { layers, .. } => {
                layers.len() <= 4096
                    && layers
                        .iter()
                        .all(|l| l.layer < 4096 && l.experts <= l.capacity)
                    && layers.windows(2).all(|pair| pair[0].layer < pair[1].layer)
            }
            Self::Kv {
                reused,
                prefill,
                capacity,
            } => reused.checked_add(*prefill).is_some_and(|n| n <= *capacity),
            Self::KvState { position, capacity } => position <= capacity,
            Self::Context { prompt, limit, .. } => prompt <= limit,
            Self::Stop { reason } => reason.len() <= 128,
        }
    }
}

struct Observer {
    callback: Box<dyn Fn(Event)>,
    last_sample: RefCell<[Option<Instant>; 2]>,
}

enum Sample {
    Resources,
    Sequence,
}
thread_local! {
    static OBSERVERS: RefCell<Vec<Rc<Observer>>> = const { RefCell::new(Vec::new()) };
}

/// Dropping a subscription removes only its own observer, even with nested inference scopes.
pub struct Subscription(Rc<Observer>);

pub fn subscribe(callback: impl Fn(Event) + 'static) -> Subscription {
    let observer = Rc::new(Observer {
        callback: Box::new(callback),
        last_sample: RefCell::new([None; 2]),
    });
    OBSERVERS.with(|active| active.borrow_mut().push(observer.clone()));
    Subscription(observer)
}

impl Drop for Subscription {
    fn drop(&mut self) {
        OBSERVERS.with(|active| active.borrow_mut().retain(|o| !Rc::ptr_eq(o, &self.0)));
    }
}

/// The snapshot is constructed only when a subscriber is active. Sampled updates are bounded
/// to four per second per category; lifecycle events and final values use `force = true`.
pub fn emit(force: bool, snapshot: impl FnOnce() -> Event) {
    emit_sample(Sample::Resources, force, snapshot);
}

/// Sequence progress has its own throttle, so frequent per-layer expert samples cannot hide it.
pub fn sequence(position: usize, capacity: usize, force: bool) {
    emit_sample(Sample::Sequence, force, || Event::KvState {
        position,
        capacity,
    });
}

fn emit_sample(sample: Sample, force: bool, snapshot: impl FnOnce() -> Event) {
    let observer = OBSERVERS.with(|active| active.borrow().last().cloned());
    let Some(observer) = observer else { return };
    let now = Instant::now();
    if !force {
        let mut samples = observer.last_sample.borrow_mut();
        let last = &mut samples[sample as usize];
        if last.is_some_and(|last| now.duration_since(last) < Duration::from_millis(250)) {
            return;
        }
        *last = Some(now);
    }
    (observer.callback)(snapshot());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observations_are_opt_in_scoped_and_final_samples_bypass_throttling() {
        emit(true, || {
            panic!("disabled observation must not construct a snapshot")
        });
        let events = Rc::new(RefCell::new(Vec::new()));
        let received = events.clone();
        let subscription = subscribe(move |event| received.borrow_mut().push(event));
        let snapshot = || Event::Kv {
            reused: 2,
            prefill: 3,
            capacity: 10,
        };
        emit(false, snapshot);
        emit(false, || panic!("sample should be throttled"));
        sequence(5, 10, false);
        sequence(6, 10, false);
        emit(true, snapshot);
        drop(subscription);
        emit(true, || panic!("observer must be removed on drop"));
        assert_eq!(
            *events.borrow(),
            vec![
                snapshot(),
                Event::KvState {
                    position: 5,
                    capacity: 10
                },
                snapshot()
            ]
        );
        assert!(!Event::KvState {
            position: 11,
            capacity: 10
        }
        .is_valid());
        assert!(!Event::Kv {
            reused: 9,
            prefill: 2,
            capacity: 10
        }
        .is_valid());
    }
}
