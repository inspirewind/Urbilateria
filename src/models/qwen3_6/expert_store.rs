//! Packed BF16 routed-expert slice loading and deterministic per-layer caching.

use crate::execution::{install, worker_threads};
use crate::model::{Bf16Matrix, WeightMatrix};
use crate::models::qwen3_8::expert::{
    Qwen38Expert as Qwen36Expert, Qwen38ExpertError as Qwen36ExpertError,
};
use crate::profiling::{span, ProfileStage};
use crate::runtime::cache::{GroupedAccessPlan, LayerLruCache};
use crate::runtime::ExpertTelemetry;
use crate::storage::{DType, TensorIndex, WeightLoadError};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum Qwen36ExpertStoreError {
    Invalid(String),
    InvalidLayer { layer: usize, layers: usize },
    InvalidExpert { expert: usize, experts: usize },
    Weight(WeightLoadError),
    Expert(Qwen36ExpertError),
    Budget { required: u64, maximum: u64 },
}

impl fmt::Display for Qwen36ExpertStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(formatter, "invalid Qwen3.6 expert store: {reason}"),
            Self::InvalidLayer { layer, layers } => {
                write!(formatter, "Qwen3.6 expert layer {layer} is outside 0..{layers}")
            }
            Self::InvalidExpert { expert, experts } => {
                write!(formatter, "Qwen3.6 expert {expert} is outside 0..{experts}")
            }
            Self::Weight(error) => error.fmt(formatter),
            Self::Expert(error) => error.fmt(formatter),
            Self::Budget { required, maximum } => write!(
                formatter,
                "Qwen3.6 routed expert needs {required} resident bytes, per-expert limit is {maximum}"
            ),
        }
    }
}

impl std::error::Error for Qwen36ExpertStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight(error) => Some(error),
            Self::Expert(error) => Some(error),
            Self::Invalid(_)
            | Self::InvalidLayer { .. }
            | Self::InvalidExpert { .. }
            | Self::Budget { .. } => None,
        }
    }
}

impl From<WeightLoadError> for Qwen36ExpertStoreError {
    fn from(value: WeightLoadError) -> Self {
        Self::Weight(value)
    }
}

impl From<Qwen36ExpertError> for Qwen36ExpertStoreError {
    fn from(value: Qwen36ExpertError) -> Self {
        Self::Expert(value)
    }
}

#[derive(Debug)]
pub struct Qwen36LoadedExpert {
    pub value: Qwen36Expert,
    resident_bytes: u64,
    payload_bytes: u64,
}

impl Qwen36LoadedExpert {
    pub fn load(
        index: &TensorIndex,
        layer: usize,
        expert: usize,
        hidden: usize,
        intermediate: usize,
        maximum_resident_bytes: u64,
    ) -> Result<Self, Qwen36ExpertStoreError> {
        Self::load_reusing(
            index,
            layer,
            expert,
            hidden,
            intermediate,
            maximum_resident_bytes,
            None,
        )
    }

    fn load_reusing(
        index: &TensorIndex,
        layer: usize,
        expert: usize,
        hidden: usize,
        intermediate: usize,
        maximum_resident_bytes: u64,
        reuse: Option<[Vec<u8>; 3]>,
    ) -> Result<Self, Qwen36ExpertStoreError> {
        if hidden == 0 || intermediate == 0 || maximum_resident_bytes == 0 {
            return Err(Qwen36ExpertStoreError::Invalid(
                "hidden/intermediate dimensions and per-expert budget must be non-zero".to_owned(),
            ));
        }
        if layer >= 40 || expert >= 256 {
            return Err(Qwen36ExpertStoreError::Invalid(
                "expert index outside release geometry".into(),
            ));
        }
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
        let resident_bytes = Self::inspect_layer(index, &prefix, hidden, intermediate)?;
        if resident_bytes > maximum_resident_bytes {
            return Err(Qwen36ExpertStoreError::Budget {
                required: resident_bytes,
                maximum: maximum_resident_bytes,
            });
        }
        let matrix_bytes = hidden * intermediate * 2;
        let mut buffers = reuse.unwrap_or_default().into_iter();
        let mut load = |name: &str,
                        offset: usize,
                        rows,
                        cols|
         -> Result<WeightMatrix, Qwen36ExpertStoreError> {
            let bytes = index
                .read_range_reusing(
                    name,
                    offset as u64,
                    matrix_bytes,
                    buffers
                        .next()
                        .expect("one buffer for each expert projection"),
                )
                .map_err(WeightLoadError::from)?;
            let matrix =
                Bf16Matrix::from_le_bytes(rows, cols, bytes).map_err(WeightLoadError::from)?;
            Ok(WeightMatrix::Bf16(matrix))
        };
        let gate_up = format!("{prefix}.gate_up_proj");
        let down = format!("{prefix}.down_proj");
        let value = Qwen36Expert::new(
            load(&gate_up, expert * 2 * matrix_bytes, intermediate, hidden)?,
            load(
                &gate_up,
                (expert * 2 + 1) * matrix_bytes,
                intermediate,
                hidden,
            )?,
            load(&down, expert * matrix_bytes, hidden, intermediate)?,
        )?;
        let payload_bytes = resident_bytes;
        Ok(Self {
            value,
            resident_bytes,
            payload_bytes,
        })
    }

    pub fn inspect_resident_bytes(
        index: &TensorIndex,
        hidden: usize,
        intermediate: usize,
    ) -> Result<u64, Qwen36ExpertStoreError> {
        Self::inspect_layer(
            index,
            "model.language_model.layers.0.mlp.experts",
            hidden,
            intermediate,
        )
    }

    fn inspect_layer(
        index: &TensorIndex,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
    ) -> Result<u64, Qwen36ExpertStoreError> {
        if hidden != 2048 || intermediate != 512 {
            return Err(Qwen36ExpertStoreError::Invalid(
                "unsupported expert geometry".into(),
            ));
        }
        for (suffix, shape) in [
            (
                "gate_up_proj",
                [256, (2 * intermediate) as u64, hidden as u64],
            ),
            ("down_proj", [256, hidden as u64, intermediate as u64]),
        ] {
            let name = format!("{prefix}.{suffix}");
            let tensor = index.require(&name).map_err(WeightLoadError::from)?;
            if tensor.dtype != DType::Bf16 || tensor.shape != shape {
                return Err(Qwen36ExpertStoreError::Invalid(format!(
                    "invalid packed BF16 expert {name}"
                )));
            }
        }
        Ok((hidden * intermediate * 6) as u64)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

#[derive(Debug)]
pub(super) enum Qwen36GroupedAccess {
    Ordered(Vec<usize>),
    Planned(GroupedAccessPlan),
    Initial(GroupedAccessPlan),
}

impl Qwen36GroupedAccess {
    pub(super) fn order(&self) -> &[usize] {
        match self {
            Self::Ordered(order) => order,
            Self::Planned(plan) | Self::Initial(plan) => plan.order(),
        }
    }
}

/// Final survivors staged inside the initially empty layer's cache reservation.
/// Each successful value moves into the cache once, without another read or allocation.
#[derive(Debug, Default)]
pub(super) struct Qwen36PrefetchedExperts {
    pending: BTreeMap<usize, Result<Arc<Qwen36LoadedExpert>, Qwen36ExpertStoreError>>,
}

#[derive(Debug)]
pub struct Qwen36ExpertStore {
    index: Arc<TensorIndex>,
    layers: usize,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    maximum_expert_bytes: u64,
    cache: LayerLruCache<Arc<Qwen36LoadedExpert>>,
    // One displaced or uncached expert occupies the existing transient-expert
    // reserve. Only exclusively owned buffers can be overwritten by the next miss.
    recycled: Option<Arc<Qwen36LoadedExpert>>,
    telemetry: ExpertTelemetry,
}

impl Qwen36ExpertStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new_shared(
        index: Arc<TensorIndex>,
        layers: usize,
        experts: usize,
        hidden: usize,
        intermediate: usize,
        slots_per_layer: usize,
        maximum_expert_bytes: u64,
    ) -> Result<Self, Qwen36ExpertStoreError> {
        if layers == 0
            || experts == 0
            || hidden == 0
            || intermediate == 0
            || maximum_expert_bytes == 0
            || slots_per_layer > experts
        {
            return Err(Qwen36ExpertStoreError::Invalid(
                "expert cache geometry or budget is invalid".to_owned(),
            ));
        }
        Ok(Self {
            index,
            layers,
            experts,
            hidden,
            intermediate,
            maximum_expert_bytes,
            cache: LayerLruCache::new(layers, slots_per_layer),
            recycled: None,
            telemetry: ExpertTelemetry::default(),
        })
    }

    pub fn acquire(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<Arc<Qwen36LoadedExpert>, Qwen36ExpertStoreError> {
        self.acquire_with_plan(layer, expert, None)
    }

    pub(super) fn acquire_grouped(
        &mut self,
        layer: usize,
        expert: usize,
        schedule: &Qwen36GroupedAccess,
        prefetched: &mut Qwen36PrefetchedExperts,
    ) -> Result<Arc<Qwen36LoadedExpert>, Qwen36ExpertStoreError> {
        match schedule {
            Qwen36GroupedAccess::Ordered(_) => self.acquire(layer, expert),
            Qwen36GroupedAccess::Planned(plan) | Qwen36GroupedAccess::Initial(plan) => {
                debug_assert_eq!(layer, plan.layer());
                if let Some(loaded) = prefetched.pending.remove(&expert) {
                    self.admit_loaded(layer, expert, Some(plan), loaded?)
                } else {
                    self.acquire_with_plan(layer, expert, Some(plan))
                }
            }
        }
    }

    pub(super) fn prefetch_grouped(
        &self,
        schedule: &Qwen36GroupedAccess,
    ) -> Qwen36PrefetchedExperts {
        let Qwen36GroupedAccess::Initial(plan) = schedule else {
            return Qwen36PrefetchedExperts::default();
        };
        let layer = plan.layer();
        debug_assert!(self.cache.layer_is_empty(layer));
        let ids = plan.retained_keys().collect::<Vec<_>>();
        debug_assert!(ids.len() <= self.cache.layer_capacity(layer));
        let context = crate::profiling::capture_context();
        // Staged survivors plus already admitted entries never exceed the cache slots.
        // Non-final experts use the existing one-expert transient reserve. Keep errors
        // by expert ID so the runtime still reports their original request positions.
        let pending = install(|| {
            ids.par_iter()
                .map(|&id| {
                    let result = context.enter(|| {
                        let _profile = span(ProfileStage::Qwen36ExpertLoad);
                        Qwen36LoadedExpert::load(
                            &self.index,
                            layer,
                            id,
                            self.hidden,
                            self.intermediate,
                            self.maximum_expert_bytes,
                        )
                        .map(Arc::new)
                    });
                    (id, result)
                })
                .collect::<Vec<_>>()
        });
        Qwen36PrefetchedExperts {
            pending: pending.into_iter().collect(),
        }
    }

    fn acquire_with_plan(
        &mut self,
        layer: usize,
        expert: usize,
        plan: Option<&GroupedAccessPlan>,
    ) -> Result<Arc<Qwen36LoadedExpert>, Qwen36ExpertStoreError> {
        if layer >= self.layers {
            return Err(Qwen36ExpertStoreError::InvalidLayer {
                layer,
                layers: self.layers,
            });
        }
        if expert >= self.experts {
            return Err(Qwen36ExpertStoreError::InvalidExpert {
                expert,
                experts: self.experts,
            });
        }
        if self.cache.contains(layer, expert) {
            return self.cache.access(
                &mut self.telemetry,
                layer,
                expert,
                || unreachable!("known cache hit cannot load"),
                |value| Ok(Arc::clone(value)),
            );
        }

        // Load before allowing the LRU to evict: a short/corrupt shard must not discard a usable
        // cached expert.
        let _profile = span(ProfileStage::Qwen36ExpertLoad);
        let reuse = self
            .recycled
            .take()
            .and_then(|expert| Arc::try_unwrap(expert).ok())
            .and_then(|expert| expert.value.into_bf16_buffers());
        let loaded = Qwen36LoadedExpert::load_reusing(
            &self.index,
            layer,
            expert,
            self.hidden,
            self.intermediate,
            self.maximum_expert_bytes,
            reuse,
        )?;
        self.admit_loaded(layer, expert, plan, Arc::new(loaded))
    }

    fn admit_loaded(
        &mut self,
        layer: usize,
        expert: usize,
        plan: Option<&GroupedAccessPlan>,
        loaded: Arc<Qwen36LoadedExpert>,
    ) -> Result<Arc<Qwen36LoadedExpert>, Qwen36ExpertStoreError> {
        let resident_bytes = loaded.resident_bytes();
        let payload_bytes = loaded.payload_bytes();
        let load = || Ok::<_, Qwen36ExpertStoreError>((loaded, resident_bytes, payload_bytes));
        let execute = |value: &mut Arc<Qwen36LoadedExpert>| {
            Ok::<_, Qwen36ExpertStoreError>(Arc::clone(value))
        };
        let (selected, evicted) = match plan {
            Some(plan) => self.cache.access_grouped_with_evicted(
                &mut self.telemetry,
                plan,
                expert,
                load,
                execute,
            ),
            None => {
                self.cache
                    .access_with_evicted(&mut self.telemetry, layer, expert, load, execute)
            }
        }?;
        self.recycled = evicted;
        Ok(selected)
    }

    /// Stage a selected set only when every value remains charged to the cache.
    /// Otherwise the caller must use the one-transient-expert streaming path.
    pub fn acquire_retained(
        &mut self,
        layer: usize,
        ids: &[usize],
    ) -> Result<Option<Vec<Arc<Qwen36LoadedExpert>>>, Qwen36ExpertStoreError> {
        if layer >= self.layers {
            return Err(Qwen36ExpertStoreError::InvalidLayer {
                layer,
                layers: self.layers,
            });
        }
        if !self.cache.can_insert_without_eviction(layer, ids) {
            return Ok(None);
        }
        let mut missing = Vec::new();
        for &id in ids {
            if !self.cache.contains(layer, id) && !missing.contains(&id) {
                missing.push(id);
            }
        }
        if missing.len() < 2 {
            return ids
                .iter()
                .map(|&id| self.acquire(layer, id))
                .collect::<Result<Vec<_>, _>>()
                .map(Some);
        }
        let context = crate::profiling::capture_context();
        // No eviction is needed, so all in-flight payloads already fit the cache reservation.
        // Reads are positional and each worker owns its buffers; finite validation stays enabled.
        let loaded = install(|| {
            missing
                .par_iter()
                .map(|&id| {
                    context.enter(|| {
                        let _profile = span(ProfileStage::Qwen36ExpertLoad);
                        Qwen36LoadedExpert::load(
                            &self.index,
                            layer,
                            id,
                            self.hidden,
                            self.intermediate,
                            self.maximum_expert_bytes,
                        )
                        .map(Arc::new)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut loaded = loaded.into_iter().map(Some).collect::<Vec<_>>();
        let mut selected = Vec::with_capacity(ids.len());
        for &id in ids {
            selected.push(self.cache.access(
                &mut self.telemetry,
                layer,
                id,
                || {
                    let position = missing
                        .iter()
                        .position(|&key| key == id)
                        .expect("missing expert was prefetched");
                    let value = loaded[position]
                        .take()
                        .expect("each missing expert is inserted once");
                    let resident = value.resident_bytes();
                    let payload = value.payload_bytes();
                    Ok::<_, Qwen36ExpertStoreError>((value, resident, payload))
                },
                |value| Ok(Arc::clone(value)),
            )?);
        }
        Ok(Some(selected))
    }

    /// Choose a bounded grouped schedule without mutating cache contents or counters.
    pub(super) fn grouped_access_order(
        &self,
        layer: usize,
        ids: &[usize],
        experts_per_token: usize,
    ) -> Option<Qwen36GroupedAccess> {
        let initial = worker_threads() > 1
            && self.cache.layer_is_empty(layer)
            && self.cache.layer_capacity(layer) >= experts_per_token;
        if let Some(plan) = self.cache.grouped_access_plan(layer, ids) {
            return Some(if initial {
                Qwen36GroupedAccess::Initial(plan)
            } else {
                Qwen36GroupedAccess::Planned(plan)
            });
        }
        self.cache
            .grouped_access_order(layer, ids)
            .map(Qwen36GroupedAccess::Ordered)
    }

    pub(super) fn finish_grouped_accesses(
        &mut self,
        layer: usize,
        ids: &[usize],
        schedule: &Qwen36GroupedAccess,
    ) {
        match schedule {
            Qwen36GroupedAccess::Ordered(order) => {
                self.cache.finish_grouped_accesses(layer, ids, order.len());
            }
            Qwen36GroupedAccess::Planned(plan) | Qwen36GroupedAccess::Initial(plan) => {
                self.cache.finish_grouped_plan(plan);
            }
        }
    }

    pub fn telemetry(&self) -> &ExpertTelemetry {
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn packed_experts_read_only_the_selected_gate_up_and_down_slices() {
        let directory =
            std::env::temp_dir().join(format!("urb-qwen36-packed-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let matrix_bytes = 2048u64 * 512 * 2;
        let gate_bytes = 256 * 2 * matrix_bytes;
        let prefix = "model.language_model.layers.0.mlp.experts";
        let header = serde_json::json!({
            format!("{prefix}.gate_up_proj"): {"dtype":"BF16", "shape":[256,1024,2048], "data_offsets":[0,gate_bytes]},
            format!("{prefix}.down_proj"): {"dtype":"BF16", "shape":[256,2048,512], "data_offsets":[gate_bytes,gate_bytes+256*matrix_bytes]},
        }).to_string();
        let mut file = std::fs::File::create(directory.join("packed.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        let start = 8 + header.len() as u64;
        // Sparse payload: only one expert's first gate/up/down entries are nonzero.
        file.set_len(start + gate_bytes + 256 * matrix_bytes)
            .unwrap();
        for offset in [
            255 * 2 * matrix_bytes,
            (255 * 2 + 1) * matrix_bytes,
            gate_bytes + 255 * matrix_bytes,
        ] {
            file.seek(SeekFrom::Start(start + offset)).unwrap();
            file.write_all(&0x3f80u16.to_le_bytes()).unwrap(); // 1.0 BF16
        }
        drop(file);
        let index = TensorIndex::open(&directory).unwrap();
        let expert = Qwen36LoadedExpert::load(&index, 0, 255, 2048, 512, 3 * matrix_bytes).unwrap();
        assert_eq!(expert.resident_bytes(), 3 * matrix_bytes);
        assert_eq!(expert.payload_bytes(), 3 * matrix_bytes);
        let mut input = vec![0.0; 2048];
        input[0] = 1.0;
        let output = expert.value.forward(&input).unwrap();
        assert_eq!(output[0], 0.73046875); // BF16 SiLU(1), then identity up/down.
        assert!(output[1..].iter().all(|&x| x == 0.0));
        let zero = Qwen36LoadedExpert::load(&index, 0, 0, 2048, 512, 3 * matrix_bytes).unwrap();
        assert!(zero
            .value
            .forward(&input)
            .unwrap()
            .iter()
            .all(|&x| x == 0.0));
        assert!(matches!(
            Qwen36LoadedExpert::load(&index, 0, 255, 2048, 512, 3 * matrix_bytes - 1),
            Err(Qwen36ExpertStoreError::Budget { .. })
        ));
        assert!(Qwen36LoadedExpert::load(&index, 0, 256, 2048, 512, 3 * matrix_bytes).is_err());
        // Replacing a native expert overwrites every byte without reallocating its projections.
        let buffers = expert.value.into_bf16_buffers().unwrap();
        let addresses = buffers.each_ref().map(|bytes| bytes.as_ptr());
        let reused = Qwen36LoadedExpert::load_reusing(
            &index,
            0,
            0,
            2048,
            512,
            3 * matrix_bytes,
            Some(buffers),
        )
        .unwrap();
        let buffers = reused.value.into_bf16_buffers().unwrap();
        assert_eq!(addresses, buffers.each_ref().map(|bytes| bytes.as_ptr()));
        assert!(buffers.iter().flatten().all(|&byte| byte == 0));
        drop(buffers);
        let index = Arc::new(index);
        let mut store = Qwen36ExpertStore::new_shared(
            Arc::clone(&index),
            1,
            256,
            2048,
            512,
            2,
            3 * matrix_bytes,
        )
        .unwrap();
        let selected = store.acquire_retained(0, &[255, 0]).unwrap().unwrap();
        assert_eq!(selected[0].value.forward(&input).unwrap(), output);
        assert_eq!(store.telemetry().misses, 2);
        assert_eq!(store.telemetry().resident_bytes, 6 * matrix_bytes);
        let before = store.telemetry().clone();
        assert!(store.acquire_retained(0, &[1, 2]).unwrap().is_none());
        assert_eq!(store.telemetry(), &before);
        store.acquire_retained(0, &[255, 0]).unwrap().unwrap();
        assert_eq!(store.telemetry().hits, 2);
        assert_eq!(store.telemetry().bytes_read, 6 * matrix_bytes);
        drop(store);
        let new_store = |slots| {
            Qwen36ExpertStore::new_shared(
                Arc::clone(&index),
                1,
                256,
                2048,
                512,
                slots,
                3 * matrix_bytes,
            )
            .unwrap()
        };
        for slots in [0, 1, 2] {
            let mut store = new_store(slots);
            let ids = [255, 0, 1, 255, 0, 1];
            let before = store.telemetry().clone();
            let schedule = store.grouped_access_order(0, &ids, 2).unwrap();
            assert_eq!(
                matches!(schedule, Qwen36GroupedAccess::Initial(_)),
                slots >= 2 && worker_threads() > 1,
            );
            assert_eq!(store.telemetry(), &before);
            store.acquire(0, 255).unwrap();
            assert!(store.grouped_access_order(0, &ids, 2).is_some());
        }
        // Only the final survivors occupy staged cache slots. Admission moves their
        // exact payloads, while all other groups share one transient expert buffer.
        let mut initial = new_store(2);
        let mut reference = new_store(2);
        let accesses = [255, 0, 1, 255, 0, 1];
        let schedule =
            Qwen36GroupedAccess::Initial(initial.cache.grouped_access_plan(0, &accesses).unwrap());
        let before = initial.telemetry().clone();
        let mut prefetched = initial.prefetch_grouped(&schedule);
        assert_eq!(
            prefetched.pending.keys().copied().collect::<Vec<_>>(),
            [0, 1]
        );
        assert!(initial.cache.layer_is_empty(0));
        assert_eq!(initial.telemetry(), &before);
        let addresses = prefetched
            .pending
            .iter()
            .map(|(&id, value)| (id, Arc::as_ptr(value.as_ref().unwrap())))
            .collect::<BTreeMap<_, _>>();
        for &id in schedule.order() {
            let expert = initial
                .acquire_grouped(0, id, &schedule, &mut prefetched)
                .unwrap();
            if let Some(&address) = addresses.get(&id) {
                assert_eq!(Arc::as_ptr(&expert), address);
            }
            if id == 255 {
                assert_eq!(expert.value.forward(&input).unwrap(), output);
                assert!(!initial.cache.contains(0, id));
            }
            assert!(
                prefetched.pending.len()
                    + initial.telemetry().resident_experts
                    + usize::from(initial.recycled.is_some())
                    <= 3
            );
        }
        assert!(prefetched.pending.is_empty());
        initial.finish_grouped_accesses(0, &accesses, &schedule);
        assert_eq!(initial.telemetry().misses, 3);
        assert_eq!(initial.telemetry().bytes_read, 9 * matrix_bytes);
        assert_eq!(initial.telemetry().resident_bytes, 6 * matrix_bytes);
        for id in accesses {
            reference.acquire(0, id).unwrap();
        }
        for id in [255, 0, 1, 2, 0] {
            let a = initial.telemetry().clone();
            let b = reference.telemetry().clone();
            initial.acquire(0, id).unwrap();
            reference.acquire(0, id).unwrap();
            assert_eq!(
                initial.telemetry().misses - a.misses,
                reference.telemetry().misses - b.misses
            );
        }
        drop(initial);
        drop(reference);
        let mut serial = new_store(2);
        let mut planned = new_store(2);
        for id in [255, 0] {
            serial.acquire(0, id).unwrap();
            planned.acquire(0, id).unwrap();
        }
        let accesses = [255, 1, 2, 255];
        let schedule = planned.grouped_access_order(0, &accesses, 1).unwrap();
        assert!(matches!(schedule, Qwen36GroupedAccess::Planned(_)));
        let original = planned.cache.peek(0, 255).unwrap().clone();
        let before = planned.telemetry().clone();
        let mut prefetched = planned.prefetch_grouped(&schedule);
        assert!(prefetched.pending.is_empty());
        for &id in schedule.order() {
            let expert = planned
                .acquire_grouped(0, id, &schedule, &mut prefetched)
                .unwrap();
            assert!(planned.cache.contains(0, 255));
            assert_eq!(planned.telemetry().resident_bytes, 6 * matrix_bytes);
            if id == 255 {
                assert!(Arc::ptr_eq(&original, &expert));
                assert_eq!(expert.value.forward(&input).unwrap(), output);
            }
            if id == 1 {
                assert!(!planned.cache.contains(0, 1));
                assert!(planned.recycled.is_some());
            }
        }
        planned.finish_grouped_accesses(0, &accesses, &schedule);
        assert_eq!(planned.telemetry().misses - before.misses, 2);
        assert_eq!(
            planned.telemetry().bytes_read - before.bytes_read,
            6 * matrix_bytes
        );
        for id in accesses {
            serial.acquire(0, id).unwrap();
        }
        assert_eq!(serial.telemetry().misses - before.misses, 3);
        for id in [0, 255, 2, 3, 255] {
            let a = serial.telemetry().clone();
            let b = planned.telemetry().clone();
            serial.acquire(0, id).unwrap();
            planned.acquire(0, id).unwrap();
            assert_eq!(
                serial.telemetry().misses - a.misses,
                planned.telemetry().misses - b.misses
            );
            assert_eq!(
                serial.telemetry().hits - a.hits,
                planned.telemetry().hits - b.hits
            );
        }
        assert_eq!(original.value.forward(&input).unwrap(), output);
        drop(original);
        drop(planned);
        drop(serial);
        let mut sequential = new_store(2);
        let mut grouped = new_store(2);
        let accesses = [255, 0, 255, 0, 255];
        for id in accesses {
            sequential.acquire(0, id).unwrap();
        }
        let staged = grouped.acquire_retained(0, &accesses).unwrap().unwrap();
        assert_eq!(staged.len(), accesses.len());
        assert!(Arc::ptr_eq(&staged[0], &staged[2]));
        assert_eq!(grouped.telemetry(), sequential.telemetry());
        drop(staged);
        // The next eviction must choose the same victim after replaying duplicate accesses.
        for id in [1, 255, 0] {
            grouped.acquire(0, id).unwrap();
            sequential.acquire(0, id).unwrap();
            assert_eq!(grouped.telemetry(), sequential.telemetry());
        }
        drop(grouped);
        drop(sequential);

        // A caller may still hold an uncached or evicted expert. Neither replacement may
        // overwrite those buffers, and later misses must still work once the caller drops it.
        for slots in [0, 1] {
            let mut store = new_store(slots);
            let held = store.acquire(0, 255).unwrap();
            for id in [0, 1, 0] {
                let selected = store.acquire(0, id).unwrap();
                assert!(selected
                    .value
                    .forward(&input)
                    .unwrap()
                    .iter()
                    .all(|&x| x == 0.0));
                assert_eq!(held.value.forward(&input).unwrap(), output);
                assert_eq!(
                    store.telemetry().resident_bytes,
                    slots as u64 * 3 * matrix_bytes
                );
            }
            drop(held);
            assert_eq!(
                store
                    .acquire(0, 255)
                    .unwrap()
                    .value
                    .forward(&input)
                    .unwrap(),
                output
            );
        }

        // Failure after taking the spare must preserve the cached value and logical LRU
        // counters. Corrupt only the synthetic fixture, then repair it to verify recovery.
        let mut store = new_store(1);
        store.acquire(0, 0).unwrap();
        store.acquire(0, 255).unwrap();
        assert!(store.recycled.is_some());
        let before = store.telemetry().clone();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(directory.join("packed.safetensors"))
            .unwrap();
        file.seek(SeekFrom::Start(start + 2 * matrix_bytes))
            .unwrap();
        file.write_all(&0x7f80u16.to_le_bytes()).unwrap(); // BF16 infinity.
        let mut planned = new_store(1);
        planned.acquire(0, 0).unwrap();
        planned.acquire(0, 255).unwrap();
        let schedule = planned.grouped_access_order(0, &[255, 1, 255], 1).unwrap();
        let mut prefetched = planned.prefetch_grouped(&schedule);
        planned
            .acquire_grouped(0, 255, &schedule, &mut prefetched)
            .unwrap();
        assert!(planned.recycled.is_some());
        let planned_before = planned.telemetry().clone();
        assert!(planned
            .acquire_grouped(0, 1, &schedule, &mut prefetched)
            .is_err());
        assert_eq!(planned.telemetry(), &planned_before);
        assert!(planned.cache.contains(0, 255));
        assert_eq!(
            planned
                .cache
                .peek(0, 255)
                .unwrap()
                .value
                .forward(&input)
                .unwrap(),
            output
        );
        drop(planned);
        let mut initial = new_store(1);
        let schedule = Qwen36GroupedAccess::Initial(
            initial
                .cache
                .grouped_access_plan(0, &[255, 1, 255, 1])
                .unwrap(),
        );
        let mut prefetched = initial.prefetch_grouped(&schedule);
        assert!(prefetched.pending.get(&1).unwrap().is_err());
        assert_eq!(initial.telemetry(), &ExpertTelemetry::default());
        initial
            .acquire_grouped(0, 255, &schedule, &mut prefetched)
            .unwrap();
        let before_failed_admission = initial.telemetry().clone();
        assert!(initial
            .acquire_grouped(0, 1, &schedule, &mut prefetched)
            .is_err());
        assert_eq!(initial.telemetry(), &before_failed_admission);
        assert!(initial.cache.layer_is_empty(0));
        assert_eq!(
            initial
                .recycled
                .as_ref()
                .unwrap()
                .value
                .forward(&input)
                .unwrap(),
            output
        );
        drop(initial);
        assert!(store.acquire(0, 1).is_err());
        assert_eq!(store.telemetry(), &before);
        assert!(store.cache.contains(0, 255));
        assert_eq!(
            store
                .acquire(0, 255)
                .unwrap()
                .value
                .forward(&input)
                .unwrap(),
            output
        );
        assert_eq!(store.telemetry().hits, before.hits + 1);
        file.seek(SeekFrom::Start(start + 2 * matrix_bytes))
            .unwrap();
        file.write_all(&0u16.to_le_bytes()).unwrap();
        assert!(store
            .acquire(0, 1)
            .unwrap()
            .value
            .forward(&input)
            .unwrap()
            .iter()
            .all(|&x| x == 0.0));
        drop(file);
        drop(store);
        drop(index);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
