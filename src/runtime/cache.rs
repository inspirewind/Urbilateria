//! Model-agnostic deterministic per-layer LRU mechanics.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertTelemetry {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
    pub resident_experts: usize,
    pub resident_bytes: u64,
}

impl ExpertTelemetry {
    pub fn accesses(&self) -> u64 {
        self.hits.saturating_add(self.misses)
    }

    /// Fraction of expert-cache accesses served without loading an expert.
    ///
    /// `None` distinguishes a runtime that made no routed-expert accesses from a measured 0%
    /// cache hit rate.
    pub fn hit_rate(&self) -> Option<f64> {
        let accesses = self.accesses();
        (accesses != 0).then(|| self.hits as f64 / accesses as f64)
    }
}

#[derive(Debug)]
struct Entry<T> {
    key: usize,
    last_used: u64,
    bytes: u64,
    value: T,
}

/// A read-only schedule for a single layer, consumed without interleaved cache accesses.
/// Resident selected keys come first; only the original sequence's final survivors are admitted.
#[derive(Debug)]
pub(crate) struct GroupedAccessPlan {
    layer: usize,
    order: Vec<usize>,
    retained: Vec<(usize, u64)>,
    initial_clock: u64,
    final_clock: u64,
}

impl GroupedAccessPlan {
    pub(crate) fn order(&self) -> &[usize] {
        &self.order
    }

    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    pub(crate) fn retained_keys(&self) -> impl Iterator<Item = usize> + '_ {
        self.retained.iter().map(|&(key, _)| key)
    }

    fn retained_position(&self, key: usize) -> Result<usize, usize> {
        self.retained.binary_search_by_key(&key, |&(key, _)| key)
    }
}

#[derive(Debug)]
pub(crate) struct LayerLruCache<T> {
    layers: Vec<Vec<Entry<T>>>,
    layer_capacities: Vec<usize>,
    capacity_budget: usize,
    clock: u64,
}

impl<T> LayerLruCache<T> {
    pub(crate) fn residency(&self) -> Vec<super::telemetry::LayerCache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer, entries)| super::telemetry::LayerCache {
                layer,
                experts: entries.len(),
                capacity: self.layer_capacities[layer],
                bytes: entries
                    .iter()
                    .fold(0u64, |sum, entry| sum.saturating_add(entry.bytes)),
            })
            .collect()
    }

    pub(crate) fn new(layers: usize, slots_per_layer: usize) -> Self {
        Self::with_layer_capacities(vec![slots_per_layer; layers])
    }

    pub(crate) fn with_layer_capacities(layer_capacities: Vec<usize>) -> Self {
        let capacity_budget = layer_capacities.iter().copied().sum();
        Self {
            layers: (0..layer_capacities.len()).map(|_| Vec::new()).collect(),
            layer_capacities,
            capacity_budget,
            clock: 0,
        }
    }

    pub(crate) fn access<E, R>(
        &mut self,
        telemetry: &mut ExpertTelemetry,
        layer: usize,
        key: usize,
        load: impl FnOnce() -> Result<(T, u64, u64), E>,
        execute: impl FnOnce(&mut T) -> Result<R, E>,
    ) -> Result<R, E> {
        self.access_with_evicted(telemetry, layer, key, load, execute)
            .map(|(result, _evicted)| result)
    }

    /// Equivalent to [`Self::access`], but returns ownership of an entry displaced by a miss.
    /// Callers with same-shaped values can recycle its backing allocation after all readers drop.
    pub(crate) fn access_with_evicted<E, R>(
        &mut self,
        telemetry: &mut ExpertTelemetry,
        layer: usize,
        key: usize,
        load: impl FnOnce() -> Result<(T, u64, u64), E>,
        execute: impl FnOnce(&mut T) -> Result<R, E>,
    ) -> Result<(R, Option<T>), E> {
        self.clock = self.clock.saturating_add(1);
        if let Some(position) = self.layers[layer].iter().position(|entry| entry.key == key) {
            telemetry.hits = telemetry.hits.saturating_add(1);
            let entry = &mut self.layers[layer][position];
            entry.last_used = self.clock;
            return execute(&mut entry.value).map(|result| (result, None));
        }

        telemetry.misses = telemetry.misses.saturating_add(1);
        let layer_capacity = self.layer_capacities[layer];
        if layer_capacity == 0 {
            let (mut value, _resident_bytes, read_bytes) = load()?;
            telemetry.bytes_read = telemetry.bytes_read.saturating_add(read_bytes);
            let result = execute(&mut value)?;
            return Ok((result, Some(value)));
        }
        let evicted = if self.layers[layer].len() == layer_capacity {
            let position = self.layers[layer]
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| (entry.last_used, entry.key))
                .map(|(position, _)| position)
                .expect("a full non-zero cache contains an entry");
            let entry = self.layers[layer].remove(position);
            telemetry.evictions = telemetry.evictions.saturating_add(1);
            telemetry.resident_experts = telemetry.resident_experts.saturating_sub(1);
            telemetry.resident_bytes = telemetry.resident_bytes.saturating_sub(entry.bytes);
            Some(entry.value)
        } else {
            None
        };
        let (value, bytes, read_bytes) = load()?;
        telemetry.bytes_read = telemetry.bytes_read.saturating_add(read_bytes);
        telemetry.resident_experts = telemetry.resident_experts.saturating_add(1);
        telemetry.resident_bytes = telemetry.resident_bytes.saturating_add(bytes);
        self.layers[layer].push(Entry {
            key,
            last_used: self.clock,
            bytes,
            value,
        });
        let position = self.layers[layer].len() - 1;
        execute(&mut self.layers[layer][position].value).map(|result| (result, evicted))
    }

    pub(crate) fn contains(&self, layer: usize, key: usize) -> bool {
        self.layers[layer].iter().any(|entry| entry.key == key)
    }

    /// Returns whether a layer has no resident entries.
    ///
    /// A layer-major prefill backend can use this to select a grouped streaming path whose
    /// access order is chosen up front. Starting from an empty layer makes it possible to retain
    /// exactly the same final LRU hot set by processing distinct keys in last-use order.
    pub(crate) fn layer_is_empty(&self, layer: usize) -> bool {
        self.layers[layer].is_empty()
    }

    pub(crate) fn layer_capacity(&self, layer: usize) -> usize {
        self.layer_capacities[layer]
    }

    /// Evicts every resident entry from one layer without changing its configured capacity.
    ///
    /// This supports bounded chunk scheduling: a caller can finish all work that references one
    /// expert chunk, recycle its backing buffers, and use the same slots for the next chunk.
    pub(crate) fn drain_layer(&mut self, telemetry: &mut ExpertTelemetry, layer: usize) -> Vec<T> {
        let entries = std::mem::take(&mut self.layers[layer]);
        telemetry.evictions = telemetry.evictions.saturating_add(entries.len() as u64);
        telemetry.resident_experts = telemetry.resident_experts.saturating_sub(entries.len());
        let bytes = entries
            .iter()
            .fold(0u64, |sum, entry| sum.saturating_add(entry.bytes));
        telemetry.resident_bytes = telemetry.resident_bytes.saturating_sub(bytes);
        entries.into_iter().map(|entry| entry.value).collect()
    }

    /// Borrows a resident value without changing recency or telemetry. Callers can use this to
    /// stage independent work, then replay the logical access through [`Self::access`] once an
    /// associated fallible batch has succeeded.
    pub(crate) fn peek(&self, layer: usize, key: usize) -> Option<&T> {
        self.layers[layer]
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }

    /// Returns whether every currently missing key can be inserted without changing eviction
    /// semantics. Callers may safely preload such keys concurrently, then replay accesses in the
    /// original order through [`Self::access`].
    pub(crate) fn can_insert_without_eviction(&self, layer: usize, keys: &[usize]) -> bool {
        let layer_capacity = self.layer_capacities[layer];
        if layer_capacity == 0 {
            return false;
        }
        let mut missing = Vec::new();
        for &key in keys {
            if !self.contains(layer, key) && !missing.contains(&key) {
                missing.push(key);
            }
        }
        self.layers[layer].len().saturating_add(missing.len()) <= layer_capacity
    }

    /// Collapse repeated accesses in last-use order only when doing so adds no cache misses.
    /// The surviving keys have the same final LRU order as the complete sequence. A warm
    /// cache can nevertheless incur extra intermediate misses, so compare both schedules.
    pub(crate) fn grouped_access_order(&self, layer: usize, keys: &[usize]) -> Option<Vec<usize>> {
        // Once timestamps saturate, expert-ID tie breaking no longer represents recency.
        self.clock.checked_add(u64::try_from(keys.len()).ok()?)?;
        let mut grouped = Vec::new();
        for &key in keys.iter().rev() {
            if !grouped.contains(&key) {
                grouped.push(key);
            }
        }
        if grouped.len() == keys.len() {
            return None;
        }
        grouped.reverse();
        (self.projected_misses(layer, &grouped) <= self.projected_misses(layer, keys))
            .then_some(grouped)
    }

    /// After every grouped load succeeds, recover the complete sequence's timestamps without
    /// inventing cache hits or reads. This also preserves subsequent clock-saturation behavior.
    pub(crate) fn finish_grouped_accesses(&mut self, layer: usize, keys: &[usize], groups: usize) {
        let initial_clock = self.clock - groups as u64;
        self.clock = initial_clock + keys.len() as u64;
        for entry in &mut self.layers[layer] {
            if let Some(position) = keys.iter().rposition(|&key| key == entry.key) {
                entry.last_used = initial_clock + position as u64 + 1;
            }
        }
    }

    /// Preserve all initially resident selected keys until their computation has finished,
    /// then admit only the ordered sequence's final hot set. This reaches the lower bound
    /// of one load per initially absent distinct key without changing the final cache state.
    pub(crate) fn grouped_access_plan(
        &self,
        layer: usize,
        keys: &[usize],
    ) -> Option<GroupedAccessPlan> {
        if self.layer_capacities[layer] == 0 || keys.is_empty() {
            return None;
        }
        let final_clock = self.clock.checked_add(u64::try_from(keys.len()).ok()?)?;
        let mut order = Vec::new();
        for &key in keys.iter().rev() {
            if !order.contains(&key) {
                order.push(key);
            }
        }
        order.reverse();
        let (mut retained, original_misses) = self.projected_accesses(layer, keys);
        let minimum_misses = order
            .iter()
            .filter(|&&key| !self.contains(layer, key))
            .count();
        if order.len() == keys.len() && minimum_misses == original_misses {
            return None;
        }
        // Stable partition keeps last-use order within hits and misses. Once all resident
        // keys have been used, unprotected entries can be evicted without another load.
        order.sort_by_key(|&key| !self.contains(layer, key));
        retained.sort_unstable_by_key(|&(key, _)| key);
        Some(GroupedAccessPlan {
            layer,
            order,
            retained,
            initial_clock: self.clock,
            final_clock,
        })
    }

    /// Access one scheduled group. A non-final miss stays transient; a final miss may evict
    /// only an entry absent from the original final hot set. The returned transient/displaced
    /// value can be recycled before the next miss. Loading succeeds before any eviction or
    /// telemetry mutation on a miss.
    pub(crate) fn access_grouped_with_evicted<E, R>(
        &mut self,
        telemetry: &mut ExpertTelemetry,
        plan: &GroupedAccessPlan,
        key: usize,
        load: impl FnOnce() -> Result<(T, u64, u64), E>,
        execute: impl FnOnce(&mut T) -> Result<R, E>,
    ) -> Result<(R, Option<T>), E> {
        let layer = plan.layer;
        if self.contains(layer, key) {
            return self.access_with_evicted(telemetry, layer, key, load, execute);
        }
        let (mut value, bytes, read_bytes) = load()?;
        self.clock = self.clock.saturating_add(1);
        telemetry.misses = telemetry.misses.saturating_add(1);
        telemetry.bytes_read = telemetry.bytes_read.saturating_add(read_bytes);
        if plan.retained_position(key).is_err() {
            let result = execute(&mut value)?;
            return Ok((result, Some(value)));
        }
        let evicted = if self.layers[layer].len() == self.layer_capacities[layer] {
            let position = self.layers[layer]
                .iter()
                .enumerate()
                .filter(|(_, entry)| plan.retained_position(entry.key).is_err())
                .min_by_key(|(_, entry)| (entry.last_used, entry.key))
                .map(|(position, _)| position)
                .expect("an absent final key leaves an unprotected cache slot");
            let entry = self.layers[layer].remove(position);
            telemetry.evictions = telemetry.evictions.saturating_add(1);
            telemetry.resident_experts = telemetry.resident_experts.saturating_sub(1);
            telemetry.resident_bytes = telemetry.resident_bytes.saturating_sub(entry.bytes);
            Some(entry.value)
        } else {
            None
        };
        telemetry.resident_experts = telemetry.resident_experts.saturating_add(1);
        telemetry.resident_bytes = telemetry.resident_bytes.saturating_add(bytes);
        self.layers[layer].push(Entry {
            key,
            last_used: self.clock,
            bytes,
            value,
        });
        let position = self.layers[layer].len() - 1;
        execute(&mut self.layers[layer][position].value).map(|result| (result, evicted))
    }

    /// Publish the original clock and timestamps after every scheduled load succeeds.
    pub(crate) fn finish_grouped_plan(&mut self, plan: &GroupedAccessPlan) {
        debug_assert_eq!(self.clock, plan.initial_clock + plan.order.len() as u64);
        debug_assert_eq!(self.layers[plan.layer].len(), plan.retained.len());
        for entry in &mut self.layers[plan.layer] {
            let position = plan
                .retained_position(entry.key)
                .expect("only final keys survive");
            entry.last_used = plan.retained[position].1;
        }
        self.clock = plan.final_clock;
    }

    fn projected_misses(&self, layer: usize, keys: &[usize]) -> usize {
        self.projected_accesses(layer, keys).1
    }

    fn projected_accesses(&self, layer: usize, keys: &[usize]) -> (Vec<(usize, u64)>, usize) {
        let capacity = self.layer_capacities[layer];
        if capacity == 0 {
            return (Vec::new(), keys.len());
        }
        let mut entries = self.layers[layer]
            .iter()
            .map(|entry| (entry.key, entry.last_used))
            .collect::<Vec<_>>();
        let mut clock = self.clock;
        let mut misses = 0;
        for &key in keys {
            clock = clock.saturating_add(1);
            if let Some(entry) = entries.iter_mut().find(|entry| entry.0 == key) {
                entry.1 = clock;
                continue;
            }
            misses += 1;
            if entries.len() == capacity {
                let victim = entries
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &(key, used))| (used, key))
                    .map(|(index, _)| index)
                    .expect("a full non-zero cache contains an entry");
                entries.remove(victim);
            }
            entries.push((key, clock));
        }
        (entries, misses)
    }

    /// Makes an ordered, duplicate-free batch safe to preload concurrently while preserving the
    /// exact hit/miss and final-LRU behavior of replaying its accesses one by one.
    ///
    /// We first simulate the ordered accesses against the current recencies, then remove only the
    /// original entries that the serial execution would evict. A duplicate-free batch no larger
    /// than the cache cannot evict a value inserted earlier in that same batch: every insertion has
    /// a newer timestamp than an as-yet untouched original victim. Replaying the real accesses can
    /// therefore fill the resulting holes without eviction, including reloading an initially
    /// resident key when an earlier miss would have displaced it before its turn.
    pub(crate) fn prepare_ordered_batch(
        &mut self,
        telemetry: &mut ExpertTelemetry,
        layer: usize,
        keys: &[usize],
    ) -> Option<Vec<T>> {
        let layer_capacity = self.layer_capacities[layer];
        if layer_capacity == 0
            || keys.is_empty()
            || keys.len() > layer_capacity
            || keys
                .iter()
                .enumerate()
                .any(|(index, key)| keys[..index].contains(key))
        {
            return None;
        }

        let mut simulated = self.layers[layer]
            .iter()
            .map(|entry| (entry.key, entry.last_used))
            .collect::<Vec<_>>();
        let mut clock = self.clock;
        let mut evicted_keys = Vec::new();
        for &key in keys {
            clock = clock.saturating_add(1);
            if let Some(position) = simulated
                .iter()
                .position(|&(resident_key, _)| resident_key == key)
            {
                simulated[position].1 = clock;
                continue;
            }
            if simulated.len() == layer_capacity {
                let position = simulated
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &(resident_key, last_used))| (last_used, resident_key))
                    .map(|(position, _)| position)
                    .expect("a full non-zero simulated cache contains an entry");
                evicted_keys.push(simulated.remove(position).0);
            }
            simulated.push((key, clock));
        }
        if evicted_keys
            .iter()
            .any(|key| !self.layers[layer].iter().any(|entry| entry.key == *key))
        {
            return None;
        }

        let mut evicted = Vec::with_capacity(evicted_keys.len());
        for key in evicted_keys {
            let position = self.layers[layer]
                .iter()
                .position(|entry| entry.key == key)
                .expect("every simulated eviction refers to an original entry");
            let entry = self.layers[layer].remove(position);
            telemetry.evictions = telemetry.evictions.saturating_add(1);
            telemetry.resident_experts = telemetry.resident_experts.saturating_sub(1);
            telemetry.resident_bytes = telemetry.resident_bytes.saturating_sub(entry.bytes);
            evicted.push(entry.value);
        }
        debug_assert!(self.can_insert_without_eviction(layer, keys));
        Some(evicted)
    }

    /// Temporarily lends a layer every globally budgeted slot not currently occupied by another
    /// layer. This is safe for layer-major prefill, where later layers have not populated their
    /// caches yet. The returned steady-state capacity must be passed to
    /// [`Self::restore_layer_capacity`] after the layer finishes.
    pub(crate) fn lend_unused_capacity(&mut self, layer: usize, maximum: usize) -> usize {
        let steady_capacity = self.layer_capacities[layer];
        let occupied_elsewhere = self
            .layers
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != layer)
            .map(|(_, entries)| entries.len())
            .sum::<usize>();
        let available = self.capacity_budget.saturating_sub(occupied_elsewhere);
        self.layer_capacities[layer] = available.min(maximum);
        debug_assert!(self.layer_capacities[layer] >= self.layers[layer].len());
        steady_capacity
    }

    /// Restores a temporarily enlarged layer and returns displaced values in oldest-first order.
    /// Trimming an enlarged LRU to N entries retains exactly the same N most recently distinct
    /// keys that a capacity-N LRU would contain after the same access stream.
    pub(crate) fn restore_layer_capacity(
        &mut self,
        telemetry: &mut ExpertTelemetry,
        layer: usize,
        capacity: usize,
    ) -> Vec<T> {
        self.layer_capacities[layer] = capacity;
        let mut evicted = Vec::with_capacity(self.layers[layer].len().saturating_sub(capacity));
        while self.layers[layer].len() > capacity {
            let position = self.layers[layer]
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| (entry.last_used, entry.key))
                .map(|(position, _)| position)
                .expect("an oversized cache contains an entry");
            let entry = self.layers[layer].remove(position);
            telemetry.evictions = telemetry.evictions.saturating_add(1);
            telemetry.resident_experts = telemetry.resident_experts.saturating_sub(1);
            telemetry.resident_bytes = telemetry.resident_bytes.saturating_sub(entry.bytes);
            evicted.push(entry.value);
        }
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_policy_is_independent_of_cached_value_type() {
        let mut cache = LayerLruCache::new(1, 1);
        let mut telemetry = ExpertTelemetry::default();
        let first = cache
            .access(
                &mut telemetry,
                0,
                7,
                || Ok::<_, ()>((10, 4, 3)),
                |value| Ok(*value),
            )
            .unwrap();
        let hit = cache
            .access(
                &mut telemetry,
                0,
                7,
                || Ok::<_, ()>((20, 4, 3)),
                |value| Ok(*value),
            )
            .unwrap();
        let second = cache
            .access(
                &mut telemetry,
                0,
                8,
                || Ok::<_, ()>((30, 8, 6)),
                |value| Ok(*value),
            )
            .unwrap();
        assert_eq!((first, hit, second), (10, 10, 30));
        assert_eq!(
            (telemetry.hits, telemetry.misses, telemetry.evictions),
            (1, 2, 1)
        );
        assert_eq!(telemetry.resident_bytes, 8);
        assert_eq!(telemetry.bytes_read, 9);
    }

    #[test]
    fn displaced_value_is_returned_without_changing_lru_accounting() {
        let mut cache = LayerLruCache::new(1, 1);
        let mut telemetry = ExpertTelemetry::default();
        cache
            .access(
                &mut telemetry,
                0,
                7,
                || Ok::<_, ()>((vec![1, 2, 3], 3, 3)),
                |_| Ok(()),
            )
            .unwrap();
        let (length, evicted) = cache
            .access_with_evicted(
                &mut telemetry,
                0,
                8,
                || Ok::<_, ()>((vec![4, 5], 2, 2)),
                |value| Ok(value.len()),
            )
            .unwrap();
        assert_eq!(length, 2);
        assert_eq!(evicted, Some(vec![1, 2, 3]));
        assert_eq!(telemetry.misses, 2);
        assert_eq!(telemetry.evictions, 1);
        assert_eq!(telemetry.resident_experts, 1);
        assert_eq!(telemetry.resident_bytes, 2);
        assert_eq!(telemetry.bytes_read, 5);
    }

    #[test]
    fn concurrent_preload_is_allowed_only_without_eviction() {
        let mut cache = LayerLruCache::new(1, 1);
        let mut telemetry = ExpertTelemetry::default();
        assert!(cache.can_insert_without_eviction(0, &[7, 7]));
        assert!(!cache.can_insert_without_eviction(0, &[7, 8]));
        cache
            .access(
                &mut telemetry,
                0,
                7,
                || Ok::<_, ()>((10, 4, 3)),
                |value| Ok(*value),
            )
            .unwrap();
        assert!(cache.can_insert_without_eviction(0, &[7]));
        assert!(!cache.can_insert_without_eviction(0, &[8]));
    }

    #[test]
    fn protected_grouping_reaches_minimum_reads_and_preserves_future_lru() {
        fn apply(
            cache: &mut LayerLruCache<usize>,
            telemetry: &mut ExpertTelemetry,
            keys: &[usize],
        ) {
            for &key in keys {
                cache
                    .access(
                        telemetry,
                        0,
                        key,
                        || Ok::<_, ()>((key, key as u64 + 1, 7)),
                        |value| Ok(*value),
                    )
                    .unwrap();
            }
        }
        fn snapshot(cache: &LayerLruCache<usize>) -> Vec<(usize, u64, usize)> {
            let mut values = cache.layers[0]
                .iter()
                .map(|e| (e.key, e.last_used, e.value))
                .collect::<Vec<_>>();
            values.sort_unstable_by_key(|e| e.0);
            values
        }
        for capacity in 1..=4 {
            for seed in [&[][..], &[0, 1], &[2, 0, 1, 2], &[3, 1, 0, 2]] {
                for encoded in 0..4096 {
                    let keys = (0..6).map(|i| (encoded >> (i * 2)) & 3).collect::<Vec<_>>();
                    let mut serial = LayerLruCache::new(1, capacity);
                    let mut grouped = LayerLruCache::new(1, capacity);
                    let mut a = ExpertTelemetry::default();
                    let mut b = ExpertTelemetry::default();
                    apply(&mut serial, &mut a, seed);
                    apply(&mut grouped, &mut b, seed);
                    let initial = snapshot(&grouped);
                    let initial_clock = grouped.clock;
                    let mut missing = keys
                        .iter()
                        .copied()
                        .filter(|&key| !grouped.contains(0, key))
                        .collect::<Vec<_>>();
                    missing.sort_unstable();
                    missing.dedup();
                    let plan = grouped.grouped_access_plan(0, &keys);
                    assert_eq!(snapshot(&grouped), initial);
                    assert_eq!(grouped.clock, initial_clock);
                    apply(&mut serial, &mut a, &keys);
                    if let Some(plan) = plan {
                        let before = b.clone();
                        for &key in plan.order() {
                            let (value, _retired) = grouped
                                .access_grouped_with_evicted(
                                    &mut b,
                                    &plan,
                                    key,
                                    || Ok::<_, ()>((key, key as u64 + 1, 7)),
                                    |value| Ok(*value),
                                )
                                .unwrap();
                            assert_eq!(value, key);
                            assert!(b.resident_experts <= capacity);
                        }
                        grouped.finish_grouped_plan(&plan);
                        assert_eq!(b.misses - before.misses, missing.len() as u64);
                        assert_eq!(b.bytes_read - before.bytes_read, missing.len() as u64 * 7);
                        assert_eq!(b.accesses() - before.accesses(), plan.order().len() as u64);
                        assert!(b.misses <= a.misses);
                        assert_eq!(a.resident_bytes, b.resident_bytes);
                        assert_eq!(a.resident_experts, b.resident_experts);
                        assert_eq!(snapshot(&serial), snapshot(&grouped));
                        assert_eq!(serial.clock, grouped.clock);
                        let old_a = a.clone();
                        let old_b = b.clone();
                        apply(&mut serial, &mut a, &[4, 1, 0, 4, 3]);
                        apply(&mut grouped, &mut b, &[4, 1, 0, 4, 3]);
                        assert_eq!(a.misses - old_a.misses, b.misses - old_b.misses);
                        assert_eq!(a.hits - old_a.hits, b.hits - old_b.hits);
                        assert_eq!(snapshot(&serial), snapshot(&grouped));
                    }
                }
            }
        }
        let mut serial = LayerLruCache::new(1, 2);
        let mut grouped = LayerLruCache::new(1, 2);
        let mut a = ExpertTelemetry::default();
        let mut b = ExpertTelemetry::default();
        apply(&mut serial, &mut a, &[0, 1]);
        apply(&mut grouped, &mut b, &[0, 1]);
        let keys = [0, 2, 0];
        assert!(grouped.grouped_access_order(0, &keys).is_none());
        assert_eq!(
            grouped.grouped_access_plan(0, &keys).unwrap().order(),
            &[0, 2]
        );
        grouped.clock = u64::MAX - 2;
        assert!(grouped.grouped_access_plan(0, &keys).is_none());
        serial.clock = u64::MAX - 3;
        grouped.clock = u64::MAX - 3;
        let plan = grouped.grouped_access_plan(0, &keys).unwrap();
        apply(&mut serial, &mut a, &keys);
        for &key in plan.order() {
            grouped
                .access_grouped_with_evicted(
                    &mut b,
                    &plan,
                    key,
                    || Ok::<_, ()>((key, key as u64 + 1, 7)),
                    |_| Ok(()),
                )
                .unwrap();
        }
        grouped.finish_grouped_plan(&plan);
        assert_eq!(grouped.clock, u64::MAX);
        assert_eq!(snapshot(&serial), snapshot(&grouped));
        apply(&mut serial, &mut a, &[3, 0, 3, 2, 1]);
        apply(&mut grouped, &mut b, &[3, 0, 3, 2, 1]);
        assert_eq!(snapshot(&serial), snapshot(&grouped));
        assert!(grouped.grouped_access_plan(0, &keys).is_none());
        assert!(LayerLruCache::<usize>::new(1, 0)
            .grouped_access_plan(0, &keys)
            .is_none());
    }

    #[test]
    fn protected_grouping_keeps_one_transient_value_and_preserves_cache_on_failure() {
        use std::cell::Cell;
        use std::rc::Rc;
        #[derive(Debug)]
        struct Value {
            key: usize,
            live: Rc<Cell<usize>>,
        }
        impl Drop for Value {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
            }
        }
        let live = Rc::new(Cell::new(0));
        let make = |key| {
            live.set(live.get() + 1);
            Rc::new(Value {
                key,
                live: Rc::clone(&live),
            })
        };
        let mut cache = LayerLruCache::new(1, 2);
        let mut telemetry = ExpertTelemetry::default();
        for key in [0, 1] {
            cache
                .access(
                    &mut telemetry,
                    0,
                    key,
                    || Ok::<_, ()>((make(key), 11, 7)),
                    |_| Ok(()),
                )
                .unwrap();
        }
        // Resident 0 survives; 2 and 3 stream without admission; final miss 4 replaces 1.
        let keys = [0, 2, 3, 4, 0];
        let plan = cache.grouped_access_plan(0, &keys).unwrap();
        assert_eq!(plan.order(), &[0, 2, 3, 4]);
        for key in [2, 4] {
            let clock = cache.clock;
            let before = telemetry.clone();
            let entries = cache.layers[0]
                .iter()
                .map(|e| (e.key, e.last_used))
                .collect::<Vec<_>>();
            assert!(cache
                .access_grouped_with_evicted(
                    &mut telemetry,
                    &plan,
                    key,
                    || Err::<(Rc<Value>, u64, u64), _>(()),
                    |_| Ok(())
                )
                .is_err());
            assert_eq!(telemetry, before);
            assert_eq!(cache.clock, clock);
            assert_eq!(
                cache.layers[0]
                    .iter()
                    .map(|e| (e.key, e.last_used))
                    .collect::<Vec<_>>(),
                entries
            );
            assert_eq!(live.get(), 2);
        }
        let mut recycled = None;
        for &key in plan.order() {
            if !cache.contains(0, key) {
                drop(recycled.take());
            }
            let (selected, displaced) = cache
                .access_grouped_with_evicted(
                    &mut telemetry,
                    &plan,
                    key,
                    || {
                        let value = make(key);
                        assert!(live.get() <= 3);
                        Ok::<_, ()>((value, 11, 7))
                    },
                    |value| Ok(Rc::clone(value)),
                )
                .unwrap();
            assert_eq!(selected.key, key);
            assert!(live.get() <= 3);
            if displaced.is_some() {
                recycled = displaced;
            }
            drop(selected);
        }
        cache.finish_grouped_plan(&plan);
        assert!(cache.contains(0, 0) && cache.contains(0, 4));
        assert_eq!(telemetry.resident_experts, 2);
        assert_eq!(telemetry.misses, 5); // two seed loads and three distinct misses
        drop(recycled);
        drop(cache);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn grouped_accesses_preserve_lru_and_never_add_projected_misses() {
        fn apply(
            cache: &mut LayerLruCache<usize>,
            telemetry: &mut ExpertTelemetry,
            keys: &[usize],
        ) {
            for &key in keys {
                cache
                    .access(
                        telemetry,
                        0,
                        key,
                        || Ok::<_, ()>((key, key as u64 + 1, 7)),
                        |_| Ok(()),
                    )
                    .unwrap();
            }
        }
        fn recency(cache: &LayerLruCache<usize>) -> Vec<usize> {
            let mut entries = cache.layers[0].iter().collect::<Vec<_>>();
            entries.sort_by_key(|entry| (entry.last_used, entry.key));
            entries.into_iter().map(|entry| entry.key).collect()
        }
        for capacity in 0..=4 {
            for seed in [&[][..], &[0, 1], &[2, 0, 1, 2], &[3, 1, 0, 2]] {
                for encoded in 0..4096 {
                    let keys = (0..6).map(|i| (encoded >> (i * 2)) & 3).collect::<Vec<_>>();
                    let mut serial = LayerLruCache::new(1, capacity);
                    let mut grouped = LayerLruCache::new(1, capacity);
                    let mut a = ExpertTelemetry::default();
                    let mut b = ExpertTelemetry::default();
                    apply(&mut serial, &mut a, seed);
                    apply(&mut grouped, &mut b, seed);
                    let initial = recency(&grouped);
                    let clock = grouped.clock;
                    let expected_misses = serial.projected_misses(0, &keys);
                    let order = grouped.grouped_access_order(0, &keys);
                    assert_eq!(recency(&grouped), initial);
                    assert_eq!(grouped.clock, clock);
                    let before = a.misses;
                    apply(&mut serial, &mut a, &keys);
                    assert_eq!(a.misses - before, expected_misses as u64);
                    if let Some(order) = order {
                        apply(&mut grouped, &mut b, &order);
                        grouped.finish_grouped_accesses(0, &keys, order.len());
                        assert!(b.misses <= a.misses);
                        assert_eq!(serial.clock, grouped.clock);
                        for entry in &serial.layers[0] {
                            assert_eq!(
                                grouped.layers[0]
                                    .iter()
                                    .find(|other| other.key == entry.key)
                                    .unwrap()
                                    .last_used,
                                entry.last_used,
                            );
                        }
                        assert_eq!(recency(&serial), recency(&grouped));
                        assert_eq!(a.resident_bytes, b.resident_bytes);
                        let old_a = a.clone();
                        let old_b = b.clone();
                        apply(&mut serial, &mut a, &[4, 1, 0, 4, 3]);
                        apply(&mut grouped, &mut b, &[4, 1, 0, 4, 3]);
                        assert_eq!(a.misses - old_a.misses, b.misses - old_b.misses);
                        assert_eq!(a.hits - old_a.hits, b.hits - old_b.hits);
                        assert_eq!(recency(&serial), recency(&grouped));
                    }
                }
            }
        }
        let mut cache = LayerLruCache::new(1, 2);
        let mut telemetry = ExpertTelemetry::default();
        apply(&mut cache, &mut telemetry, &[0, 1]);
        // Touching 0 early protects it: [0,2,0] has one miss, but [2,0] has two.
        assert_eq!(cache.projected_misses(0, &[0, 2, 0]), 1);
        assert_eq!(cache.projected_misses(0, &[2, 0]), 2);
        assert!(cache.grouped_access_order(0, &[0, 2, 0]).is_none());
        assert!(cache.grouped_access_order(0, &[0, 1]).is_none());
        cache.clock = u64::MAX - 1;
        assert!(cache.grouped_access_order(0, &[0, 0]).is_none());
        cache.clock = u64::MAX - 3;
        let keys = [0, 1, 0];
        let grouped = cache.grouped_access_order(0, &keys).unwrap();
        apply(&mut cache, &mut telemetry, &grouped);
        cache.finish_grouped_accesses(0, &keys, grouped.len());
        assert_eq!(cache.clock, u64::MAX);
        assert_eq!(
            cache.layers[0]
                .iter()
                .find(|entry| entry.key == 0)
                .unwrap()
                .last_used,
            u64::MAX
        );
        assert_eq!(
            cache.layers[0]
                .iter()
                .find(|entry| entry.key == 1)
                .unwrap()
                .last_used,
            u64::MAX - 1
        );
        apply(&mut cache, &mut telemetry, &[2]);
        assert!(cache.contains(0, 0));
        assert!(!cache.contains(0, 1));
    }

    #[test]
    fn ordered_batch_preparation_matches_serial_lru_and_telemetry() {
        fn seeded_cache() -> (LayerLruCache<usize>, ExpertTelemetry) {
            let mut cache = LayerLruCache::new(1, 4);
            let mut telemetry = ExpertTelemetry::default();
            for key in [1, 2, 3, 4] {
                cache
                    .access(
                        &mut telemetry,
                        0,
                        key,
                        || Ok::<_, ()>((key * 10, 4, 3)),
                        |_| Ok(()),
                    )
                    .unwrap();
            }
            (cache, telemetry)
        }

        let accesses = [5, 1, 6];
        let (mut serial, mut serial_telemetry) = seeded_cache();
        for &key in &accesses {
            serial
                .access(
                    &mut serial_telemetry,
                    0,
                    key,
                    || Ok::<_, ()>((key * 10, 4, 3)),
                    |_| Ok(()),
                )
                .unwrap();
        }

        let (mut prepared, mut prepared_telemetry) = seeded_cache();
        assert_eq!(
            prepared.prepare_ordered_batch(&mut prepared_telemetry, 0, &accesses),
            Some(vec![10, 20, 30])
        );
        assert!(prepared.can_insert_without_eviction(0, &accesses));
        for &key in &accesses {
            prepared
                .access(
                    &mut prepared_telemetry,
                    0,
                    key,
                    || Ok::<_, ()>((key * 10, 4, 3)),
                    |_| Ok(()),
                )
                .unwrap();
        }

        assert_eq!(prepared_telemetry, serial_telemetry);
        for key in 1..=6 {
            assert_eq!(prepared.contains(0, key), serial.contains(0, key));
        }
    }

    #[test]
    fn peek_does_not_change_recency_or_telemetry() {
        let mut cache = LayerLruCache::new(1, 2);
        let mut telemetry = ExpertTelemetry::default();
        for key in [1, 2] {
            cache
                .access(
                    &mut telemetry,
                    0,
                    key,
                    || Ok::<_, ()>((key * 10, 4, 3)),
                    |_| Ok(()),
                )
                .unwrap();
        }
        let before = telemetry.clone();
        assert_eq!(cache.peek(0, 1), Some(&10));
        assert_eq!(telemetry, before);
        cache
            .access(&mut telemetry, 0, 3, || Ok::<_, ()>((30, 4, 3)), |_| Ok(()))
            .unwrap();
        assert!(!cache.contains(0, 1));
    }

    #[test]
    fn layer_empty_state_tracks_residency() {
        let mut cache = LayerLruCache::new(2, 1);
        let mut telemetry = ExpertTelemetry::default();
        assert!(cache.layer_is_empty(0));
        assert!(cache.layer_is_empty(1));
        cache
            .access(&mut telemetry, 1, 7, || Ok::<_, ()>((70, 4, 3)), |_| Ok(()))
            .unwrap();
        assert!(cache.layer_is_empty(0));
        assert!(!cache.layer_is_empty(1));
        let before = telemetry.clone();
        assert_eq!(
            cache.residency(),
            vec![
                super::super::telemetry::LayerCache {
                    layer: 0,
                    experts: 0,
                    capacity: 1,
                    bytes: 0
                },
                super::super::telemetry::LayerCache {
                    layer: 1,
                    experts: 1,
                    capacity: 1,
                    bytes: 4
                },
            ]
        );
        assert_eq!(telemetry, before);
        cache
            .access(&mut telemetry, 1, 8, || Ok::<_, ()>((80, 9, 8)), |_| Ok(()))
            .unwrap();
        assert_eq!(cache.residency()[1].bytes, 9);
        assert_eq!(cache.residency()[1].experts, 1);
        cache.drain_layer(&mut telemetry, 1);
        assert_eq!(cache.residency()[1].bytes, 0);
        assert_eq!(cache.residency()[1].capacity, 1);
    }

    #[test]
    fn draining_a_layer_preserves_capacity_and_updates_residency() {
        let mut cache = LayerLruCache::new(1, 2);
        let mut telemetry = ExpertTelemetry::default();
        for key in [7, 8] {
            cache
                .access(
                    &mut telemetry,
                    0,
                    key,
                    || Ok::<_, ()>((key * 10, 4, 3)),
                    |_| Ok(()),
                )
                .unwrap();
        }
        assert_eq!(cache.layer_capacity(0), 2);
        assert_eq!(cache.drain_layer(&mut telemetry, 0), [70, 80]);
        assert!(cache.layer_is_empty(0));
        assert_eq!(cache.layer_capacity(0), 2);
        assert_eq!(telemetry.evictions, 2);
        assert_eq!(telemetry.resident_experts, 0);
        assert_eq!(telemetry.resident_bytes, 0);
    }

    #[test]
    fn per_layer_capacities_use_spare_slots_without_weakening_other_layers() {
        let mut cache = LayerLruCache::with_layer_capacities(vec![1, 2]);
        let mut telemetry = ExpertTelemetry::default();
        for (layer, key) in [(0, 1), (1, 1), (1, 2)] {
            cache
                .access(
                    &mut telemetry,
                    layer,
                    key,
                    || Ok::<_, ()>((key, 4, 3)),
                    |_| Ok(()),
                )
                .unwrap();
        }
        assert_eq!(telemetry.resident_experts, 3);
        assert!(cache.contains(0, 1));
        assert!(cache.contains(1, 1));
        assert!(cache.contains(1, 2));
    }

    #[test]
    fn unused_global_capacity_can_be_lent_then_trimmed_back_to_recent_entries() {
        let mut cache = LayerLruCache::with_layer_capacities(vec![2, 2]);
        let mut telemetry = ExpertTelemetry::default();
        let steady = cache.lend_unused_capacity(0, 10);
        assert_eq!(steady, 2);
        for key in [1, 2, 3, 4] {
            cache
                .access(
                    &mut telemetry,
                    0,
                    key,
                    || Ok::<_, ()>((key * 10, 4, 3)),
                    |_| Ok(()),
                )
                .unwrap();
        }
        assert_eq!(telemetry.resident_experts, 4);
        let evicted = cache.restore_layer_capacity(&mut telemetry, 0, steady);
        assert_eq!(evicted, vec![10, 20]);
        assert!(!cache.contains(0, 1));
        assert!(!cache.contains(0, 2));
        assert!(cache.contains(0, 3));
        assert!(cache.contains(0, 4));
        assert_eq!(telemetry.resident_experts, 2);
        assert_eq!(telemetry.evictions, 2);

        for key in [5, 6] {
            cache
                .access(
                    &mut telemetry,
                    1,
                    key,
                    || Ok::<_, ()>((key * 10, 4, 3)),
                    |_| Ok(()),
                )
                .unwrap();
        }
        let steady = cache.lend_unused_capacity(0, 10);
        assert_eq!(steady, 2);
        cache.restore_layer_capacity(&mut telemetry, 0, steady);
    }

    #[test]
    fn hit_rate_distinguishes_no_accesses_from_no_hits() {
        assert_eq!(ExpertTelemetry::default().hit_rate(), None);
        let telemetry = ExpertTelemetry {
            hits: 3,
            misses: 1,
            ..ExpertTelemetry::default()
        };
        assert_eq!(telemetry.accesses(), 4);
        assert_eq!(telemetry.hit_rate(), Some(0.75));
    }
}
