//! Model-agnostic deterministic per-layer LRU mechanics.

use serde::Serialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
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

#[derive(Debug)]
pub(crate) struct LayerLruCache<T> {
    layers: Vec<Vec<Entry<T>>>,
    layer_capacities: Vec<usize>,
    capacity_budget: usize,
    clock: u64,
}

impl<T> LayerLruCache<T> {
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
