//! Packed BF16 routed-expert slice loading and deterministic per-layer caching.

use crate::execution::install;
use crate::model::{Bf16Matrix, WeightMatrix};
use crate::models::qwen3_8::expert::{
    Qwen38Expert as Qwen36Expert, Qwen38ExpertError as Qwen36ExpertError,
};
use crate::profiling::{span, ProfileStage};
use crate::runtime::cache::LayerLruCache;
use crate::runtime::ExpertTelemetry;
use crate::storage::{DType, TensorIndex, WeightLoadError};
use rayon::prelude::*;
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
        let load = |name: &str,
                    offset: usize,
                    rows,
                    cols|
         -> Result<WeightMatrix, Qwen36ExpertStoreError> {
            let bytes = index
                .read_range(name, offset as u64, matrix_bytes)
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
pub struct Qwen36ExpertStore {
    index: Arc<TensorIndex>,
    layers: usize,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    maximum_expert_bytes: u64,
    cache: LayerLruCache<Arc<Qwen36LoadedExpert>>,
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
            telemetry: ExpertTelemetry::default(),
        })
    }

    pub fn acquire(
        &mut self,
        layer: usize,
        expert: usize,
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
        let loaded = Qwen36LoadedExpert::load(
            &self.index,
            layer,
            expert,
            self.hidden,
            self.intermediate,
            self.maximum_expert_bytes,
        )?;
        let resident_bytes = loaded.resident_bytes();
        let payload_bytes = loaded.payload_bytes();
        let loaded = Arc::new(loaded);
        self.cache.access(
            &mut self.telemetry,
            layer,
            expert,
            || Ok((loaded, resident_bytes, payload_bytes)),
            |value| Ok(Arc::clone(value)),
        )
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
        let new_store = || {
            Qwen36ExpertStore::new_shared(
                Arc::clone(&index),
                1,
                256,
                2048,
                512,
                2,
                3 * matrix_bytes,
            )
            .unwrap()
        };
        let mut sequential = new_store();
        let mut grouped = new_store();
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
        drop(index);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
