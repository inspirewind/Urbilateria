//! GLM expert checkpoint locator and execution cache.

use crate::model::{GatedMlp, MlpError};
use crate::profiling::{span, ProfileStage};
use crate::runtime::ExpertTelemetry;
use crate::storage::{
    inspect_weight_matrix, load_weight_matrices, SafetensorError, TensorIndex, WeightLoadError,
};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum ExpertStoreError {
    InvalidGeometry(String),
    InvalidLayer {
        layer: usize,
        layers: usize,
    },
    InvalidExpert {
        expert: usize,
        experts: usize,
    },
    InvalidInput {
        expected: usize,
        got: usize,
    },
    Budget {
        layer: usize,
        expert: usize,
        required: u64,
        maximum: u64,
    },
    Checkpoint(SafetensorError),
    Load(WeightLoadError),
    Execute(MlpError),
}

impl fmt::Display for ExpertStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(reason) => write!(f, "invalid expert store geometry: {reason}"),
            Self::InvalidLayer { layer, layers } => {
                write!(f, "expert layer {layer} is outside 0..{layers}")
            }
            Self::InvalidExpert { expert, experts } => {
                write!(f, "expert ID {expert} is outside 0..{experts}")
            }
            Self::InvalidInput { expected, got } => {
                write!(f, "expert input needs {expected} values, got {got}")
            }
            Self::Budget {
                layer,
                expert,
                required,
                maximum,
            } => write!(
                f,
                "layer {layer} expert {expert} needs {required} bytes, per-expert limit is {maximum}"
            ),
            Self::Checkpoint(error) => error.fmt(f),
            Self::Load(error) => error.fmt(f),
            Self::Execute(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ExpertStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            Self::Load(error) => Some(error),
            Self::Execute(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SafetensorError> for ExpertStoreError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<WeightLoadError> for ExpertStoreError {
    fn from(value: WeightLoadError) -> Self {
        Self::Load(value)
    }
}

impl From<MlpError> for ExpertStoreError {
    fn from(value: MlpError) -> Self {
        Self::Execute(value)
    }
}

#[derive(Debug)]
struct CachedExpert {
    expert: usize,
    last_used: u64,
    bytes: u64,
    mlp: GatedMlp,
}

/// Synchronous deterministic per-layer LRU for exact routed experts.
///
/// A zero-slot cache remains valid: the selected expert is loaded, executed once, and dropped.
/// Cache state may change after an I/O failure, but model/KV state is owned elsewhere and can be
/// rolled back independently.
#[derive(Debug)]
pub struct ExpertStore {
    index: Arc<TensorIndex>,
    caches: Vec<Vec<CachedExpert>>,
    slots_per_layer: usize,
    experts_per_layer: usize,
    hidden_size: usize,
    intermediate_size: usize,
    maximum_expert_bytes: u64,
    clock: u64,
    telemetry: ExpertTelemetry,
}

impl ExpertStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: TensorIndex,
        num_layers: usize,
        experts_per_layer: usize,
        hidden_size: usize,
        intermediate_size: usize,
        slots_per_layer: usize,
        maximum_expert_bytes: u64,
    ) -> Result<Self, ExpertStoreError> {
        Self::new_shared(
            Arc::new(index),
            num_layers,
            experts_per_layer,
            hidden_size,
            intermediate_size,
            slots_per_layer,
            maximum_expert_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_shared(
        index: Arc<TensorIndex>,
        num_layers: usize,
        experts_per_layer: usize,
        hidden_size: usize,
        intermediate_size: usize,
        slots_per_layer: usize,
        maximum_expert_bytes: u64,
    ) -> Result<Self, ExpertStoreError> {
        if num_layers == 0
            || experts_per_layer == 0
            || hidden_size == 0
            || intermediate_size == 0
            || maximum_expert_bytes == 0
        {
            return Err(ExpertStoreError::InvalidGeometry(
                "layers, experts, dimensions, and byte limit must be non-zero".to_owned(),
            ));
        }
        Ok(Self {
            index,
            caches: (0..num_layers).map(|_| Vec::new()).collect(),
            slots_per_layer,
            experts_per_layer,
            hidden_size,
            intermediate_size,
            maximum_expert_bytes,
            clock: 0,
            telemetry: ExpertTelemetry::default(),
        })
    }

    pub fn telemetry(&self) -> &ExpertTelemetry {
        &self.telemetry
    }

    pub fn expert_count(&self) -> usize {
        self.experts_per_layer
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn reset_telemetry(&mut self) {
        let resident_experts = self.telemetry.resident_experts;
        let resident_bytes = self.telemetry.resident_bytes;
        self.telemetry = ExpertTelemetry {
            resident_experts,
            resident_bytes,
            ..ExpertTelemetry::default()
        };
    }

    pub fn execute(
        &mut self,
        layer: usize,
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, ExpertStoreError> {
        self.validate_request(layer, expert, input)?;
        self.clock = self.clock.checked_add(1).ok_or_else(|| {
            ExpertStoreError::InvalidGeometry("LRU clock overflowed u64".to_owned())
        })?;

        if let Some(position) = self.caches[layer]
            .iter()
            .position(|entry| entry.expert == expert)
        {
            self.telemetry.hits += 1;
            self.caches[layer][position].last_used = self.clock;
            let _profile = span(ProfileStage::GlmExpertCompute);
            return Ok(self.caches[layer][position].mlp.forward(input)?);
        }

        self.telemetry.misses += 1;
        if self.slots_per_layer == 0 {
            let (mlp, bytes) = {
                let mut profile = span(ProfileStage::GlmExpertLoad);
                let loaded = self.load(layer, expert)?;
                profile.add_logical_bytes(loaded.1);
                loaded
            };
            self.telemetry.bytes_read = self.telemetry.bytes_read.saturating_add(bytes);
            let _profile = span(ProfileStage::GlmExpertCompute);
            return Ok(mlp.forward(input)?);
        }

        if self.caches[layer].len() == self.slots_per_layer {
            let position = self.caches[layer]
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| (entry.last_used, entry.expert))
                .map(|(position, _)| position)
                .expect("a full non-zero cache has an entry");
            let evicted = self.caches[layer].remove(position);
            self.telemetry.evictions += 1;
            self.telemetry.resident_experts -= 1;
            self.telemetry.resident_bytes =
                self.telemetry.resident_bytes.saturating_sub(evicted.bytes);
        }

        let (mlp, bytes) = {
            let mut profile = span(ProfileStage::GlmExpertLoad);
            let loaded = self.load(layer, expert)?;
            profile.add_logical_bytes(loaded.1);
            loaded
        };
        self.telemetry.bytes_read = self.telemetry.bytes_read.saturating_add(bytes);
        self.telemetry.resident_experts += 1;
        self.telemetry.resident_bytes = self.telemetry.resident_bytes.saturating_add(bytes);
        self.caches[layer].push(CachedExpert {
            expert,
            last_used: self.clock,
            bytes,
            mlp,
        });
        let position = self.caches[layer].len() - 1;
        let _profile = span(ProfileStage::GlmExpertCompute);
        Ok(self.caches[layer][position].mlp.forward(input)?)
    }

    fn validate_request(
        &self,
        layer: usize,
        expert: usize,
        input: &[f32],
    ) -> Result<(), ExpertStoreError> {
        if layer >= self.caches.len() {
            return Err(ExpertStoreError::InvalidLayer {
                layer,
                layers: self.caches.len(),
            });
        }
        if expert >= self.experts_per_layer {
            return Err(ExpertStoreError::InvalidExpert {
                expert,
                experts: self.experts_per_layer,
            });
        }
        if input.len() != self.hidden_size {
            return Err(ExpertStoreError::InvalidInput {
                expected: self.hidden_size,
                got: input.len(),
            });
        }
        Ok(())
    }

    fn load(&self, layer: usize, expert: usize) -> Result<(GatedMlp, u64), ExpertStoreError> {
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let gate_name = format!("{prefix}.gate_proj.weight");
        let up_name = format!("{prefix}.up_proj.weight");
        let down_name = format!("{prefix}.down_proj.weight");
        let mut bytes = 0u64;
        for (name, rows, cols) in [
            (&gate_name, self.intermediate_size, self.hidden_size),
            (&up_name, self.intermediate_size, self.hidden_size),
            (&down_name, self.hidden_size, self.intermediate_size),
        ] {
            bytes = bytes
                .checked_add(inspect_weight_matrix(&self.index, name, rows, cols)?.resident_bytes)
                .ok_or_else(|| {
                    ExpertStoreError::InvalidGeometry("expert bytes overflow".to_owned())
                })?;
        }
        if bytes > self.maximum_expert_bytes {
            return Err(ExpertStoreError::Budget {
                layer,
                expert,
                required: bytes,
                maximum: self.maximum_expert_bytes,
            });
        }
        let matrices = load_weight_matrices(
            &self.index,
            &[
                (gate_name.as_str(), self.intermediate_size, self.hidden_size),
                (up_name.as_str(), self.intermediate_size, self.hidden_size),
                (down_name.as_str(), self.hidden_size, self.intermediate_size),
            ],
            self.maximum_expert_bytes,
        )?;
        let mut matrices = matrices.into_iter();
        let gate = matrices.next().expect("three requested expert matrices");
        let up = matrices.next().expect("three requested expert matrices");
        let down = matrices.next().expect("three requested expert matrices");
        Ok((GatedMlp::new_mixed(gate, up, down)?, bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "urbilateria_expert_store_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    fn write_experts(path: &std::path::Path) {
        let mut tensors = BTreeMap::new();
        let mut payload = Vec::new();
        for expert in 0..2 {
            for projection in ["down_proj", "gate_proj", "up_proj"] {
                let name = format!("model.layers.0.mlp.experts.{expert}.{projection}.weight");
                let start = payload.len();
                payload.extend([0x99, 0x99]);
                tensors.insert(
                    name.clone(),
                    serde_json::json!({"dtype":"U8", "shape":[2], "data_offsets":[start,payload.len()]}),
                );
                let scale_start = payload.len();
                payload.extend(1.0f32.to_le_bytes());
                payload.extend(1.0f32.to_le_bytes());
                tensors.insert(
                    format!("{name}.qs"),
                    serde_json::json!({"dtype":"F32", "shape":[2], "data_offsets":[scale_start,payload.len()]}),
                );
            }
        }
        let mut header = serde_json::to_vec(&tensors).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(payload);
        fs::write(path, output).unwrap();
    }

    #[test]
    fn deterministic_lru_records_hits_misses_bytes_and_evictions() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_experts(&dir.join("experts.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let mut store = ExpertStore::new(index, 1, 2, 2, 2, 1, 64).unwrap();
        let first = store.execute(0, 0, &[0.5, -0.25]).unwrap();
        let hit = store.execute(0, 0, &[0.5, -0.25]).unwrap();
        let second = store.execute(0, 1, &[0.5, -0.25]).unwrap();
        assert_eq!(first, hit);
        assert_eq!(first, second);
        assert_eq!(
            store.telemetry(),
            &ExpertTelemetry {
                hits: 1,
                misses: 2,
                evictions: 1,
                bytes_read: 60,
                resident_experts: 1,
                resident_bytes: 30,
            }
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_slots_executes_without_retaining_an_expert() {
        let dir = fixture_dir();
        fs::create_dir_all(&dir).unwrap();
        write_experts(&dir.join("experts.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let mut store = ExpertStore::new(index, 1, 2, 2, 2, 0, 64).unwrap();
        store.execute(0, 0, &[1.0, 0.0]).unwrap();
        store.execute(0, 0, &[1.0, 0.0]).unwrap();
        assert_eq!(store.telemetry().hits, 0);
        assert_eq!(store.telemetry().misses, 2);
        assert_eq!(store.telemetry().resident_experts, 0);
        assert_eq!(store.telemetry().resident_bytes, 0);
        fs::remove_dir_all(dir).unwrap();
    }
}
