//! Native Kimi-K3 routed-expert loading and deterministic per-layer caching.
//!
//! Each routed expert is stored as three MXFP4 matrices.  The loader validates all six
//! safetensors entries (packed weights plus E8M0 scale sidecars), authorizes the complete
//! resident allocation, and then asks the shared storage layer to read the batch in physical
//! shard order.  Cache misses are loaded before touching the LRU so a failed load cannot evict a
//! previously usable expert.

use crate::model::{WeightError, WeightMatrix};
use crate::runtime::cache::LayerLruCache;
use crate::runtime::ExpertTelemetry;
use crate::storage::{
    inspect_weight_matrix, load_weight_matrices, DType, TensorIndex, WeightFormat, WeightLoadError,
};
use std::fmt;
use std::sync::Arc;

/// Exact resident payload of one official `[3584, 3072]` Kimi-K3 routed expert.
pub const KIMI_K3_EXPERT_RESIDENT_BYTES: u64 = 17_547_264;

const MXFP4_GROUP_SIZE: usize = 32;

/// The six checkpoint entries that make up one Kimi-K3 routed expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiK3ExpertTensorNames {
    pub w1_packed: String,
    pub w1_scale: String,
    pub w2_packed: String,
    pub w2_scale: String,
    pub w3_packed: String,
    pub w3_scale: String,
}

impl KimiK3ExpertTensorNames {
    pub fn new(layer: usize, expert: usize) -> Self {
        let prefix =
            format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}");
        Self {
            w1_packed: format!("{prefix}.w1.weight_packed"),
            w1_scale: format!("{prefix}.w1.weight_scale"),
            w2_packed: format!("{prefix}.w2.weight_packed"),
            w2_scale: format!("{prefix}.w2.weight_scale"),
            w3_packed: format!("{prefix}.w3.weight_packed"),
            w3_scale: format!("{prefix}.w3.weight_scale"),
        }
    }

    /// Returns names in projection order, with each packed tensor immediately followed by its
    /// scale sidecar.
    pub fn all(&self) -> [&str; 6] {
        [
            &self.w1_packed,
            &self.w1_scale,
            &self.w2_packed,
            &self.w2_scale,
            &self.w3_packed,
            &self.w3_scale,
        ]
    }
}

#[derive(Debug)]
pub enum KimiK3ExpertError {
    InvalidGeometry(String),
    InvalidLayer {
        layer: usize,
        layers: usize,
    },
    InvalidExpert {
        layer: usize,
        expert: usize,
        experts: usize,
    },
    Tensor {
        layer: usize,
        expert: usize,
        name: String,
        source: WeightLoadError,
    },
    NativeFormat {
        layer: usize,
        expert: usize,
        name: String,
        reason: String,
    },
    Budget {
        layer: usize,
        expert: usize,
        name: String,
        required: u64,
        maximum: u64,
    },
    BatchLoad {
        layer: usize,
        expert: usize,
        name: String,
        source: WeightLoadError,
    },
    Projection {
        layer: usize,
        expert: usize,
        name: String,
        source: WeightError,
    },
}

impl fmt::Display for KimiK3ExpertError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(reason) => {
                write!(formatter, "invalid Kimi-K3 expert geometry: {reason}")
            }
            Self::InvalidLayer { layer, layers } => {
                write!(formatter, "Kimi-K3 expert layer {layer} is outside 0..{layers}")
            }
            Self::InvalidExpert {
                layer,
                expert,
                experts,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} expert {expert} is outside 0..{experts}"
            ),
            Self::Tensor {
                layer,
                expert,
                name,
                source,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} expert {expert} tensor {name:?}: {source}"
            ),
            Self::NativeFormat {
                layer,
                expert,
                name,
                reason,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} expert {expert} tensor {name:?} is not native MXFP4: {reason}"
            ),
            Self::Budget {
                layer,
                expert,
                name,
                required,
                maximum,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} expert {expert} batch beginning at {name:?} needs {required} resident bytes, per-expert limit is {maximum}"
            ),
            Self::BatchLoad {
                layer,
                expert,
                name,
                source,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} expert {expert} batch beginning at {name:?}: {source}"
            ),
            Self::Projection {
                layer,
                expert,
                name,
                source,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} expert {expert} projection {name:?}: {source}"
            ),
        }
    }
}

impl std::error::Error for KimiK3ExpertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor { source, .. } | Self::BatchLoad { source, .. } => Some(source),
            Self::Projection { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Three native MXFP4 projections for one Kimi-K3 routed expert.
#[derive(Debug)]
pub struct KimiK3Expert {
    layer: usize,
    expert: usize,
    names: KimiK3ExpertTensorNames,
    w1: WeightMatrix,
    w2: WeightMatrix,
    w3: WeightMatrix,
    resident_bytes: u64,
    payload_bytes: u64,
}

impl KimiK3Expert {
    /// Loads `w1`, `w2`, and `w3` together after a metadata-only aggregate budget check.
    ///
    /// The logical geometry is `w1/w3 = [moe_intermediate_size, latent_size]` and
    /// `w2 = [latent_size, moe_intermediate_size]`.
    pub fn load(
        index: &TensorIndex,
        layer: usize,
        expert: usize,
        latent_size: usize,
        moe_intermediate_size: usize,
        maximum_resident_bytes: u64,
    ) -> Result<Self, KimiK3ExpertError> {
        validate_dimensions(latent_size, moe_intermediate_size)?;
        if maximum_resident_bytes == 0 {
            return Err(KimiK3ExpertError::InvalidGeometry(
                "maximum resident bytes must be non-zero".to_owned(),
            ));
        }

        let names = KimiK3ExpertTensorNames::new(layer, expert);
        let specifications = [
            (
                names.w1_packed.as_str(),
                names.w1_scale.as_str(),
                moe_intermediate_size,
                latent_size,
            ),
            (
                names.w2_packed.as_str(),
                names.w2_scale.as_str(),
                latent_size,
                moe_intermediate_size,
            ),
            (
                names.w3_packed.as_str(),
                names.w3_scale.as_str(),
                moe_intermediate_size,
                latent_size,
            ),
        ];

        // Inspection touches only retained safetensors headers.  Keep the aggregate authorization
        // ahead of the single batch loader call so an undersized budget performs no payload I/O.
        let mut resident_bytes = 0u64;
        let mut payload_bytes = 0u64;
        for (packed, scale, rows, columns) in specifications {
            let bytes = inspect_native_matrix(index, layer, expert, packed, scale, rows, columns)?;
            resident_bytes = resident_bytes.checked_add(bytes).ok_or_else(|| {
                KimiK3ExpertError::InvalidGeometry(
                    "routed expert resident byte count overflows u64".to_owned(),
                )
            })?;
            for name in [packed, scale] {
                let length = index
                    .require(name)
                    .map_err(WeightLoadError::from)
                    .map_err(|source| KimiK3ExpertError::Tensor {
                        layer,
                        expert,
                        name: name.to_owned(),
                        source,
                    })?
                    .data_len;
                payload_bytes = payload_bytes.checked_add(length).ok_or_else(|| {
                    KimiK3ExpertError::InvalidGeometry(
                        "routed expert payload byte count overflows u64".to_owned(),
                    )
                })?;
            }
        }
        if resident_bytes > maximum_resident_bytes {
            return Err(KimiK3ExpertError::Budget {
                layer,
                expert,
                name: names.w1_packed.clone(),
                required: resident_bytes,
                maximum: maximum_resident_bytes,
            });
        }

        let matrix_specs = [
            (names.w1_packed.as_str(), moe_intermediate_size, latent_size),
            (names.w2_packed.as_str(), latent_size, moe_intermediate_size),
            (names.w3_packed.as_str(), moe_intermediate_size, latent_size),
        ];
        let matrices = load_weight_matrices(index, &matrix_specs, maximum_resident_bytes).map_err(
            |source| KimiK3ExpertError::BatchLoad {
                layer,
                expert,
                name: names.w1_packed.clone(),
                source,
            },
        )?;
        let mut matrices = matrices.into_iter();
        let w1 = matrices.next().expect("three requested Kimi-K3 matrices");
        let w2 = matrices.next().expect("three requested Kimi-K3 matrices");
        let w3 = matrices.next().expect("three requested Kimi-K3 matrices");
        debug_assert!(matrices.next().is_none());
        let loaded_bytes = [&w1, &w2, &w3]
            .into_iter()
            .try_fold(0u64, |total, matrix| {
                total
                    .checked_add(matrix.resident_bytes() as u64)
                    .ok_or_else(|| {
                        KimiK3ExpertError::InvalidGeometry(
                            "loaded routed expert byte count overflows u64".to_owned(),
                        )
                    })
            })?;
        if loaded_bytes != resident_bytes {
            return Err(KimiK3ExpertError::NativeFormat {
                layer,
                expert,
                name: names.w1_packed.clone(),
                reason: format!(
                    "metadata accounts for {resident_bytes} resident bytes but loaded matrices use {loaded_bytes}"
                ),
            });
        }

        Ok(Self {
            layer,
            expert,
            names,
            w1,
            w2,
            w3,
            resident_bytes,
            payload_bytes,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn expert(&self) -> usize {
        self.expert
    }

    pub fn tensor_names(&self) -> &KimiK3ExpertTensorNames {
        &self.names
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Exact bytes fetched from the six native tensors when this expert is loaded.
    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn project_w1(&self, input: &[f32]) -> Result<Vec<f32>, KimiK3ExpertError> {
        self.project(&self.w1, &self.names.w1_packed, input)
    }

    pub fn project_w2(&self, input: &[f32]) -> Result<Vec<f32>, KimiK3ExpertError> {
        self.project(&self.w2, &self.names.w2_packed, input)
    }

    pub fn project_w3(&self, input: &[f32]) -> Result<Vec<f32>, KimiK3ExpertError> {
        self.project(&self.w3, &self.names.w3_packed, input)
    }

    fn project(
        &self,
        matrix: &WeightMatrix,
        name: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, KimiK3ExpertError> {
        matrix
            .matvec(input)
            .map_err(|source| KimiK3ExpertError::Projection {
                layer: self.layer,
                expert: self.expert,
                name: name.to_owned(),
                source,
            })
    }
}

/// Synchronous deterministic per-layer LRU for native Kimi-K3 routed experts.
///
/// With zero slots, every acquisition loads a transient expert and retains no cache residency.
/// A miss is fully loaded and decoded before the shared LRU is allowed to evict an entry, making
/// insertion transactional with respect to checkpoint and budget failures.
#[derive(Debug)]
pub struct KimiK3ExpertStore {
    index: Arc<TensorIndex>,
    layers: usize,
    experts_per_layer: usize,
    latent_size: usize,
    moe_intermediate_size: usize,
    maximum_expert_bytes: u64,
    cache: LayerLruCache<Arc<KimiK3Expert>>,
    telemetry: ExpertTelemetry,
}

impl KimiK3ExpertStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: TensorIndex,
        layers: usize,
        experts_per_layer: usize,
        latent_size: usize,
        moe_intermediate_size: usize,
        slots_per_layer: usize,
        maximum_expert_bytes: u64,
    ) -> Result<Self, KimiK3ExpertError> {
        Self::new_shared(
            Arc::new(index),
            layers,
            experts_per_layer,
            latent_size,
            moe_intermediate_size,
            slots_per_layer,
            maximum_expert_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_shared(
        index: Arc<TensorIndex>,
        layers: usize,
        experts_per_layer: usize,
        latent_size: usize,
        moe_intermediate_size: usize,
        slots_per_layer: usize,
        maximum_expert_bytes: u64,
    ) -> Result<Self, KimiK3ExpertError> {
        validate_dimensions(latent_size, moe_intermediate_size)?;
        if layers == 0 || experts_per_layer == 0 || maximum_expert_bytes == 0 {
            return Err(KimiK3ExpertError::InvalidGeometry(
                "layers, experts per layer, and maximum expert bytes must be non-zero".to_owned(),
            ));
        }
        if slots_per_layer > experts_per_layer {
            return Err(KimiK3ExpertError::InvalidGeometry(format!(
                "{slots_per_layer} cache slots per layer exceed {experts_per_layer} experts"
            )));
        }
        Ok(Self {
            index,
            layers,
            experts_per_layer,
            latent_size,
            moe_intermediate_size,
            maximum_expert_bytes,
            cache: LayerLruCache::new(layers, slots_per_layer),
            telemetry: ExpertTelemetry::default(),
        })
    }

    pub fn acquire(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<Arc<KimiK3Expert>, KimiK3ExpertError> {
        self.validate_key(layer, expert)?;

        if self.cache.contains(layer, expert) {
            return self.cache.access(
                &mut self.telemetry,
                layer,
                expert,
                || -> Result<_, KimiK3ExpertError> {
                    unreachable!("a known Kimi-K3 cache hit never invokes its loader")
                },
                |value| Ok(Arc::clone(value)),
            );
        }

        // This is intentionally before `LayerLruCache::access`: its generic miss path evicts
        // before invoking the loader.  Supplying an already successful load makes that infallible
        // insertion step transactional for Kimi-K3.
        let loaded = KimiK3Expert::load(
            &self.index,
            layer,
            expert,
            self.latent_size,
            self.moe_intermediate_size,
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

    pub fn telemetry(&self) -> &ExpertTelemetry {
        &self.telemetry
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

    pub fn is_cached(&self, layer: usize, expert: usize) -> bool {
        layer < self.layers && expert < self.experts_per_layer && self.cache.contains(layer, expert)
    }

    fn validate_key(&self, layer: usize, expert: usize) -> Result<(), KimiK3ExpertError> {
        if layer >= self.layers {
            return Err(KimiK3ExpertError::InvalidLayer {
                layer,
                layers: self.layers,
            });
        }
        if expert >= self.experts_per_layer {
            return Err(KimiK3ExpertError::InvalidExpert {
                layer,
                expert,
                experts: self.experts_per_layer,
            });
        }
        Ok(())
    }
}

fn validate_dimensions(
    latent_size: usize,
    moe_intermediate_size: usize,
) -> Result<(), KimiK3ExpertError> {
    if latent_size == 0 || moe_intermediate_size == 0 {
        return Err(KimiK3ExpertError::InvalidGeometry(
            "latent and MoE intermediate dimensions must be non-zero".to_owned(),
        ));
    }
    if latent_size % MXFP4_GROUP_SIZE != 0 || moe_intermediate_size % MXFP4_GROUP_SIZE != 0 {
        return Err(KimiK3ExpertError::InvalidGeometry(format!(
            "native MXFP4 dimensions must be multiples of {MXFP4_GROUP_SIZE}, got latent={latent_size} and intermediate={moe_intermediate_size}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn inspect_native_matrix(
    index: &TensorIndex,
    layer: usize,
    expert: usize,
    packed_name: &str,
    scale_name: &str,
    rows: usize,
    columns: usize,
) -> Result<u64, KimiK3ExpertError> {
    let contextual = |name: &str, source| KimiK3ExpertError::Tensor {
        layer,
        expert,
        name: name.to_owned(),
        source,
    };
    let packed = index
        .require(packed_name)
        .map_err(WeightLoadError::from)
        .map_err(|source| contextual(packed_name, source))?;
    if packed.dtype != DType::U8 {
        return Err(KimiK3ExpertError::NativeFormat {
            layer,
            expert,
            name: packed_name.to_owned(),
            reason: format!("packed dtype must be U8, got {}", packed.dtype),
        });
    }
    let scale = index
        .require(scale_name)
        .map_err(WeightLoadError::from)
        .map_err(|source| contextual(scale_name, source))?;
    if scale.dtype != DType::U8 {
        return Err(KimiK3ExpertError::NativeFormat {
            layer,
            expert,
            name: scale_name.to_owned(),
            reason: format!("E8M0 scale dtype must be U8, got {}", scale.dtype),
        });
    }
    let layout = inspect_weight_matrix(index, packed_name, rows, columns)
        .map_err(|source| contextual(packed_name, source))?;
    if layout.format != (WeightFormat::MxFp4E2M1 { group_size: 32 }) {
        return Err(KimiK3ExpertError::NativeFormat {
            layer,
            expert,
            name: packed_name.to_owned(),
            reason: format!("expected E2M1 group-32 layout, got {:?}", layout.format),
        });
    }
    Ok(layout.resident_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    const LATENT: usize = 32;
    const INTERMEDIATE: usize = 32;
    const TINY_EXPERT_BYTES: u64 = 1_632;

    fn fixture_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "urbilateria_kimi_k3_expert_{label}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    fn append_tensor(
        tensors: &mut BTreeMap<String, serde_json::Value>,
        payload: &mut Vec<u8>,
        name: String,
        dtype: &str,
        shape: [usize; 2],
        bytes: impl IntoIterator<Item = u8>,
    ) {
        let start = payload.len();
        payload.extend(bytes);
        tensors.insert(
            name,
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }

    fn write_fixture(path: &Path, experts: usize, invalid_scale_expert: Option<usize>) {
        let mut tensors = BTreeMap::new();
        let mut payload = Vec::new();
        for expert in 0..experts {
            let names = KimiK3ExpertTensorNames::new(0, expert);
            for (packed, scale, rows, columns) in [
                (names.w1_packed, names.w1_scale, INTERMEDIATE, LATENT),
                (names.w2_packed, names.w2_scale, LATENT, INTERMEDIATE),
                (names.w3_packed, names.w3_scale, INTERMEDIATE, LATENT),
            ] {
                append_tensor(
                    &mut tensors,
                    &mut payload,
                    packed,
                    "U8",
                    [rows, columns / 2],
                    std::iter::repeat_n(0x11, rows * columns / 2),
                );
                let scale_byte = if invalid_scale_expert == Some(expert) {
                    0xff
                } else {
                    127
                };
                append_tensor(
                    &mut tensors,
                    &mut payload,
                    scale,
                    "U8",
                    [rows, columns / MXFP4_GROUP_SIZE],
                    std::iter::repeat_n(scale_byte, rows * columns / MXFP4_GROUP_SIZE),
                );
            }
        }
        let mut header = serde_json::to_vec(&tensors).expect("serialize fixture header");
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(payload);
        fs::write(path, output).expect("write fixture shard");
    }

    #[test]
    fn official_geometry_has_exact_native_residency() {
        let logical = 3_584u64 * 3_072;
        let packed = logical / 2;
        let scales = logical / 32;
        assert_eq!(3 * (packed + scales), KIMI_K3_EXPERT_RESIDENT_BYTES);
    }

    #[test]
    fn names_and_native_matrix_projections_are_exact() {
        let dir = fixture_dir("load");
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("tiny.safetensors"), 1, None);
        let index = TensorIndex::open(&dir).unwrap();
        let expert =
            KimiK3Expert::load(&index, 0, 0, LATENT, INTERMEDIATE, TINY_EXPERT_BYTES).unwrap();

        assert_eq!(
            expert.tensor_names().all(),
            [
                "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed",
                "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale",
                "language_model.model.layers.0.block_sparse_moe.experts.0.w2.weight_packed",
                "language_model.model.layers.0.block_sparse_moe.experts.0.w2.weight_scale",
                "language_model.model.layers.0.block_sparse_moe.experts.0.w3.weight_packed",
                "language_model.model.layers.0.block_sparse_moe.experts.0.w3.weight_scale",
            ]
        );
        assert_eq!(expert.resident_bytes(), TINY_EXPERT_BYTES);
        assert_eq!(expert.payload_bytes(), TINY_EXPERT_BYTES);
        assert_eq!(expert.project_w1(&[1.0; LATENT]).unwrap(), vec![16.0; 32]);
        assert_eq!(expert.project_w3(&[1.0; LATENT]).unwrap(), vec![16.0; 32]);
        assert_eq!(
            expert.project_w2(&[1.0; INTERMEDIATE]).unwrap(),
            vec![16.0; 32]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn aggregate_budget_is_rejected_before_payload_io() {
        let dir = fixture_dir("budget");
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("tiny.safetensors");
        write_fixture(&shard, 1, None);
        let index = TensorIndex::open(&dir).unwrap();
        // The retained index has complete metadata, but any attempted payload read now fails.
        // Receiving Budget therefore proves the aggregate authorization precedes payload I/O.
        fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&shard)
            .unwrap();
        let error = KimiK3Expert::load(&index, 0, 0, LATENT, INTERMEDIATE, TINY_EXPERT_BYTES - 1)
            .unwrap_err();
        assert!(matches!(
            error,
            KimiK3ExpertError::Budget {
                layer: 0,
                expert: 0,
                ref name,
                required: TINY_EXPERT_BYTES,
                maximum
            } if name.ends_with(".w1.weight_packed") && maximum == TINY_EXPERT_BYTES - 1
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_miss_does_not_evict_or_mutate_cache_telemetry() {
        let dir = fixture_dir("rollback");
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("tiny.safetensors"), 2, Some(1));
        let index = TensorIndex::open(&dir).unwrap();
        let mut store =
            KimiK3ExpertStore::new(index, 1, 2, LATENT, INTERMEDIATE, 1, TINY_EXPERT_BYTES)
                .unwrap();

        let first = store.acquire(0, 0).unwrap();
        let before = store.telemetry().clone();
        let error = store.acquire(0, 1).unwrap_err();
        assert!(matches!(error, KimiK3ExpertError::BatchLoad { .. }));
        assert!(store.is_cached(0, 0));
        assert!(!store.is_cached(0, 1));
        assert_eq!(store.telemetry(), &before);
        let hit = store.acquire(0, 0).unwrap();
        assert!(Arc::ptr_eq(&first, &hit));
        assert_eq!(store.telemetry().hits, 1);
        assert_eq!(store.telemetry().misses, 1);
        assert_eq!(store.telemetry().evictions, 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_slots_streams_without_residency() {
        let dir = fixture_dir("zero_slots");
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("tiny.safetensors"), 1, None);
        let index = TensorIndex::open(&dir).unwrap();
        let mut store =
            KimiK3ExpertStore::new(index, 1, 1, LATENT, INTERMEDIATE, 0, TINY_EXPERT_BYTES)
                .unwrap();

        let first = store.acquire(0, 0).unwrap();
        let second = store.acquire(0, 0).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(
            store.telemetry(),
            &ExpertTelemetry {
                hits: 0,
                misses: 2,
                evictions: 0,
                bytes_read: 2 * TINY_EXPERT_BYTES,
                resident_experts: 0,
                resident_bytes: 0,
            }
        );
        assert!(!store.is_cached(0, 0));
        fs::remove_dir_all(dir).unwrap();
    }
}
