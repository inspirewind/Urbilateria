//! Native block-FP8 routed-expert loading and deterministic per-layer caching.

use super::expert::{Qwen38Expert, Qwen38ExpertError};
use crate::runtime::cache::LayerLruCache;
use crate::runtime::ExpertTelemetry;
use crate::storage::{
    inspect_weight_matrix, load_weight_matrices, TensorIndex, WeightFormat, WeightLoadError,
};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum Qwen38ExpertStoreError {
    Invalid(String),
    InvalidLayer { layer: usize, layers: usize },
    InvalidExpert { expert: usize, experts: usize },
    Weight(WeightLoadError),
    Expert(Qwen38ExpertError),
    Budget { required: u64, maximum: u64 },
}

impl fmt::Display for Qwen38ExpertStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(formatter, "invalid Qwen3.8 expert store: {reason}"),
            Self::InvalidLayer { layer, layers } => {
                write!(formatter, "Qwen3.8 expert layer {layer} is outside 0..{layers}")
            }
            Self::InvalidExpert { expert, experts } => {
                write!(formatter, "Qwen3.8 expert {expert} is outside 0..{experts}")
            }
            Self::Weight(error) => error.fmt(formatter),
            Self::Expert(error) => error.fmt(formatter),
            Self::Budget { required, maximum } => write!(
                formatter,
                "Qwen3.8 routed expert needs {required} resident bytes, per-expert limit is {maximum}"
            ),
        }
    }
}

impl std::error::Error for Qwen38ExpertStoreError {
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

impl From<WeightLoadError> for Qwen38ExpertStoreError {
    fn from(value: WeightLoadError) -> Self {
        Self::Weight(value)
    }
}

impl From<Qwen38ExpertError> for Qwen38ExpertStoreError {
    fn from(value: Qwen38ExpertError) -> Self {
        Self::Expert(value)
    }
}

#[derive(Debug)]
pub struct Qwen38LoadedExpert {
    pub value: Qwen38Expert,
    resident_bytes: u64,
    payload_bytes: u64,
}

impl Qwen38LoadedExpert {
    pub fn load(
        index: &TensorIndex,
        layer: usize,
        expert: usize,
        hidden: usize,
        intermediate: usize,
        maximum_resident_bytes: u64,
    ) -> Result<Self, Qwen38ExpertStoreError> {
        if hidden == 0 || intermediate == 0 || maximum_resident_bytes == 0 {
            return Err(Qwen38ExpertStoreError::Invalid(
                "hidden/intermediate dimensions and per-expert budget must be non-zero".to_owned(),
            ));
        }
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let specifications = [
            (format!("{prefix}.gate_proj.weight"), intermediate, hidden),
            (format!("{prefix}.up_proj.weight"), intermediate, hidden),
            (format!("{prefix}.down_proj.weight"), hidden, intermediate),
        ];
        let mut resident_bytes = 0u64;
        let mut payload_bytes = 0u64;
        for (name, rows, columns) in &specifications {
            let layout = inspect_weight_matrix(index, name, *rows, *columns)?;
            if !matches!(
                layout.format,
                WeightFormat::BlockFp8E4M3Bf16ScaleInv {
                    block_rows: 128,
                    block_cols: 128
                }
            ) {
                return Err(Qwen38ExpertStoreError::Invalid(format!(
                    "routed tensor {name:?} is not 128x128 E4M3/BF16-scale_inv"
                )));
            }
            resident_bytes = resident_bytes
                .checked_add(layout.resident_bytes)
                .ok_or_else(|| {
                    Qwen38ExpertStoreError::Invalid(
                        "routed expert resident bytes overflow".to_owned(),
                    )
                })?;
            let scale_name = name
                .strip_suffix(".weight")
                .map(|base| format!("{base}.weight_scale_inv"))
                .ok_or_else(|| {
                    Qwen38ExpertStoreError::Invalid(format!(
                        "routed tensor name {name:?} does not end in .weight"
                    ))
                })?;
            for tensor_name in [name.as_str(), scale_name.as_str()] {
                payload_bytes = payload_bytes
                    .checked_add(
                        index
                            .require(tensor_name)
                            .map_err(WeightLoadError::from)?
                            .data_len,
                    )
                    .ok_or_else(|| {
                        Qwen38ExpertStoreError::Invalid(
                            "routed expert payload bytes overflow".to_owned(),
                        )
                    })?;
            }
        }
        if resident_bytes > maximum_resident_bytes {
            return Err(Qwen38ExpertStoreError::Budget {
                required: resident_bytes,
                maximum: maximum_resident_bytes,
            });
        }
        let borrowed = specifications
            .iter()
            .map(|(name, rows, columns)| (name.as_str(), *rows, *columns))
            .collect::<Vec<_>>();
        let mut matrices =
            load_weight_matrices(index, &borrowed, maximum_resident_bytes)?.into_iter();
        let value = Qwen38Expert::new(
            matrices.next().expect("three Qwen3.8 expert matrices"),
            matrices.next().expect("three Qwen3.8 expert matrices"),
            matrices.next().expect("three Qwen3.8 expert matrices"),
        )?;
        if matrices.next().is_some() || value.resident_bytes() as u64 != resident_bytes {
            return Err(Qwen38ExpertStoreError::Invalid(
                "loaded routed expert disagrees with metadata accounting".to_owned(),
            ));
        }
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
    ) -> Result<u64, Qwen38ExpertStoreError> {
        // Every released expert has the same shapes and representation, so layer 0/expert 0 is a
        // complete metadata oracle for the cache planner.
        let prefix = "model.layers.0.mlp.experts.0";
        [
            (format!("{prefix}.gate_proj.weight"), intermediate, hidden),
            (format!("{prefix}.up_proj.weight"), intermediate, hidden),
            (format!("{prefix}.down_proj.weight"), hidden, intermediate),
        ]
        .iter()
        .try_fold(0u64, |total, (name, rows, columns)| {
            let layout = inspect_weight_matrix(index, name, *rows, *columns)?;
            if !matches!(layout.format, WeightFormat::BlockFp8E4M3Bf16ScaleInv { .. }) {
                return Err(Qwen38ExpertStoreError::Invalid(format!(
                    "routed tensor {name:?} is not native Qwen block-FP8"
                )));
            }
            total.checked_add(layout.resident_bytes).ok_or_else(|| {
                Qwen38ExpertStoreError::Invalid("expert byte count overflows".to_owned())
            })
        })
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

#[derive(Debug)]
pub struct Qwen38ExpertStore {
    index: Arc<TensorIndex>,
    layers: usize,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    maximum_expert_bytes: u64,
    cache: LayerLruCache<Arc<Qwen38LoadedExpert>>,
    telemetry: ExpertTelemetry,
}

impl Qwen38ExpertStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new_shared(
        index: Arc<TensorIndex>,
        layers: usize,
        experts: usize,
        hidden: usize,
        intermediate: usize,
        slots_per_layer: usize,
        maximum_expert_bytes: u64,
    ) -> Result<Self, Qwen38ExpertStoreError> {
        if layers == 0
            || experts == 0
            || hidden == 0
            || intermediate == 0
            || maximum_expert_bytes == 0
            || slots_per_layer > experts
        {
            return Err(Qwen38ExpertStoreError::Invalid(
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
    ) -> Result<Arc<Qwen38LoadedExpert>, Qwen38ExpertStoreError> {
        if layer >= self.layers {
            return Err(Qwen38ExpertStoreError::InvalidLayer {
                layer,
                layers: self.layers,
            });
        }
        if expert >= self.experts {
            return Err(Qwen38ExpertStoreError::InvalidExpert {
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
        let loaded = Qwen38LoadedExpert::load(
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

    pub fn telemetry(&self) -> &ExpertTelemetry {
        &self.telemetry
    }
}
