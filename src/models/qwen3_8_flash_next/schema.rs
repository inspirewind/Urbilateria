//! Header-only validation of the Qwen3.8-Flash-Next-35B-A3B checkpoint ABI.
//!
//! The Hugging Face weight index is useful before weights are downloaded: it proves that the
//! advertised tensor-name set, shard-name set, and aggregate payload size match this adapter.
//! Once shards exist, [`inspect_requirements`] additionally checks every safetensors header's
//! dtype, shape, payload length, and assignment to the shard named by the HF index.

use super::{
    Qwen38FlashNextConfig, Qwen38FlashNextGenerationConfig, BOS_TOKEN_ID, IM_END_TOKEN_ID,
};
use crate::config::ConfigError;
use crate::storage::{DType, SafetensorError, TensorIndex, TensorInfo};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const RELEASE_TENSOR_COUNT: usize = 1658;
pub const RELEASE_SHARD_COUNT: usize = 131;
pub const RELEASE_TOTAL_SIZE: u64 = 359999963128;
pub const RELEASE_LOGICAL_PARAMETER_COUNT: u64 = 179999981424;
pub const BASE_TENSOR_COUNT: usize = 1294;
pub const MTP_TENSOR_COUNT: usize = 31;
pub const BASE_ROUTED_EXPERT_COUNT: usize = 24576;
pub const BASE_LAYER_COUNT: usize = 48;
pub const EXPERT_COUNT: usize = 512;
pub const FULL_ATTENTION_LAYER_COUNT: usize = 12;
pub const LINEAR_ATTENTION_LAYER_COUNT: usize = 36;

const MAX_HF_INDEX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
pub enum SchemaError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: Option<PathBuf>,
        source: serde_json::Error,
    },
    Config(ConfigError),
    Checkpoint(SafetensorError),
    Invalid(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            Self::Json {
                path: Some(path),
                source,
            } => write!(f, "invalid JSON in {}: {source}", path.display()),
            Self::Json { path: None, source } => {
                write!(f, "invalid HF weight-index JSON: {source}")
            }
            Self::Config(error) => error.fmt(f),
            Self::Checkpoint(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "invalid Qwen3.8-Flash-Next checkpoint: {reason}"),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Config(error) => Some(error),
            Self::Checkpoint(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<SafetensorError> for SchemaError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<ConfigError> for SchemaError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl TensorSpec {
    fn new(name: impl Into<String>, dtype: DType, shape: &[u64]) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    pub fn element_count(&self) -> Result<u64, SchemaError> {
        self.shape.iter().try_fold(1u64, |product, &dimension| {
            product.checked_mul(dimension).ok_or_else(|| {
                SchemaError::Invalid(format!("tensor {:?} element count overflows", self.name))
            })
        })
    }

    pub fn payload_bytes(&self) -> Result<u64, SchemaError> {
        let element_bytes = self.dtype.element_bytes().ok_or_else(|| {
            SchemaError::Invalid(format!(
                "tensor {:?} has unknown dtype {}",
                self.name, self.dtype
            ))
        })?;
        self.element_count()?
            .checked_mul(element_bytes)
            .ok_or_else(|| {
                SchemaError::Invalid(format!("tensor {:?} payload size overflows", self.name))
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen38FlashNextRequirements {
    pub required_tensor_count: usize,
    pub checkpoint_tensor_count: usize,
    pub checkpoint_shard_count: usize,
    pub base_layer_count: usize,
    pub linear_attention_layer_count: usize,
    pub full_attention_layer_count: usize,
    pub routed_expert_count_per_layer: usize,
    pub mtp_layer_count: usize,
    pub checkpoint_payload_bytes: u64,
    /// All BF16 checkpoint parameters, including vision and MTP.
    pub logical_parameter_count: u64,
    pub base_tensor_count: usize,
    pub mtp_tensor_count: usize,
    pub experts_per_layer: usize,
    pub selected_experts_per_token: usize,
    pub base_routed_expert_count: usize,
    pub mtp_routed_expert_count: usize,
    pub stop_token_ids: Vec<u32>,
}

/// Metadata-only preflight result. No `.safetensors` file is opened to produce this report.
pub type Qwen38FlashNextManifestReport = Qwen38FlashNextRequirements;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HfWeightIndex {
    pub total_size: u64,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHfWeightIndex {
    metadata: RawMetadata,
    weight_map: StrictWeightMap,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMetadata {
    total_size: f64,
}

#[derive(Debug, Default)]
struct StrictWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for StrictWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct WeightMapVisitor;

        impl<'de> Visitor<'de> for WeightMapVisitor {
            type Value = StrictWeightMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a weight_map object with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some(name) = map.next_key::<String>()? {
                    if values.contains_key(&name) {
                        return Err(de::Error::custom(format!(
                            "duplicate weight_map tensor {name:?}"
                        )));
                    }
                    values.insert(name, map.next_value::<String>()?);
                }
                Ok(StrictWeightMap(values))
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

impl HfWeightIndex {
    pub fn load(model_dir: &Path) -> Result<Self, SchemaError> {
        let path = model_dir.join("model.safetensors.index.json");
        let file = File::open(&path).map_err(|source| SchemaError::Read {
            path: path.clone(),
            source,
        })?;
        let file_len = file
            .metadata()
            .map_err(|source| SchemaError::Read {
                path: path.clone(),
                source,
            })?
            .len();
        if file_len > MAX_HF_INDEX_BYTES {
            return invalid(format!(
                "HF weight index is {file_len} bytes; refusing metadata larger than {MAX_HF_INDEX_BYTES} bytes"
            ));
        }
        let capacity = usize::try_from(file_len).map_err(|_| {
            SchemaError::Invalid(format!(
                "HF weight index length {file_len} does not fit usize"
            ))
        })?;
        let mut json = String::with_capacity(capacity);
        file.take(MAX_HF_INDEX_BYTES + 1)
            .read_to_string(&mut json)
            .map_err(|source| SchemaError::Read {
                path: path.clone(),
                source,
            })?;
        if json.len() as u64 > MAX_HF_INDEX_BYTES {
            return invalid(format!(
                "HF weight index grew beyond the {MAX_HF_INDEX_BYTES}-byte metadata limit while reading"
            ));
        }
        Self::from_json_str_at(&json, Some(path))
    }

    pub fn from_json_str(json: &str) -> Result<Self, SchemaError> {
        Self::from_json_str_at(json, None)
    }

    fn from_json_str_at(json: &str, path: Option<PathBuf>) -> Result<Self, SchemaError> {
        let raw: RawHfWeightIndex =
            serde_json::from_str(json).map_err(|source| SchemaError::Json {
                path: path.clone(),
                source,
            })?;
        if raw.metadata.total_size != RELEASE_TOTAL_SIZE as f64 {
            return invalid("HF metadata.total_size differs from the release");
        }
        let index = Self {
            total_size: raw.metadata.total_size as u64,
            weight_map: raw.weight_map.0,
        };
        index.validate()?;
        Ok(index)
    }

    /// Validates information available before any weight shard is present.
    pub fn validate(&self) -> Result<(), SchemaError> {
        let specs = official_tensor_specs()?;
        self.validate_against_specs(&specs)
    }

    fn validate_against_specs(&self, specs: &[TensorSpec]) -> Result<(), SchemaError> {
        if self.total_size != RELEASE_TOTAL_SIZE {
            return invalid(format!(
                "HF metadata.total_size={} but the release requires {RELEASE_TOTAL_SIZE}",
                self.total_size
            ));
        }
        if self.weight_map.len() != RELEASE_TENSOR_COUNT {
            return invalid(format!(
                "HF weight_map has {} names; expected {RELEASE_TENSOR_COUNT}",
                self.weight_map.len()
            ));
        }

        let expected_names: BTreeSet<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        let actual_names: BTreeSet<&str> = self.weight_map.keys().map(String::as_str).collect();
        if actual_names != expected_names {
            let missing = expected_names.difference(&actual_names).next();
            let unexpected = actual_names.difference(&expected_names).next();
            return invalid(format!(
                "HF tensor names differ from the release (first missing={missing:?}, first unexpected={unexpected:?})"
            ));
        }

        let mut shards = BTreeSet::new();
        for (tensor, shard) in &self.weight_map {
            let number = parse_release_shard_name(shard).ok_or_else(|| {
                SchemaError::Invalid(format!(
                    "tensor {tensor:?} points to non-release shard name {shard:?}"
                ))
            })?;
            shards.insert(number);
        }
        let expected_shards: BTreeSet<usize> = (1..=RELEASE_SHARD_COUNT).collect();
        if shards != expected_shards {
            let missing = expected_shards.difference(&shards).next();
            return invalid(format!(
                "HF weight_map does not cover all {RELEASE_SHARD_COUNT} shards (first missing={missing:?})"
            ));
        }
        Ok(())
    }

    pub fn shard_for(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }
}

/// Validates every release artifact available before downloading the 131 weight shards.
///
/// This reads only `model.safetensors.index.json` and `generation_config.json`. The supplied
/// model config is validated as the exact Qwen3.8-Flash-Next release config, including its text and vision geometry.
pub fn inspect_manifest(
    config: &Qwen38FlashNextConfig,
    model_dir: &Path,
) -> Result<Qwen38FlashNextRequirements, SchemaError> {
    config.validate()?;
    let hf_index = HfWeightIndex::load(model_dir)?;
    let generation = Qwen38FlashNextGenerationConfig::load(model_dir)?;
    let specs = official_tensor_specs()?;
    hf_index.validate_against_specs(&specs)?;

    let logical_parameter_count = specs
        .iter()
        .filter(|spec| spec.dtype == DType::Bf16)
        .try_fold(0u64, |total, spec| {
            total.checked_add(spec.element_count()?).ok_or_else(|| {
                SchemaError::Invalid("logical parameter count overflows u64".to_owned())
            })
        })?;
    if logical_parameter_count != RELEASE_LOGICAL_PARAMETER_COUNT {
        return invalid(format!(
            "internal schema has {logical_parameter_count} logical parameters; expected {RELEASE_LOGICAL_PARAMETER_COUNT}"
        ));
    }

    Ok(Qwen38FlashNextRequirements {
        required_tensor_count: RELEASE_TENSOR_COUNT,
        checkpoint_tensor_count: hf_index.weight_map.len(),
        checkpoint_shard_count: hf_index.weight_map.values().collect::<BTreeSet<_>>().len(),
        base_layer_count: BASE_LAYER_COUNT,
        linear_attention_layer_count: LINEAR_ATTENTION_LAYER_COUNT,
        full_attention_layer_count: FULL_ATTENTION_LAYER_COUNT,
        routed_expert_count_per_layer: EXPERT_COUNT,
        mtp_layer_count: config.mtp_num_hidden_layers,
        checkpoint_payload_bytes: hf_index.total_size,
        logical_parameter_count,
        base_tensor_count: BASE_TENSOR_COUNT,
        mtp_tensor_count: MTP_TENSOR_COUNT,
        experts_per_layer: EXPERT_COUNT,
        selected_experts_per_token: config.num_experts_per_tok,
        base_routed_expert_count: BASE_ROUTED_EXPERT_COUNT,
        mtp_routed_expert_count: EXPERT_COUNT,
        stop_token_ids: generation.eos_token_id,
    })
}

/// Pinned BF16/I64 header ABI of the local release, including vision, MTP and split PLE tables.
pub fn official_tensor_specs() -> Result<Vec<TensorSpec>, SchemaError> {
    #[derive(Deserialize)]
    struct RawSpec {
        name: String,
        dtype: String,
        shape: Vec<u64>,
    }
    let raw: Vec<RawSpec> = serde_json::from_str(include_str!("tensor_schema.json"))
        .map_err(|source| SchemaError::Json { path: None, source })?;
    let specs = raw
        .into_iter()
        .map(|r| TensorSpec::new(r.name, DType::parse(&r.dtype), &r.shape))
        .collect::<Vec<_>>();
    validate_generated_specs(&specs)?;
    Ok(specs)
}

fn validate_generated_specs(specs: &[TensorSpec]) -> Result<(), SchemaError> {
    if specs.len() != RELEASE_TENSOR_COUNT {
        return invalid(format!(
            "internal tensor schema generated {} tensors, expected {RELEASE_TENSOR_COUNT}",
            specs.len()
        ));
    }
    let mut names = HashSet::with_capacity(specs.len());
    let mut payload_bytes = 0u64;
    for spec in specs {
        if !names.insert(spec.name.as_str()) {
            return invalid(format!("internal tensor schema duplicates {:?}", spec.name));
        }
        payload_bytes = payload_bytes
            .checked_add(spec.payload_bytes()?)
            .ok_or_else(|| SchemaError::Invalid("checkpoint payload size overflows".to_owned()))?;
    }
    if payload_bytes != RELEASE_TOTAL_SIZE {
        return invalid(format!(
            "internal tensor schema totals {payload_bytes} bytes, expected {RELEASE_TOTAL_SIZE}"
        ));
    }
    Ok(())
}

/// Validates all available safetensors headers without loading tensor payloads.
pub fn inspect_requirements(
    config: &Qwen38FlashNextConfig,
    hf_index: &HfWeightIndex,
    index: &TensorIndex,
) -> Result<Qwen38FlashNextRequirements, SchemaError> {
    config.validate()?;
    let specs = official_tensor_specs()?;
    hf_index.validate_against_specs(&specs)?;

    if index.names().count() != RELEASE_TENSOR_COUNT {
        return invalid(format!(
            "safetensors headers expose {} tensors, expected {RELEASE_TENSOR_COUNT}",
            index.names().count()
        ));
    }
    if index.shards().len() != RELEASE_SHARD_COUNT {
        return invalid(format!(
            "checkpoint has {} shards, expected {RELEASE_SHARD_COUNT}",
            index.shards().len()
        ));
    }
    let actual_shards: BTreeSet<String> = index
        .shards()
        .iter()
        .map(|shard| shard_file_name(&shard.path))
        .collect::<Result<_, _>>()?;
    let expected_shards: BTreeSet<String> =
        (1..=RELEASE_SHARD_COUNT).map(release_shard_name).collect();
    if actual_shards != expected_shards {
        let missing = expected_shards.difference(&actual_shards).next();
        let unexpected = actual_shards.difference(&expected_shards).next();
        return invalid(format!(
            "safetensors shard names differ from the release (first missing={missing:?}, first unexpected={unexpected:?})"
        ));
    }

    for spec in &specs {
        let tensor = index.require(&spec.name)?;
        let expected_shard = hf_index.shard_for(&spec.name).ok_or_else(|| {
            SchemaError::Invalid(format!("HF index omits tensor {:?}", spec.name))
        })?;
        validate_tensor_info(spec, tensor, expected_shard)?;
    }
    if index.total_payload_bytes() != RELEASE_TOTAL_SIZE {
        return invalid(format!(
            "safetensors payload is {} bytes; expected {RELEASE_TOTAL_SIZE}",
            index.total_payload_bytes()
        ));
    }

    Ok(Qwen38FlashNextRequirements {
        required_tensor_count: RELEASE_TENSOR_COUNT,
        checkpoint_tensor_count: index.names().count(),
        checkpoint_shard_count: index.shards().len(),
        base_layer_count: BASE_LAYER_COUNT,
        linear_attention_layer_count: LINEAR_ATTENTION_LAYER_COUNT,
        full_attention_layer_count: FULL_ATTENTION_LAYER_COUNT,
        routed_expert_count_per_layer: EXPERT_COUNT,
        mtp_layer_count: 1,
        checkpoint_payload_bytes: index.total_payload_bytes(),
        logical_parameter_count: RELEASE_LOGICAL_PARAMETER_COUNT,
        base_tensor_count: BASE_TENSOR_COUNT,
        mtp_tensor_count: MTP_TENSOR_COUNT,
        experts_per_layer: EXPERT_COUNT,
        selected_experts_per_token: config.num_experts_per_tok,
        base_routed_expert_count: BASE_ROUTED_EXPERT_COUNT,
        mtp_routed_expert_count: EXPERT_COUNT,
        stop_token_ids: vec![IM_END_TOKEN_ID, BOS_TOKEN_ID],
    })
}

/// Convenience preflight for a downloaded model directory.
pub fn validate_checkpoint(
    model_dir: &Path,
    config: &Qwen38FlashNextConfig,
) -> Result<Qwen38FlashNextRequirements, SchemaError> {
    let hf_index = HfWeightIndex::load(model_dir)?;
    let tensor_index = TensorIndex::open(model_dir)?;
    inspect_requirements(config, &hf_index, &tensor_index)
}

fn validate_tensor_info(
    spec: &TensorSpec,
    tensor: &TensorInfo,
    expected_shard: &str,
) -> Result<(), SchemaError> {
    if tensor.name != spec.name {
        return invalid(format!(
            "requested tensor {:?} but header entry is {:?}",
            spec.name, tensor.name
        ));
    }
    if tensor.dtype != spec.dtype {
        return invalid(format!(
            "tensor {:?} dtype is {}, expected {}",
            spec.name, tensor.dtype, spec.dtype
        ));
    }
    if tensor.shape != spec.shape {
        return invalid(format!(
            "tensor {:?} shape is {:?}, expected {:?}",
            spec.name, tensor.shape, spec.shape
        ));
    }
    let expected_elements = spec.element_count()?;
    if tensor.declared_elements != expected_elements {
        return invalid(format!(
            "tensor {:?} declares {} elements, expected {expected_elements}",
            spec.name, tensor.declared_elements
        ));
    }
    let expected_bytes = spec.payload_bytes()?;
    if tensor.data_len != expected_bytes {
        return invalid(format!(
            "tensor {:?} payload is {} bytes, expected {expected_bytes}",
            spec.name, tensor.data_len
        ));
    }
    let actual_shard = shard_file_name(&tensor.shard)?;
    if actual_shard != expected_shard {
        return invalid(format!(
            "tensor {:?} is in {actual_shard:?}, HF index requires {expected_shard:?}",
            spec.name
        ));
    }
    Ok(())
}

fn shard_file_name(path: &Path) -> Result<String, SchemaError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            SchemaError::Invalid(format!(
                "shard path {} has no UTF-8 file name",
                path.display()
            ))
        })
}

fn release_shard_name(number: usize) -> String {
    format!("model-{number:05}-of-{RELEASE_SHARD_COUNT:05}.safetensors")
}

fn parse_release_shard_name(name: &str) -> Option<usize> {
    let body = name.strip_prefix("model-")?.strip_suffix(".safetensors")?;
    let (number, total) = body.split_once("-of-")?;
    if number.len() != 5
        || total.len() != 5
        || !number.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
        || total.parse::<usize>().ok()? != RELEASE_SHARD_COUNT
    {
        return None;
    }
    let number = number.parse::<usize>().ok()?;
    (1..=RELEASE_SHARD_COUNT)
        .contains(&number)
        .then_some(number)
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, SchemaError> {
    Err(SchemaError::Invalid(reason.into()))
}
