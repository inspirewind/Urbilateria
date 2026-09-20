//! One model-family dispatch shared by the plain CLI and interactive frontends.

use super::{analyze_checkpoint, CheckpointReport};
use crate::models::{deepseek_v41, hy4, qwen3_8};
use crate::storage::TensorIndex;
use crate::{CommonModelConfig, ModelConfig};
use serde::Serialize;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Inspection {
    pub model_path: PathBuf,
    pub model: CommonModelConfig,
    pub report: InspectionReport,
}

/// Untagged serialization preserves each family's existing `urb inspect --json` format.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum InspectionReport {
    Checkpoint(Box<CheckpointReport>),
    Qwen38(Box<qwen3_8::schema::Qwen38ManifestReport>),
    Hy4(Box<hy4::schema::Hy4Requirements>),
    DeepseekV41(Box<deepseek_v41::schema::DeepseekV41Requirements>),
}

/// Reads configuration and manifest/header metadata without loading tensor payloads.
pub fn inspect_checkpoint(model_dir: &Path) -> Result<Inspection, Box<dyn Error + Send + Sync>> {
    let config = ModelConfig::load(model_dir)?;
    let model = config.common();
    let report = match config {
        ModelConfig::Qwen38(config) => InspectionReport::Qwen38(Box::new(
            qwen3_8::schema::inspect_manifest(&config, model_dir)?,
        )),
        ModelConfig::Hy4(config) => {
            let index = TensorIndex::open(model_dir)?;
            InspectionReport::Hy4(Box::new(hy4::schema::inspect_requirements(
                &config, &index, 1, 0,
            )?))
        }
        ModelConfig::DeepseekV41(config) => {
            let index = TensorIndex::open(model_dir)?;
            InspectionReport::DeepseekV41(Box::new(deepseek_v41::schema::inspect_requirements(
                &config, &index, 1, 0,
            )?))
        }
        _ => InspectionReport::Checkpoint(Box::new(analyze_checkpoint(model_dir)?)),
    };
    Ok(Inspection {
        model_path: model_dir.to_owned(),
        model,
        report,
    })
}
