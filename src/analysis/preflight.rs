//! Header-only schema/runtime preflight with each model family's original result format.

use crate::models::{deepseek_v4, deepseek_v41, glm, hy4, kimi_k3, qwen3_8};
use crate::storage::TensorIndex;
use crate::{CommonModelConfig, ModelConfig};
use serde::Serialize;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreflightOptions {
    pub context: usize,
    pub expert_slots: usize,
    pub partial: bool,
}

impl Default for PreflightOptions {
    fn default() -> Self {
        Self {
            context: 1,
            expert_slots: 0,
            partial: false,
        }
    }
}

impl PreflightOptions {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.context == 0 {
            return Err("--context must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Preflight {
    pub model_path: PathBuf,
    pub model: CommonModelConfig,
    pub report: PreflightReport,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum PreflightReport {
    Glm(Box<glm::runtime::GlmRuntimeRequirements>),
    DeepseekV4(Box<deepseek_v4::schema::DeepseekV4Requirements>),
    DeepseekV41(Box<deepseek_v41::schema::DeepseekV41Requirements>),
    Hy4(Box<hy4::schema::Hy4Requirements>),
    KimiK3(Box<kimi_k3::schema::KimiK3Requirements>),
    KimiK3Partial(Box<kimi_k3::schema::KimiK3PartialLayers>),
    Qwen38(Box<qwen3_8::Qwen38RuntimeRequirements>),
}

pub fn preflight_checkpoint(
    model_dir: &Path,
    options: PreflightOptions,
) -> Result<Preflight, Box<dyn Error + Send + Sync>> {
    options.validate()?;
    let config = ModelConfig::load(model_dir)?;
    let model = config.common();
    let PreflightOptions {
        context,
        expert_slots,
        partial,
    } = options;
    if partial {
        let error = match &config {
            ModelConfig::KimiK3(_) => None,
            ModelConfig::Glm52(_) | ModelConfig::DeepseekV4(_) => Some("--partial is currently specific to an in-progress Kimi-K3 transfer"),
            ModelConfig::DeepseekV41(_) => Some("DeepSeek-V4.1 preflight validates the complete native release; --partial applies only to Kimi-K3 shard transfer"),
            ModelConfig::Hy4(_) => Some("Hy4 preflight validates the complete 130-shard release; --partial is Kimi-K3-specific"),
            ModelConfig::Qwen38(_) => Some("Qwen3.8 runtime preflight validates the complete official checkpoint; --partial applies only to Kimi-K3 shard transfer"),
        };
        if let Some(error) = error {
            return Err(error.into());
        }
    }
    let report = match config {
        ModelConfig::Glm52(_) => PreflightReport::Glm(Box::new(
            glm::runtime::GlmRuntimeModel::inspect_requirements(model_dir, context, expert_slots)?,
        )),
        ModelConfig::DeepseekV4(config) => {
            let index = TensorIndex::open(model_dir)?;
            PreflightReport::DeepseekV4(Box::new(deepseek_v4::schema::inspect_requirements(
                &config,
                &index,
                context,
                expert_slots,
            )?))
        }
        ModelConfig::DeepseekV41(config) => {
            let index = TensorIndex::open(model_dir)?;
            PreflightReport::DeepseekV41(Box::new(deepseek_v41::schema::inspect_requirements(
                &config,
                &index,
                context,
                expert_slots,
            )?))
        }
        ModelConfig::Hy4(config) => {
            let index = TensorIndex::open(model_dir)?;
            PreflightReport::Hy4(Box::new(hy4::schema::inspect_requirements(
                &config,
                &index,
                context,
                expert_slots,
            )?))
        }
        ModelConfig::KimiK3(config) => {
            let index = TensorIndex::open(model_dir)?;
            if partial {
                PreflightReport::KimiK3Partial(Box::new(kimi_k3::schema::inspect_available_layers(
                    &config, &index,
                )?))
            } else {
                PreflightReport::KimiK3(Box::new(kimi_k3::schema::inspect_requirements(
                    &config, &index,
                )?))
            }
        }
        ModelConfig::Qwen38(_) => PreflightReport::Qwen38(Box::new(
            qwen3_8::Qwen38RuntimeModel::inspect_requirements(model_dir, context, expert_slots)?,
        )),
    };
    Ok(Preflight {
        model_path: model_dir.to_owned(),
        model,
        report,
    })
}
