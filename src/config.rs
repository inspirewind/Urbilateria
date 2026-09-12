//! Model-family detection and the small cross-model configuration view.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub use crate::models::deepseek_v4::DeepseekV4Config;
pub use crate::models::deepseek_v41::DeepseekV41Config;
pub use crate::models::glm::{GlmConfig, RopeParameters, TokenIds, MLA_LATENT_NORM_EPS};
pub use crate::models::hy4::Hy4Config;
pub use crate::models::kimi_k3::KimiK3Config;
pub use crate::models::qwen3_8::Qwen38Config;

#[derive(Debug)]
pub enum ConfigError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: Option<PathBuf>,
        source: serde_json::Error,
    },
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            Self::Json {
                path: Some(path),
                source,
            } => write!(f, "invalid JSON in {}: {source}", path.display()),
            Self::Json { path: None, source } => write!(f, "invalid config JSON: {source}"),
            Self::Invalid(reason) => write!(f, "invalid model configuration: {reason}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Glm52,
    DeepseekV4,
    DeepseekV41,
    Hy4,
    KimiK3,
    Qwen38,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Glm52 => f.write_str("GLM-5.2"),
            Self::DeepseekV4 => f.write_str("DeepSeek-V4"),
            Self::DeepseekV41 => f.write_str("DeepSeek-V4.1-Flash"),
            Self::Hy4 => f.write_str("Hy4-preview-FP8"),
            Self::KimiK3 => f.write_str("Kimi-K3"),
            Self::Qwen38 => f.write_str("Qwen3.8-2.4T-A95B-FP8"),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ModelConfig {
    Glm52(GlmConfig),
    DeepseekV4(DeepseekV4Config),
    DeepseekV41(DeepseekV41Config),
    Hy4(Hy4Config),
    KimiK3(KimiK3Config),
    Qwen38(Qwen38Config),
}

/// The deliberately small view needed by model-agnostic CLI and generation code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommonModelConfig {
    pub family: ModelFamily,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub eos_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct ModelTypeProbe {
    model_type: String,
}

impl ModelConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = model_dir.as_ref().join("config.json");
        let json = fs::read_to_string(&path).map_err(|source| ConfigError::Read {
            path: path.clone(),
            source,
        })?;
        Self::from_json_str_at(&json, Some(path))
    }

    pub fn from_json_str(json: &str) -> Result<Self, ConfigError> {
        Self::from_json_str_at(json, None)
    }

    fn from_json_str_at(json: &str, path: Option<PathBuf>) -> Result<Self, ConfigError> {
        let probe: ModelTypeProbe =
            serde_json::from_str(json).map_err(|source| ConfigError::Json {
                path: path.clone(),
                source,
            })?;
        match probe.model_type.as_str() {
            "glm_moe_dsa" => GlmConfig::from_json_str(json).map(Self::Glm52),
            "deepseek_v4" => DeepseekV4Config::from_json_str(json).map(Self::DeepseekV4),
            "deepseek_v41" => DeepseekV41Config::from_json_str(json).map(Self::DeepseekV41),
            "hy_v4" => Hy4Config::from_json_str(json).map(Self::Hy4),
            "kimi_k3" => KimiK3Config::from_json_str(json).map(Self::KimiK3),
            "qwen3_5_moe_text" => Qwen38Config::from_json_str(json).map(Self::Qwen38),
            other => Err(ConfigError::Invalid(format!(
                "unsupported model_type={other:?}; supported values are \"glm_moe_dsa\", \"deepseek_v4\", \"deepseek_v41\", \"hy_v4\", \"kimi_k3\", and the release-specific \"qwen3_5_moe_text\""
            ))),
        }
    }

    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Glm52(_) => ModelFamily::Glm52,
            Self::DeepseekV4(_) => ModelFamily::DeepseekV4,
            Self::DeepseekV41(_) => ModelFamily::DeepseekV41,
            Self::Hy4(_) => ModelFamily::Hy4,
            Self::KimiK3(_) => ModelFamily::KimiK3,
            Self::Qwen38(_) => ModelFamily::Qwen38,
        }
    }

    pub fn common(&self) -> CommonModelConfig {
        match self {
            Self::Glm52(config) => CommonModelConfig {
                family: ModelFamily::Glm52,
                model_type: config.model_type.clone(),
                vocab_size: config.vocab_size,
                hidden_size: config.hidden_size,
                num_hidden_layers: config.num_hidden_layers,
                max_position_embeddings: config.max_position_embeddings,
                eos_token_ids: config.eos_token_id.as_slice().to_vec(),
            },
            Self::DeepseekV4(config) => CommonModelConfig {
                family: ModelFamily::DeepseekV4,
                model_type: config.model_type.clone(),
                vocab_size: config.vocab_size,
                hidden_size: config.hidden_size,
                num_hidden_layers: config.num_hidden_layers,
                max_position_embeddings: config.max_position_embeddings,
                eos_token_ids: vec![config.eos_token_id],
            },
            Self::DeepseekV41(config) => CommonModelConfig {
                family: ModelFamily::DeepseekV41,
                model_type: config.model_type.clone(),
                vocab_size: config.text_config.vocab_size,
                hidden_size: config.text_config.hidden_size,
                num_hidden_layers: config.text_config.num_hidden_layers,
                max_position_embeddings: config.text_config.max_position_embeddings,
                eos_token_ids: vec![config.eos_token_id],
            },
            Self::Hy4(config) => CommonModelConfig {
                family: ModelFamily::Hy4,
                model_type: config.model_type.clone(),
                vocab_size: config.vocab_size,
                hidden_size: config.hidden_size,
                num_hidden_layers: config.num_hidden_layers,
                max_position_embeddings: config.max_position_embeddings,
                eos_token_ids: vec![config.eos_token_id],
            },
            Self::KimiK3(config) => CommonModelConfig {
                family: ModelFamily::KimiK3,
                model_type: config.model_type.clone(),
                vocab_size: config.text_config.vocab_size,
                hidden_size: config.text_config.hidden_size,
                num_hidden_layers: config.text_config.num_hidden_layers,
                max_position_embeddings: config.text_config.max_position_embeddings,
                eos_token_ids: vec![config.eos_token_id],
            },
            Self::Qwen38(config) => CommonModelConfig {
                family: ModelFamily::Qwen38,
                model_type: config.model_type.clone(),
                vocab_size: config.vocab_size,
                hidden_size: config.hidden_size,
                num_hidden_layers: config.num_hidden_layers,
                max_position_embeddings: config.max_position_embeddings,
                // The effective chat stop set lives in generation_config.json and is validated
                // by Qwen38ReleaseConfig. This config-only view deliberately exposes only the
                // EOS identity declared by config.json.
                eos_token_ids: vec![config.eos_token_id],
            },
        }
    }

    pub fn as_glm52(&self) -> Option<&GlmConfig> {
        match self {
            Self::Glm52(config) => Some(config),
            Self::DeepseekV4(_)
            | Self::DeepseekV41(_)
            | Self::Hy4(_)
            | Self::KimiK3(_)
            | Self::Qwen38(_) => None,
        }
    }

    pub fn as_deepseek_v4(&self) -> Option<&DeepseekV4Config> {
        match self {
            Self::Glm52(_)
            | Self::DeepseekV41(_)
            | Self::Hy4(_)
            | Self::KimiK3(_)
            | Self::Qwen38(_) => None,
            Self::DeepseekV4(config) => Some(config),
        }
    }

    pub fn as_deepseek_v41(&self) -> Option<&DeepseekV41Config> {
        match self {
            Self::DeepseekV41(config) => Some(config),
            Self::Glm52(_)
            | Self::DeepseekV4(_)
            | Self::Hy4(_)
            | Self::KimiK3(_)
            | Self::Qwen38(_) => None,
        }
    }

    pub fn as_hy4(&self) -> Option<&Hy4Config> {
        match self {
            Self::Hy4(config) => Some(config),
            Self::Glm52(_)
            | Self::DeepseekV4(_)
            | Self::DeepseekV41(_)
            | Self::KimiK3(_)
            | Self::Qwen38(_) => None,
        }
    }

    pub fn as_kimi_k3(&self) -> Option<&KimiK3Config> {
        match self {
            Self::Glm52(_)
            | Self::DeepseekV4(_)
            | Self::DeepseekV41(_)
            | Self::Hy4(_)
            | Self::Qwen38(_) => None,
            Self::KimiK3(config) => Some(config),
        }
    }

    pub fn as_qwen38(&self) -> Option<&Qwen38Config> {
        match self {
            Self::Glm52(_)
            | Self::DeepseekV4(_)
            | Self::DeepseekV41(_)
            | Self::Hy4(_)
            | Self::KimiK3(_) => None,
            Self::Qwen38(config) => Some(config),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_family_before_family_deserialization() {
        let error = ModelConfig::from_json_str(r#"{"model_type":"mystery"}"#).unwrap_err();
        assert!(error.to_string().contains("unsupported model_type"));
    }
}
