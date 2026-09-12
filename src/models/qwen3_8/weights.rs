//! Strict layer-wise loading for the Qwen3.8 text decoder.
//!
//! Routed experts are deliberately excluded from [`Qwen38LayerWeights`].  The roughly one
//! gigabyte BF16 decoder trunk is loaded one layer at a time, while selected FP8 experts are
//! acquired independently by the runtime's per-layer LRU.

use super::Qwen38Config;
use crate::model::WeightMatrix;
use crate::storage::{
    inspect_compact_bf16_matrix, load_compact_bf16_matrices, load_reference_values,
    load_reference_vector, DType, TensorIndex, TensorLoadError, WeightLoadError,
};
use std::fmt;

const ROOT: &str = "model.layers";

#[derive(Debug)]
pub enum Qwen38LayerWeightError {
    Invalid(String),
    InvalidLayer {
        layer: usize,
        layers: usize,
    },
    Matrix(WeightLoadError),
    Vector(TensorLoadError),
    Budget {
        layer: usize,
        required: u64,
        maximum: u64,
    },
}

impl fmt::Display for Qwen38LayerWeightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(formatter, "invalid Qwen3.8 layer weights: {reason}"),
            Self::InvalidLayer { layer, layers } => {
                write!(formatter, "Qwen3.8 layer {layer} is outside 0..{layers}")
            }
            Self::Matrix(error) => error.fmt(formatter),
            Self::Vector(error) => error.fmt(formatter),
            Self::Budget {
                layer,
                required,
                maximum,
            } => write!(
                formatter,
                "Qwen3.8 layer {layer} needs {required} resident bytes, layer budget is {maximum}"
            ),
        }
    }
}

impl std::error::Error for Qwen38LayerWeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            Self::Vector(error) => Some(error),
            Self::Invalid(_) | Self::InvalidLayer { .. } | Self::Budget { .. } => None,
        }
    }
}

impl From<WeightLoadError> for Qwen38LayerWeightError {
    fn from(value: WeightLoadError) -> Self {
        Self::Matrix(value)
    }
}

impl From<TensorLoadError> for Qwen38LayerWeightError {
    fn from(value: TensorLoadError) -> Self {
        Self::Vector(value)
    }
}

#[derive(Debug)]
pub struct Qwen38FullAttentionWeights {
    pub query: WeightMatrix,
    pub key: WeightMatrix,
    pub value: WeightMatrix,
    pub output: WeightMatrix,
    pub query_norm: Vec<f32>,
    pub key_norm: Vec<f32>,
}

#[derive(Debug)]
pub struct Qwen38LinearAttentionWeights {
    pub qkv: WeightMatrix,
    pub z: WeightMatrix,
    pub beta: WeightMatrix,
    pub decay: WeightMatrix,
    pub output: WeightMatrix,
    /// Flattened `[conv_channels, 1, kernel]`, oldest tap first.
    pub convolution: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub output_norm: Vec<f32>,
}

#[derive(Debug)]
pub enum Qwen38AttentionWeights {
    Full(Box<Qwen38FullAttentionWeights>),
    Linear(Box<Qwen38LinearAttentionWeights>),
}

#[derive(Debug)]
pub struct Qwen38MoeTrunkWeights {
    pub router: WeightMatrix,
    pub shared_gate: WeightMatrix,
    pub shared_expert_gate: WeightMatrix,
    pub shared_expert_up: WeightMatrix,
    pub shared_expert_down: WeightMatrix,
}

#[derive(Debug)]
pub struct Qwen38LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub attention: Qwen38AttentionWeights,
    pub moe: Qwen38MoeTrunkWeights,
}

impl Qwen38LayerWeights {
    /// Loads one complete non-routed decoder trunk after an aggregate metadata-only budget check.
    pub fn load(
        config: &Qwen38Config,
        index: &TensorIndex,
        layer: usize,
        maximum_resident_bytes: u64,
    ) -> Result<Self, Qwen38LayerWeightError> {
        config
            .validate()
            .map_err(|error| Qwen38LayerWeightError::Invalid(error.to_string()))?;
        if layer >= config.num_hidden_layers {
            return Err(Qwen38LayerWeightError::InvalidLayer {
                layer,
                layers: config.num_hidden_layers,
            });
        }
        let shape = LayerShape::from_config(config, layer)?;
        let resident_bytes = inspect_layer_resident_bytes(index, &shape)?;
        if resident_bytes > maximum_resident_bytes {
            return Err(Qwen38LayerWeightError::Budget {
                layer,
                required: resident_bytes,
                maximum: maximum_resident_bytes,
            });
        }
        load_layer(index, shape, resident_bytes)
    }

    pub fn inspect_resident_bytes(
        config: &Qwen38Config,
        index: &TensorIndex,
        layer: usize,
    ) -> Result<u64, Qwen38LayerWeightError> {
        config
            .validate()
            .map_err(|error| Qwen38LayerWeightError::Invalid(error.to_string()))?;
        if layer >= config.num_hidden_layers {
            return Err(Qwen38LayerWeightError::InvalidLayer {
                layer,
                layers: config.num_hidden_layers,
            });
        }
        inspect_layer_resident_bytes(index, &LayerShape::from_config(config, layer)?)
    }
}

#[derive(Debug, Clone)]
struct LayerShape {
    layer: usize,
    hidden: usize,
    routed_experts: usize,
    shared_intermediate: usize,
    attention: AttentionShape,
}

#[derive(Debug, Clone)]
enum AttentionShape {
    Full {
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    Linear {
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        kernel: usize,
    },
}

impl LayerShape {
    fn from_config(config: &Qwen38Config, layer: usize) -> Result<Self, Qwen38LayerWeightError> {
        let attention = match config.is_full_attention_layer(layer) {
            Some(true) => AttentionShape::Full {
                query_heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
            },
            Some(false) => AttentionShape::Linear {
                key_heads: config.linear_num_key_heads,
                value_heads: config.linear_num_value_heads,
                key_head_dim: config.linear_key_head_dim,
                value_head_dim: config.linear_value_head_dim,
                kernel: config.linear_conv_kernel_dim,
            },
            None => {
                return Err(Qwen38LayerWeightError::InvalidLayer {
                    layer,
                    layers: config.num_hidden_layers,
                })
            }
        };
        Ok(Self {
            layer,
            hidden: config.hidden_size,
            routed_experts: config.num_experts,
            shared_intermediate: config.shared_expert_intermediate_size,
            attention,
        })
    }

    fn prefix(&self) -> String {
        format!("{ROOT}.{}", self.layer)
    }

    fn matrices(&self) -> Result<Vec<(String, usize, usize)>, Qwen38LayerWeightError> {
        let prefix = self.prefix();
        let mut matrices = Vec::new();
        match self.attention {
            AttentionShape::Full {
                query_heads,
                kv_heads,
                head_dim,
            } => {
                let query = checked_product(&[query_heads, head_dim, 2], "full query rows")?;
                let kv = checked_product(&[kv_heads, head_dim], "full KV rows")?;
                let output = checked_product(&[query_heads, head_dim], "full output columns")?;
                matrices.extend([
                    (
                        format!("{prefix}.self_attn.q_proj.weight"),
                        query,
                        self.hidden,
                    ),
                    (format!("{prefix}.self_attn.k_proj.weight"), kv, self.hidden),
                    (format!("{prefix}.self_attn.v_proj.weight"), kv, self.hidden),
                    (
                        format!("{prefix}.self_attn.o_proj.weight"),
                        self.hidden,
                        output,
                    ),
                ]);
            }
            AttentionShape::Linear {
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                ..
            } => {
                let key = checked_product(&[key_heads, key_head_dim], "linear key width")?;
                let value = checked_product(&[value_heads, value_head_dim], "linear value width")?;
                let qkv = key
                    .checked_mul(2)
                    .and_then(|twice| twice.checked_add(value))
                    .ok_or_else(|| {
                        Qwen38LayerWeightError::Invalid(
                            "linear QKV projection width overflows".to_owned(),
                        )
                    })?;
                matrices.extend([
                    (
                        format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        qkv,
                        self.hidden,
                    ),
                    (
                        format!("{prefix}.linear_attn.in_proj_z.weight"),
                        value,
                        self.hidden,
                    ),
                    (
                        format!("{prefix}.linear_attn.in_proj_b.weight"),
                        value_heads,
                        self.hidden,
                    ),
                    (
                        format!("{prefix}.linear_attn.in_proj_a.weight"),
                        value_heads,
                        self.hidden,
                    ),
                    (
                        format!("{prefix}.linear_attn.out_proj.weight"),
                        self.hidden,
                        value,
                    ),
                ]);
            }
        }
        matrices.extend([
            (
                format!("{prefix}.mlp.gate.weight"),
                self.routed_experts,
                self.hidden,
            ),
            (
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                1,
                self.hidden,
            ),
            (
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                self.shared_intermediate,
                self.hidden,
            ),
            (
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                self.shared_intermediate,
                self.hidden,
            ),
            (
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                self.hidden,
                self.shared_intermediate,
            ),
        ]);
        Ok(matrices)
    }

    fn vectors(&self) -> Result<Vec<(String, Vec<u64>)>, Qwen38LayerWeightError> {
        let prefix = self.prefix();
        let mut vectors = vec![
            (
                format!("{prefix}.input_layernorm.weight"),
                vec![self.hidden as u64],
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![self.hidden as u64],
            ),
        ];
        match self.attention {
            AttentionShape::Full { head_dim, .. } => vectors.extend([
                (
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![head_dim as u64],
                ),
                (
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![head_dim as u64],
                ),
            ]),
            AttentionShape::Linear {
                value_heads,
                value_head_dim,
                key_heads,
                key_head_dim,
                kernel,
            } => {
                let key = checked_product(&[key_heads, key_head_dim], "linear key width")?;
                let value = checked_product(&[value_heads, value_head_dim], "linear value width")?;
                let channels = key
                    .checked_mul(2)
                    .and_then(|twice| twice.checked_add(value))
                    .ok_or_else(|| {
                        Qwen38LayerWeightError::Invalid(
                            "linear convolution width overflows".to_owned(),
                        )
                    })?;
                vectors.extend([
                    (
                        format!("{prefix}.linear_attn.A_log"),
                        vec![value_heads as u64],
                    ),
                    (
                        format!("{prefix}.linear_attn.dt_bias"),
                        vec![value_heads as u64],
                    ),
                    (
                        format!("{prefix}.linear_attn.norm.weight"),
                        vec![value_head_dim as u64],
                    ),
                    (
                        format!("{prefix}.linear_attn.conv1d.weight"),
                        vec![channels as u64, 1, kernel as u64],
                    ),
                ]);
            }
        }
        Ok(vectors)
    }
}

fn inspect_layer_resident_bytes(
    index: &TensorIndex,
    shape: &LayerShape,
) -> Result<u64, Qwen38LayerWeightError> {
    let mut bytes = 0u64;
    for (name, rows, cols) in shape.matrices()? {
        bytes = bytes
            .checked_add(inspect_compact_bf16_matrix(index, &name, rows, cols)?.resident_bytes)
            .ok_or_else(|| {
                Qwen38LayerWeightError::Invalid("layer matrix bytes overflow".to_owned())
            })?;
    }
    for (name, expected_shape) in shape.vectors()? {
        let tensor = index.require(&name).map_err(TensorLoadError::from)?;
        if tensor.dtype != DType::Bf16 || tensor.shape != expected_shape {
            return Err(Qwen38LayerWeightError::Vector(
                TensorLoadError::InvalidShape(format!(
                    "tensor {name:?} must be BF16 {expected_shape:?}, got {} {:?}",
                    tensor.dtype, tensor.shape
                )),
            ));
        }
        let length = expected_shape.iter().try_fold(1u64, |product, dimension| {
            product.checked_mul(*dimension).ok_or_else(|| {
                Qwen38LayerWeightError::Invalid("layer vector elements overflow".to_owned())
            })
        })?;
        bytes = bytes
            .checked_add(length.checked_mul(4).ok_or_else(|| {
                Qwen38LayerWeightError::Invalid("layer vector bytes overflow".to_owned())
            })?)
            .ok_or_else(|| {
                Qwen38LayerWeightError::Invalid("layer resident bytes overflow".to_owned())
            })?;
    }
    Ok(bytes)
}

fn load_layer(
    index: &TensorIndex,
    shape: LayerShape,
    resident_bytes: u64,
) -> Result<Qwen38LayerWeights, Qwen38LayerWeightError> {
    let matrix_specs = shape.matrices()?;
    let borrowed = matrix_specs
        .iter()
        .map(|(name, rows, cols)| (name.as_str(), *rows, *cols))
        .collect::<Vec<_>>();
    let mut matrices = load_compact_bf16_matrices(index, &borrowed, resident_bytes)?.into_iter();
    let prefix = shape.prefix();
    let input_norm = load_reference_vector(
        index,
        &format!("{prefix}.input_layernorm.weight"),
        shape.hidden,
    )?;
    let post_attention_norm = load_reference_vector(
        index,
        &format!("{prefix}.post_attention_layernorm.weight"),
        shape.hidden,
    )?;

    let attention = match shape.attention {
        AttentionShape::Full { head_dim, .. } => {
            Qwen38AttentionWeights::Full(Box::new(Qwen38FullAttentionWeights {
                query: next_matrix(&mut matrices)?,
                key: next_matrix(&mut matrices)?,
                value: next_matrix(&mut matrices)?,
                output: next_matrix(&mut matrices)?,
                query_norm: load_reference_vector(
                    index,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    head_dim,
                )?,
                key_norm: load_reference_vector(
                    index,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    head_dim,
                )?,
            }))
        }
        AttentionShape::Linear {
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            kernel,
        } => {
            let channels = checked_product(&[key_heads, key_head_dim], "linear key width")?
                .checked_mul(2)
                .and_then(|value| value.checked_add(value_heads.checked_mul(value_head_dim)?))
                .ok_or_else(|| {
                    Qwen38LayerWeightError::Invalid(
                        "linear convolution channel count overflows".to_owned(),
                    )
                })?;
            Qwen38AttentionWeights::Linear(Box::new(Qwen38LinearAttentionWeights {
                qkv: next_matrix(&mut matrices)?,
                z: next_matrix(&mut matrices)?,
                beta: next_matrix(&mut matrices)?,
                decay: next_matrix(&mut matrices)?,
                output: next_matrix(&mut matrices)?,
                convolution: {
                    let name = format!("{prefix}.linear_attn.conv1d.weight");
                    let tensor = index.require(&name).map_err(TensorLoadError::from)?;
                    let expected = [channels as u64, 1, kernel as u64];
                    if tensor.shape != expected {
                        return Err(Qwen38LayerWeightError::Vector(
                            TensorLoadError::InvalidShape(format!(
                                "tensor {name:?} must declare {expected:?}, got {:?}",
                                tensor.shape
                            )),
                        ));
                    }
                    load_reference_values(index, &name)?
                },
                a_log: load_reference_vector(
                    index,
                    &format!("{prefix}.linear_attn.A_log"),
                    value_heads,
                )?,
                dt_bias: load_reference_vector(
                    index,
                    &format!("{prefix}.linear_attn.dt_bias"),
                    value_heads,
                )?,
                output_norm: load_reference_vector(
                    index,
                    &format!("{prefix}.linear_attn.norm.weight"),
                    value_head_dim,
                )?,
            }))
        }
    };

    let moe = Qwen38MoeTrunkWeights {
        router: next_matrix(&mut matrices)?,
        shared_expert_gate: next_matrix(&mut matrices)?,
        shared_gate: next_matrix(&mut matrices)?,
        shared_expert_up: next_matrix(&mut matrices)?,
        shared_expert_down: next_matrix(&mut matrices)?,
    };
    if matrices.next().is_some() {
        return Err(Qwen38LayerWeightError::Invalid(
            "layer loader left unassigned matrices".to_owned(),
        ));
    }
    Ok(Qwen38LayerWeights {
        input_norm,
        post_attention_norm,
        attention,
        moe,
    })
}

fn next_matrix(
    matrices: &mut impl Iterator<Item = WeightMatrix>,
) -> Result<WeightMatrix, Qwen38LayerWeightError> {
    matrices
        .next()
        .ok_or_else(|| Qwen38LayerWeightError::Invalid("layer matrix batch ended early".to_owned()))
}

fn checked_product(
    factors: &[usize],
    description: &'static str,
) -> Result<usize, Qwen38LayerWeightError> {
    factors.iter().try_fold(1usize, |product, factor| {
        product.checked_mul(*factor).ok_or_else(|| {
            Qwen38LayerWeightError::Invalid(format!("{description} overflows usize"))
        })
    })
}
