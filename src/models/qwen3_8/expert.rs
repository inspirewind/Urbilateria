//! Scalar Qwen3.8 SwiGLU expert used by the correctness runtime.
//!
//! Qwen's routed and shared experts have the same equation but may have different
//! intermediate widths:
//!
//! `down_proj(SiLU(gate_proj(x)) * up_proj(x))`.
//!
//! The checkpoint stores every projection as `[output, input]`.  Keeping the
//! projections behind [`WeightMatrix`] lets the same forward path exercise dense
//! tiny fixtures, BF16 trunk weights, and block-FP8 routed experts.

use super::math::{linear, round_to_bf16, round_to_bf16_in_place, uses_bf16_output};
use crate::math::silu;
use crate::model::{WeightError, WeightMatrix};
use std::fmt;

#[derive(Debug)]
pub enum Qwen38ExpertError {
    Weight {
        projection: &'static str,
        source: WeightError,
    },
    InvalidShape(String),
    NonFinite {
        operation: &'static str,
        index: usize,
    },
}

impl fmt::Display for Qwen38ExpertError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight { projection, source } => {
                write!(formatter, "Qwen3.8 expert {projection}: {source}")
            }
            Self::InvalidShape(reason) => {
                write!(formatter, "invalid Qwen3.8 expert shape: {reason}")
            }
            Self::NonFinite { operation, index } => write!(
                formatter,
                "Qwen3.8 expert {operation} produced NaN or infinity at index {index}"
            ),
        }
    }
}

impl std::error::Error for Qwen38ExpertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight { source, .. } => Some(source),
            Self::InvalidShape(_) | Self::NonFinite { .. } => None,
        }
    }
}

/// One bias-free Qwen3.8 SwiGLU expert.
#[derive(Debug, Clone)]
pub struct Qwen38Expert {
    gate_proj: WeightMatrix,
    up_proj: WeightMatrix,
    down_proj: WeightMatrix,
}

impl Qwen38Expert {
    /// Builds an expert from `[intermediate, hidden]` gate/up projections and a
    /// `[hidden, intermediate]` down projection.
    pub fn new(
        gate_proj: WeightMatrix,
        up_proj: WeightMatrix,
        down_proj: WeightMatrix,
    ) -> Result<Self, Qwen38ExpertError> {
        let hidden_size = gate_proj.cols();
        let intermediate_size = gate_proj.rows();
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(Qwen38ExpertError::InvalidShape(
                "hidden and intermediate dimensions must be non-zero".to_owned(),
            ));
        }
        if up_proj.rows() != intermediate_size || up_proj.cols() != hidden_size {
            return Err(Qwen38ExpertError::InvalidShape(format!(
                "up_proj is [{}, {}], expected [{intermediate_size}, {hidden_size}]",
                up_proj.rows(),
                up_proj.cols()
            )));
        }
        if down_proj.rows() != hidden_size || down_proj.cols() != intermediate_size {
            return Err(Qwen38ExpertError::InvalidShape(format!(
                "down_proj is [{}, {}], expected [{hidden_size}, {intermediate_size}]",
                down_proj.rows(),
                down_proj.cols()
            )));
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.gate_proj.cols()
    }

    pub fn intermediate_size(&self) -> usize {
        self.gate_proj.rows()
    }

    pub fn resident_bytes(&self) -> usize {
        self.gate_proj
            .resident_bytes()
            .saturating_add(self.up_proj.resident_bytes())
            .saturating_add(self.down_proj.resident_bytes())
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Qwen38ExpertError> {
        Self::forward_projections(&self.gate_proj, &self.up_proj, &self.down_proj, input)
    }

    pub(crate) fn forward_projections(
        gate_proj: &WeightMatrix,
        up_proj: &WeightMatrix,
        down_proj: &WeightMatrix,
        input: &[f32],
    ) -> Result<Vec<f32>, Qwen38ExpertError> {
        let hidden_size = gate_proj.cols();
        let intermediate_size = gate_proj.rows();
        if up_proj.rows() != intermediate_size
            || up_proj.cols() != hidden_size
            || down_proj.rows() != hidden_size
            || down_proj.cols() != intermediate_size
        {
            return Err(Qwen38ExpertError::InvalidShape(
                "expert projection shapes disagree".to_owned(),
            ));
        }
        if input.len() != hidden_size {
            return Err(Qwen38ExpertError::InvalidShape(format!(
                "input has length {}, expected hidden size {}",
                input.len(),
                hidden_size
            )));
        }
        validate_finite("input", input)?;

        let bf16_output = uses_bf16_output(gate_proj);
        let gate = linear(gate_proj, input).map_err(|source| Qwen38ExpertError::Weight {
            projection: "gate_proj",
            source,
        })?;
        let up = linear(up_proj, input).map_err(|source| Qwen38ExpertError::Weight {
            projection: "up_proj",
            source,
        })?;
        validate_projection("gate_proj", &gate, intermediate_size)?;
        validate_projection("up_proj", &up, intermediate_size)?;

        let mut activated = Vec::with_capacity(intermediate_size);
        for (index, (gate, up)) in gate.into_iter().zip(up).enumerate() {
            let activated_gate = if bf16_output {
                round_to_bf16(silu(gate)).map_err(|source| Qwen38ExpertError::Weight {
                    projection: "SwiGLU gate cast",
                    source,
                })?
            } else {
                silu(gate)
            };
            let value = activated_gate * up;
            if !value.is_finite() {
                return Err(Qwen38ExpertError::NonFinite {
                    operation: "SwiGLU activation",
                    index,
                });
            }
            activated.push(value);
        }
        if bf16_output {
            round_to_bf16_in_place(&mut activated).map_err(|source| Qwen38ExpertError::Weight {
                projection: "SwiGLU activation cast",
                source,
            })?;
        }

        let output = linear(down_proj, &activated).map_err(|source| Qwen38ExpertError::Weight {
            projection: "down_proj",
            source,
        })?;
        validate_projection("down_proj", &output, hidden_size)?;
        Ok(output)
    }
}

fn validate_projection(
    operation: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), Qwen38ExpertError> {
    if values.len() != expected {
        return Err(Qwen38ExpertError::InvalidShape(format!(
            "{operation} returned {} values, expected {expected}",
            values.len()
        )));
    }
    validate_finite(operation, values)
}

fn validate_finite(operation: &'static str, values: &[f32]) -> Result<(), Qwen38ExpertError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(Qwen38ExpertError::NonFinite { operation, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DenseMatrix;

    fn dense(rows: usize, columns: usize, values: Vec<f32>) -> WeightMatrix {
        WeightMatrix::F32(DenseMatrix::new(rows, columns, values).unwrap())
    }

    #[test]
    fn expert_matches_explicit_swiglu_equation() {
        let expert = Qwen38Expert::new(
            dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            dense(2, 2, vec![2.0, 0.0, 0.0, 3.0]),
            dense(2, 2, vec![1.0, 1.0, 1.0, -1.0]),
        )
        .unwrap();

        let output = expert.forward(&[1.0, -2.0]).unwrap();
        let first = silu(1.0) * 2.0;
        let second = silu(-2.0) * -6.0;
        assert!((output[0] - (first + second)).abs() < 1e-6);
        assert!((output[1] - (first - second)).abs() < 1e-6);
    }

    #[test]
    fn expert_geometry_is_not_tied_to_release_dimensions() {
        let expert = Qwen38Expert::new(
            dense(3, 2, vec![0.0; 6]),
            dense(3, 2, vec![0.0; 6]),
            dense(2, 3, vec![0.0; 6]),
        )
        .unwrap();
        assert_eq!(expert.hidden_size(), 2);
        assert_eq!(expert.intermediate_size(), 3);
        assert_eq!(expert.forward(&[4.0, -1.0]).unwrap(), [0.0, 0.0]);
    }

    #[test]
    fn rejects_bad_down_shape_and_non_finite_input() {
        assert!(matches!(
            Qwen38Expert::new(
                dense(3, 2, vec![0.0; 6]),
                dense(3, 2, vec![0.0; 6]),
                dense(3, 2, vec![0.0; 6]),
            ),
            Err(Qwen38ExpertError::InvalidShape(_))
        ));

        let expert = Qwen38Expert::new(
            dense(1, 1, vec![1.0]),
            dense(1, 1, vec![1.0]),
            dense(1, 1, vec![1.0]),
        )
        .unwrap();
        assert!(matches!(
            expert.forward(&[f32::NAN]),
            Err(Qwen38ExpertError::NonFinite {
                operation: "input",
                index: 0
            })
        ));
    }
}
