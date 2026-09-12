use super::{DenseMatrix, MatrixError, WeightError, WeightMatrix};
use crate::math::silu;
use std::fmt;

#[derive(Debug)]
pub enum MlpError {
    Matrix(MatrixError),
    Weight(WeightError),
    InvalidShape(String),
}

impl fmt::Display for MlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(f),
            Self::Weight(error) => error.fmt(f),
            Self::InvalidShape(reason) => write!(f, "invalid gated MLP shape: {reason}"),
        }
    }
}

impl std::error::Error for MlpError {}

impl From<MatrixError> for MlpError {
    fn from(value: MatrixError) -> Self {
        Self::Matrix(value)
    }
}

impl From<WeightError> for MlpError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

/// GLM's bias-free SwiGLU-style `down(SiLU(gate(x)) * up(x))` sublayer.
#[derive(Debug, Clone)]
pub struct GatedMlp {
    gate: WeightMatrix,
    up: WeightMatrix,
    down: WeightMatrix,
}

impl GatedMlp {
    pub fn new(gate: DenseMatrix, up: DenseMatrix, down: DenseMatrix) -> Result<Self, MlpError> {
        Self::new_mixed(gate.into(), up.into(), down.into())
    }

    pub fn new_mixed(
        gate: WeightMatrix,
        up: WeightMatrix,
        down: WeightMatrix,
    ) -> Result<Self, MlpError> {
        if gate.rows() != up.rows() || gate.cols() != up.cols() {
            return Err(MlpError::InvalidShape(format!(
                "gate is [{},{}], up is [{},{}]",
                gate.rows(),
                gate.cols(),
                up.rows(),
                up.cols()
            )));
        }
        if down.cols() != gate.rows() || down.rows() != gate.cols() {
            return Err(MlpError::InvalidShape(format!(
                "down [{},{}] must map intermediate {} back to hidden {}",
                down.rows(),
                down.cols(),
                gate.rows(),
                gate.cols()
            )));
        }
        Ok(Self { gate, up, down })
    }

    pub fn hidden_size(&self) -> usize {
        self.gate.cols()
    }

    pub fn intermediate_size(&self) -> usize {
        self.gate.rows()
    }

    pub fn resident_bytes(&self) -> usize {
        self.gate
            .resident_bytes()
            .saturating_add(self.up.resident_bytes())
            .saturating_add(self.down.resident_bytes())
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, MlpError> {
        let gate = self.gate.matvec(input)?;
        let up = self.up.matvec(input)?;
        let activated: Vec<f32> = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| silu(gate) * up)
            .collect();
        Ok(self.down.matvec(&activated)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_mlp_matches_explicit_formula() {
        let gate = DenseMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let up = DenseMatrix::new(2, 2, vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let down = DenseMatrix::new(2, 2, vec![1.0, 1.0, 1.0, -1.0]).unwrap();
        let mlp = GatedMlp::new(gate, up, down).unwrap();
        let output = mlp.forward(&[1.0, -2.0]).unwrap();
        let first = silu(1.0) * 2.0;
        let second = silu(-2.0) * -6.0;
        assert!((output[0] - (first + second)).abs() < 1e-6);
        assert!((output[1] - (first - second)).abs() < 1e-6);
    }

    #[test]
    fn rejects_transposed_down_projection() {
        let gate = DenseMatrix::zeros(3, 2).unwrap();
        let up = DenseMatrix::zeros(3, 2).unwrap();
        let wrong_down = DenseMatrix::zeros(3, 2).unwrap();
        assert!(GatedMlp::new(gate, up, wrong_down).is_err());
    }
}
