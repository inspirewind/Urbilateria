//! Storage-independent matrix and neural-network building blocks.

mod bf16;
mod matrix;
mod mlp;
mod weight;

pub use bf16::Bf16Matrix;
pub use matrix::{DenseMatrix, MatrixError};
pub use mlp::{GatedMlp, MlpError};
pub use weight::{WeightError, WeightMatrix};
