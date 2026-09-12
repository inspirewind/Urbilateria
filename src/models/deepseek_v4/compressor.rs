//! Incremental learned KV compression shared by main attention and the ratio-4 indexer.

use super::math::{linear, paired_rope, round_to_bf16_in_place, DeepseekMathError};
use crate::math::{normalized_hadamard, simulate_e2m1_activation, simulate_e4m3_activation};
use crate::model::{WeightError, WeightMatrix};
use std::fmt;

#[derive(Debug)]
pub enum CompressorError {
    Invalid(String),
    Math(DeepseekMathError),
    Weight(WeightError),
}

impl fmt::Display for CompressorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid DeepSeek-V4 compressor: {reason}"),
            Self::Math(error) => error.fmt(f),
            Self::Weight(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CompressorError {}

impl From<DeepseekMathError> for CompressorError {
    fn from(value: DeepseekMathError) -> Self {
        Self::Math(value)
    }
}

impl From<WeightError> for CompressorError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

#[derive(Debug, Clone)]
pub struct CompressorWeights {
    ratio: usize,
    head_dim: usize,
    rope_head_dim: usize,
    rotate: bool,
    ape: WeightMatrix,
    wkv: WeightMatrix,
    wgate: WeightMatrix,
    norm: Vec<f32>,
    norm_eps: f32,
}

impl CompressorWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ratio: usize,
        head_dim: usize,
        rope_head_dim: usize,
        rotate: bool,
        ape: WeightMatrix,
        wkv: WeightMatrix,
        wgate: WeightMatrix,
        norm: Vec<f32>,
        norm_eps: f32,
    ) -> Result<Self, CompressorError> {
        let coefficient = if ratio == 4 { 2 } else { 1 };
        let width = coefficient * head_dim;
        if ratio == 0
            || head_dim == 0
            || rope_head_dim == 0
            || rope_head_dim > head_dim
            || ape.rows() != ratio
            || ape.cols() != width
            || wkv.rows() != width
            || wgate.rows() != width
            || wkv.cols() != wgate.cols()
            || norm.len() != head_dim
            || !norm_eps.is_finite()
            || norm_eps <= 0.0
        {
            return Err(CompressorError::Invalid(
                "weights do not match ratio/head/input geometry".to_owned(),
            ));
        }
        Ok(Self {
            ratio,
            head_dim,
            rope_head_dim,
            rotate,
            ape,
            wkv,
            wgate,
            norm,
            norm_eps,
        })
    }

    pub fn ratio(&self) -> usize {
        self.ratio
    }

    pub fn input_size(&self) -> usize {
        self.wkv.cols()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_token<'a>(
        &self,
        input: &[f32],
        position: usize,
        state: &'a mut CompressorState,
        rope_base: f32,
        original_sequence_length: Option<usize>,
        rope_factor: f32,
        beta_fast: usize,
        beta_slow: usize,
    ) -> Result<Option<&'a [f32]>, CompressorError> {
        if input.len() != self.input_size()
            || state.ratio != self.ratio
            || state.head_dim != self.head_dim
            || state.overlap != (self.ratio == 4)
            || state.next_position != position
        {
            return Err(CompressorError::Invalid(format!(
                "input/state mismatch at position {position}"
            )));
        }
        let width = state.width();
        let slot = position % self.ratio;
        let target_slot = if state.overlap {
            self.ratio + slot
        } else {
            slot
        };
        let kv = linear(&self.wkv, input)?;
        let mut score = linear(&self.wgate, input)?;
        let ape = self.ape.row(slot)?;
        for (score, ape) in score.iter_mut().zip(ape) {
            *score += ape;
        }
        state.kv[target_slot * width..(target_slot + 1) * width].copy_from_slice(&kv);
        state.score[target_slot * width..(target_slot + 1) * width].copy_from_slice(&score);
        state.next_position += 1;
        if (position + 1) % self.ratio != 0 {
            return Ok(None);
        }

        let mut compressed = vec![0.0; self.head_dim];
        for (feature, output) in compressed.iter_mut().enumerate() {
            if state.overlap {
                let candidates = (0..self.ratio)
                    .map(|row| (row, feature))
                    .chain((0..self.ratio).map(|row| (self.ratio + row, self.head_dim + feature)));
                *output = weighted_pool(&state.kv, &state.score, width, candidates);
            } else {
                *output = weighted_pool(
                    &state.kv,
                    &state.score,
                    width,
                    (0..self.ratio).map(|row| (row, feature)),
                );
            }
        }
        // Official compression pools in FP32, casts to the model dtype, then applies RMSNorm.
        round_to_bf16_in_place(&mut compressed)?;
        rms_norm_in_place(&mut compressed, &self.norm, self.norm_eps)?;
        paired_rope(
            &mut compressed[self.head_dim - self.rope_head_dim..],
            position + 1 - self.ratio,
            rope_base,
            original_sequence_length,
            rope_factor,
            beta_fast,
            beta_slow,
            false,
        )?;
        if self.rotate {
            normalized_hadamard(&mut compressed)
                .map_err(|error| CompressorError::Invalid(error.to_string()))?;
            round_to_bf16_in_place(&mut compressed)?;
            compressed = simulate_e2m1_activation(&compressed, 32)
                .map_err(|error| CompressorError::Invalid(error.to_string()))?;
        } else {
            let non_rope = self.head_dim - self.rope_head_dim;
            if non_rope > 0 {
                let quantized = simulate_e4m3_activation(&compressed[..non_rope], 64)
                    .map_err(|error| CompressorError::Invalid(error.to_string()))?;
                compressed[..non_rope].copy_from_slice(&quantized);
            }
        }
        state.compressed.push(compressed);

        if state.overlap {
            for row in 0..self.ratio {
                let current = (self.ratio + row) * width;
                let previous = row * width;
                state.kv.copy_within(current..current + width, previous);
                state.score.copy_within(current..current + width, previous);
                state.kv[current..current + width].fill(0.0);
                state.score[current..current + width].fill(f32::NEG_INFINITY);
            }
        } else {
            state.kv.fill(0.0);
            state.score.fill(f32::NEG_INFINITY);
        }
        Ok(state.compressed.last().map(Vec::as_slice))
    }
}

#[derive(Debug, Clone)]
pub struct CompressorState {
    ratio: usize,
    head_dim: usize,
    overlap: bool,
    next_position: usize,
    kv: Vec<f32>,
    score: Vec<f32>,
    compressed: Vec<Vec<f32>>,
}

impl CompressorState {
    pub fn new(ratio: usize, head_dim: usize) -> Result<Self, CompressorError> {
        if ratio == 0 || head_dim == 0 {
            return Err(CompressorError::Invalid(
                "ratio and head dimension must be non-zero".to_owned(),
            ));
        }
        let coefficient = if ratio == 4 { 2 } else { 1 };
        let rows = coefficient * ratio;
        let width = coefficient * head_dim;
        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| CompressorError::Invalid("state size overflows".to_owned()))?;
        Ok(Self {
            ratio,
            head_dim,
            overlap: ratio == 4,
            next_position: 0,
            kv: vec![0.0; elements],
            score: vec![f32::NEG_INFINITY; elements],
            compressed: Vec::new(),
        })
    }

    fn width(&self) -> usize {
        (1 + usize::from(self.overlap)) * self.head_dim
    }

    pub fn compressed(&self) -> &[Vec<f32>] {
        &self.compressed
    }

    pub fn stored_f32_elements(&self) -> usize {
        self.kv.len() + self.score.len() + self.compressed.iter().map(Vec::len).sum::<usize>()
    }
}

fn weighted_pool(
    values: &[f32],
    scores: &[f32],
    width: usize,
    candidates: impl Iterator<Item = (usize, usize)>,
) -> f32 {
    let candidates = candidates.collect::<Vec<_>>();
    let maximum = candidates
        .iter()
        .map(|&(row, feature)| scores[row * width + feature])
        .fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return 0.0;
    }
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (row, feature) in candidates {
        let weight = f64::from((scores[row * width + feature] - maximum).exp());
        numerator += weight * f64::from(values[row * width + feature]);
        denominator += weight;
    }
    (numerator / denominator) as f32
}

fn rms_norm_in_place(values: &mut [f32], weight: &[f32], eps: f32) -> Result<(), CompressorError> {
    let mean_square = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / values.len() as f64;
    let scale = (mean_square + f64::from(eps)).sqrt().recip() as f32;
    for ((value, &weight), index) in values.iter_mut().zip(weight).zip(0..) {
        *value *= scale * weight;
        if !value.is_finite() {
            return Err(CompressorError::Invalid(format!(
                "RMSNorm produced a non-finite value at {index}"
            )));
        }
    }
    round_to_bf16_in_place(values)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DenseMatrix;

    fn dense(rows: usize, cols: usize, values: Vec<f32>) -> WeightMatrix {
        DenseMatrix::new(rows, cols, values).unwrap().into()
    }

    #[test]
    fn non_overlapping_compressor_emits_only_complete_groups() {
        let weights = CompressorWeights::new(
            2,
            2,
            2,
            false,
            dense(2, 2, vec![0.0; 4]),
            dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            dense(2, 2, vec![0.0; 4]),
            vec![1.0, 1.0],
            1e-6,
        )
        .unwrap();
        let mut state = CompressorState::new(2, 2).unwrap();
        assert!(weights
            .forward_token(&[1.0, 0.0], 0, &mut state, 10_000.0, None, 1.0, 32, 1)
            .unwrap()
            .is_none());
        let output = weights
            .forward_token(&[3.0, 0.0], 1, &mut state, 10_000.0, None, 1.0, 32, 1)
            .unwrap()
            .unwrap()
            .to_vec();
        assert_eq!(output[0], 1.4140625); // BF16-rounded sqrt(2)
        assert_eq!(state.compressed().len(), 1);
    }
}
