//! Incremental learned KV compression shared by main attention and the ratio-4 indexer.

use super::math::{
    linear_with_mx_activation_batch, paired_rope, prepare_mx_activation, round_to_bf16_in_place,
    DeepseekMathError,
};
use crate::math::{normalized_hadamard, simulate_e2m1_activation, simulate_e4m3_activation};
use crate::model::{WeightError, WeightMatrix};
use std::fmt;
use std::sync::Arc;

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
    norm: Arc<[f32]>,
    norm_eps: f32,
}

#[derive(Debug)]
pub(super) struct PreparedCompressor {
    kv: Vec<f32>,
    score: Vec<f32>,
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
        norm: impl Into<Arc<[f32]>>,
        norm_eps: f32,
    ) -> Result<Self, CompressorError> {
        let norm = norm.into();
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

    /// Returns the three native projection matrices when a streamed decoder layer is retired.
    /// The caller can then reuse their owned checkpoint buffers for a later same-shaped layer.
    pub(super) fn into_weight_matrices(self) -> [WeightMatrix; 3] {
        [self.ape, self.wkv, self.wgate]
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
        let prepared = self
            .prepare_batch(&[input])?
            .pop()
            .expect("one compressor input produces one prepared projection");
        self.forward_prepared(
            prepared,
            position,
            state,
            rope_base,
            original_sequence_length,
            rope_factor,
            beta_fast,
            beta_slow,
        )
    }

    pub(super) fn prepare_batch(
        &self,
        inputs: &[&[f32]],
    ) -> Result<Vec<PreparedCompressor>, CompressorError> {
        if inputs.is_empty() || inputs.iter().any(|input| input.len() != self.input_size()) {
            return Err(CompressorError::Invalid(
                "projection batch has invalid input geometry".to_owned(),
            ));
        }
        let batch = inputs.len();
        let mut input = Vec::with_capacity(batch.saturating_mul(self.input_size()));
        let uses_mx = self.wkv.uses_mx_activation_quantization()
            || self.wgate.uses_mx_activation_quantization();
        let mut mx_input = Vec::with_capacity(if uses_mx { input.capacity() } else { 0 });
        for &token in inputs {
            input.extend_from_slice(token);
            if uses_mx {
                mx_input.extend(prepare_mx_activation(token)?);
            }
        }
        let kv = linear_with_mx_activation_batch(&self.wkv, &input, &mx_input, batch)?;
        let score = linear_with_mx_activation_batch(&self.wgate, &input, &mx_input, batch)?;
        let width = self.wkv.rows();
        Ok(kv
            .chunks_exact(width)
            .zip(score.chunks_exact(width))
            .map(|(kv, score)| PreparedCompressor {
                kv: kv.to_vec(),
                score: score.to_vec(),
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_prepared<'a>(
        &self,
        prepared: PreparedCompressor,
        position: usize,
        state: &'a mut CompressorState,
        rope_base: f32,
        original_sequence_length: Option<usize>,
        rope_factor: f32,
        beta_fast: usize,
        beta_slow: usize,
    ) -> Result<Option<&'a [f32]>, CompressorError> {
        if state.ratio != self.ratio
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
        let PreparedCompressor { kv, mut score } = prepared;
        if kv.len() != width || score.len() != width {
            return Err(CompressorError::Invalid(
                "prepared projection has invalid output geometry".to_owned(),
            ));
        }
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

/// Fixed-size transactional snapshot. Compressed history is append-only during one token, so a
/// length marker is sufficient; only the rolling pooling workspace needs to be copied.
#[derive(Debug)]
pub(crate) struct CompressorCheckpoint {
    next_position: usize,
    kv: Vec<f32>,
    score: Vec<f32>,
    compressed_len: usize,
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

    pub(crate) fn checkpoint(&self) -> CompressorCheckpoint {
        CompressorCheckpoint {
            next_position: self.next_position,
            kv: self.kv.clone(),
            score: self.score.clone(),
            compressed_len: self.compressed.len(),
        }
    }

    pub(crate) fn restore(&mut self, checkpoint: CompressorCheckpoint) {
        self.next_position = checkpoint.next_position;
        self.kv = checkpoint.kv;
        self.score = checkpoint.score;
        self.compressed.truncate(checkpoint.compressed_len);
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

    #[test]
    fn checkpoint_restores_rolling_workspace_and_truncates_history() {
        let mut state = CompressorState::new(2, 2).unwrap();
        state.next_position = 3;
        state.kv.fill(1.0);
        state.score.fill(2.0);
        state.compressed.push(vec![3.0, 4.0]);
        let checkpoint = state.checkpoint();

        state.next_position = 4;
        state.kv.fill(5.0);
        state.score.fill(6.0);
        state.compressed.push(vec![7.0, 8.0]);
        state.restore(checkpoint);

        assert_eq!(state.next_position, 3);
        assert!(state.kv.iter().all(|&value| value == 1.0));
        assert!(state.score.iter().all(|&value| value == 2.0));
        assert_eq!(state.compressed, [vec![3.0, 4.0]]);
    }

    #[test]
    fn prepared_batch_matches_incremental_overlap_state_exactly() {
        let weights = CompressorWeights::new(
            4,
            2,
            2,
            true,
            dense(
                4,
                4,
                vec![
                    0.1, -0.2, 0.3, -0.4, -0.2, 0.3, -0.4, 0.5, 0.3, -0.4, 0.5, -0.6, -0.4, 0.5,
                    -0.6, 0.7,
                ],
            ),
            dense(4, 2, vec![0.5, -0.25, -0.75, 0.5, 0.25, 0.75, -0.5, -0.25]),
            dense(4, 2, vec![0.2, -0.1, -0.3, 0.4, 0.5, 0.2, -0.4, 0.3]),
            vec![1.0, 0.75],
            1e-6,
        )
        .unwrap();
        let inputs = (0..9)
            .map(|token| vec![token as f32 * 0.125 - 0.5, 0.75 - token as f32 * 0.0625])
            .collect::<Vec<_>>();

        let mut sequential_state = CompressorState::new(4, 2).unwrap();
        let mut sequential_outputs = Vec::new();
        for (position, input) in inputs.iter().enumerate() {
            sequential_outputs.push(
                weights
                    .forward_token(
                        input,
                        position,
                        &mut sequential_state,
                        10_000.0,
                        Some(4),
                        2.0,
                        32,
                        1,
                    )
                    .unwrap()
                    .map(<[f32]>::to_vec),
            );
        }

        let input_refs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let prepared = weights.prepare_batch(&input_refs).unwrap();
        let mut prepared_state = CompressorState::new(4, 2).unwrap();
        let mut prepared_outputs = Vec::new();
        for (position, prepared) in prepared.into_iter().enumerate() {
            prepared_outputs.push(
                weights
                    .forward_prepared(
                        prepared,
                        position,
                        &mut prepared_state,
                        10_000.0,
                        Some(4),
                        2.0,
                        32,
                        1,
                    )
                    .unwrap()
                    .map(<[f32]>::to_vec),
            );
        }

        assert_eq!(prepared_outputs, sequential_outputs);
        assert_eq!(prepared_state.next_position, sequential_state.next_position);
        assert_eq!(prepared_state.kv, sequential_state.kv);
        assert_eq!(prepared_state.score, sequential_state.score);
        assert_eq!(prepared_state.compressed, sequential_state.compressed);
    }
}
