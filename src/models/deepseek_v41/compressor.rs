//! Incremental learned CSA2 compression state for DeepSeek-V4.1.
//!
//! V4.1 pools non-overlapping groups. Unlike the older V4 adapter, it has no APE tensor and no
//! ratio-4 overlapping two-bank state; keeping this implementation separate prevents accidental
//! reuse of the old graph.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CompressorError {
    Invalid(String),
}

impl fmt::Display for CompressorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(
                formatter,
                "invalid DeepSeek-V4.1 compressor state: {reason}"
            ),
        }
    }
}

impl std::error::Error for CompressorError {}

/// Carries only the unfinished compression group across prefill/decode boundaries.
#[derive(Debug, Clone)]
pub struct CompressorState {
    ratio: usize,
    width: usize,
    next_position: usize,
    values: Vec<f32>,
    scores: Vec<f32>,
}

impl CompressorState {
    pub fn new(ratio: usize, width: usize) -> Result<Self, CompressorError> {
        if ratio == 0 || width == 0 {
            return Err(CompressorError::Invalid(
                "ratio and width must be non-zero".to_owned(),
            ));
        }
        let elements = ratio
            .checked_mul(width)
            .ok_or_else(|| CompressorError::Invalid("state geometry overflows".to_owned()))?;
        Ok(Self {
            ratio,
            width,
            next_position: 0,
            values: vec![0.0; elements],
            scores: vec![f32::NEG_INFINITY; elements],
        })
    }

    /// Accepts the outputs of `wkv` and `wgate` for one token. Ratio-one compressors have no gate;
    /// pass `None` and the projected row is emitted immediately.
    pub fn push(
        &mut self,
        position: usize,
        projected: &[f32],
        gate_scores: Option<&[f32]>,
    ) -> Result<Option<Vec<f32>>, CompressorError> {
        if position != self.next_position
            || projected.len() != self.width
            || projected.iter().any(|value| !value.is_finite())
        {
            return Err(CompressorError::Invalid(format!(
                "expected position {} and {} finite projected values",
                self.next_position, self.width
            )));
        }
        if self.ratio == 1 {
            if gate_scores.is_some() {
                return Err(CompressorError::Invalid(
                    "ratio-one compression must not use wgate".to_owned(),
                ));
            }
            self.next_position += 1;
            return Ok(Some(projected.to_vec()));
        }
        let scores = gate_scores.ok_or_else(|| {
            CompressorError::Invalid("pooled compression requires wgate scores".to_owned())
        })?;
        if scores.len() != self.width || scores.iter().any(|value| !value.is_finite()) {
            return Err(CompressorError::Invalid(format!(
                "wgate must contain {} finite values",
                self.width
            )));
        }
        let slot = position % self.ratio;
        let start = slot * self.width;
        self.values[start..start + self.width].copy_from_slice(projected);
        self.scores[start..start + self.width].copy_from_slice(scores);
        self.next_position += 1;
        if self.next_position % self.ratio != 0 {
            return Ok(None);
        }
        let mut pooled = vec![0.0; self.width];
        for (column, output) in pooled.iter_mut().enumerate() {
            let maximum = (0..self.ratio)
                .map(|row| self.scores[row * self.width + column])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;
            for row in 0..self.ratio {
                let offset = row * self.width + column;
                let weight = f64::from((self.scores[offset] - maximum).exp());
                numerator += weight * f64::from(self.values[offset]);
                denominator += weight;
            }
            *output = (numerator / denominator) as f32;
        }
        self.values.fill(0.0);
        self.scores.fill(f32::NEG_INFINITY);
        Ok(Some(pooled))
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn pending_tokens(&self) -> usize {
        self.next_position % self.ratio
    }

    pub fn state_bytes(&self) -> usize {
        (self.values.len() + self.scores.len()) * size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_two_pools_each_feature_with_its_own_softmax() {
        let mut state = CompressorState::new(2, 2).unwrap();
        assert_eq!(
            state.push(0, &[1.0, 10.0], Some(&[0.0, 10.0])).unwrap(),
            None
        );
        let pooled = state
            .push(1, &[3.0, 20.0], Some(&[0.0, 0.0]))
            .unwrap()
            .unwrap();
        assert_eq!(pooled[0], 2.0);
        assert!((pooled[1] - 10.000454).abs() < 1e-5);
        assert_eq!(state.pending_tokens(), 0);
    }

    #[test]
    fn prefill_tail_is_carried_into_decode() {
        let mut state = CompressorState::new(4, 1).unwrap();
        for position in 0..3 {
            assert!(state
                .push(position, &[position as f32], Some(&[0.0]))
                .unwrap()
                .is_none());
        }
        assert_eq!(state.pending_tokens(), 3);
        assert_eq!(
            state.push(3, &[3.0], Some(&[0.0])).unwrap(),
            Some(vec![1.5])
        );
    }

    #[test]
    fn ratio_one_has_no_gate_or_partial_state() {
        let mut state = CompressorState::new(1, 2).unwrap();
        assert_eq!(
            state.push(0, &[2.0, 3.0], None).unwrap(),
            Some(vec![2.0, 3.0])
        );
        assert_eq!(state.pending_tokens(), 0);
        assert!(state.push(1, &[2.0, 3.0], Some(&[0.0, 0.0])).is_err());
    }
}
