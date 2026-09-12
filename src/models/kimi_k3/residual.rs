//! Transactional cross-layer AttnRes state for one Kimi K3 token.
//!
//! K3 does not use a conventional residual addition around each sublayer. Every block boundary
//! snapshots the incoming hidden state, and later AttnRes calls mix those raw snapshots with the
//! current running prefix. Keeping the pending boundary snapshot in the continuation objects makes
//! a failed attention or MLP leave the persistent stack unchanged.

use super::math::{attn_res, KimiK3MathError};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnResidualError {
    InvalidGeometry(&'static str),
    LayerOrder {
        expected: usize,
        got: usize,
    },
    Shape {
        value: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite {
        value: &'static str,
        index: usize,
    },
    Overflow(&'static str),
    Allocation {
        value: &'static str,
        elements: usize,
    },
    Math(KimiK3MathError),
}

impl fmt::Display for AttnResidualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(reason) => write!(f, "invalid Kimi K3 AttnRes state: {reason}"),
            Self::LayerOrder { expected, got } => {
                write!(f, "Kimi K3 AttnRes expected layer {expected}, got {got}")
            }
            Self::Shape {
                value,
                expected,
                got,
            } => write!(
                f,
                "Kimi K3 AttnRes {value} has length {got}, expected {expected}"
            ),
            Self::NonFinite { value, index } => {
                write!(f, "Kimi K3 AttnRes {value}[{index}] is NaN or infinity")
            }
            Self::Overflow(expression) => {
                write!(
                    f,
                    "integer overflow while computing Kimi K3 AttnRes {expression}"
                )
            }
            Self::Allocation { value, elements } => write!(
                f,
                "cannot allocate {elements} elements for Kimi K3 AttnRes {value}"
            ),
            Self::Math(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for AttnResidualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KimiK3MathError> for AttnResidualError {
    fn from(value: KimiK3MathError) -> Self {
        Self::Math(value)
    }
}

/// Persistent block snapshots while one token traverses the decoder stack.
#[derive(Debug, Clone, PartialEq)]
pub struct AttnResidualState {
    hidden_size: usize,
    block_size: usize,
    next_layer: usize,
    snapshots: Vec<f32>,
}

/// Opaque state between a layer's attention input and attention output.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionContinuation {
    layer: usize,
    prefix_sum: Option<Vec<f32>>,
    pending_snapshot: Option<Vec<f32>>,
}

/// Opaque state between a layer's MLP input and MLP output.
#[derive(Debug, Clone, PartialEq)]
pub struct MlpContinuation {
    layer: usize,
    prefix_sum: Option<Vec<f32>>,
    pending_snapshot: Option<Vec<f32>>,
}

impl AttnResidualState {
    pub fn new(hidden_size: usize, block_size: usize) -> Result<Self, AttnResidualError> {
        if hidden_size == 0 {
            return Err(AttnResidualError::InvalidGeometry(
                "hidden_size must be non-zero",
            ));
        }
        if block_size == 0 {
            return Err(AttnResidualError::InvalidGeometry(
                "block_size must be non-zero",
            ));
        }
        Ok(Self {
            hidden_size,
            block_size,
            next_layer: 0,
            snapshots: Vec::new(),
        })
    }

    pub fn next_layer(&self) -> usize {
        self.next_layer
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len() / self.hidden_size
    }

    pub fn snapshots(&self) -> &[f32] {
        &self.snapshots
    }

    /// Computes the AttnRes-aggregated attention input and records, but does not yet commit, a
    /// block boundary snapshot. The decoder runtime must apply `input_layernorm` to this result
    /// before invoking KDA or MLA.
    pub fn begin_layer(
        &self,
        layer: usize,
        hidden: &[f32],
        folded_attention_weight: &[f32],
        epsilon: f32,
    ) -> Result<(Vec<f32>, AttentionContinuation), AttnResidualError> {
        self.expect_layer(layer)?;
        self.validate_hidden("hidden", hidden)?;
        self.validate_folded(folded_attention_weight, epsilon)?;

        let attention_input = if self.snapshots.is_empty() {
            hidden.to_vec()
        } else {
            let sources = self.sources_with(hidden, None)?;
            attn_res(
                &sources,
                self.snapshot_count() + 1,
                self.hidden_size,
                folded_attention_weight,
                epsilon,
            )?
        };
        let boundary = layer % self.block_size == 0;
        Ok((
            attention_input,
            AttentionContinuation {
                layer,
                prefix_sum: (!boundary).then(|| hidden.to_vec()),
                pending_snapshot: boundary.then(|| hidden.to_vec()),
            },
        ))
    }

    /// Adds attention to the running prefix, then computes the unconditional pre-MLP AttnRes.
    /// The decoder runtime must apply `post_attention_layernorm` to the result before the MLP.
    pub fn after_attention(
        &self,
        continuation: AttentionContinuation,
        attention_output: &[f32],
        folded_mlp_weight: &[f32],
        epsilon: f32,
    ) -> Result<(Vec<f32>, MlpContinuation), AttnResidualError> {
        self.expect_layer(continuation.layer)?;
        self.validate_hidden("attention output", attention_output)?;
        self.validate_folded(folded_mlp_weight, epsilon)?;
        self.validate_attention_continuation(
            continuation.prefix_sum.as_deref(),
            continuation.pending_snapshot.as_deref(),
        )?;

        let prefix_sum = match continuation.prefix_sum {
            Some(mut prefix) => {
                add_in_place(&mut prefix, attention_output, "attention residual")?;
                Some(prefix)
            }
            None => Some(attention_output.to_vec()),
        };
        let prefix = prefix_sum
            .as_deref()
            .expect("attention always establishes a running prefix");
        let sources = self.sources_with(prefix, continuation.pending_snapshot.as_deref())?;
        let source_count = self
            .snapshot_count()
            .checked_add(usize::from(continuation.pending_snapshot.is_some()))
            .and_then(|count| count.checked_add(1))
            .ok_or(AttnResidualError::Overflow("source count"))?;
        let mlp_input = attn_res(
            &sources,
            source_count,
            self.hidden_size,
            folded_mlp_weight,
            epsilon,
        )?;
        Ok((
            mlp_input,
            MlpContinuation {
                layer: continuation.layer,
                prefix_sum,
                pending_snapshot: continuation.pending_snapshot,
            },
        ))
    }

    /// Adds the MLP output and atomically commits this layer's boundary snapshot and layer index.
    pub fn finish_layer(
        &mut self,
        continuation: MlpContinuation,
        mlp_output: &[f32],
    ) -> Result<Vec<f32>, AttnResidualError> {
        self.expect_layer(continuation.layer)?;
        self.validate_hidden("MLP output", mlp_output)?;
        let prefix =
            continuation
                .prefix_sum
                .as_deref()
                .ok_or(AttnResidualError::InvalidGeometry(
                    "an MLP continuation must contain the running prefix",
                ))?;
        self.validate_hidden("continuation prefix", prefix)?;
        if let Some(snapshot) = continuation.pending_snapshot.as_deref() {
            self.validate_hidden("pending snapshot", snapshot)?;
        }
        let mut hidden = continuation
            .prefix_sum
            .expect("attention always establishes a running prefix");
        add_in_place(&mut hidden, mlp_output, "MLP residual")?;
        let next_layer = self
            .next_layer
            .checked_add(1)
            .ok_or(AttnResidualError::Overflow("next layer"))?;

        if let Some(snapshot) = continuation.pending_snapshot {
            let total = self
                .snapshots
                .len()
                .checked_add(snapshot.len())
                .ok_or(AttnResidualError::Overflow("snapshot stack length"))?;
            self.snapshots.try_reserve(snapshot.len()).map_err(|_| {
                AttnResidualError::Allocation {
                    value: "snapshot stack",
                    elements: total,
                }
            })?;
            self.snapshots.extend(snapshot);
        }
        self.next_layer = next_layer;
        Ok(hidden)
    }

    /// Applies the model-level AttnRes over all block snapshots and the final running prefix.
    pub fn finalize(
        &self,
        hidden: &[f32],
        folded_output_weight: &[f32],
        epsilon: f32,
    ) -> Result<Vec<f32>, AttnResidualError> {
        if self.next_layer == 0 || self.snapshots.is_empty() {
            return Err(AttnResidualError::InvalidGeometry(
                "finalization requires at least one completed boundary layer",
            ));
        }
        self.validate_hidden("final hidden", hidden)?;
        self.validate_folded(folded_output_weight, epsilon)?;
        let sources = self.sources_with(hidden, None)?;
        Ok(attn_res(
            &sources,
            self.snapshot_count() + 1,
            self.hidden_size,
            folded_output_weight,
            epsilon,
        )?)
    }

    fn expect_layer(&self, layer: usize) -> Result<(), AttnResidualError> {
        if layer != self.next_layer {
            return Err(AttnResidualError::LayerOrder {
                expected: self.next_layer,
                got: layer,
            });
        }
        Ok(())
    }

    fn validate_hidden(
        &self,
        value: &'static str,
        hidden: &[f32],
    ) -> Result<(), AttnResidualError> {
        if hidden.len() != self.hidden_size {
            return Err(AttnResidualError::Shape {
                value,
                expected: self.hidden_size,
                got: hidden.len(),
            });
        }
        if let Some(index) = hidden.iter().position(|item| !item.is_finite()) {
            return Err(AttnResidualError::NonFinite { value, index });
        }
        Ok(())
    }

    fn validate_folded(&self, folded: &[f32], epsilon: f32) -> Result<(), AttnResidualError> {
        self.validate_hidden("folded weight", folded)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(AttnResidualError::InvalidGeometry(
                "epsilon must be finite and strictly positive",
            ));
        }
        Ok(())
    }

    fn validate_attention_continuation(
        &self,
        prefix_sum: Option<&[f32]>,
        pending_snapshot: Option<&[f32]>,
    ) -> Result<(), AttnResidualError> {
        if prefix_sum.is_some() == pending_snapshot.is_some() {
            return Err(AttnResidualError::InvalidGeometry(
                "a layer continuation must contain exactly one prefix or boundary snapshot",
            ));
        }
        if let Some(prefix) = prefix_sum {
            self.validate_hidden("continuation prefix", prefix)?;
        }
        if let Some(snapshot) = pending_snapshot {
            self.validate_hidden("pending snapshot", snapshot)?;
        }
        Ok(())
    }

    fn sources_with(
        &self,
        tail: &[f32],
        pending_snapshot: Option<&[f32]>,
    ) -> Result<Vec<f32>, AttnResidualError> {
        let pending = pending_snapshot.map_or(0, <[f32]>::len);
        let elements = self
            .snapshots
            .len()
            .checked_add(pending)
            .and_then(|value| value.checked_add(tail.len()))
            .ok_or(AttnResidualError::Overflow("source stack length"))?;
        let mut sources = Vec::new();
        sources
            .try_reserve_exact(elements)
            .map_err(|_| AttnResidualError::Allocation {
                value: "source stack",
                elements,
            })?;
        sources.extend_from_slice(&self.snapshots);
        if let Some(snapshot) = pending_snapshot {
            sources.extend_from_slice(snapshot);
        }
        sources.extend_from_slice(tail);
        Ok(sources)
    }
}

fn add_in_place(
    left: &mut [f32],
    right: &[f32],
    value: &'static str,
) -> Result<(), AttnResidualError> {
    for (index, (left, right)) in left.iter_mut().zip(right).enumerate() {
        let sum = f64::from(*left) + f64::from(*right);
        if !sum.is_finite() || sum < f64::from(f32::MIN) || sum > f64::from(f32::MAX) {
            return Err(AttnResidualError::NonFinite { value, index });
        }
        *left = sum as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_layer(
        state: &mut AttnResidualState,
        layer: usize,
        hidden: f32,
        attention: f32,
        mlp: f32,
    ) -> (f32, f32, f32) {
        let fold = [0.0];
        let (attention_input, continuation) =
            state.begin_layer(layer, &[hidden], &fold, 1.0e-5).unwrap();
        let (mlp_input, continuation) = state
            .after_attention(continuation, &[attention], &fold, 1.0e-5)
            .unwrap();
        let output = state.finish_layer(continuation, &[mlp]).unwrap();
        (attention_input[0], mlp_input[0], output[0])
    }

    #[test]
    fn boundary_snapshot_order_matches_the_decoder_contract() {
        let mut state = AttnResidualState::new(1, 2).unwrap();
        assert_eq!(run_layer(&mut state, 0, 10.0, 2.0, 3.0), (10.0, 6.0, 5.0));
        assert_eq!(state.snapshots(), &[10.0]);
        assert_eq!(run_layer(&mut state, 1, 5.0, 1.0, 2.0), (7.5, 8.0, 8.0));
        let layer_two = run_layer(&mut state, 2, 8.0, 4.0, 1.0);
        assert_eq!(layer_two.0, 9.0);
        assert!((layer_two.1 - 22.0 / 3.0).abs() < 1.0e-6);
        assert_eq!(layer_two.2, 5.0);
        assert_eq!(state.snapshots(), &[10.0, 8.0]);
        let final_hidden = state.finalize(&[5.0], &[0.0], 1.0e-5).unwrap();
        assert!((final_hidden[0] - 23.0 / 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn failures_do_not_commit_pending_boundary_or_layer_index() {
        let mut state = AttnResidualState::new(2, 12).unwrap();
        let (_, continuation) = state
            .begin_layer(0, &[1.0, 2.0], &[0.0, 0.0], 1.0e-5)
            .unwrap();
        assert!(state
            .after_attention(continuation.clone(), &[f32::NAN, 0.0], &[0.0, 0.0], 1.0e-5)
            .is_err());
        assert_eq!(state.next_layer(), 0);
        assert_eq!(state.snapshot_count(), 0);

        let (_, mlp_continuation) = state
            .after_attention(continuation, &[3.0, 4.0], &[0.0, 0.0], 1.0e-5)
            .unwrap();
        assert!(state
            .finish_layer(mlp_continuation, &[f32::INFINITY, 0.0])
            .is_err());
        assert_eq!(state.next_layer(), 0);
        assert_eq!(state.snapshot_count(), 0);
    }

    #[test]
    fn invalid_geometry_shapes_and_layer_order_are_rejected() {
        assert!(AttnResidualState::new(0, 12).is_err());
        assert!(AttnResidualState::new(2, 0).is_err());
        let state = AttnResidualState::new(2, 12).unwrap();
        assert!(matches!(
            state.begin_layer(1, &[0.0, 0.0], &[0.0, 0.0], 1.0e-5),
            Err(AttnResidualError::LayerOrder { .. })
        ));
        assert!(state.begin_layer(0, &[0.0], &[0.0, 0.0], 1.0e-5).is_err());
        assert!(state.begin_layer(0, &[0.0, 0.0], &[0.0], 1.0e-5).is_err());
        assert!(state.finalize(&[0.0, 0.0], &[0.0, 0.0], 1.0e-5).is_err());
    }
}
