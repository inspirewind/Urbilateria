//! Scalar reference math for the Kimi K3 text decoder.
//!
//! These routines intentionally favor explicit shapes, stable reductions, and useful errors over
//! kernel-level optimization. Runtime kernels can use them as correctness oracles while retaining
//! their own reusable scratch buffers.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiK3MathError {
    Shape {
        operation: &'static str,
        argument: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidValue {
        operation: &'static str,
        argument: &'static str,
        reason: &'static str,
    },
    NonFinite {
        operation: &'static str,
        argument: &'static str,
        index: usize,
    },
    Overflow {
        operation: &'static str,
        expression: &'static str,
    },
}

impl fmt::Display for KimiK3MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                operation,
                argument,
                expected,
                got,
            } => write!(
                f,
                "Kimi K3 {operation}: {argument} has length {got}, expected {expected}"
            ),
            Self::InvalidValue {
                operation,
                argument,
                reason,
            } => write!(f, "Kimi K3 {operation}: invalid {argument}: {reason}"),
            Self::NonFinite {
                operation,
                argument,
                index,
            } => write!(
                f,
                "Kimi K3 {operation}: {argument}[{index}] is NaN or infinity"
            ),
            Self::Overflow {
                operation,
                expression,
            } => write!(
                f,
                "Kimi K3 {operation}: integer overflow while computing {expression}"
            ),
        }
    }
}

impl std::error::Error for KimiK3MathError {}

/// Applies K3's bounded SiTU-GLU activation to separate gate and up branches.
///
/// For gate `g` and up value `u`, the activation is
/// `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`. In particular,
/// the sigmoid receives the original, uncapped gate.
pub fn situ_glu(
    gate: &[f32],
    up: &[f32],
    beta: f32,
    linear_beta: f32,
) -> Result<Vec<f32>, KimiK3MathError> {
    const OPERATION: &str = "SiTU-GLU";
    require_nonempty(OPERATION, "gate", gate.len())?;
    expect_len(OPERATION, "up", gate.len(), up.len())?;
    require_positive(OPERATION, "beta", beta)?;
    require_positive(OPERATION, "linear_beta", linear_beta)?;
    validate_finite(OPERATION, "gate", gate)?;
    validate_finite(OPERATION, "up", up)?;

    let beta = f64::from(beta);
    let linear_beta = f64::from(linear_beta);
    gate.iter()
        .zip(up)
        .enumerate()
        .map(|(index, (&gate, &up))| {
            let gate = f64::from(gate);
            let up = f64::from(up);
            let capped_gate = beta * (gate / beta).tanh();
            let capped_up = linear_beta * (up / linear_beta).tanh();
            checked_f32(
                OPERATION,
                "output",
                index,
                capped_gate * sigmoid(gate) * capped_up,
            )
        })
        .collect()
}

/// L2-normalizes one non-empty vector in place using `1 / sqrt(sum(x^2) + epsilon)`.
///
/// This is deliberately not RMS normalization: there is no division by the vector width.
pub fn l2_normalize_in_place(values: &mut [f32], epsilon: f32) -> Result<(), KimiK3MathError> {
    const OPERATION: &str = "L2 normalization";
    require_nonempty(OPERATION, "values", values.len())?;
    require_positive(OPERATION, "epsilon", epsilon)?;
    validate_finite(OPERATION, "values", values)?;

    let square_sum = values.iter().try_fold(0.0f64, |sum, &value| {
        let value = f64::from(value);
        let sum = sum + value * value;
        if sum.is_finite() {
            Ok(sum)
        } else {
            Err(KimiK3MathError::NonFinite {
                operation: OPERATION,
                argument: "sum of squares",
                index: 0,
            })
        }
    })?;
    let inverse_norm = (square_sum + f64::from(epsilon)).sqrt().recip();
    if !inverse_norm.is_finite() {
        return Err(KimiK3MathError::NonFinite {
            operation: OPERATION,
            argument: "inverse norm",
            index: 0,
        });
    }

    // Once the norm is finite, every result is bounded by one in magnitude. There can be no
    // late error after this point, so mutating the caller's slice is transactional with respect
    // to all validation failures above.
    for value in values {
        *value = (f64::from(*value) * inverse_norm) as f32;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct KdaDecay {
    /// Log-space forget gate, constrained to `[lower_bound, 0]`.
    pub gate: Vec<f32>,
    /// Multiplicative state decay, constrained to `(0, 1]`.
    pub alpha: Vec<f32>,
}

/// Computes KDA's per-channel decay while indexing `a_log` once per head.
///
/// `z` and `dt_bias` are flattened `[head_count, head_dim]` arrays. `a_log` must contain
/// exactly one value per head; callers loading a checkpoint with padded storage must slice it to
/// `head_count` before calling this function.
pub fn kda_decay(
    z: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    head_count: usize,
    head_dim: usize,
    lower_bound: f32,
) -> Result<KdaDecay, KimiK3MathError> {
    const OPERATION: &str = "KDA decay";
    require_nonzero(OPERATION, "head_count", head_count)?;
    require_nonzero(OPERATION, "head_dim", head_dim)?;
    let element_count = checked_product(OPERATION, "head_count * head_dim", head_count, head_dim)?;
    expect_len(OPERATION, "z", element_count, z.len())?;
    expect_len(OPERATION, "dt_bias", element_count, dt_bias.len())?;
    expect_len(OPERATION, "a_log", head_count, a_log.len())?;
    if !lower_bound.is_finite() || lower_bound >= 0.0 {
        return Err(KimiK3MathError::InvalidValue {
            operation: OPERATION,
            argument: "lower_bound",
            reason: "must be finite and strictly negative",
        });
    }
    validate_finite(OPERATION, "z", z)?;
    validate_finite(OPERATION, "dt_bias", dt_bias)?;
    validate_finite(OPERATION, "a_log", a_log)?;

    let mut gate = Vec::with_capacity(element_count);
    let mut alpha = Vec::with_capacity(element_count);
    let lower_bound = f64::from(lower_bound);
    for (head, &a_log) in a_log.iter().enumerate() {
        let rate = f64::from(a_log).exp();
        if !rate.is_finite() {
            return Err(KimiK3MathError::NonFinite {
                operation: OPERATION,
                argument: "exp(a_log)",
                index: head,
            });
        }
        let offset = head * head_dim;
        for channel in 0..head_dim {
            let index = offset + channel;
            let shifted = f64::from(z[index]) + f64::from(dt_bias[index]);
            let argument = rate * shifted;
            if !argument.is_finite() {
                return Err(KimiK3MathError::NonFinite {
                    operation: OPERATION,
                    argument: "decay argument",
                    index,
                });
            }
            let gate_value =
                checked_f32(OPERATION, "gate", index, lower_bound * sigmoid(argument))?;
            let alpha_value = checked_f32(OPERATION, "alpha", index, f64::from(gate_value).exp())?;
            gate.push(gate_value);
            alpha.push(alpha_value);
        }
    }
    Ok(KdaDecay { gate, alpha })
}

/// Advances one KDA head by one token and returns that head's output.
///
/// `state` is row-major `[key_dim, value_dim]`. The operation is ordered as follows:
/// decay each key row, read `state^T * key`, apply the rank-one delta update, then read the
/// already-updated state with `query / sqrt(key_dim)`. The query accepted by this API is
/// therefore unscaled. State is committed only after all calculations succeed.
#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent_step(
    state: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    alpha: &[f32],
    beta: f32,
    key_dim: usize,
    value_dim: usize,
) -> Result<Vec<f32>, KimiK3MathError> {
    const OPERATION: &str = "KDA recurrent step";
    require_nonzero(OPERATION, "key_dim", key_dim)?;
    require_nonzero(OPERATION, "value_dim", value_dim)?;
    let state_len = checked_product(OPERATION, "key_dim * value_dim", key_dim, value_dim)?;
    expect_len(OPERATION, "state", state_len, state.len())?;
    expect_len(OPERATION, "query", key_dim, query.len())?;
    expect_len(OPERATION, "key", key_dim, key.len())?;
    expect_len(OPERATION, "value", value_dim, value.len())?;
    expect_len(OPERATION, "alpha", key_dim, alpha.len())?;
    if !beta.is_finite() || !(0.0..=1.0).contains(&beta) {
        return Err(KimiK3MathError::InvalidValue {
            operation: OPERATION,
            argument: "beta",
            reason: "must be finite and in [0, 1]",
        });
    }
    validate_finite(OPERATION, "state", state)?;
    validate_finite(OPERATION, "query", query)?;
    validate_finite(OPERATION, "key", key)?;
    validate_finite(OPERATION, "value", value)?;
    validate_finite(OPERATION, "alpha", alpha)?;
    if alpha.iter().any(|&value| !(0.0..=1.0).contains(&value)) {
        return Err(KimiK3MathError::InvalidValue {
            operation: OPERATION,
            argument: "alpha",
            reason: "all entries must be in [0, 1]",
        });
    }

    // Work on a candidate copy so overflow cannot leave the persistent sequence state half
    // decayed or half updated.
    let mut next_state = Vec::with_capacity(state_len);
    for (row, (state_row, &row_alpha)) in state.chunks_exact(value_dim).zip(alpha).enumerate() {
        for (column, &state_value) in state_row.iter().enumerate() {
            let index = row * value_dim + column;
            next_state.push(checked_f32(
                OPERATION,
                "decayed state",
                index,
                f64::from(state_value) * f64::from(row_alpha),
            )?);
        }
    }

    let mut prediction = vec![0.0f64; value_dim];
    for (row, &key_value) in key.iter().enumerate() {
        let key_value = f64::from(key_value);
        let state_row = &next_state[row * value_dim..(row + 1) * value_dim];
        for column in 0..value_dim {
            prediction[column] += key_value * f64::from(state_row[column]);
            if !prediction[column].is_finite() {
                return Err(KimiK3MathError::NonFinite {
                    operation: OPERATION,
                    argument: "prediction",
                    index: column,
                });
            }
        }
    }

    let beta = f64::from(beta);
    for (row, &key_value) in key.iter().enumerate() {
        let key_value = f64::from(key_value);
        for column in 0..value_dim {
            let index = row * value_dim + column;
            let correction = beta * (f64::from(value[column]) - prediction[column]);
            next_state[index] = checked_f32(
                OPERATION,
                "updated state",
                index,
                f64::from(next_state[index]) + key_value * correction,
            )?;
        }
    }

    let query_scale = (key_dim as f64).sqrt().recip();
    if !query_scale.is_finite() {
        return Err(KimiK3MathError::NonFinite {
            operation: OPERATION,
            argument: "query scale",
            index: 0,
        });
    }
    let mut output = vec![0.0f64; value_dim];
    for (row, &query_value) in query.iter().enumerate() {
        let scaled_query = f64::from(query_value) * query_scale;
        for column in 0..value_dim {
            output[column] += scaled_query * f64::from(next_state[row * value_dim + column]);
            if !output[column].is_finite() {
                return Err(KimiK3MathError::NonFinite {
                    operation: OPERATION,
                    argument: "output",
                    index: column,
                });
            }
        }
    }
    let output = output
        .into_iter()
        .enumerate()
        .map(|(index, value)| checked_f32(OPERATION, "output", index, value))
        .collect::<Result<Vec<_>, _>>()?;

    state.copy_from_slice(&next_state);
    Ok(output)
}

/// Pre-folds the element-wise AttnRes norm and projection vectors for reuse by every token.
pub fn fold_attn_res_weights(
    norm_weight: &[f32],
    projection_weight: &[f32],
) -> Result<Vec<f32>, KimiK3MathError> {
    const OPERATION: &str = "AttnRes weight folding";
    require_nonempty(OPERATION, "norm_weight", norm_weight.len())?;
    expect_len(
        OPERATION,
        "projection_weight",
        norm_weight.len(),
        projection_weight.len(),
    )?;
    validate_finite(OPERATION, "norm_weight", norm_weight)?;
    validate_finite(OPERATION, "projection_weight", projection_weight)?;
    norm_weight
        .iter()
        .zip(projection_weight)
        .enumerate()
        .map(|(index, (&norm, &projection))| {
            checked_f32(
                OPERATION,
                "folded_weight",
                index,
                f64::from(norm) * f64::from(projection),
            )
        })
        .collect()
}

/// Aggregates a source stack using K3's attention residual.
///
/// `sources` is source-major `[source_count, hidden_size]`. Scores use RMS-normalized
/// sources as keys and the pre-folded `norm_weight * projection_weight` vector. The values
/// mixed by the resulting softmax probabilities are the raw, unnormalized sources.
pub fn attn_res(
    sources: &[f32],
    source_count: usize,
    hidden_size: usize,
    folded_weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, KimiK3MathError> {
    const OPERATION: &str = "AttnRes";
    require_nonzero(OPERATION, "source_count", source_count)?;
    require_nonzero(OPERATION, "hidden_size", hidden_size)?;
    let source_len = checked_product(
        OPERATION,
        "source_count * hidden_size",
        source_count,
        hidden_size,
    )?;
    expect_len(OPERATION, "sources", source_len, sources.len())?;
    expect_len(OPERATION, "folded_weight", hidden_size, folded_weight.len())?;
    require_positive(OPERATION, "epsilon", epsilon)?;
    validate_finite(OPERATION, "sources", sources)?;
    validate_finite(OPERATION, "folded_weight", folded_weight)?;

    let mut scores = Vec::with_capacity(source_count);
    for source_index in 0..source_count {
        let source = &sources[source_index * hidden_size..(source_index + 1) * hidden_size];
        let square_sum = source.iter().try_fold(0.0f64, |sum, &value| {
            let value = f64::from(value);
            let next = sum + value * value;
            if next.is_finite() {
                Ok(next)
            } else {
                Err(KimiK3MathError::NonFinite {
                    operation: OPERATION,
                    argument: "source sum of squares",
                    index: source_index,
                })
            }
        })?;
        let mean_square = square_sum / hidden_size as f64;
        let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip();
        if !inverse_rms.is_finite() {
            return Err(KimiK3MathError::NonFinite {
                operation: OPERATION,
                argument: "inverse RMS",
                index: source_index,
            });
        }
        let score = source
            .iter()
            .zip(folded_weight)
            .map(|(&value, &folded)| f64::from(value) * inverse_rms * f64::from(folded))
            .sum::<f64>();
        if !score.is_finite() {
            return Err(KimiK3MathError::NonFinite {
                operation: OPERATION,
                argument: "score",
                index: source_index,
            });
        }
        scores.push(score);
    }

    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut denominator = 0.0f64;
    for score in &mut scores {
        *score = (*score - maximum).exp();
        denominator += *score;
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(KimiK3MathError::NonFinite {
            operation: OPERATION,
            argument: "softmax denominator",
            index: 0,
        });
    }

    let mut output = vec![0.0f64; hidden_size];
    for source_index in 0..source_count {
        let probability = scores[source_index] / denominator;
        let source = &sources[source_index * hidden_size..(source_index + 1) * hidden_size];
        for (destination, &value) in output.iter_mut().zip(source) {
            *destination += probability * f64::from(value);
        }
    }
    output
        .into_iter()
        .enumerate()
        .map(|(index, value)| checked_f32(OPERATION, "output", index, value))
        .collect()
}

fn checked_product(
    operation: &'static str,
    expression: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, KimiK3MathError> {
    left.checked_mul(right).ok_or(KimiK3MathError::Overflow {
        operation,
        expression,
    })
}

fn expect_len(
    operation: &'static str,
    argument: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), KimiK3MathError> {
    if expected == got {
        Ok(())
    } else {
        Err(KimiK3MathError::Shape {
            operation,
            argument,
            expected,
            got,
        })
    }
}

fn require_nonempty(
    operation: &'static str,
    argument: &'static str,
    length: usize,
) -> Result<(), KimiK3MathError> {
    if length > 0 {
        Ok(())
    } else {
        Err(KimiK3MathError::InvalidValue {
            operation,
            argument,
            reason: "must not be empty",
        })
    }
}

fn require_nonzero(
    operation: &'static str,
    argument: &'static str,
    value: usize,
) -> Result<(), KimiK3MathError> {
    if value > 0 {
        Ok(())
    } else {
        Err(KimiK3MathError::InvalidValue {
            operation,
            argument,
            reason: "must be greater than zero",
        })
    }
}

fn require_positive(
    operation: &'static str,
    argument: &'static str,
    value: f32,
) -> Result<(), KimiK3MathError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KimiK3MathError::InvalidValue {
            operation,
            argument,
            reason: "must be finite and strictly positive",
        })
    }
}

fn validate_finite(
    operation: &'static str,
    argument: &'static str,
    values: &[f32],
) -> Result<(), KimiK3MathError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        Err(KimiK3MathError::NonFinite {
            operation,
            argument,
            index,
        })
    } else {
        Ok(())
    }
}

fn checked_f32(
    operation: &'static str,
    argument: &'static str,
    index: usize,
    value: f64,
) -> Result<f32, KimiK3MathError> {
    let narrowed = value as f32;
    if value.is_finite() && narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(KimiK3MathError::NonFinite {
            operation,
            argument,
            index,
        })
    }
}

#[inline]
fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }

    #[test]
    fn situ_glu_uses_the_uncapped_gate_for_sigmoid_and_stays_bounded() {
        let output = situ_glu(&[-1_000.0, 4.0], &[1_000.0, 25.0], 4.0, 25.0).unwrap();
        assert_eq!(output[0], -0.0);

        let expected = 4.0f32
            * (4.0f32 / 4.0).tanh()
            * (1.0 / (1.0 + (-4.0f32).exp()))
            * 25.0
            * (25.0f32 / 25.0).tanh();
        assert_close(output[1], expected, 1e-5);
        assert!(output.iter().all(|value| value.abs() <= 100.0));
    }

    #[test]
    fn l2_normalization_uses_sum_of_squares_not_mean() {
        let mut values = [3.0, 4.0];
        l2_normalize_in_place(&mut values, 1e-6).unwrap();
        let inverse = (25.0f32 + 1e-6).sqrt().recip();
        assert_close(values[0], 3.0 * inverse, 1e-6);
        assert_close(values[1], 4.0 * inverse, 1e-6);
    }

    #[test]
    fn kda_decay_uses_one_a_log_value_per_head() {
        let decay = kda_decay(
            &[0.0, 1.0, 0.0, 1.0],
            &[0.0; 4],
            &[0.0, 2.0f32.ln()],
            2,
            2,
            -5.0,
        )
        .unwrap();

        assert_close(decay.gate[0], -2.5, 1e-6);
        assert_close(decay.gate[1], -5.0 / (1.0 + (-1.0f32).exp()), 1e-6);
        assert_close(decay.gate[2], -2.5, 1e-6);
        assert_close(decay.gate[3], -5.0 / (1.0 + (-2.0f32).exp()), 1e-6);
        for (&gate, &alpha) in decay.gate.iter().zip(&decay.alpha) {
            assert_close(alpha, gate.exp(), 1e-6);
        }
    }

    #[test]
    fn kda_step_reads_after_decay_writes_before_output() {
        let mut state = [2.0];
        let output =
            kda_recurrent_step(&mut state, &[2.0], &[0.5], &[3.0], &[0.5], 0.5, 1, 1).unwrap();

        // decay: 2 -> 1; prediction: .5; write: 1 + .5*.5*(3-.5) = 1.625;
        // output uses that updated state and q=2.
        assert_close(state[0], 1.625, 1e-6);
        assert_close(output[0], 3.25, 1e-6);
    }

    #[test]
    fn kda_step_applies_inverse_sqrt_key_dimension_to_query() {
        let mut state = [0.0; 4];
        let output = kda_recurrent_step(
            &mut state,
            &[1.0; 4],
            &[1.0, 0.0, 0.0, 0.0],
            &[2.0],
            &[1.0; 4],
            0.5,
            4,
            1,
        )
        .unwrap();

        assert_eq!(state, [1.0, 0.0, 0.0, 0.0]);
        assert_close(output[0], 0.5, 1e-6);
    }

    #[test]
    fn failed_kda_step_does_not_partially_modify_state() {
        let mut state = [1.0];
        let original = state;
        let error = kda_recurrent_step(&mut state, &[f32::NAN], &[1.0], &[1.0], &[0.5], 0.5, 1, 1)
            .unwrap_err();
        assert!(matches!(error, KimiK3MathError::NonFinite { .. }));
        assert_eq!(state, original);
    }

    #[test]
    fn attn_res_scores_normalized_keys_but_mixes_raw_values() {
        let epsilon = 1e-6;
        let sources = [3.0, 4.0, 0.0, 2.0];
        let output = attn_res(&sources, 2, 2, &[1.0, 0.0], epsilon).unwrap();

        let first_score = 3.0 / (12.5f32 + epsilon).sqrt();
        let first_probability = first_score.exp() / (first_score.exp() + 1.0);
        assert_close(output[0], first_probability * 3.0, 1e-6);
        assert_close(
            output[1],
            first_probability * 4.0 + (1.0 - first_probability) * 2.0,
            1e-6,
        );
        // A normalized-value mix would be close to one, not the raw scale retained here.
        assert!(output[1] > 2.5);
    }

    #[test]
    fn attn_res_weights_are_elementwise_folded() {
        let folded = fold_attn_res_weights(&[2.0, -3.0], &[4.0, 5.0]).unwrap();
        assert_eq!(folded, [8.0, -15.0]);
    }

    #[test]
    fn shape_overflow_is_reported_before_slice_validation() {
        let error = attn_res(&[], usize::MAX, 2, &[], 1e-6).unwrap_err();
        assert!(matches!(error, KimiK3MathError::Overflow { .. }));
    }

    #[test]
    fn invalid_shapes_and_non_finite_values_are_rejected() {
        assert!(matches!(
            situ_glu(&[1.0], &[], 4.0, 25.0),
            Err(KimiK3MathError::Shape { .. })
        ));
        assert!(matches!(
            kda_decay(&[0.0; 4], &[0.0; 4], &[0.0; 4], 2, 2, -5.0),
            Err(KimiK3MathError::Shape {
                argument: "a_log",
                ..
            })
        ));
        assert!(matches!(
            fold_attn_res_weights(&[f32::INFINITY], &[1.0]),
            Err(KimiK3MathError::NonFinite { .. })
        ));
    }
}
