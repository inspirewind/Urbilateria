use super::math::{linear, round_to_bf16, round_to_bf16_in_place, uses_bf16_output};
use super::norm::{gated_rms_norm, gated_rms_norm_bf16, NormError};
use crate::model::{DenseMatrix, WeightError, WeightMatrix};
use std::fmt;

const QK_L2_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaNetGeometry {
    pub hidden_size: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub norm_eps: f32,
}

impl DeltaNetGeometry {
    pub fn validate(self) -> Result<(), DeltaNetError> {
        if self.hidden_size == 0
            || self.num_key_heads == 0
            || self.num_value_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.conv_kernel_size == 0
        {
            return Err(DeltaNetError::InvalidGeometry(
                "all Gated DeltaNet dimensions must be non-zero".to_owned(),
            ));
        }
        if self.num_value_heads % self.num_key_heads != 0 {
            return Err(DeltaNetError::InvalidGeometry(format!(
                "value heads {} must be divisible by key heads {}",
                self.num_value_heads, self.num_key_heads
            )));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(DeltaNetError::InvalidGeometry(
                "RMSNorm epsilon must be finite and positive".to_owned(),
            ));
        }
        self.key_width()?;
        self.value_width()?;
        self.conv_width()?;
        checked_product(
            "convolution state",
            &[self.conv_width()?, self.conv_kernel_size],
        )?;
        checked_product(
            "recurrent state",
            &[self.num_value_heads, self.key_head_dim, self.value_head_dim],
        )?;
        Ok(())
    }

    pub fn key_width(self) -> Result<usize, DeltaNetError> {
        checked_product("key width", &[self.num_key_heads, self.key_head_dim])
    }

    pub fn value_width(self) -> Result<usize, DeltaNetError> {
        checked_product("value width", &[self.num_value_heads, self.value_head_dim])
    }

    pub fn conv_width(self) -> Result<usize, DeltaNetError> {
        self.key_width()?
            .checked_mul(2)
            .and_then(|two_keys| two_keys.checked_add(self.value_width().ok()?))
            .ok_or_else(|| {
                DeltaNetError::InvalidGeometry("QKV convolution width overflows usize".to_owned())
            })
    }
}

#[derive(Debug)]
pub enum DeltaNetError {
    Weight(WeightError),
    Norm(NormError),
    InvalidGeometry(String),
    InvalidShape(String),
    NonFinite(&'static str),
    Allocation(String),
    StateGeometry {
        expected: DeltaNetGeometry,
        got: DeltaNetGeometry,
    },
    StateLengthOverflow,
}

impl fmt::Display for DeltaNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight(error) => error.fmt(f),
            Self::Norm(error) => error.fmt(f),
            Self::InvalidGeometry(reason) => {
                write!(f, "invalid Qwen Gated DeltaNet geometry: {reason}")
            }
            Self::InvalidShape(reason) => {
                write!(f, "invalid Qwen Gated DeltaNet shape: {reason}")
            }
            Self::NonFinite(name) => {
                write!(f, "Qwen Gated DeltaNet {name} contains NaN or infinity")
            }
            Self::Allocation(reason) => {
                write!(f, "cannot allocate Qwen Gated DeltaNet state: {reason}")
            }
            Self::StateGeometry { expected, got } => {
                write!(
                    f,
                    "DeltaNet state geometry {got:?} does not match {expected:?}"
                )
            }
            Self::StateLengthOverflow => f.write_str("DeltaNet token count overflows usize"),
        }
    }
}

impl std::error::Error for DeltaNetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight(error) => Some(error),
            Self::Norm(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WeightError> for DeltaNetError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

impl From<NormError> for DeltaNetError {
    fn from(value: NormError) -> Self {
        Self::Norm(value)
    }
}

/// The convolution window and FP32 recurrent matrix for one DeltaNet layer and batch item.
#[derive(Debug, Clone)]
pub struct DeltaNetState {
    geometry: DeltaNetGeometry,
    conv: Vec<f32>,
    recurrent: Vec<f32>,
    tokens: usize,
}

impl DeltaNetState {
    pub fn new(geometry: DeltaNetGeometry) -> Result<Self, DeltaNetError> {
        geometry.validate()?;
        let conv_len = checked_product(
            "convolution state",
            &[geometry.conv_width()?, geometry.conv_kernel_size],
        )?;
        let recurrent_len = checked_product(
            "recurrent state",
            &[
                geometry.num_value_heads,
                geometry.key_head_dim,
                geometry.value_head_dim,
            ],
        )?;
        let mut conv = Vec::new();
        conv.try_reserve_exact(conv_len).map_err(|error| {
            DeltaNetError::Allocation(format!("{conv_len} convolution values: {error}"))
        })?;
        conv.resize(conv_len, 0.0);
        let mut recurrent = Vec::new();
        recurrent
            .try_reserve_exact(recurrent_len)
            .map_err(|error| {
                DeltaNetError::Allocation(format!("{recurrent_len} recurrent values: {error}"))
            })?;
        recurrent.resize(recurrent_len, 0.0);
        Ok(Self {
            geometry,
            conv,
            recurrent,
            tokens: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    pub fn stored_f32_elements(&self) -> usize {
        self.conv.len().saturating_add(self.recurrent.len())
    }

    pub fn conv_state(&self) -> &[f32] {
        &self.conv
    }

    pub fn recurrent_state(&self) -> &[f32] {
        &self.recurrent
    }

    pub fn clear(&mut self) {
        self.conv.fill(0.0);
        self.recurrent.fill(0.0);
        self.tokens = 0;
    }
}

/// Scalar, one-token Qwen3.8 Gated DeltaNet reference.
#[derive(Debug, Clone)]
pub struct GatedDeltaNet {
    geometry: DeltaNetGeometry,
    in_proj_qkv: WeightMatrix,
    in_proj_z: WeightMatrix,
    in_proj_b: WeightMatrix,
    in_proj_a: WeightMatrix,
    conv_weight: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    norm_weight: Vec<f32>,
    out_proj: WeightMatrix,
    bf16_activations: bool,
}

impl GatedDeltaNet {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        geometry: DeltaNetGeometry,
        in_proj_qkv: DenseMatrix,
        in_proj_z: DenseMatrix,
        in_proj_b: DenseMatrix,
        in_proj_a: DenseMatrix,
        conv_weight: Vec<f32>,
        dt_bias: Vec<f32>,
        a_log: Vec<f32>,
        norm_weight: Vec<f32>,
        out_proj: DenseMatrix,
    ) -> Result<Self, DeltaNetError> {
        Self::new_mixed(
            geometry,
            in_proj_qkv.into(),
            in_proj_z.into(),
            in_proj_b.into(),
            in_proj_a.into(),
            conv_weight,
            dt_bias,
            a_log,
            norm_weight,
            out_proj.into(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_mixed(
        geometry: DeltaNetGeometry,
        in_proj_qkv: WeightMatrix,
        in_proj_z: WeightMatrix,
        in_proj_b: WeightMatrix,
        in_proj_a: WeightMatrix,
        conv_weight: Vec<f32>,
        dt_bias: Vec<f32>,
        a_log: Vec<f32>,
        norm_weight: Vec<f32>,
        out_proj: WeightMatrix,
    ) -> Result<Self, DeltaNetError> {
        geometry.validate()?;
        let conv_width = geometry.conv_width()?;
        let value_width = geometry.value_width()?;
        expect_matrix(
            "in_proj_qkv",
            &in_proj_qkv,
            conv_width,
            geometry.hidden_size,
        )?;
        expect_matrix("in_proj_z", &in_proj_z, value_width, geometry.hidden_size)?;
        expect_matrix(
            "in_proj_b",
            &in_proj_b,
            geometry.num_value_heads,
            geometry.hidden_size,
        )?;
        expect_matrix(
            "in_proj_a",
            &in_proj_a,
            geometry.num_value_heads,
            geometry.hidden_size,
        )?;
        expect_matrix("out_proj", &out_proj, geometry.hidden_size, value_width)?;
        expect_vector(
            "conv_weight",
            &conv_weight,
            checked_product(
                "convolution weights",
                &[conv_width, geometry.conv_kernel_size],
            )?,
        )?;
        expect_vector("dt_bias", &dt_bias, geometry.num_value_heads)?;
        expect_vector("A_log", &a_log, geometry.num_value_heads)?;
        expect_vector("norm_weight", &norm_weight, geometry.value_head_dim)?;
        if a_log.iter().any(|value| !value.exp().is_finite()) {
            return Err(DeltaNetError::NonFinite("exp(A_log)"));
        }
        let bf16_activations = uses_bf16_output(&in_proj_qkv);
        if [
            uses_bf16_output(&in_proj_z),
            uses_bf16_output(&in_proj_b),
            uses_bf16_output(&in_proj_a),
            uses_bf16_output(&out_proj),
        ]
        .into_iter()
        .any(|value| value != bf16_activations)
        {
            return Err(DeltaNetError::InvalidShape(
                "DeltaNet projections must share one activation dtype".to_owned(),
            ));
        }
        Ok(Self {
            geometry,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv_weight,
            dt_bias,
            a_log,
            norm_weight,
            out_proj,
            bf16_activations,
        })
    }

    pub fn geometry(&self) -> DeltaNetGeometry {
        self.geometry
    }

    pub(crate) fn uses_bf16_activations(&self) -> bool {
        self.bf16_activations
    }

    pub fn new_state(&self) -> Result<DeltaNetState, DeltaNetError> {
        DeltaNetState::new(self.geometry)
    }

    pub fn forward_token(
        &self,
        input: &[f32],
        state: &mut DeltaNetState,
    ) -> Result<Vec<f32>, DeltaNetError> {
        if state.geometry != self.geometry {
            return Err(DeltaNetError::StateGeometry {
                expected: self.geometry,
                got: state.geometry,
            });
        }
        expect_input(input, self.geometry.hidden_size)?;
        let next_tokens = state
            .tokens
            .checked_add(1)
            .ok_or(DeltaNetError::StateLengthOverflow)?;

        let mixed_qkv = linear(&self.in_proj_qkv, input)?;
        let z = linear(&self.in_proj_z, input)?;
        let b = linear(&self.in_proj_b, input)?;
        let a = linear(&self.in_proj_a, input)?;
        ensure_finite("QKV projection", &mixed_qkv)?;
        ensure_finite("z projection", &z)?;
        ensure_finite("beta projection", &b)?;
        ensure_finite("decay projection", &a)?;

        let conv_width = self.geometry.conv_width()?;
        let kernel = self.geometry.conv_kernel_size;
        let mut next_conv = vec![0.0; state.conv.len()];
        let mut convolved = vec![0.0; conv_width];
        for channel in 0..conv_width {
            let start = channel * kernel;
            let old = &state.conv[start..start + kernel];
            let new = &mut next_conv[start..start + kernel];
            if kernel > 1 {
                new[..kernel - 1].copy_from_slice(&old[1..]);
            }
            new[kernel - 1] = mixed_qkv[channel];
            let convolution = new
                .iter()
                .zip(&self.conv_weight[start..start + kernel])
                .map(|(&value, &weight)| f64::from(value) * f64::from(weight))
                .sum::<f64>() as f32;
            let convolution = if self.bf16_activations {
                round_to_bf16(convolution)?
            } else {
                convolution
            };
            convolved[channel] = silu(convolution);
            if self.bf16_activations {
                convolved[channel] = round_to_bf16(convolved[channel])?;
            }
        }
        ensure_finite("causal convolution", &convolved)?;

        let key_width = self.geometry.key_width()?;
        let value_width = self.geometry.value_width()?;
        let query = &convolved[..key_width];
        let key = &convolved[key_width..2 * key_width];
        let value = &convolved[2 * key_width..2 * key_width + value_width];

        let repeats = self.geometry.num_value_heads / self.geometry.num_key_heads;
        let query_scale = (self.geometry.key_head_dim as f32).sqrt().recip();
        let mut next_recurrent = state.recurrent.clone();
        let mut core_output = vec![0.0; value_width];
        for value_head in 0..self.geometry.num_value_heads {
            let key_head = value_head / repeats;
            let key_start = key_head * self.geometry.key_head_dim;
            let normalized_query = l2_normalize(
                &query[key_start..key_start + self.geometry.key_head_dim],
                self.bf16_activations,
            )?;
            let normalized_key = l2_normalize(
                &key[key_start..key_start + self.geometry.key_head_dim],
                self.bf16_activations,
            )?;
            let value_start = value_head * self.geometry.value_head_dim;
            let head_value = &value[value_start..value_start + self.geometry.value_head_dim];

            let beta = sigmoid(b[value_head]);
            let beta = if self.bf16_activations {
                round_to_bf16(beta)?
            } else {
                beta
            };
            let decay_log =
                -self.a_log[value_head].exp() * softplus(a[value_head] + self.dt_bias[value_head]);
            if !decay_log.is_finite() {
                return Err(DeltaNetError::NonFinite("decay log"));
            }
            let decay = decay_log.exp();
            let state_start =
                value_head * self.geometry.key_head_dim * self.geometry.value_head_dim;
            let head_state = &mut next_recurrent[state_start
                ..state_start + self.geometry.key_head_dim * self.geometry.value_head_dim];
            for state_value in head_state.iter_mut() {
                *state_value *= decay;
            }

            let mut delta = vec![0.0; self.geometry.value_head_dim];
            for value_dimension in 0..self.geometry.value_head_dim {
                let memory = (0..self.geometry.key_head_dim)
                    .map(|key_dimension| {
                        f64::from(
                            head_state
                                [key_dimension * self.geometry.value_head_dim + value_dimension],
                        ) * f64::from(normalized_key[key_dimension])
                    })
                    .sum::<f64>() as f32;
                delta[value_dimension] = (head_value[value_dimension] - memory) * beta;
            }
            for key_dimension in 0..self.geometry.key_head_dim {
                for value_dimension in 0..self.geometry.value_head_dim {
                    head_state[key_dimension * self.geometry.value_head_dim + value_dimension] +=
                        normalized_key[key_dimension] * delta[value_dimension];
                }
            }
            for value_dimension in 0..self.geometry.value_head_dim {
                core_output[value_start + value_dimension] = (0..self.geometry.key_head_dim)
                    .map(|key_dimension| {
                        f64::from(
                            head_state
                                [key_dimension * self.geometry.value_head_dim + value_dimension],
                        ) * f64::from(normalized_query[key_dimension] * query_scale)
                    })
                    .sum::<f64>()
                    as f32;
            }
        }
        ensure_finite("recurrent state", &next_recurrent)?;
        ensure_finite("recurrent output", &core_output)?;
        if self.bf16_activations {
            round_to_bf16_in_place(&mut core_output)?;
        }

        let mut gated_output = Vec::with_capacity(value_width);
        for value_head in 0..self.geometry.num_value_heads {
            let start = value_head * self.geometry.value_head_dim;
            let normalized = if self.bf16_activations {
                gated_rms_norm_bf16(
                    &core_output[start..start + self.geometry.value_head_dim],
                    &self.norm_weight,
                    &z[start..start + self.geometry.value_head_dim],
                    self.geometry.norm_eps,
                )?
            } else {
                gated_rms_norm(
                    &core_output[start..start + self.geometry.value_head_dim],
                    &self.norm_weight,
                    &z[start..start + self.geometry.value_head_dim],
                    self.geometry.norm_eps,
                )?
            };
            gated_output.extend(normalized);
        }
        let output = linear(&self.out_proj, &gated_output)?;
        ensure_finite("output projection", &output)?;

        // Commit both state components only once the complete token succeeds.
        state.conv = next_conv;
        state.recurrent = next_recurrent;
        state.tokens = next_tokens;
        Ok(output)
    }
}

fn l2_normalize(values: &[f32], bf16_activations: bool) -> Result<Vec<f32>, DeltaNetError> {
    ensure_finite("Q/K vector", values)?;
    let squared_norm = if bf16_activations {
        let terms = values
            .iter()
            .map(|&value| round_to_bf16(value * value))
            .collect::<Result<Vec<_>, _>>()?;
        round_to_bf16(terms.into_iter().sum::<f32>())?
    } else {
        values
            .iter()
            .fold(0.0f32, |sum, &value| sum + value * value)
    };
    let shifted = if bf16_activations {
        round_to_bf16(squared_norm + QK_L2_EPSILON)?
    } else {
        squared_norm + QK_L2_EPSILON
    };
    let inverse = if bf16_activations {
        round_to_bf16(shifted.sqrt().recip())?
    } else {
        shifted.sqrt().recip()
    };
    if !inverse.is_finite() {
        return Err(DeltaNetError::NonFinite("Q/K normalization factor"));
    }
    let normalized = values
        .iter()
        .map(|&value| {
            if bf16_activations {
                round_to_bf16(value * inverse)
            } else {
                Ok(value * inverse)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(normalized)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize, DeltaNetError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            DeltaNetError::InvalidGeometry(format!("{name} dimensions overflow usize: {factors:?}"))
        })
    })
}

fn expect_matrix(
    name: &str,
    matrix: &WeightMatrix,
    rows: usize,
    columns: usize,
) -> Result<(), DeltaNetError> {
    if matrix.rows() != rows || matrix.cols() != columns {
        return Err(DeltaNetError::InvalidShape(format!(
            "{name} must be [{rows}, {columns}], got [{}, {}]",
            matrix.rows(),
            matrix.cols()
        )));
    }
    Ok(())
}

fn expect_vector(name: &str, vector: &[f32], length: usize) -> Result<(), DeltaNetError> {
    if vector.len() != length {
        return Err(DeltaNetError::InvalidShape(format!(
            "{name} must have {length} values, got {}",
            vector.len()
        )));
    }
    ensure_finite("weight vector", vector)
}

fn expect_input(input: &[f32], length: usize) -> Result<(), DeltaNetError> {
    if input.len() != length {
        return Err(DeltaNetError::InvalidShape(format!(
            "input must have {length} values, got {}",
            input.len()
        )));
    }
    ensure_finite("input", input)
}

fn ensure_finite(name: &'static str, values: &[f32]) -> Result<(), DeltaNetError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DeltaNetError::NonFinite(name));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> DenseMatrix {
        DenseMatrix::new(rows, columns, values).unwrap()
    }

    fn tiny_delta_net() -> GatedDeltaNet {
        let geometry = DeltaNetGeometry {
            hidden_size: 2,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 1,
            value_head_dim: 2,
            conv_kernel_size: 2,
            norm_eps: 1e-6,
        };
        // q=x0, k=x0, v=[x1, x0+x1].
        let qkv = matrix(4, 2, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let z = matrix(2, 2, vec![1.0, 0.0, 1.0, 0.0]);
        let zero_head = matrix(1, 2, vec![0.0, 0.0]);
        let output = matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        GatedDeltaNet::new(
            geometry,
            qkv,
            z,
            zero_head.clone(),
            zero_head,
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.0],
            vec![0.0],
            vec![1.0, 1.0],
            output,
        )
        .unwrap()
    }

    #[test]
    fn one_token_updates_full_conv_window_and_fp32_recurrence() {
        let delta_net = tiny_delta_net();
        let mut state = delta_net.new_state().unwrap();
        let output = delta_net.forward_token(&[1.0, 2.0], &mut state).unwrap();
        assert_eq!(state.len(), 1);
        assert_eq!(
            state.conv_state(),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
        );
        let normalized_key = silu(1.0) / (silu(1.0).powi(2) + QK_L2_EPSILON).sqrt();
        let expected_state = [
            normalized_key * silu(2.0) * 0.5,
            normalized_key * silu(3.0) * 0.5,
        ];
        for (&actual, expected) in state.recurrent_state().iter().zip(expected_state) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output[1] > output[0]);
    }

    #[test]
    fn repeated_step_decays_then_updates_recurrent_state() {
        let delta_net = tiny_delta_net();
        let mut state = delta_net.new_state().unwrap();
        delta_net.forward_token(&[1.0, 2.0], &mut state).unwrap();
        let first = state.recurrent_state().to_vec();
        delta_net.forward_token(&[1.0, 2.0], &mut state).unwrap();
        assert_eq!(state.len(), 2);
        assert!(state.recurrent_state()[0] > first[0]);
        assert!(state.recurrent_state()[1] > first[1]);
    }

    #[test]
    fn invalid_input_leaves_both_states_unchanged() {
        let delta_net = tiny_delta_net();
        let mut state = delta_net.new_state().unwrap();
        let before = state.clone();
        let error = delta_net
            .forward_token(&[f32::NAN, 0.0], &mut state)
            .unwrap_err();
        assert!(matches!(error, DeltaNetError::NonFinite("input")));
        assert_eq!(state.tokens, before.tokens);
        assert_eq!(state.conv, before.conv);
        assert_eq!(state.recurrent, before.recurrent);
    }
}
