use super::math::{linear, linear_batch, round_to_bf16, round_to_bf16_in_place, uses_bf16_output};
use super::norm::{gated_rms_norm, gated_rms_norm_bf16, NormError};
use crate::execution::{install, should_parallelize};
use crate::model::{DenseMatrix, WeightError, WeightMatrix};
use crate::profiling::{span, ProfileStage};
use rayon::prelude::*;
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

/// One transactional working state, reusable across sequential layers of the same geometry.
/// It is execution scratch, so session checkpoints need only copy the committed causal state.
#[derive(Debug, Default)]
pub(crate) struct DeltaNetWorkspace {
    state: Option<DeltaNetState>,
}

impl DeltaNetWorkspace {
    fn prepare(&mut self, source: &DeltaNetState) -> &mut DeltaNetState {
        match self.state.as_mut() {
            Some(state) if state.geometry == source.geometry => {
                state.conv.clone_from(&source.conv);
                state.recurrent.clone_from(&source.recurrent);
                state.tokens = source.tokens;
            }
            _ => self.state = Some(source.clone()),
        }
        self.state.as_mut().expect("workspace was initialized")
    }
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

/// One-token Qwen Gated DeltaNet with ordered recurrent reductions.
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
        self.forward_token_with_workspace(input, state, &mut DeltaNetWorkspace::default())
    }

    pub(crate) fn forward_token_with_workspace(
        &self,
        input: &[f32],
        state: &mut DeltaNetState,
        workspace: &mut DeltaNetWorkspace,
    ) -> Result<Vec<f32>, DeltaNetError> {
        if state.geometry != self.geometry {
            return Err(DeltaNetError::StateGeometry {
                expected: self.geometry,
                got: state.geometry,
            });
        }
        expect_input(input, self.geometry.hidden_size)?;
        state
            .tokens
            .checked_add(1)
            .ok_or(DeltaNetError::StateLengthOverflow)?;

        let projection_profile = span(ProfileStage::QwenDeltaInputProjection);
        let mixed_qkv = linear(&self.in_proj_qkv, input)?;
        let z = linear(&self.in_proj_z, input)?;
        let b = linear(&self.in_proj_b, input)?;
        let a = linear(&self.in_proj_a, input)?;
        drop(projection_profile);
        let state_profile = span(ProfileStage::QwenDeltaStateUpdate);
        let next_state = workspace.prepare(state);
        let gated_output = self.recurrent_step(&mixed_qkv, &z, &b, &a, next_state)?;
        drop(state_profile);
        let _output_profile = span(ProfileStage::QwenDeltaOutputProjection);
        let output = linear(&self.out_proj, &gated_output)?;
        ensure_finite("output projection", &output)?;
        std::mem::swap(state, next_state);
        Ok(output)
    }

    /// Prefills consecutive tokens with shared projection weights and ordered recurrence.
    /// The caller bounds the batch; neither state component changes if any token fails.
    pub fn forward_tokens(
        &self,
        input: &[f32],
        batch: usize,
        state: &mut DeltaNetState,
    ) -> Result<Vec<f32>, DeltaNetError> {
        self.forward_tokens_with_workspace(input, batch, state, &mut DeltaNetWorkspace::default())
    }

    pub(crate) fn forward_tokens_with_workspace(
        &self,
        input: &[f32],
        batch: usize,
        state: &mut DeltaNetState,
        workspace: &mut DeltaNetWorkspace,
    ) -> Result<Vec<f32>, DeltaNetError> {
        if batch == 1 {
            return self.forward_token_with_workspace(input, state, workspace);
        }
        if batch == 0 {
            return Err(DeltaNetError::InvalidShape(
                "prefill batch must be non-zero".into(),
            ));
        }
        if state.geometry != self.geometry {
            return Err(DeltaNetError::StateGeometry {
                expected: self.geometry,
                got: state.geometry,
            });
        }
        let length = batch
            .checked_mul(self.geometry.hidden_size)
            .ok_or_else(|| DeltaNetError::InvalidShape("prefill input length overflows".into()))?;
        expect_input(input, length)?;
        state
            .tokens
            .checked_add(batch)
            .ok_or(DeltaNetError::StateLengthOverflow)?;
        let projection_profile = span(ProfileStage::QwenDeltaInputProjection);
        let qkv = linear_batch(&self.in_proj_qkv, input, batch)?;
        let z = linear_batch(&self.in_proj_z, input, batch)?;
        let b = linear_batch(&self.in_proj_b, input, batch)?;
        let a = linear_batch(&self.in_proj_a, input, batch)?;
        drop(projection_profile);
        let conv_width = self.geometry.conv_width()?;
        let value_width = self.geometry.value_width()?;
        let heads = self.geometry.num_value_heads;
        let state_profile = span(ProfileStage::QwenDeltaStateUpdate);
        let next_state = workspace.prepare(state);
        let mut gated = Vec::with_capacity(batch * value_width);
        for token in 0..batch {
            let output = self.recurrent_step(
                &qkv[token * conv_width..(token + 1) * conv_width],
                &z[token * value_width..(token + 1) * value_width],
                &b[token * heads..(token + 1) * heads],
                &a[token * heads..(token + 1) * heads],
                next_state,
            )?;
            gated.extend(output);
        }
        drop(state_profile);
        let _output_profile = span(ProfileStage::QwenDeltaOutputProjection);
        let output = linear_batch(&self.out_proj, &gated, batch)?;
        ensure_finite("output projection", &output)?;
        std::mem::swap(state, next_state);
        Ok(output)
    }

    // Mutates only a private transaction workspace. Callers publish it after the complete
    // token/batch output projection succeeds, so a partial update may be discarded on error.
    #[inline(always)]
    fn recurrent_step(
        &self,
        mixed_qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        state: &mut DeltaNetState,
    ) -> Result<Vec<f32>, DeltaNetError> {
        let next_tokens = state
            .tokens
            .checked_add(1)
            .ok_or(DeltaNetError::StateLengthOverflow)?;
        ensure_finite("QKV projection", mixed_qkv)?;
        ensure_finite("z projection", z)?;
        ensure_finite("beta projection", b)?;
        ensure_finite("decay projection", a)?;

        let conv_width = self.geometry.conv_width()?;
        let kernel = self.geometry.conv_kernel_size;
        let mut convolved = vec![0.0; conv_width];
        for channel in 0..conv_width {
            let start = channel * kernel;
            let new = &mut state.conv[start..start + kernel];
            if kernel > 1 {
                new.copy_within(1.., 0);
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
        let normalized_queries = query
            .chunks_exact(self.geometry.key_head_dim)
            .map(|head| l2_normalize(head, self.bf16_activations))
            .collect::<Result<Vec<_>, _>>()?;
        let normalized_keys = key
            .chunks_exact(self.geometry.key_head_dim)
            .map(|head| l2_normalize(head, self.bf16_activations))
            .collect::<Result<Vec<_>, _>>()?;
        let next_recurrent = &mut state.recurrent;
        let mut core_output = vec![0.0; value_width];
        let head_state_len = self.geometry.key_head_dim * self.geometry.value_head_dim;
        let update_head = |value_head: usize,
                           head_state: &mut [f32],
                           head_output: &mut [f32]|
         -> Result<(), DeltaNetError> {
            let key_head = value_head / repeats;
            let normalized_query = &normalized_queries[key_head];
            let normalized_key = &normalized_keys[key_head];
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
            for state_value in head_state.iter_mut() {
                *state_value *= decay;
            }

            // The native 128-wide heads use stack scratch. Wider geometries keep a bounded
            // per-head fallback, while the output slice temporarily holds the rank-one delta.
            let mut local_sums = [0.0f64; 128];
            let mut wide_sums;
            let sums = if self.geometry.value_head_dim <= local_sums.len() {
                &mut local_sums[..self.geometry.value_head_dim]
            } else {
                wide_sums = vec![0.0; self.geometry.value_head_dim];
                &mut wide_sums
            };
            project_recurrent_state_into(head_state, normalized_key, 1.0, sums);
            for ((delta, &value), &memory) in
                head_output.iter_mut().zip(head_value).zip(sums.iter())
            {
                *delta = (value - memory as f32) * beta;
            }
            for (row, &key) in head_state
                .chunks_exact_mut(self.geometry.value_head_dim)
                .zip(normalized_key)
            {
                for (value, &delta) in row.iter_mut().zip(head_output.iter()) {
                    *value += key * delta;
                }
            }
            project_recurrent_state_into(head_state, normalized_query, query_scale, sums);
            for (output, &sum) in head_output.iter_mut().zip(sums.iter()) {
                *output = sum as f32;
            }
            Ok(())
        };
        // Heads own disjoint state and output slices. Scheduling changes no reduction order.
        if should_parallelize(self.geometry.num_value_heads, next_recurrent.len()) {
            install(|| {
                next_recurrent
                    .par_chunks_mut(head_state_len)
                    .zip(core_output.par_chunks_mut(self.geometry.value_head_dim))
                    .enumerate()
                    .try_for_each(|(head, (state, output))| update_head(head, state, output))
            })?;
        } else {
            for (head, (state, output)) in next_recurrent
                .chunks_mut(head_state_len)
                .zip(core_output.chunks_mut(self.geometry.value_head_dim))
                .enumerate()
            {
                update_head(head, state, output)?;
            }
        }
        ensure_finite("recurrent state", next_recurrent)?;
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
        state.tokens = next_tokens;
        Ok(gated_output)
    }
}

/// Ordered F64 reduction, identical to the column-wise reference, with contiguous loads.
fn project_recurrent_state_into(state: &[f32], vector: &[f32], scale: f32, sums: &mut [f64]) {
    sums.fill(-0.0);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and the helper bounds all loads/stores to full lanes.
        unsafe { project_recurrent_state_avx2(state, vector, scale, sums) };
        return;
    }
    for (row, &factor) in state.chunks_exact(sums.len()).zip(vector) {
        let factor = f64::from(factor * scale);
        for (sum, &value) in sums.iter_mut().zip(row) {
            *sum += f64::from(value) * factor;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_recurrent_state_avx2(
    state: &[f32],
    vector: &[f32],
    scale: f32,
    sums: &mut [f64],
) {
    use std::arch::x86_64::*;
    let width = sums.len();
    let complete = width / 4 * 4;
    for (row, &factor) in state.chunks_exact(width).zip(vector) {
        let factor = f64::from(factor * scale);
        let factors = _mm256_set1_pd(factor);
        for column in (0..complete).step_by(4) {
            let values = _mm256_cvtps_pd(_mm_loadu_ps(row.as_ptr().add(column)));
            let previous = _mm256_loadu_pd(sums.as_ptr().add(column));
            // Separate multiply and add preserve the scalar F64 rounding; do not use FMA.
            let next = _mm256_add_pd(previous, _mm256_mul_pd(values, factors));
            _mm256_storeu_pd(sums.as_mut_ptr().add(column), next);
        }
        for column in complete..width {
            sums[column] += f64::from(row[column]) * factor;
        }
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
    if !values_are_finite(values) {
        return Err(DeltaNetError::NonFinite(name));
    }
    Ok(())
}

fn values_are_finite(values: &[f32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and the helper only loads complete eight-value chunks.
        return unsafe { values_are_finite_avx2(values) };
    }
    values.iter().all(|value| value.is_finite())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn values_are_finite_avx2(values: &[f32]) -> bool {
    use std::arch::x86_64::*;
    let exponent = _mm256_set1_epi32(0x7f80_0000);
    let mut chunks = values.chunks_exact(8);
    for chunk in &mut chunks {
        let bits = _mm256_loadu_si256(chunk.as_ptr().cast());
        let nonfinite = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent);
        if _mm256_movemask_epi8(nonfinite) != 0 {
            return false;
        }
    }
    chunks.remainder().iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_scan_handles_unaligned_tails_and_all_exponent_patterns() {
        let finite = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x807f_ffff),
        ];
        let nonfinite = [
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7fc0_0000,
            0xffc0_0000,
            0x7fff_ffff,
            0xffff_ffff,
        ];
        for offset in 0..8 {
            for length in 0..=65 {
                let mut buffer = (0..offset + length)
                    .map(|i| finite[i % finite.len()])
                    .collect::<Vec<_>>();
                let values = &mut buffer[offset..];
                assert!(values_are_finite(values));
                for position in 0..length {
                    let old = values[position];
                    for bits in nonfinite {
                        values[position] = f32::from_bits(bits);
                        assert!(
                            !values_are_finite(values),
                            "offset={offset} length={length} position={position}"
                        );
                    }
                    values[position] = old;
                }
            }
        }
        // Every sign/exponent/high-mantissa pattern with representative low mantissas,
        // including signaling NaNs and values immediately below infinity.
        for low in [0, 1, 0x7fff, 0xffff] {
            for high in 0..=u16::MAX {
                let value = f32::from_bits((u32::from(high) << 16) | low);
                assert_eq!(values_are_finite(&[value; 8]), value.is_finite());
            }
        }
    }

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

    fn tiny_delta_net_with_dtype(bf16: bool) -> GatedDeltaNet {
        let mut module = tiny_delta_net();
        if bf16 {
            for weight in [
                &mut module.in_proj_qkv,
                &mut module.in_proj_z,
                &mut module.in_proj_b,
                &mut module.in_proj_a,
                &mut module.out_proj,
            ] {
                let WeightMatrix::F32(dense) = weight else {
                    unreachable!()
                };
                let bytes = dense
                    .data()
                    .iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>();
                *weight = WeightMatrix::Bf16(
                    crate::model::Bf16Matrix::from_le_bytes(dense.rows(), dense.cols(), bytes)
                        .unwrap(),
                );
            }
            module.bf16_activations = true;
        }
        module
    }

    #[test]
    fn batched_prefill_preserves_outputs_and_existing_state_bits() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for bf16 in [false, true] {
            let module = tiny_delta_net_with_dtype(bf16);
            for batch in [2, 3, 8, 19, 32, 33, 128, 129] {
                let input = (0..batch * 2)
                    .map(|i| ((i * 19 % 41) as f32 - 20.0) / 16.0)
                    .collect::<Vec<_>>();
                let mut sequential = module.new_state().unwrap();
                module.forward_token(&[0.25, 0.5], &mut sequential).unwrap();
                let mut batched = sequential.clone();
                let mut expected = Vec::new();
                for token in input.chunks_exact(2) {
                    expected.extend(module.forward_token(token, &mut sequential).unwrap());
                }
                let actual = module.forward_tokens(&input, batch, &mut batched).unwrap();
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(bits(&batched.conv), bits(&sequential.conv));
                assert_eq!(bits(&batched.recurrent), bits(&sequential.recurrent));
                assert_eq!(batched.tokens, sequential.tokens);
            }
        }
    }

    #[test]
    fn batch_output_failure_preserves_both_state_components() {
        let mut module = tiny_delta_net();
        module.out_proj = matrix(2, 2, vec![f32::MAX; 4]).into();
        let mut state = module.new_state().unwrap();
        module.forward_token(&[0.0, 0.0], &mut state).unwrap();
        let before = state.clone();
        assert!(module
            .forward_tokens(&[1.0, 2.0, 2.0, 3.0], 2, &mut state)
            .is_err());
        assert_eq!(state.tokens, before.tokens);
        assert_eq!(state.conv, before.conv);
        assert_eq!(state.recurrent, before.recurrent);
        assert!(module.forward_tokens(&[], 0, &mut state).is_err());
        assert_eq!(state.tokens, before.tokens);
    }

    #[test]
    fn shared_workspace_preserves_bits_and_reuses_buffers_across_layers() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let pointers = |states: &[DeltaNetState], workspace: &DeltaNetWorkspace| {
            let mut pointers = states
                .iter()
                .chain(workspace.state.iter())
                .map(|state| {
                    (
                        state.conv.as_ptr() as usize,
                        state.recurrent.as_ptr() as usize,
                    )
                })
                .collect::<Vec<_>>();
            pointers.sort_unstable();
            pointers
        };
        for bf16 in [false, true] {
            let module = tiny_delta_net_with_dtype(bf16);
            let mut states = [module.new_state().unwrap(), module.new_state().unwrap()];
            let mut references = states.clone();
            let mut workspace = DeltaNetWorkspace::default();
            module
                .forward_token_with_workspace(&[0.25, 0.5], &mut states[0], &mut workspace)
                .unwrap();
            module
                .forward_token(&[0.25, 0.5], &mut references[0])
                .unwrap();
            let allocations = pointers(&states, &workspace);
            for (layer, batch) in [(1, 3), (0, 1), (1, 8), (0, 2), (1, 32), (0, 19)] {
                let input = (0..batch * 2)
                    .map(|i| ((i * 19 % 41) as f32 - 20.0) / 16.0)
                    .collect::<Vec<_>>();
                let actual = module
                    .forward_tokens_with_workspace(
                        &input,
                        batch,
                        &mut states[layer],
                        &mut workspace,
                    )
                    .unwrap();
                let expected = module
                    .forward_tokens(&input, batch, &mut references[layer])
                    .unwrap();
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(bits(&states[layer].conv), bits(&references[layer].conv));
                assert_eq!(
                    bits(&states[layer].recurrent),
                    bits(&references[layer].recurrent)
                );
                assert_eq!(states[layer].tokens, references[layer].tokens);
                assert_eq!(pointers(&states, &workspace), allocations);
            }
        }
    }

    #[test]
    fn workspace_recovers_after_output_failure_and_geometry_change() {
        let mut module = tiny_delta_net();
        let mut state = module.new_state().unwrap();
        let mut workspace = DeltaNetWorkspace::default();
        module
            .forward_token_with_workspace(&[0.0, 0.0], &mut state, &mut workspace)
            .unwrap();
        let before = state.clone();
        let output_projection = module.out_proj.clone();
        module.out_proj = matrix(2, 2, vec![f32::MAX; 4]).into();
        assert!(module
            .forward_tokens_with_workspace(&[1.0, 2.0, 2.0, 3.0], 2, &mut state, &mut workspace)
            .is_err());
        assert_eq!(state.tokens, before.tokens);
        assert_eq!(state.conv, before.conv);
        assert_eq!(state.recurrent, before.recurrent);
        module.out_proj = output_projection;
        let mut reference = before;
        let expected = module
            .forward_tokens(&[1.0, 2.0, 2.0, 3.0], 2, &mut reference)
            .unwrap();
        let actual = module
            .forward_tokens_with_workspace(&[1.0, 2.0, 2.0, 3.0], 2, &mut state, &mut workspace)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(state.recurrent, reference.recurrent);
        assert_eq!(state.conv, reference.conv);
        assert_eq!(state.tokens, reference.tokens);

        module.geometry.conv_kernel_size = 3;
        module.conv_weight = [0.0, 0.0, 1.0].repeat(module.geometry.conv_width().unwrap());
        let mut different = module.new_state().unwrap();
        let mut reference = different.clone();
        let expected = module.forward_token(&[0.25, 0.5], &mut reference).unwrap();
        let actual = module
            .forward_token_with_workspace(&[0.25, 0.5], &mut different, &mut workspace)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(different.conv, reference.conv);
        assert_eq!(different.recurrent, reference.recurrent);
    }

    #[test]
    fn contiguous_state_projection_preserves_ordered_reductions() {
        fn project_recurrent_state(
            state: &[f32],
            vector: &[f32],
            values: usize,
            scale: f32,
        ) -> Vec<f32> {
            let mut sums = vec![0.0; values];
            project_recurrent_state_into(state, vector, scale, &mut sums);
            sums.into_iter().map(|value| value as f32).collect()
        }
        for (keys, values) in [(1, 1), (3, 5), (17, 31), (128, 128), (129, 257)] {
            let state = (0..keys * values)
                .map(|i| ((i * 137 % 251) as f32 - 125.0) / 37.0)
                .collect::<Vec<_>>();
            let vector = (0..keys)
                .map(|i| ((i * 19 % 97) as f32 - 48.0) / 31.0)
                .collect::<Vec<_>>();
            for scale in [1.0, (keys as f32).sqrt().recip()] {
                let actual = project_recurrent_state(&state, &vector, values, scale);
                let expected = (0..values)
                    .map(|v| {
                        (0..keys)
                            .map(|k| {
                                f64::from(state[k * values + v]) * f64::from(vector[k] * scale)
                            })
                            .sum::<f64>() as f32
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        assert_eq!(
            project_recurrent_state(&[-0.0], &[1.0], 1, 1.0)[0].to_bits(),
            ([-0.0f64].into_iter().sum::<f64>() as f32).to_bits()
        );
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
    fn late_head_failure_leaves_large_state_unchanged() {
        let geometry = DeltaNetGeometry {
            hidden_size: 1,
            num_key_heads: 1,
            num_value_heads: 16,
            key_head_dim: 32,
            value_head_dim: 128,
            conv_kernel_size: 2,
            norm_eps: 1e-6,
        };
        let conv_width = geometry.conv_width().unwrap();
        let value_width = geometry.value_width().unwrap();
        let zeros = |rows, cols| DenseMatrix::new(rows, cols, vec![0.0; rows * cols]).unwrap();
        let mut a_log = vec![0.0; geometry.num_value_heads];
        let mut dt_bias = a_log.clone();
        // exp(88) is finite, but its product with softplus(3) overflows in the last head.
        a_log[geometry.num_value_heads - 1] = 88.0;
        dt_bias[geometry.num_value_heads - 1] = 3.0;
        let module = GatedDeltaNet::new(
            geometry,
            zeros(conv_width, 1),
            zeros(value_width, 1),
            zeros(geometry.num_value_heads, 1),
            zeros(geometry.num_value_heads, 1),
            vec![0.5; conv_width * geometry.conv_kernel_size],
            dt_bias,
            a_log,
            vec![1.0; geometry.value_head_dim],
            zeros(1, value_width),
        )
        .unwrap();
        let mut state = module.new_state().unwrap();
        state.conv.fill(0.5);
        state.recurrent.fill(1.0);
        state.tokens = 7;
        let before = state.clone();
        let error = module.forward_token(&[1.0], &mut state).unwrap_err();
        assert!(matches!(error, DeltaNetError::NonFinite("decay log")));
        assert_eq!(state.tokens, before.tokens);
        assert_eq!(state.conv, before.conv);
        assert_eq!(state.recurrent, before.recurrent);
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
