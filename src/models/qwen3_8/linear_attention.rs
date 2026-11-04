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
    sigmoid_output_gate: bool,
    qk_norm_fp32: bool,
    out_proj: WeightMatrix,
    bf16_activations: bool,
}

struct RecurrentInput {
    convolved: Vec<f32>,
    queries: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
}

// Decode previously completed each of these checks for all heads before the next
// phase. Independent heads may finish together, but failures retain that priority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RecurrentCheck {
    Update,
    State,
    Output,
    OutputCast,
    Gate,
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
            sigmoid_output_gate: false,
            qk_norm_fp32: false,
            out_proj,
            bf16_activations,
        })
    }

    /// Flash-Next normalizes Q/K and updates recurrent state in FP32, then uses a sigmoid output gate.
    pub(crate) fn with_flash_next_semantics(mut self) -> Self {
        self.sigmoid_output_gate = true;
        self.qk_norm_fp32 = true;
        self
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
        let gated = if should_parallelize(heads, next_state.recurrent.len()) {
            self.recurrent_batch(&qkv, &z, &b, &a, next_state)?
        } else {
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
            gated
        };
        drop(state_profile);
        let _output_profile = span(ProfileStage::QwenDeltaOutputProjection);
        let output = linear_batch(&self.out_proj, &gated, batch)?;
        ensure_finite("output projection", &output)?;
        std::mem::swap(state, next_state);
        Ok(output)
    }

    /// Each worker keeps a head's state through the entire chunk, in token order. The
    /// convolution windows are prepared in token order before heads run independently.
    /// All mutations remain in the caller's unpublished transaction workspace.
    fn recurrent_batch(
        &self,
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        state: &mut DeltaNetState,
    ) -> Result<Vec<f32>, DeltaNetError> {
        ensure_finite("z projection", z)?;
        ensure_finite("beta projection", b)?;
        ensure_finite("decay projection", a)?;
        let heads = self.geometry.num_value_heads;
        let batch = b.len() / heads;
        let width = self.geometry.value_head_dim;
        let value_width = self.geometry.value_width()?;
        let next_tokens = state
            .tokens
            .checked_add(batch)
            .ok_or(DeltaNetError::StateLengthOverflow)?;
        let inputs = qkv
            .chunks_exact(self.geometry.conv_width()?)
            .map(|token| self.prepare_recurrent_input(token, &mut state.conv))
            .collect::<Result<Vec<_>, _>>()?;
        let mut by_head = vec![0.0; batch * value_width];
        install(|| {
            state
                .recurrent
                .par_chunks_mut(self.geometry.key_head_dim * width)
                .zip(by_head.par_chunks_mut(batch * width))
                .enumerate()
                .try_for_each(|(head, (head_state, outputs))| {
                    let mut core = vec![0.0; width];
                    for (token, output) in outputs.chunks_exact_mut(width).enumerate() {
                        self.update_recurrent_head(
                            head,
                            &inputs[token],
                            (b[token * heads + head], a[token * heads + head]),
                            head_state,
                            &mut core,
                        )?;
                        // Validate every token before advancing this head to the next one.
                        ensure_finite("recurrent state", head_state)?;
                        let start = token * value_width + head * width;
                        self.gate_recurrent_output(&mut core, &z[start..start + width], output)?;
                    }
                    Ok::<_, DeltaNetError>(())
                })
        })?;
        let mut gated = vec![0.0; by_head.len()];
        for (head, outputs) in by_head.chunks_exact(batch * width).enumerate() {
            for (token, output) in outputs.chunks_exact(width).enumerate() {
                let start = token * value_width + head * width;
                gated[start..start + width].copy_from_slice(output);
            }
        }
        state.tokens = next_tokens;
        Ok(gated)
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

        let input = self.prepare_recurrent_input(mixed_qkv, &mut state.conv)?;
        let value_width = self.geometry.value_width()?;
        let next_recurrent = &mut state.recurrent;
        let head_state_len = self.geometry.key_head_dim * self.geometry.value_head_dim;
        if should_parallelize(self.geometry.num_value_heads, next_recurrent.len()) {
            let mut gated_output = vec![0.0; value_width];
            let results = install(|| {
                next_recurrent
                    .par_chunks_mut(head_state_len)
                    .zip(gated_output.par_chunks_mut(self.geometry.value_head_dim))
                    .zip(z.par_chunks(self.geometry.value_head_dim))
                    .enumerate()
                    .map(|(head, ((state, output), gate))| {
                        // State validation and per-head normalization stay on the worker
                        // that just updated these values; each reduction keeps its order.
                        self.update_recurrent_head(head, &input, (b[head], a[head]), state, output)
                            .map_err(|error| (RecurrentCheck::Update, error))?;
                        ensure_finite("recurrent state", state)
                            .map_err(|error| (RecurrentCheck::State, error))?;
                        ensure_finite("recurrent output", output)
                            .map_err(|error| (RecurrentCheck::Output, error))?;
                        if self.bf16_activations {
                            round_to_bf16_in_place(output).map_err(|error| {
                                (RecurrentCheck::OutputCast, DeltaNetError::from(error))
                            })?;
                        }
                        let normalized = if self.sigmoid_output_gate {
                            super::norm::gated_rms_norm_sigmoid(
                                output,
                                &self.norm_weight,
                                gate,
                                self.geometry.norm_eps,
                                self.bf16_activations,
                            )
                        } else if self.bf16_activations {
                            gated_rms_norm_bf16(
                                output,
                                &self.norm_weight,
                                gate,
                                self.geometry.norm_eps,
                            )
                        } else {
                            gated_rms_norm(output, &self.norm_weight, gate, self.geometry.norm_eps)
                        }
                        .map_err(|error| (RecurrentCheck::Gate, DeltaNetError::from(error)))?;
                        output.copy_from_slice(&normalized);
                        Ok(())
                    })
                    .collect::<Vec<_>>()
            });
            if let Some((_, error)) = results
                .into_iter()
                .filter_map(Result::err)
                .min_by_key(|(phase, _)| *phase)
            {
                return Err(error);
            }
            state.tokens = next_tokens;
            return Ok(gated_output);
        }
        let mut core_output = vec![0.0; value_width];
        for (head, (state, output)) in next_recurrent
            .chunks_mut(head_state_len)
            .zip(core_output.chunks_mut(self.geometry.value_head_dim))
            .enumerate()
        {
            self.update_recurrent_head(head, &input, (b[head], a[head]), state, output)?;
        }
        ensure_finite("recurrent state", next_recurrent)?;
        let mut gated_output = vec![0.0; value_width];
        self.gate_recurrent_output(&mut core_output, z, &mut gated_output)?;
        state.tokens = next_tokens;
        Ok(gated_output)
    }

    fn prepare_recurrent_input(
        &self,
        mixed_qkv: &[f32],
        conv: &mut [f32],
    ) -> Result<RecurrentInput, DeltaNetError> {
        ensure_finite("QKV projection", mixed_qkv)?;
        let conv_width = self.geometry.conv_width()?;
        let kernel = self.geometry.conv_kernel_size;
        let mut convolved = vec![0.0; conv_width];
        for channel in 0..conv_width {
            let start = channel * kernel;
            let new = &mut conv[start..start + kernel];
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
        let query = &convolved[..key_width];
        let key = &convolved[key_width..2 * key_width];
        let queries = query
            .chunks_exact(self.geometry.key_head_dim)
            .map(|head| l2_normalize(head, self.bf16_activations && !self.qk_norm_fp32))
            .collect::<Result<Vec<_>, _>>()?;
        let keys = key
            .chunks_exact(self.geometry.key_head_dim)
            .map(|head| l2_normalize(head, self.bf16_activations && !self.qk_norm_fp32))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RecurrentInput {
            convolved,
            queries,
            keys,
        })
    }

    fn update_recurrent_head(
        &self,
        value_head: usize,
        input: &RecurrentInput,
        (b, a): (f32, f32),
        head_state: &mut [f32],
        head_output: &mut [f32],
    ) -> Result<(), DeltaNetError> {
        let repeats = self.geometry.num_value_heads / self.geometry.num_key_heads;
        let query_scale = (self.geometry.key_head_dim as f32).sqrt().recip();
        let key_head = value_head / repeats;
        let normalized_query = &input.queries[key_head];
        let normalized_key = &input.keys[key_head];
        let value_start =
            2 * self.geometry.key_width()? + value_head * self.geometry.value_head_dim;
        let head_value = &input.convolved[value_start..value_start + self.geometry.value_head_dim];

        let beta = sigmoid(b);
        let beta = if self.bf16_activations {
            round_to_bf16(beta)?
        } else {
            beta
        };
        let decay_log = -self.a_log[value_head].exp() * softplus(a + self.dt_bias[value_head]);
        if !decay_log.is_finite() {
            return Err(DeltaNetError::NonFinite("decay log"));
        }
        let decay = decay_log.exp();
        if self.qk_norm_fp32 {
            // Flash-Next's reference casts Q/K to FP32 BEFORE L2 normalization, then
            // performs the recurrence in FP32. Keep the older Qwen path unchanged.
            let width = self.geometry.value_head_dim;
            for value in head_state.iter_mut() {
                *value *= decay;
            }
            for column in 0..width {
                let mut memory = 0.0f32;
                for (row, &key) in normalized_key.iter().enumerate() {
                    memory += head_state[row * width + column] * key;
                }
                head_output[column] = (head_value[column] - memory) * beta;
            }
            for (row, &key) in normalized_key.iter().enumerate() {
                for (column, &delta) in head_output.iter().enumerate() {
                    head_state[row * width + column] += key * delta;
                }
            }
            for (column, output) in head_output.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for (row, &query) in normalized_query.iter().enumerate() {
                    sum += head_state[row * width + column] * (query * query_scale);
                }
                *output = sum;
            }
            return Ok(());
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
        decay_and_project_recurrent_state(head_state, decay, normalized_key, sums);
        for ((delta, &value), &memory) in head_output.iter_mut().zip(head_value).zip(sums.iter()) {
            *delta = (value - memory as f32) * beta;
        }
        update_and_project_recurrent_state(
            head_state,
            normalized_key,
            head_output,
            normalized_query,
            query_scale,
            sums,
        );
        for (output, &sum) in head_output.iter_mut().zip(sums.iter()) {
            *output = sum as f32;
        }
        Ok(())
    }

    fn gate_recurrent_output(
        &self,
        core_output: &mut [f32],
        z: &[f32],
        gated_output: &mut [f32],
    ) -> Result<(), DeltaNetError> {
        debug_assert_eq!(core_output.len(), z.len());
        debug_assert_eq!(core_output.len(), gated_output.len());
        ensure_finite("recurrent output", core_output)?;
        if self.bf16_activations {
            round_to_bf16_in_place(core_output)?;
        }

        for ((core, z), output) in core_output
            .chunks_exact(self.geometry.value_head_dim)
            .zip(z.chunks_exact(self.geometry.value_head_dim))
            .zip(gated_output.chunks_exact_mut(self.geometry.value_head_dim))
        {
            let normalized = if self.sigmoid_output_gate {
                super::norm::gated_rms_norm_sigmoid(
                    core,
                    &self.norm_weight,
                    z,
                    self.geometry.norm_eps,
                    self.bf16_activations,
                )?
            } else if self.bf16_activations {
                gated_rms_norm_bf16(core, &self.norm_weight, z, self.geometry.norm_eps)?
            } else {
                gated_rms_norm(core, &self.norm_weight, z, self.geometry.norm_eps)?
            };
            output.copy_from_slice(&normalized);
        }
        Ok(())
    }
}

/// Decay each F32 state entry before consuming it in its ordered F64 key projection.
fn decay_and_project_recurrent_state(state: &mut [f32], decay: f32, key: &[f32], sums: &mut [f64]) {
    debug_assert_eq!(state.len(), key.len() * sums.len());
    for value in state.iter_mut() {
        *value *= decay;
    }
    project_recurrent_state_into(state, key, 1.0, sums);
}

/// Apply the F32 rank-one update before its ordered F64 query projection, retaining
/// separate multiplication, addition, and widening operations.
fn update_and_project_recurrent_state(
    state: &mut [f32],
    key: &[f32],
    delta: &[f32],
    query: &[f32],
    scale: f32,
    sums: &mut [f64],
) {
    debug_assert_eq!(state.len(), key.len() * sums.len());
    debug_assert_eq!(key.len(), query.len());
    debug_assert_eq!(delta.len(), sums.len());
    for (row, &key) in state.chunks_exact_mut(sums.len()).zip(key) {
        for (value, &delta) in row.iter_mut().zip(delta) {
            *value += key * delta;
        }
    }
    project_recurrent_state_into(state, query, scale, sums);
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
        with_dtype(tiny_delta_net(), bf16)
    }

    fn with_dtype(mut module: GatedDeltaNet, bf16: bool) -> GatedDeltaNet {
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

    fn decode_module(keys: usize, width: usize, heads: usize, bf16: bool) -> GatedDeltaNet {
        let geometry = DeltaNetGeometry {
            hidden_size: 1,
            num_key_heads: 2,
            num_value_heads: heads,
            key_head_dim: keys,
            value_head_dim: width,
            conv_kernel_size: 4,
            norm_eps: 1e-6,
        };
        let conv_width = geometry.conv_width().unwrap();
        let value_width = geometry.value_width().unwrap();
        let zeros = |rows, columns| matrix(rows, columns, vec![0.0; rows * columns]);
        with_dtype(
            GatedDeltaNet::new(
                geometry,
                zeros(conv_width, 1),
                zeros(value_width, 1),
                zeros(heads, 1),
                zeros(heads, 1),
                [0.125, -0.25, 0.375, 0.5].repeat(conv_width),
                vec![0.125; heads],
                vec![-0.25; heads],
                vec![0.875; width],
                zeros(1, value_width),
            )
            .unwrap(),
            bf16,
        )
    }

    // The original decode schedule: update all heads, then validate all state and
    // core outputs before normalization. This is independent of the fused schedule.
    fn staged_recurrent_step(
        module: &GatedDeltaNet,
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        state: &mut DeltaNetState,
    ) -> Result<Vec<f32>, DeltaNetError> {
        let next_tokens = state
            .tokens
            .checked_add(1)
            .ok_or(DeltaNetError::StateLengthOverflow)?;
        ensure_finite("QKV projection", qkv)?;
        ensure_finite("z projection", z)?;
        ensure_finite("beta projection", b)?;
        ensure_finite("decay projection", a)?;
        let input = module.prepare_recurrent_input(qkv, &mut state.conv)?;
        let width = module.geometry.value_head_dim;
        let mut core = vec![0.0; module.geometry.value_width()?];
        for (head, (state, output)) in state
            .recurrent
            .chunks_exact_mut(module.geometry.key_head_dim * width)
            .zip(core.chunks_exact_mut(width))
            .enumerate()
        {
            module.update_recurrent_head(head, &input, (b[head], a[head]), state, output)?;
        }
        ensure_finite("recurrent state", &state.recurrent)?;
        let mut output = vec![0.0; core.len()];
        module.gate_recurrent_output(&mut core, z, &mut output)?;
        state.tokens = next_tokens;
        Ok(output)
    }

    #[test]
    fn decode_head_completion_preserves_staged_state_and_output_bits() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for bf16 in [false, true] {
            for (keys, width, heads) in [(4, 5, 2), (128, 128, 32), (65, 129, 8)] {
                let module = decode_module(keys, width, heads, bf16);
                let row = |length, offset| {
                    (0..length)
                        .map(|i| ((i * 19 + offset) % 97) as f32 / 64.0 - 0.75)
                        .collect::<Vec<_>>()
                };
                let mut expected = module.new_state().unwrap();
                expected.conv = row(expected.conv.len(), 5);
                expected.recurrent = row(expected.recurrent.len(), 7);
                expected.recurrent[..4].copy_from_slice(&[
                    -0.0,
                    0.0,
                    f32::from_bits(1),
                    -f32::MIN_POSITIVE,
                ]);
                expected.tokens = 7;
                let mut actual = expected.clone();
                for token in 0..4 {
                    let qkv = row(module.geometry.conv_width().unwrap(), token * 13);
                    let z = row(module.geometry.value_width().unwrap(), token * 17);
                    let b = row(heads, token * 23);
                    let a = row(heads, token * 31);
                    let reference =
                        staged_recurrent_step(&module, &qkv, &z, &b, &a, &mut expected).unwrap();
                    let result = module
                        .recurrent_step(&qkv, &z, &b, &a, &mut actual)
                        .unwrap();
                    assert_eq!(bits(&result), bits(&reference));
                    assert_eq!(bits(&actual.conv), bits(&expected.conv));
                    assert_eq!(bits(&actual.recurrent), bits(&expected.recurrent));
                    assert_eq!(actual.tokens, expected.tokens);
                }
            }
        }
    }

    #[test]
    fn decode_head_completion_preserves_global_check_priority() {
        for bf16 in [false, true] {
            for phase in 0..if bf16 { 3 } else { 2 } {
                let mut module = decode_module(128, 128, 16, bf16);
                module.norm_weight.fill(f32::MAX);
                module.conv_weight =
                    [0.0, 0.0, 0.0, 1.0].repeat(module.geometry.conv_width().unwrap());
                let mut state = module.new_state().unwrap();
                state.recurrent.fill(0.125);
                let last = state.recurrent.len() - 128 * 128;
                state.recurrent[last..].fill(f32::MAX);
                let mut qkv = vec![1.0; module.geometry.conv_width().unwrap()];
                let z = vec![10.0; module.geometry.value_width().unwrap()];
                let b = vec![0.0; 16];
                let a = b.clone();
                if phase == 1 {
                    // A late decay failure precedes earlier state or gate failures.
                    module.a_log[15] = 88.0;
                    module.dt_bias[15] = 3.0;
                    state.recurrent[..128 * 128].fill(f32::MAX);
                } else if phase == 2 {
                    // Zero keys retain finite state. The last head overflows only its
                    // BF16 core cast, which precedes the first head's norm failure.
                    qkv[..module.geometry.key_width().unwrap()].fill(2.0);
                    qkv[module.geometry.key_width().unwrap()..].fill(0.0);
                    module.a_log.fill(-100.0);
                }
                let reference =
                    staged_recurrent_step(&module, &qkv, &z, &b, &a, &mut state.clone())
                        .unwrap_err();
                let actual = module
                    .recurrent_step(&qkv, &z, &b, &a, &mut state)
                    .unwrap_err();
                assert_eq!(actual.to_string(), reference.to_string());
                match phase {
                    0 => assert!(matches!(
                        reference,
                        DeltaNetError::NonFinite("recurrent state")
                    )),
                    1 => assert!(matches!(reference, DeltaNetError::NonFinite("decay log"))),
                    2 => assert!(
                        matches!(reference, DeltaNetError::Weight(_)),
                        "{reference:?}"
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn head_major_prefill_preserves_state_outputs_and_continuation_bits() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (keys, values) in [(128, 128), (65, 129)] {
            let geometry = DeltaNetGeometry {
                hidden_size: 2,
                num_key_heads: 2,
                num_value_heads: 8,
                key_head_dim: keys,
                value_head_dim: values,
                conv_kernel_size: 4,
                norm_eps: 1e-6,
            };
            let conv_width = geometry.conv_width().unwrap();
            let value_width = geometry.value_width().unwrap();
            let projection = |rows, cols, seed| {
                matrix(
                    rows,
                    cols,
                    (0..rows * cols)
                        .map(|i| ((i * 19 + seed) % 97) as f32 / 128.0 - 0.375)
                        .collect(),
                )
            };
            let module = GatedDeltaNet::new(
                geometry,
                projection(conv_width, 2, 1),
                projection(value_width, 2, 3),
                projection(8, 2, 5),
                projection(8, 2, 7),
                [0.125, -0.25, 0.375, 0.5].repeat(conv_width),
                vec![0.125; 8],
                vec![-0.25; 8],
                vec![0.875; values],
                projection(2, value_width, 11),
            )
            .unwrap();
            for bf16 in [false, true] {
                let module = with_dtype(module.clone(), bf16);
                for batch in [2, 3, 9] {
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
                    let expected = module
                        .forward_token(&[-0.25, 0.75], &mut sequential)
                        .unwrap();
                    let actual = module.forward_token(&[-0.25, 0.75], &mut batched).unwrap();
                    assert_eq!(bits(&actual), bits(&expected));
                    assert_eq!(bits(&batched.recurrent), bits(&sequential.recurrent));
                }
            }
        }
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
    fn head_state_updates_preserve_staged_fp32_and_f64_bits() {
        let bits = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let project = |state: &[f32], vector: &[f32], scale: f32, width: usize| {
            (0..width)
                .map(|column| {
                    vector.iter().enumerate().fold(-0.0f64, |sum, (row, &x)| {
                        sum + f64::from(state[row * width + column]) * f64::from(x * scale)
                    })
                })
                .collect::<Vec<_>>()
        };
        for (keys, width) in [
            (1, 1),
            (2, 3),
            (3, 4),
            (5, 7),
            (7, 8),
            (13, 15),
            (17, 16),
            (19, 17),
            (128, 128),
            (129, 257),
        ] {
            let key = (0..keys)
                .map(|i| ((i * 19 % 97) as f32 - 48.0) / 127.0)
                .collect::<Vec<_>>();
            let query = (0..keys)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0)
                .collect::<Vec<_>>();
            let delta = (0..width)
                .map(|i| ((i * 31 % 103) as f32 - 51.0) / 47.0)
                .collect::<Vec<_>>();
            let mut initial = (0..keys * width)
                .map(|i| ((i * 137 % 251) as f32 - 125.0) / 37.0)
                .collect::<Vec<_>>();
            for (value, boundary) in initial.iter_mut().zip([
                -0.0,
                0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                f32::MAX * 0.25,
                f32::MIN * 0.25,
            ]) {
                *value = boundary;
            }
            for decay in [-0.0, 0.0, 1e-7, 0.5, 0.991, 1.0] {
                for scale in [1.0, (keys as f32).sqrt().recip()] {
                    let mut expected = initial.clone();
                    let mut actual = expected.clone();
                    let mut sums = vec![f64::NAN; width];
                    for _ in 0..3 {
                        for value in &mut expected {
                            *value *= decay;
                        }
                        let reference = project(&expected, &key, 1.0, width);
                        decay_and_project_recurrent_state(&mut actual, decay, &key, &mut sums);
                        assert_eq!(bits(&actual), bits(&expected));
                        assert_eq!(
                            sums.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                        for (row, &key) in expected.chunks_exact_mut(width).zip(&key) {
                            for (value, &delta) in row.iter_mut().zip(&delta) {
                                *value += key * delta;
                            }
                        }
                        let reference = project(&expected, &query, scale, width);
                        update_and_project_recurrent_state(
                            &mut actual,
                            &key,
                            &delta,
                            &query,
                            scale,
                            &mut sums,
                        );
                        assert_eq!(bits(&actual), bits(&expected));
                        assert_eq!(
                            sums.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn recurrent_overflow_does_not_commit_parallel_head_updates() {
        let geometry = DeltaNetGeometry {
            hidden_size: 1,
            num_key_heads: 1,
            num_value_heads: 16,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 2,
            norm_eps: 1e-6,
        };
        let conv_width = geometry.conv_width().unwrap();
        let value_width = geometry.value_width().unwrap();
        let zeros = |rows, cols| DenseMatrix::new(rows, cols, vec![0.0; rows * cols]).unwrap();
        let module = GatedDeltaNet::new(
            geometry,
            DenseMatrix::new(conv_width, 1, vec![1.0; conv_width]).unwrap(),
            zeros(value_width, 1),
            zeros(geometry.num_value_heads, 1),
            zeros(geometry.num_value_heads, 1),
            vec![0.5; conv_width * geometry.conv_kernel_size],
            vec![0.0; geometry.num_value_heads],
            vec![0.0; geometry.num_value_heads],
            vec![1.0; geometry.value_head_dim],
            zeros(1, value_width),
        )
        .unwrap();
        let mut state = module.new_state().unwrap();
        state.conv.fill(0.5);
        state.recurrent.fill(0.125);
        // The last head's finite state overflows when its key projection narrows to F32.
        // Other heads can already have completed their updates when this is discovered.
        state.recurrent[(geometry.num_value_heads - 1) * 128 * 128..].fill(f32::MAX);
        state.tokens = 7;
        let before = state.clone();
        let mut workspace = DeltaNetWorkspace::default();
        for input in [&[1.0][..], &[1.0, 1.0][..]] {
            let error = module
                .forward_tokens_with_workspace(input, input.len(), &mut state, &mut workspace)
                .unwrap_err();
            assert!(matches!(error, DeltaNetError::NonFinite("recurrent state")));
            assert_eq!(state.tokens, before.tokens);
            assert_eq!(state.conv, before.conv);
            assert_eq!(state.recurrent, before.recurrent);

            // A later layer can reuse the partially updated transaction after a failure.
            // Its committed state must overwrite every failed head before any work resumes.
            let mut reference = before.clone();
            reference.recurrent.fill(0.125);
            let mut recovered = reference.clone();
            let expected = module
                .forward_tokens(input, input.len(), &mut reference)
                .unwrap();
            let actual = module
                .forward_tokens_with_workspace(input, input.len(), &mut recovered, &mut workspace)
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(recovered.tokens, reference.tokens);
            assert_eq!(recovered.conv, reference.conv);
            assert_eq!(recovered.recurrent, reference.recurrent);
        }

        // The final head succeeds on token one but overflows its decay on token two.
        // Other heads may already have processed the entire chunk when it fails.
        let mut late_failure = module;
        late_failure.a_log[geometry.num_value_heads - 1] = 2.0f32.ln();
        let mut decay_weights = vec![0.0; geometry.num_value_heads];
        decay_weights[geometry.num_value_heads - 1] = f32::MAX * 0.5;
        late_failure.in_proj_a = matrix(geometry.num_value_heads, 1, decay_weights).into();
        let mut batched = late_failure.new_state().unwrap();
        let before = batched.clone();
        let mut sequential = before.clone();
        late_failure.forward_token(&[1.0], &mut sequential).unwrap();
        assert!(matches!(
            late_failure.forward_token(&[2.0], &mut sequential),
            Err(DeltaNetError::NonFinite("decay log"))
        ));
        assert!(matches!(
            late_failure.forward_tokens_with_workspace(
                &[1.0, 2.0],
                2,
                &mut batched,
                &mut workspace
            ),
            Err(DeltaNetError::NonFinite("decay log"))
        ));
        assert_eq!(batched.tokens, before.tokens);
        assert_eq!(batched.conv, before.conv);
        assert_eq!(batched.recurrent, before.recurrent);
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
