//! Storage-independent scalar references for Kimi K3's two attention mechanisms.
//!
//! The checkpoint loader is deliberately absent from this module. Callers inject every matrix
//! projection through [`Projector`], which keeps the formulas usable with BF16, MXFP4, synthetic
//! test matrices, or a future device backend. Persistent sequence contents and positions are
//! updated transactionally. A failed cache reservation may retain harmless spare `Vec` capacity,
//! but never changes logical cache length or values.

use super::math::{kda_decay, kda_recurrent_step, l2_normalize_in_place, KimiK3MathError};
use std::fmt;

/// MLA cache vectors grow in bounded token chunks instead of `Vec`'s geometric policy.
pub const MLA_CACHE_GROWTH_TOKENS: usize = 256;

/// Logical matrix projections used by the scalar attention references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Projection {
    KdaQuery,
    KdaKey,
    KdaValue,
    KdaBeta,
    KdaDecayA,
    KdaDecayB,
    KdaOutputGate,
    KdaOutput,
    MlaQueryA,
    MlaQueryB,
    MlaKvA,
    MlaKvB,
    MlaOutputGate,
    MlaOutput,
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::KdaQuery => "KDA q_proj",
            Self::KdaKey => "KDA k_proj",
            Self::KdaValue => "KDA v_proj",
            Self::KdaBeta => "KDA b_proj",
            Self::KdaDecayA => "KDA f_a_proj",
            Self::KdaDecayB => "KDA f_b_proj",
            Self::KdaOutputGate => "KDA g_proj",
            Self::KdaOutput => "KDA o_proj",
            Self::MlaQueryA => "MLA q_a_proj",
            Self::MlaQueryB => "MLA q_b_proj",
            Self::MlaKvA => "MLA kv_a_proj_with_mqa",
            Self::MlaKvB => "MLA kv_b_proj",
            Self::MlaOutputGate => "MLA g_proj",
            Self::MlaOutput => "MLA o_proj",
        })
    }
}

/// Minimal matrix interface required by the attention references.
///
/// `expected_output` is supplied both as a useful allocation hint and as a contract. The caller
/// still validates the returned vector, so an erroneous backend cannot silently corrupt state.
/// Implementations must be deterministic and side-effect-free for a fixed projection and input:
/// bounded MLA deliberately invokes `MlaKvB` twice per cached source, once for scoring and once
/// for value accumulation.
pub trait Projector {
    type Error: fmt::Display;

    fn project(
        &mut self,
        projection: Projection,
        input: &[f32],
        expected_output: usize,
    ) -> Result<Vec<f32>, Self::Error>;
}

impl<F, E> Projector for F
where
    F: FnMut(Projection, &[f32], usize) -> Result<Vec<f32>, E>,
    E: fmt::Display,
{
    type Error = E;

    fn project(
        &mut self,
        projection: Projection,
        input: &[f32],
        expected_output: usize,
    ) -> Result<Vec<f32>, Self::Error> {
        self(projection, input, expected_output)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    InvalidGeometry {
        field: &'static str,
        reason: &'static str,
    },
    Shape {
        argument: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite {
        argument: &'static str,
        index: usize,
    },
    Overflow {
        expression: &'static str,
    },
    CachePosition {
        expected: usize,
        got: usize,
    },
    CacheCapacity {
        capacity: usize,
        position: usize,
    },
    Allocation {
        value: &'static str,
        additional_elements: usize,
        message: String,
    },
    Projection {
        projection: Projection,
        message: String,
    },
    Math(KimiK3MathError),
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry { field, reason } => {
                write!(f, "invalid Kimi K3 attention geometry `{field}`: {reason}")
            }
            Self::Shape {
                argument,
                expected,
                got,
            } => write!(
                f,
                "Kimi K3 attention `{argument}` has length {got}, expected {expected}"
            ),
            Self::NonFinite { argument, index } => write!(
                f,
                "Kimi K3 attention `{argument}`[{index}] is NaN or infinity"
            ),
            Self::Overflow { expression } => write!(
                f,
                "integer overflow while computing Kimi K3 attention {expression}"
            ),
            Self::CachePosition { expected, got } => write!(
                f,
                "Kimi K3 attention cache expects position {expected}, got {got}"
            ),
            Self::CacheCapacity { capacity, position } => write!(
                f,
                "Kimi K3 MLA cache position {position} exceeds capacity {capacity}"
            ),
            Self::Allocation {
                value,
                additional_elements,
                message,
            } => write!(
                f,
                "cannot reserve {additional_elements} additional F32 elements for Kimi K3 {value}: {message}"
            ),
            Self::Projection {
                projection,
                message,
            } => write!(f, "{projection} failed: {message}"),
            Self::Math(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for AttentionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KimiK3MathError> for AttentionError {
    fn from(value: KimiK3MathError) -> Self {
        Self::Math(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KdaGeometry {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_epsilon: f32,
    pub gate_lower_bound: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlaGeometry {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// The checkpoint calls this the RoPE width. Kimi K3 deliberately applies no rotation, so it
    /// is a shared, position-independent NoPE slot in this implementation.
    pub qk_nope_slot_dim: usize,
    pub value_head_dim: usize,
    pub rms_epsilon: f32,
}

/// Non-matrix KDA tensors. Matrix weights are supplied through [`Projector`].
#[derive(Debug, Clone, Copy)]
pub struct KdaParameters<'a> {
    /// Flattened `[num_heads * head_dim, conv_kernel_size]`, oldest tap first.
    pub query_conv: &'a [f32],
    pub key_conv: &'a [f32],
    pub value_conv: &'a [f32],
    /// The released checkpoint stores 128 entries for 96 heads. Entries after `num_heads` are
    /// padding and are validated but never used.
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    /// Head-local RMSNorm gain, shared by all heads.
    pub output_norm: &'a [f32],
}

/// Non-matrix MLA tensors. Matrix weights are supplied through [`Projector`].
#[derive(Debug, Clone, Copy)]
pub struct MlaParameters<'a> {
    pub query_a_norm: &'a [f32],
    pub kv_a_norm: &'a [f32],
}

/// Fixed-size per-sequence KDA state.
#[derive(Debug, Clone, PartialEq)]
pub struct KdaState {
    position: usize,
    recurrent: Vec<f32>,
    query_history: Vec<f32>,
    key_history: Vec<f32>,
    value_history: Vec<f32>,
}

impl KdaState {
    pub fn new(geometry: &KdaGeometry) -> Result<Self, AttentionError> {
        let sizes = validate_kda_geometry(geometry)?;
        validate_vec_len::<f32>(sizes.recurrent, "KDA recurrent state allocation")?;
        validate_vec_len::<f32>(sizes.history, "KDA convolution history allocation")?;
        Ok(Self {
            position: 0,
            recurrent: vec![0.0; sizes.recurrent],
            query_history: vec![0.0; sizes.history],
            key_history: vec![0.0; sizes.history],
            value_history: vec![0.0; sizes.history],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        geometry: &KdaGeometry,
        position: usize,
        recurrent: Vec<f32>,
        query_history: Vec<f32>,
        key_history: Vec<f32>,
        value_history: Vec<f32>,
    ) -> Result<Self, AttentionError> {
        let state = Self {
            position,
            recurrent,
            query_history,
            key_history,
            value_history,
        };
        validate_kda_state(&state, geometry, validate_kda_geometry(geometry)?)?;
        Ok(state)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn recurrent(&self) -> &[f32] {
        &self.recurrent
    }

    /// Raw, pre-convolution query projections in channel-major oldest-to-newest order.
    pub fn query_history(&self) -> &[f32] {
        &self.query_history
    }

    pub fn key_history(&self) -> &[f32] {
        &self.key_history
    }

    pub fn value_history(&self) -> &[f32] {
        &self.value_history
    }
}

/// Compressed MLA cache: one normalized KV latent and one shared unrotated slot per token.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaCache {
    capacity: usize,
    len: usize,
    latent: Vec<f32>,
    shared_nope_slots: Vec<f32>,
}

impl MlaCache {
    pub fn new(capacity: usize, geometry: &MlaGeometry) -> Result<Self, AttentionError> {
        let _ = validate_mla_geometry(geometry)?;
        let latent_capacity = checked_mul(
            capacity,
            geometry.kv_lora_rank,
            "cache capacity * kv_lora_rank",
        )?;
        let slot_capacity = checked_mul(
            capacity,
            geometry.qk_nope_slot_dim,
            "cache capacity * qk_nope_slot_dim",
        )?;
        validate_vec_len::<f32>(latent_capacity, "MLA latent cache address space")?;
        validate_vec_len::<f32>(slot_capacity, "MLA shared slot cache address space")?;
        Ok(Self {
            capacity,
            len: 0,
            // Capacity is a logical bound, not an eager multi-gigabyte allocation. The official
            // million-token context would otherwise reserve 576M floats per MLA layer up front.
            latent: Vec::new(),
            shared_nope_slots: Vec::new(),
        })
    }

    pub fn from_parts(
        capacity: usize,
        len: usize,
        latent: Vec<f32>,
        shared_nope_slots: Vec<f32>,
        geometry: &MlaGeometry,
    ) -> Result<Self, AttentionError> {
        let cache = Self {
            capacity,
            len,
            latent,
            shared_nope_slots,
        };
        validate_mla_cache(&cache, geometry, validate_mla_geometry(geometry)?)?;
        Ok(cache)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn normalized_latents(&self) -> &[f32] {
        &self.latent
    }

    pub(crate) fn truncate(&mut self, tokens: usize) -> Result<(), AttentionError> {
        if tokens > self.len {
            return Err(AttentionError::CachePosition {
                expected: self.len,
                got: tokens,
            });
        }
        if self.len != 0 {
            self.latent
                .truncate(tokens * (self.latent.len() / self.len));
            self.shared_nope_slots
                .truncate(tokens * (self.shared_nope_slots.len() / self.len));
        }
        self.len = tokens;
        Ok(())
    }

    pub fn shared_nope_slots(&self) -> &[f32] {
        &self.shared_nope_slots
    }

    /// Allocated latent-vector capacity in F32 elements, including bounded growth slack.
    pub fn allocated_latent_f32_capacity(&self) -> usize {
        self.latent.capacity()
    }

    /// Allocated shared-slot capacity in F32 elements, including bounded growth slack.
    pub fn allocated_nope_slot_f32_capacity(&self) -> usize {
        self.shared_nope_slots.capacity()
    }

    pub fn compressed_floats_per_token(
        &self,
        geometry: &MlaGeometry,
    ) -> Result<usize, AttentionError> {
        Ok(validate_mla_geometry(geometry)?.kv_a)
    }
}

#[derive(Debug, Clone, Copy)]
struct KdaSizes {
    channels: usize,
    history: usize,
    recurrent_per_head: usize,
    recurrent: usize,
}

#[derive(Debug, Clone, Copy)]
struct MlaSizes {
    query_head: usize,
    query: usize,
    kv_head: usize,
    kv: usize,
    gated_output: usize,
    kv_a: usize,
}

/// Variable F32 scratch used by the bounded two-pass MLA attention sweep.
///
/// Fixed-width query, context, and projection vectors do not grow with context length and are not
/// included. The only context-sized region is one score/probability per head and source. A single
/// source's expanded KV is reused across both passes instead of retaining every expanded source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaTwoPassScratch {
    pub score_probability_f32: usize,
    pub transient_expanded_kv_f32: usize,
}

impl MlaTwoPassScratch {
    pub fn total_f32(self) -> Result<usize, AttentionError> {
        self.score_probability_f32
            .checked_add(self.transient_expanded_kv_f32)
            .ok_or(AttentionError::Overflow {
                expression: "MLA score/probability + transient expanded KV scratch",
            })
    }

    pub fn total_bytes(self) -> Result<usize, AttentionError> {
        self.total_f32()?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(AttentionError::Overflow {
                expression: "MLA two-pass F32 scratch bytes",
            })
    }
}

/// Computes MLA's context-dependent two-pass F32 scratch without allocating it.
pub fn mla_two_pass_scratch(
    geometry: &MlaGeometry,
    source_count: usize,
) -> Result<MlaTwoPassScratch, AttentionError> {
    let sizes = validate_mla_geometry(geometry)?;
    require_nonzero("source_count", source_count)?;
    Ok(MlaTwoPassScratch {
        score_probability_f32: checked_mul(
            geometry.num_heads,
            source_count,
            "MLA num_heads * source_count score/probability scratch",
        )?,
        transient_expanded_kv_f32: sizes.kv,
    })
}

/// Maximum unused F32 capacity retained per MLA layer by the 256-token growth policy.
///
/// The final partial chunk is smaller, so this is a conservative planner margin. For the released
/// 512+64 geometry it is 147,456 F32 values (589,824 bytes) per layer, or 13.5 MiB over 24 layers.
pub fn mla_cache_growth_slack_f32_per_layer(
    geometry: &MlaGeometry,
) -> Result<usize, AttentionError> {
    let sizes = validate_mla_geometry(geometry)?;
    checked_mul(
        MLA_CACHE_GROWTH_TOKENS,
        sizes.kv_a,
        "MLA cache growth tokens * compressed F32 width",
    )
}

/// Applies one causal depthwise-convolution token and its fused SiLU.
///
/// `history` is channel-major `[channels, kernel_size - 1]` and holds raw projection
/// values, not prior convolved or activated values. Taps are ordered oldest to current. The
/// history is committed only if every output is finite.
pub fn causal_depthwise_conv_step(
    projected: &[f32],
    weights: &[f32],
    history: &mut [f32],
    channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, AttentionError> {
    require_nonzero("channels", channels)?;
    require_nonzero("kernel_size", kernel_size)?;
    expect_len("projected", channels, projected.len())?;
    let weight_len = checked_mul(channels, kernel_size, "channels * kernel_size")?;
    expect_len("convolution weights", weight_len, weights.len())?;
    let history_width = kernel_size - 1;
    let history_len = checked_mul(channels, history_width, "channels * (kernel_size - 1)")?;
    expect_len("convolution history", history_len, history.len())?;
    validate_finite("projected", projected)?;
    validate_finite("convolution weights", weights)?;
    validate_finite("convolution history", history)?;

    let mut next_history = history.to_vec();
    let mut output = Vec::with_capacity(channels);
    for channel in 0..channels {
        let weight = &weights[channel * kernel_size..(channel + 1) * kernel_size];
        let channel_history = &history[channel * history_width..(channel + 1) * history_width];
        let mut sum = 0.0f64;
        for (&value, &tap) in channel_history.iter().zip(weight) {
            sum += f64::from(value) * f64::from(tap);
            if !sum.is_finite() {
                return Err(AttentionError::NonFinite {
                    argument: "convolution accumulator",
                    index: channel,
                });
            }
        }
        sum += f64::from(projected[channel]) * f64::from(weight[history_width]);
        let sum = checked_f32("convolution accumulator", channel, sum)?;
        output.push(checked_f32(
            "convolution output",
            channel,
            f64::from(sum) * sigmoid(f64::from(sum)),
        )?);

        if history_width > 0 {
            let offset = channel * history_width;
            next_history[offset..offset + history_width].rotate_left(1);
            next_history[offset + history_width - 1] = projected[channel];
        }
    }

    history.copy_from_slice(&next_history);
    Ok(output)
}

/// Advances one token through Kimi Delta Attention.
#[allow(clippy::too_many_arguments)]
pub fn kda_step<P: Projector>(
    geometry: &KdaGeometry,
    parameters: KdaParameters<'_>,
    state: &mut KdaState,
    position: usize,
    hidden: &[f32],
    projector: &mut P,
) -> Result<Vec<f32>, AttentionError> {
    let sizes = validate_kda_geometry(geometry)?;
    validate_kda_parameters(parameters, geometry, sizes)?;
    validate_kda_state(state, geometry, sizes)?;
    expect_position(state.position, position)?;
    let next_position = position.checked_add(1).ok_or(AttentionError::Overflow {
        expression: "KDA cache position + 1",
    })?;
    expect_len("hidden", geometry.hidden_size, hidden.len())?;
    validate_finite("hidden", hidden)?;

    // Every mutation below targets this candidate. Even a final o_proj failure leaves the
    // caller's recurrent and convolution states untouched.
    let mut candidate = state.clone();
    let q_projected = project_checked(projector, Projection::KdaQuery, hidden, sizes.channels)?;
    let k_projected = project_checked(projector, Projection::KdaKey, hidden, sizes.channels)?;
    let v_projected = project_checked(projector, Projection::KdaValue, hidden, sizes.channels)?;
    let beta_logits = project_checked(projector, Projection::KdaBeta, hidden, geometry.num_heads)?;
    let decay_rank = project_checked(projector, Projection::KdaDecayA, hidden, geometry.head_dim)?;
    let z = project_checked(
        projector,
        Projection::KdaDecayB,
        &decay_rank,
        sizes.channels,
    )?;
    let output_gate =
        project_checked(projector, Projection::KdaOutputGate, hidden, sizes.channels)?;

    let mut query = causal_depthwise_conv_step(
        &q_projected,
        parameters.query_conv,
        &mut candidate.query_history,
        sizes.channels,
        geometry.conv_kernel_size,
    )?;
    let mut key = causal_depthwise_conv_step(
        &k_projected,
        parameters.key_conv,
        &mut candidate.key_history,
        sizes.channels,
        geometry.conv_kernel_size,
    )?;
    let value = causal_depthwise_conv_step(
        &v_projected,
        parameters.value_conv,
        &mut candidate.value_history,
        sizes.channels,
        geometry.conv_kernel_size,
    )?;

    for head in 0..geometry.num_heads {
        let start = head * geometry.head_dim;
        let end = start + geometry.head_dim;
        l2_normalize_in_place(&mut query[start..end], 1.0e-6)?;
        l2_normalize_in_place(&mut key[start..end], 1.0e-6)?;
    }
    let beta = beta_logits
        .iter()
        .map(|&value| sigmoid(f64::from(value)) as f32)
        .collect::<Vec<_>>();
    let decay = kda_decay(
        &z,
        parameters.dt_bias,
        &parameters.a_log[..geometry.num_heads],
        geometry.num_heads,
        geometry.head_dim,
        geometry.gate_lower_bound,
    )?;

    let mut recurrent_output = Vec::with_capacity(sizes.channels);
    for (head, &head_beta) in beta.iter().enumerate() {
        let vector_start = head * geometry.head_dim;
        let vector_end = vector_start + geometry.head_dim;
        let state_start = head * sizes.recurrent_per_head;
        let state_end = state_start + sizes.recurrent_per_head;
        recurrent_output.extend(kda_recurrent_step(
            &mut candidate.recurrent[state_start..state_end],
            &query[vector_start..vector_end],
            &key[vector_start..vector_end],
            &value[vector_start..vector_end],
            &decay.alpha[vector_start..vector_end],
            head_beta,
            geometry.head_dim,
            geometry.head_dim,
        )?);
    }

    for head in 0..geometry.num_heads {
        let start = head * geometry.head_dim;
        let end = start + geometry.head_dim;
        rms_norm_in_place(
            &mut recurrent_output[start..end],
            parameters.output_norm,
            geometry.rms_epsilon,
            "KDA head output",
        )?;
    }
    for (index, (output, &gate)) in recurrent_output.iter_mut().zip(&output_gate).enumerate() {
        *output = checked_f32(
            "KDA gated output",
            index,
            f64::from(*output) * sigmoid(f64::from(gate)),
        )?;
    }
    let output = project_checked(
        projector,
        Projection::KdaOutput,
        &recurrent_output,
        geometry.hidden_size,
    )?;

    candidate.position = next_position;
    *state = candidate;
    Ok(output)
}

/// Transactional multi-token convenience wrapper over [`kda_step`].
#[allow(clippy::too_many_arguments)]
pub fn kda_prefill<P: Projector>(
    geometry: &KdaGeometry,
    parameters: KdaParameters<'_>,
    state: &mut KdaState,
    start_position: usize,
    hidden: &[f32],
    token_count: usize,
    projector: &mut P,
) -> Result<Vec<f32>, AttentionError> {
    let sizes = validate_kda_geometry(geometry)?;
    validate_kda_parameters(parameters, geometry, sizes)?;
    validate_kda_state(state, geometry, sizes)?;
    let expected = checked_mul(
        token_count,
        geometry.hidden_size,
        "token_count * hidden_size",
    )?;
    expect_len("prefill hidden", expected, hidden.len())?;
    validate_finite("prefill hidden", hidden)?;
    expect_position(state.position, start_position)?;
    let mut candidate = state.clone();
    validate_vec_len::<f32>(expected, "KDA prefill output allocation")?;
    let mut output = Vec::with_capacity(expected);
    for token in 0..token_count {
        let position = start_position
            .checked_add(token)
            .ok_or(AttentionError::Overflow {
                expression: "KDA prefill start_position + token",
            })?;
        let input = &hidden[token * geometry.hidden_size..(token + 1) * geometry.hidden_size];
        output.extend(kda_step(
            geometry,
            parameters,
            &mut candidate,
            position,
            input,
            projector,
        )?);
    }
    *state = candidate;
    Ok(output)
}

/// Advances one token through gated NoPE MLA using a compressed latent cache.
#[allow(clippy::too_many_arguments)]
pub fn mla_step<P: Projector>(
    geometry: &MlaGeometry,
    parameters: MlaParameters<'_>,
    cache: &mut MlaCache,
    position: usize,
    hidden: &[f32],
    projector: &mut P,
) -> Result<Vec<f32>, AttentionError> {
    let sizes = validate_mla_geometry(geometry)?;
    validate_mla_parameters(parameters, geometry)?;
    validate_mla_cache(cache, geometry, sizes)?;
    expect_position(cache.len, position)?;
    if position >= cache.capacity {
        return Err(AttentionError::CacheCapacity {
            capacity: cache.capacity,
            position,
        });
    }
    let next_position = position.checked_add(1).ok_or(AttentionError::Overflow {
        expression: "MLA cache position + 1",
    })?;
    expect_len("hidden", geometry.hidden_size, hidden.len())?;
    validate_finite("hidden", hidden)?;

    let query_rank = project_checked(
        projector,
        Projection::MlaQueryA,
        hidden,
        geometry.q_lora_rank,
    )?;
    let mut query_rank = query_rank;
    rms_norm_in_place(
        &mut query_rank,
        parameters.query_a_norm,
        geometry.rms_epsilon,
        "MLA query A",
    )?;
    let query = project_checked(projector, Projection::MlaQueryB, &query_rank, sizes.query)?;

    let compressed = project_checked(projector, Projection::MlaKvA, hidden, sizes.kv_a)?;
    let mut current_latent = compressed[..geometry.kv_lora_rank].to_vec();
    let current_slot = &compressed[geometry.kv_lora_rank..];
    rms_norm_in_place(
        &mut current_latent,
        parameters.kv_a_norm,
        geometry.rms_epsilon,
        "MLA KV A",
    )?;

    // The persistent cache is exactly kv_lora_rank + qk_nope_slot_dim floats per token
    // (512 + 64 in Kimi K3). A two-pass sweep bounds transient expansion to one source:
    // pass one computes all head/source probabilities, pass two re-runs kv_b and accumulates
    // values. This deliberately spends twice the historical kv_b work to avoid an
    // O(context * heads * expanded_kv_width) allocation.
    let source_count = next_position;
    let scale = (sizes.query_head as f64).sqrt().recip();
    if !scale.is_finite() {
        return Err(AttentionError::NonFinite {
            argument: "MLA attention scale",
            index: 0,
        });
    }
    let scratch = mla_two_pass_scratch(geometry, source_count)?;
    validate_vec_len::<f32>(
        scratch.score_probability_f32,
        "MLA score/probability allocation",
    )?;
    validate_vec_len::<f32>(sizes.kv, "MLA transient expanded KV allocation")?;
    let mut score_probability = vec![0.0f32; scratch.score_probability_f32];

    // Pass one: expand one source, score it against every head, then immediately discard it.
    for source in 0..source_count {
        let latent = if source == position {
            &current_latent
        } else {
            let start = source * geometry.kv_lora_rank;
            &cache.latent[start..start + geometry.kv_lora_rank]
        };
        let expanded = project_checked(projector, Projection::MlaKvB, latent, sizes.kv)?;
        let slot = if source == position {
            current_slot
        } else {
            let start = source * geometry.qk_nope_slot_dim;
            &cache.shared_nope_slots[start..start + geometry.qk_nope_slot_dim]
        };
        for head in 0..geometry.num_heads {
            let query_start = head * sizes.query_head;
            let query_head = &query[query_start..query_start + sizes.query_head];
            let expanded_start = head * sizes.kv_head;
            let key = &expanded[expanded_start..expanded_start + geometry.qk_nope_head_dim];
            let mut dot = 0.0f64;
            for (&q, &k) in query_head[..geometry.qk_nope_head_dim].iter().zip(key) {
                dot += f64::from(q) * f64::from(k);
            }
            for (&q, &k) in query_head[geometry.qk_nope_head_dim..].iter().zip(slot) {
                dot += f64::from(q) * f64::from(k);
            }
            let index = head * source_count + source;
            score_probability[index] = checked_f32("MLA attention score", index, dot * scale)?;
        }
    }
    for head_scores in score_probability.chunks_exact_mut(source_count) {
        softmax_f32_in_place(head_scores)?;
    }

    // Pass two: repeat each kv_b expansion and accumulate only its value slice. The probability
    // matrix is retained, but no historical expanded K/V survives an iteration.
    validate_vec_len::<f64>(sizes.gated_output, "MLA context allocation")?;
    let mut context_accumulator = vec![0.0f64; sizes.gated_output];
    for source in 0..source_count {
        let latent = if source == position {
            &current_latent
        } else {
            let start = source * geometry.kv_lora_rank;
            &cache.latent[start..start + geometry.kv_lora_rank]
        };
        let expanded = project_checked(projector, Projection::MlaKvB, latent, sizes.kv)?;
        for head in 0..geometry.num_heads {
            let probability = f64::from(score_probability[head * source_count + source]);
            let output_start = head * geometry.value_head_dim;
            let value_start = head * sizes.kv_head + geometry.qk_nope_head_dim;
            let value = &expanded[value_start..value_start + geometry.value_head_dim];
            for (channel, &element) in value.iter().enumerate() {
                let index = output_start + channel;
                context_accumulator[index] = checked_f64(
                    "MLA attention context",
                    index,
                    context_accumulator[index] + probability * f64::from(element),
                )?;
            }
        }
    }
    drop(score_probability);

    // Gated MLA has no post-attention norm: the sigmoid gate is applied directly before o_proj.
    let mut context = context_accumulator
        .into_iter()
        .enumerate()
        .map(|(index, value)| checked_f32("MLA attention context", index, value))
        .collect::<Result<Vec<_>, _>>()?;
    let gate = project_checked(
        projector,
        Projection::MlaOutputGate,
        hidden,
        sizes.gated_output,
    )?;
    for (index, (output, &gate)) in context.iter_mut().zip(&gate).enumerate() {
        *output = checked_f32(
            "MLA gated output",
            index,
            f64::from(*output) * sigmoid(f64::from(gate)),
        )?;
    }
    let output = project_checked(
        projector,
        Projection::MlaOutput,
        &context,
        geometry.hidden_size,
    )?;

    // Reserve both vectors before changing either logical length. If the second reservation fails,
    // the first vector may retain extra capacity, but cache position, lengths, and contents remain
    // unchanged. Cloning all historical cache data merely to roll back capacity would make decode
    // O(context^2) in memory traffic.
    reserve_mla_cache_append_capacity(cache, geometry)?;
    cache.latent.extend_from_slice(&current_latent);
    cache.shared_nope_slots.extend_from_slice(current_slot);
    cache.len += 1;
    Ok(output)
}

/// Transactional causal prefill over [`mla_step`].
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill<P: Projector>(
    geometry: &MlaGeometry,
    parameters: MlaParameters<'_>,
    cache: &mut MlaCache,
    start_position: usize,
    hidden: &[f32],
    token_count: usize,
    projector: &mut P,
) -> Result<Vec<f32>, AttentionError> {
    let sizes = validate_mla_geometry(geometry)?;
    validate_mla_parameters(parameters, geometry)?;
    validate_mla_cache(cache, geometry, sizes)?;
    let expected = checked_mul(
        token_count,
        geometry.hidden_size,
        "token_count * hidden_size",
    )?;
    expect_len("prefill hidden", expected, hidden.len())?;
    validate_finite("prefill hidden", hidden)?;
    expect_position(cache.len, start_position)?;
    let mut candidate = cache.clone();
    validate_vec_len::<f32>(expected, "MLA prefill output allocation")?;
    let mut output = Vec::with_capacity(expected);
    for token in 0..token_count {
        let position = start_position
            .checked_add(token)
            .ok_or(AttentionError::Overflow {
                expression: "MLA prefill start_position + token",
            })?;
        let input = &hidden[token * geometry.hidden_size..(token + 1) * geometry.hidden_size];
        output.extend(mla_step(
            geometry,
            parameters,
            &mut candidate,
            position,
            input,
            projector,
        )?);
    }
    *cache = candidate;
    Ok(output)
}

fn validate_kda_geometry(geometry: &KdaGeometry) -> Result<KdaSizes, AttentionError> {
    require_nonzero("hidden_size", geometry.hidden_size)?;
    require_nonzero("num_heads", geometry.num_heads)?;
    require_nonzero("head_dim", geometry.head_dim)?;
    require_nonzero("conv_kernel_size", geometry.conv_kernel_size)?;
    require_positive("rms_epsilon", geometry.rms_epsilon)?;
    if !geometry.gate_lower_bound.is_finite() || geometry.gate_lower_bound >= 0.0 {
        return Err(AttentionError::InvalidGeometry {
            field: "gate_lower_bound",
            reason: "must be finite and strictly negative",
        });
    }
    let channels = checked_mul(
        geometry.num_heads,
        geometry.head_dim,
        "num_heads * head_dim",
    )?;
    let history = checked_mul(
        channels,
        geometry.conv_kernel_size - 1,
        "channels * (conv_kernel_size - 1)",
    )?;
    let recurrent_per_head =
        checked_mul(geometry.head_dim, geometry.head_dim, "head_dim * head_dim")?;
    let recurrent = checked_mul(
        geometry.num_heads,
        recurrent_per_head,
        "num_heads * head_dim * head_dim",
    )?;
    Ok(KdaSizes {
        channels,
        history,
        recurrent_per_head,
        recurrent,
    })
}

fn validate_mla_geometry(geometry: &MlaGeometry) -> Result<MlaSizes, AttentionError> {
    require_nonzero("hidden_size", geometry.hidden_size)?;
    require_nonzero("num_heads", geometry.num_heads)?;
    require_nonzero("q_lora_rank", geometry.q_lora_rank)?;
    require_nonzero("kv_lora_rank", geometry.kv_lora_rank)?;
    require_nonzero("qk_nope_head_dim", geometry.qk_nope_head_dim)?;
    require_nonzero("qk_nope_slot_dim", geometry.qk_nope_slot_dim)?;
    require_nonzero("value_head_dim", geometry.value_head_dim)?;
    require_positive("rms_epsilon", geometry.rms_epsilon)?;
    let query_head = checked_add(
        geometry.qk_nope_head_dim,
        geometry.qk_nope_slot_dim,
        "qk_nope_head_dim + qk_nope_slot_dim",
    )?;
    let query = checked_mul(geometry.num_heads, query_head, "num_heads * query_head_dim")?;
    let kv_head = checked_add(
        geometry.qk_nope_head_dim,
        geometry.value_head_dim,
        "qk_nope_head_dim + value_head_dim",
    )?;
    let kv = checked_mul(geometry.num_heads, kv_head, "num_heads * KV head width")?;
    let gated_output = checked_mul(
        geometry.num_heads,
        geometry.value_head_dim,
        "num_heads * value_head_dim",
    )?;
    let kv_a = checked_add(
        geometry.kv_lora_rank,
        geometry.qk_nope_slot_dim,
        "kv_lora_rank + qk_nope_slot_dim",
    )?;
    Ok(MlaSizes {
        query_head,
        query,
        kv_head,
        kv,
        gated_output,
        kv_a,
    })
}

fn validate_kda_parameters(
    parameters: KdaParameters<'_>,
    geometry: &KdaGeometry,
    sizes: KdaSizes,
) -> Result<(), AttentionError> {
    let conv_len = checked_mul(
        sizes.channels,
        geometry.conv_kernel_size,
        "KDA channels * conv_kernel_size",
    )?;
    for (name, values) in [
        ("query_conv", parameters.query_conv),
        ("key_conv", parameters.key_conv),
        ("value_conv", parameters.value_conv),
    ] {
        expect_len(name, conv_len, values.len())?;
        validate_finite(name, values)?;
    }
    if parameters.a_log.len() < geometry.num_heads {
        return Err(AttentionError::Shape {
            argument: "a_log",
            expected: geometry.num_heads,
            got: parameters.a_log.len(),
        });
    }
    validate_finite("a_log", parameters.a_log)?;
    expect_len("dt_bias", sizes.channels, parameters.dt_bias.len())?;
    validate_finite("dt_bias", parameters.dt_bias)?;
    expect_len(
        "output_norm",
        geometry.head_dim,
        parameters.output_norm.len(),
    )?;
    validate_finite("output_norm", parameters.output_norm)?;
    Ok(())
}

fn validate_mla_parameters(
    parameters: MlaParameters<'_>,
    geometry: &MlaGeometry,
) -> Result<(), AttentionError> {
    expect_len(
        "query_a_norm",
        geometry.q_lora_rank,
        parameters.query_a_norm.len(),
    )?;
    validate_finite("query_a_norm", parameters.query_a_norm)?;
    expect_len(
        "kv_a_norm",
        geometry.kv_lora_rank,
        parameters.kv_a_norm.len(),
    )?;
    validate_finite("kv_a_norm", parameters.kv_a_norm)?;
    Ok(())
}

fn validate_kda_state(
    state: &KdaState,
    _geometry: &KdaGeometry,
    sizes: KdaSizes,
) -> Result<(), AttentionError> {
    expect_len(
        "KDA recurrent state",
        sizes.recurrent,
        state.recurrent.len(),
    )?;
    expect_len(
        "KDA query convolution history",
        sizes.history,
        state.query_history.len(),
    )?;
    expect_len(
        "KDA key convolution history",
        sizes.history,
        state.key_history.len(),
    )?;
    expect_len(
        "KDA value convolution history",
        sizes.history,
        state.value_history.len(),
    )?;
    validate_finite("KDA recurrent state", &state.recurrent)?;
    validate_finite("KDA query convolution history", &state.query_history)?;
    validate_finite("KDA key convolution history", &state.key_history)?;
    validate_finite("KDA value convolution history", &state.value_history)?;
    Ok(())
}

fn validate_mla_cache(
    cache: &MlaCache,
    geometry: &MlaGeometry,
    _sizes: MlaSizes,
) -> Result<(), AttentionError> {
    if cache.len > cache.capacity {
        return Err(AttentionError::CacheCapacity {
            capacity: cache.capacity,
            position: cache.len,
        });
    }
    let latent_capacity = checked_mul(
        cache.capacity,
        geometry.kv_lora_rank,
        "cache capacity * kv_lora_rank",
    )?;
    let slot_capacity = checked_mul(
        cache.capacity,
        geometry.qk_nope_slot_dim,
        "cache capacity * qk_nope_slot_dim",
    )?;
    validate_vec_len::<f32>(latent_capacity, "MLA latent cache address space")?;
    validate_vec_len::<f32>(slot_capacity, "MLA shared slot cache address space")?;
    let latent_len = checked_mul(cache.len, geometry.kv_lora_rank, "cache len * kv_lora_rank")?;
    let slot_len = checked_mul(
        cache.len,
        geometry.qk_nope_slot_dim,
        "cache len * qk_nope_slot_dim",
    )?;
    expect_len("MLA cached latents", latent_len, cache.latent.len())?;
    expect_len(
        "MLA cached shared NoPE slots",
        slot_len,
        cache.shared_nope_slots.len(),
    )?;
    validate_finite("MLA cached latents", &cache.latent)?;
    validate_finite("MLA cached shared NoPE slots", &cache.shared_nope_slots)?;
    Ok(())
}

fn reserve_mla_cache_append_capacity(
    cache: &mut MlaCache,
    geometry: &MlaGeometry,
) -> Result<(), AttentionError> {
    let remaining_tokens =
        cache
            .capacity
            .checked_sub(cache.len)
            .ok_or(AttentionError::CacheCapacity {
                capacity: cache.capacity,
                position: cache.len,
            })?;
    if remaining_tokens == 0 {
        return Err(AttentionError::CacheCapacity {
            capacity: cache.capacity,
            position: cache.len,
        });
    }
    let chunk_tokens = remaining_tokens.min(MLA_CACHE_GROWTH_TOKENS);
    let latent_chunk = checked_mul(
        chunk_tokens,
        geometry.kv_lora_rank,
        "MLA cache growth tokens * kv_lora_rank",
    )?;
    let slot_chunk = checked_mul(
        chunk_tokens,
        geometry.qk_nope_slot_dim,
        "MLA cache growth tokens * qk_nope_slot_dim",
    )?;
    let latent_required = checked_add(
        cache.latent.len(),
        geometry.kv_lora_rank,
        "MLA latent cache length + one token",
    )?;
    let slot_required = checked_add(
        cache.shared_nope_slots.len(),
        geometry.qk_nope_slot_dim,
        "MLA shared slot cache length + one token",
    )?;
    let latent_additional = usize::from(cache.latent.capacity() < latent_required) * latent_chunk;
    let slot_additional =
        usize::from(cache.shared_nope_slots.capacity() < slot_required) * slot_chunk;
    try_reserve_mla_cache_vectors(cache, latent_additional, slot_additional)
}

fn try_reserve_mla_cache_vectors(
    cache: &mut MlaCache,
    latent_additional: usize,
    slot_additional: usize,
) -> Result<(), AttentionError> {
    if latent_additional > 0 {
        cache
            .latent
            .try_reserve_exact(latent_additional)
            .map_err(|error| AttentionError::Allocation {
                value: "MLA normalized latent cache",
                additional_elements: latent_additional,
                message: error.to_string(),
            })?;
    }
    if slot_additional > 0 {
        cache
            .shared_nope_slots
            .try_reserve_exact(slot_additional)
            .map_err(|error| AttentionError::Allocation {
                value: "MLA shared NoPE slot cache",
                additional_elements: slot_additional,
                message: error.to_string(),
            })?;
    }
    Ok(())
}

fn project_checked<P: Projector>(
    projector: &mut P,
    projection: Projection,
    input: &[f32],
    expected_output: usize,
) -> Result<Vec<f32>, AttentionError> {
    validate_finite("projection input", input)?;
    let output = projector
        .project(projection, input, expected_output)
        .map_err(|error| AttentionError::Projection {
            projection,
            message: error.to_string(),
        })?;
    if output.len() != expected_output {
        return Err(AttentionError::Shape {
            argument: "projection output",
            expected: expected_output,
            got: output.len(),
        });
    }
    validate_finite("projection output", &output)?;
    Ok(output)
}

fn rms_norm_in_place(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
    argument: &'static str,
) -> Result<(), AttentionError> {
    expect_len(argument, weight.len(), values.len())?;
    require_positive("rms_epsilon", epsilon)?;
    validate_finite(argument, values)?;
    validate_finite("RMSNorm weight", weight)?;
    let mut square_sum = 0.0f64;
    for &value in values.iter() {
        square_sum += f64::from(value) * f64::from(value);
        if !square_sum.is_finite() {
            return Err(AttentionError::NonFinite { argument, index: 0 });
        }
    }
    let inverse_rms = (square_sum / values.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    let mut candidate = Vec::with_capacity(values.len());
    for (index, (&value, &gain)) in values.iter().zip(weight).enumerate() {
        candidate.push(checked_f32(
            argument,
            index,
            f64::from(value) * inverse_rms * f64::from(gain),
        )?);
    }
    values.copy_from_slice(&candidate);
    Ok(())
}

fn softmax_f32_in_place(values: &mut [f32]) -> Result<(), AttentionError> {
    if values.is_empty() {
        return Err(AttentionError::InvalidGeometry {
            field: "softmax source count",
            reason: "must be non-zero",
        });
    }
    validate_finite("MLA softmax scores", values)?;
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0f64;
    for (index, value) in values.iter_mut().enumerate() {
        *value = checked_f32(
            "MLA softmax exponential",
            index,
            (f64::from(*value) - f64::from(maximum)).exp(),
        )?;
        denominator += f64::from(*value);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(AttentionError::NonFinite {
            argument: "MLA softmax denominator",
            index: 0,
        });
    }
    for (index, value) in values.iter_mut().enumerate() {
        *value = checked_f32(
            "MLA softmax probability",
            index,
            f64::from(*value) / denominator,
        )?;
    }
    Ok(())
}

fn expect_position(expected: usize, got: usize) -> Result<(), AttentionError> {
    if expected == got {
        Ok(())
    } else {
        Err(AttentionError::CachePosition { expected, got })
    }
}

fn expect_len(argument: &'static str, expected: usize, got: usize) -> Result<(), AttentionError> {
    if expected == got {
        Ok(())
    } else {
        Err(AttentionError::Shape {
            argument,
            expected,
            got,
        })
    }
}

fn validate_finite(argument: &'static str, values: &[f32]) -> Result<(), AttentionError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(AttentionError::NonFinite { argument, index })
    } else {
        Ok(())
    }
}

fn require_nonzero(field: &'static str, value: usize) -> Result<(), AttentionError> {
    if value == 0 {
        Err(AttentionError::InvalidGeometry {
            field,
            reason: "must be non-zero",
        })
    } else {
        Ok(())
    }
}

fn require_positive(field: &'static str, value: f32) -> Result<(), AttentionError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AttentionError::InvalidGeometry {
            field,
            reason: "must be finite and strictly positive",
        })
    }
}

fn checked_mul(
    left: usize,
    right: usize,
    expression: &'static str,
) -> Result<usize, AttentionError> {
    left.checked_mul(right)
        .ok_or(AttentionError::Overflow { expression })
}

fn checked_add(
    left: usize,
    right: usize,
    expression: &'static str,
) -> Result<usize, AttentionError> {
    left.checked_add(right)
        .ok_or(AttentionError::Overflow { expression })
}

fn validate_vec_len<T>(length: usize, expression: &'static str) -> Result<(), AttentionError> {
    let maximum = (isize::MAX as usize) / std::mem::size_of::<T>();
    if length <= maximum {
        Ok(())
    } else {
        Err(AttentionError::Overflow { expression })
    }
}

fn checked_f32(argument: &'static str, index: usize, value: f64) -> Result<f32, AttentionError> {
    let narrowed = value as f32;
    if value.is_finite() && narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(AttentionError::NonFinite { argument, index })
    }
}

fn checked_f64(argument: &'static str, index: usize, value: f64) -> Result<f64, AttentionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AttentionError::NonFinite { argument, index })
    }
}

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
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    #[test]
    fn annotated_closures_can_serve_as_projectors() {
        let mut projector = |_projection: Projection,
                             input: &[f32],
                             expected: usize|
         -> Result<Vec<f32>, &'static str> {
            assert_eq!(input, &[2.0]);
            Ok(vec![3.0; expected])
        };
        assert_eq!(
            Projector::project(&mut projector, Projection::KdaQuery, &[2.0], 2).unwrap(),
            vec![3.0, 3.0]
        );
    }

    #[test]
    fn short_convolution_uses_oldest_to_current_raw_history() {
        let mut history = vec![1.0, 2.0, 5.0, 6.0];
        let output = causal_depthwise_conv_step(
            &[3.0, 4.0],
            &[1.0, 10.0, 100.0, -1.0, 2.0, 0.5],
            &mut history,
            2,
            3,
        )
        .unwrap();
        assert_close(output[0], 321.0, 1.0e-5);
        assert_close(output[1], 9.0 * (1.0 / (1.0 + (-9.0f32).exp())), 1.0e-6);
        assert_eq!(history, vec![2.0, 3.0, 6.0, 4.0]);

        let before = history.clone();
        assert!(causal_depthwise_conv_step(
            &[f32::NAN, 1.0],
            &[1.0, 10.0, 100.0, -1.0, 2.0, 0.5],
            &mut history,
            2,
            3,
        )
        .is_err());
        assert_eq!(history, before);
    }

    #[derive(Clone)]
    struct TinyKdaProjector {
        fail_on: Option<Projection>,
    }

    impl Projector for TinyKdaProjector {
        type Error = &'static str;

        fn project(
            &mut self,
            projection: Projection,
            input: &[f32],
            expected_output: usize,
        ) -> Result<Vec<f32>, Self::Error> {
            if self.fail_on == Some(projection) {
                return Err("injected failure");
            }
            let output = match projection {
                Projection::KdaQuery | Projection::KdaKey | Projection::KdaValue => {
                    vec![input[0]]
                }
                Projection::KdaBeta
                | Projection::KdaDecayA
                | Projection::KdaDecayB
                | Projection::KdaOutputGate => vec![0.0],
                Projection::KdaOutput => vec![input[0]],
                _ => return Err("unexpected projection"),
            };
            assert_eq!(output.len(), expected_output);
            Ok(output)
        }
    }

    fn tiny_kda_geometry(kernel: usize) -> KdaGeometry {
        KdaGeometry {
            hidden_size: 1,
            num_heads: 1,
            head_dim: 1,
            conv_kernel_size: kernel,
            rms_epsilon: 1.0e-5,
            gate_lower_bound: -5.0,
        }
    }

    fn tiny_kda_parameters(kernel: usize) -> KdaParameters<'static> {
        let conv = match kernel {
            1 => &[1.0][..],
            3 => &[0.0, 0.0, 1.0][..],
            _ => panic!("unsupported test kernel"),
        };
        KdaParameters {
            query_conv: conv,
            key_conv: conv,
            value_conv: conv,
            // Deliberately padded: only the first head entry is consumed.
            a_log: &[0.0, 17.0],
            dt_bias: &[0.0],
            output_norm: &[1.0],
        }
    }

    #[test]
    fn kda_first_token_matches_scalar_formula() {
        let geometry = tiny_kda_geometry(1);
        let mut state = KdaState::new(&geometry).unwrap();
        let mut projector = TinyKdaProjector { fail_on: None };
        let output = kda_step(
            &geometry,
            tiny_kda_parameters(1),
            &mut state,
            0,
            &[1.0],
            &mut projector,
        )
        .unwrap();

        let conv = 1.0f64 / (1.0 + (-1.0f64).exp());
        let normalized = conv / (conv * conv + 1.0e-6).sqrt();
        let recurrent = normalized * (0.5 * conv * normalized);
        let normed = recurrent / (recurrent * recurrent + 1.0e-5).sqrt();
        assert_close(output[0], (0.5 * normed) as f32, 2.0e-6);
        assert_eq!(state.position(), 1);
    }

    #[test]
    fn kda_history_order_and_late_failure_are_transactional() {
        let geometry = tiny_kda_geometry(3);
        let parameters = tiny_kda_parameters(3);
        let mut state = KdaState::new(&geometry).unwrap();
        let mut projector = TinyKdaProjector { fail_on: None };
        kda_step(&geometry, parameters, &mut state, 0, &[1.0], &mut projector).unwrap();
        kda_step(&geometry, parameters, &mut state, 1, &[2.0], &mut projector).unwrap();
        assert_eq!(state.query_history(), &[1.0, 2.0]);
        assert_eq!(state.key_history(), &[1.0, 2.0]);
        assert_eq!(state.value_history(), &[1.0, 2.0]);

        let before = state.clone();
        let mut failing = TinyKdaProjector {
            fail_on: Some(Projection::KdaOutput),
        };
        let error =
            kda_step(&geometry, parameters, &mut state, 2, &[3.0], &mut failing).unwrap_err();
        assert!(matches!(
            error,
            AttentionError::Projection {
                projection: Projection::KdaOutput,
                ..
            }
        ));
        assert_eq!(state, before);

        assert!(kda_step(&geometry, parameters, &mut state, 9, &[3.0], &mut projector,).is_err());
        assert_eq!(state, before);
        assert!(kda_step(
            &geometry,
            parameters,
            &mut state,
            2,
            &[f32::INFINITY],
            &mut projector,
        )
        .is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn kda_prefill_matches_incremental_state_and_outputs() {
        let geometry = tiny_kda_geometry(3);
        let parameters = tiny_kda_parameters(3);
        let tokens = [1.0, 2.0, -0.5];

        let mut prefill_state = KdaState::new(&geometry).unwrap();
        let mut prefill_projector = TinyKdaProjector { fail_on: None };
        let prefill = kda_prefill(
            &geometry,
            parameters,
            &mut prefill_state,
            0,
            &tokens,
            tokens.len(),
            &mut prefill_projector,
        )
        .unwrap();

        let mut incremental_state = KdaState::new(&geometry).unwrap();
        let mut incremental_projector = TinyKdaProjector { fail_on: None };
        let mut incremental = Vec::new();
        for (position, token) in tokens.into_iter().enumerate() {
            incremental.extend(
                kda_step(
                    &geometry,
                    parameters,
                    &mut incremental_state,
                    position,
                    &[token],
                    &mut incremental_projector,
                )
                .unwrap(),
            );
        }
        assert_eq!(prefill, incremental);
        assert_eq!(prefill_state, incremental_state);
    }

    #[derive(Clone)]
    struct TinyMlaProjector {
        fail_on: Option<Projection>,
        kv_b_calls: usize,
    }

    impl Projector for TinyMlaProjector {
        type Error = &'static str;

        fn project(
            &mut self,
            projection: Projection,
            input: &[f32],
            expected_output: usize,
        ) -> Result<Vec<f32>, Self::Error> {
            if self.fail_on == Some(projection) {
                return Err("injected failure");
            }
            if projection == Projection::MlaKvB {
                self.kv_b_calls += 1;
            }
            let output = match projection {
                Projection::MlaQueryA | Projection::MlaQueryB => input.to_vec(),
                Projection::MlaKvA => vec![input[0], input[1], input[1]],
                Projection::MlaKvB => vec![input[0], input[1]],
                Projection::MlaOutputGate => vec![0.0],
                Projection::MlaOutput => vec![input[0], -input[0]],
                _ => return Err("unexpected projection"),
            };
            assert_eq!(output.len(), expected_output);
            Ok(output)
        }
    }

    fn tiny_mla_geometry() -> MlaGeometry {
        MlaGeometry {
            hidden_size: 2,
            num_heads: 1,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 1,
            qk_nope_slot_dim: 1,
            value_head_dim: 1,
            rms_epsilon: 1.0e-6,
        }
    }

    fn official_mla_geometry() -> MlaGeometry {
        MlaGeometry {
            hidden_size: 7_168,
            num_heads: 96,
            q_lora_rank: 1_536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_nope_slot_dim: 64,
            value_head_dim: 128,
            rms_epsilon: 1.0e-6,
        }
    }

    fn tiny_mla_parameters() -> MlaParameters<'static> {
        MlaParameters {
            query_a_norm: &[1.0, 1.0],
            kv_a_norm: &[1.0, 1.0],
        }
    }

    #[test]
    fn mla_one_token_is_hand_calculable_and_cache_is_compressed() {
        let geometry = tiny_mla_geometry();
        let mut cache = MlaCache::new(4, &geometry).unwrap();
        let mut projector = TinyMlaProjector {
            fail_on: None,
            kv_b_calls: 0,
        };
        let output = mla_step(
            &geometry,
            tiny_mla_parameters(),
            &mut cache,
            0,
            &[3.0, 4.0],
            &mut projector,
        )
        .unwrap();
        let inverse_rms = (12.5f64 + 1.0e-6).sqrt().recip();
        let expected = (4.0 * inverse_rms * 0.5) as f32;
        assert_close(output[0], expected, 1.0e-6);
        assert_close(output[1], -expected, 1.0e-6);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.normalized_latents().len(), 2);
        assert_eq!(cache.shared_nope_slots(), &[4.0]);
        assert_eq!(cache.compressed_floats_per_token(&geometry).unwrap(), 3);
        assert_eq!(projector.kv_b_calls, 2);

        let official = official_mla_geometry();
        assert_eq!(
            MlaCache::new(0, &official)
                .unwrap()
                .compressed_floats_per_token(&official)
                .unwrap(),
            576
        );
    }

    #[test]
    fn mla_two_pass_scratch_stays_bounded_and_large_capacity_is_lazy() {
        let geometry = official_mla_geometry();
        let cache = MlaCache::new(1_048_576, &geometry).unwrap();
        assert_eq!(cache.capacity(), 1_048_576);
        assert!(cache.is_empty());
        assert_eq!(cache.latent.capacity(), 0);
        assert_eq!(cache.shared_nope_slots.capacity(), 0);

        let scratch_32k = mla_two_pass_scratch(&geometry, 32_768).unwrap();
        assert_eq!(scratch_32k.score_probability_f32, 3_145_728);
        assert_eq!(scratch_32k.transient_expanded_kv_f32, 24_576);
        assert_eq!(scratch_32k.total_bytes().unwrap(), 12_681_216);

        let scratch_max = mla_two_pass_scratch(&geometry, 1_048_576).unwrap();
        assert_eq!(scratch_max.score_probability_f32, 100_663_296);
        assert_eq!(scratch_max.transient_expanded_kv_f32, 24_576);
        assert_eq!(scratch_max.total_bytes().unwrap(), 402_751_488);
        assert!(scratch_max.total_bytes().unwrap() < 512 * 1024 * 1024);
        let per_layer_slack = mla_cache_growth_slack_f32_per_layer(&geometry).unwrap();
        assert_eq!(per_layer_slack, 147_456);
        assert_eq!(per_layer_slack * 4 * 24, 14_155_776);

        let old_expanded_32k = 32_768usize * 96 * (128 + 128) * 4;
        assert_eq!(old_expanded_32k, 3_221_225_472);
        assert!(scratch_32k.total_bytes().unwrap() * 200 < old_expanded_32k);
    }

    #[test]
    fn mla_cache_growth_is_chunk_bounded_instead_of_geometric() {
        let geometry = tiny_mla_geometry();
        let mut cache = MlaCache::new(600, &geometry).unwrap();
        let latent_chunk = MLA_CACHE_GROWTH_TOKENS * geometry.kv_lora_rank;
        let slot_chunk = MLA_CACHE_GROWTH_TOKENS * geometry.qk_nope_slot_dim;
        let mut previous_latent_capacity = 0;
        let mut previous_slot_capacity = 0;

        for position in 0..600 {
            reserve_mla_cache_append_capacity(&mut cache, &geometry).unwrap();
            let latent_capacity = cache.allocated_latent_f32_capacity();
            let slot_capacity = cache.allocated_nope_slot_f32_capacity();
            if latent_capacity != previous_latent_capacity {
                assert!(latent_capacity - previous_latent_capacity <= latent_chunk);
            }
            if slot_capacity != previous_slot_capacity {
                assert!(slot_capacity - previous_slot_capacity <= slot_chunk);
            }
            cache.latent.extend_from_slice(&[1.0, 2.0]);
            cache.shared_nope_slots.push(3.0);
            cache.len += 1;
            assert_eq!(cache.len, position + 1);
            assert!(cache.latent.capacity() - cache.latent.len() < latent_chunk);
            assert!(
                cache.shared_nope_slots.capacity() - cache.shared_nope_slots.len() < slot_chunk
            );
            previous_latent_capacity = latent_capacity;
            previous_slot_capacity = slot_capacity;
        }
        assert_eq!(cache.latent.capacity(), 600 * geometry.kv_lora_rank);
        assert_eq!(
            cache.shared_nope_slots.capacity(),
            600 * geometry.qk_nope_slot_dim
        );
    }

    #[test]
    fn failed_second_cache_reservation_preserves_logical_state() {
        let geometry = tiny_mla_geometry();
        let mut cache = MlaCache::new(4, &geometry).unwrap();
        cache.latent.extend_from_slice(&[1.0, 2.0]);
        cache.shared_nope_slots.push(3.0);
        cache.len = 1;
        cache.latent.shrink_to_fit();
        cache.shared_nope_slots.shrink_to_fit();
        let before = cache.clone();
        let old_latent_capacity = cache.latent.capacity();

        let error = try_reserve_mla_cache_vectors(&mut cache, 1, usize::MAX).unwrap_err();
        assert!(matches!(
            error,
            AttentionError::Allocation {
                value: "MLA shared NoPE slot cache",
                ..
            }
        ));
        // Vec capacity may change after the first successful reservation. Logical sequence state
        // (position, lengths, and values) remains identical and can be retried safely.
        assert_eq!(cache, before);
        assert!(cache.latent.capacity() >= old_latent_capacity);
        assert_eq!(cache.len(), before.len());
    }

    #[test]
    fn mla_prefill_matches_incremental_and_nope_slot_is_never_rotated() {
        let geometry = tiny_mla_geometry();
        let parameters = tiny_mla_parameters();
        let tokens = [3.0, 4.0, 4.0, 3.0];

        let mut prefill_cache = MlaCache::new(2, &geometry).unwrap();
        let mut prefill_projector = TinyMlaProjector {
            fail_on: None,
            kv_b_calls: 0,
        };
        let prefill = mla_prefill(
            &geometry,
            parameters,
            &mut prefill_cache,
            0,
            &tokens,
            2,
            &mut prefill_projector,
        )
        .unwrap();

        let mut incremental_cache = MlaCache::new(2, &geometry).unwrap();
        let mut incremental_projector = TinyMlaProjector {
            fail_on: None,
            kv_b_calls: 0,
        };
        let mut incremental = Vec::new();
        incremental.extend(
            mla_step(
                &geometry,
                parameters,
                &mut incremental_cache,
                0,
                &tokens[..2],
                &mut incremental_projector,
            )
            .unwrap(),
        );
        incremental.extend(
            mla_step(
                &geometry,
                parameters,
                &mut incremental_cache,
                1,
                &tokens[2..],
                &mut incremental_projector,
            )
            .unwrap(),
        );
        assert_eq!(prefill, incremental);
        assert_eq!(prefill_cache, incremental_cache);
        assert_eq!(prefill_projector.kv_b_calls, 6);
        assert_eq!(incremental_projector.kv_b_calls, 6);
        // These are the raw kv_a slot values from positions zero and one. A RoPE path would
        // alter the second pair before caching or scoring.
        assert_eq!(prefill_cache.shared_nope_slots(), &[4.0, 3.0]);

        let inv = (12.5f64 + 1.0e-6).sqrt().recip();
        let q_nope = 4.0 * inv;
        let q_slot = 3.0 * inv;
        let key0 = 3.0 * inv;
        let key1 = 4.0 * inv;
        let score0 = (q_nope * key0 + q_slot * 4.0) / 2.0f64.sqrt();
        let score1 = (q_nope * key1 + q_slot * 3.0) / 2.0f64.sqrt();
        let probability0 = (score0 - score0.max(score1)).exp()
            / ((score0 - score0.max(score1)).exp() + (score1 - score0.max(score1)).exp());
        let context = probability0 * (4.0 * inv) + (1.0 - probability0) * (3.0 * inv);
        assert_close(prefill[2], (0.5 * context) as f32, 2.0e-6);
    }

    #[test]
    fn mla_position_capacity_and_projection_failures_roll_back() {
        let geometry = tiny_mla_geometry();
        let parameters = tiny_mla_parameters();
        let mut cache = MlaCache::new(1, &geometry).unwrap();
        let empty = cache.clone();
        let mut projector = TinyMlaProjector {
            fail_on: None,
            kv_b_calls: 0,
        };
        assert!(matches!(
            mla_step(
                &geometry,
                parameters,
                &mut cache,
                1,
                &[3.0, 4.0],
                &mut projector,
            ),
            Err(AttentionError::CachePosition { .. })
        ));
        assert_eq!(cache, empty);

        let mut failing = TinyMlaProjector {
            fail_on: Some(Projection::MlaOutput),
            kv_b_calls: 0,
        };
        assert!(mla_step(
            &geometry,
            parameters,
            &mut cache,
            0,
            &[3.0, 4.0],
            &mut failing,
        )
        .is_err());
        assert_eq!(cache, empty);

        mla_step(
            &geometry,
            parameters,
            &mut cache,
            0,
            &[3.0, 4.0],
            &mut projector,
        )
        .unwrap();
        let full = cache.clone();
        assert!(matches!(
            mla_step(
                &geometry,
                parameters,
                &mut cache,
                1,
                &[4.0, 3.0],
                &mut projector,
            ),
            Err(AttentionError::CacheCapacity { .. })
        ));
        assert_eq!(cache, full);
    }

    #[test]
    fn geometry_overflow_and_nonfinite_restored_state_are_rejected() {
        let bad_kda = KdaGeometry {
            hidden_size: 1,
            num_heads: usize::MAX,
            head_dim: 2,
            conv_kernel_size: 1,
            rms_epsilon: 1.0e-6,
            gate_lower_bound: -5.0,
        };
        assert!(matches!(
            KdaState::new(&bad_kda),
            Err(AttentionError::Overflow { .. })
        ));

        let geometry = tiny_mla_geometry();
        assert!(matches!(
            MlaCache::from_parts(1, 1, vec![f32::NAN, 0.0], vec![1.0], &geometry),
            Err(AttentionError::NonFinite { .. })
        ));
        let bad_mla = MlaGeometry {
            num_heads: usize::MAX,
            qk_nope_head_dim: 2,
            ..geometry
        };
        assert!(matches!(
            MlaCache::new(0, &bad_mla),
            Err(AttentionError::Overflow { .. })
        ));
    }
}
