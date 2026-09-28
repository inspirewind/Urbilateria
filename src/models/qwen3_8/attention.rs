use super::math::{linear, linear_batch, round_to_bf16, round_to_bf16_in_place, uses_bf16_output};
use super::norm::{zero_centered_rms_norm, NormError};
use crate::execution::{install, should_parallelize};
use crate::math::{softmax_in_place, MathError};
use crate::model::{DenseMatrix, WeightError, WeightMatrix};
use rayon::prelude::*;
use std::fmt;

// Head tasks allocate their own scores and run several reductions. Require more work
// than a matrix dispatch to amortize scheduling; native heads reach this at 64 tokens.
const PARALLEL_HEAD_MIN_WORK: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullAttentionGeometry {
    pub hidden_size: usize,
    pub num_query_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
}

impl FullAttentionGeometry {
    pub fn validate(self) -> Result<(), AttentionError> {
        if self.hidden_size == 0
            || self.num_query_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.rotary_dim == 0
        {
            return Err(AttentionError::InvalidGeometry(
                "all full-attention dimensions must be non-zero".to_owned(),
            ));
        }
        if self.num_query_heads % self.num_key_value_heads != 0 {
            return Err(AttentionError::InvalidGeometry(format!(
                "query heads {} must be divisible by KV heads {}",
                self.num_query_heads, self.num_key_value_heads
            )));
        }
        if self.rotary_dim > self.head_dim || self.rotary_dim % 2 != 0 {
            return Err(AttentionError::InvalidGeometry(format!(
                "rotary dimension {} must be even and no larger than head dimension {}",
                self.rotary_dim, self.head_dim
            )));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(AttentionError::InvalidGeometry(
                "RMSNorm epsilon must be finite and positive".to_owned(),
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(AttentionError::InvalidGeometry(
                "RoPE theta must be finite and positive".to_owned(),
            ));
        }
        checked_product(
            "query projection",
            &[self.num_query_heads, self.head_dim, 2],
        )?;
        checked_product("KV projection", &[self.num_key_value_heads, self.head_dim])?;
        checked_product("attention context", &[self.num_query_heads, self.head_dim])?;
        Ok(())
    }

    fn query_width(self) -> Result<usize, AttentionError> {
        checked_product(
            "query projection",
            &[self.num_query_heads, self.head_dim, 2],
        )
    }

    fn kv_width(self) -> Result<usize, AttentionError> {
        checked_product("KV projection", &[self.num_key_value_heads, self.head_dim])
    }

    fn context_width(self) -> Result<usize, AttentionError> {
        checked_product("attention context", &[self.num_query_heads, self.head_dim])
    }
}

#[derive(Debug)]
pub enum AttentionError {
    Weight(WeightError),
    Math(MathError),
    Norm(NormError),
    InvalidGeometry(String),
    InvalidShape(String),
    NonFinite(&'static str),
    Allocation(String),
    CacheGeometry {
        expected_heads: usize,
        expected_head_dim: usize,
        got_heads: usize,
        got_head_dim: usize,
    },
    CachePosition {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight(error) => error.fmt(f),
            Self::Math(error) => error.fmt(f),
            Self::Norm(error) => error.fmt(f),
            Self::InvalidGeometry(reason) => {
                write!(f, "invalid Qwen full-attention geometry: {reason}")
            }
            Self::InvalidShape(reason) => write!(f, "invalid Qwen full-attention shape: {reason}"),
            Self::NonFinite(name) => {
                write!(f, "Qwen full-attention {name} contains NaN or infinity")
            }
            Self::Allocation(reason) => {
                write!(f, "cannot allocate Qwen full-attention cache: {reason}")
            }
            Self::CacheGeometry {
                expected_heads,
                expected_head_dim,
                got_heads,
                got_head_dim,
            } => write!(
                f,
                "Qwen KV cache geometry [{got_heads}, {got_head_dim}] does not match [{expected_heads}, {expected_head_dim}]"
            ),
            Self::CachePosition { expected, got } => {
                write!(f, "Qwen KV cache expected position {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for AttentionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weight(error) => Some(error),
            Self::Math(error) => Some(error),
            Self::Norm(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WeightError> for AttentionError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

impl From<MathError> for AttentionError {
    fn from(value: MathError) -> Self {
        Self::Math(value)
    }
}

impl From<NormError> for AttentionError {
    fn from(value: NormError) -> Self {
        Self::Norm(value)
    }
}

/// One full-attention layer's unexpanded GQA KV cache.
#[derive(Debug, Clone)]
pub struct FullAttentionCache {
    num_key_value_heads: usize,
    head_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl FullAttentionCache {
    pub fn new(num_key_value_heads: usize, head_dim: usize) -> Result<Self, AttentionError> {
        Self::with_capacity(num_key_value_heads, head_dim, 0)
    }

    pub fn with_capacity(
        num_key_value_heads: usize,
        head_dim: usize,
        tokens: usize,
    ) -> Result<Self, AttentionError> {
        if num_key_value_heads == 0 || head_dim == 0 {
            return Err(AttentionError::InvalidGeometry(
                "KV cache dimensions must be non-zero".to_owned(),
            ));
        }
        let per_token = checked_product("KV cache token", &[num_key_value_heads, head_dim])?;
        let capacity = checked_product("KV cache capacity", &[per_token, tokens])?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(capacity).map_err(|error| {
            AttentionError::Allocation(format!("{capacity} key values: {error}"))
        })?;
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(|error| {
            AttentionError::Allocation(format!("{capacity} value values: {error}"))
        })?;
        Ok(Self {
            num_key_value_heads,
            head_dim,
            keys,
            values,
        })
    }

    pub fn len(&self) -> usize {
        self.keys.len() / (self.num_key_value_heads * self.head_dim)
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn stored_f32_elements(&self) -> usize {
        self.keys.len().saturating_add(self.values.len())
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    pub fn truncate(&mut self, tokens: usize) -> Result<(), AttentionError> {
        if tokens > self.len() {
            return Err(AttentionError::CachePosition {
                expected: self.len(),
                got: tokens,
            });
        }
        let length = tokens * self.num_key_value_heads * self.head_dim;
        self.keys.truncate(length);
        self.values.truncate(length);
        Ok(())
    }

    fn row(&self, values: bool, token: usize, head: usize) -> &[f32] {
        let source = if values { &self.values } else { &self.keys };
        let start = (token * self.num_key_value_heads + head) * self.head_dim;
        &source[start..start + self.head_dim]
    }

    fn push(&mut self, key: &[f32], value: &[f32]) {
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
    }
}

/// Bias-free gated Qwen GQA with ordered per-head reductions for a single batch item.
#[derive(Debug, Clone)]
pub struct FullAttention {
    geometry: FullAttentionGeometry,
    q_proj: WeightMatrix,
    k_proj: WeightMatrix,
    v_proj: WeightMatrix,
    o_proj: WeightMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    bf16_activations: bool,
}

impl FullAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        geometry: FullAttentionGeometry,
        q_proj: DenseMatrix,
        k_proj: DenseMatrix,
        v_proj: DenseMatrix,
        o_proj: DenseMatrix,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    ) -> Result<Self, AttentionError> {
        Self::new_mixed(
            geometry,
            q_proj.into(),
            k_proj.into(),
            v_proj.into(),
            o_proj.into(),
            q_norm,
            k_norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_mixed(
        geometry: FullAttentionGeometry,
        q_proj: WeightMatrix,
        k_proj: WeightMatrix,
        v_proj: WeightMatrix,
        o_proj: WeightMatrix,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    ) -> Result<Self, AttentionError> {
        geometry.validate()?;
        expect_matrix(
            "q_proj",
            &q_proj,
            geometry.query_width()?,
            geometry.hidden_size,
        )?;
        expect_matrix(
            "k_proj",
            &k_proj,
            geometry.kv_width()?,
            geometry.hidden_size,
        )?;
        expect_matrix(
            "v_proj",
            &v_proj,
            geometry.kv_width()?,
            geometry.hidden_size,
        )?;
        expect_matrix(
            "o_proj",
            &o_proj,
            geometry.hidden_size,
            geometry.context_width()?,
        )?;
        expect_vector("q_norm", &q_norm, geometry.head_dim)?;
        expect_vector("k_norm", &k_norm, geometry.head_dim)?;
        let bf16_activations = uses_bf16_output(&q_proj);
        if [
            uses_bf16_output(&k_proj),
            uses_bf16_output(&v_proj),
            uses_bf16_output(&o_proj),
        ]
        .into_iter()
        .any(|value| value != bf16_activations)
        {
            return Err(AttentionError::InvalidShape(
                "full-attention projections must share one activation dtype".to_owned(),
            ));
        }
        Ok(Self {
            geometry,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            bf16_activations,
        })
    }

    pub fn geometry(&self) -> FullAttentionGeometry {
        self.geometry
    }

    pub(crate) fn uses_bf16_activations(&self) -> bool {
        self.bf16_activations
    }

    pub fn new_cache(&self) -> Result<FullAttentionCache, AttentionError> {
        FullAttentionCache::new(self.geometry.num_key_value_heads, self.geometry.head_dim)
    }

    pub fn new_cache_with_capacity(
        &self,
        tokens: usize,
    ) -> Result<FullAttentionCache, AttentionError> {
        FullAttentionCache::with_capacity(
            self.geometry.num_key_value_heads,
            self.geometry.head_dim,
            tokens,
        )
    }

    pub fn forward_token(
        &self,
        input: &[f32],
        position: usize,
        cache: &mut FullAttentionCache,
    ) -> Result<Vec<f32>, AttentionError> {
        self.validate_cache(cache)?;
        if position != cache.len() {
            return Err(AttentionError::CachePosition {
                expected: cache.len(),
                got: position,
            });
        }
        expect_input(input, self.geometry.hidden_size)?;

        let projected_query = linear(&self.q_proj, input)?;
        let projected_key = linear(&self.k_proj, input)?;
        let value = linear(&self.v_proj, input)?;
        let (context, key) =
            self.projected_context(&projected_query, &projected_key, &value, position, cache)?;
        let output = linear(&self.o_proj, &context)?;
        ensure_finite("output projection", &output)?;
        cache.push(&key, &value);
        Ok(output)
    }

    /// Shares projection weights across consecutive inputs while retaining causal KV order.
    /// Failed batches restore the original cache length, including output-projection errors.
    pub fn forward_tokens(
        &self,
        input: &[f32],
        batch: usize,
        position: usize,
        cache: &mut FullAttentionCache,
    ) -> Result<Vec<f32>, AttentionError> {
        if batch == 1 {
            return self.forward_token(input, position, cache);
        }
        if batch == 0 {
            return Err(AttentionError::InvalidShape(
                "prefill batch must be non-zero".into(),
            ));
        }
        self.validate_cache(cache)?;
        if position != cache.len() {
            return Err(AttentionError::CachePosition {
                expected: cache.len(),
                got: position,
            });
        }
        position
            .checked_add(batch)
            .ok_or_else(|| AttentionError::InvalidShape("prefill position overflows".into()))?;
        let length = batch
            .checked_mul(self.geometry.hidden_size)
            .ok_or_else(|| AttentionError::InvalidShape("prefill input length overflows".into()))?;
        expect_input(input, length)?;
        let queries = linear_batch(&self.q_proj, input, batch)?;
        let keys = linear_batch(&self.k_proj, input, batch)?;
        let values = linear_batch(&self.v_proj, input, batch)?;
        let query_width = self.geometry.query_width()?;
        let kv_width = self.geometry.kv_width()?;
        let original_length = cache.len();
        let result = (|| {
            let mut contexts = Vec::with_capacity(batch * self.geometry.context_width()?);
            for token in 0..batch {
                let value = &values[token * kv_width..(token + 1) * kv_width];
                let (context, key) = self.projected_context(
                    &queries[token * query_width..(token + 1) * query_width],
                    &keys[token * kv_width..(token + 1) * kv_width],
                    value,
                    position + token,
                    cache,
                )?;
                contexts.extend(context);
                cache.push(&key, value);
            }
            let output = linear_batch(&self.o_proj, &contexts, batch)?;
            ensure_finite("output projection", &output)?;
            Ok(output)
        })();
        if result.is_err() {
            cache.truncate(original_length)?;
        }
        result
    }

    #[inline(always)]
    fn projected_context(
        &self,
        projected_query: &[f32],
        projected_key: &[f32],
        value: &[f32],
        position: usize,
        cache: &FullAttentionCache,
    ) -> Result<(Vec<f32>, Vec<f32>), AttentionError> {
        let mut query = Vec::with_capacity(self.geometry.context_width()?);
        let mut gate = Vec::with_capacity(self.geometry.context_width()?);
        for head in projected_query.chunks_exact(self.geometry.head_dim * 2) {
            let mut normalized = zero_centered_rms_norm(
                &head[..self.geometry.head_dim],
                &self.q_norm,
                self.geometry.norm_eps,
            )?;
            if self.bf16_activations {
                round_to_bf16_in_place(&mut normalized)?;
            }
            query.extend(normalized);
            gate.extend_from_slice(&head[self.geometry.head_dim..]);
        }
        ensure_finite("query gate", &gate)?;

        let mut key = Vec::with_capacity(self.geometry.kv_width()?);
        for head in projected_key.chunks_exact(self.geometry.head_dim) {
            let mut normalized =
                zero_centered_rms_norm(head, &self.k_norm, self.geometry.norm_eps)?;
            if self.bf16_activations {
                round_to_bf16_in_place(&mut normalized)?;
            }
            key.extend(normalized);
        }
        ensure_finite("value projection", value)?;

        for head in query.chunks_exact_mut(self.geometry.head_dim) {
            half_split_rope(
                &mut head[..self.geometry.rotary_dim],
                position,
                self.geometry.rope_theta,
            )?;
        }
        for head in key.chunks_exact_mut(self.geometry.head_dim) {
            half_split_rope(
                &mut head[..self.geometry.rotary_dim],
                position,
                self.geometry.rope_theta,
            )?;
        }
        if self.bf16_activations {
            round_to_bf16_in_place(&mut query)?;
            round_to_bf16_in_place(&mut key)?;
        }

        let work = self
            .geometry
            .context_width()?
            .saturating_mul(cache.len().saturating_add(1));
        let mut context = self.head_context(
            &query,
            &key,
            value,
            cache,
            work >= PARALLEL_HEAD_MIN_WORK
                && should_parallelize(self.geometry.num_query_heads, work),
        )?;
        ensure_finite("attention context", &context)?;

        for (value, &gate) in context.iter_mut().zip(&gate) {
            let gate = sigmoid(gate);
            let gate = if self.bf16_activations {
                round_to_bf16(gate)?
            } else {
                gate
            };
            *value *= gate;
            if self.bf16_activations {
                *value = round_to_bf16(*value)?;
            }
        }
        ensure_finite("gated attention context", &context)?;
        Ok((context, key))
    }

    fn head_context(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        cache: &FullAttentionCache,
        parallel: bool,
    ) -> Result<Vec<f32>, AttentionError> {
        let groups = self.geometry.num_query_heads / self.geometry.num_key_value_heads;
        let mut context = vec![0.0; self.geometry.context_width()?];
        let scale = (self.geometry.head_dim as f32).sqrt().recip();
        let attend_head = |query_head: usize, output: &mut [f32]| -> Result<(), AttentionError> {
            let kv_head = query_head / groups;
            let q_start = query_head * self.geometry.head_dim;
            let q = &query[q_start..q_start + self.geometry.head_dim];
            let current_start = kv_head * self.geometry.head_dim;
            let current_key = &key[current_start..current_start + self.geometry.head_dim];
            let current_value = &value[current_start..current_start + self.geometry.head_dim];

            let mut scores = Vec::with_capacity(cache.len() + 1);
            for token in 0..cache.len() {
                let score = dot(q, cache.row(false, token, kv_head)) * scale;
                scores.push(if self.bf16_activations {
                    round_to_bf16(score)?
                } else {
                    score
                });
            }
            let score = dot(q, current_key) * scale;
            scores.push(if self.bf16_activations {
                round_to_bf16(score)?
            } else {
                score
            });
            softmax_in_place(&mut scores)?;
            if self.bf16_activations {
                round_to_bf16_in_place(&mut scores)?;
            }

            // Native 256-wide heads need no heap scratch. Traverse complete value rows
            // while retaining the original token-order F64 sum for every dimension.
            let mut local_sums = [0.0f64; 256];
            let mut wide_sums;
            let sums = if self.geometry.head_dim <= local_sums.len() {
                &mut local_sums[..self.geometry.head_dim]
            } else {
                wide_sums = vec![0.0; self.geometry.head_dim];
                &mut wide_sums
            };
            weighted_values_into(cache, kv_head, current_value, &scores, sums);
            for (output, &sum) in output.iter_mut().zip(sums.iter()) {
                let value = sum as f32;
                *output = if self.bf16_activations {
                    round_to_bf16(value)?
                } else {
                    value
                };
            }
            Ok(())
        };
        if parallel {
            // Heads share read-only KV rows and own disjoint output slices. Scores and
            // reductions remain token-ordered within each head; errors retain head order.
            let results = install(|| {
                context
                    .par_chunks_mut(self.geometry.head_dim)
                    .enumerate()
                    .map(|(head, output)| attend_head(head, output))
                    .collect::<Vec<_>>()
            });
            for result in results {
                result?;
            }
        } else {
            for (head, output) in context.chunks_mut(self.geometry.head_dim).enumerate() {
                attend_head(head, output)?;
            }
        }
        Ok(context)
    }

    fn validate_cache(&self, cache: &FullAttentionCache) -> Result<(), AttentionError> {
        if cache.num_key_value_heads != self.geometry.num_key_value_heads
            || cache.head_dim != self.geometry.head_dim
        {
            return Err(AttentionError::CacheGeometry {
                expected_heads: self.geometry.num_key_value_heads,
                expected_head_dim: self.geometry.head_dim,
                got_heads: cache.num_key_value_heads,
                got_head_dim: cache.head_dim,
            });
        }
        Ok(())
    }
}

/// Each dimension retains its columnwise reference sum, starting at positive zero.
fn weighted_values_into(
    cache: &FullAttentionCache,
    head: usize,
    current: &[f32],
    probabilities: &[f32],
    sums: &mut [f64],
) {
    assert_eq!(sums.len(), cache.head_dim);
    assert_eq!(current.len(), cache.head_dim);
    assert_eq!(probabilities.len(), cache.len() + 1);
    assert!(head < cache.num_key_value_heads);
    sums.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected, all rows have the checked width, and SIMD accesses
        // cover only complete groups of four values. The tail is scalar.
        unsafe { weighted_values_avx2(cache, head, current, probabilities, sums) };
        return;
    }
    for (token, &probability) in probabilities.iter().enumerate() {
        let values = if token == cache.len() {
            current
        } else {
            cache.row(true, token, head)
        };
        for (sum, &value) in sums.iter_mut().zip(values) {
            *sum += f64::from(probability) * f64::from(value);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn weighted_values_avx2(
    cache: &FullAttentionCache,
    head: usize,
    current: &[f32],
    probabilities: &[f32],
    sums: &mut [f64],
) {
    use std::arch::x86_64::*;
    let complete = sums.len() / 4 * 4;
    for (token, &probability) in probabilities.iter().enumerate() {
        let values = if token == cache.len() {
            current
        } else {
            cache.row(true, token, head)
        };
        let weight = _mm256_set1_pd(f64::from(probability));
        for column in (0..complete).step_by(4) {
            let value = _mm256_cvtps_pd(_mm_loadu_ps(values.as_ptr().add(column)));
            let previous = _mm256_loadu_pd(sums.as_ptr().add(column));
            // Do not fuse or reassociate the reference F64 multiply/add operations.
            let next = _mm256_add_pd(previous, _mm256_mul_pd(weight, value));
            _mm256_storeu_pd(sums.as_mut_ptr().add(column), next);
        }
        for column in complete..sums.len() {
            sums[column] += f64::from(probability) * f64::from(values[column]);
        }
    }
}

fn half_split_rope(values: &mut [f32], position: usize, theta: f32) -> Result<(), AttentionError> {
    if values.is_empty() || values.len() % 2 != 0 {
        return Err(AttentionError::InvalidGeometry(
            "rotary slice must be non-empty and even".to_owned(),
        ));
    }
    ensure_finite("rotary input", values)?;
    let input = values.to_vec();
    let half = values.len() / 2;
    for dimension in 0..half {
        let inverse_frequency = theta.powf(-2.0 * dimension as f32 / values.len() as f32);
        let angle = position as f32 * inverse_frequency;
        let (sine, cosine) = angle.sin_cos();
        let first = input[dimension];
        let second = input[half + dimension];
        values[dimension] = first * cosine - second * sine;
        values[half + dimension] = second * cosine + first * sine;
    }
    ensure_finite("rotary output", values)
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize, AttentionError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            AttentionError::InvalidGeometry(format!(
                "{name} dimensions overflow usize: {factors:?}"
            ))
        })
    })
}

fn expect_matrix(
    name: &str,
    matrix: &WeightMatrix,
    rows: usize,
    columns: usize,
) -> Result<(), AttentionError> {
    if matrix.rows() != rows || matrix.cols() != columns {
        return Err(AttentionError::InvalidShape(format!(
            "{name} must be [{rows}, {columns}], got [{}, {}]",
            matrix.rows(),
            matrix.cols()
        )));
    }
    Ok(())
}

fn expect_vector(name: &str, vector: &[f32], length: usize) -> Result<(), AttentionError> {
    if vector.len() != length {
        return Err(AttentionError::InvalidShape(format!(
            "{name} must have {length} values, got {}",
            vector.len()
        )));
    }
    ensure_finite("normalization weight", vector)
}

fn expect_input(input: &[f32], length: usize) -> Result<(), AttentionError> {
    if input.len() != length {
        return Err(AttentionError::InvalidShape(format!(
            "input must have {length} values, got {}",
            input.len()
        )));
    }
    ensure_finite("input", input)
}

fn ensure_finite(name: &'static str, values: &[f32]) -> Result<(), AttentionError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(AttentionError::NonFinite(name));
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum::<f64>() as f32
}

fn sigmoid(value: f32) -> f32 {
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

    fn multi_head_attention(width: usize, bf16: bool) -> FullAttention {
        let geometry = FullAttentionGeometry {
            hidden_size: 2,
            num_query_heads: 16,
            num_key_value_heads: 2,
            head_dim: width,
            rotary_dim: 2,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let zeros = |rows, columns| {
            if bf16 {
                WeightMatrix::Bf16(
                    crate::model::Bf16Matrix::from_le_bytes(
                        rows,
                        columns,
                        vec![0; rows * columns * 2],
                    )
                    .unwrap(),
                )
            } else {
                matrix(rows, columns, vec![0.0; rows * columns]).into()
            }
        };
        FullAttention::new_mixed(
            geometry,
            zeros(geometry.query_width().unwrap(), 2),
            zeros(geometry.kv_width().unwrap(), 2),
            zeros(geometry.kv_width().unwrap(), 2),
            zeros(2, geometry.context_width().unwrap()),
            vec![0.0; width],
            vec![0.0; width],
        )
        .unwrap()
    }

    #[test]
    fn parallel_heads_preserve_serial_context_bits() {
        for bf16 in [false, true] {
            for (width, tokens) in [
                (5, 0),
                (5, 17),
                (128, 33),
                (256, 65),
                (257, 129),
                (256, 513),
            ] {
                let module = multi_head_attention(width, bf16);
                let mut cache = module.new_cache().unwrap();
                let row = |length, offset| {
                    (0..length)
                        .map(|i| ((i * 19 + offset) % 251) as f32 / 97.0 - 1.25)
                        .collect::<Vec<_>>()
                };
                for token in 0..tokens {
                    cache.push(&row(2 * width, token * 13), &row(2 * width, token * 31));
                }
                let query = row(16 * width, 17);
                let key = row(2 * width, 37);
                let value = row(2 * width, 53);
                let expected = module
                    .head_context(&query, &key, &value, &cache, false)
                    .unwrap();
                let actual = module
                    .head_context(&query, &key, &value, &cache, true)
                    .unwrap();
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn parallel_head_output_failure_preserves_existing_cache() {
        let mut module = multi_head_attention(256, false);
        let kv = module.geometry.kv_width().unwrap();
        let context = module.geometry.context_width().unwrap();
        module.v_proj = matrix(kv, 2, vec![1.0; kv * 2]).into();
        module.o_proj = matrix(2, context, vec![f32::MAX; 2 * context]).into();
        let mut cache = module.new_cache().unwrap();
        for _ in 0..65 {
            cache.push(&vec![0.0; kv], &vec![0.0; kv]);
        }
        let before = cache.clone();
        assert!(module.forward_token(&[1.0, 1.0], 65, &mut cache).is_err());
        assert_eq!(cache.keys, before.keys);
        assert_eq!(cache.values, before.values);
        // The first batch token succeeds; the second overflows the output projection.
        assert!(module
            .forward_tokens(&[0.0, 0.0, 1.0, 1.0], 2, 65, &mut cache)
            .is_err());
        assert_eq!(cache.keys, before.keys);
        assert_eq!(cache.values, before.values);
    }

    #[test]
    fn contiguous_values_preserve_columnwise_f64_sums_and_tails() {
        for (heads, width, tokens) in [
            (1, 1, 0),
            (2, 3, 5),
            (3, 5, 17),
            (2, 256, 33),
            (2, 257, 65),
            (2, 256, 513),
        ] {
            let mut cache = FullAttentionCache::with_capacity(heads, width, tokens).unwrap();
            for token in 0..tokens {
                let values = (0..heads * width)
                    .map(|i| ((i * 137 + token * 19) % 251) as f32 / 37.0 - 3.0)
                    .collect::<Vec<_>>();
                cache.push(&vec![0.0; heads * width], &values);
            }
            let probabilities = (0..=tokens)
                .map(|i| (i * 23 % 71) as f32 / 97.0 - 0.25)
                .collect::<Vec<_>>();
            let current = (0..width)
                .map(|i| (i * 17 % 47) as f32 / 31.0 - 1.0)
                .collect::<Vec<_>>();
            for head in 0..heads {
                let mut actual = vec![17.0; width];
                weighted_values_into(&cache, head, &current, &probabilities, &mut actual);
                for dimension in 0..width {
                    let mut expected = 0.0f64;
                    for (token, &probability) in probabilities[..tokens].iter().enumerate() {
                        expected += f64::from(probability)
                            * f64::from(cache.row(true, token, head)[dimension]);
                    }
                    expected += f64::from(probabilities[tokens]) * f64::from(current[dimension]);
                    assert_eq!(actual[dimension].to_bits(), expected.to_bits());
                }
            }
        }
        let mut cache = FullAttentionCache::new(1, 5).unwrap();
        cache.push(&[0.0; 5], &[-0.0; 5]);
        let mut sums = [1.0; 5];
        weighted_values_into(&cache, 0, &[-0.0; 5], &[1.0, 1.0], &mut sums);
        assert!(sums.iter().all(|x| x.to_bits() == 0.0f64.to_bits()));
    }

    fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> DenseMatrix {
        DenseMatrix::new(rows, columns, values).unwrap()
    }

    fn tiny_attention() -> FullAttention {
        let geometry = FullAttentionGeometry {
            hidden_size: 2,
            num_query_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        // q_proj is laid out as [head, query_then_gate, head_dim].
        let q_proj = matrix(
            8,
            2,
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // head 0
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // head 1
            ],
        );
        let identity = matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let output = matrix(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        FullAttention::new(
            geometry,
            q_proj,
            identity.clone(),
            identity,
            output,
            vec![0.0; 2],
            vec![0.0; 2],
        )
        .unwrap()
    }

    #[test]
    fn batched_prefill_preserves_outputs_and_existing_state_bits() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for bf16 in [false, true] {
            let mut module = tiny_attention();
            if bf16 {
                for weight in [
                    &mut module.q_proj,
                    &mut module.k_proj,
                    &mut module.v_proj,
                    &mut module.o_proj,
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
            for batch in [2, 3, 8, 19, 32, 33, 128, 129] {
                let input = (0..batch * 2)
                    .map(|i| ((i * 19 % 41) as f32 - 20.0) / 16.0)
                    .collect::<Vec<_>>();
                let mut sequential = module.new_cache().unwrap();
                module
                    .forward_token(&[0.25, 0.5], 0, &mut sequential)
                    .unwrap();
                let mut batched = sequential.clone();
                let mut expected = Vec::new();
                for (offset, token) in input.chunks_exact(2).enumerate() {
                    expected.extend(
                        module
                            .forward_token(token, offset + 1, &mut sequential)
                            .unwrap(),
                    );
                }
                let actual = module
                    .forward_tokens(&input, batch, 1, &mut batched)
                    .unwrap();
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(bits(&batched.keys), bits(&sequential.keys));
                assert_eq!(bits(&batched.values), bits(&sequential.values));
            }
        }
    }

    #[test]
    fn batch_output_failure_rolls_back_existing_kv_cache() {
        let mut module = tiny_attention();
        module.o_proj = matrix(2, 4, vec![f32::MAX; 8]).into();
        let mut cache = module.new_cache().unwrap();
        module.forward_token(&[0.0, 0.0], 0, &mut cache).unwrap();
        let before = cache.clone();
        assert!(module
            .forward_tokens(&[0.25, 0.5, 10.0, 10.0], 2, 1, &mut cache)
            .is_err());
        assert_eq!(cache.keys, before.keys);
        assert_eq!(cache.values, before.values);
        assert!(module.forward_tokens(&[], 0, 1, &mut cache).is_err());
        assert_eq!(cache.len(), before.len());
    }

    #[test]
    fn one_token_gqa_applies_output_gate_and_stores_unexpanded_kv() {
        let attention = tiny_attention();
        let mut cache = attention.new_cache().unwrap();
        let output = attention
            .forward_token(&[2.0, -1.0], 0, &mut cache)
            .unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] + 0.5).abs() < 1e-6);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stored_f32_elements(), 4);
    }

    #[test]
    fn position_gap_does_not_mutate_cache() {
        let attention = tiny_attention();
        let mut cache = attention.new_cache().unwrap();
        let error = attention
            .forward_token(&[1.0, 2.0], 1, &mut cache)
            .unwrap_err();
        assert!(matches!(
            error,
            AttentionError::CachePosition {
                expected: 0,
                got: 1
            }
        ));
        assert!(cache.is_empty());
    }

    #[test]
    fn partial_rope_uses_half_split_rotation() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        half_split_rope(&mut values, 1, 1.0).unwrap();
        let (sine, cosine) = 1.0f32.sin_cos();
        let expected = [
            cosine - 3.0 * sine,
            2.0 * cosine - 4.0 * sine,
            3.0 * cosine + sine,
            4.0 * cosine + 2.0 * sine,
        ];
        for (actual, expected) in values.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
