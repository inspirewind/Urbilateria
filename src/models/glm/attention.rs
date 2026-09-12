use crate::math::{interleaved_rope, rms_norm, softmax_in_place, MathError};
use crate::model::{DenseMatrix, MatrixError, WeightError, WeightMatrix};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    DenseReconstruction,
    Absorbed,
}

#[derive(Debug, Clone, Copy)]
pub struct MlaGeometry {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Epsilon for the two internal Q/KV latent norms. The official implementation uses
    /// the RMSNorm constructor default (`1e-6`) here, not the decoder-block `1e-5` value.
    pub latent_norm_eps: f32,
    pub rope_theta: f32,
}

impl MlaGeometry {
    pub fn q_head_dim(self) -> Result<usize, AttentionError> {
        self.qk_nope_head_dim
            .checked_add(self.qk_rope_head_dim)
            .ok_or_else(|| AttentionError::InvalidGeometry("Q head dimension overflows".to_owned()))
    }

    pub fn compressed_elements_per_token(self) -> Result<usize, AttentionError> {
        self.kv_lora_rank
            .checked_add(self.qk_rope_head_dim)
            .ok_or_else(|| {
                AttentionError::InvalidGeometry(
                    "compressed KV elements per token overflows".to_owned(),
                )
            })
    }

    fn validate(self) -> Result<(), AttentionError> {
        if self.hidden_size == 0
            || self.num_heads == 0
            || self.q_lora_rank == 0
            || self.kv_lora_rank == 0
            || self.qk_nope_head_dim == 0
            || self.qk_rope_head_dim == 0
            || self.v_head_dim == 0
        {
            return Err(AttentionError::InvalidGeometry(
                "all MLA dimensions must be non-zero".to_owned(),
            ));
        }
        if self.qk_rope_head_dim % 2 != 0 {
            return Err(AttentionError::InvalidGeometry(
                "rotary head dimension must be even".to_owned(),
            ));
        }
        if !self.latent_norm_eps.is_finite() || self.latent_norm_eps <= 0.0 {
            return Err(AttentionError::InvalidGeometry(
                "RMSNorm epsilon must be finite and positive".to_owned(),
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(AttentionError::InvalidGeometry(
                "RoPE theta must be finite and positive".to_owned(),
            ));
        }
        checked_product("query projection", &[self.num_heads, self.q_head_dim()?])?;
        checked_product(
            "KV reconstruction",
            &[
                self.num_heads,
                self.qk_nope_head_dim
                    .checked_add(self.v_head_dim)
                    .ok_or_else(|| {
                        AttentionError::InvalidGeometry("KV head dimension overflows".to_owned())
                    })?,
            ],
        )?;
        checked_product("attention context", &[self.num_heads, self.v_head_dim])?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum AttentionError {
    Matrix(MatrixError),
    Weight(WeightError),
    Math(MathError),
    InvalidGeometry(String),
    InvalidShape(String),
    Allocation(String),
    CacheGeometry {
        expected_latent: usize,
        expected_rope: usize,
        got_latent: usize,
        got_rope: usize,
    },
    CachePosition {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(f),
            Self::Weight(error) => error.fmt(f),
            Self::Math(error) => error.fmt(f),
            Self::InvalidGeometry(reason) => write!(f, "invalid MLA geometry: {reason}"),
            Self::InvalidShape(reason) => write!(f, "invalid MLA shape: {reason}"),
            Self::Allocation(reason) => write!(f, "cannot allocate MLA cache: {reason}"),
            Self::CacheGeometry {
                expected_latent,
                expected_rope,
                got_latent,
                got_rope,
            } => write!(
                f,
                "MLA cache geometry [{got_latent}, {got_rope}] does not match [{expected_latent}, {expected_rope}]"
            ),
            Self::CachePosition { expected, got } => {
                write!(f, "MLA cache expected position {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for AttentionError {}

impl From<MatrixError> for AttentionError {
    fn from(value: MatrixError) -> Self {
        Self::Matrix(value)
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

/// One layer's compressed MLA cache.
///
/// Each token stores only the normalized `kv_lora_rank` latent and the shared rotary key,
/// rather than one full K/V pair per attention head.
#[derive(Debug, Clone)]
pub struct MlaCache {
    kv_lora_rank: usize,
    rope_dim: usize,
    latent: Vec<f32>,
    rope: Vec<f32>,
}

impl MlaCache {
    pub fn new(kv_lora_rank: usize, rope_dim: usize) -> Result<Self, AttentionError> {
        Self::with_capacity(kv_lora_rank, rope_dim, 0)
    }

    pub fn with_capacity(
        kv_lora_rank: usize,
        rope_dim: usize,
        tokens: usize,
    ) -> Result<Self, AttentionError> {
        if kv_lora_rank == 0 || rope_dim == 0 {
            return Err(AttentionError::InvalidGeometry(
                "cache dimensions must be non-zero".to_owned(),
            ));
        }
        let latent_capacity = kv_lora_rank.checked_mul(tokens).ok_or_else(|| {
            AttentionError::Allocation("latent capacity overflows usize".to_owned())
        })?;
        let rope_capacity = rope_dim.checked_mul(tokens).ok_or_else(|| {
            AttentionError::Allocation("rotary capacity overflows usize".to_owned())
        })?;
        let mut latent = Vec::new();
        latent.try_reserve_exact(latent_capacity).map_err(|error| {
            AttentionError::Allocation(format!("{latent_capacity} latent values: {error}"))
        })?;
        let mut rope = Vec::new();
        rope.try_reserve_exact(rope_capacity).map_err(|error| {
            AttentionError::Allocation(format!("{rope_capacity} rotary values: {error}"))
        })?;
        Ok(Self {
            kv_lora_rank,
            rope_dim,
            latent,
            rope,
        })
    }

    pub fn len(&self) -> usize {
        self.latent.len() / self.kv_lora_rank
    }

    pub fn is_empty(&self) -> bool {
        self.latent.is_empty()
    }

    pub fn stored_f32_elements(&self) -> usize {
        self.latent.len().saturating_add(self.rope.len())
    }

    pub fn clear(&mut self) {
        self.latent.clear();
        self.rope.clear();
    }

    pub(crate) fn truncate(&mut self, tokens: usize) -> Result<(), AttentionError> {
        if tokens > self.len() {
            return Err(AttentionError::CachePosition {
                expected: self.len(),
                got: tokens,
            });
        }
        self.latent.truncate(tokens * self.kv_lora_rank);
        self.rope.truncate(tokens * self.rope_dim);
        Ok(())
    }

    fn latent_row(&self, position: usize) -> &[f32] {
        &self.latent[position * self.kv_lora_rank..(position + 1) * self.kv_lora_rank]
    }

    fn rope_row(&self, position: usize) -> &[f32] {
        &self.rope[position * self.rope_dim..(position + 1) * self.rope_dim]
    }

    fn push(&mut self, latent: &[f32], rope: &[f32]) {
        self.latent.extend_from_slice(latent);
        self.rope.extend_from_slice(rope);
    }
}

/// Scalar, bias-free GLM MLA reference with both algebraically equivalent decode paths.
#[derive(Debug, Clone)]
pub struct MlaAttention {
    geometry: MlaGeometry,
    q_a: WeightMatrix,
    q_a_norm: Vec<f32>,
    q_b: WeightMatrix,
    kv_a: WeightMatrix,
    kv_a_norm: Vec<f32>,
    kv_b: WeightMatrix,
    output: WeightMatrix,
}

impl MlaAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        geometry: MlaGeometry,
        q_a: DenseMatrix,
        q_a_norm: Vec<f32>,
        q_b: DenseMatrix,
        kv_a: DenseMatrix,
        kv_a_norm: Vec<f32>,
        kv_b: DenseMatrix,
        output: DenseMatrix,
    ) -> Result<Self, AttentionError> {
        Self::new_mixed(
            geometry,
            q_a.into(),
            q_a_norm,
            q_b.into(),
            kv_a.into(),
            kv_a_norm,
            kv_b.into(),
            output.into(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_mixed(
        geometry: MlaGeometry,
        q_a: WeightMatrix,
        q_a_norm: Vec<f32>,
        q_b: WeightMatrix,
        kv_a: WeightMatrix,
        kv_a_norm: Vec<f32>,
        kv_b: WeightMatrix,
        output: WeightMatrix,
    ) -> Result<Self, AttentionError> {
        geometry.validate()?;
        let q_head = geometry.q_head_dim()?;
        let q_rows = checked_product("q_b rows", &[geometry.num_heads, q_head])?;
        let kv_head = geometry
            .qk_nope_head_dim
            .checked_add(geometry.v_head_dim)
            .ok_or_else(|| {
                AttentionError::InvalidGeometry("KV head dimension overflows".to_owned())
            })?;
        let kv_rows = checked_product("kv_b rows", &[geometry.num_heads, kv_head])?;
        let context = checked_product(
            "output projection columns",
            &[geometry.num_heads, geometry.v_head_dim],
        )?;
        let compressed = geometry.compressed_elements_per_token()?;

        expect_matrix("q_a", &q_a, geometry.q_lora_rank, geometry.hidden_size)?;
        expect_vector("q_a_norm", &q_a_norm, geometry.q_lora_rank)?;
        expect_matrix("q_b", &q_b, q_rows, geometry.q_lora_rank)?;
        expect_matrix("kv_a", &kv_a, compressed, geometry.hidden_size)?;
        expect_vector("kv_a_norm", &kv_a_norm, geometry.kv_lora_rank)?;
        expect_matrix("kv_b", &kv_b, kv_rows, geometry.kv_lora_rank)?;
        expect_matrix("output", &output, geometry.hidden_size, context)?;

        Ok(Self {
            geometry,
            q_a,
            q_a_norm,
            q_b,
            kv_a,
            kv_a_norm,
            kv_b,
            output,
        })
    }

    pub fn geometry(&self) -> MlaGeometry {
        self.geometry
    }

    pub fn new_cache(&self) -> Result<MlaCache, AttentionError> {
        MlaCache::new(self.geometry.kv_lora_rank, self.geometry.qk_rope_head_dim)
    }

    pub fn new_cache_with_capacity(&self, tokens: usize) -> Result<MlaCache, AttentionError> {
        MlaCache::with_capacity(
            self.geometry.kv_lora_rank,
            self.geometry.qk_rope_head_dim,
            tokens,
        )
    }

    /// Runs consecutive causal tokens and appends them to `cache` in order.
    pub fn forward(
        &self,
        input: &[f32],
        token_count: usize,
        position_base: usize,
        cache: &mut MlaCache,
        mode: AttentionMode,
    ) -> Result<Vec<f32>, AttentionError> {
        let expected =
            checked_product("attention input", &[token_count, self.geometry.hidden_size])?;
        if input.len() != expected {
            return Err(AttentionError::InvalidShape(format!(
                "input needs {expected} values, got {}",
                input.len()
            )));
        }
        if token_count == 0 {
            return Ok(Vec::new());
        }
        let mut result = Vec::with_capacity(expected);
        for (offset, token) in input.chunks_exact(self.geometry.hidden_size).enumerate() {
            result.extend(self.forward_token(token, position_base + offset, cache, mode)?);
        }
        Ok(result)
    }

    pub fn forward_token(
        &self,
        input: &[f32],
        position: usize,
        cache: &mut MlaCache,
        mode: AttentionMode,
    ) -> Result<Vec<f32>, AttentionError> {
        self.validate_cache(cache)?;
        if position != cache.len() {
            return Err(AttentionError::CachePosition {
                expected: cache.len(),
                got: position,
            });
        }

        let q_low = self.q_a.matvec(input)?;
        let q_low = rms_norm(&q_low, &self.q_a_norm, self.geometry.latent_norm_eps)?;
        let mut query = self.q_b.matvec(&q_low)?;
        let q_head = self.geometry.q_head_dim()?;
        for head in 0..self.geometry.num_heads {
            let start = head * q_head + self.geometry.qk_nope_head_dim;
            interleaved_rope(
                &mut query[start..start + self.geometry.qk_rope_head_dim],
                position,
                self.geometry.rope_theta,
            )?;
        }

        let compressed = self.kv_a.matvec(input)?;
        let mut latent = rms_norm(
            &compressed[..self.geometry.kv_lora_rank],
            &self.kv_a_norm,
            self.geometry.latent_norm_eps,
        )?;
        let mut rope = compressed[self.geometry.kv_lora_rank..].to_vec();
        interleaved_rope(&mut rope, position, self.geometry.rope_theta)?;
        cache.push(&latent, &rope);

        let context = match mode {
            AttentionMode::DenseReconstruction => self.dense_context(&query, cache)?,
            AttentionMode::Absorbed => self.absorbed_context(&query, cache)?,
        };
        // Make ownership obvious: neither attention path should retain a token-local buffer.
        latent.clear();
        Ok(self.output.matvec(&context)?)
    }

    fn validate_cache(&self, cache: &MlaCache) -> Result<(), AttentionError> {
        if cache.kv_lora_rank != self.geometry.kv_lora_rank
            || cache.rope_dim != self.geometry.qk_rope_head_dim
        {
            return Err(AttentionError::CacheGeometry {
                expected_latent: self.geometry.kv_lora_rank,
                expected_rope: self.geometry.qk_rope_head_dim,
                got_latent: cache.kv_lora_rank,
                got_rope: cache.rope_dim,
            });
        }
        Ok(())
    }

    fn dense_context(&self, query: &[f32], cache: &MlaCache) -> Result<Vec<f32>, AttentionError> {
        let g = self.geometry;
        let q_head = g.q_head_dim()?;
        let kv_head = g.qk_nope_head_dim + g.v_head_dim;
        let mut reconstructed = Vec::with_capacity(cache.len());
        for position in 0..cache.len() {
            reconstructed.push(self.kv_b.matvec(cache.latent_row(position))?);
        }

        let mut context = Vec::with_capacity(g.num_heads * g.v_head_dim);
        let scale = (q_head as f32).sqrt().recip();
        for head in 0..g.num_heads {
            let q_base = head * q_head;
            let q_nope = &query[q_base..q_base + g.qk_nope_head_dim];
            let q_rope = &query[q_base + g.qk_nope_head_dim..q_base + q_head];
            let kv_base = head * kv_head;
            let mut scores = Vec::with_capacity(cache.len());
            for (position, row) in reconstructed.iter().enumerate() {
                let key = &row[kv_base..kv_base + g.qk_nope_head_dim];
                let score = dot(q_nope, key) + dot(q_rope, cache.rope_row(position));
                scores.push(score * scale);
            }
            softmax_in_place(&mut scores)?;

            for value_dimension in 0..g.v_head_dim {
                let mut sum = 0.0f64;
                for (position, &probability) in scores.iter().enumerate() {
                    let value =
                        reconstructed[position][kv_base + g.qk_nope_head_dim + value_dimension];
                    sum += f64::from(probability) * f64::from(value);
                }
                context.push(sum as f32);
            }
        }
        Ok(context)
    }

    fn absorbed_context(
        &self,
        query: &[f32],
        cache: &MlaCache,
    ) -> Result<Vec<f32>, AttentionError> {
        let g = self.geometry;
        let q_head = g.q_head_dim()?;
        let kv_head = g.qk_nope_head_dim + g.v_head_dim;
        let scale = (q_head as f32).sqrt().recip();
        let mut context = Vec::with_capacity(g.num_heads * g.v_head_dim);

        for head in 0..g.num_heads {
            let q_base = head * q_head;
            let q_nope = &query[q_base..q_base + g.qk_nope_head_dim];
            let q_rope = &query[q_base + g.qk_nope_head_dim..q_base + q_head];
            let kv_base = head * kv_head;

            // q_nope · (W_k L) = (W_k^T q_nope) · L.
            let absorbed_query = self.kv_b.transpose_rows_matvec(kv_base, q_nope)?;

            let mut scores = Vec::with_capacity(cache.len());
            for position in 0..cache.len() {
                let score = dot(&absorbed_query, cache.latent_row(position))
                    + dot(q_rope, cache.rope_row(position));
                scores.push(score * scale);
            }
            softmax_in_place(&mut scores)?;

            let mut mixed_latent = vec![0.0f64; g.kv_lora_rank];
            for (position, &probability) in scores.iter().enumerate() {
                for (accumulator, &value) in mixed_latent.iter_mut().zip(cache.latent_row(position))
                {
                    *accumulator += f64::from(probability) * f64::from(value);
                }
            }
            let mixed_latent: Vec<f32> =
                mixed_latent.into_iter().map(|value| value as f32).collect();
            context.extend(self.kv_b.matvec_rows(
                kv_base + g.qk_nope_head_dim,
                g.v_head_dim,
                &mixed_latent,
            )?);
        }
        Ok(context)
    }
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize, AttentionError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            AttentionError::InvalidGeometry(format!("{name} dimensions overflow: {factors:?}"))
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
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(AttentionError::InvalidShape(format!(
            "{name} contains NaN or infinity"
        )));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn random_values(count: usize, seed: &mut u32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 17;
                *seed ^= *seed << 5;
                ((*seed as f64 / u32::MAX as f64) - 0.5) as f32 * 0.6
            })
            .collect()
    }

    fn random_matrix(rows: usize, columns: usize, seed: &mut u32) -> DenseMatrix {
        DenseMatrix::new(rows, columns, random_values(rows * columns, seed)).unwrap()
    }

    fn tiny_attention() -> MlaAttention {
        let geometry = MlaGeometry {
            hidden_size: 6,
            num_heads: 2,
            q_lora_rank: 4,
            kv_lora_rank: 3,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            v_head_dim: 3,
            latent_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let mut seed = 0x1234_5678;
        MlaAttention::new(
            geometry,
            random_matrix(4, 6, &mut seed),
            vec![0.9, 1.0, 1.1, 1.2],
            random_matrix(8, 4, &mut seed),
            random_matrix(5, 6, &mut seed),
            vec![0.8, 1.0, 1.2],
            random_matrix(10, 3, &mut seed),
            random_matrix(6, 6, &mut seed),
        )
        .unwrap()
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
            assert!((a - b).abs() <= tolerance, "index {index}: {a} versus {b}");
        }
    }

    #[test]
    fn absorbed_and_reconstructed_paths_match() {
        let attention = tiny_attention();
        let mut seed = 0xfeed_beef;
        let input = random_values(4 * 6, &mut seed);
        let mut dense_cache = attention.new_cache().unwrap();
        let dense = attention
            .forward(
                &input,
                4,
                0,
                &mut dense_cache,
                AttentionMode::DenseReconstruction,
            )
            .unwrap();
        let mut absorbed_cache = attention.new_cache().unwrap();
        let absorbed = attention
            .forward(&input, 4, 0, &mut absorbed_cache, AttentionMode::Absorbed)
            .unwrap();
        assert_close(&dense, &absorbed, 2e-6);
    }

    #[test]
    fn cache_has_only_latent_and_shared_rope_key() {
        let attention = tiny_attention();
        let mut cache = attention.new_cache().unwrap();
        let input = vec![0.1; 3 * 6];
        attention
            .forward(&input, 3, 0, &mut cache, AttentionMode::Absorbed)
            .unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stored_f32_elements(), 3 * (3 + 2));
    }

    #[test]
    fn cache_rejects_position_gaps_before_mutating() {
        let attention = tiny_attention();
        let mut cache = attention.new_cache().unwrap();
        let error = attention
            .forward_token(&[0.0; 6], 1, &mut cache, AttentionMode::Absorbed)
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
    fn prefill_then_decode_matches_one_contiguous_call() {
        let attention = tiny_attention();
        let mut seed = 99;
        let input = random_values(5 * 6, &mut seed);
        let mut contiguous_cache = attention.new_cache().unwrap();
        let contiguous = attention
            .forward(&input, 5, 0, &mut contiguous_cache, AttentionMode::Absorbed)
            .unwrap();

        let mut incremental_cache = attention.new_cache().unwrap();
        let mut incremental = attention
            .forward(
                &input[..4 * 6],
                4,
                0,
                &mut incremental_cache,
                AttentionMode::Absorbed,
            )
            .unwrap();
        incremental.extend(
            attention
                .forward_token(
                    &input[4 * 6..],
                    4,
                    &mut incremental_cache,
                    AttentionMode::Absorbed,
                )
                .unwrap(),
        );
        assert_eq!(contiguous, incremental);
    }
}
