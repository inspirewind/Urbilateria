//! Flash-Next-specific HC, PLE and QSA. BF16 materialization follows the reference eager path.
use super::runtime::{Error, Result};
use super::Qwen38FlashNextConfig;
use crate::math::{silu, softmax_in_place};
use crate::model::WeightMatrix;
use crate::models::qwen3_8::math::{linear, round_to_bf16 as bf, round_to_bf16_in_place};
use crate::models::qwen3_8::norm::zero_centered_rms_norm;
use crate::storage::{
    load_compact_bf16_matrix, load_reference_matrix_row, load_reference_values, TensorIndex,
};
use std::collections::BTreeMap;

pub(super) struct Weights {
    matrices: BTreeMap<String, WeightMatrix>,
    vectors: BTreeMap<String, Vec<f32>>,
}
impl Weights {
    pub fn load(index: &TensorIndex, prefix: &str) -> Result<Self> {
        let mut result = Self {
            matrices: BTreeMap::new(),
            vectors: BTreeMap::new(),
        };
        for tensor in index.tensors().filter(|t| t.name.starts_with(prefix)) {
            let name = &tensor.name[prefix.len()..];
            if name.contains("mlp.experts.") || name.contains("ple_embedding.") {
                continue;
            }
            if tensor.shape.len() == 2 {
                result.matrices.insert(
                    name.into(),
                    load_compact_bf16_matrix(
                        index,
                        &tensor.name,
                        tensor.shape[0] as usize,
                        tensor.shape[1] as usize,
                        tensor.data_len,
                    )?,
                );
            } else {
                result
                    .vectors
                    .insert(name.into(), load_reference_values(index, &tensor.name)?);
            }
        }
        Ok(result)
    }
    pub fn matrix(&mut self, name: &str) -> Result<WeightMatrix> {
        self.matrices
            .remove(name)
            .ok_or_else(|| Error(format!("missing matrix {name}")))
    }
    pub fn vector(&mut self, name: &str) -> Result<Vec<f32>> {
        self.vectors
            .remove(name)
            .ok_or_else(|| Error(format!("missing vector {name}")))
    }
    pub fn finish(self) -> Result<()> {
        if !self.matrices.is_empty() || !self.vectors.is_empty() {
            return Err(Error(format!(
                "unconsumed weights: {:?} {:?}",
                self.matrices.keys(),
                self.vectors.keys()
            )));
        }
        Ok(())
    }
}

fn norm(input: &[f32], weight: &[f32], group: usize, eps: f32) -> Result<Vec<f32>> {
    if input.len() != weight.len() || group == 0 || input.len() % group != 0 {
        return Err(Error("invalid grouped RMSNorm geometry".into()));
    }
    let mut output = Vec::with_capacity(input.len());
    for (x, w) in input.chunks_exact(group).zip(weight.chunks_exact(group)) {
        output.extend(zero_centered_rms_norm(x, w, eps)?);
    }
    round_to_bf16_in_place(&mut output)?;
    Ok(output)
}

pub(super) struct HyperConnection {
    norm: Vec<f32>,
    down: WeightMatrix,
    up: WeightMatrix,
    inject: Option<WeightMatrix>,
    hidden: usize,
    count: usize,
    eps: f32,
}
impl HyperConnection {
    pub fn take(
        w: &mut Weights,
        prefix: &str,
        config: &Qwen38FlashNextConfig,
        combine: bool,
    ) -> Result<Self> {
        Ok(Self {
            norm: w.vector(&format!("{prefix}hc_norm.weight"))?,
            down: w.matrix(&format!("{prefix}input_mix_weight_down.weight"))?,
            up: w.matrix(&format!("{prefix}input_mix_weight_up.weight"))?,
            inject: if combine {
                Some(w.matrix(&format!("{prefix}block_inject_weight.weight"))?)
            } else {
                None
            },
            hidden: config.hidden_size,
            count: config.hc_count,
            eps: config.rms_norm_eps as f32,
        })
    }
    pub fn mix(&self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let normalized = norm(input, &self.norm, self.hidden, self.eps)?;
        let mut low = linear(&self.down, &normalized)?;
        for x in &mut low {
            *x = bf(silu(bf(*x / self.count as f32)?))?;
        }
        let gates = linear(&self.up, &low)?;
        let mut mixed = vec![0.0; self.hidden];
        for (i, (&x, &gate)) in normalized.iter().zip(&gates).enumerate() {
            mixed[i % self.hidden] += bf(x * bf(sigmoid(gate))?)?;
        }
        for x in &mut mixed {
            *x = bf(*x / self.count as f32)?;
        }
        let injection = if let Some(w) = &self.inject {
            linear(w, &normalized)?
                .into_iter()
                .map(|v| Ok(bf(2.0 * bf(sigmoid(bf(v / self.count as f32)?))?)?))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok((mixed, injection))
    }
    pub fn combine(&self, residual: &mut [f32], output: &[f32], injection: &[f32]) -> Result<()> {
        if residual.len() != self.count * self.hidden
            || output.len() != self.hidden
            || injection.len() != self.count
        {
            return Err(Error("invalid HC combine geometry".into()));
        }
        for (stream, &gate) in residual.chunks_exact_mut(self.hidden).zip(injection) {
            for (r, &x) in stream.iter_mut().zip(output) {
                *r = bf(*r + bf(x * gate)?)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct PleState {
    pub history: Vec<Vec<f32>>,
    pub tokens: Vec<u32>,
}
impl PleState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            tokens: Vec::new(),
        }
    }
}

pub(super) struct Ple {
    key: WeightMatrix,
    value: WeightMatrix,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    conv: Vec<f32>,
    multipliers: Vec<i64>,
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    prefix: String,
    hidden: usize,
    count: usize,
    eps: f32,
    eos: u32,
    heads_per_ngram: usize,
    ngram_size: usize,
    rows_per_shard: usize,
    embed_width: usize,
    kernel: usize,
}
impl Ple {
    pub fn take(
        w: &mut Weights,
        index: &TensorIndex,
        layer: usize,
        c: &Qwen38FlashNextConfig,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.ple.ple_embedding.");
        let ints = |name: &str, n: usize| -> Result<Vec<i64>> {
            let bytes = index.read_range(&format!("{prefix}{name}"), 0, n * 8)?;
            Ok(bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        let heads = (c.ngram_size - 1) * c.heads_per_ngram;
        let sizes = ints("ngram_heads_vocab_sizes", heads)?;
        let offsets = ints("ngram_heads_offsets", heads)?;
        if sizes.iter().any(|&s| s <= 0)
            || offsets.first() != Some(&0)
            || offsets
                .windows(2)
                .zip(&sizes)
                .any(|(o, s)| o[1] - o[0] != *s)
        {
            return Err(Error("invalid PLE hash table metadata".into()));
        }
        let rows = index
            .require(&format!("{prefix}ngram_embedding.shard_0.weight"))?
            .shape[0] as usize;
        Ok(Self {
            multipliers: ints("layer_multipliers", c.ngram_size)?,
            sizes,
            offsets,
            prefix,
            key: w.matrix("ple.key_proj.weight")?,
            value: w.matrix("ple.value_proj.weight")?,
            norm_key: w.vector("ple.norm_key.weight")?,
            norm_query: w.vector("ple.norm_query.weight")?,
            norm_conv: w.vector("ple.norm_conv.weight")?,
            conv: w.vector("ple.conv1d.weight")?,
            hidden: c.hidden_size,
            count: c.hc_count,
            eps: c.rms_norm_eps as f32,
            eos: c.eos_token_id,
            heads_per_ngram: c.heads_per_ngram,
            ngram_size: c.ngram_size,
            rows_per_shard: rows,
            embed_width: c.ple_embed_dim / heads,
            kernel: c.ple_conv_kernel_size,
        })
    }
    pub fn forward(
        &self,
        index: &TensorIndex,
        input: &[f32],
        token: u32,
        state: &mut PleState,
    ) -> Result<Vec<f32>> {
        let ids = ngram_ids(
            token,
            &state.tokens,
            self.eos,
            &self.multipliers,
            &self.sizes,
            &self.offsets,
            self.heads_per_ngram,
        )?;
        let mut embedded = Vec::with_capacity(ids.len() * self.embed_width);
        for row in ids {
            let shard = row / self.rows_per_shard;
            embedded.extend(load_reference_matrix_row(
                index,
                &format!("{}ngram_embedding.shard_{shard}.weight", self.prefix),
                row % self.rows_per_shard,
                self.rows_per_shard,
                self.embed_width,
            )?);
        }
        let output = self.forward_embedded(input, &embedded, state)?;
        if token == self.eos {
            state.tokens.clear();
        } else {
            state.tokens.push(token);
            if state.tokens.len() >= self.ngram_size {
                state.tokens.remove(0);
            }
        }
        Ok(output)
    }
    fn forward_embedded(
        &self,
        input: &[f32],
        embedded: &[f32],
        state: &mut PleState,
    ) -> Result<Vec<f32>> {
        let key = norm(
            &linear(&self.key, embedded)?,
            &self.norm_key,
            self.hidden,
            self.eps,
        )?;
        let query = norm(input, &self.norm_query, self.hidden, self.eps)?;
        let value = linear(&self.value, embedded)?;
        let mut gated = Vec::with_capacity(self.hidden * self.count);
        for (k, q) in key
            .chunks_exact(self.hidden)
            .zip(query.chunks_exact(self.hidden))
        {
            let mut score = 0.0;
            for (&k, &q) in k.iter().zip(q) {
                score += bf(k * q)?;
            }
            score = bf(bf(score)? / (self.hidden as f32).sqrt())?;
            let root = bf(bf(score.abs().max(1e-6))?.sqrt())?;
            let gate = bf(sigmoid(bf(
                root * if score == 0.0 { 0.0 } else { score.signum() }
            )?))?;
            for &v in &value {
                gated.push(bf(gate * v)?);
            }
        }
        let normalized = norm(&gated, &self.norm_conv, self.hidden, self.eps)?;
        let history_len = (self.kernel - 1) * self.ngram_size;
        for (channel, output) in gated.iter_mut().enumerate() {
            let mut sum = normalized[channel] * self.conv[channel * self.kernel + self.kernel - 1];
            for tap in 0..self.kernel - 1 {
                let lag = (self.kernel - 1 - tap) * self.ngram_size;
                if state.history.len() >= lag {
                    sum += state.history[state.history.len() - lag][channel]
                        * self.conv[channel * self.kernel + tap];
                }
            }
            *output = bf(*output + bf(silu(bf(sum)?))?)?;
        }
        if state.history.len() == history_len {
            state.history.remove(0);
        }
        state.history.push(normalized);
        Ok(gated)
    }
}

pub(super) fn ngram_ids(
    token: u32,
    history: &[u32],
    eos: u32,
    multipliers: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    heads: usize,
) -> Result<Vec<usize>> {
    if multipliers.len() < 2
        || heads == 0
        || sizes.len() != (multipliers.len() - 1) * heads
        || offsets.len() != sizes.len()
    {
        return Err(Error("invalid n-gram geometry".into()));
    }
    let mut mixed = (token as i64).wrapping_mul(multipliers[0]);
    let mut result = Vec::new();
    let segment = history
        .iter()
        .rposition(|&v| v == eos)
        .map_or(history, |i| &history[i + 1..]);
    for (lag, &multiplier) in multipliers.iter().enumerate().skip(1) {
        let previous = segment.len().checked_sub(lag).map_or(eos, |i| segment[i]);
        mixed ^= (previous as i64).wrapping_mul(multiplier);
        for head in (lag - 1) * heads..lag * heads {
            if sizes[head] <= 0 || offsets[head] < 0 {
                return Err(Error("invalid n-gram metadata".into()));
            }
            result.push(
                mixed
                    .rem_euclid(sizes[head])
                    .checked_add(offsets[head])
                    .ok_or_else(|| Error("n-gram row overflows".into()))? as usize,
            );
        }
    }
    Ok(result)
}

#[derive(Clone)]
pub(super) struct QsaState {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub blocks: Vec<Vec<f32>>,
    pub tail: Vec<Vec<f32>>,
}
impl QsaState {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            blocks: Vec::new(),
            tail: Vec::new(),
        }
    }
    pub fn truncate(&mut self, position: usize, blocks: usize, tail: Vec<Vec<f32>>) {
        self.keys.truncate(position);
        self.values.truncate(position);
        self.blocks.truncate(blocks);
        self.tail = tail;
    }
}
pub(super) struct Qsa {
    q: WeightMatrix,
    k: WeightMatrix,
    v: WeightMatrix,
    out: WeightMatrix,
    index: WeightMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    iq_norm: Vec<f32>,
    ik_norm: Vec<f32>,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    index_heads: usize,
    index_dim: usize,
    ratio: usize,
    budget: usize,
    rotary: usize,
    theta: f32,
    eps: f32,
}
impl Qsa {
    pub fn take(w: &mut Weights, c: &Qwen38FlashNextConfig) -> Result<Self> {
        Ok(Self {
            q: w.matrix("self_attn.q_proj.weight")?,
            k: w.matrix("self_attn.k_proj.weight")?,
            v: w.matrix("self_attn.v_proj.weight")?,
            out: w.matrix("self_attn.o_proj.weight")?,
            index: w.matrix("self_attn.indexer.index_qk_proj.weight")?,
            q_norm: w.vector("self_attn.q_norm.weight")?,
            k_norm: w.vector("self_attn.k_norm.weight")?,
            iq_norm: w.vector("self_attn.indexer.q_layernorm.weight")?,
            ik_norm: w.vector("self_attn.indexer.k_layernorm.weight")?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            dim: c.head_dim,
            index_heads: c.indexer_n_heads,
            index_dim: c.indexer_head_dim,
            ratio: c.indexer_compress_ratio,
            budget: c.indexer_budget,
            rotary: (c.head_dim as f64 * c.partial_rotary_factor) as usize,
            theta: c.rope_parameters.rope_theta as f32,
            eps: c.rms_norm_eps as f32,
        })
    }
    fn rope(&self, x: &mut [f32], position: usize) -> Result<()> {
        // QSA uses the SAME 64 rotary dimensions as GQA, even though its head dimension is 128.
        let half = self.rotary / 2;
        for i in 0..half {
            let angle = position as f32 * self.theta.powf(-2.0 * i as f32 / self.rotary as f32);
            let (sin, cos) = angle.sin_cos();
            // The reference rotary operator materializes each BF16 product before addition.
            let (a, b) = (x[i], x[i + half]);
            let (sin, cos) = (bf(sin)?, bf(cos)?);
            x[i] = bf(bf(a * cos)? - bf(b * sin)?)?;
            x[i + half] = bf(bf(b * cos)? + bf(a * sin)?)?;
        }
        round_to_bf16_in_place(x)?;
        Ok(())
    }
    pub fn forward(
        &self,
        input: &[f32],
        position: usize,
        state: &mut QsaState,
    ) -> Result<Vec<f32>> {
        if state.keys.len() != position {
            return Err(Error("QSA cache position mismatch".into()));
        }
        let iqk = linear(&self.index, input)?;
        let mut iq = Vec::new();
        for h in iqk[..self.index_heads * self.index_dim].chunks_exact(self.index_dim) {
            let mut q = norm(h, &self.iq_norm, self.index_dim, self.eps)?;
            self.rope(&mut q, position)?;
            iq.push(q);
        }
        state
            .tail
            .push(iqk[self.index_heads * self.index_dim..].to_vec());
        if state.tail.len() == self.ratio {
            let mut pooled = vec![0.0; self.index_dim];
            for k in &state.tail {
                for (p, &v) in pooled.iter_mut().zip(k) {
                    *p += v;
                }
            }
            for p in &mut pooled {
                *p = bf(*p / self.ratio as f32)?;
            }
            pooled = norm(&pooled, &self.ik_norm, self.index_dim, self.eps)?;
            self.rope(&mut pooled, position + 1 - self.ratio)?;
            state.blocks.push(pooled);
            state.tail.clear();
        }
        let selected = select_blocks(
            &iq,
            &state.blocks,
            self.budget / self.ratio,
            self.ratio,
            position + 1,
        );
        let qg = linear(&self.q, input)?;
        let kp = linear(&self.k, input)?;
        let value = linear(&self.v, input)?;
        let mut key = Vec::new();
        for h in kp.chunks_exact(self.dim) {
            let mut k = norm(h, &self.k_norm, self.dim, self.eps)?;
            self.rope(&mut k, position)?;
            key.extend(k);
        }
        state.keys.push(key);
        state.values.push(value);
        let mut context = Vec::with_capacity(self.heads * self.dim);
        for (head, qg) in qg.chunks_exact(2 * self.dim).enumerate() {
            let mut query = norm(&qg[..self.dim], &self.q_norm, self.dim, self.eps)?;
            self.rope(&mut query, position)?;
            let start = (head / (self.heads / self.kv_heads)) * self.dim;
            let mut scores = Vec::with_capacity(selected.len());
            for &t in &selected {
                let score = query
                    .iter()
                    .zip(&state.keys[t][start..start + self.dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                scores.push(bf(bf(score)? / (self.dim as f32).sqrt())?);
            }
            softmax_in_place(&mut scores)?;
            round_to_bf16_in_place(&mut scores)?;
            let mut out = vec![0.0; self.dim];
            for (&t, &prob) in selected.iter().zip(&scores) {
                for (o, &v) in out
                    .iter_mut()
                    .zip(&state.values[t][start..start + self.dim])
                {
                    *o += prob * v;
                }
            }
            for (v, &gate) in out.iter_mut().zip(&qg[self.dim..]) {
                *v = bf(bf(*v)? * bf(sigmoid(gate))?)?;
            }
            context.extend(out);
        }
        Ok(linear(&self.out, &context)?)
    }
}

pub(super) fn select_blocks(
    query: &[Vec<f32>],
    blocks: &[Vec<f32>],
    topk: usize,
    ratio: usize,
    length: usize,
) -> Vec<usize> {
    let mut ranked = blocks
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let score = query
                .iter()
                .map(|q| q.iter().zip(k).map(|(q, k)| q * k).sum::<f32>().max(0.0))
                .sum::<f32>();
            (i, score / (k.len() as f32).sqrt())
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut selected = ranked
        .iter()
        .take(topk)
        .flat_map(|(i, _)| i * ratio..(i + 1) * ratio)
        .collect::<Vec<_>>();
    selected.extend(blocks.len() * ratio..length);
    // Reference scatters to a mask, so attention reduces in original token order.
    selected.sort_unstable();
    selected
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Bf16Matrix;
    use crate::models::qwen3_8::linear_attention::{DeltaNetGeometry, GatedDeltaNet};
    use serde_json::Value;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/qwen3_8_flash_next_tiny.json"
        ))
        .unwrap()
    }
    fn values(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn weights(v: &Value, prefix: &str) -> Weights {
        let mut w = Weights {
            matrices: BTreeMap::new(),
            vectors: BTreeMap::new(),
        };
        for (name, t) in v.as_object().unwrap() {
            let shape = t["shape"].as_array().unwrap();
            let data = values(&t["values"]);
            if shape.len() == 2 {
                let bytes = data
                    .iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect();
                w.matrices.insert(
                    format!("{prefix}{name}"),
                    WeightMatrix::Bf16(
                        Bf16Matrix::from_le_bytes(
                            shape[0].as_u64().unwrap() as usize,
                            shape[1].as_u64().unwrap() as usize,
                            bytes,
                        )
                        .unwrap(),
                    ),
                );
            } else {
                w.vectors.insert(format!("{prefix}{name}"), data);
            }
        }
        w
    }
    fn config() -> Qwen38FlashNextConfig {
        let mut c =
            Qwen38FlashNextConfig::from_json_str(include_str!("release_config.json")).unwrap();
        c.text_config.hidden_size = 4;
        c.text_config.hc_count = 2;
        c.text_config.hc_lowrank = 3;
        c.text_config.head_dim = 8;
        c.text_config.num_attention_heads = 2;
        c.text_config.num_key_value_heads = 1;
        c.text_config.indexer_n_heads = 2;
        c.text_config.indexer_head_dim = 8;
        c.text_config.indexer_budget = 4;
        c.text_config.indexer_compress_ratio = 2;
        c.text_config.partial_rotary_factor = 0.5;
        c.text_config.rope_parameters.rope_theta = 10000.;
        c
    }
    fn check(actual: &[f32], expected: &Value, label: &str) {
        let expected = values(expected);
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - e).abs() <= 0.00006 + e.abs() * 0.008,
                "{label}[{i}]: {a} != {e}"
            );
        }
    }
    #[test]
    fn hc_matches_independent_bf16_reference() {
        let f = fixture();
        let f = &f["hc"];
        let mut w = weights(&f["weights"], "");
        let hc = HyperConnection::take(&mut w, "", &config(), true).unwrap();
        w.finish().unwrap();
        for (i, input) in f["inputs"].as_array().unwrap().iter().enumerate() {
            let (mixed, injection) = hc.mix(&values(input)).unwrap();
            check(&mixed, &f["mixed"][i], "HC mix");
            check(&injection, &f["injection"][i], "HC inject");
        }
    }
    #[test]
    fn qsa_matches_reference_through_block_selection_and_tail() {
        let f = fixture();
        let f = &f["qsa"];
        let mut w = weights(&f["weights"], "self_attn.");
        let qsa = Qsa::take(&mut w, &config()).unwrap();
        w.finish().unwrap();
        let mut state = QsaState::new();
        for (i, input) in f["inputs"].as_array().unwrap().iter().enumerate() {
            check(
                &qsa.forward(&values(input), i, &mut state).unwrap(),
                &f["outputs"][i],
                &format!("QSA token {i}"),
            );
        }
    }
    #[test]
    fn delta_sigmoid_matches_reference_without_changing_silu_default() {
        let f = fixture();
        let f = &f["delta"];
        let mut w = weights(&f["weights"], "");
        let delta = GatedDeltaNet::new_mixed(
            DeltaNetGeometry {
                hidden_size: 4,
                num_key_heads: 1,
                num_value_heads: 2,
                key_head_dim: 4,
                value_head_dim: 4,
                conv_kernel_size: 3,
                norm_eps: 1e-6,
            },
            w.matrix("in_proj_qkv.weight").unwrap(),
            w.matrix("in_proj_z.weight").unwrap(),
            w.matrix("in_proj_b.weight").unwrap(),
            w.matrix("in_proj_a.weight").unwrap(),
            w.vector("conv1d.weight").unwrap(),
            w.vector("dt_bias").unwrap(),
            w.vector("A_log").unwrap(),
            w.vector("norm.weight").unwrap(),
            w.matrix("out_proj.weight").unwrap(),
        )
        .unwrap()
        .with_flash_next_semantics();
        w.finish().unwrap();
        let mut state = delta.new_state().unwrap();
        for (i, input) in f["inputs"].as_array().unwrap().iter().enumerate() {
            check(
                &delta.forward_token(&values(input), &mut state).unwrap(),
                &f["outputs"][i],
                &format!("Delta token {i}"),
            );
        }
    }
    #[test]
    fn ple_matches_reference_across_all_dilated_convolution_taps() {
        let f = fixture();
        let f = &f["ple"];
        let mut w = weights(&f["weights"], "");
        let ple = Ple {
            key: w.matrix("key_proj.weight").unwrap(),
            value: w.matrix("value_proj.weight").unwrap(),
            norm_key: w.vector("norm_key.weight").unwrap(),
            norm_query: w.vector("norm_query.weight").unwrap(),
            norm_conv: w.vector("norm_conv.weight").unwrap(),
            conv: w.vector("conv1d.weight").unwrap(),
            multipliers: vec![],
            sizes: vec![],
            offsets: vec![],
            prefix: String::new(),
            hidden: 4,
            count: 2,
            eps: 1e-6,
            eos: 0,
            heads_per_ngram: 1,
            ngram_size: 3,
            rows_per_shard: 0,
            embed_width: 4,
            kernel: 4,
        };
        w.finish().unwrap();
        let mut state = PleState::new();
        for (i, input) in f["inputs"].as_array().unwrap().iter().enumerate() {
            check(
                &ple.forward_embedded(&values(input), &values(&f["embeddings"][i]), &mut state)
                    .unwrap(),
                &f["outputs"][i],
                &format!("PLE token {i}"),
            );
        }
    }
    #[test]
    fn ngram_hash_respects_eos_boundaries_and_signed_remainder() {
        let m = [i64::MAX, 5, 7];
        let sizes = [17, 19];
        let offsets = [0, 17];
        let ids = ngram_ids(3, &[8, 99, 2], 99, &m, &sizes, &offsets, 1).unwrap();
        assert_eq!(
            ids,
            ngram_ids(3, &[2], 99, &m, &sizes, &offsets, 1).unwrap()
        );
        let mixed = 3i64.wrapping_mul(m[0]) ^ 2i64.wrapping_mul(m[1]);
        assert_eq!(
            ids,
            vec![
                mixed.rem_euclid(17) as usize,
                ((mixed ^ (99 * 7)).rem_euclid(19) + 17) as usize
            ]
        );
        assert_ne!(
            ids,
            ngram_ids(3, &[8, 2], 99, &m, &sizes, &offsets, 1).unwrap()
        );
    }
    #[test]
    fn qsa_keeps_incomplete_tail_and_original_reduction_order() {
        let q = vec![vec![1., 0.]];
        let k = vec![vec![1., 0.], vec![3., 0.], vec![2., 0.]];
        assert_eq!(select_blocks(&q, &k, 1, 2, 7), [2, 3, 6]);
        assert_eq!(select_blocks(&q, &k, 2, 2, 7), [2, 3, 4, 5, 6]);
    }
}
