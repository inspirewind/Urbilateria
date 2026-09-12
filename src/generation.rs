//! Backend-independent autoregressive decoding.
//!
//! The generator owns sampling and stop semantics, while a backend owns model state and logits.
//! Keeping this boundary small lets every family-specific bounded-memory runtime use exactly the
//! same token loop.

use crate::profiling::{span, span_with_work, ProfileStage};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

/// Minimal causal model contract needed by autoregressive generation.
pub trait CausalDecoder {
    type State;
    type Error: Error;

    fn new_state(&self) -> Result<Self::State, Self::Error>;
    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error>;

    /// Advances an entire prompt and returns only the logits for its final token.
    ///
    /// The default preserves the token-at-a-time execution contract. Backends with streamed
    /// weights may override this to execute the prompt layer by layer, avoiding repeated weight
    /// loads and intermediate LM-head projections.
    fn prefill(&self, prompt: &[u32], state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        let mut logits = Vec::new();
        for &token in prompt {
            let _profile = span(ProfileStage::Prefill);
            logits = self.forward_token(token, state)?;
        }
        Ok(logits)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodeStrategy {
    Greedy,
    TopP {
        temperature: f32,
        top_p: f32,
        seed: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub strategy: DecodeStrategy,
    /// Optional vocabulary-sized mask. `false` entries can never be sampled.
    ///
    /// This is used when a model's LM head contains reserved rows that are absent from the
    /// tokenizer. Keeping the restriction in the sampler prevents an undecodable token from
    /// being selected in the first place.
    pub allowed_token_mask: Option<Vec<bool>>,
}

impl GenerationConfig {
    pub fn greedy(max_new_tokens: usize, eos_token_ids: Vec<u32>) -> Self {
        Self {
            max_new_tokens,
            eos_token_ids,
            strategy: DecodeStrategy::Greedy,
            allowed_token_mask: None,
        }
    }

    pub fn validate(&self) -> Result<(), GenerationConfigError> {
        if self.max_new_tokens == 0 {
            return Err(GenerationConfigError(
                "max_new_tokens must be greater than zero".to_owned(),
            ));
        }
        if self.max_new_tokens > 1_000_000 {
            return Err(GenerationConfigError(
                "max_new_tokens must not exceed 1,000,000".to_owned(),
            ));
        }
        let mut eos = self.eos_token_ids.clone();
        eos.sort_unstable();
        if eos.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(GenerationConfigError(
                "eos_token_ids must not contain duplicates".to_owned(),
            ));
        }
        if let DecodeStrategy::TopP {
            temperature, top_p, ..
        } = self.strategy
        {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(GenerationConfigError(
                    "temperature must be finite and greater than zero".to_owned(),
                ));
            }
            if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
                return Err(GenerationConfigError(
                    "top_p must be finite and in (0, 1]".to_owned(),
                ));
            }
        }
        if self
            .allowed_token_mask
            .as_ref()
            .is_some_and(|mask| !mask.iter().any(|allowed| *allowed))
        {
            return Err(GenerationConfigError(
                "allowed_token_mask must permit at least one token".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationConfigError(pub String);

impl fmt::Display for GenerationConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for GenerationConfigError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Eos(u32),
    MaxNewTokens,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    pub generated_tokens: Vec<u32>,
    pub stop_reason: StopReason,
}

#[derive(Debug)]
pub enum GenerationError<E, C = Infallible> {
    Config(GenerationConfigError),
    EmptyPrompt,
    InvalidLogits(String),
    Model(E),
    Callback(C),
}

impl<E: fmt::Display, C: fmt::Display> fmt::Display for GenerationError<E, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => write!(f, "invalid generation config: {error}"),
            Self::EmptyPrompt => f.write_str("generation prompt must contain at least one token"),
            Self::InvalidLogits(reason) => write!(f, "invalid model logits: {reason}"),
            Self::Model(error) => write!(f, "model execution failed: {error}"),
            Self::Callback(error) => write!(f, "generation callback failed: {error}"),
        }
    }
}

impl<E: Error + 'static, C: Error + 'static> Error for GenerationError<E, C> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Model(error) => Some(error),
            Self::Callback(error) => Some(error),
            Self::EmptyPrompt | Self::InvalidLogits(_) => None,
        }
    }
}

/// Generates tokens and calls `on_token` immediately after every selection, including EOS.
pub fn generate<M, F>(
    model: &M,
    prompt: &[u32],
    config: &GenerationConfig,
    mut on_token: F,
) -> Result<GenerationOutput, GenerationError<M::Error>>
where
    M: CausalDecoder,
    F: FnMut(u32),
{
    try_generate(model, prompt, config, |token| {
        on_token(token);
        Ok::<(), Infallible>(())
    })
}

/// Fallible form of [`generate`], used by streaming frontends so decode or output errors stop
/// model execution immediately instead of wasting another expensive forward pass.
pub fn try_generate<M, F, C>(
    model: &M,
    prompt: &[u32],
    config: &GenerationConfig,
    on_token: F,
) -> Result<GenerationOutput, GenerationError<M::Error, C>>
where
    M: CausalDecoder,
    F: FnMut(u32) -> Result<(), C>,
{
    let mut state = model.new_state().map_err(GenerationError::Model)?;
    try_generate_with_state(model, prompt, config, &mut state, on_token)
}

/// Fallible generation using caller-owned model state, allowing runtimes to report cache and
/// routing telemetry after the token loop.
///
/// On success, `state` contains the prompt and every generated token except the final selected
/// token. Advancing that final token would require an otherwise-unused forward pass after EOS or
/// `max_new_tokens`. To continue later, pass the final token back as the next suffix token.
pub fn try_generate_with_state<M, F, C>(
    model: &M,
    prompt: &[u32],
    config: &GenerationConfig,
    state: &mut M::State,
    mut on_token: F,
) -> Result<GenerationOutput, GenerationError<M::Error, C>>
where
    M: CausalDecoder,
    F: FnMut(u32) -> Result<(), C>,
{
    config.validate().map_err(GenerationError::Config)?;
    if prompt.is_empty() {
        return Err(GenerationError::EmptyPrompt);
    }

    let mut time_to_first_token = Some(span(ProfileStage::TimeToFirstToken));
    let mut logits = model
        .prefill(prompt, state)
        .map_err(GenerationError::Model)?;
    validate_logits(&logits)?;
    validate_decode_domain(&logits, config.allowed_token_mask.as_deref())?;
    validate_eos_domain(&logits, config)?;

    let mut rng = match config.strategy {
        DecodeStrategy::Greedy => None,
        DecodeStrategy::TopP { seed, .. } => Some(SplitMix64::new(seed)),
    };
    let mut generated = Vec::with_capacity(config.max_new_tokens);
    for step in 0..config.max_new_tokens {
        let token = select_token(
            &logits,
            &config.strategy,
            config.allowed_token_mask.as_deref(),
            rng.as_mut(),
        )?;
        generated.push(token);
        drop(time_to_first_token.take());
        on_token(token).map_err(GenerationError::Callback)?;
        if config.eos_token_ids.contains(&token) {
            return Ok(GenerationOutput {
                generated_tokens: generated,
                stop_reason: StopReason::Eos(token),
            });
        }
        if step + 1 < config.max_new_tokens {
            let _profile = span_with_work(ProfileStage::Decode, 1);
            logits = model
                .forward_token(token, state)
                .map_err(GenerationError::Model)?;
            drop(_profile);
            validate_logits(&logits)?;
            validate_decode_domain(&logits, config.allowed_token_mask.as_deref())?;
            validate_eos_domain(&logits, config)?;
        }
    }
    Ok(GenerationOutput {
        generated_tokens: generated,
        stop_reason: StopReason::MaxNewTokens,
    })
}

fn validate_logits<E, C>(logits: &[f32]) -> Result<(), GenerationError<E, C>> {
    if logits.is_empty() {
        return Err(GenerationError::InvalidLogits(
            "the vocabulary is empty".to_owned(),
        ));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(GenerationError::InvalidLogits(
            "NaN or infinity was produced".to_owned(),
        ));
    }
    if logits.len() > u32::MAX as usize {
        return Err(GenerationError::InvalidLogits(
            "vocabulary size does not fit u32 token IDs".to_owned(),
        ));
    }
    Ok(())
}

fn select_token<E, C>(
    logits: &[f32],
    strategy: &DecodeStrategy,
    allowed_token_mask: Option<&[bool]>,
    rng: Option<&mut SplitMix64>,
) -> Result<u32, GenerationError<E, C>> {
    validate_logits(logits)?;
    validate_decode_domain(logits, allowed_token_mask)?;
    match strategy {
        DecodeStrategy::Greedy => logits
            .iter()
            .enumerate()
            .filter(|(id, _)| allowed_token_mask.is_none_or(|mask| mask[*id]))
            .max_by(|(left_id, left), (right_id, right)| {
                left.total_cmp(right).then_with(|| right_id.cmp(left_id))
            })
            .map(|(id, _)| id as u32)
            .ok_or_else(|| GenerationError::InvalidLogits("the vocabulary is empty".to_owned())),
        DecodeStrategy::TopP {
            temperature, top_p, ..
        } => sample_top_p(
            logits,
            *temperature,
            *top_p,
            allowed_token_mask,
            rng.ok_or_else(|| {
                GenerationError::InvalidLogits("top-p sampler has no RNG state".to_owned())
            })?,
        ),
    }
}

fn sample_top_p<E, C>(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    allowed_token_mask: Option<&[bool]>,
    rng: &mut SplitMix64,
) -> Result<u32, GenerationError<E, C>> {
    let inverse_temperature = 1.0f64 / f64::from(temperature);
    let mut ranked: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .filter(|(id, _)| allowed_token_mask.is_none_or(|mask| mask[*id]))
        .map(|(id, &logit)| (id as u32, f64::from(logit) * inverse_temperature))
        .collect();
    ranked.sort_unstable_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    let maximum = ranked[0].1;
    for (_, value) in &mut ranked {
        *value = (*value - maximum).exp();
    }
    let total: f64 = ranked.iter().map(|(_, weight)| *weight).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(GenerationError::InvalidLogits(
            "softmax normalization is not finite and positive".to_owned(),
        ));
    }
    let threshold = total * f64::from(top_p);
    let mut kept_sum = 0.0f64;
    let mut kept = 0usize;
    for (_, weight) in &ranked {
        kept_sum += *weight;
        kept += 1;
        if kept_sum >= threshold {
            break;
        }
    }
    let draw = rng.unit_f64() * kept_sum;
    let mut cumulative = 0.0f64;
    for &(token, weight) in &ranked[..kept] {
        cumulative += weight;
        if draw < cumulative {
            return Ok(token);
        }
    }
    Ok(ranked[kept - 1].0)
}

fn validate_decode_domain<E, C>(
    logits: &[f32],
    allowed_token_mask: Option<&[bool]>,
) -> Result<(), GenerationError<E, C>> {
    if let Some(mask) = allowed_token_mask {
        if mask.len() != logits.len() {
            return Err(GenerationError::InvalidLogits(format!(
                "allowed token mask has {} entries for {} logits",
                mask.len(),
                logits.len()
            )));
        }
        if !mask.iter().any(|allowed| *allowed) {
            return Err(GenerationError::InvalidLogits(
                "allowed token mask rejects the complete vocabulary".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_eos_domain<E, C>(
    logits: &[f32],
    config: &GenerationConfig,
) -> Result<(), GenerationError<E, C>> {
    for &token in &config.eos_token_ids {
        let id = token as usize;
        if id >= logits.len() {
            return Err(GenerationError::InvalidLogits(format!(
                "EOS token ID {token} is outside the {}-row vocabulary",
                logits.len()
            )));
        }
        if config
            .allowed_token_mask
            .as_ref()
            .is_some_and(|mask| !mask[id])
        {
            return Err(GenerationError::InvalidLogits(format!(
                "EOS token ID {token} is suppressed by the allowed token mask"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::{ProfileSession, ProfileStage};
    use std::cell::Cell;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ScriptError(&'static str);

    impl fmt::Display for ScriptError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for ScriptError {}

    struct ScriptedModel;

    impl CausalDecoder for ScriptedModel {
        type State = Vec<u32>;
        type Error = ScriptError;

        fn new_state(&self) -> Result<Self::State, Self::Error> {
            Ok(Vec::new())
        }

        fn forward_token(
            &self,
            token: u32,
            state: &mut Self::State,
        ) -> Result<Vec<f32>, Self::Error> {
            state.push(token);
            Ok(match token {
                1 => vec![0.0, 0.0, 4.0, -1.0],
                2 => vec![0.0, 0.0, -1.0, 4.0],
                _ => vec![0.0; 4],
            })
        }
    }

    #[test]
    fn greedy_prefills_then_stops_on_eos() {
        let mut streamed = Vec::new();
        let output = generate(
            &ScriptedModel,
            &[1],
            &GenerationConfig::greedy(8, vec![3]),
            |token| streamed.push(token),
        )
        .unwrap();
        assert_eq!(streamed, vec![2, 3]);
        assert_eq!(output.generated_tokens, vec![2, 3]);
        assert_eq!(output.stop_reason, StopReason::Eos(3));
    }

    #[test]
    fn profiling_counts_only_actual_prefill_and_decode_forwards() {
        let profile = ProfileSession::start();
        generate(
            &ScriptedModel,
            &[1, 2],
            &GenerationConfig::greedy(3, Vec::new()),
            |_| {},
        )
        .unwrap();
        let report = profile.finish();
        assert_eq!(report.stage(ProfileStage::Prefill).unwrap().calls, 2);
        let decode = report.stage(ProfileStage::Decode).unwrap();
        assert_eq!(decode.calls, 2);
        assert_eq!(decode.work_items, 2);
        assert_eq!(
            report.stage(ProfileStage::TimeToFirstToken).unwrap().calls,
            1
        );

        let profile = ProfileSession::start();
        generate(
            &ScriptedModel,
            &[1],
            &GenerationConfig::greedy(1, Vec::new()),
            |_| {},
        )
        .unwrap();
        let report = profile.finish();
        assert_eq!(report.stage(ProfileStage::Prefill).unwrap().calls, 1);
        assert!(report.stage(ProfileStage::Decode).is_none());
        assert_eq!(
            report.stage(ProfileStage::TimeToFirstToken).unwrap().calls,
            1
        );
    }

    #[test]
    fn greedy_ties_choose_lower_id_and_max_tokens_stops() {
        let output = generate(
            &ScriptedModel,
            &[0],
            &GenerationConfig::greedy(2, Vec::new()),
            |_| {},
        )
        .unwrap();
        assert_eq!(output.generated_tokens, vec![0, 0]);
        assert_eq!(output.stop_reason, StopReason::MaxNewTokens);
    }

    #[test]
    fn top_p_is_reproducible_for_a_fixed_seed() {
        let config = GenerationConfig {
            max_new_tokens: 12,
            eos_token_ids: Vec::new(),
            strategy: DecodeStrategy::TopP {
                temperature: 0.8,
                top_p: 0.95,
                seed: 1234,
            },
            allowed_token_mask: None,
        };
        let left = generate(&ScriptedModel, &[0], &config, |_| {}).unwrap();
        let right = generate(&ScriptedModel, &[0], &config, |_| {}).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn rejects_empty_prompt_and_invalid_sampling_config() {
        assert!(matches!(
            generate(
                &ScriptedModel,
                &[],
                &GenerationConfig::greedy(1, Vec::new()),
                |_| {}
            ),
            Err(GenerationError::EmptyPrompt)
        ));
        let invalid = GenerationConfig {
            max_new_tokens: 1,
            eos_token_ids: Vec::new(),
            strategy: DecodeStrategy::TopP {
                temperature: 1.0,
                top_p: 0.0,
                seed: 0,
            },
            allowed_token_mask: None,
        };
        assert!(matches!(
            generate(&ScriptedModel, &[1], &invalid, |_| {}),
            Err(GenerationError::Config(_))
        ));
    }

    #[test]
    fn fallible_callback_stops_before_another_forward() {
        struct CountingModel(Cell<usize>);
        impl CausalDecoder for CountingModel {
            type State = ();
            type Error = ScriptError;

            fn new_state(&self) -> Result<Self::State, Self::Error> {
                Ok(())
            }

            fn forward_token(
                &self,
                _token: u32,
                _state: &mut Self::State,
            ) -> Result<Vec<f32>, Self::Error> {
                self.0.set(self.0.get() + 1);
                Ok(vec![1.0, 0.0])
            }
        }

        let model = CountingModel(Cell::new(0));
        let result = try_generate(
            &model,
            &[1],
            &GenerationConfig::greedy(4, Vec::new()),
            |_| Err(ScriptError("output closed")),
        );
        assert!(matches!(
            result,
            Err(GenerationError::Callback(ScriptError("output closed")))
        ));
        assert_eq!(model.0.get(), 1);
    }

    #[test]
    fn token_mask_prevents_sampling_undecodable_rows() {
        let mut config = GenerationConfig::greedy(1, Vec::new());
        config.allowed_token_mask = Some(vec![true, true, false, true]);
        let output = generate(&ScriptedModel, &[1], &config, |_| {}).unwrap();
        assert_eq!(output.generated_tokens, vec![0]);

        config.allowed_token_mask = Some(vec![true; 3]);
        assert!(matches!(
            generate(&ScriptedModel, &[1], &config, |_| {}),
            Err(GenerationError::InvalidLogits(reason)) if reason.contains("mask")
        ));

        config.eos_token_ids = vec![2];
        config.allowed_token_mask = Some(vec![true, true, false, true]);
        assert!(matches!(
            generate(&ScriptedModel, &[1], &config, |_| {}),
            Err(GenerationError::InvalidLogits(reason)) if reason.contains("EOS")
        ));
    }

    #[test]
    fn caller_owned_state_excludes_only_the_final_selected_token() {
        let mut state = Vec::new();
        let output = try_generate_with_state(
            &ScriptedModel,
            &[1],
            &GenerationConfig::greedy(2, Vec::new()),
            &mut state,
            |_| Ok::<(), Infallible>(()),
        )
        .unwrap();
        assert_eq!(output.generated_tokens, vec![2, 3]);
        assert_eq!(state, vec![1, 2]);
    }
}
