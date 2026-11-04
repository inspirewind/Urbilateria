//! A single resident model and exact, token-validated conversation cache.

use crate::{chat::EncodedPrompt, progress, StreamError};
use std::any::Any;
use std::error::Error;
use std::io::Write;
use urbilateria::analysis::ResourcePlan;
use urbilateria::generation::{
    try_generate_with_state, CausalDecoder, GenerationConfig, GenerationError, StopReason,
};
use urbilateria::models::kimi_k3::tokenizer::KimiK3StreamingDecoder;
use urbilateria::profiling::{span, ProfileStage};
use urbilateria::runtime::session::SessionState;
use urbilateria::runtime::{
    session::observe_experts,
    telemetry::{self, Event as RuntimeEvent},
};
use urbilateria::tokenizer::{StreamingDecoder, TokenizerError};

#[derive(Default)]
pub struct Cache {
    pub enabled: bool,
    saved: Option<Box<dyn Any>>,
    key: Option<(String, bool, bool)>,
}

impl Cache {
    pub fn resident() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
    pub fn clear(&mut self) {
        self.saved = None;
    }
    pub fn configure(&mut self, ram: &str, raw: bool, thinking: bool) {
        let key = (ram.to_owned(), raw, thinking);
        if self.key.as_ref() != Some(&key) {
            self.clear();
        }
        self.key = Some(key);
    }
    pub fn take<M>(&mut self, required: usize) -> Option<Live<M>>
    where
        M: CausalDecoder + 'static,
        M::State: SessionState + 'static,
    {
        let live = self.saved.take()?.downcast::<Live<M>>().ok()?;
        if required <= live.capacity {
            Some(*live)
        } else {
            eprintln!(
                "kv cache: rebuilding for context growth ({required} > {})",
                live.capacity
            );
            None
        }
    }
}

pub struct Live<M: CausalDecoder>
where
    M::State: SessionState,
{
    model: M,
    state: Option<M::State>,
    tokens: Vec<u32>,
    checkpoint: Option<(usize, <M::State as SessionState>::Checkpoint)>,
    capacity: usize,
}

impl<M> Live<M>
where
    M: CausalDecoder,
    M::State: SessionState,
    M::Error: 'static,
{
    pub fn new(model: M, capacity: usize) -> Result<Self, Box<dyn Error>> {
        let _profile = span(ProfileStage::StateInit);
        let state = model.new_state()?;
        Ok(Self {
            model,
            state: Some(state),
            tokens: vec![],
            checkpoint: None,
            capacity,
        })
    }

    fn prepare(&mut self, prompt: &EncodedPrompt, retain: bool) -> Result<usize, Box<dyn Error>> {
        let boundary = if retain { prompt.checkpoint } else { 0 };
        let matches =
            |length: usize| length <= boundary && prompt.tokens.starts_with(&self.tokens[..length]);
        let reused = if matches(self.tokens.len()) {
            self.tokens.len()
        } else if self
            .checkpoint
            .as_ref()
            .is_some_and(|(length, _)| matches(*length))
        {
            let (length, checkpoint) = self.checkpoint.take().unwrap();
            self.state.as_mut().unwrap().restore(checkpoint)?;
            length
        } else {
            // Free sequence and expert buffers before allocating a fresh state.
            self.checkpoint = None;
            self.state = None;
            self.state = Some(self.model.new_state()?);
            0
        };
        self.checkpoint = None;
        self.tokens.truncate(reused);
        telemetry::emit(true, || RuntimeEvent::Kv {
            reused,
            prefill: prompt.tokens.len() - reused,
            capacity: self.capacity,
        });
        if retain && boundary > reused {
            Observed {
                model: &self.model,
                capacity: self.capacity,
            }
            .prefill(
                &prompt.tokens[reused..boundary],
                self.state.as_mut().unwrap(),
            )?;
        }
        if retain && boundary > 0 {
            self.checkpoint = Some((boundary, self.state.as_ref().unwrap().checkpoint()));
        }
        eprintln!(
            "kv cache: reused={reused} tokens, prefill={} tokens, capacity={}",
            prompt.tokens.len() - reused,
            self.capacity
        );
        Ok(if retain { boundary } else { reused })
    }
}

pub trait TextDecoder {
    fn push(&mut self, token: u32) -> Result<String, TokenizerError>;
    fn finish(&mut self) -> String;
}
impl TextDecoder for StreamingDecoder<'_> {
    fn push(&mut self, token: u32) -> Result<String, TokenizerError> {
        self.push(token)
    }
    fn finish(&mut self) -> String {
        self.finish()
    }
}
impl TextDecoder for KimiK3StreamingDecoder<'_> {
    fn push(&mut self, token: u32) -> Result<String, TokenizerError> {
        self.push(token)
    }
    fn finish(&mut self) -> String {
        self.finish()
    }
}

pub fn generate<M, D, W>(
    cache: &mut Cache,
    mut live: Live<M>,
    prompt: EncodedPrompt,
    config: &GenerationConfig,
    mut decoder: D,
    output: &mut W,
    progress: &mut progress::Progress,
) -> Result<(), Box<dyn Error>>
where
    M: CausalDecoder + 'static,
    M::State: SessionState + 'static,
    M::Error: 'static,
    D: TextDecoder,
    W: Write,
{
    if let Some(weights) = live.state.as_ref().unwrap().weight_residency() {
        telemetry::emit(true, || RuntimeEvent::Weights(weights));
    }
    observe_experts(live.state.as_ref().unwrap(), true);
    let suffix = live.prepare(&prompt, cache.enabled)?;
    let generated = match try_generate_with_state(
        &Observed {
            model: &live.model,
            capacity: live.capacity,
        },
        &prompt.tokens[suffix..],
        config,
        live.state.as_mut().unwrap(),
        |token| -> Result<(), StreamError> {
            progress.token();
            if !config.eos_token_ids.contains(&token) {
                output.write_all(decoder.push(token)?.as_bytes())?;
                output.flush()?;
            }
            Ok(())
        },
    ) {
        Ok(generated) => generated,
        Err(GenerationError::Callback(StreamError::Io(error)))
            if error.kind() == std::io::ErrorKind::BrokenPipe =>
        {
            return Ok(())
        }
        Err(error) => return Err(Box::new(error)),
    };
    output.write_all(decoder.finish().as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    let telemetry = live.state.as_ref().unwrap().expert_telemetry();
    telemetry::sequence(live.state.as_ref().unwrap().position(), live.capacity, true);
    observe_experts(live.state.as_ref().unwrap(), true);
    telemetry::emit(true, || RuntimeEvent::Stop {
        reason: match generated.stop_reason {
            StopReason::Eos(token) => format!("EOS {token}"),
            StopReason::MaxNewTokens => "Token limit".into(),
        },
    });
    eprintln!("done: new_tokens={}, stop={}, expert hits={}, misses={}, evictions={}, expert_payload_bytes_read={}",
        generated.generated_tokens.len(), match generated.stop_reason {
            StopReason::Eos(token) => format!("eos:{token}"), StopReason::MaxNewTokens => "max_new_tokens".to_owned(),
        }, telemetry.hits, telemetry.misses, telemetry.evictions, telemetry.bytes_read);
    if cache.enabled {
        live.tokens = prompt.tokens;
        // The final sampled token has not been forwarded; native rendering supplies it again
        // when needed. Never claim its KV exists (including EOS and token-limited replies).
        live.tokens
            .extend_from_slice(&generated.generated_tokens[..generated.generated_tokens.len() - 1]);
        debug_assert_eq!(live.state.as_ref().unwrap().position(), live.tokens.len());
        cache.saved = Some(Box::new(live));
    }
    Ok(())
}

/// Observe state between forward calls without changing sampling, callbacks or model numerics.
struct Observed<'a, M> {
    model: &'a M,
    capacity: usize,
}
impl<M: CausalDecoder> CausalDecoder for Observed<'_, M>
where
    M::State: SessionState,
{
    type State = M::State;
    type Error = M::Error;
    fn new_state(&self) -> Result<Self::State, Self::Error> {
        self.model.new_state()
    }
    fn forward_token(&self, token: u32, state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        let result = self.model.forward_token(token, state);
        if result.is_ok() {
            telemetry::sequence(state.position(), self.capacity, false);
        }
        observe_experts(state, false);
        result
    }
    fn prefill(&self, prompt: &[u32], state: &mut Self::State) -> Result<Vec<f32>, Self::Error> {
        let result = self.model.prefill(prompt, state);
        if result.is_ok() {
            telemetry::sequence(state.position(), self.capacity, true);
        }
        observe_experts(state, true);
        result
    }
}

/// Reserve growing context up front, shrinking to the exact request if RAM is tight.
/// The closure includes the additional recurrent/window checkpoint in its memory accounting.
pub fn plan<T>(
    enabled: bool,
    required: usize,
    limit: usize,
    mut build: impl FnMut(usize) -> Result<(T, bool), Box<dyn Error>>,
) -> Result<(usize, T), Box<dyn Error>> {
    let mut capacity = if enabled {
        required
            .checked_next_power_of_two()
            .unwrap_or(limit)
            .max(2048)
            .min(limit)
    } else {
        required
    };
    loop {
        let (plan, fits) = build(capacity)?;
        if fits || capacity == required {
            return Ok((capacity, plan));
        }
        capacity = (capacity / 2).max(required);
    }
}

pub fn resource_plan(
    enabled: bool,
    required: usize,
    limit: usize,
    ram: u64,
    mut build: impl FnMut(usize, u64) -> Result<(ResourcePlan, u64), Box<dyn Error>>,
) -> Result<(usize, ResourcePlan), Box<dyn Error>> {
    plan(enabled, required, limit, |capacity| {
        let (mut plan, checkpoint_bytes) = build(capacity, ram)?;
        if enabled && checkpoint_bytes != 0 {
            plan = build(capacity, ram.saturating_sub(checkpoint_bytes))?.0;
        }
        let fits = plan.feasible;
        Ok((plan, fits))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::convert::Infallible;
    use std::rc::Rc;
    use std::time::Instant;
    use urbilateria::runtime::ExpertTelemetry;

    #[derive(Default)]
    struct Trace {
        initialized: usize,
        forwarded: Vec<u32>,
    }
    struct Decoder(Rc<RefCell<Trace>>);
    struct State {
        tokens: Vec<u32>,
        telemetry: ExpertTelemetry,
    }
    impl SessionState for State {
        type Checkpoint = usize;
        fn position(&self) -> usize {
            self.tokens.len()
        }
        fn checkpoint(&self) -> usize {
            self.tokens.len()
        }
        fn restore(&mut self, length: usize) -> Result<(), Box<dyn Error>> {
            self.tokens.truncate(length);
            Ok(())
        }
        fn expert_telemetry(&self) -> &ExpertTelemetry {
            &self.telemetry
        }
    }
    impl CausalDecoder for Decoder {
        type State = State;
        type Error = Infallible;
        fn new_state(&self) -> Result<State, Infallible> {
            self.0.borrow_mut().initialized += 1;
            Ok(State {
                tokens: vec![],
                telemetry: ExpertTelemetry::default(),
            })
        }
        fn forward_token(&self, token: u32, state: &mut State) -> Result<Vec<f32>, Infallible> {
            self.0.borrow_mut().forwarded.push(token);
            state.tokens.push(token);
            let selected = state
                .tokens
                .iter()
                .fold(0usize, |v, &t| (v * 3 + t as usize) % 11);
            let mut logits = vec![0.; 11];
            logits[selected] = 1.;
            Ok(logits)
        }
    }
    struct Text;
    impl TextDecoder for Text {
        fn push(&mut self, token: u32) -> Result<String, TokenizerError> {
            Ok(format!("{token},"))
        }
        fn finish(&mut self) -> String {
            String::new()
        }
    }
    fn turn(
        cache: &mut Cache,
        live: Live<Decoder>,
        tokens: &[u32],
        checkpoint: usize,
        count: usize,
    ) -> String {
        let mut output = Vec::new();
        generate(
            cache,
            live,
            EncodedPrompt {
                tokens: tokens.to_vec(),
                checkpoint,
            },
            &GenerationConfig::greedy(count, vec![]),
            Text,
            &mut output,
            &mut progress::Progress::new(Instant::now(), progress::Mode::Summary),
        )
        .unwrap();
        String::from_utf8(output).unwrap()
    }
    fn cold(tokens: &[u32], count: usize) -> String {
        turn(
            &mut Cache::default(),
            Live::new(Decoder(Rc::default()), 64).unwrap(),
            tokens,
            0,
            count,
        )
    }

    #[test]
    fn append_reuses_forwarded_tokens_and_feeds_the_unforwarded_final_token_once() {
        let trace = Rc::new(RefCell::new(Trace::default()));
        let mut cache = Cache::resident();
        let first = turn(
            &mut cache,
            Live::new(Decoder(trace.clone()), 64).unwrap(),
            &[1, 2, 3],
            2,
            3,
        );
        let selected = first
            .trim()
            .trim_end_matches(',')
            .split(',')
            .map(|v| v.parse::<u32>().unwrap())
            .collect::<Vec<_>>();
        let live = cache.take::<Decoder>(64).unwrap();
        assert_eq!(live.tokens, [1, 2, 3, selected[0], selected[1]]);
        let mut prompt = live.tokens.clone();
        prompt.extend([selected[2], 8, 9]);
        trace.borrow_mut().forwarded.clear();
        let result = turn(&mut cache, live, &prompt, prompt.len() - 1, 2);
        assert_eq!(result, cold(&prompt, 2));
        assert_eq!(&trace.borrow().forwarded[..3], &[selected[2], 8, 9]);
        assert_eq!(trace.borrow().initialized, 1);
    }

    #[test]
    fn kv_snapshots_count_forwarded_positions_across_reuse_and_reset() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let received = events.clone();
        let _subscription = telemetry::subscribe(move |event| received.borrow_mut().push(event));
        let mut cache = Cache::resident();
        turn(
            &mut cache,
            Live::new(Decoder(Rc::default()), 64).unwrap(),
            &[1, 2, 3],
            2,
            3,
        );
        let check = |reused, prefill, cached| {
            let events = events.borrow();
            assert!(events.contains(&RuntimeEvent::Kv {
                reused,
                prefill,
                capacity: 64
            }));
            let positions: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    RuntimeEvent::KvState {
                        position,
                        capacity: 64,
                    } => Some(*position),
                    _ => None,
                })
                .collect();
            assert_eq!(positions.last(), Some(&cached));
            assert!(positions.windows(2).all(|p| p[0] <= p[1]));
            assert!(events.iter().all(RuntimeEvent::is_valid));
        };
        check(0, 3, 5); // Three selected tokens, but the last one has not been forwarded.
        let live = cache.take::<Decoder>(64).unwrap();
        let mut prompt = live.tokens.clone();
        prompt.extend([8, 9]);
        events.borrow_mut().clear();
        turn(&mut cache, live, &prompt, 6, 2);
        check(5, 2, 8);
        let live = cache.take::<Decoder>(64).unwrap();
        events.borrow_mut().clear();
        turn(&mut cache, live, &[7, 6], 1, 1);
        check(0, 2, 2);
    }

    #[test]
    fn rewritten_reasoning_rolls_back_and_changed_history_rebuilds_without_stale_kv() {
        let trace = Rc::new(RefCell::new(Trace::default()));
        let mut cache = Cache::resident();
        turn(
            &mut cache,
            Live::new(Decoder(trace.clone()), 64).unwrap(),
            &[1, 2, 3, 4],
            2,
            4,
        );
        trace.borrow_mut().forwarded.clear();
        let live = cache.take::<Decoder>(32).unwrap();
        let prompt = [1, 2, 9, 10];
        assert_eq!(turn(&mut cache, live, &prompt, 3, 2), cold(&prompt, 2));
        assert_eq!(&trace.borrow().forwarded[..2], &[9, 10]);
        assert_eq!(trace.borrow().initialized, 1);
        trace.borrow_mut().forwarded.clear();
        let live = cache.take::<Decoder>(32).unwrap();
        let prompt = [8, 7, 6];
        assert_eq!(turn(&mut cache, live, &prompt, 2, 1), cold(&prompt, 1));
        assert_eq!(&trace.borrow().forwarded[..3], &prompt);
        assert_eq!(trace.borrow().initialized, 2);
        assert!(cache.take::<Decoder>(65).is_none());
    }

    #[test]
    fn context_reservation_shrinks_to_the_request_under_memory_pressure() {
        let mut tried = vec![];
        let (capacity, _) = plan(true, 130, 8192, |n| {
            tried.push(n);
            Ok((n, n <= 130))
        })
        .unwrap();
        assert_eq!(capacity, 130);
        assert_eq!(tried, [2048, 1024, 512, 256, 130]);
        assert_eq!(plan(false, 130, 8192, |n| Ok((n, true))).unwrap().0, 130);
    }
}
