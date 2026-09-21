//! Lightweight generation metrics, independent of optional profiling and terminal rendering.

use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::{Duration, Instant};

pub const JSON_PREFIX: &str = "URB_PROGRESS ";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    #[default]
    Running,
    Complete,
    Failed,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub status: Status,
    pub prompt_tokens: Option<usize>,
    /// Selected token IDs, including EOS, matching GenerationOutput::generated_tokens.
    pub generated_tokens: usize,
    pub total_tokens: Option<usize>,
    pub elapsed_seconds: f64,
    /// Request start through selection of the first token, including loading and prefill.
    pub ttft_seconds: Option<f64>,
    /// (generated_tokens - 1) / (last_token_time - first_token_time).
    pub decode_tokens_per_second: Option<f64>,
}

impl Snapshot {
    #[cfg(feature = "ui")]
    pub fn is_valid(&self) -> bool {
        let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
        nonnegative(self.elapsed_seconds)
            && self
                .ttft_seconds
                .is_none_or(|ttft| nonnegative(ttft) && ttft <= self.elapsed_seconds)
            && self.decode_tokens_per_second.is_none_or(nonnegative)
            && self.total_tokens
                == self
                    .prompt_tokens
                    .and_then(|prompt| prompt.checked_add(self.generated_tokens))
            && (self.generated_tokens != 0 || self.ttft_seconds.is_none())
            && (self.generated_tokens >= 2 || self.decode_tokens_per_second.is_none())
    }

    pub fn summary(&self) -> String {
        format!(
            "input={} · output={} · total={} tok · {} tok/s · TTFT {} · {:.2}s",
            count(self.prompt_tokens),
            self.generated_tokens,
            count(self.total_tokens),
            self.decode_tokens_per_second
                .map(|rate| format!("{rate:.2}"))
                .unwrap_or_else(|| "—".into()),
            self.ttft_seconds
                .map(|seconds| format!("{seconds:.2}s"))
                .unwrap_or_else(|| "—".into()),
            self.elapsed_seconds,
        )
    }
}

fn count(value: Option<usize>) -> String {
    value
        .map(|count| count.to_string())
        .unwrap_or_else(|| "—".into())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Summary,
    Live,
    Json,
}

struct Timing {
    started: Instant,
    prompt_tokens: Option<usize>,
    generated_tokens: usize,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
}

impl Timing {
    fn new(started: Instant) -> Self {
        Self {
            started,
            prompt_tokens: None,
            generated_tokens: 0,
            first_token: None,
            last_token: None,
        }
    }

    fn token(&mut self, now: Instant) {
        self.generated_tokens += 1;
        self.first_token.get_or_insert(now);
        self.last_token = Some(now);
    }

    fn snapshot(&self, now: Instant, status: Status) -> Snapshot {
        let decode_tokens_per_second =
            self.first_token
                .zip(self.last_token)
                .and_then(|(first, last)| {
                    let seconds = last.duration_since(first).as_secs_f64();
                    (self.generated_tokens > 1 && seconds > 0.0)
                        .then(|| (self.generated_tokens - 1) as f64 / seconds)
                });
        Snapshot {
            status,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            total_tokens: self
                .prompt_tokens
                .and_then(|prompt| prompt.checked_add(self.generated_tokens)),
            elapsed_seconds: now.duration_since(self.started).as_secs_f64(),
            ttft_seconds: self
                .first_token
                .map(|first| first.duration_since(self.started).as_secs_f64()),
            decode_tokens_per_second,
        }
    }
}

pub struct Progress {
    timing: Timing,
    mode: Mode,
    last_emit: Instant,
}

impl Progress {
    pub fn new(started: Instant, mode: Mode) -> Self {
        let mut progress = Self {
            timing: Timing::new(started),
            mode,
            last_emit: started,
        };
        progress.emit(Status::Running, true);
        progress
    }

    pub fn prompt(&mut self, tokens: usize) {
        self.timing.prompt_tokens = Some(tokens);
        self.emit(Status::Running, true);
    }

    pub fn token(&mut self) {
        self.timing.token(Instant::now());
        self.emit(Status::Running, self.timing.generated_tokens == 1);
    }

    pub fn finish(&mut self, success: bool) {
        self.emit(
            if success {
                Status::Complete
            } else {
                Status::Failed
            },
            true,
        );
    }

    fn emit(&mut self, status: Status, force: bool) {
        if self.mode == Mode::Summary
            && (status == Status::Running || self.timing.prompt_tokens.is_none())
        {
            return;
        }
        let now = Instant::now();
        let interval = if self.mode == Mode::Json {
            Duration::from_millis(100)
        } else {
            Duration::from_secs(1)
        };
        if !force && now.duration_since(self.last_emit) < interval {
            return;
        }
        self.last_emit = now;
        // Keep stdout byte-for-byte suitable for piping. No cursor control: stdout and stderr
        // may share a terminal, and repainting stderr would erase the generated response.
        let _ = write_snapshot(
            &mut io::stderr().lock(),
            self.mode,
            &self.timing.snapshot(now, status),
        );
    }
}

fn write_snapshot(output: &mut impl Write, mode: Mode, snapshot: &Snapshot) -> io::Result<()> {
    if mode == Mode::Json {
        output.write_all(JSON_PREFIX.as_bytes())?;
        serde_json::to_writer(&mut *output, snapshot)?;
        output.write_all(b"\n")?;
    } else {
        writeln!(output, "metrics: {}", snapshot.summary())?;
    }
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttft_includes_loading_but_decode_rate_excludes_the_first_token() {
        let start = Instant::now();
        let mut timing = Timing::new(start);
        timing.prompt_tokens = Some(6);
        let loading = timing.snapshot(start + Duration::from_secs(2), Status::Running);
        assert_eq!(loading.total_tokens, Some(6));
        assert_eq!(loading.ttft_seconds, None);
        timing.token(start + Duration::from_secs(10));
        let first = timing.snapshot(start + Duration::from_secs(10), Status::Running);
        assert_eq!(first.ttft_seconds, Some(10.0));
        assert_eq!(first.decode_tokens_per_second, None);
        timing.token(start + Duration::from_secs(12));
        timing.token(start + Duration::from_secs(14)); // Could be EOS; it is still a selected ID.
        let final_stats = timing.snapshot(start + Duration::from_secs(20), Status::Complete);
        assert_eq!(final_stats.generated_tokens, 3);
        assert_eq!(final_stats.total_tokens, Some(9));
        assert_eq!(final_stats.decode_tokens_per_second, Some(0.5));
        assert_eq!(final_stats.elapsed_seconds, 20.0);
        assert_eq!(final_stats.ttft_seconds, Some(10.0));
    }

    #[test]
    fn empty_single_token_and_zero_interval_do_not_produce_infinite_rates() {
        let start = Instant::now();
        let mut timing = Timing::new(start);
        assert_eq!(
            timing
                .snapshot(start, Status::Failed)
                .decode_tokens_per_second,
            None
        );
        timing.token(start);
        timing.token(start);
        let snapshot = timing.snapshot(start, Status::Complete);
        assert_eq!(snapshot.decode_tokens_per_second, None);
        assert_eq!(snapshot.ttft_seconds, Some(0.0));
        let mut json = Vec::new();
        write_snapshot(&mut json, Mode::Json, &snapshot).unwrap();
        let line = std::str::from_utf8(&json).unwrap();
        assert_eq!(
            serde_json::from_str::<Snapshot>(line.strip_prefix(JSON_PREFIX).unwrap()).unwrap(),
            snapshot
        );
        let mut human = Vec::new();
        write_snapshot(&mut human, Mode::Summary, &snapshot).unwrap();
        let human = String::from_utf8(human).unwrap();
        assert!(human.contains("output=2"));
        assert!(human.contains("TTFT 0.00s"));
        assert!(!human.contains('\r'));
    }
}
