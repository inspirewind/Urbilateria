//! Sequence checkpoints for conversation reuse. Weight and expert caches stay in the live state.

use super::ExpertTelemetry;
use std::error::Error;

/// A checkpoint belongs to the state that created it. Restoring consumes it; callers must not
/// restore past a reset or use it with another model instance. Append-only KV arrays retain
/// lengths; recurrent and sliding-window state must retain values as well.
pub trait SessionState {
    type Checkpoint;

    fn position(&self) -> usize;
    fn checkpoint(&self) -> Self::Checkpoint;
    fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<(), Box<dyn Error>>;
    fn expert_telemetry(&self) -> &ExpertTelemetry;
}
