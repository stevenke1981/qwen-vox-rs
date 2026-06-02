//! Core TTS decoder trait.

use crate::config::DecoderConfig;
use crate::error::VoxResult;

/// Primary streaming decoder interface.
///
/// Implementations must be `Send + Sync` to support concurrent
/// multi-codebook processing and async I/O.
pub trait TtsDecoder: Send + Sync {
    /// Initialize decoder — load weights and vocoder.
    fn new(config: DecoderConfig) -> VoxResult<Self>
    where
        Self: Sized;

    /// Stream tokens in, return PCM audio chunk (f32 samples, mono).
    fn decode_chunk(&mut self, tokens: &[u16]) -> VoxResult<Vec<f32>>;

    /// Reset internal state (e.g., for session switch).
    fn reset_state(&mut self);

    /// Return the current decoder mode name.
    fn mode_name(&self) -> &'static str;
}
