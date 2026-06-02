//! Vocoder — waveform synthesis from acoustic features.
//!
//! Converts mel-spectrogram or codec features into raw PCM audio.
//! Architecture TBD pending analysis of the original Qwen3-TTS vocoder.

use crate::error::VoxResult;
use candle_core::Tensor;

/// Vocoder trait for pluggable waveform synthesis backends.
pub trait Vocoder: Send + Sync {
    /// Synthesize PCM audio from acoustic features.
    ///
    /// # Arguments
    /// * `features` - Acoustic feature tensor (shape depends on vocoder type)
    ///
    /// # Returns
    /// Mono PCM samples as `Vec<f32>`, normalized to [-1.0, 1.0]
    fn synthesize(&self, features: &Tensor) -> VoxResult<Vec<f32>>;

    /// Return the expected sample rate of the output audio.
    fn sample_rate(&self) -> u32;
}

/// Placeholder vocoder that generates silence.
///
/// Used during Phase 0-2 before the real vocoder is implemented.
pub struct SilenceVocoder {
    sample_rate: u32,
}

impl SilenceVocoder {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }
}

impl Vocoder for SilenceVocoder {
    fn synthesize(&self, features: &Tensor) -> VoxResult<Vec<f32>> {
        // Generate silence proportional to feature length
        let num_frames = features.dim(0).unwrap_or(1);
        let samples_per_frame = self.sample_rate as usize / 25; // ~960 samples/frame at 24kHz
        Ok(vec![0.0; num_frames * samples_per_frame])
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
