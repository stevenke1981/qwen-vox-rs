//! Speaker Encoder (ECAPA-TDNN + Attentive Statistics Pooling)
//!
//! Extracts speaker embeddings from reference audio (mel spectrogram).
//! Architecture: ECAPA-TDNN with SE-Res2Blocks + ASP pooling.
//!
//! Weight prefixes observed:
//! - `asp.*` — Attentive Statistics Pooling parameters
//! - `blocks.*` — TDNN / Res2Net blocks

use crate::error::{VoxError, VoxResult};
use crate::weights::WeightStore;
use candle_core::Tensor;

/// Speaker Encoder producing fixed-dimensional speaker embeddings.
/// Input: mel spectrogram [B, n_mels, T]
/// Output: speaker embedding [B, embed_dim]
pub struct SpeakerEncoder {
    // TODO: Define internal layers based on actual weight inspection
    // Placeholder fields — replace with real layers after weight analysis
    _placeholder: (),
}

impl SpeakerEncoder {
    /// Load from WeightStore (expects "asp.*" and "blocks.*" prefixes).
    pub fn from_store(store: &WeightStore) -> VoxResult<Self> {
        // Verify expected prefixes exist
        let asp_keys = store.get_prefix("asp");
        let block_keys = store.get_prefix("blocks");

        if asp_keys.is_empty() && block_keys.is_empty() {
            return Err(VoxError::WeightLoad(
                "Speaker encoder weights must contain 'asp' or 'blocks' prefixes".to_string(),
            ));
        }

        // TODO: Implement actual layer construction after inspecting weight shapes
        // For now, return a stub that will fail on forward until implemented.
        Ok(Self { _placeholder: () })
    }

    /// Forward pass: mel [B, n_mels, T] → embedding [B, embed_dim]
    pub fn forward(&self, _mel: &Tensor) -> candle_core::Result<Tensor> {
        // TODO: Implement ECAPA-TDNN forward
        // 1. Frame-level TDNN / Res2Net blocks
        // 2. Attentive Statistics Pooling (ASP)
        // 3. Final projection to embedding
        unimplemented!("SpeakerEncoder::forward not yet implemented — requires weight inspection")
    }

    /// Get expected embedding dimension (e.g., 256 or 512).
    pub fn embed_dim(&self) -> usize {
        // TODO: Return actual dimension after loading
        256
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_speaker_encoder_construction_without_weights() {
        // This test verifies the module compiles and from_store fails gracefully
        // when no relevant prefixes are present.
        let store = WeightStore::new(Device::Cpu);
        let result = SpeakerEncoder::from_store(&store);
        assert!(result.is_err());
    }
}
