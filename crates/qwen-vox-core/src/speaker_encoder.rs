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
    prefix: String,
    mel_dim: usize,
    embed_dim: usize,
}

impl SpeakerEncoder {
    /// Load from WeightStore.
    ///
    /// Accepts either a component store with `asp.*` / `blocks.*` keys or a
    /// whole Qwen3-TTS model store with `speaker_encoder.*` keys.
    pub fn from_store(store: &WeightStore) -> VoxResult<Self> {
        let prefix = if Self::has_component(store, "speaker_encoder.") {
            "speaker_encoder."
        } else if Self::has_component(store, "") {
            ""
        } else {
            return Err(VoxError::WeightLoad(
                "Speaker encoder weights must contain 'speaker_encoder.asp'/'speaker_encoder.blocks' or bare 'asp'/'blocks' prefixes".to_string(),
            ));
        };

        let first_conv = store.require(&format!("{prefix}blocks.0.conv.weight"))?;
        let first_conv_dims = first_conv.dims();
        if first_conv_dims.len() != 3 {
            return Err(VoxError::ShapeMismatch {
                expected: vec![512, 128, 5],
                actual: first_conv_dims.to_vec(),
            });
        };
        let mel_dim = first_conv_dims[1];

        let fc = store.require(&format!("{prefix}fc.weight"))?;
        let fc_dims = fc.dims();
        if fc_dims.len() != 3 || fc_dims[2] != 1 {
            return Err(VoxError::ShapeMismatch {
                expected: vec![1024, 3072, 1],
                actual: fc_dims.to_vec(),
            });
        }

        Ok(Self {
            prefix: prefix.to_string(),
            mel_dim,
            embed_dim: fc_dims[0],
        })
    }

    /// Forward pass: mel [B, n_mels, T] → embedding [B, embed_dim]
    pub fn forward(&self, _mel: &Tensor) -> candle_core::Result<Tensor> {
        Err(candle_core::Error::Msg(
            "SpeakerEncoder::forward not yet implemented: ECAPA-TDNN / ASP layers still need Rust parity with official Qwen3-TTS".into(),
        ))
    }

    /// Prefix detected in the loaded weight store.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Number of mel bins expected by the speaker encoder.
    pub fn mel_dim(&self) -> usize {
        self.mel_dim
    }

    /// Get expected embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn has_component(store: &WeightStore, prefix: &str) -> bool {
        store.get(&format!("{prefix}asp.conv.weight")).is_some()
            && store
                .get(&format!("{prefix}blocks.0.conv.weight"))
                .is_some()
            && store.get(&format!("{prefix}fc.weight")).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_speaker_encoder_construction_without_weights() {
        // This test verifies the module compiles and from_store fails gracefully
        // when no relevant prefixes are present.
        let store = WeightStore::new(Device::Cpu);
        let result = SpeakerEncoder::from_store(&store);
        assert!(result.is_err());
    }

    #[test]
    fn test_speaker_encoder_loads_whole_model_prefix() {
        let store = minimal_store("speaker_encoder.");
        let encoder = SpeakerEncoder::from_store(&store).unwrap();
        assert_eq!(encoder.prefix(), "speaker_encoder.");
        assert_eq!(encoder.mel_dim(), 128);
        assert_eq!(encoder.embed_dim(), 1024);
    }

    #[test]
    fn test_speaker_encoder_loads_bare_component_prefix() {
        let store = minimal_store("");
        let encoder = SpeakerEncoder::from_store(&store).unwrap();
        assert_eq!(encoder.prefix(), "");
        assert_eq!(encoder.mel_dim(), 128);
        assert_eq!(encoder.embed_dim(), 1024);
    }

    #[test]
    fn test_speaker_encoder_forward_returns_error_not_panic() {
        let store = minimal_store("");
        let encoder = SpeakerEncoder::from_store(&store).unwrap();
        let mel = Tensor::zeros((1, 128, 10), DType::F32, &Device::Cpu).unwrap();
        assert!(encoder.forward(&mel).is_err());
    }

    fn minimal_store(prefix: &str) -> WeightStore {
        let device = Device::Cpu;
        let mut store = WeightStore::new(device.clone());
        store.insert_tensor(
            format!("{prefix}asp.conv.weight"),
            Tensor::zeros((1536, 128, 1), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            format!("{prefix}blocks.0.conv.weight"),
            Tensor::zeros((512, 128, 5), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            format!("{prefix}fc.weight"),
            Tensor::zeros((1024, 3072, 1), DType::F32, &device).unwrap(),
        );
        store
    }
}
