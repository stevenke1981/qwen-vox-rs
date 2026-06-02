//! Multi-Token Prediction (MTP) head for 12 Hz mode.
//!
//! Generates acoustic codebook tokens from the causal convolution
//! output using a lightweight prediction head.

use crate::error::VoxResult;
use candle_core::Tensor;

/// MTP prediction head.
///
/// Takes the causal conv output and predicts the next set of
/// acoustic codebook tokens across all 16 layers.
pub struct MtpHead {
    /// Linear projection weights: `[embed_dim, num_layers * vocab_size]`.
    projection: Tensor,
    /// Number of codebook layers.
    num_layers: usize,
    /// Vocabulary size per layer.
    vocab_size: usize,
}

impl MtpHead {
    /// Create from preloaded projection weights.
    pub fn from_weights(
        projection: Tensor,
        num_layers: usize,
        vocab_size: usize,
    ) -> VoxResult<Self> {
        Ok(Self {
            projection,
            num_layers,
            vocab_size,
        })
    }

    /// Predict next tokens from causal conv output.
    ///
    /// # Arguments
    /// * `conv_output` - Tensor of shape `[embed_dim]` from causal conv
    ///
    /// # Returns
    /// Predicted token indices: `Vec<u16>` of length `num_layers`
    pub fn predict(&self, conv_output: &Tensor) -> VoxResult<Vec<u16>> {
        // Project to logits: [num_layers * vocab_size]
        let logits = conv_output
            .unsqueeze(0)?
            .matmul(&self.projection.t()?)?
            .squeeze(0)?;

        // Reshape to [num_layers, vocab_size] and argmax per layer
        let logits_2d = logits.reshape((self.num_layers, self.vocab_size))?;

        let mut tokens = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let layer_logits = logits_2d.narrow(0, layer, 1)?.squeeze(0)?;
            let token_idx = layer_logits.argmax(0)?.to_scalar::<u32>()?;
            tokens.push(token_idx as u16);
        }

        Ok(tokens)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
