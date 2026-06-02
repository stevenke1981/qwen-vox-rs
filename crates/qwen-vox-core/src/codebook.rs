//! Multi-codebook embedding lookup with rayon parallelism.
//!
//! Memory layout: flat 1-D contiguous `Vec<u16>`,
//! indexed as `layer * num_frames + frame`.

use crate::error::{VoxError, VoxResult};
use candle_core::Tensor;
use rayon::prelude::*;

/// Codebook vocabulary size (default 2048).
pub const DEFAULT_VOCAB_SIZE: usize = 2048;

/// Number of quantizer layers in 12 Hz mode.
pub const NUM_CODEBOOK_LAYERS: usize = 16;

/// Multi-layer codebook embedding table.
///
/// Stores preloaded weight matrices for all 16 quantizer layers.
/// Lookup is branch-free: direct pointer offset into contiguous memory.
pub struct CodebookEmbedding {
    /// Embedding weight tensor: shape `[num_layers, vocab_size, embed_dim]`.
    weights: Tensor,
    /// Vocabulary size per layer.
    vocab_size: usize,
    /// Embedding dimension.
    embed_dim: usize,
    /// Number of layers.
    num_layers: usize,
}

impl CodebookEmbedding {
    /// Create from a preloaded weight tensor.
    ///
    /// # Arguments
    /// * `weights` - Tensor of shape `[num_layers, vocab_size, embed_dim]`
    pub fn from_weights(weights: Tensor) -> VoxResult<Self> {
        let dims = weights
            .dims3()
            .map_err(|e| VoxError::WeightLoad(format!("codebook weights must be 3-D: {e}")))?;
        Ok(Self {
            weights,
            num_layers: dims.0,
            vocab_size: dims.1,
            embed_dim: dims.2,
        })
    }

    /// Lookup embeddings for a batch of token indices across all layers.
    ///
    /// Uses `rayon::par_iter` to parallelize across the 16 codebook layers.
    ///
    /// # Arguments
    /// * `tokens` - Flat token buffer: `tokens[layer * num_frames + frame]`
    /// * `num_frames` - Number of frames in this chunk
    ///
    /// # Returns
    /// Tensor of shape `[num_layers, num_frames, embed_dim]`
    pub fn lookup(&self, tokens: &[u16], num_frames: usize) -> VoxResult<Tensor> {
        let expected_len = self.num_layers * num_frames;
        if tokens.len() != expected_len {
            return Err(VoxError::ShapeMismatch {
                expected: vec![expected_len],
                actual: vec![tokens.len()],
            });
        }

        // Validate all indices are in range (branch-free in hot path later)
        for (i, &tok) in tokens.iter().enumerate() {
            if tok as usize >= self.vocab_size {
                let layer = i / num_frames;
                return Err(VoxError::CodebookOutOfRange {
                    layer,
                    index: tok,
                    vocab_size: self.vocab_size,
                });
            }
        }

        // Parallel lookup across layers
        let layer_results: Vec<VoxResult<Tensor>> = (0..self.num_layers)
            .into_par_iter()
            .map(|layer_idx| {
                let start = layer_idx * num_frames;
                let end = start + num_frames;
                let layer_tokens = &tokens[start..end];

                // Extract this layer's weight slice: [vocab_size, embed_dim]
                let layer_weight = self.weights.narrow(0, layer_idx, 1)?.squeeze(0)?;

                // Gather embeddings for each token index
                let indices: Vec<u32> = layer_tokens.iter().map(|&t| t as u32).collect();
                let index_tensor = Tensor::new(&indices[..], layer_weight.device())?;
                let embeddings = layer_weight.index_select(&index_tensor, 0)?;

                Ok(embeddings)
            })
            .collect();

        // Stack results: [num_layers, num_frames, embed_dim]
        let results: Vec<Tensor> = layer_results.into_iter().collect::<VoxResult<Vec<_>>>()?;
        let stacked = Tensor::stack(&results, 0)?;
        Ok(stacked)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}
