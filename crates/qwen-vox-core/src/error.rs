//! Error types for qwen-vox-core.

use thiserror::Error;

/// Unified error type for all decoder operations.
#[derive(Debug, Error)]
pub enum VoxError {
    #[error("weight loading failed: {0}")]
    WeightLoad(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("codebook index out of range: layer={layer}, index={index}, vocab_size={vocab_size}")]
    CodebookOutOfRange {
        layer: usize,
        index: u16,
        vocab_size: usize,
    },

    #[error("ring buffer overflow: capacity={capacity}")]
    RingBufferOverflow { capacity: usize },

    #[error("inference error: {0}")]
    Inference(String),

    #[error("decoder not initialized")]
    NotInitialized,

    #[error("invalid decoder mode: {0}")]
    InvalidMode(String),

    #[error(
        "numerical alignment check failed: cosine_sim={cosine_sim:.6}, threshold={threshold:.6}"
    )]
    AlignmentFailed { cosine_sim: f64, threshold: f64 },

    #[error("vocoder error: {0}")]
    Vocoder(String),

    #[error("streaming error: {0}")]
    Stream(String),

    #[error("device error: {0}")]
    Device(String),

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

/// Convenience alias.
pub type VoxResult<T> = Result<T, VoxError>;
