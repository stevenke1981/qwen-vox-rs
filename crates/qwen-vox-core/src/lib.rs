//! # qwen-vox-core
//!
//! Qwen3-TTS codec decoder core library — pure Rust, Candle-based.
//!
//! Supports dual-mode decoding:
//! - **12 Hz** — real-time interactive (Causal ConvNet + MTP)
//! - **25 Hz** — high-quality synthesis (Block-wise Flow Matching DiT)

pub mod alignment;
pub mod causal_conv;
pub mod codebook;
pub mod config;
pub mod conv_decoder;
pub mod custom_ops;
pub mod decoder;
pub mod device;
pub mod error;
pub mod flow_matching;
pub mod mtp;
pub mod pipeline;

pub mod quantizer;
pub mod sampling;
pub mod speaker_encoder;
pub mod speech_synth;
pub mod stream;
pub mod talker;
pub mod tokenizer;
pub mod transformer;
pub mod vocoder;
pub mod weights;

// Re-export primary public API
pub use config::{DecoderConfig, DecoderMode};
pub use decoder::TtsDecoder;
pub use error::VoxError;
pub use error::VoxResult;
pub use quantizer::{
    load_decoder_codebooks, CodePredictor, EuclideanCodebook, ResidualVectorQuantizer,
    SplitResidualVectorQuantizer,
};
pub use speech_synth::{synthesize_formant_speech, FormantSynthConfig};
pub use weights::{ComponentWeights, WeightStore};
