//! Configuration types for the TTS decoder.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Decoder operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoderMode {
    /// 12 Hz real-time interactive mode.
    /// Causal ConvNet + MTP, 16-layer codebook parallel.
    Realtime12Hz,

    /// 25 Hz high-quality synthesis mode.
    /// Block-wise Flow Matching DiT.
    HighQuality25Hz,
}

impl DecoderMode {
    /// Target frame rate in Hz.
    pub fn frame_rate_hz(&self) -> u32 {
        match self {
            Self::Realtime12Hz => 12,
            Self::HighQuality25Hz => 25,
        }
    }

    /// Number of codebook layers.
    pub fn num_codebook_layers(&self) -> usize {
        match self {
            Self::Realtime12Hz => 16,
            Self::HighQuality25Hz => 1,
        }
    }
}

/// Device selection for tensor computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    /// CUDA GPU (device index 0). Requires `cuda` feature.
    Cuda,
    /// Apple Metal GPU. Requires `metal` feature.
    Metal,
    /// CPU fallback (always available).
    Cpu,
}

impl DeviceKind {
    /// Convert to a `candle_core::Device`, returning an error if the device is
    /// unavailable (e.g. `Cuda` without the `cuda` feature or without a GPU).
    pub fn to_candle_device(&self) -> crate::error::VoxResult<candle_core::Device> {
        use crate::error::VoxError;
        match self {
            Self::Cpu => Ok(candle_core::Device::Cpu),
            #[cfg(feature = "cuda")]
            Self::Cuda => candle_core::Device::cuda_if_available(0)
                .map_err(|e| VoxError::Device(format!("CUDA unavailable: {e}"))),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda => Err(VoxError::Device(
                "CUDA requested but feature 'cuda' is not enabled. \
                 Rebuild with --features cuda"
                    .into(),
            )),
            #[cfg(feature = "metal")]
            Self::Metal => candle_core::Device::metal_if_available(0)
                .map_err(|e| VoxError::Device(format!("Metal unavailable: {e}"))),
            #[cfg(not(feature = "metal"))]
            Self::Metal => Err(VoxError::Device(
                "Metal requested but feature 'metal' is not enabled. \
                 Rebuild with --features metal"
                    .into(),
            )),
        }
    }
}

impl std::str::FromStr for DeviceKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            "metal" | "mps" => Ok(Self::Metal),
            _ => Err(format!(
                "unknown device '{s}': expected 'cpu', 'cuda'/'gpu', or 'metal'/'mps'"
            )),
        }
    }
}

/// Full decoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Operating mode.
    pub mode: DecoderMode,

    /// Compute device.
    pub device: DeviceKind,

    /// Path to SafeTensors weight file.
    pub weights_path: PathBuf,

    /// Path to tokenizer.json.
    pub tokenizer_path: PathBuf,

    /// Codebook vocabulary size (default: 2048).
    pub vocab_size: usize,

    /// Audio sample rate in Hz (default: 24000).
    pub sample_rate: u32,

    /// Ring buffer capacity for causal conv (frames).
    pub ring_buffer_capacity: usize,

    /// First-packet latency target in milliseconds.
    pub latency_target_ms: u64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            mode: DecoderMode::Realtime12Hz,
            device: DeviceKind::Cpu,
            weights_path: PathBuf::from("weights/model.safetensors"),
            tokenizer_path: PathBuf::from("weights/tokenizer.json"),
            vocab_size: 2048,
            sample_rate: 24_000,
            ring_buffer_capacity: 64,
            latency_target_ms: 97,
        }
    }
}
