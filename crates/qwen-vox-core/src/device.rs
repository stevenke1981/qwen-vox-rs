//! Device management for qwen-vox-core.
//!
//! Provides `DeviceManager` — a convenience layer for selecting
//! and querying the compute device (CPU, CUDA GPU, Metal GPU).
//!
//! # Usage
//!
//! ```ignore
//! use qwen_vox_core::device::DeviceManager;
//!
//! // Auto-detect: try CUDA (if feature enabled), fall back to CPU
//! let mgr = DeviceManager::new("cuda")?;
//! let device = mgr.device();
//! println!("Running on: {device:?}");
//! ```

use crate::config::DeviceKind;
use crate::error::{VoxError, VoxResult};

/// Manages compute device selection and lifecycle.
///
/// Wraps a `candle_core::Device` with helpers for:
/// - Parsing device strings (`"cpu"`, `"cuda"`, `"metal"`)
/// - Attempting CUDA/Metal and falling back to CPU on failure
/// - Reporting the active device
#[derive(Debug, Clone)]
pub struct DeviceManager {
    active: candle_core::Device,
    kind: DeviceKind,
}

impl DeviceManager {
    /// Create a `DeviceManager` for the given device kind.
    ///
    /// For `Cpu`: always succeeds.
    /// For `Cuda` / `Metal`: tries the GPU, falling back to CPU if unavailable
    /// (warns on fallback).
    pub fn new(kind: DeviceKind) -> Self {
        let (active, actual_kind) = match kind.to_candle_device() {
            Ok(d) => (d, kind),
            Err(e) => {
                tracing::warn!("{e} — falling back to CPU");
                (candle_core::Device::Cpu, DeviceKind::Cpu)
            }
        };
        tracing::info!("Device: {active:?}");
        Self {
            active,
            kind: actual_kind,
        }
    }

    /// Parse a device string and create a `DeviceManager`.
    ///
    /// Accepts: `"cpu"`, `"cuda"`, `"gpu"`, `"metal"`, `"mps"`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> VoxResult<Self> {
        let kind: DeviceKind = s.parse().map_err(|e: String| VoxError::Device(e))?;
        Ok(Self::new(kind))
    }

    /// Returns a reference to the active `candle_core::Device`.
    pub fn device(&self) -> &candle_core::Device {
        &self.active
    }

    /// Returns the `DeviceKind` of the *requested* device (before fallback).
    pub fn kind(&self) -> DeviceKind {
        self.kind
    }

    /// Returns the `DeviceKind` of the *actual* device (after any fallback).
    pub fn active_kind(&self) -> DeviceKind {
        if matches!(&self.active, candle_core::Device::Cpu) {
            DeviceKind::Cpu
        } else if matches!(&self.active, candle_core::Device::Cuda(_)) {
            DeviceKind::Cuda
        } else if matches!(&self.active, candle_core::Device::Metal(_)) {
            DeviceKind::Metal
        } else {
            DeviceKind::Cpu
        }
    }

    /// Returns true if the active device is a CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(&self.active, candle_core::Device::Cpu)
    }

    /// Returns true if the active device is a CUDA GPU.
    pub fn is_cuda(&self) -> bool {
        matches!(&self.active, candle_core::Device::Cuda(_))
    }

    /// Returns true if the active device is a Metal GPU.
    pub fn is_metal(&self) -> bool {
        matches!(&self.active, candle_core::Device::Metal(_))
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new(DeviceKind::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_cpu() {
        let mgr = DeviceManager::new(DeviceKind::Cpu);
        assert!(mgr.is_cpu());
        assert!(!mgr.is_cuda());
        assert!(!mgr.is_metal());
    }

    #[test]
    fn test_device_manager_from_str() {
        let mgr = DeviceManager::from_str("cpu").unwrap();
        assert!(mgr.is_cpu());

        // Unknown string → error
        assert!(DeviceManager::from_str("quantum").is_err());
    }

    #[test]
    fn test_device_manager_default() {
        let mgr = DeviceManager::default();
        assert!(mgr.is_cpu());
    }
}
