//! Weight loader for Qwen3-TTS SafeTensors files.
//!
//! Loads converted weights from SafeTensors format into Candle tensors.

use crate::error::{VoxError, VoxResult};
use candle_core::{safetensors::Load, DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Loaded weight collection.
pub struct WeightStore {
    /// All loaded tensors indexed by name.
    tensors: HashMap<String, Tensor>,
    /// Device used for loading.
    device: Device,
}

impl WeightStore {
    /// Create an empty WeightStore (useful for testing or incremental loading).
    pub fn new(device: Device) -> Self {
        Self {
            tensors: HashMap::new(),
            device,
        }
    }

    /// Insert a tensor into the store (useful for tests).
    pub fn insert_tensor(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.tensors.insert(name.into(), tensor);
    }

    /// Load weights from a SafeTensors file.
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> VoxResult<Self> {
        let path = path.as_ref();
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(|e| VoxError::WeightLoad(format!("failed to mmap {}: {e}", path.display())))?;
        let mut tensors = HashMap::new();

        let force_f16 = std::env::var_os("QWEN_VOX_FORCE_F16_WEIGHTS").is_some();
        for (name, view) in st.tensors() {
            let mut tensor = view.load(device).map_err(|e| {
                VoxError::WeightLoad(format!("failed to load tensor '{name}': {e}"))
            })?;
            if force_f16 && tensor.dtype() == DType::BF16 {
                tensor = tensor.to_dtype(DType::F16).map_err(|e| {
                    VoxError::WeightLoad(format!("failed to convert tensor '{name}' to f16: {e}"))
                })?;
            }

            tensors.insert(name.to_string(), tensor);
        }

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get a tensor by name, returning error if not found.
    pub fn require(&self, name: &str) -> VoxResult<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| VoxError::WeightLoad(format!("required tensor not found: {name}")))
    }

    /// Get all tensors with a given prefix.
    pub fn get_prefix(&self, prefix: &str) -> HashMap<String, &Tensor> {
        self.tensors
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// List all tensor names.
    pub fn names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Get the number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Weight loader for specific model components.
pub struct ComponentWeights {
    store: WeightStore,
    prefix: String,
}

impl ComponentWeights {
    /// Create from a weight store with a specific prefix.
    pub fn new(store: WeightStore, prefix: impl Into<String>) -> Self {
        Self {
            store,
            prefix: prefix.into(),
        }
    }

    /// Get a tensor by relative name (prefix is prepended).
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.store.get(&full_name)
    }

    /// Get a tensor by relative name, returning error if not found.
    pub fn require(&self, name: &str) -> VoxResult<&Tensor> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.store.require(&full_name)
    }

    /// Get all tensors under this component.
    pub fn all(&self) -> HashMap<String, &Tensor> {
        self.store.get_prefix(&self.prefix)
    }

    /// Get the underlying store.
    pub fn store(&self) -> &WeightStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_store_empty() {
        let store = WeightStore {
            tensors: HashMap::new(),
            device: Device::Cpu,
        };
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_component_weights_prefix() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "talker.layers.0.attn.q.weight".to_string(),
            Tensor::zeros((10, 10), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "talker.layers.0.attn.k.weight".to_string(),
            Tensor::zeros((10, 10), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "speaker.blocks.0.conv.weight".to_string(),
            Tensor::zeros((10, 10), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );

        let store = WeightStore {
            tensors,
            device: Device::Cpu,
        };

        let talker = ComponentWeights::new(store, "talker");
        assert!(talker.get("layers.0.attn.q.weight").is_some());
        assert!(talker.get("layers.0.attn.k.weight").is_some());
        assert!(talker.get("layers.1.attn.q.weight").is_none());
    }
}
