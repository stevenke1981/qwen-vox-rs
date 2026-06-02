//! Numerical alignment test framework.
//!
//! Compares Candle (Rust) intermediate tensors against PyTorch reference
//! tensors exported by `tools/export_activations.py`.
//!
//! Each test loads a pair of SafeTensors files (reference + actual),
//! computes cosine similarity, and asserts ≥ 0.999.

use crate::error::{VoxError, VoxResult};
use candle_core::{Device, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Cosine similarity between two flat f32 slices.
///
/// Returns a value in [-1.0, 1.0]. Higher = more similar.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return if dot.abs() < 1e-12 { 1.0 } else { 0.0 };
    }

    dot / (norm_a * norm_b)
}

/// Maximum absolute difference between two flat f32 slices.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0_f64, f64::max)
}

/// Mean absolute difference between two flat f32 slices.
pub fn mean_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum();
    sum / a.len() as f64
}

/// Load all tensors from a SafeTensors file into a HashMap.
pub fn load_safetensors(path: impl AsRef<Path>) -> VoxResult<HashMap<String, Tensor>> {
    let path = path.as_ref();
    let data = std::fs::read(path)
        .map_err(|e| VoxError::WeightLoad(format!("failed to read {}: {e}", path.display())))?;

    let st = SafeTensors::deserialize(&data)
        .map_err(|e| VoxError::WeightLoad(format!("failed to parse safetensors: {e}")))?;

    let device = Device::Cpu;
    let mut map = HashMap::new();

    for (name, view) in st.tensors() {
        let shape = view.shape().to_vec();
        let dtype = view.dtype();
        let data = view.data();

        // Convert to f32 Tensor
        let tensor = match dtype {
            safetensors::Dtype::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Tensor::from_vec(floats, &shape[..], &device)?
            }
            safetensors::Dtype::F16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Tensor::from_vec(floats, &shape[..], &device)?
            }
            safetensors::Dtype::I64 => {
                let ints: Vec<i64> = data
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                    .collect();
                Tensor::from_vec(ints, &shape[..], &device)?
            }
            _ => {
                return Err(VoxError::WeightLoad(format!(
                    "unsupported dtype {dtype:?} for tensor '{name}'"
                )));
            }
        };

        map.insert(name.to_string(), tensor);
    }

    Ok(map)
}

/// Result of a single alignment comparison.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub tensor_name: String,
    pub cosine_sim: f64,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub shape: Vec<usize>,
    pub passed: bool,
}

/// Run alignment check between two sets of tensors.
///
/// For each tensor name present in both `reference` and `actual`,
/// computes cosine similarity and checks against `threshold`.
pub fn compare_tensors(
    reference: &HashMap<String, Tensor>,
    actual: &HashMap<String, Tensor>,
    threshold: f64,
) -> Vec<AlignmentResult> {
    let mut results = Vec::new();

    for (name, ref_tensor) in reference {
        if let Some(act_tensor) = actual.get(name) {
            // Check shape match
            if ref_tensor.shape() != act_tensor.shape() {
                results.push(AlignmentResult {
                    tensor_name: name.clone(),
                    cosine_sim: 0.0,
                    max_abs: f64::MAX,
                    mean_abs: f64::MAX,
                    shape: ref_tensor.dims().to_vec(),
                    passed: false,
                });
                continue;
            }

            let ref_vec: Vec<f32> = ref_tensor.flatten_all().unwrap().to_vec1().unwrap();
            let act_vec: Vec<f32> = act_tensor.flatten_all().unwrap().to_vec1().unwrap();

            let cos = cosine_similarity(&ref_vec, &act_vec);
            let max_d = max_abs_diff(&ref_vec, &act_vec);
            let mean_d = mean_abs_diff(&ref_vec, &act_vec);

            results.push(AlignmentResult {
                tensor_name: name.clone(),
                cosine_sim: cos,
                max_abs: max_d,
                mean_abs: mean_d,
                shape: ref_tensor.dims().to_vec(),
                passed: cos >= threshold,
            });
        }
    }

    results.sort_by(|a, b| a.tensor_name.cmp(&b.tensor_name));
    results
}

/// Print alignment results in a formatted table.
pub fn print_results(results: &[AlignmentResult]) {
    println!();
    println!(
        "{:<50} {:>10} {:>10} {:>10} {:>6}",
        "Tensor", "Cosine", "MaxAbs", "MeanAbs", "Pass"
    );
    println!("{}", "-".repeat(90));

    let mut pass_count = 0;
    let mut fail_count = 0;

    for r in results {
        let status = if r.passed { "✅" } else { "❌" };
        if r.passed {
            pass_count += 1;
        } else {
            fail_count += 1;
        }
        println!(
            "{:<50} {:>10.6} {:>10.6} {:>10.6} {:>6}",
            r.tensor_name, r.cosine_sim, r.max_abs, r.mean_abs, status
        );
    }

    println!("{}", "-".repeat(90));
    println!("Total: {} passed, {} failed", pass_count, fail_count);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have cosine=1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have cosine≈0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "opposite vectors should have cosine=-1.0, got {sim}"
        );
    }

    #[test]
    fn test_max_abs_diff() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.2, 2.7];
        let d = max_abs_diff(&a, &b);
        assert!((d - 0.3).abs() < 1e-6, "expected max_abs=0.3, got {d}");
    }

    #[test]
    fn test_mean_abs_diff() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 3.1];
        let d = mean_abs_diff(&a, &b);
        assert!((d - 0.1).abs() < 1e-6, "expected mean_abs=0.1, got {d}");
    }

    #[test]
    fn test_cosine_near_alignment() {
        // Simulate near-alignment: small perturbation
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = a.iter().map(|&x| x + 0.001).collect();
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim > 0.999,
            "near-identical vectors should have cosine > 0.999, got {sim}"
        );
    }
}
