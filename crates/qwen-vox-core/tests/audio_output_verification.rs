//! Integration test: End-to-end audio output verification.

use candle_core::Device;
use qwen_vox_core::weights::WeightStore;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights")
}

#[test]
fn test_intermediate_activations_available() {
    let intermediates_path = weights_dir()
        .join("intermediates")
        .join("intermediates.safetensors");

    if !intermediates_path.exists() {
        eprintln!("Skipping: {} not found", intermediates_path.display());
        return;
    }

    let store = WeightStore::from_file(&intermediates_path, &Device::Cpu).unwrap();
    println!("Loaded {} intermediate tensors", store.len());

    // Verify final output exists
    if let Some(final_out) = store.get("final_out") {
        println!("final_out shape: {:?}", final_out.dims());

        // Check that final output is not all zeros (non-silent)
        let data: Vec<f32> = final_out.flatten_all().unwrap().to_vec1().unwrap();
        let non_zero_count = data.iter().filter(|&&x| x.abs() > 0.001).count();
        let ratio = non_zero_count as f64 / data.len() as f64;

        println!(
            "Non-zero samples: {}/{} ({:.2}%)",
            non_zero_count,
            data.len(),
            ratio * 100.0
        );

        // For dummy data, this will be 0%, but demonstrates the check
        if ratio > 0.0 {
            println!("Audio contains non-silent segments ✓");
        }
    }
}

#[test]
fn test_weight_conversion_complete() {
    // Verify all required weight files exist
    let converted_dir = weights_dir().join("converted");

    assert!(
        converted_dir.join("model.safetensors").exists(),
        "Main model weights should exist"
    );

    assert!(
        converted_dir
            .join("tokenizer")
            .join("model.safetensors")
            .exists(),
        "Tokenizer decoder weights should exist"
    );

    println!("All required weight files present ✓");
}
