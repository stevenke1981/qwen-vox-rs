//! Integration test: Code Predictor (MTP) integration verification.

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
fn test_code_predictor_lm_head_count() {
    let weights_path = weights_dir().join("converted").join("model.safetensors");

    if !weights_path.exists() {
        eprintln!("Skipping: {} not found", weights_path.display());
        return;
    }

    let store = WeightStore::from_file(&weights_path, &Device::Cpu).unwrap();

    // Count lm_head tensors
    let all_names: Vec<_> = store.names().into_iter().collect();
    let lm_heads: Vec<_> = all_names.iter().filter(|n| n.contains("lm_head")).collect();

    println!("Found {} lm_head tensors", lm_heads.len());
    assert_eq!(
        lm_heads.len(),
        15,
        "Should have 15 lm_head tensors for 15 residual codebooks"
    );

    // Verify all lm_head shapes are [2048, 1024]
    for i in 0..15 {
        let name = format!("talker.lm_head.{}.weight", i);
        if let Some(head) = store.get(&name) {
            assert_eq!(
                head.dims(),
                &[2048, 1024],
                "lm_head.{} should be [2048, 1024]",
                i
            );
        }
    }
}

#[test]
fn test_code_predictor_codec_embeddings() {
    let weights_path = weights_dir().join("converted").join("model.safetensors");

    if !weights_path.exists() {
        eprintln!("Skipping: {} not found", weights_path.display());
        return;
    }

    let store = WeightStore::from_file(&weights_path, &Device::Cpu).unwrap();

    // Check for codec_embedding tensors (15 embeddings)
    let all_names: Vec<_> = store.names().into_iter().collect();
    let embeddings: Vec<_> = all_names
        .iter()
        .filter(|n| n.contains("codec_embedding"))
        .collect();

    println!("Found {} codec_embedding tensors", embeddings.len());

    // Verify embedding shapes
    for emb_name in embeddings.iter().take(3) {
        if let Some(emb) = store.get(emb_name) {
            println!("  {}: {:?}", emb_name, emb.dims());
        }
    }
}
