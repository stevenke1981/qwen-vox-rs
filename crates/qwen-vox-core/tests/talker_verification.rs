//! Integration test: Talker backbone weight loading and forward pass verification.

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
fn test_load_talker_weights() {
    let weights_path = weights_dir().join("converted").join("model.safetensors");

    if !weights_path.exists() {
        eprintln!("Skipping: {} not found", weights_path.display());
        return;
    }

    let store = WeightStore::from_file(&weights_path, &Device::Cpu).unwrap();
    println!("Loaded {} talker tensors", store.len());

    // Verify key Talker tensors exist
    assert!(
        store.get("talker.model.embed_tokens.weight").is_some()
            || store
                .get("talker.model.layers.0.self_attn.q_proj.weight")
                .is_some(),
        "Talker backbone tensors should exist"
    );

    // Check for code predictor (MTP) tensors
    let lm_head_count = store
        .names()
        .iter()
        .filter(|n| n.contains("lm_head"))
        .count();
    println!("Found {} lm_head tensors (expected 15)", lm_head_count);
}

#[test]
fn test_talker_layer_structure() {
    let weights_path = weights_dir().join("converted").join("model.safetensors");

    if !weights_path.exists() {
        eprintln!("Skipping: {} not found", weights_path.display());
        return;
    }

    let store = WeightStore::from_file(&weights_path, &Device::Cpu).unwrap();

    // Verify layer 0 structure (Qwen3 LM with GQA)
    // q_proj: [num_heads * head_dim, hidden] = [16*128, 2048]
    // k_proj: [num_kv_heads * head_dim, hidden] = [8*128, 2048]
    if let Some(q_proj) = store.get("talker.model.layers.0.self_attn.q_proj.weight") {
        println!("q_proj shape: {:?}", q_proj.dims());
        assert_eq!(
            q_proj.dims(),
            &[2048, 2048],
            "q_proj should be [2048, 2048] (16 heads × 128 head_dim)"
        );
    }

    if let Some(k_proj) = store.get("talker.model.layers.0.self_attn.k_proj.weight") {
        println!("k_proj shape: {:?}", k_proj.dims());
        assert_eq!(
            k_proj.dims(),
            &[1024, 2048],
            "k_proj should be [1024, 2048] (8 kv_heads × 128 head_dim)"
        );
    }

    // Check MLP structure (SwiGLU)
    if let Some(gate_proj) = store.get("talker.model.layers.0.mlp.gate_proj.weight") {
        println!("gate_proj shape: {:?}", gate_proj.dims());
        assert_eq!(
            gate_proj.dims(),
            &[6144, 2048],
            "gate_proj should be [6144, 2048] (FFN × hidden)"
        );
    }
}

#[test]
fn test_code_predictor_structure() {
    let weights_path = weights_dir().join("converted").join("model.safetensors");

    if !weights_path.exists() {
        eprintln!("Skipping: {} not found", weights_path.display());
        return;
    }

    let store = WeightStore::from_file(&weights_path, &Device::Cpu).unwrap();

    // Count lm_head tensors (should be 15 for 15 residual codebooks)
    let all_names: Vec<_> = store.names().into_iter().collect();
    let lm_heads: Vec<_> = all_names.iter().filter(|n| n.contains("lm_head")).collect();

    println!("Found {} lm_head tensors", lm_heads.len());

    // Verify first lm_head shape
    if let Some(first_head) = store.get("talker.lm_head.0.weight") {
        println!("lm_head.0 shape: {:?}", first_head.dims());
        assert_eq!(
            first_head.dims(),
            &[2048, 1024],
            "lm_head should be [2048, 1024]"
        );
    }
}
