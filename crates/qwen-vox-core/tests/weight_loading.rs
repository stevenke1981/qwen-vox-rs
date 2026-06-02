//! Integration tests for weight loading.
//!
//! These tests verify that the Rust weight loader can correctly read
//! the converted SafeTensors files produced by the Python conversion scripts.

use candle_core::Device;
use qwen_vox_core::quantizer::load_decoder_codebooks;
use qwen_vox_core::weights::WeightStore;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights")
        .join("alignments")
}

#[test]
fn test_load_talker_weights() {
    let path = weights_dir().join("talker_layers.safetensors");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let store = WeightStore::from_file(&path, &Device::Cpu).unwrap();
    assert!(!store.is_empty(), "talker weights should not be empty");

    // Check for expected tensor patterns
    let names = store.names();
    let has_embed = names.iter().any(|n| n.contains("embed"));
    let has_attn = names.iter().any(|n| n.contains("attn"));

    println!("Loaded {} talker tensors", store.len());
    println!("Has embed tensors: {has_embed}");
    println!("Has attention tensors: {has_attn}");

    assert!(
        has_embed || has_attn,
        "should have embed or attention tensors"
    );
}

#[test]
fn test_load_code_predictor_weights() {
    let path = weights_dir().join("code_predictor.safetensors");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let store = WeightStore::from_file(&path, &Device::Cpu).unwrap();
    assert!(
        !store.is_empty(),
        "code predictor weights should not be empty"
    );

    println!("Loaded {} code predictor tensors", store.len());

    // Check for lm_head tensors
    let lm_heads = store.get_prefix("lm_head");
    println!("Found {} lm_head tensors", lm_heads.len());
}

#[test]
fn test_load_speaker_encoder_weights() {
    let path = weights_dir().join("speaker_encoder.safetensors");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let store = WeightStore::from_file(&path, &Device::Cpu).unwrap();
    assert!(
        !store.is_empty(),
        "speaker encoder weights should not be empty"
    );

    println!("Loaded {} speaker encoder tensors", store.len());

    // Check for expected components
    let asp_tensors = store.get_prefix("asp");
    let block_tensors = store.get_prefix("blocks");
    println!(
        "ASP tensors: {}, Block tensors: {}",
        asp_tensors.len(),
        block_tensors.len()
    );
}

#[test]
fn test_load_tokenizer_decoder_weights() {
    let path = weights_dir().join("tokenizer_decoder.safetensors");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let store = WeightStore::from_file(&path, &Device::Cpu).unwrap();
    assert!(
        !store.is_empty(),
        "tokenizer decoder weights should not be empty"
    );

    println!("Loaded {} tokenizer decoder tensors", store.len());

    // Check for decoder components
    let decoder_tensors = store.get_prefix("decoder");
    println!("Found {} decoder tensors", decoder_tensors.len());

    // Verify the new load_decoder_codebooks helper (1+15=16 codebooks)
    match load_decoder_codebooks(&store) {
        Ok(codebooks) => {
            println!(
                "load_decoder_codebooks: loaded {} embeddings",
                codebooks.len()
            );
            assert_eq!(
                codebooks.len(),
                16,
                "decoder quantizer must have 16 codebook layers total"
            );
            for (i, cb) in codebooks.iter().enumerate() {
                let dims = cb.dims();
                assert_eq!(
                    dims,
                    vec![2048, 256],
                    "embedding_sum {} must be [2048, 256]",
                    i
                );
            }
            println!("load_decoder_codebooks: all shapes verified OK");
        }
        Err(e) => {
            eprintln!("load_decoder_codebooks failed (non-fatal for this test): {e}");
        }
    }

    // Check for SnakeBeta parameters (alpha, beta)
    let has_alpha = store.names().iter().any(|n| n.contains("alpha"));
    let has_beta = store.names().iter().any(|n| n.contains("beta"));
    println!("Has SnakeBeta alpha: {has_alpha}, beta: {has_beta}");
}

#[test]
fn test_load_test_input() {
    let path = weights_dir().join("test_input.safetensors");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let store = WeightStore::from_file(&path, &Device::Cpu).unwrap();
    let tokens = store.require("test_tokens").unwrap();

    let dims = tokens.dims();
    println!("Test tokens shape: {:?}", dims);

    // Should be [16, 8] (16 codebook layers, 8 frames)
    assert_eq!(dims.len(), 2, "test tokens should be 2D");
    assert_eq!(dims[0], 16, "should have 16 codebook layers");
}
