# qwen-vox-rs

Rust workspace for Qwen-style speech generation experiments.

This repository contains:

- `qwen-vox-core`: Candle-based model building blocks, tokenizer/weight loading, codec decoder modules, and a Rust-only fallback speech synthesizer.
- `qwen-vox-cli`: command-line WAV generation.

The CLI is intentionally usable without Python, libtorch, ONNX Runtime, or PyTorch FFI.

## Quick Start

```powershell
cargo run -p qwen-vox-cli -- generate `
  --text "Hello from Rust speech synthesis" `
  --output out/hello.wav
```

Traditional Chinese text also works through the fallback synthesizer:

```powershell
cargo run -p qwen-vox-cli -- generate `
  --text "你好，這是 Rust 產生的人聲。" `
  --output out/hello-zh.wav `
  --pitch 170 `
  --speed 0.95
```

The generated file is a 24 kHz, 16-bit mono WAV.

## Current Generation Path

`qwen-vox-cli generate` currently uses a pure Rust formant synthesizer so the project can produce audible, non-silent speech immediately. The Candle modules for tokenizer, talker, codec decoding, flow matching, and weight loading remain in `qwen-vox-core` for the neural Qwen3-TTS path.

This means:

- Rust-only speech output works now.
- Large model weights are not committed.
- Full neural Qwen3-TTS inference still depends on completing model-specific wiring and validation.

## Model Files

Large `.safetensors`, `.pt`, `.bin`, `weights/`, and `models/` paths are ignored by Git. Keep downloaded or converted model assets locally under `weights/` or `models/`.

The existing tests can use local files such as:

- `weights/converted/model.safetensors`
- `weights/converted/tokenizer/model.safetensors`
- `weights/intermediates/intermediates.safetensors`

## Validation

```powershell
cargo fmt
cargo test
cargo clippy --all-targets -- -D warnings
```

## Repository Layout

```text
crates/
  qwen-vox-core/
    src/
      speech_synth.rs      # Rust-only audible speech fallback
      pipeline.rs          # codec decode pipeline
      talker.rs            # autoregressive code generation scaffold
      tokenizer.rs         # tokenizer loader
      weights.rs           # safetensors loading
  qwen-vox-cli/
    src/main.rs            # CLI WAV generation
```
