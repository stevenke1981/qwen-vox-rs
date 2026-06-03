# qwen-vox-rs

Rust command-line Qwen3-TTS speech generation.

This repository contains:

- `qwen-vox-core`: Candle-based Qwen3-TTS tokenizer/weight loading, talker, codec decoder modules, and alignment tests.
- `qwen-vox-cli`: command-line Qwen3-TTS WAV generation.

The CLI is intentionally usable without Python, libtorch, ONNX Runtime, or PyTorch FFI.
With the local Qwen3-TTS weights in `weights/hf_original`, the Rust CLI can
generate audible human speech.

## Quick Start

```powershell
cargo run -p qwen-vox-cli --features cuda -- generate `
  --device cuda `
  --text "Hello from Qwen three TTS." `
  --output out/qwen3.wav `
  --language english
```

Chinese text:

```powershell
cargo run -p qwen-vox-cli --features cuda -- generate `
  --device cuda `
  --text "你好，這是 Qwen3 TTS 產生的語音。" `
  --output out/qwen3-zh.wav `
  --language chinese `
  --speaker vivian
```

The generated file is a 24 kHz, 16-bit mono WAV. The 12 Hz tokenizer runs at 12.5 codec frames per second; each codec frame decodes to 1,920 samples. By default `--max-frames 0` auto-estimates a frame cap from text length. Pass an explicit `--max-frames N` to override it.

## Release Builds

Build both local Windows release binaries:

```powershell
.\scripts\build_release.ps1
```

Outputs:

- `dist\qwen-vox-cpu.exe`
- `dist\qwen-vox-cuda.exe`
- `dist\BUILD_INFO.txt`

Run the CUDA binary:

```powershell
.\dist\qwen-vox-cuda.exe generate `
  --device cuda `
  --language chinese `
  --speaker vivian `
  --text "你好，這是 Qwen3 TTS 產生的語音。" `
  --output out\speech.wav
```

Run the CPU binary:

```powershell
.\dist\qwen-vox-cpu.exe generate `
  --device cpu `
  --language chinese `
  --speaker vivian `
  --text "你好，這是 Qwen3 TTS 產生的語音。" `
  --output out\speech-cpu.wav
```

CPU generation works as a pure Rust path, but the full 1.7B Qwen3-TTS model is
slow on CPU. CUDA is recommended for practical generation.

## Current Generation Path

`qwen-vox-cli generate` now uses the local Qwen3-TTS model path:

1. Tokenize a ChatML-style text prompt.
2. Load `weights/hf_original/model.safetensors`.
3. Build the Qwen3-TTS talker and generate 16-level codec frames.
4. Load `weights/hf_original/speech_tokenizer/model.safetensors`.
5. Decode codec frames to a 24 kHz WAV.

This means:

- The CLI no longer uses Windows SAPI or the Rust formant fallback for `generate`.
- CUDA builds are strongly recommended.
- Large model weights are not committed.
- The Qwen3 talker uses transformer KV cache for incremental generation, but CPU generation is still not practical for the full model.

## Model Files

Large `.safetensors`, `.pt`, `.bin`, `weights/`, and `models/` paths are ignored by Git. Keep downloaded or converted model assets locally under `weights/` or `models/`.

The current CLI expects local files such as:

- `weights/hf_original/model.safetensors`
- `weights/hf_original/speech_tokenizer/model.safetensors`
- `weights/hf_original/tokenizer_config.json`
- `weights/hf_original/vocab.json`
- `weights/hf_original/merges.txt`

## Validation

```powershell
cargo fmt
cargo test
cargo clippy --all-targets -- -D warnings
cargo check -p qwen-vox-cli --features cuda
.\scripts\build_release.ps1
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
scripts/
  build_release.ps1        # CPU/CUDA Windows release build
```
