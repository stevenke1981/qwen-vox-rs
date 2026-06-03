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

Dialect and Taiwan Taigi / Minnan usage notes are documented in
[`docs/qwen3_tts_minnan_hokkien_guide.md`](docs/qwen3_tts_minnan_hokkien_guide.md).
The Rust CLI currently supports CustomVoice preset generation; Base voice clone
and Voice Design / instruct control are tracked as follow-up implementation
work.

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
  --seed 42 `
  --speed 1.0 `
  --text "你好，這是 Qwen3 TTS 產生的語音。" `
  --output out\speech.wav
```

Run the CPU binary:

```powershell
.\dist\qwen-vox-cpu.exe generate `
  --device cpu `
  --language chinese `
  --speaker vivian `
  --seed 42 `
  --text "你好，這是 Qwen3 TTS 產生的語音。" `
  --output out\speech-cpu.wav
```

CPU generation works as a pure Rust path, but the full 1.7B Qwen3-TTS model is
slow on CPU. CUDA is recommended for practical generation.

## Voice Clone Status

Official Qwen3-TTS voice cloning is a Base-model feature. The CustomVoice
weights in `weights/hf_original` provide preset speakers such as `vivian`, but
do not include `speaker_encoder_config`; official Hugging Face code rejects
`generate_voice_clone()` for CustomVoice weights.

This repository now exposes the clone CLI entry point and validates the model
type:

```powershell
.\dist\qwen-vox-cuda.exe clone `
  --model-dir weights\model-0.6b `
  --device cuda `
  --ref-audio reference.wav `
  --ref-text "Reference transcript text." `
  --text "Text to synthesize." `
  --output out\clone.wav
```

The next implementation work is Base speaker encoder support and dynamic 0.6B
talker loading. Until that is complete, `clone` reports an explicit error after
validating that Base weights and reference inputs are present.

Useful generation options:

- `--seed 42`: makes sampling reproducible for the same text, speaker, language,
  and sampling options. `--temperature 0` uses argmax and is deterministic even
  without a seed.
- `--speed 1.2`: shortens the decoded waveform for faster speech. Values below
  `1.0` slow the output down. Current implementation is a lightweight
  post-decode duration scale, not a pitch-preserving neural prosody control.
- `--max-frames 96`: caps generated codec frames. Higher values allow longer
  speech but increase generation time.

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

Default CLI paths expect this local structure:

```text
weights/
  hf_original/
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    vocab.json
    merges.txt
    speech_tokenizer/
      model.safetensors
  model-0.6b/
    config.json
    model.safetensors
    tokenizer_config.json
    vocab.json
    speech_tokenizer/
      model.safetensors
```

The Rust tokenizer can load either `tokenizer.json` or the Hugging Face
directory files, but keep `vocab.json`, `merges.txt`, and
`tokenizer_config.json` with the model directory for parity diagnostics.

## Performance Notes

The current CLI is a one-shot process: every `generate` invocation loads the
talker weights and speech tokenizer weights before generation. That startup cost
is visible in short runs. The generation kernel is also a straightforward Candle
implementation with manual attention and per-frame/per-residual incremental
steps; it is correct enough for audible speech, but it is not yet optimized to
match fused CUDA implementations.

Next performance work:

- Keep a long-lived model process/server so repeated requests reuse loaded
  weights.
- Replace manual attention paths with fused attention where available.
- Reduce per-residual allocation and JSON/debug overhead in normal generation.
- Profile 96-frame CUDA generation separately for load time vs token generation
  time.

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
