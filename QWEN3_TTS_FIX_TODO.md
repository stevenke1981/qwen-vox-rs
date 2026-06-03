# Qwen3-TTS Rust Fix TODO

## Current Status

- The CLI must use the in-app Qwen3-TTS Rust path, not formant fallback, SAPI, Edge TTS, Python TTS, or external services.
- CUDA build works on this machine:
  - `cargo check -p qwen-vox-cli --features cuda`
- Qwen3 generation now reaches model inference and writes WAV files after fixing the first performance/runtime blockers.
- Current generated WAV is still unacceptable: it is mostly high-frequency noise and does not sound like real human speech.
- Upstream generation flow and Rust divergence checklist are documented in:
  - `QWEN3_TTS_GENERATION_FLOW.md`

## Fixes Already Applied

- `crates/qwen-vox-core/src/weights.rs`
  - Changed `WeightStore::from_file` from manual SafeTensors parsing with F16/BF16 -> F32 expansion to Candle mmap SafeTensors loading.
  - This preserves BF16/F16 weights and avoids expanding the 4GB model into roughly 8GB of f32 tensors.

- `crates/qwen-vox-core/src/custom_ops.rs`
  - Added dtype alignment for small helper tensors used in LayerScale, SnakeBeta, attention scaling, and attention masks.
  - This fixed the CUDA runtime error:
    `dtype mismatch in mul, lhs: BF16, rhs: F32`.

- `crates/qwen-vox-cli/src/main.rs`
  - Added phase timing logs around tokenizer load, talker weight load, talker build, decoder weight load, codec frame generation, and waveform decode.
  - Added safe WAV post-processing:
    - remove non-finite samples
    - remove DC offset
    - peak-normalize to `0.90`
    - apply a soft limiter before writing
  - Added `--debug-frames` logging for generated codec frames.

- `crates/qwen-vox-core/src/sampling.rs`
  - Added temperature/top-k/top-p/repetition-penalty sampling for q0.

- `crates/qwen-vox-core/src/talker.rs`
  - Added speaker token insertion support for the CustomVoice prompt path.
  - Added incremental backbone generation with K/V cache.

- `crates/qwen-vox-core/src/conv_decoder.rs`
  - Fixed tokenizer decoder upsample strides to `[8, 5, 4, 3]`, matching `decode_upsample_rate=1920`.

- `crates/qwen-vox-core/src/transformer.rs`
  - Added interleaved RoPE for Q/K.
  - RoPE is applied before K/V cache append.
  - Cached incremental generation uses the existing cache length as the position offset.
  - Optional Q/K per-head RMSNorm now uses the block epsilon instead of a hard-coded `1e-5`.

- `crates/qwen-vox-core/src/talker.rs`
  - Enabled `rope_theta=1000000` for talker backbone and code predictor.
  - Changed talker/code predictor RMSNorm epsilon, including Q/K norm epsilon, to `1e-6`, matching `weights/hf_original/config.json`.

- `crates/qwen-vox-core/src/pipeline.rs`
  - Enabled `rope_theta=10000` for speech tokenizer decoder pre-transformer.

## Immediate Evidence

- `--max-frames 1` completed and wrote a WAV of the expected frame duration:
  - 1 generated codec frame -> 1920 samples -> 0.080s at 24 kHz.
- After RoPE:
  - `out/qwen3_rope_smoke.wav` completed on CUDA with `--max-frames 1`.
  - `out/qwen3_rope_16frames.wav` completed on CUDA with `--max-frames 16`.
  - `out/qwen3_rope_eps_16frames.wav` completed on CUDA after Q/K norm epsilon alignment.
  - 16 frames -> 30720 samples -> 1.280s at 24 kHz.
  - latest codec frame diagnostics: code range `[6, 2041]`, unique q0 values `13/16`, repeated consecutive frames `0`.
- The audio is non-silent, but still sounds like high-frequency noise.
- The remaining issue is now more likely model alignment than final WAV gain:
  - generated codec frames may be invalid
  - talker transformer may be numerically wrong
  - tokenizer decoder may still be numerically wrong for known-good frames
- FFmpeg metrics for `out/qwen3_rope_eps_16frames.wav` still look unlike normal speech:
  - RMS level: `-2.73 dB`
  - max volume: `-0.9 dB`
  - peak count: `15696`
  - crest factor: `1.23`
  - zero crossing rate: `0.0865`

## Root-Cause TODO

1. Verify Qwen3 prompt and codec prefill against `weights/hf_original/config.json` and upstream Python.
   - Current Rust code manually builds:
     - `codec_think_id = 2154`
     - `codec_think_bos_id = 2156`
     - language id
     - `codec_think_eos_id = 2157`
     - `codec_pad_id = 2148`
     - `codec_bos_id = 2149`
   - Confirm this sequence matches the upstream generation template exactly.
   - Confirm whether `custom_voice` requires a speaker id or reference speaker embedding. The full model config has speaker IDs such as `serena`, `vivian`, `ryan`, etc.; the current CLI does not expose a speaker option.

2. Replace hard-coded speaker IDs with config-driven speaker IDs.
   - Parse `spk_id` from `weights/hf_original/config.json`.
   - Reject unknown speaker names at CLI parse time.
   - Keep `vivian` or another official speaker as the default.

3. Replace pure argmax generation if upstream uses sampling.
   - Current `Talker::predict_codes` uses argmax for q0 and residual codes.
   - Check upstream settings for temperature, top-k, top-p, repetition penalty, EOS handling, and special token suppression.
   - Implement the exact decoding strategy before judging audio quality.

4. Numerically align the talker against upstream Python.
   - Export a tiny upstream Python trace for:
     - tokenized prompt IDs
     - prefill embeddings
     - first hidden state
     - first q0 logits
     - first residual logits
   - Add Rust tests that compare these tensors with tolerances.
   - Fix mismatches layer by layer.

5. Numerically align the speech tokenizer decoder.
   - Feed known-good codec frames from upstream Python into the Rust decoder.
   - Compare intermediate outputs:
     - RVQ decode
     - pre-conv
     - pre-transformer
     - upsample stages
     - post-transformer decoder waveform
   - If known-good codec frames sound bad in Rust, the decoder path is wrong.
   - If known-good codec frames sound good in Rust, the talker generation path is wrong.

6. Add audio-quality validation.
   - Write a small Rust or PowerShell checker for WAV peak, RMS, clipping ratio, duration, and non-finite sample count.
   - Fail tests if RMS is speech-impossible or clipping ratio is high.
   - Keep listening tests/manual QA as a final gate, because metrics alone cannot prove human-like speech.

7. Improve speed after correctness.
   - Use release builds for real generation:
     - `cargo run -p qwen-vox-cli --release --features cuda -- generate --device cuda ...`
   - Avoid reloading model weights for every utterance by adding a long-lived serve/batch mode.
   - Add dynamic config support before using `weights/model-0.6b`; current `Talker` constants are hard-coded for the 1.7B/custom_voice hidden size.

## Suggested Next Commands

```powershell
cargo check -p qwen-vox-cli --features cuda
cargo test -p qwen-vox-core custom_ops::tests
cargo test -p qwen-vox-core transformer -- --nocapture
cargo run -p qwen-vox-cli --features cuda -- generate --device cuda --text "Hello from Qwen three TTS." --output out\qwen3_cuda_16frames.wav --language english --max-frames 16
```

## Next Code Change

Start with TODO 1 plus TODO 4:

- Follow `QWEN3_TTS_GENERATION_FLOW.md`.
- Export a known-good upstream Python trace for the same text/speaker/language prompt.
- Compare the Rust prompt token IDs, prefill embedding sequence, first backbone hidden state, first q0 logits, and first residual logits.
- If q0 logits already diverge before the codec decoder, fix talker prompt/position/cache alignment.
- If known-good upstream codec frames still decode as noise in Rust, fix the tokenizer decoder path.
