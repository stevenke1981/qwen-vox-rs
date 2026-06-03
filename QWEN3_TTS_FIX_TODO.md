# Qwen3-TTS Rust Fix TODO

## Current Status

- The CLI must use the in-app Qwen3-TTS Rust path, not formant fallback, SAPI, Edge TTS, Python TTS, or external services.
- CUDA build works on this machine:
  - `cargo check -p qwen-vox-cli --features cuda`
- Qwen3 generation now reaches model inference and writes WAV files after fixing the first performance/runtime blockers.
- Current generated WAV is still unacceptable: it sounds clipped/distorted and does not sound like real human speech.

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

## Immediate Evidence

- `--max-frames 1` completed and wrote `out/qwen3_cuda_smoke.wav`.
- `--max-frames 4` completed and wrote `out/qwen3_cuda_4frames.wav`.
- `--max-frames 16` completed and wrote `out/qwen3_cuda_16frames.wav`.
- The 16-frame file is non-silent but too hot:
  - duration: about 1.28s
  - peak: about 0.999969
  - RMS: about 0.8195
- That RMS is far too high for normal speech and explains the audible clipping/破音 symptom, but clipping may be only a symptom. The generated codec frames may also be wrong.

## Root-Cause TODO

1. Add safe WAV post-processing before writing.
   - Remove non-finite samples.
   - Remove DC offset.
   - Peak-normalize to around `0.90`.
   - Add a simple soft limiter instead of hard clamp.
   - This should reduce 破音, but it will not make bad codec frames sound human.

2. Log generated codec frames for short prompts.
   - Print the first N `[q0..q15]` frames behind a debug flag.
   - Check for repeated frames, out-of-range codes, early EOS, or obviously collapsed argmax output.
   - Compare frame distribution against known-good Python Qwen3-TTS output if available.

3. Verify Qwen3 prompt and codec prefill against `weights/hf_original/config.json`.
   - Current Rust code manually builds:
     - `codec_think_id = 2154`
     - `codec_think_bos_id = 2156`
     - language id
     - `codec_think_eos_id = 2157`
     - `codec_pad_id = 2148`
     - `codec_bos_id = 2149`
   - Confirm this sequence matches the upstream generation template exactly.
   - Confirm whether `custom_voice` requires a speaker id or reference speaker embedding. The full model config has speaker IDs such as `serena`, `vivian`, `ryan`, etc.; the current CLI does not expose a speaker option.

4. Implement speaker selection if required.
   - Add CLI option such as `--speaker serena`.
   - Insert the correct speaker token/prefix according to upstream Qwen3-TTS behavior.
   - Add validation against `spk_id` in config instead of hard-coded guesses.

5. Replace pure argmax generation if upstream uses sampling.
   - Current `Talker::predict_codes` uses argmax for q0 and residual codes.
   - Check upstream settings for temperature, top-k, top-p, repetition penalty, EOS handling, and special token suppression.
   - Implement the exact decoding strategy before judging audio quality.

6. Numerically align the talker.
   - Export a tiny upstream Python trace for:
     - tokenized prompt IDs
     - prefill embeddings
     - first hidden state
     - first q0 logits
     - first residual logits
   - Add Rust tests that compare these tensors with tolerances.
   - Fix mismatches layer by layer.

7. Numerically align the speech tokenizer decoder.
   - Feed known-good codec frames from upstream Python into the Rust decoder.
   - Compare intermediate outputs:
     - RVQ decode
     - pre-conv
     - pre-transformer
     - upsample stages
     - post-transformer decoder waveform
   - If known-good codec frames sound bad in Rust, the decoder path is wrong.
   - If known-good codec frames sound good in Rust, the talker generation path is wrong.

8. Add audio-quality validation.
   - Write a small Rust or PowerShell checker for WAV peak, RMS, clipping ratio, duration, and non-finite sample count.
   - Fail tests if RMS is speech-impossible or clipping ratio is high.
   - Keep listening tests/manual QA as a final gate, because metrics alone cannot prove human-like speech.

9. Improve speed after correctness.
   - Use release builds for real generation:
     - `cargo run -p qwen-vox-cli --release --features cuda -- generate --device cuda ...`
   - Avoid reloading model weights for every utterance by adding a long-lived serve/batch mode.
   - Add dynamic config support before using `weights/model-0.6b`; current `Talker` constants are hard-coded for the 1.7B/custom_voice hidden size.

## Suggested Next Commands

```powershell
cargo check -p qwen-vox-cli --features cuda
cargo test -p qwen-vox-core custom_ops::tests
cargo test -p qwen-vox-core transformer::tests::test_grouped_query_attention_shape_mha_and_gqa
cargo run -p qwen-vox-cli --features cuda -- generate --device cuda --text "Hello from Qwen three TTS." --output out\qwen3_cuda_16frames.wav --language english --max-frames 16
```

## Next Code Change

Start with TODO 1 plus TODO 2:

- Add `normalize_waveform_for_wav(samples: &mut [f32])`.
- Add tests for clipping prevention.
- Add optional debug logging of generated codec frames.

Then run the same 16-frame prompt again. If it is less distorted but still not human-like, move immediately to TODO 3 through TODO 7 instead of tuning audio gain further.
