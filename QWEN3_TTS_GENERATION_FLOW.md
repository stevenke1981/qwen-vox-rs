# Qwen3-TTS Generation Flow and Rust Debug Plan

This file records how upstream Qwen3-TTS generates speech and how the Rust
implementation should be checked step by step. The immediate goal is not to
tune output gain. The goal is to find the first tensor/code path that diverges
from upstream and causes high-frequency noise instead of human speech.

## Upstream Sources

- Hugging Face Space wrapper:
  <https://huggingface.co/spaces/Qwen/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py>
- Hugging Face Space model implementation:
  <https://huggingface.co/spaces/Qwen/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py>
- Local config used by this repo:
  `weights/hf_original/config.json`
- Speech tokenizer config:
  `weights/hf_original/speech_tokenizer/config.json`

## Verified Human-Voice Reference

The first verified official end-to-end reference is documented in:

- `OFFICIAL_QWEN3_TTS_REFERENCE_FLOW.md`

The output file is:

- `out/official_qwen3_reference.wav`

This file is confirmed by listening as normal human speech. It is the current
golden reference for Rust end-to-end work.

Do not use `out/python_decoder_test_output.wav` as a speech-quality reference:
that file decodes fixed test tokens and can sound like high-frequency noise even
when the decoder is numerically useful.

## Official CustomVoice Flow

1. Build assistant text.
   - Format:
     `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
   - The wrapper tokenizes this through the official processor.

2. Merge generation defaults.
   - Main talker defaults:
     - `do_sample=true`
     - `top_k=50`
     - `top_p=1.0`
     - `temperature=0.9`
     - `repetition_penalty=1.05`
     - `max_new_tokens=2048`
   - Subtalker/code-predictor defaults:
     - `subtalker_dosample=true`
     - `subtalker_top_k=50`
     - `subtalker_top_p=1.0`
     - `subtalker_temperature=0.9`

3. CustomVoice wrapper calls `model.generate(...)`.
   - Inputs:
     - `input_ids`
     - optional `instruct_ids`
     - `languages`
     - `speakers`
     - `non_streaming_mode=true` by default for CustomVoice
   - Output:
     - `talker_codes_list`
     - hidden states for optional downstream use

4. Build language and speaker codec prefill.
   - If language is `auto`:
     `[codec_nothink_id, codec_think_bos_id, codec_think_eos_id]`
   - Else:
     `[codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]`
   - Then append either:
     - speaker embedding / speaker token equivalent, if present
     - `[codec_pad_id, codec_bos_id]`
   - Config values for the current 1.7B CustomVoice model:
     - `codec_think_id=2154`
     - `codec_nothink_id=2155`
     - `codec_think_bos_id=2156`
     - `codec_think_eos_id=2157`
     - `codec_pad_id=2148`
     - `codec_bos_id=2149`
     - `codec_eos_token_id=2150`

5. Build initial talker embeddings.
   - Role prefix:
     text projection of `input_id[:, :3]`
   - Codec/text prefill:
     `tts_pad_embed * (codec_len - 2) + tts_bos_embed`, added to codec
     embeddings except the last codec token.
   - Streaming-style mode:
     - Add only the first text token plus final codec BOS embedding.
     - `trailing_text_hidden = text_projection(input_id[:, 4:-5]) + tts_eos_embed`
   - Non-streaming mode:
     - Remove the first-token streaming append.
     - Add all target text tokens plus `tts_eos_embed`, each paired with
       `codec_pad_id`.
     - Append `tts_pad_embed + codec_bos_id`.
     - Set `trailing_text_hidden = tts_pad_embed`.

6. Pad batched prompts and build attention masks.
   - Official code left-pads variable-length prompt embeddings.
   - `talker_attention_mask` is passed into generation.
   - RoPE position ids are derived from `attention_mask.cumsum(...)`, not only
     from cache length.

7. Talker autoregressive generation.
   - Prefill stage:
     - Run the talker transformer with `inputs_embeds`.
     - Produce q0 logits from `codec_head(hidden_states)`.
   - Generation stage:
     - Use the previous q0 token embedding.
     - Run the residual code predictor with:
       `inputs_embeds = cat(past_hidden, last_id_hidden)`
     - Generate residual codes q1..q15 with the code predictor's
       `generate(...)` path.
     - Sum q0 embedding plus residual code embeddings into the next frame
       embedding.
     - Add `trailing_text_hidden[generation_step]` if available; otherwise add
       `tts_pad_embed`.
     - Run the main talker with KV cache and updated `cache_position`.

8. Stop condition.
   - Decode stops when q0 equals `codec_eos_token_id`.
   - Effective codes are trimmed before speech tokenizer decode.

9. Speech tokenizer decode.
   - Official wrapper calls:
     `speech_tokenizer.decode([{"audio_codes": codes}])`
   - For voice clone with reference codes, upstream prepends reference codes
     during decode and cuts the reference waveform portion afterward.
   - CustomVoice without reference codes decodes generated codes directly.

## Rust Current Flow

1. CLI builds the same assistant prompt string in `qwen3_prompt`.

2. CLI tokenizes with `tokenizer.json`.

3. CLI loads:
   - talker weights: `weights/hf_original/model.safetensors`
   - decoder weights: `weights/hf_original/speech_tokenizer/model.safetensors`

4. `Talker::generate_qwen3_base` builds codec prefill:
   - `[2154, 2156, language_id, 2157, speaker_id, 2148, 2149]`
   - or `[2155, 2156, 2157, speaker_id?, 2148, 2149]` for auto

5. Rust now uses upstream-style CustomVoice `non_streaming_mode=true` embedding
   layout:
   - role prefix
   - pad/BOS codec prefill
   - all target text tokens plus TTS_EOS paired with codec PAD
   - final `tts_pad + codec_bos`
   - later codec frames add only `tts_pad_embed`

6. Rust talker now has:
   - KV cache
   - interleaved RoPE
   - talker/code predictor `rope_theta=1000000`
   - tokenizer pre-transformer `rope_theta=10000`
   - talker RMSNorm eps `1e-6`

7. Rust `predict_codes` differs from upstream:
   - q0 is sampled from `codec_head(last_hidden)`.
   - q1..q15 now use a cached code-predictor path:
     - prefill `[past_hidden, q0_hidden]` with a causal mask
     - generate q1 from `lm_head[0]`
     - feed one residual code embedding per step through the code-predictor
       cache
   - This is closer to upstream `code_predictor.generate(...)`, but still needs
     logits-level trace comparison.

8. Rust speech tokenizer decode:
   - Reorganizes `[frame][level]` into 16 code tensors `[B, T]`.
   - `SplitRVQ.decode`
   - pre-conv
   - pre-transformer
   - upsample stages
   - tokenizer decoder conv stack
   - WAV post-processing and write

## Known Evidence

- Rust CUDA generation completes.
- 16 frames decode to the correct duration:
  - 16 frames -> 30720 samples -> 1.280s at 24 kHz
- Latest sample:
  - `out/qwen3_cp_cache_16frames.wav`
  - code range `[1, 2040]`
  - unique q0 values `11/16`
  - repeated consecutive frames `0`
- Audio is still not human speech.
- Metrics still look wrong:
  - RMS level about `-2.73 dB`
  - peak count about `15754`
  - crest factor about `1.23`
  - Non-streaming prompt alignment and cached residual generation did not fix
    the noise.

## Highest-Risk Divergence Points

1. Prompt embedding layout.
   - Official CustomVoice defaults to `non_streaming_mode=true`.
   - Rust has been changed to this layout, but the exact token IDs, embedding
     sequence length, and first hidden state still need numeric trace
     comparison.

2. Residual code predictor generation.
   - Official code predictor uses its own `generate(...)` state machine.
   - Rust now uses causal prefill plus KV cache and sampling for residual
     codes.
   - If q0 is correct but q1..q15 are wrong, the decoder will output noise.
   - This still needs logits-level trace comparison against upstream.

3. RoPE position ids and attention mask.
   - Official talker derives position ids from attention masks and supports
     left padding and cache position deltas.
   - Rust currently uses simple cache length offsets.
   - For single-sample non-padded prompts this may be close, but it is not yet
     proven numerically identical.

4. Speaker handling.
   - Rust uses hard-coded speaker token ids from config.
   - Need confirm whether CustomVoice path expects speaker token embedding
     from codec embedding, a speaker embedding vector, or either depending on
     model type.

5. Speech tokenizer decoder alignment.
   - If known-good upstream codec frames decode as noise in Rust, the bug is in
     the decoder path rather than the talker.

## Step-By-Step Debug Plan

### Step 1: Freeze a deterministic upstream trace

Use the same prompt, language, speaker, and max frames as Rust:

- text: `Hello from Qwen three TTS.`
- language: `english`
- speaker: `vivian`
- mode: CustomVoice
- `non_streaming_mode=true`
- `do_sample=false` first, to remove randomness
- `subtalker_dosample=false` first, to remove residual-code randomness
- max frames: 2 or 4

Export from Python:

- assistant prompt string
- token IDs
- language id
- speaker id or speaker embedding source
- codec prefill IDs
- talker input embedding length
- talker attention mask
- position ids / cache positions
- first prefill hidden state checksum
- first q0 logits top 20
- first generated q0
- first residual code predictor logits top 20 for q1
- first full frame `[q0..q15]`
- decoded waveform stats

### Step 2: Add Rust trace hooks

Add a debug flag that can dump:

- prompt token IDs
- codec prefill IDs
- input hidden shape
- trailing text hidden shape
- first hidden checksum
- q0 logits top 20
- residual logits top 20 per code group
- generated frames
- decoder stage shapes and simple checksums

Use machine-readable JSON or JSONL so the Python/Rust traces can be diffed.

### Step 3: Match prompt construction first

Before touching attention math, compare:

- prompt string bytes
- token IDs
- codec prefill IDs
- speaker selection
- talker input embedding sequence layout

Expected first code change if mismatch is confirmed:

- Fix the Rust `non_streaming_mode` layout until token IDs, sequence length,
  and q0 logits match upstream.

### Step 4: Match the first q0 logits

If prompt construction matches but q0 logits differ:

- Compare text projection output.
- Compare first transformer layer output.
- Compare RoPE position ids.
- Compare attention mask shape and values.
- Compare final norm and codec head output.

Stop at the first layer whose checksum diverges.

### Step 5: Match residual codes q1..q15

If q0 matches but the frame is wrong:

- Replace the manual Rust residual loop with a code-predictor generation path
  matching upstream:
  - prefill with `[past_hidden, last_id_hidden]`
  - maintain `generation_steps`
  - use the correct residual embedding table per step
  - apply subtalker sampling or deterministic settings exactly

### Step 6: Verify decoder independently

Feed known-good upstream frames directly into Rust:

- If Rust decoder output is human-like, focus on talker generation.
- If Rust decoder output is still noise, compare decoder stages:
  - `SplitRVQ.decode`
  - pre-conv
  - pre-transformer
  - upsample stages
  - tokenizer decoder conv stack

### Step 7: Restore sampling after deterministic alignment

After deterministic frames match:

- Re-enable upstream defaults:
  - `do_sample=true`
  - `temperature=0.9`
  - `top_k=50`
  - `top_p=1.0`
  - `repetition_penalty=1.05`
  - `subtalker_dosample=true`
  - `subtalker_temperature=0.9`

Then run a longer phrase and listen.

## Immediate Next Coding Task

Implement the trace harness before changing more model math:

1. Add Rust JSON trace output for prompt/prelude/q0/residual/frames.
2. Add a small Python upstream trace script under `tools/` or `scripts/`.
3. Run both with deterministic generation settings.
4. Compare and fix the earliest mismatch.

The most likely first functional fix is adding upstream-compatible
`non_streaming_mode=true` prompt embedding layout to Rust CustomVoice
generation.
