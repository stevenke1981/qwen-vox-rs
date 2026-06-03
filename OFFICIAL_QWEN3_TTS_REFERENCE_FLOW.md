# Official Qwen3-TTS Reference Flow

Goal: keep a verified upstream path that produces audible human speech, then
rewrite the Rust path to match it step by step.

Verified reference output:

- `out/official_qwen3_reference.wav`
- Text: `你好，這是官方 Qwen3 TTS 參考語音。`
- Language: `chinese`
- Speaker: `vivian`
- Output sample rate: 24000 Hz
- Generated codec frames: 50 frames, 16 codebooks each
- Duration: about 3.976875 seconds
- Audio check: audible human voice, 0 clipped samples

Important distinction:

- `out/python_decoder_test_output.wav` is not a speech reference. It is only a
  decoder-only numerical reference for fixed test tokens.
- `out/official_qwen3_reference.wav` is the real end-to-end speech reference.

## Upstream Files

Official source files used for the reference run:

- `qwen_tts/inference/qwen3_tts_model.py`
- `qwen_tts/inference/qwen3_tts_tokenizer.py`
- `qwen_tts/core/models/modeling_qwen3_tts.py`
- `qwen_tts/core/models/configuration_qwen3_tts.py`
- `qwen_tts/core/models/processing_qwen3_tts.py`
- `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
- `qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py`

Local weights used by the reference run:

- Main model: `weights/hf_original/model.safetensors`
- Main config: `weights/hf_original/config.json`
- Generation config: `weights/hf_original/generation_config.json`
- Text tokenizer files: `weights/hf_original/vocab.json`, `weights/hf_original/merges.txt`
- Speech tokenizer: `weights/hf_original/speech_tokenizer/model.safetensors`

## Official End-To-End Flow

### 1. Load model and tokenizer

The official model class is:

- `Qwen3TTSForConditionalGeneration`

The official speech tokenizer model for this repo is:

- `Qwen3TTSTokenizerV2Model`
- `model_type = qwen3_tts_tokenizer_12hz`

The model loader attaches the speech tokenizer during
`Qwen3TTSForConditionalGeneration.from_pretrained(...)`.

For the local reference script, the official Python classes are loaded from
`out/` and registered with Hugging Face `AutoConfig` / `AutoModel`. This avoids
network downloads and uses the local `weights/hf_original` directory.

### 2. Build the CustomVoice assistant prompt

Official wrapper prompt format:

```text
<|im_start|>assistant
{text}<|im_end|>
<|im_start|>assistant
```

For the verified reference:

```text
<|im_start|>assistant
你好，這是官方 Qwen3 TTS 參考語音。<|im_end|>
<|im_start|>assistant
```

The prompt is tokenized by `Qwen3TTSProcessor`, which wraps the local Qwen text
tokenizer.

### 3. Apply generation defaults

The upstream generation defaults are read from:

```text
weights/hf_original/generation_config.json
```

The verified reference run used:

- `do_sample = true`
- `top_k = 50`
- `top_p = 1.0`
- `temperature = 0.9`
- `repetition_penalty = 1.05`
- `subtalker_dosample = true`
- `subtalker_top_k = 50`
- `subtalker_top_p = 1.0`
- `subtalker_temperature = 0.9`
- `max_new_tokens = 64`

The official wrapper default for CustomVoice is:

- `non_streaming_mode = true`

### 4. Call official model.generate

The CustomVoice wrapper calls:

```python
talker_codes_list, _ = model.generate(
    input_ids=input_ids,
    instruct_ids=instruct_ids,
    languages=languages,
    speakers=speakers,
    non_streaming_mode=True,
    **generation_kwargs,
)
```

For the verified reference:

- `input_ids = [tokenized assistant prompt]`
- `instruct_ids = [None]`
- `languages = ["chinese"]`
- `speakers = ["vivian"]`

The output is a list of codec tensors:

```text
talker_codes_list[0].shape == (50, 16)
```

Shape meaning:

- axis 0: codec frames
- axis 1: 16 RVQ/codebook levels
- q0 is the semantic/primary code
- q1..q15 are residual codec codes

### 5. Official prompt embedding inside model.generate

For each sample, upstream builds talker input embeddings as follows.

Text-side special embeddings:

- `tts_bos_embed = text_projection(text_embedding(tts_bos_token_id))`
- `tts_eos_embed = text_projection(text_embedding(tts_eos_token_id))`
- `tts_pad_embed = text_projection(text_embedding(tts_pad_token_id))`

Codec prefill for a fixed language:

```text
[codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
```

For CustomVoice speaker mode, the speaker embedding/token is inserted before:

```text
[codec_pad_id, codec_bos_id]
```

Then the official non-streaming layout is:

1. role prefix: text projection of `input_id[:, :3]`
2. codec/text prefill:
   - `tts_pad_embed` repeated for codec-prefix positions
   - `tts_bos_embed`
   - added to codec embeddings except final codec BOS
3. all target text tokens plus `tts_eos_embed`, each added to codec PAD
4. final `tts_pad_embed + codec_bos_embedding`
5. `trailing_text_hidden = tts_pad_embed`

### 6. Main talker generation

The official talker uses Hugging Face `GenerationMixin`.

It passes:

- `inputs_embeds`
- `attention_mask`
- `trailing_text_hidden`
- `tts_pad_embed`

The model builds causal masks and position IDs from the attention mask and cache
position. KV cache is updated during generation.

At each generated frame:

1. q0 is generated from the main talker `codec_head`.
2. The code predictor/subtalker generates q1..q15.
3. q0 and residual code embeddings are summed into the next frame embedding.
4. `trailing_text_hidden[generation_step]` is added when available; otherwise
   `tts_pad_embed` is added.
5. Generation stops when q0 equals `codec_eos_token_id`.

### 7. Speech tokenizer decode

The official wrapper decodes generated codec frames with:

```python
wavs, fs = model.speech_tokenizer.decode(
    [{"audio_codes": c} for c in talker_codes_list]
)
```

For the 12Hz tokenizer:

- input code shape per sample: `(frames, 16)`
- decode output: list of float32 mono waveforms
- sample rate: 24000 Hz

### 8. Save WAV

The verified reference saves the first waveform:

```python
soundfile.write("out/official_qwen3_reference.wav", wav, 24000)
```

Observed stats:

```text
samples  = 95445
duration = 3.976875 seconds
min      = -0.474609375
max      = 0.60546875
rms_db   = about -20.35 dB
clip     = 0
```

## Rust Rewrite Targets

The Rust implementation should match the official flow in this order:

1. Prompt/tokenization parity
   - Same assistant prompt string.
   - Same token IDs for the same text.

2. Codec prefill parity
   - Same language ID.
   - Same speaker handling for `vivian`.
   - Same `[codec_think..., speaker, codec_pad, codec_bos]` sequence.

3. Non-streaming embedding parity
   - Same role prefix.
   - Same `tts_pad/tts_bos/tts_eos` embedding layout.
   - Same `trailing_text_hidden = tts_pad_embed`.

4. Talker generation parity
   - Same causal attention mask.
   - Same position IDs/cache positions.
   - Same q0 logits for the first frame.
   - Same q1..q15 residual code predictor logits.

5. Codec frame parity
   - Rust should be able to reproduce or closely match official codec frames
     when using deterministic generation settings.
   - With sampling enabled, compare distributions and audio quality rather
     than exact frame equality.

6. Decoder parity
   - Known-good official codec frames must decode cleanly in Rust.
   - Decoder-only test tokens are not enough for speech validation.

7. End-to-end acceptance
   - Compiled Rust CLI produces a WAV that is audibly human speech.
   - No clipping.
   - Duration and frame count are plausible.

## Reusable Reference Script

Use:

```powershell
python tools\generate_official_reference.py --text "你好，這是官方 Qwen3 TTS 參考語音。" --language chinese --speaker vivian --max-new-tokens 64 --output out\official_qwen3_reference.wav --codes-output out\official_qwen3_reference_codes.npy
```

The first run is slow because it loads the full upstream Python model and
speech tokenizer. It is a reference/debug tool, not the runtime target.
