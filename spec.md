# Qwen3-TTS Rust Rewrite — Technical Specification

> **Version**: 1.1.0  
> **Created**: 2026-06  
> **Updated**: 2026-06-03 (official HF verification)  
> **Status**: Draft  
> **Scope**: Codec Decode module for **Qwen3-TTS-Tokenizer-12Hz** (official name; actual rate 12.5 Hz) — Python/PyTorch → pure Rust port using Candle.

**Official Reference** (verified via Hugging Face):
- Model: `Qwen/Qwen3-TTS-Tokenizer-12Hz`
- Paper: Qwen3-TTS Technical Report (arXiv 2601.15621)
- Key facts:
  - Marketed as "12Hz" / "Qwen3-TTS-Tokenizer-12Hz".
  - **Actual operating rate: 12.5 Hz** (80 ms per frame, 1920 samples @ 24 kHz).
  - 16-layer multi-codebook (1 semantic + 15 acoustic RVQ).
  - Decoder: lightweight **causal ConvNet** (no DiT for this tokenizer; DiT/flow-matching is for the separate 25 Hz tokenizer).
  - Enables ultra-low first-packet latency (~97 ms end-to-end in full system).

**Note on naming confusion**: Some internal docs (e.g. copied from other ports) refer to "Decoder12Hz". This is the **same component** as the official 12Hz tokenizer's causal decoder. The "12" vs "12.5" difference is marketing vs precise rate; the current implementation correctly uses 12.5 Hz (see `TOKENIZER_FRAME_RATE_HZ`).

---

## 1. Project Overview

Port the **Qwen3-TTS-Tokenizer-12Hz Codec Decoder** (the causal ConvNet path) from Python/PyTorch to pure Rust (Candle).

This is the low-latency streaming tokenizer (distinct from the 25 Hz tokenizer which uses block-wise DiT + flow matching for higher quality).

Target latency (full system context): first packet ≤ 97 ms.

Core engineering challenges:
1. Multi-codebook parallel processing (16 quantizer layers)
2. Streaming causal convolution inference with ring-buffer state
3. Low-latency first-packet output

---

## 2. Technology Stack Constraints

| Category | Choice | Notes |
|---|---|---|
| Language | Rust (Edition 2021+) | |
| Tensor backend | **Candle** | CUDA / Metal GPU preferred; CPU fallback |
| Concurrency | **rayon** (multi-codebook parallel) + **tokio** (async streaming I/O) | |
| Serialization | **serde** / **safetensors-rs** | Weight loading |
| Audio I/O | **hound** / **symphonia** | WAV / PCM encode-decode |

### Forbidden Dependencies

- ❌ libtorch
- ❌ PyTorch
- ❌ Python FFI
- ❌ ONNX Runtime

---

## 3. Core Functional Specifications

### 3.1 Tokenizer Decoding (Qwen3-TTS-Tokenizer-12Hz focus)

This project targets the **official Qwen3-TTS-Tokenizer-12Hz** codec decoder (causal ConvNet path).

| Tokenizer (Official) | Marketing Name | Actual Rate | Decoder Architecture (this port) | Notes |
|---|----|----|----|----|
| Qwen3-TTS-Tokenizer-12Hz | 12Hz | **12.5 Hz** (80 ms/frame) | RVQ (16 codebooks) → pre_conv → 8-layer pre_transformer (GQA) → 2× upsample → 4-block causal SEANet-style conv decoder (dilations 1,3,9) + final Snake+Conv | Lightweight causal ConvNet, full left-context streaming, 1920× total upsample to 24 kHz |
| Qwen3-TTS-Tokenizer-25Hz | 25Hz | 25 Hz | Single-codebook + block-wise DiT + flow matching (BigVGAN vocoder) | **Not in current scope** of this codec-decoder port |

The old "dual Decoder12Hz / Decoder25Hz" view in earlier drafts was a misunderstanding from mixing ports. The two tokenizers are separate models on HF; their decoders are architecturally different. This spec and the `CodecDecoder` / `pipeline.rs` implement the 12Hz causal path.

### 3.2 Multi-Codebook Processing (12 Hz Mode)

#### Memory Layout

Flat 1-D contiguous memory: `Vec<u16>`

```
index = layer * num_frames + frame
```

#### Parallel Strategy

All 16 quantizer-layer embedding lookups **must** execute via `rayon::par_iter`.

#### Lookup Optimization

- Preload the full 2048-entry codebook weight matrix.
- Access via raw pointer offset — **no conditional branches**.

#### State Management

- Causal convolution hidden state uses a **fixed-size ring buffer**.
- **No runtime dynamic reallocation** permitted.

### 3.3 Streaming Inference Interface

```rust
pub trait TtsDecoder: Send + Sync {
    /// Initialize decoder — load weights and vocoder.
    fn new(config: DecoderConfig) -> Result<Self>;

    /// Stream tokens in, return PCM audio chunk.
    fn decode_chunk(&mut self, tokens: &[u16]) -> Result<Vec<f32>>;

    /// Reset internal state (session switch).
    fn reset_state(&mut self);
}
```

### 3.4 Precision & Robustness

| Requirement | Target |
|---|---|
| Numerical alignment | Layer-wise intermediate tensor cosine similarity ≥ **0.999** vs PyTorch reference |
| Fault tolerance | Single-layer codebook decode failure → auto-fill with upper-layer mean; **no inference interruption** |
| Tokenizer config | Must load pre-tokenizer rules from official `tokenizer.json` — **no hardcoding** |

---

## 4. Non-Functional Requirements

| Metric | Target |
|---|---|
| Binary size (GPU) | ≤ 50 MB |
| Binary size (CPU) | ≤ 30 MB |
| Cold start (excl. weight load) | ≤ 200 ms |
| Peak VRAM (12 Hz) | ≤ 1.5 GB |
| Peak RAM (12 Hz) | ≤ 3 GB |

### Platform Support

| Platform | Architecture | GPU Backend |
|---|---|---|
| Linux | x86_64 | CUDA |
| macOS | ARM64 | Metal |
| Windows | x86_64 | CUDA / CPU |

**Current focus**: 12Hz causal ConvNet decoder (the path used by Qwen3-TTS-12Hz models for low-latency streaming). The 25 Hz DiT path is a separate tokenizer and is not the target of the current `CodecDecoder` implementation.

---

## 5. Acceptance Criteria (for 12Hz Tokenizer Decoder)

1. **Qwen3-TTS-Tokenizer-12Hz (12.5 Hz) causal decode** — 8-frame batch decode produces correct-length 24 kHz waveform (1920 samples/frame). Layer-wise cosine similarity vs PyTorch reference **≥ 0.999** on key stages (RVQ out, pre_conv, pre_transformer, upsample, decoder blocks, final).
2. **Causality & streaming readiness** — all convolutions are strictly causal; state can be maintained across frames without future leakage.
3. **Numerical robustness** — no NaN/Inf on real code sequences; output clamped to [-1, 1]; non-silent, reasonable RMS on valid input.
4. **Regression** — `cargo test --release` (including alignment_verification and audio_output_verification) passes 100%.
5. **Weight compatibility** — loads from both full `speech_tokenizer/model.safetensors` (hf_original / converted) and extracted `alignments/tokenizer_decoder.safetensors`.

(25 Hz / DiT / flow-matching acceptance is out of scope for this spec until a separate tokenizer port is started.)

---

## 6. Architecture Overview — Qwen3-TTS-Tokenizer-12Hz Codec Decoder (Current Implementation)

The 12Hz tokenizer (official `Qwen/Qwen3-TTS-Tokenizer-12Hz`) uses a **16-codebook (1+15 RVQ) causal ConvNet decoder**. This is what `CodecDecoder` in `pipeline.rs` implements.

```
Input: 16 code tensors [B, T]  (u32/i64, 0-based into 2048-entry codebooks)
          │
          ▼
┌────────────────────────────────────────────────────────────┐
│ 1. SplitResidualVectorQuantizer (RVQ decode)               │
│    - rvq_first (semantic, 1 layer) + rvq_rest (acoustic,15)│
│    - codebook lookup + residual sum → [B, 512, T]          │
└────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────┐
│ 2. pre_conv (CausalConv1d k=3, 512→1024)                   │
└────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────┐
│ 3. pre_transformer (8 layers GQA + SwiGLU, causal mask)    │
│    [B, 1024, T] → transpose → [B, T, 1024] → blocks → ...  │
└────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────┐
│ 4. UpsampleStage × 2 (ConvTranspose stride-2 + ConvNeXt)   │
│    → [B, 1024, T×4]                                        │
└────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────┐
│ 5. TokenizerDecoder (post-transformer conv stack)          │
│    - decoder.0.conv (pre-proj)                             │
│    - 4× DecoderBlock (SnakeBeta + CausalTransConv +        │
│      3× ResidualUnit dil=1,3,9)  upsample rates [8,5,4,3]  │
│    - final SnakeBeta + Conv1d → [B, 1, T×1920]             │
│    - clamp [-1, 1]                                         │
└────────────────────────────────────────────────────────────┘
          │
          ▼
Output: waveform [B, 1, samples] f32 @ 24 kHz
(1920 samples per input frame → exactly 12.5 Hz)
```

**Key properties (official + current code)**:
- Total upsample: 4 (from the two UpsampleStages) × 480 (8×5×4×3 from decoder blocks) = 1920×.
- All convs are strictly causal (left-pad only for encode-side, right-crop only for transposed convs).
- SnakeBeta: `y = x + sin²(α·x) / (β + 1e-9)` with `α=exp(α_log)`, `β=exp(β_log)`.
- No DiT / flow-matching in this path (those belong to the separate 25 Hz tokenizer).

(The old simplistic "CodebookEmb + CausalConv + MTP" diagram was from a different port and does not match the actual Qwen3-TTS-12Hz decoder stack.)

---

## 7. Module Breakdown (Qwen3-TTS-Tokenizer-12Hz Codec Decoder)

| Module (current crate) | Responsibility | Notes |
|---|----|----|
| `quantizer` (RVQ + EuclideanCodebook) | 16-layer codebook lookup + residual sum (semantic + 15 acoustic) | Uses normalized `embedding_sum / cluster_usage` |
| `causal_conv` | `CausalConv1dLayer`, `CausalConvTranspose1dLayer` (left-pad, right-crop only) | Strict causality |
| `conv_decoder` | `ResidualUnit` (dil 1/3/9), `DecoderBlock`, `TokenizerDecoder` (full SEANet-style stack) | SnakeBeta, upsample [8,5,4,3] |
| `pipeline` (`CodecDecoder`) | Full end-to-end: quantizer → pre_conv → pre_transformer (8L) → upsample×2 → tokenizer_decoder | The main "12Hz causal decoder" entry point |
| `transformer` | GQA pre_transformer blocks (causal mask, RoPE, SwiGLU, LayerScale) | 8 layers for the 12Hz path |
| `weights` / `ComponentWeights` | SafeTensors loading with "decoder." prefix scoping | Supports both full speech_tokenizer and extracted decoder-only safetensors |
| `custom_ops` | `snake_beta`, `causal_pad_left`, `causal_crop_right`, `causal_mask` | |
| `alignment` (tests) | Cosine similarity helpers for PyTorch reference verification | Target ≥ 0.999 layer-wise |

**Out of scope for this codec-decoder port** (belong to the separate 25 Hz tokenizer or full LM):
- Flow-matching DiT / block-wise diffusion
- BigVGAN vocoder
- The 25 Hz single-codebook path

(The old "MTP Head" and simple "CodebookEmb" modules listed in earlier drafts do not apply to the decoder side of the 12Hz tokenizer; MTP lives in the talker / code-predictor for token *generation*.)

---

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Candle lacks required ops (e.g., grouped conv1d) | Medium | High | Implement custom CUDA kernel via `candle-core` backend extension |
| Numerical drift exceeds 0.999 cosine threshold | Medium | High | Layer-by-layer comparison harness against PyTorch reference tensors |
| Ring buffer state corruption under concurrent access | Low | Critical | Enforce `Send + Sync` bounds; use `Arc<Mutex<_>>` only for state reset path |
| Binary size exceeds 50 MB with CUDA | Medium | Medium | Strip symbols (`strip = true`), LTO, and conditional feature gates |
| Metal backend maturity on macOS | Medium | Medium | CPU fallback path; defer Metal optimization to Phase 2 |

---

## 9. Milestones (12Hz Tokenizer Decoder focus)

| Phase | Deliverable | Target |
|---|----|----|
| **Phase 0** | Scaffold, WeightStore, ComponentWeights, basic quantizer | Done |
| **Phase 1** | Full `CodecDecoder` (RVQ + pre_conv + pre_transformer + upsample + TokenizerDecoder) on CPU | Done (basic forward works) |
| **Phase 2** | Causal correctness, SnakeBeta, transposed-conv right-crop, dilations 1/3/9 verified in unit tests | In progress |
| **Phase 3** | Numerical alignment harness using `weights/intermediates/` + `weights/alignments/tokenizer_decoder.safetensors`; cosine ≥ 0.999 on real activations | Next priority |
| **Phase 4** | `decoder_test` + integration tests produce bit-identical or high-cosine matching audio vs Python reference on the same tokens/weights |  |
| **Phase 5** | Talker / code-predictor integration (if in scope), full end-to-end with real text frontend tokens |  |
| **Later** | 25 Hz (DiT) tokenizer decoder, Metal optimization, production streaming | Out of current 12Hz decoder scope |

---

## 10. Open Questions & Notes

**Resolved / Verified (2026-06-03 via official HF + paper)**:
- Tokenizer name: Qwen3-TTS-Tokenizer-12Hz (marketed "12Hz", operates at **12.5 Hz**).
- Decoder for this tokenizer: pure causal ConvNet (no DiT). The DiT path belongs to the separate 25 Hz tokenizer.
- Sample rate: 24 kHz. 1920 samples per codec frame.
- Weights: provided as SafeTensors (both full speech_tokenizer and extracted decoder subsets exist in this repo under `weights/`).

**Remaining (for full system, not just codec decoder)**:
- Exact talker + MTP architecture for token *generation* (this spec is decoder-only).
- Whether the current `hf_original/speech_tokenizer` vs `alignments/tokenizer_decoder.safetensors` + `test_input.safetensors` are bit-compatible for numerical alignment tests.
- Full end-to-end latency numbers on this Rust port (target inherited from paper: ~97 ms first packet in the complete system).

**Note on copied documentation**: Several files under `docs/` (e.g. `codec_decode_voice_fix_2026-06-02.md`) were brought in from other Qwen3-TTS Rust ports. They contain valuable implementation lessons (causal padding, SnakeBeta formula, right-only crop on transposed conv, dilation schedule, full-batch vs per-frame decode) but use class names ("Decoder12Hz") and file layouts from those ports. Treat them as reference, not literal spec for this codebase. The canonical architecture is the one in `crates/qwen-vox-core/src/pipeline.rs` + `conv_decoder.rs`.
