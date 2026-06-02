# Qwen3-TTS Rust Rewrite — Technical Specification

> **Version**: 1.0.0  
> **Created**: 2026-06  
> **Status**: Draft  
> **Scope**: Codec Decode module — Python/PyTorch → pure Rust port

---

## 1. Project Overview

Port the **Qwen3-TTS Codec Decode** module from Python/PyTorch to pure Rust. The rewritten module must support dual-mode decoding:

- **12 Hz** — real-time interactive (causal convolution, ≤97 ms first packet)
- **25 Hz** — high-quality synthesis (block-wise flow matching DiT, ≤300 ms first packet)

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

### 3.1 Dual-Mode Tokenizer Decoding

| Mode | Tokenizer | Decoder Architecture | Latency Target | Key Characteristics |
|---|---|---|---|---|
| Real-time Interactive | 12 Hz | Causal ConvNet + MTP | First packet ≤ 97 ms | 16-layer independent codebook parallel, zero look-ahead |
| High-quality Synthesis | 25 Hz | Block-wise Flow Matching DiT | First packet ≤ 300 ms | Single-stage token prediction, chunked diffusion |

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

---

## 5. Acceptance Criteria

1. **12 Hz mode** — continuous 10-second audio generation, first-packet latency **P99 ≤ 97 ms**.
2. **25 Hz mode** — generated audio PESQ score deviation from Python reference **≤ 0.05**.
3. **Regression suite** — 1000 variable-length text end-to-end tests: **zero crashes, zero silent segments**.
4. **Test suite** — `cargo test --release` passes 100%, including numerical-alignment unit tests.

---

## 6. Architecture Overview (Preliminary)

```
┌─────────────────────────────────────────────────────────┐
│                    TtsDecoder (trait)                    │
├──────────────────────┬──────────────────────────────────┤
│   Decoder12Hz        │        Decoder25Hz               │
│   ┌──────────────┐   │   ┌────────────────────────┐     │
│   │ CodebookEmb  │   │   │ FlowMatchingDiT        │     │
│   │ (16 layers)  │   │   │ (block-wise diffusion) │     │
│   │ rayon par    │   │   └────────────────────────┘     │
│   └──────────────┘   │   ┌────────────────────────┐     │
│   ┌──────────────┐   │   │ Chunk Scheduler        │     │
│   │ CausalConv   │   │   └────────────────────────┘     │
│   │ (ring buf)   │   │                                  │
│   └──────────────┘   │                                  │
│   ┌──────────────┐   │                                  │
│   │ MTP Head     │   │                                  │
│   └──────────────┘   │                                  │
├──────────────────────┴──────────────────────────────────┤
│              Vocoder (HiFi-GAN / Vocos)                  │
├─────────────────────────────────────────────────────────┤
│              Audio Output (PCM f32 / WAV)                │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Module Breakdown (Proposed)

| Module | Responsibility | Key Crate |
|---|---|---|
| `codebook` | Embedding lookup, 16-layer parallel decode | `rayon`, `candle` |
| `causal_conv` | Streaming causal convolution with ring buffer | `candle` |
| `mtp` | Multi-token prediction head | `candle` |
| `flow_matching` | Block-wise flow matching DiT (25 Hz) | `candle` |
| `vocoder` | Waveform synthesis from mel/codec tokens | `candle` |
| `tokenizer` | Load `tokenizer.json`, encode text → tokens | `serde_json` |
| `stream` | Async chunk scheduling, tokio channels | `tokio` |
| `weights` | SafeTensors weight loading & validation | `safetensors-rs` |
| `audio` | WAV/PCM encode-decode | `hound`, `symphonia` |

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

## 9. Milestones

| Phase | Deliverable | Target |
|---|---|---|
| **Phase 0** | Project scaffold, weight loader, tokenizer | Week 1 |
| **Phase 1** | 12 Hz codebook + causal conv (CPU) | Week 2–3 |
| **Phase 2** | 12 Hz GPU (CUDA), numerical alignment verified | Week 4 |
| **Phase 3** | 25 Hz flow matching DiT | Week 5–6 |
| **Phase 4** | Vocoder integration, streaming pipeline | Week 7 |
| **Phase 5** | Regression suite, perf benchmarks, Metal port | Week 8 |

---

## 10. Open Questions

- [ ] Which vocoder does Qwen3-TTS use? (HiFi-GAN, Vocos, or custom?)
- [ ] Are the 16 codebook layers independent or have cross-layer attention?
- [ ] What is the exact MTP (multi-token prediction) architecture?
- [ ] Is there an official SafeTensors weight file available, or must we convert from PyTorch `.pt`?
- [ ] What sample rate does the vocoder output? (24 kHz assumed)
