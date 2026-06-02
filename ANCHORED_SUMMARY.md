> 最後更新：2026-06-02 | 工作階段：qwen-vox-rs Candle 0.9 相容性修復 & pipeline 模組建置
> 追蹤範圍：crates/qwen-vox-core/src/ 核心模組

## 專案目標

Qwen3-TTS 語音合成模型的 Rust 推理實作，基於 Candle 0.9.2 ML 框架。

## 已完成的工作

### 1. Candle 0.8.4 → 0.9.2 API 相容性修復

| 問題 | 檔案 | 修復內容 |
|------|------|----------|
| `candle_nn::layer_norm` 不存在 | `pipeline.rs` | 改用 `candle_nn::LayerNorm` struct + `.forward()` |
| `Tensor::full(val, size, DType, device)` API 改變 | `pipeline.rs` | 移除 DType 參數 → `Tensor::full(val, size, device)` |
| `Tensor::randn(mean, std, shape, DType, device)` API 改變 | `pipeline.rs` | 移除 DType 參數 → `Tensor::randn(mean, std, shape, device)` |
| `conv_transpose1d` 參數順序錯誤 | `conv_decoder.rs:146` | `(weight, stride, 0, 0, 1, 1)` → `(weight, 0, 0, stride, 1, 1)` |
| `layer_scale` 對 3D `[B,C,T]` layout 無法 broadcast | `custom_ops.rs` | 新增 `layer_scale_3d(x, gamma, channel_dim)` |
| `DecoderBlock`/`TokenizerDecoder` fields 非 pub | `conv_decoder.rs` | 加上 `pub(crate)` 供內部測試存取 |

### 2. pipeline 模組 — 新增與驗證

- **`src/pipeline.rs`** — CodecDecoder 完整管線（quantizer → pre_conv → pre_transformer → upsample×2 → tokenizer_decoder）
- **`lib.rs`** — 加入 `pub mod pipeline;`
- ConvNeXtBlock, UpsampleStage, 完整的 shape propagation test

### 3. 測試狀態

**全部 52 tests passing**（36 unit + 16 integration）

#### Unit tests (36)
- `custom_ops`: 7 tests — causal pad/crop, snake_beta, layer_scale, masking
- `conv_decoder`: 6 tests — causal_conv1d, conv_transpose, residual_unit, decoder_block, tokenizer_decoder
- `pipeline`: 6 tests — gelu_erf, convnext_block, upsample_stage, quantizer_only, full_pipeline
- `transformer`: 5 tests — rms_norm, gqa, swiglu, transformer_block, stack
- `quantizer`: 3 tests — euclidean, rvq, code_predictor
- `tokenizer`: 2 tests — special_tokens, config_deserialize
- `weights`: 2 tests — component_weights, weight_store
- `alignment`: 6 tests — cosine, mean_abs, max_abs

#### Integration tests (16)
- `alignment_verification`: 4 tests
- `audio_output_verification`: 2 tests
- `code_predictor_verification`: 2 tests
- `talker_verification`: 3 tests
- `weight_loading`: 5 tests

### 4. 警告清理（已完成）

- ✅ `alignment_verification.rs:10` — 移除未使用的 `ComponentWeights` import
- ✅ `custom_ops.rs:68` — 將 `(d0, d1, d2)` 改為 `(_d0, _d1, _d2)`

**目前狀態：零警告、零錯誤**

## 模組架構

```
crates/qwen-vox-core/src/
├── lib.rs                  # Root
├── error.rs                # VoxError / VoxResult
├── weights.rs              # WeightStore / ComponentWeights
├── custom_ops.rs           # layer_scale, layer_scale_3d, causal_pad_left/right, snake_beta
├── transformer.rs          # RmsNorm, TransformerBlock, TransformerStack
├── conv_decoder.rs         # CausalConv1dLayer, CausalConvTranspose1dLayer,
│                           #   ResidualUnit, DecoderBlock, TokenizerDecoder
├── quantizer.rs            # ResidualVectorQuantizer, SplitResidualVectorQuantizer
├── alignment.rs            # 語音對齊
├── pipeline.rs             # ConvNeXtBlock, UpsampleStage, CodecDecoder
└── tokenizer.rs            # 文字 tokenizer
```

## Candle 0.9.2 API 重點筆記

```rust
// conv_transpose1d 參數順序（與 0.8 不同!）
x.conv_transpose1d(&weight, padding, output_padding, stride, dilation, groups)

// Tensor 建構（已移除 DType 參數）
Tensor::full(val, size, &device)
Tensor::randn(mean, std, shape, &device)

// LayerNorm（改用 struct API）
let ln = candle_nn::LayerNorm::new(weight, bias, eps);
h = ln.forward(&h)?;

// 3D layer_scale（需指定 channel 維度）
layer_scale_3d(&x, &gamma, 1)  // [B, C, T] layout
layer_scale_3d(&x, &gamma, 2)  // [B, T, C] layout
```

## 下一步可能的工作

- Speaker encoder 整合
- 端到端 TTS 推論管線
- 音頻輸出驗證
- 效能優化
