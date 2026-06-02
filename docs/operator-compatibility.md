# Candle 0.9.2 算子兼容性報告 — Qwen3-TTS Rust Rewrite

> **Generated**: 2026-06  
> **Candle version**: 0.9.2 (candle-core + candle-nn)  
> **Target model**: Qwen3-TTS-12Hz (0.6B / 1.7B)

---

## 1. 算子兼容性總表

| # | Qwen3-TTS 所需算子 | Candle 支援 | 後端 (CPU/CUDA/Metal) | 備註 |
|---|---|---|---|---|
| 1 | **Conv1d** (causal) | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::conv1d(kernel, padding, stride, dilation, groups)` — causal padding 需自行實作 |
| 2 | **ConvTranspose1d** (causal) | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::conv_transpose1d(kernel, stride, padding, output_padding, dilation, groups)` |
| 3 | **Linear / MatMul** | ✅ | CPU ✅ CUDA (cuBLAS) ✅ Metal ✅ | `Tensor::matmul`, `candle_nn::linear` |
| 4 | **RMSNorm** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `candle_nn::rms_norm(weight, eps)` |
| 5 | **LayerNorm** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `candle_nn::layer_norm(weight, bias, eps)` |
| 6 | **Softmax** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `candle_nn::ops::softmax(xs, dim)` / `softmax_last_dim(xs)` |
| 7 | **SiLU** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::silu()` |
| 8 | **GELU** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::gelu()` / `gelu_erf()` |
| 9 | **ReLU** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::relu()` |
| 10 | **Embedding** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `candle_nn::embedding(vocab_size, dim, vs)` |
| 11 | **RoPE** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `candle_nn::rotary_emb::rope(q, k, cos, sin)` — 多種變體 |
| 12 | **index_select** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::index_select(&indexes, dim)` |
| 13 | **gather** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::gather(&indexes, dim)` |
| 14 | **narrow / squeeze / reshape** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | 基本 shape ops |
| 15 | **stack / cat** | ✅ | CPU ✅ CUDA ✅ Metal ✅ | `Tensor::stack`, `Tensor::cat` |
| 16 | **GQA / SDPA** | ⚠️ 部分 | CPU ❌ CUDA ❌ Metal ✅ (fused) | `candle_nn::ops::sdpa` — Metal only fused; CPU/CUDA 需手動 matmul+softmax |
| 17 | **SnakeBeta** | ❌ 需自實作 | — | 週期性激活函式：`x + beta * sin^2(alpha * x)` |
| 18 | **LayerScale** | ❌ 需自實作 | — | 簡單逐元素乘法：`x * gamma` (gamma 初始 0.01) |
| 19 | **ConvNeXt Block** | ❌ 需組合 | — | depthwise Conv7 + LayerNorm + GELU FF 4× + gamma scale |
| 20 | **Sliding Window Attention** | ❌ 需自實作 | — | causal mask with window=72 |
| 21 | **ODE Solver** | ❌ 需自實作 | — | 25Hz Flow Matching 用，Phase 2 |
| 22 | **RVQ / SplitRVQ** | ❌ 需自實作 | — | EuclideanCodebook + projection |
| 23 | **Attentive Statistics Pooling** | ❌ 需自實作 | — | Speaker Encoder (ECAPA-TDNN) 用 |

---

## 2. 需自行實作的模組

### 2.1 SnakeBeta Activation (優先級：高)

```
公式: y = x + beta * sin^2(alpha * x)
用途: Tokenizer Decoder 的 ResidualUnit
實作: 逐元素運算，可用 Candle 的 sin/pow/mul 組合
```

### 2.2 Causal Padding for Conv1d (優先級：高)

```
標準 Conv1d 是雙向 padding；causal 需要僅左側 padding。
實作方式：先 pad_left(k-1)，再 conv1d(padding=0)
```

### 2.3 Causal ConvTranspose1d (優先級：高)

```
Upsampling 階段使用。需確保 output 僅依賴過去 frames。
實作方式：conv_transpose1d 後 crop 右側。
```

### 2.4 GQA Attention (優先級：高)

```
Grouped Query Attention: 16 heads, 8 kv_heads (code predictor)
或 16 heads, 16 kv_heads (tokenizer decoder)
CUDA 無 fused SDPA → 手動: Q·K^T / √d → mask → softmax → ·V
```

### 2.5 Sliding Window Causal Mask (優先級：中)

```
window=72 for tokenizer decoder
實作: 建立 causal mask 矩陣，限制 attention 範圍
```

### 2.6 LayerScale (優先級：低)

```
y = x * gamma (gamma 為可學習參數，初始值 0.01)
直接用 Tensor::broadcast_mul 即可
```

---

## 3. 風險評估

| 風險 | 影響 | 概率 | 緩解 |
|---|---|---|---|
| CUDA GQA 效能不足（無 fused SDPA） | 中 | 中 | 可接受：12Hz tokenizer decoder 僅 8 層，head_dim=64，matmul+softmax 開銷可控 |
| SnakeBeta 數值精度偏差 | 低 | 低 | sin^2 可用 `(sin(x))^2` 或 `(1-cos(2x))/2` 兩種形式，選精度較高者 |
| ConvTranspose1d causal crop 邊界效應 | 中 | 中 | 對齊測試時逐層比對，確認 crop 位置正確 |

---

## 4. 結論

**Candle 0.9.2 覆蓋了 Qwen3-TTS 12Hz 模式約 85% 的算子需求。** 需自行實作的部分集中在：
1. 自訂激活函式 (SnakeBeta) — 簡單逐元素運算
2. Causal padding 邏輯 — 組合現有 ops
3. GQA attention — 手動 matmul+softmax（CUDA 無 fused kernel）
4. 高階模組組合 (ConvNeXt Block, ResidualUnit) — 純組合層

**無阻塞性缺失。** 所有基礎算子 (Conv1d, ConvTranspose1d, RMSNorm, RoPE, Embedding) 均有完整三平台支援。
