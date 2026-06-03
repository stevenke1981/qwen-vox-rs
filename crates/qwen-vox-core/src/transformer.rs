//! Transformer building blocks for the Qwen3-TTS decoder (Candle 0.9+).
//!
//! Implements the core components used by the pre_transformer (tokenizer decoder)
//! and potentially the main talker transformer / DiT:
//! - RmsNorm (pre-norm)
//! - SwiGLU (gated MLP)
//! - GroupedQueryAttention (with optional per-head Q/K RMSNorm)
//! - TransformerBlock (pre-norm + residual + optional LayerScale)
//! - TransformerStack (with optional input/output projections and final norm)
//!
//! All weight tensors follow PyTorch Linear convention: [out_features, in_features].
//! Use `matmul(&w.t()?)` for projections.
//!
//! Custom ops (layer_scale, grouped_query_attention) are imported from crate::custom_ops.

use candle_core::{DType, Error, Result, Tensor};

use crate::custom_ops::{grouped_query_attention, layer_scale_3d};
use crate::error::{VoxError, VoxResult};

/// Root-mean-square normalization.
///
/// Wraps a learnable weight (gamma) of shape `[hidden_size]`.
/// Forward: `x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight`
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Construct from a pre-loaded weight tensor and epsilon.
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Apply RMSNorm.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Prefer the optimized op from candle-nn (supports CPU/CUDA/Metal).
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

/// SwiGLU gated feed-forward network (used as MLP in transformer blocks).
///
/// Equivalent to: `down( silu(x @ gate.T) * (x @ up.T) )`
///
/// Weights are [out, in]:
/// - gate_proj / up_proj: [intermediate_size, hidden_size]
/// - down_proj: [hidden_size, intermediate_size]
#[derive(Debug, Clone)]
pub struct SwiGLU {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl SwiGLU {
    /// Construct from pre-loaded gate, up, down projection weights.
    pub fn from_weights(gate: Tensor, up: Tensor, down: Tensor) -> VoxResult<Self> {
        // Basic shape validation (allow candle errors to surface via From)
        let inter = gate.dim(0)?;
        let hidden = gate.dim(1)?;
        if up.dim(0)? != inter || up.dim(1)? != hidden {
            return Err(VoxError::ShapeMismatch {
                expected: vec![inter, hidden],
                actual: vec![up.dim(0)?, up.dim(1)?],
            });
        }
        if down.dim(0)? != hidden || down.dim(1)? != inter {
            return Err(VoxError::ShapeMismatch {
                expected: vec![hidden, inter],
                actual: vec![down.dim(0)?, down.dim(1)?],
            });
        }
        Ok(Self {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        })
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = x.broadcast_matmul(&self.gate_proj.t()?)?;
        let up = x.broadcast_matmul(&self.up_proj.t()?)?;
        let gate = gate.silu()?;
        let hidden = gate.mul(&up)?;
        hidden.broadcast_matmul(&self.down_proj.t()?)
    }
}

/// Grouped Query Attention (GQA) with optional Q/K per-head RMSNorm.
///
/// Projections:
/// - q_proj: [q_dim, hidden] where q_dim = num_heads * head_dim
/// - k_proj / v_proj: [kv_dim, hidden] where kv_dim = num_kv_heads * head_dim
/// - o_proj: [hidden, q_dim]
///
/// q_norm / k_norm (if present): [head_dim] — applied after reshape to heads.
#[derive(Debug, Clone)]
pub struct GroupedQueryAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    norm_eps: f64,
    rope_theta: Option<f64>,
}

/// Per-layer K/V cache for autoregressive decoding.
#[derive(Debug, Clone, Default)]
pub struct AttentionCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
}

impl AttentionCache {
    /// Number of cached key/value positions.
    pub fn seq_len(&self) -> Result<usize> {
        match &self.key {
            Some(key) => key.dim(2),
            None => Ok(0),
        }
    }

    fn append(&mut self, key: Tensor, value: Tensor) -> Result<(Tensor, Tensor)> {
        let key = if let Some(prev) = &self.key {
            Tensor::cat(&[prev, &key], 2)?
        } else {
            key
        };
        let value = if let Some(prev) = &self.value {
            Tensor::cat(&[prev, &value], 2)?
        } else {
            value
        };

        self.key = Some(key.clone());
        self.value = Some(value.clone());
        Ok((key, value))
    }
}

impl GroupedQueryAttention {
    /// Construct from pre-loaded projection weights and optional Q/K norms.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        o: Tensor,
        q_norm: Option<Tensor>,
        k_norm: Option<Tensor>,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> VoxResult<Self> {
        let q_dim = q.dim(0)?;
        let hidden = q.dim(1)?;

        if num_heads == 0 {
            return Err(VoxError::Other("num_heads must be > 0".to_string()));
        }
        let head_dim = q_dim / num_heads;
        if num_heads * head_dim != q_dim {
            return Err(VoxError::ShapeMismatch {
                expected: vec![num_heads * head_dim],
                actual: vec![q_dim],
            });
        }

        let kv_dim = k.dim(0)?;
        if num_kv_heads * head_dim != kv_dim {
            return Err(VoxError::ShapeMismatch {
                expected: vec![num_kv_heads * head_dim],
                actual: vec![kv_dim],
            });
        }

        // Validate other projections share the hidden / q_dim
        if k.dim(1)? != hidden || v.dim(1)? != hidden {
            return Err(VoxError::ShapeMismatch {
                expected: vec![hidden],
                actual: vec![k.dim(1)?],
            });
        }
        let o_out = o.dim(0)?;
        let o_in = o.dim(1)?;
        if o_out != hidden || o_in != q_dim {
            return Err(VoxError::ShapeMismatch {
                expected: vec![hidden, q_dim],
                actual: vec![o_out, o_in],
            });
        }

        // Validate optional per-head norms
        if let Some(ref qn) = q_norm {
            if qn.dim(0)? != head_dim {
                return Err(VoxError::ShapeMismatch {
                    expected: vec![head_dim],
                    actual: vec![qn.dim(0)?],
                });
            }
        }
        if let Some(ref kn) = k_norm {
            if kn.dim(0)? != head_dim {
                return Err(VoxError::ShapeMismatch {
                    expected: vec![head_dim],
                    actual: vec![kn.dim(0)?],
                });
            }
        }

        Ok(Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            norm_eps: 1e-5,
            rope_theta: None,
        })
    }

    /// Set the epsilon used by optional per-head Q/K RMSNorm.
    pub fn with_norm_eps(mut self, eps: f64) -> Self {
        self.norm_eps = eps;
        self
    }

    /// Enable interleaved rotary position embeddings for Q/K.
    pub fn with_rope_theta(mut self, theta: f64) -> Self {
        self.rope_theta = Some(theta);
        self
    }

    /// Forward pass with optional attention mask (shape [seq_len, seq_len]).
    ///
    /// Expects `x` of shape [batch, seq_len, hidden].
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass that appends projected K/V tensors into an autoregressive cache.
    ///
    /// The query length is the input length, while cached keys/values may be longer.
    /// This supports a full prompt prefill followed by one-token incremental steps.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        mut cache: Option<&mut AttentionCache>,
    ) -> Result<Tensor> {
        let (b, s, _hidden) = x.dims3()?;

        // Linear projections (use broadcast_matmul for 3D x @ 2D weight)
        let q = x.broadcast_matmul(&self.q_proj.t()?)?;
        let k = x.broadcast_matmul(&self.k_proj.t()?)?;
        let v = x.broadcast_matmul(&self.v_proj.t()?)?;

        // Reshape to [b, s, n_h, d] then transpose -> [b, n_h, s, d]
        // Make contiguous because transpose/reshape can produce non-contiguous strides
        // that later matmuls inside grouped_query_attention dislike.
        let q = q
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, s, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, s, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Optional per-head Q/K RMSNorm (weight shape [head_dim], broadcasts over b/nh/s)
        let q = if let Some(ref qn) = self.q_norm {
            candle_nn::ops::rms_norm(&q, qn, self.norm_eps as f32)?
        } else {
            q
        };
        let k = if let Some(ref kn) = self.k_norm {
            candle_nn::ops::rms_norm(&k, kn, self.norm_eps as f32)?
        } else {
            k
        };

        let position_offset = if let Some(ref cache) = cache {
            cache.seq_len()?
        } else {
            0
        };

        let (q, k) = if let Some(theta) = self.rope_theta {
            (
                apply_rotary_embedding(&q, position_offset, theta)?,
                apply_rotary_embedding(&k, position_offset, theta)?,
            )
        } else {
            (q, k)
        };

        let (k, v) = if let Some(cache) = cache.take() {
            cache.append(k, v)?
        } else {
            (k, v)
        };

        // Core GQA (handles KV head repetition internally)
        let attn = grouped_query_attention(&q, &k, &v, mask, self.num_heads, self.num_kv_heads)?;

        // Reshape back: [b, n_h, s, d] -> [b, s, n_h, d] -> [b, s, q_dim]
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;

        // Output projection
        attn.broadcast_matmul(&self.o_proj.t()?)
    }
}

/// Apply Qwen/HF half-split RoPE to a `[batch, heads, seq, head_dim]` tensor.
///
/// Official Qwen3-TTS uses `rotate_half(x) = cat(-x[..., half:], x[..., :half])`,
/// not adjacent even/odd pairing. Cached decoding rotates only the new
/// positions, using the prior cache length as offset.
fn apply_rotary_embedding(x: &Tensor, position_offset: usize, theta: f64) -> Result<Tensor> {
    let (_b, _h, s, d) = x.dims4()?;
    if d % 2 != 0 {
        return Err(Error::Msg(format!("RoPE head_dim must be even, got {d}")));
    }

    let half = d / 2;
    let device = x.device();
    let original_dtype = x.dtype();
    let x_work = if original_dtype == DType::F32 {
        x.clone()
    } else {
        x.to_dtype(DType::F32)?
    };

    let mut cos = Vec::with_capacity(s * half);
    let mut sin = Vec::with_capacity(s * half);
    for pos in position_offset..position_offset + s {
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f64) / d as f64);
            let angle = pos as f64 * inv_freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }

    let cos = Tensor::from_vec(cos, (s, half), device)?.reshape((1, 1, s, half))?;
    let sin = Tensor::from_vec(sin, (s, half), device)?.reshape((1, 1, s, half))?;

    let x1 = x_work.narrow(3, 0, half)?;
    let x2 = x_work.narrow(3, half, half)?;

    let x1_cos = x1.broadcast_mul(&cos)?;
    let x2_sin = x2.broadcast_mul(&sin)?;
    let out_first = x1_cos.broadcast_sub(&x2_sin)?;

    let x2_cos = x2.broadcast_mul(&cos)?;
    let x1_sin = x1.broadcast_mul(&sin)?;
    let out_second = x2_cos.broadcast_add(&x1_sin)?;

    let out = Tensor::cat(&[&out_first, &out_second], 3)?;

    if original_dtype == DType::F32 {
        Ok(out)
    } else {
        out.to_dtype(original_dtype)
    }
}

/// Single pre-norm transformer block with SwiGLU and optional LayerScale.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attn: GroupedQueryAttention,
    mlp: SwiGLU,
    ln1: RmsNorm,
    ln2: RmsNorm,
    attn_layer_scale: Option<Tensor>,
    mlp_layer_scale: Option<Tensor>,
}

impl TransformerBlock {
    /// Construct a block from its constituent weights.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        input_layernorm_weight: Tensor,
        post_attention_layernorm_weight: Tensor,
        attn_layer_scale: Option<Tensor>,
        mlp_layer_scale: Option<Tensor>,
        q_norm: Option<Tensor>,
        k_norm: Option<Tensor>,
        num_heads: usize,
        num_kv_heads: usize,
        eps: f64,
    ) -> VoxResult<Self> {
        let attn = GroupedQueryAttention::from_weights(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
        )?
        .with_norm_eps(eps);
        let mlp = SwiGLU::from_weights(gate_proj, up_proj, down_proj)?;
        let ln1 = RmsNorm::from_weight(input_layernorm_weight, eps);
        let ln2 = RmsNorm::from_weight(post_attention_layernorm_weight, eps);

        Ok(Self {
            attn,
            mlp,
            ln1,
            ln2,
            attn_layer_scale,
            mlp_layer_scale,
        })
    }

    /// Enable interleaved RoPE for this block's self-attention.
    pub fn with_rope_theta(mut self, theta: f64) -> Self {
        self.attn = self.attn.with_rope_theta(theta);
        self
    }

    /// Pre-norm block forward.
    ///
    /// `h = x + layer_scale( attn( ln1(x) ), attn_gamma )`  (if gamma present)
    /// `h = h + layer_scale( mlp( ln2(h) ), mlp_gamma )`     (if gamma present)
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_cache(x, mask, None)
    }

    /// Pre-norm block forward with optional attention KV cache.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut AttentionCache>,
    ) -> Result<Tensor> {
        // Attention sub-layer (pre-norm + residual + optional scale)
        let h = self.ln1.forward(x)?;
        let attn_out = self.attn.forward_with_cache(&h, mask, cache)?;
        let attn_out = if let Some(ref gamma) = self.attn_layer_scale {
            layer_scale_3d(&attn_out, gamma, 2)?
        } else {
            attn_out
        };
        let h = x.add(&attn_out)?;

        // MLP sub-layer (pre-norm + residual + optional scale)
        let h2 = self.ln2.forward(&h)?;
        let mlp_out = self.mlp.forward(&h2)?;
        let mlp_out = if let Some(ref gamma) = self.mlp_layer_scale {
            layer_scale_3d(&mlp_out, gamma, 2)?
        } else {
            mlp_out
        };
        h.add(&mlp_out)
    }
}

/// Full transformer stack: optional input proj -> N blocks -> optional norm -> optional output proj.
#[derive(Debug, Clone)]
pub struct TransformerStack {
    blocks: Vec<TransformerBlock>,
    norm: Option<RmsNorm>,
    input_proj: Option<(Tensor, Option<Tensor>)>, // (weight, bias)
    output_proj: Option<(Tensor, Option<Tensor>)>,
}

/// K/V cache for every attention layer in a [`TransformerStack`].
#[derive(Debug, Clone)]
pub struct TransformerCache {
    layers: Vec<AttentionCache>,
}

impl TransformerCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![AttentionCache::default(); num_layers],
        }
    }

    pub fn layer_seq_len(&self, layer: usize) -> Result<usize> {
        self.layers[layer].seq_len()
    }
}

impl TransformerStack {
    /// Construct from already-built blocks and optional projections/norm.
    pub fn from_blocks(
        blocks: Vec<TransformerBlock>,
        norm: Option<RmsNorm>,
        input_proj: Option<(Tensor, Option<Tensor>)>,
        output_proj: Option<(Tensor, Option<Tensor>)>,
    ) -> Self {
        Self {
            blocks,
            norm,
            input_proj,
            output_proj,
        }
    }

    /// Forward through the stack.
    ///
    /// Input `x` shape typically [batch, seq, in_dim] (in_dim may differ from hidden).
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = x.clone();

        // Optional input projection (e.g. 1024 -> 512 for pre_transformer)
        if let Some((ref w, ref bias)) = self.input_proj {
            h = h.broadcast_matmul(&w.t()?)?;
            if let Some(ref b) = bias {
                h = h.broadcast_add(b)?;
            }
        }

        // Transformer blocks
        for block in &self.blocks {
            h = block.forward(&h, mask)?;
        }

        self.forward_tail(h)
    }

    /// Create an empty autoregressive K/V cache matching this stack's layers.
    pub fn empty_cache(&self) -> TransformerCache {
        TransformerCache::new(self.blocks.len())
    }

    /// Forward through the stack while appending K/V tensors to `cache`.
    ///
    /// Call once with the full prompt to prefill, then repeatedly with `[B, 1, D]`
    /// tensors for incremental generation.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut TransformerCache,
    ) -> Result<Tensor> {
        debug_assert_eq!(self.blocks.len(), cache.layers.len());
        let mut h = x.clone();

        // Optional input projection (e.g. 1024 -> 512 for pre_transformer)
        if let Some((ref w, ref bias)) = self.input_proj {
            h = h.broadcast_matmul(&w.t()?)?;
            if let Some(ref b) = bias {
                h = h.broadcast_add(b)?;
            }
        }

        // Transformer blocks
        for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            h = block.forward_with_cache(&h, mask, Some(layer_cache))?;
        }

        self.forward_tail(h)
    }

    fn forward_tail(&self, mut h: Tensor) -> Result<Tensor> {
        // Optional final norm
        if let Some(ref n) = self.norm {
            h = n.forward(&h)?;
        }

        // Optional output projection (e.g. 512 -> 1024)
        if let Some((ref w, ref bias)) = self.output_proj {
            h = h.broadcast_matmul(&w.t()?)?;
            if let Some(ref b) = bias {
                h = h.broadcast_add(b)?;
            }
        }

        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    #[test]
    fn test_rms_norm_shape_and_basic() {
        let device = cpu();
        let hidden = 8usize;
        let weight = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let rms = RmsNorm::from_weight(weight, 1e-5);

        let x = Tensor::randn(0f32, 1.0, (2, 5, hidden), &device).unwrap();
        let y = rms.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 5, hidden]);

        // At ones weight + small eps, output should be close to input (normalized)
        let _x_mean_sq: f32 = x
            .sqr()
            .unwrap()
            .mean_keepdim(2)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap()[0][0][0];
        // Just sanity: not all zero
        let y_sum: f32 = y.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(y_sum.abs() > 0.1);
    }

    #[test]
    fn test_swiglu_shape() {
        let device = cpu();
        let hidden = 4usize;
        let inter = 8usize;

        // [inter, hidden]
        let gate = Tensor::randn(0f32, 0.1, (inter, hidden), &device).unwrap();
        let up = Tensor::randn(0f32, 0.1, (inter, hidden), &device).unwrap();
        // [hidden, inter]
        let down = Tensor::randn(0f32, 0.1, (hidden, inter), &device).unwrap();

        let swiglu = SwiGLU::from_weights(gate, up, down).unwrap();

        let x = Tensor::randn(0f32, 1.0, (3, 7, hidden), &device).unwrap();
        let y = swiglu.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 7, hidden]);
    }

    #[test]
    fn test_grouped_query_attention_shape_mha_and_gqa() {
        let device = cpu();
        let batch = 2usize;
        let seq = 6usize;
        let hidden = 8usize;
        let num_heads = 2usize;
        let head_dim = 4usize;
        let q_dim = num_heads * head_dim; // 8
        let num_kv_heads = 1usize; // test GQA repeat
        let kv_dim = num_kv_heads * head_dim; // 4

        let q = Tensor::randn(0f32, 0.1, (q_dim, hidden), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (kv_dim, hidden), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (kv_dim, hidden), &device).unwrap();
        let o = Tensor::randn(0f32, 0.1, (hidden, q_dim), &device).unwrap();

        let gqa =
            GroupedQueryAttention::from_weights(q, k, v, o, None, None, num_heads, num_kv_heads)
                .unwrap();

        let x = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device).unwrap();
        let y = gqa.forward(&x, None).unwrap();
        assert_eq!(y.dims(), &[batch, seq, hidden]);

        // With causal mask
        let mask = crate::custom_ops::causal_mask(seq, &device).unwrap();
        let y2 = gqa.forward(&x, Some(&mask)).unwrap();
        assert_eq!(y2.dims(), &[batch, seq, hidden]);

        let mut cache = AttentionCache::default();
        let y3 = gqa.forward_with_cache(&x, None, Some(&mut cache)).unwrap();
        assert_eq!(y3.dims(), &[batch, seq, hidden]);
        assert_eq!(cache.seq_len().unwrap(), seq);

        let next = Tensor::randn(0f32, 1.0, (batch, 1, hidden), &device).unwrap();
        let y4 = gqa
            .forward_with_cache(&next, None, Some(&mut cache))
            .unwrap();
        assert_eq!(y4.dims(), &[batch, 1, hidden]);
        assert_eq!(cache.seq_len().unwrap(), seq + 1);
    }

    #[test]
    fn test_rotary_embedding_shape_and_position_offset() {
        let device = cpu();
        let x = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            (1, 1, 2, 4),
            &device,
        )
        .unwrap();

        let y = apply_rotary_embedding(&x, 0, 10_000.0).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 4]);

        let first = y
            .narrow(2, 0, 1)
            .unwrap()
            .squeeze(2)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert!((first[0][0][0] - 1.0).abs() < 1e-6);
        assert!(first[0][0][1].abs() < 1e-6);
        assert!(first[0][0][2].abs() < 1e-6);
        assert!((first[0][0][3] - 1.0).abs() < 1e-6);

        let shifted = apply_rotary_embedding(&x, 1, 10_000.0).unwrap();
        let diff = shifted.broadcast_sub(&y).unwrap();
        let total_delta = diff
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(total_delta > 0.01);
    }

    #[test]
    fn test_grouped_query_attention_with_rope_cache() {
        let device = cpu();
        let batch = 1usize;
        let seq = 3usize;
        let hidden = 8usize;
        let num_heads = 2usize;
        let head_dim = 4usize;
        let q_dim = num_heads * head_dim;
        let num_kv_heads = 1usize;
        let kv_dim = num_kv_heads * head_dim;

        let q = Tensor::randn(0f32, 0.1, (q_dim, hidden), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (kv_dim, hidden), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (kv_dim, hidden), &device).unwrap();
        let o = Tensor::randn(0f32, 0.1, (hidden, q_dim), &device).unwrap();

        let gqa =
            GroupedQueryAttention::from_weights(q, k, v, o, None, None, num_heads, num_kv_heads)
                .unwrap()
                .with_rope_theta(10_000.0);

        let x = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device).unwrap();
        let mut cache = AttentionCache::default();
        let y = gqa.forward_with_cache(&x, None, Some(&mut cache)).unwrap();
        assert_eq!(y.dims(), &[batch, seq, hidden]);
        assert_eq!(cache.seq_len().unwrap(), seq);

        let next = Tensor::randn(0f32, 1.0, (batch, 1, hidden), &device).unwrap();
        let y_next = gqa
            .forward_with_cache(&next, None, Some(&mut cache))
            .unwrap();
        assert_eq!(y_next.dims(), &[batch, 1, hidden]);
        assert_eq!(cache.seq_len().unwrap(), seq + 1);
    }

    #[test]
    fn test_transformer_block_with_layer_scale() {
        let device = cpu();
        let hidden = 8usize;
        let inter = 16usize;
        let num_heads = 2usize;
        let head_dim = 4usize;
        let q_dim = num_heads * head_dim;
        let num_kv = 2usize;
        let eps = 1e-5;

        // Build sub components (small random weights)
        let q = Tensor::randn(0f32, 0.02, (q_dim, hidden), &device).unwrap();
        let k = Tensor::randn(0f32, 0.02, (q_dim, hidden), &device).unwrap();
        let v = Tensor::randn(0f32, 0.02, (q_dim, hidden), &device).unwrap();
        let o = Tensor::randn(0f32, 0.02, (hidden, q_dim), &device).unwrap();

        let gate = Tensor::randn(0f32, 0.02, (inter, hidden), &device).unwrap();
        let up = Tensor::randn(0f32, 0.02, (inter, hidden), &device).unwrap();
        let down = Tensor::randn(0f32, 0.02, (hidden, inter), &device).unwrap();

        let ln1_w = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let ln2_w = Tensor::ones(hidden, DType::F32, &device).unwrap();

        let attn_scale = Tensor::full(0.01f32, hidden, &device).unwrap();
        let mlp_scale = Tensor::full(0.01f32, hidden, &device).unwrap();

        let block = TransformerBlock::from_weights(
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
            ln1_w,
            ln2_w,
            Some(attn_scale),
            Some(mlp_scale),
            None,
            None,
            num_heads,
            num_kv,
            eps,
        )
        .unwrap();

        let x = Tensor::randn(0f32, 0.5, (1, 4, hidden), &device).unwrap();
        let y = block.forward(&x, None).unwrap();
        assert_eq!(y.dims(), &[1, 4, hidden]);
    }

    #[test]
    fn test_transformer_stack_with_projections_and_norm() {
        let device = cpu();
        let in_dim = 12usize;
        let hidden = 8usize;
        let out_dim = 10usize;
        let inter = 16usize;
        let num_heads = 2usize;
        let head_dim = 4usize;
        let q_dim = num_heads * head_dim;
        let num_kv = 2usize;
        let eps = 1e-5;
        let seq = 3usize;
        let batch = 1usize;

        // One block for stack
        let q = Tensor::randn(0f32, 0.02, (q_dim, hidden), &device).unwrap();
        let k = Tensor::randn(0f32, 0.02, (q_dim, hidden), &device).unwrap();
        let v = Tensor::randn(0f32, 0.02, (q_dim, hidden), &device).unwrap();
        let o = Tensor::randn(0f32, 0.02, (hidden, q_dim), &device).unwrap();

        let gate = Tensor::randn(0f32, 0.02, (inter, hidden), &device).unwrap();
        let up = Tensor::randn(0f32, 0.02, (inter, hidden), &device).unwrap();
        let down = Tensor::randn(0f32, 0.02, (hidden, inter), &device).unwrap();

        let ln1_w = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let ln2_w = Tensor::ones(hidden, DType::F32, &device).unwrap();

        let block = TransformerBlock::from_weights(
            q, k, v, o, gate, up, down, ln1_w, ln2_w, None, None, None, None, num_heads, num_kv,
            eps,
        )
        .unwrap();

        // Input proj: [hidden, in_dim] + bias
        let in_w = Tensor::randn(0f32, 0.02, (hidden, in_dim), &device).unwrap();
        let in_b = Tensor::zeros(hidden, DType::F32, &device).unwrap();

        // Output proj: [out_dim, hidden] + bias
        let out_w = Tensor::randn(0f32, 0.02, (out_dim, hidden), &device).unwrap();
        let out_b = Tensor::zeros(out_dim, DType::F32, &device).unwrap();

        // Final norm
        let final_norm_w = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let norm = RmsNorm::from_weight(final_norm_w, eps);

        let stack = TransformerStack::from_blocks(
            vec![block],
            Some(norm),
            Some((in_w, Some(in_b))),
            Some((out_w, Some(out_b))),
        );

        let x = Tensor::randn(0f32, 0.5, (batch, seq, in_dim), &device).unwrap();
        let y = stack.forward(&x, None).unwrap();
        assert_eq!(y.dims(), &[batch, seq, out_dim]);

        let mut cache = stack.empty_cache();
        let y_prefill = stack.forward_with_cache(&x, None, &mut cache).unwrap();
        assert_eq!(y_prefill.dims(), &[batch, seq, out_dim]);
        assert_eq!(cache.layer_seq_len(0).unwrap(), seq);

        let next = Tensor::randn(0f32, 0.5, (batch, 1, in_dim), &device).unwrap();
        let y_next = stack.forward_with_cache(&next, None, &mut cache).unwrap();
        assert_eq!(y_next.dims(), &[batch, 1, out_dim]);
        assert_eq!(cache.layer_seq_len(0).unwrap(), seq + 1);
    }

    #[test]
    fn test_stack_no_proj_no_norm() {
        let device = cpu();
        let hidden = 4usize;
        let inter = 8usize;
        let num_heads = 1usize;
        let _head_dim = 4usize;
        let q_dim = 4usize;
        let num_kv = 1usize;
        let eps = 1e-5;

        let q = Tensor::randn(0f32, 0.1, (q_dim, hidden), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (q_dim, hidden), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (q_dim, hidden), &device).unwrap();
        let o = Tensor::randn(0f32, 0.1, (hidden, q_dim), &device).unwrap();

        let gate = Tensor::randn(0f32, 0.1, (inter, hidden), &device).unwrap();
        let up = Tensor::randn(0f32, 0.1, (inter, hidden), &device).unwrap();
        let down = Tensor::randn(0f32, 0.1, (hidden, inter), &device).unwrap();

        let ln1 = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let ln2 = Tensor::ones(hidden, DType::F32, &device).unwrap();

        let block = TransformerBlock::from_weights(
            q, k, v, o, gate, up, down, ln1, ln2, None, None, None, None, num_heads, num_kv, eps,
        )
        .unwrap();

        let stack = TransformerStack::from_blocks(vec![block], None, None, None);

        let x = Tensor::randn(0f32, 1.0, (2, 3, hidden), &device).unwrap();
        let y = stack.forward(&x, None).unwrap();
        assert_eq!(y.dims(), &[2, 3, hidden]);
    }
}
