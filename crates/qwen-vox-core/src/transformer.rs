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

use candle_core::{Result, Tensor};

#[cfg(test)]
use candle_core::DType;

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
        })
    }

    /// Forward pass with optional attention mask (shape [seq_len, seq_len]).
    ///
    /// Expects `x` of shape [batch, seq_len, hidden].
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
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
            candle_nn::ops::rms_norm(&q, qn, 1e-5)?
        } else {
            q
        };
        let k = if let Some(ref kn) = self.k_norm {
            candle_nn::ops::rms_norm(&k, kn, 1e-5)?
        } else {
            k
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
        )?;
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

    /// Pre-norm block forward.
    ///
    /// `h = x + layer_scale( attn( ln1(x) ), attn_gamma )`  (if gamma present)
    /// `h = h + layer_scale( mlp( ln2(h) ), mlp_gamma )`     (if gamma present)
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Attention sub-layer (pre-norm + residual + optional scale)
        let h = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&h, mask)?;
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
