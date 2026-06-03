//! Custom operations not natively provided by Candle.
//!
//! Includes:
//! - SnakeBeta activation
//! - Causal padding utilities
//! - LayerScale
//! - Sliding window causal attention mask

use candle_core::{Result, Tensor};

fn cast_like(tensor: &Tensor, like: &Tensor) -> Result<Tensor> {
    if tensor.dtype() == like.dtype() {
        Ok(tensor.clone())
    } else {
        tensor.to_dtype(like.dtype())
    }
}

// ── SnakeBeta Activation ──────────────────────────────────────────────────

/// SnakeBeta activation: `y = x + beta * sin^2(alpha * x)`
///
/// Used in the tokenizer decoder's ResidualUnit blocks.
/// Both `alpha` and `beta` are learnable per-channel parameters.
///
/// # Arguments
/// * `x` - Input tensor of shape `[..., channels]`
/// * `alpha` - Frequency parameter, shape `[channels]`
/// * `beta` - Amplitude parameter, shape `[channels]`
pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    // sin^2(z) = (1 - cos(2z)) / 2  — numerically more stable
    let two = Tensor::new(&[2.0f32], x.device())?.to_dtype(x.dtype())?;
    let one = Tensor::new(&[1.0f32], x.device())?.to_dtype(x.dtype())?;
    let alpha = cast_like(alpha, x)?;
    let beta = cast_like(beta, x)?;

    // alpha * x
    let ax = x.broadcast_mul(&alpha.unsqueeze(0)?)?;

    // 2 * alpha * x
    let two_ax = ax.broadcast_mul(&two)?;

    // cos(2 * alpha * x)
    let cos_two_ax = two_ax.cos()?;

    // (1 - cos(2*alpha*x)) / 2 = sin^2(alpha*x)
    let sin_sq = (one.broadcast_sub(&cos_two_ax)?).broadcast_div(&two)?;

    // beta * sin^2(alpha * x)
    let modulation = beta.unsqueeze(0)?.broadcast_mul(&sin_sq)?;

    // x + beta * sin^2(alpha * x)
    x.broadcast_add(&modulation)
}

// ── LayerScale ────────────────────────────────────────────────────────────

/// LayerScale: `y = x * gamma`
///
/// Per-channel learnable scaling factor, typically initialized to 0.01.
///
/// # Arguments
/// * `x` - Input tensor
/// * `gamma` - Scale parameter, shape `[channels]`
pub fn layer_scale(x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let gamma = cast_like(gamma, x)?;
    x.broadcast_mul(&gamma.unsqueeze(0)?)
}

/// Layer scale with explicit channel dimension for 3D tensors.
///
/// For 3D inputs, `channel_dim` indicates which axis C is on:
/// - `channel_dim=1` for `[B, C, T]` → gamma reshaped to `[1, C, 1]`
/// - `channel_dim=2` for `[B, T, C]` → gamma reshaped to `[1, 1, C]`
pub fn layer_scale_3d(x: &Tensor, gamma: &Tensor, channel_dim: usize) -> Result<Tensor> {
    if x.rank() != 3 {
        return layer_scale(x, gamma);
    }
    let (_d0, _d1, _d2) = x.dims3()?;
    let gamma = cast_like(gamma, x)?;
    let gamma = match channel_dim {
        1 => gamma.reshape((1, gamma.dim(0)?, 1))?,
        2 => gamma.reshape((1, 1, gamma.dim(0)?))?,
        _ => gamma.unsqueeze(0)?.unsqueeze(0)?,
    };
    x.broadcast_mul(&gamma)
}

// ── Causal Padding ────────────────────────────────────────────────────────

/// Apply causal (left-only) padding to a 1-D signal.
///
/// Pads `kernel_size - 1` zeros on the left side, so that the
/// convolution output at position `t` depends only on inputs `[0..=t]`.
///
/// # Arguments
/// * `x` - Input tensor of shape `[batch, channels, length]`
/// * `kernel_size` - Convolution kernel size
///
/// # Returns
/// Padded tensor of shape `[batch, channels, length + kernel_size - 1]`
#[inline]
pub fn causal_pad_left(x: &Tensor, kernel_size: usize) -> Result<Tensor> {
    if kernel_size <= 1 {
        return Ok(x.clone());
    }

    let pad_size = kernel_size - 1;
    let dims = x.dims3()?;
    let (batch, channels, _length) = dims;

    // Create zero padding: [batch, channels, pad_size]
    let padding = Tensor::zeros((batch, channels, pad_size), x.dtype(), x.device())?;

    // Concatenate: [padding | x]
    Tensor::cat(&[&padding, x], 2)
}

/// Crop the right side of a 1-D signal to remove future-looking samples
/// after a causal transposed convolution.
///
/// # Arguments
/// * `x` - Input tensor of shape `[batch, channels, length]`
/// * `crop_size` - Number of samples to remove from the right
///
/// # Returns
/// Cropped tensor of shape `[batch, channels, length - crop_size]`
#[inline]
pub fn causal_crop_right(x: &Tensor, crop_size: usize) -> Result<Tensor> {
    if crop_size == 0 {
        return Ok(x.clone());
    }

    let dims = x.dims3()?;
    let target_len = dims.2 - crop_size;
    x.narrow(2, 0, target_len)
}

// ── Sliding Window Causal Attention Mask ──────────────────────────────────

/// Create a sliding window causal attention mask.
///
/// Position `(i, j)` is allowed only if `i - window < j <= i`.
///
/// # Arguments
/// * `seq_len` - Sequence length
/// * `window` - Attention window size
/// * `device` - Target device
///
/// # Returns
/// Boolean mask tensor of shape `[seq_len, seq_len]` (true = attend)
pub fn sliding_window_causal_mask(
    seq_len: usize,
    window: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];

    for i in 0..seq_len {
        let start = if i >= window { i - window + 1 } else { 0 };
        for j in start..=i {
            mask[i * seq_len + j] = 1.0;
        }
    }

    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

/// Create a standard causal (lower-triangular) attention mask.
///
/// # Arguments
/// * `seq_len` - Sequence length
/// * `device` - Target device
///
/// # Returns
/// Float mask tensor of shape `[seq_len, seq_len]` (0.0 = attend, -inf = mask)
pub fn causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mut mask = vec![neg_inf; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }

    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

// ── Grouped Query Attention (manual) ──────────────────────────────────────

/// Manual Grouped Query Attention for CUDA (no fused SDPA).
///
/// Computes: `softmax(Q·K^T / √d + mask) · V`
///
/// # Arguments
/// * `q` - Query: `[batch, num_heads, seq_len, head_dim]`
/// * `k` - Key: `[batch, num_kv_heads, seq_len, head_dim]`
/// * `v` - Value: `[batch, num_kv_heads, seq_len, head_dim]`
/// * `mask` - Optional attention mask: `[seq_len, seq_len]`
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads (for GQA)
///
/// # Returns
/// Output: `[batch, num_heads, seq_len, head_dim]`
pub fn grouped_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<Tensor> {
    let head_dim = q.dim(3)?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Repeat K/V heads to match Q heads (GQA expansion)
    let repeat_factor = num_heads / num_kv_heads;
    let (k_expanded, v_expanded) = if repeat_factor > 1 {
        let k_rep = k.repeat((1, repeat_factor, 1, 1))?;
        let v_rep = v.repeat((1, repeat_factor, 1, 1))?;
        (k_rep, v_rep)
    } else {
        (k.clone(), v.clone())
    };

    // Q·K^T: [batch, num_heads, seq_len, seq_len]
    let k_t = k_expanded.transpose(2, 3)?;
    let attn_weights = q.matmul(&k_t)?;

    // Scale
    let scale_tensor = Tensor::new(&[scale as f32], q.device())?.to_dtype(attn_weights.dtype())?;
    let attn_weights = attn_weights.broadcast_mul(&scale_tensor)?;

    // Apply mask
    let attn_weights = if let Some(m) = mask {
        let m = cast_like(m, &attn_weights)?;
        attn_weights.broadcast_add(&m.unsqueeze(0)?.unsqueeze(0)?)?
    } else {
        attn_weights
    };

    // Softmax over last dim
    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // attn_probs · V: [batch, num_heads, seq_len, head_dim]
    attn_probs.matmul(&v_expanded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_snake_beta_identity_at_zero() {
        let device = Device::Cpu;
        // At x=0, sin^2(alpha*0) = 0, so output = x = 0
        let x = Tensor::zeros((1, 4), candle_core::DType::F32, &device).unwrap();
        let alpha = Tensor::ones(4, candle_core::DType::F32, &device).unwrap();
        let beta = Tensor::ones(4, candle_core::DType::F32, &device).unwrap();

        let y = snake_beta(&x, &alpha, &beta).unwrap();
        let vals: Vec<f32> = y.to_vec2().unwrap()[0].clone();
        for v in &vals {
            assert!(v.abs() < 1e-6, "snake_beta(0) should be ~0, got {v}");
        }
    }

    #[test]
    fn test_layer_scale() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();
        let gamma = Tensor::new(&[0.5f32, 0.5, 0.5], &device).unwrap();

        let y = layer_scale(&x, &gamma).unwrap();
        let vals: Vec<f32> = y.to_vec2().unwrap()[0].clone();
        assert!((vals[0] - 0.5).abs() < 1e-6);
        assert!((vals[1] - 1.0).abs() < 1e-6);
        assert!((vals[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_causal_pad_left() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0]]], &device).unwrap(); // [1, 1, 3]
        let padded = causal_pad_left(&x, 3).unwrap(); // kernel_size=3 → pad 2

        let vals: Vec<f32> = padded
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(vals.len(), 5);
        assert_eq!(vals[0], 0.0); // pad
        assert_eq!(vals[1], 0.0); // pad
        assert_eq!(vals[2], 1.0); // original
        assert_eq!(vals[3], 2.0);
        assert_eq!(vals[4], 3.0);
    }

    #[test]
    fn test_causal_crop_right() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0, 5.0]]], &device).unwrap();
        let cropped = causal_crop_right(&x, 2).unwrap();

        let vals: Vec<f32> = cropped
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 2.0);
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = causal_mask(4, &device).unwrap();
        let vals: Vec<Vec<f32>> = mask.to_vec2().unwrap();

        // Lower triangular: 0.0 where j <= i, -inf where j > i
        assert_eq!(vals[0][0], 0.0);
        assert_eq!(vals[0][1], f32::NEG_INFINITY);
        assert_eq!(vals[1][0], 0.0);
        assert_eq!(vals[1][1], 0.0);
        assert_eq!(vals[1][2], f32::NEG_INFINITY);
        assert_eq!(vals[3][3], 0.0);
    }

    #[test]
    fn test_sliding_window_mask() {
        let device = Device::Cpu;
        let mask = sliding_window_causal_mask(5, 3, &device).unwrap();
        let vals: Vec<Vec<f32>> = mask.to_vec2().unwrap();

        // Row 4 (i=4): window=3, so attend to j=2,3,4
        assert_eq!(vals[4][0], 0.0); // outside window
        assert_eq!(vals[4][1], 0.0); // outside window
        assert_eq!(vals[4][2], 1.0); // in window
        assert_eq!(vals[4][3], 1.0); // in window
        assert_eq!(vals[4][4], 1.0); // in window
    }
}
