//! Qwen3-TTS speech tokenizer encoder.
//!
//! This implements the Mimi encoder path used by official Qwen3-TTS voice
//! clone ICL prompts:
//!
//! raw audio -> Mimi SEANet encoder -> causal transformer -> downsample -> RVQ codes.

use crate::custom_ops::{grouped_query_attention, layer_scale_3d};
use crate::error::VoxResult;
use crate::quantizer::{load_encoder_quantizer, SplitResidualVectorQuantizer};
use crate::weights::WeightStore;
use candle_core::{DType, Module, Result, Tensor};

const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const NUM_TRANSFORMER_LAYERS: usize = 8;
const NORM_EPS: f64 = 1e-5;
const ROPE_THETA: f64 = 10_000.0;
const SLIDING_WINDOW: usize = 250;
const ENCODE_DOWNSAMPLE_RATE: usize = 1_920;

/// Speech tokenizer encoder producing Qwen3-TTS codec frames.
pub struct SpeechTokenizerEncoder {
    encoder: MimiConvEncoder,
    transformer: MimiTransformer,
    downsample: MimiConv1d,
    quantizer: SplitResidualVectorQuantizer,
}

impl SpeechTokenizerEncoder {
    pub fn from_store(store: &WeightStore) -> VoxResult<Self> {
        Ok(Self {
            encoder: MimiConvEncoder::from_store(store)?,
            transformer: MimiTransformer::from_store(store)?,
            downsample: MimiConv1d::load(
                store,
                "encoder.downsample.conv",
                2,
                1,
                PadMode::Replicate,
                false,
            )?,
            quantizer: load_encoder_quantizer(store)?,
        })
    }

    /// Encode raw audio into official codec frame order `[batch, frames, 16]`.
    ///
    /// `input_values` accepts `[batch, samples]` or `[batch, 1, samples]`.
    /// When `valid_samples` is provided, the result is trimmed to
    /// `ceil(valid_samples / 1920)`, matching official padding-mask trimming.
    pub fn encode_audio_codes(
        &self,
        input_values: &Tensor,
        valid_samples: Option<usize>,
    ) -> Result<Tensor> {
        let input_values = match input_values.rank() {
            2 => input_values.unsqueeze(1)?,
            3 => input_values.clone(),
            rank => {
                return Err(candle_core::Error::Msg(format!(
                    "speech tokenizer encoder expected rank 2 or 3 audio, got {rank}"
                )))
            }
        };

        let mut h = input_values.to_dtype(self.encoder.dtype())?;
        h = self.encoder.forward(&h)?;
        h = h.transpose(1, 2)?;
        h = self.transformer.forward(&h)?;
        h = h.transpose(1, 2)?;
        h = self.downsample.forward(&h)?;

        let codes = self.quantizer.encode(&h, 16)?;
        let refs: Vec<&Tensor> = codes.iter().collect();
        let mut codes = Tensor::stack(&refs, 2)?;

        if let Some(samples) = valid_samples {
            let target_frames = samples.div_ceil(ENCODE_DOWNSAMPLE_RATE);
            let frames = codes.dim(1)?;
            codes = codes.narrow(1, 0, target_frames.min(frames))?;
        }

        Ok(codes)
    }
}

struct MimiConvEncoder {
    layers: Vec<MimiConvLayer>,
    dtype: DType,
}

impl MimiConvEncoder {
    fn from_store(store: &WeightStore) -> VoxResult<Self> {
        let layers = vec![
            MimiConvLayer::Conv(MimiConv1d::load(
                store,
                "encoder.encoder.layers.0.conv",
                1,
                1,
                PadMode::Constant,
                true,
            )?),
            MimiConvLayer::ResNet(MimiResnetBlock::load(store, "encoder.encoder.layers.1")?),
            MimiConvLayer::Elu,
            MimiConvLayer::Conv(MimiConv1d::load(
                store,
                "encoder.encoder.layers.3.conv",
                4,
                1,
                PadMode::Constant,
                true,
            )?),
            MimiConvLayer::ResNet(MimiResnetBlock::load(store, "encoder.encoder.layers.4")?),
            MimiConvLayer::Elu,
            MimiConvLayer::Conv(MimiConv1d::load(
                store,
                "encoder.encoder.layers.6.conv",
                5,
                1,
                PadMode::Constant,
                true,
            )?),
            MimiConvLayer::ResNet(MimiResnetBlock::load(store, "encoder.encoder.layers.7")?),
            MimiConvLayer::Elu,
            MimiConvLayer::Conv(MimiConv1d::load(
                store,
                "encoder.encoder.layers.9.conv",
                6,
                1,
                PadMode::Constant,
                true,
            )?),
            MimiConvLayer::ResNet(MimiResnetBlock::load(store, "encoder.encoder.layers.10")?),
            MimiConvLayer::Elu,
            MimiConvLayer::Conv(MimiConv1d::load(
                store,
                "encoder.encoder.layers.12.conv",
                8,
                1,
                PadMode::Constant,
                true,
            )?),
            MimiConvLayer::Elu,
            MimiConvLayer::Conv(MimiConv1d::load(
                store,
                "encoder.encoder.layers.14.conv",
                1,
                1,
                PadMode::Constant,
                true,
            )?),
        ];
        let dtype = layers
            .iter()
            .find_map(MimiConvLayer::dtype)
            .unwrap_or(DType::F32);
        Ok(Self { layers, dtype })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

enum MimiConvLayer {
    Conv(MimiConv1d),
    ResNet(MimiResnetBlock),
    Elu,
}

impl MimiConvLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(conv) => conv.forward(x),
            Self::ResNet(block) => block.forward(x),
            Self::Elu => elu(x),
        }
    }

    fn dtype(&self) -> Option<DType> {
        match self {
            Self::Conv(conv) => Some(conv.dtype()),
            Self::ResNet(block) => Some(block.dtype()),
            Self::Elu => None,
        }
    }
}

struct MimiResnetBlock {
    conv1: MimiConv1d,
    conv2: MimiConv1d,
}

impl MimiResnetBlock {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        Ok(Self {
            conv1: MimiConv1d::load(
                store,
                &format!("{prefix}.block.1.conv"),
                1,
                1,
                PadMode::Constant,
                true,
            )?,
            conv2: MimiConv1d::load(
                store,
                &format!("{prefix}.block.3.conv"),
                1,
                1,
                PadMode::Constant,
                true,
            )?,
        })
    }

    fn dtype(&self) -> DType {
        self.conv1.dtype()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = elu(x)?;
        let h = self.conv1.forward(&h)?;
        let h = elu(&h)?;
        let h = self.conv2.forward(&h)?;
        x.add(&h)
    }
}

#[derive(Clone, Copy)]
enum PadMode {
    Constant,
    Replicate,
}

struct MimiConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    dilation: usize,
    pad_mode: PadMode,
}

impl MimiConv1d {
    fn load(
        store: &WeightStore,
        prefix: &str,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
        bias: bool,
    ) -> VoxResult<Self> {
        Ok(Self {
            weight: store.require(&format!("{prefix}.weight"))?.clone(),
            bias: if bias {
                Some(store.require(&format!("{prefix}.bias"))?.clone())
            } else {
                None
            },
            stride,
            dilation,
            pad_mode,
        })
    }

    fn dtype(&self) -> DType {
        self.weight.dtype()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let kernel = self.weight.dim(2)?;
        let effective_kernel = (kernel - 1) * self.dilation + 1;
        let padding_total = effective_kernel.saturating_sub(self.stride);
        let extra_padding =
            extra_padding_for_conv1d(x.dim(2)?, effective_kernel, padding_total, self.stride);
        let padded = pad_1d(x, padding_total, extra_padding, self.pad_mode)?;
        let mut y = padded.conv1d(&self.weight, 0, self.stride, self.dilation, 1)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)?;
        }
        Ok(y)
    }
}

fn extra_padding_for_conv1d(
    length: usize,
    effective_kernel: usize,
    padding_total: usize,
    stride: usize,
) -> usize {
    let numerator = length as isize - effective_kernel as isize + padding_total as isize;
    let n_frames = div_ceil_signed(numerator, stride as isize);
    let ideal = n_frames * stride as isize + effective_kernel as isize - padding_total as isize;
    (ideal - length as isize).max(0) as usize
}

fn div_ceil_signed(a: isize, b: isize) -> isize {
    if a >= 0 {
        (a + b - 1) / b
    } else {
        a / b
    }
}

fn pad_1d(x: &Tensor, left: usize, right: usize, mode: PadMode) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let (batch, channels, len) = x.dims3()?;
    let padded = match mode {
        PadMode::Constant => {
            let left_pad = Tensor::zeros((batch, channels, left), x.dtype(), x.device())?;
            let right_pad = Tensor::zeros((batch, channels, right), x.dtype(), x.device())?;
            Tensor::cat(&[&left_pad, x, &right_pad], 2)?
        }
        PadMode::Replicate => {
            let first = x.narrow(2, 0, 1)?.broadcast_as((batch, channels, left))?;
            let last = x
                .narrow(2, len - 1, 1)?
                .broadcast_as((batch, channels, right))?;
            Tensor::cat(&[&first, x, &last], 2)?
        }
    };
    Ok(padded)
}

struct MimiTransformer {
    layers: Vec<MimiTransformerLayer>,
    mask: Option<Tensor>,
}

impl MimiTransformer {
    fn from_store(store: &WeightStore) -> VoxResult<Self> {
        let mut layers = Vec::with_capacity(NUM_TRANSFORMER_LAYERS);
        for i in 0..NUM_TRANSFORMER_LAYERS {
            layers.push(MimiTransformerLayer::load(
                store,
                &format!("encoder.encoder_transformer.layers.{i}"),
            )?);
        }
        Ok(Self { layers, mask: None })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let mask = match &self.mask {
            Some(mask) if mask.dim(0)? == seq_len => mask.clone(),
            _ => sliding_additive_causal_mask(seq_len, SLIDING_WINDOW, x.device())?,
        };
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &mask)?;
        }
        Ok(h)
    }
}

struct MimiTransformerLayer {
    input_layernorm: candle_nn::LayerNorm,
    post_attention_layernorm: candle_nn::LayerNorm,
    attention: MimiAttention,
    fc1: Tensor,
    fc2: Tensor,
    attn_scale: Tensor,
    mlp_scale: Tensor,
}

impl MimiTransformerLayer {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        let input_layernorm = candle_nn::LayerNorm::new(
            store
                .require(&format!("{prefix}.input_layernorm.weight"))?
                .clone(),
            store
                .require(&format!("{prefix}.input_layernorm.bias"))?
                .clone(),
            NORM_EPS,
        );
        let post_attention_layernorm = candle_nn::LayerNorm::new(
            store
                .require(&format!("{prefix}.post_attention_layernorm.weight"))?
                .clone(),
            store
                .require(&format!("{prefix}.post_attention_layernorm.bias"))?
                .clone(),
            NORM_EPS,
        );
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention: MimiAttention::load(store, &format!("{prefix}.self_attn"))?,
            fc1: store.require(&format!("{prefix}.mlp.fc1.weight"))?.clone(),
            fc2: store.require(&format!("{prefix}.mlp.fc2.weight"))?.clone(),
            attn_scale: store
                .require(&format!("{prefix}.self_attn_layer_scale.scale"))?
                .clone(),
            mlp_scale: store
                .require(&format!("{prefix}.mlp_layer_scale.scale"))?
                .clone(),
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.attention.forward(&h, mask)?;
        let h = layer_scale_3d(&h, &self.attn_scale, 2)?;
        let h = residual.add(&h)?;

        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = h.broadcast_matmul(&self.fc1.t()?)?.gelu_erf()?;
        let h = h.broadcast_matmul(&self.fc2.t()?)?;
        let h = layer_scale_3d(&h, &self.mlp_scale, 2)?;
        residual.add(&h)
    }
}

struct MimiAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
}

impl MimiAttention {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        Ok(Self {
            q_proj: store.require(&format!("{prefix}.q_proj.weight"))?.clone(),
            k_proj: store.require(&format!("{prefix}.k_proj.weight"))?.clone(),
            v_proj: store.require(&format!("{prefix}.v_proj.weight"))?.clone(),
            o_proj: store.require(&format!("{prefix}.o_proj.weight"))?.clone(),
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let q = x
            .broadcast_matmul(&self.q_proj.t()?)?
            .reshape((batch, seq_len, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = x
            .broadcast_matmul(&self.k_proj.t()?)?
            .reshape((batch, seq_len, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = x
            .broadcast_matmul(&self.v_proj.t()?)?
            .reshape((batch, seq_len, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = apply_rotary(&q, &k, ROPE_THETA)?;
        let h = grouped_query_attention(&q, &k, &v, Some(mask), NUM_HEADS, NUM_KV_HEADS)?;
        let h = h
            .transpose(1, 2)?
            .reshape((batch, seq_len, NUM_HEADS * HEAD_DIM))?;
        h.broadcast_matmul(&self.o_proj.t()?)
    }
}

fn apply_rotary(q: &Tensor, k: &Tensor, theta: f64) -> Result<(Tensor, Tensor)> {
    let (_, _, seq_len, head_dim) = q.dims4()?;
    if head_dim % 2 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "RoPE head_dim must be even, got {head_dim}"
        )));
    }
    let half = head_dim / 2;
    let device = q.device();
    let dtype = q.dtype();

    let mut cos = Vec::with_capacity(seq_len * head_dim);
    let mut sin = Vec::with_capacity(seq_len * head_dim);
    for pos in 0..seq_len {
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            cos.push(angle.cos() as f32);
        }
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            cos.push(angle.cos() as f32);
        }
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            sin.push(angle.sin() as f32);
        }
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            sin.push(angle.sin() as f32);
        }
    }
    let cos = Tensor::from_vec(cos, (1, 1, seq_len, head_dim), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, (1, 1, seq_len, head_dim), device)?.to_dtype(dtype)?;
    Ok((
        rotate_with_cos_sin(q, &cos, &sin)?,
        rotate_with_cos_sin(k, &cos, &sin)?,
    ))
}

fn rotate_with_cos_sin(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(3)?;
    let half = head_dim / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let neg_x2 = x2.neg()?;
    let rotated = Tensor::cat(&[&neg_x2, &x1], 3)?;
    x.broadcast_mul(cos)?.add(&rotated.broadcast_mul(sin)?)
}

fn sliding_additive_causal_mask(
    seq_len: usize,
    window: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        let start = if i >= window { i - window + 1 } else { 0 };
        for j in start..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

fn elu(x: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
    let positive = x.gt(&zero)?;
    let negative = x
        .exp()?
        .sub(&Tensor::ones(x.shape(), x.dtype(), x.device())?)?;
    positive.where_cond(x, &negative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::path::Path;

    #[test]
    fn test_extra_padding_matches_mimi_examples() {
        assert_eq!(extra_padding_for_conv1d(24_000, 7, 6, 1), 0);
        assert_eq!(extra_padding_for_conv1d(24_000, 8, 4, 4), 0);
        assert_eq!(extra_padding_for_conv1d(24_001, 8, 4, 4), 3);
        assert_eq!(extra_padding_for_conv1d(3, 7, 6, 1), 0);
    }

    #[test]
    fn test_sliding_additive_causal_mask() {
        let device = Device::Cpu;
        let mask = sliding_additive_causal_mask(4, 2, &device).unwrap();
        let values: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(values[0], 0.0);
        assert!(values[1].is_infinite());
        assert_eq!(values[4], 0.0);
        assert_eq!(values[5], 0.0);
        assert!(values[6].is_infinite());
        assert!(values[8].is_infinite());
        assert_eq!(values[10], 0.0);
    }

    #[test]
    #[ignore]
    fn test_speech_tokenizer_encoder_real_weights_smoke() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("weights/model-0.6b/speech_tokenizer/model.safetensors");
        if !path.exists() {
            eprintln!("missing {}", path.display());
            return;
        }
        let device = Device::Cpu;
        let store = WeightStore::from_file(path, &device).unwrap();
        let encoder = SpeechTokenizerEncoder::from_store(&store).unwrap();
        let audio = Tensor::zeros((1, 3_840), DType::F32, &device).unwrap();
        let codes = encoder.encode_audio_codes(&audio, Some(3_840)).unwrap();
        assert_eq!(codes.dims(), &[1, 2, 16]);
    }
}
