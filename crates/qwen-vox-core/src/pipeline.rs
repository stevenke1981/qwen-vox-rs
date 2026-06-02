//! Complete codec decoder pipeline for Qwen3-TTS.
//!
//! Orchestrates the full decoder path:
//!
//!    codes → SplitRVQ quantizer → pre_conv → pre_transformer
//!           → upsample (×2) → tokenizer_decoder → waveform
//!
//! Pipeline stages (`decoder.*` prefix in tokenizer weights):
//!
//! | Stage                    | Weight prefix               | Input → Output dims         |
//! |--------------------------|-----------------------------|-----------------------------|
//! | 1. SplitRVQ quantizer    | `quantizer.rvq_{first,rest}`| [B, T] × 16 → [B, 512, T]  |
//! | 2. Pre-conv (CausalConv) | `pre_conv`                  | [B, 512, T] → [B, 1024, T] |
//! | 3. Pre-transformer (8×)  | `pre_transformer`           | [B, 1024, T] → [B, 1024, T] |
//! | 4. Upsample (2×)         | `upsample.{0,1}`            | [B, 1024, T] → [B, 1024, T×4] |
//! | 5. TokenizerDecoder      | `decoder` (prefix)          | [B, 1024, T×4] → [B, 1, T×1920] |
//!
//! Total upsampling rate: 4× (upsample) × 480× (decoder blocks) = 1920×.
//! At 24 kHz, this is 12.5 codec frames per second.
//!
//! All convolutions are strictly causal.

use crate::conv_decoder::{CausalConv1dLayer, CausalConvTranspose1dLayer, TokenizerDecoder};
use crate::custom_ops::layer_scale_3d;
use crate::error::{VoxError, VoxResult};
use crate::quantizer::{
    load_decoder_codebooks, ResidualVectorQuantizer, SplitResidualVectorQuantizer,
};
use crate::speaker_encoder::SpeakerEncoder;
use crate::talker::Talker;
use crate::transformer::{RmsNorm, TransformerBlock, TransformerStack};
use crate::weights::{ComponentWeights, WeightStore};
use candle_core::{Device, Module, Result, Tensor};

// ── Constants ──────────────────────────────────────────────────────────────────

const PRE_TRANSFORMER_EPS: f64 = 1e-5;
const PRE_TRANSFORMER_HEADS: usize = 8;
const PRE_TRANSFORMER_KV_HEADS: usize = 8;
const PRE_TRANSFORMER_LAYERS: usize = 8;
const UPSAMPLE_STAGES: usize = 2;
pub const TOKENIZER_SAMPLE_RATE: u32 = 24_000;
pub const TOKENIZER_DECODE_UPSAMPLE_RATE: usize = 1_920;
pub const TOKENIZER_FRAME_RATE_HZ: f32 =
    TOKENIZER_SAMPLE_RATE as f32 / TOKENIZER_DECODE_UPSAMPLE_RATE as f32;

// ── GELU (exact, erf-based) ────────────────────────────────────────────────────

/// Exact GELU activation: `x * 0.5 * (1 + erf(x / sqrt(2)))`
fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    let sqrt2 = Tensor::new(&[std::f64::consts::SQRT_2 as f32], x.device())?;
    let half = Tensor::new(&[0.5f32], x.device())?;
    let one = Tensor::new(&[1.0f32], x.device())?;
    let x_div = x.broadcast_div(&sqrt2)?;
    let erf = x_div.erf()?;
    let inner = one.broadcast_add(&erf)?;
    let factor = half.broadcast_mul(&inner)?;
    x.broadcast_mul(&factor)
}

// ── ConvNeXt Block (upsample feature refinement) ──────────────────────────────

/// ConvNeXt-style block used inside each upsample stage.
///
/// Architecture: LayerNorm → DepthwiseConv1d → GELU → Linear(×4) → Linear(÷4) → LayerScale
///
/// Weight shapes (e.g. 1024 channels):
///   norm.{weight,bias}:    [1024]
///   dwconv.conv.{weight,bias}: [1024, 1, 7]   (depthwise, groups=1024)
///   pwconv1.{weight,bias}: [4096, 1024]         (pointwise expansion 4×)
///   pwconv2.{weight,bias}: [1024, 4096]         (pointwise contraction)
///   gamma:                [1024]               (LayerScale)
struct ConvNeXtBlock {
    layer_norm: candle_nn::LayerNorm,
    dwconv: CausalConv1dLayer,
    pwconv1_weight: Tensor,
    pwconv1_bias: Option<Tensor>,
    pwconv2_weight: Tensor,
    pwconv2_bias: Option<Tensor>,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn from_weights(weights: &ComponentWeights, prefix: &str) -> VoxResult<Self> {
        let norm_w = weights.require(&format!("{prefix}.norm.weight"))?.clone();
        let norm_b = weights.require(&format!("{prefix}.norm.bias"))?.clone();
        let layer_norm = candle_nn::LayerNorm::new(norm_w, norm_b, 1e-6);

        let dw_w = weights
            .require(&format!("{prefix}.dwconv.conv.weight"))?
            .clone();
        let dw_b = Some(
            weights
                .require(&format!("{prefix}.dwconv.conv.bias"))?
                .clone(),
        );

        // Depthwise: weight shape [C, 1, K], groups = C
        let groups = dw_w.dim(0)?;
        let dwconv = CausalConv1dLayer::from_weights(dw_w, dw_b, 1, groups, 1)?;

        let p1w = weights
            .require(&format!("{prefix}.pwconv1.weight"))?
            .clone();
        let p1b = Some(weights.require(&format!("{prefix}.pwconv1.bias"))?.clone());
        let p2w = weights
            .require(&format!("{prefix}.pwconv2.weight"))?
            .clone();
        let p2b = Some(weights.require(&format!("{prefix}.pwconv2.bias"))?.clone());
        let gamma = weights.require(&format!("{prefix}.gamma"))?.clone();

        Ok(Self {
            layer_norm,
            dwconv,
            pwconv1_weight: p1w,
            pwconv1_bias: p1b,
            pwconv2_weight: p2w,
            pwconv2_bias: p2b,
            gamma,
        })
    }

    /// Forward: [B, C, T] → [B, C, T]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();

        // Norm: [B, C, T] → [B, T, C] → LayerNorm over last dim → [B, T, C]
        let mut h = x.transpose(1, 2)?;
        h = self.layer_norm.forward(&h)?;

        // Depthwise conv: [B, T, C] → [B, C, T] → causal dw conv → [B, C, T]
        h = h.transpose(1, 2)?;
        h = self.dwconv.forward(&h)?;

        // Pointwise: [B, C, T] → [B, T, C] → GELU → pwconv1 → pwconv2
        h = h.transpose(1, 2)?;
        h = gelu_erf(&h)?;

        let b = h.dim(0)?;
        let t = h.dim(1)?;
        let c = h.dim(2)?;

        // pwconv1: expand 4× via Linear [C] → [4C]
        let flat = h.reshape((b * t, c))?;
        let flat = flat.matmul(&self.pwconv1_weight.t()?)?;
        let mut h = flat.reshape((b, t, self.pwconv1_weight.dim(0)?))?;
        if let Some(ref bias) = self.pwconv1_bias {
            h = h.broadcast_add(bias)?;
        }

        // pwconv2: contract via Linear [4C] → [C]
        let c2 = h.dim(2)?;
        let flat = h.reshape((b * t, c2))?;
        let flat = flat.matmul(&self.pwconv2_weight.t()?)?;
        let mut h = flat.reshape((b, t, self.pwconv2_weight.dim(0)?))?;
        if let Some(ref bias) = self.pwconv2_bias {
            h = h.broadcast_add(bias)?;
        }

        // [B, T, C] → [B, C, T] + LayerScale + residual
        h = h.transpose(1, 2)?;
        h = layer_scale_3d(&h, &self.gamma, 1)?;
        h.add(&shortcut)
    }
}

// ── Upsample Stage ─────────────────────────────────────────────────────────────

/// One upsampler stage: ConvTranspose1d (stride=2) → ConvNeXtBlock.
struct UpsampleStage {
    conv_transpose: CausalConvTranspose1dLayer,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    fn from_weights(weights: &ComponentWeights, stage_idx: usize) -> VoxResult<Self> {
        let prefix = format!("upsample.{stage_idx}");

        let ct_w = weights.require(&format!("{prefix}.0.conv.weight"))?.clone();
        let ct_b = Some(weights.require(&format!("{prefix}.0.conv.bias"))?.clone());
        let conv_transpose = CausalConvTranspose1dLayer::from_weights(ct_w, ct_b, 2)?;

        let convnext = ConvNeXtBlock::from_weights(weights, &format!("{prefix}.1"))?;

        Ok(Self {
            conv_transpose,
            convnext,
        })
    }

    /// Forward: [B, C, T] → [B, C, T×2]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv_transpose.forward(x)?;
        self.convnext.forward(&h)
    }
}

// ── Full Pipeline ──────────────────────────────────────────────────────────────

/// Complete codec decoder pipeline.
///
/// Decodes 16 discrete VQ code indices into a mono waveform via:
/// quantizer → pre_conv → pre_transformer → upsample → tokenizer_decoder.
pub struct CodecDecoder {
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1dLayer,
    pre_transformer: TransformerStack,
    upsample: Vec<UpsampleStage>,
    tokenizer_decoder: TokenizerDecoder,
}

impl CodecDecoder {
    /// Return the device this decoder lives on.
    pub fn device(&self) -> &Device {
        self.pre_conv.device()
    }

    /// Build the decoder from the tokenizer weight store.
    ///
    /// The tokenizer weight store must contain all `decoder.*` keys
    /// (quantizer, pre_conv, pre_transformer, upsample, conv decoder blocks).
    pub fn from_weights(store: WeightStore) -> VoxResult<Self> {
        // ── Load codebook embeddings first (before ComponentWeights wraps store) ──
        let codebooks = load_decoder_codebooks(&store)?;
        let weights = ComponentWeights::new(store, "decoder");

        // ── 1. Quantizer ──
        // rvq_first: 1 layer (index 0 from codebooks)
        let rvq_first = Self::build_rvq(&weights, "quantizer.rvq_first", &codebooks[..1])?;
        // rvq_rest: 15 layers (indices 1..=15 from codebooks)
        let rvq_rest = Self::build_rvq(&weights, "quantizer.rvq_rest", &codebooks[1..=15])?;
        let quantizer = SplitResidualVectorQuantizer::from_weights(rvq_first, rvq_rest)?;

        // ── 2. Pre-conv: 512 → 1024, k=3, causal ──
        let pc_w = weights.require("pre_conv.conv.weight")?.clone();
        let pc_b = Some(weights.require("pre_conv.conv.bias")?.clone());
        let pre_conv = CausalConv1dLayer::from_weights(pc_w, pc_b, 1, 1, 1)?;

        // ── 3. Pre-transformer: 8× GQA + SwiGLU blocks ──
        let pre_transformer = Self::build_pre_transformer(&weights)?;

        // ── 4. Upsample: 2 stages, each (ConvTranspose1d stride2 + ConvNeXt) ──
        let mut upsample = Vec::with_capacity(UPSAMPLE_STAGES);
        for i in 0..UPSAMPLE_STAGES {
            upsample.push(UpsampleStage::from_weights(&weights, i)?);
        }

        // ── 5. TokenizerDecoder (conv stack, loads decoder.decoder.* keys) ──
        let tokenizer_decoder = TokenizerDecoder::from_weights(&weights)?;

        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            tokenizer_decoder,
        })
    }

    /// Decode 16 code index tensors into a mono waveform.
    ///
    /// # Arguments
    /// * `codes` — exactly 16 tensors, each `[batch, seq_len]` (u32 or i64).
    ///   Indices are 0-based into the 2048-entry codebooks.
    ///
    /// # Returns
    /// `[batch, 1, samples]` f32, clamped to `[-1, 1]`.
    ///
    /// The 12.5 Hz tokenizer decodes each codec frame to 1,920 samples at 24 kHz.
    pub fn decode(&self, codes: &[Tensor]) -> Result<Tensor> {
        // 1. SplitRVQ → [B, 512, T]
        let mut h = self.quantizer.decode(codes)?;

        // 2. Pre-conv: [B, 512, T] → [B, 1024, T]
        h = self.pre_conv.forward(&h)?;

        // 3. Pre-transformer: [B, 1024, T] → [B, T, 1024] → blocks → [B, T, 1024] → [B, 1024, T]
        h = h.transpose(1, 2)?;
        h = self.pre_transformer.forward(&h, None)?;
        h = h.transpose(1, 2)?;

        // 4. Upsample ×2: [B, 1024, T] → [B, 1024, T×4]
        for stage in &self.upsample {
            h = stage.forward(&h)?;
        }

        // 5. TokenizerDecoder: [B, 1024, T×4] → [B, 1, T×1920]
        h = self.tokenizer_decoder.forward_post_transformer(&h)?;
        Ok(h)
    }

    // ── Weight loading helpers ──

    fn build_rvq(
        weights: &ComponentWeights,
        prefix: &str,
        codebooks: &[Tensor],
    ) -> VoxResult<ResidualVectorQuantizer> {
        let ip_k = format!("{prefix}.input_proj.weight");
        let op_k = format!("{prefix}.output_proj.weight");
        let ip_w = weights.require(&ip_k)?.clone();
        let op_w = weights.require(&op_k)?.clone();
        ResidualVectorQuantizer::from_weights(ip_w, op_w, codebooks.to_vec())
    }

    fn build_pre_transformer(weights: &ComponentWeights) -> VoxResult<TransformerStack> {
        let pf = "pre_transformer";

        // Input projection: 1024 → 512
        let in_w = weights.require(&format!("{pf}.input_proj.weight"))?.clone();
        let in_b = Some(weights.require(&format!("{pf}.input_proj.bias"))?.clone());

        // Build 8 transformer blocks
        let mut blocks = Vec::with_capacity(PRE_TRANSFORMER_LAYERS);
        for layer in 0..PRE_TRANSFORMER_LAYERS {
            let lp = format!("{pf}.layers.{layer}");

            // GQA projections
            let q = weights
                .require(&format!("{lp}.self_attn.q_proj.weight"))?
                .clone();
            let k = weights
                .require(&format!("{lp}.self_attn.k_proj.weight"))?
                .clone();
            let v = weights
                .require(&format!("{lp}.self_attn.v_proj.weight"))?
                .clone();
            let o = weights
                .require(&format!("{lp}.self_attn.o_proj.weight"))?
                .clone();

            // SwiGLU MLP
            let gate = weights
                .require(&format!("{lp}.mlp.gate_proj.weight"))?
                .clone();
            let up = weights
                .require(&format!("{lp}.mlp.up_proj.weight"))?
                .clone();
            let down = weights
                .require(&format!("{lp}.mlp.down_proj.weight"))?
                .clone();

            // LayerNorms
            let ln1 = weights
                .require(&format!("{lp}.input_layernorm.weight"))?
                .clone();
            let ln2 = weights
                .require(&format!("{lp}.post_attention_layernorm.weight"))?
                .clone();

            // LayerScale
            let als = weights
                .require(&format!("{lp}.self_attn_layer_scale.scale"))?
                .clone();
            let mls = weights
                .require(&format!("{lp}.mlp_layer_scale.scale"))?
                .clone();

            let block = TransformerBlock::from_weights(
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                ln1,
                ln2,
                Some(als),
                Some(mls),
                None, // q_norm
                None, // k_norm
                PRE_TRANSFORMER_HEADS,
                PRE_TRANSFORMER_KV_HEADS,
                PRE_TRANSFORMER_EPS,
            )?;
            blocks.push(block);
        }

        // Final RMSNorm
        let norm_w = weights.require(&format!("{pf}.norm.weight"))?.clone();
        let norm = RmsNorm::from_weight(norm_w, PRE_TRANSFORMER_EPS);

        // Output projection: 512 → 1024
        let out_w = weights
            .require(&format!("{pf}.output_proj.weight"))?
            .clone();
        let out_b = Some(weights.require(&format!("{pf}.output_proj.bias"))?.clone());

        Ok(TransformerStack::from_blocks(
            blocks,
            Some(norm),
            Some((in_w, in_b)),
            Some((out_w, out_b)),
        ))
    }
}

/// High-level TTS inference pipeline.
///
/// Composes `SpeakerEncoder` (reference audio → embedding) and `CodecDecoder`
/// (codes → waveform). The acoustic model (Talker) that maps text + speaker
/// embedding to RVQ codes is assumed to be provided externally or in a future
/// `talker` module.
pub struct TtsPipeline {
    speaker_encoder: Option<SpeakerEncoder>,
    talker: Option<Talker>,
    codec_decoder: CodecDecoder,
}

impl TtsPipeline {
    /// Create pipeline from a tokenizer weight store.
    /// Speaker encoder and Talker are optional (attach later via `with_*`).
    pub fn from_tokenizer_weights(store: WeightStore) -> VoxResult<Self> {
        let codec_decoder = CodecDecoder::from_weights(store)?;
        Ok(Self {
            speaker_encoder: None,
            talker: None,
            codec_decoder,
        })
    }

    /// Attach a pre-loaded speaker encoder.
    pub fn with_speaker_encoder(mut self, enc: SpeakerEncoder) -> Self {
        self.speaker_encoder = Some(enc);
        self
    }

    /// Attach a pre-loaded Talker (acoustic model).
    pub fn with_talker(mut self, talker: Talker) -> Self {
        self.talker = Some(talker);
        self
    }

    pub fn talker(&self) -> Option<&Talker> {
        self.talker.as_ref()
    }

    /// Extract speaker embedding from reference mel spectrogram.
    /// Returns `None` if no speaker encoder is attached.
    pub fn extract_speaker(&self, mel: &Tensor) -> candle_core::Result<Option<Tensor>> {
        match &self.speaker_encoder {
            Some(enc) => enc.forward(mel).map(Some),
            None => Ok(None),
        }
    }

    /// Decode RVQ codes to waveform.
    pub fn decode_codes(&self, codes: &[Tensor]) -> VoxResult<Tensor> {
        self.codec_decoder
            .decode(codes)
            .map_err(|e| VoxError::Inference(e.to_string()))
    }

    pub fn decode_frame_codes(&self, frames: &[[u16; 16]]) -> VoxResult<Tensor> {
        if frames.is_empty() {
            return Err(VoxError::Inference("no codec frames to decode".into()));
        }

        let num_frames = frames.len();
        let mut code_seqs: Vec<Vec<u16>> =
            (0..16).map(|_| Vec::with_capacity(num_frames)).collect();
        for frame in frames {
            for (level, &code) in frame.iter().enumerate() {
                code_seqs[level].push(code);
            }
        }

        let device = self.codec_decoder.device().clone();
        let mut code_tensors = Vec::with_capacity(16);
        for seq in &code_seqs {
            let u32_seq: Vec<u32> = seq.iter().map(|&c| c as u32).collect();
            code_tensors.push(Tensor::from_vec(u32_seq, (1, num_frames), &device)?);
        }

        self.codec_decoder
            .decode(&code_tensors)
            .map_err(|e| VoxError::Inference(format!("CodecDecoder decode: {e}")))
    }

    /// High-level synthesis entry point.
    ///
    /// 1. Runs Talker to predict RVQ codes from input phone tokens
    /// 2. Decodes codes to waveform via CodecDecoder
    ///
    /// # Arguments
    /// * `phone_tokens` — phone/semantic token IDs (BOS prepended automatically by tokenizer)
    /// * `max_frames` — maximum number of code frames to generate (default: 512)
    ///
    /// # Returns
    /// `[1, 1, samples]` f32 waveform tensor clamped to `[-1, 1]`.
    pub fn synthesize(&self, phone_tokens: &[u32], max_frames: usize) -> VoxResult<Tensor> {
        let talker = self
            .talker
            .as_ref()
            .ok_or_else(|| VoxError::Inference("Talker not attached. Use with_talker().".into()))?;

        // 1. Talker autoregressive generation: Vec<[q0..q15; 16]>
        let frames = talker.generate(phone_tokens, max_frames)?;
        if frames.is_empty() {
            return Err(VoxError::Inference("Talker generated zero frames".into()));
        }
        self.decode_frame_codes(&frames)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_gelu_erf_basic() {
        let device = cpu();
        // At x=0, GELU should be 0
        let x = Tensor::zeros((1, 3), DType::F32, &device).unwrap();
        let y = gelu_erf(&x).unwrap();
        let vals: Vec<f32> = y.to_vec2().unwrap()[0].clone();
        for v in &vals {
            assert!(v.abs() < 1e-6, "GELU(0) ≈ 0, got {v}");
        }

        // At large positive values, GELU(x) ≈ x
        let x = Tensor::new(&[[10.0f32, 20.0]], &device).unwrap();
        let y = gelu_erf(&x).unwrap();
        let vals: Vec<f32> = y.to_vec2().unwrap()[0].clone();
        assert!(
            (vals[0] - 10.0).abs() < 0.01,
            "GELU(10) ≈ 10, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 20.0).abs() < 0.01,
            "GELU(20) ≈ 20, got {}",
            vals[1]
        );
    }

    fn make_convnext_block(ch: usize, device: &Device) -> ConvNeXtBlock {
        let nw = Tensor::ones(ch, DType::F32, device).unwrap();
        let nb = Tensor::zeros(ch, DType::F32, device).unwrap();
        let layer_norm = candle_nn::LayerNorm::new(nw, nb, 1e-6);
        let dw_w = Tensor::zeros((ch, 1, 3), DType::F32, device).unwrap();
        let dw_b = Some(Tensor::zeros(ch, DType::F32, device).unwrap());
        let dwconv = CausalConv1dLayer::from_weights(dw_w, dw_b, 1, ch, 1).unwrap();
        let p1w = Tensor::zeros((ch * 2, ch), DType::F32, device).unwrap();
        let p1b = Some(Tensor::zeros(ch * 2, DType::F32, device).unwrap());
        let p2w = Tensor::zeros((ch, ch * 2), DType::F32, device).unwrap();
        let p2b = Some(Tensor::zeros(ch, DType::F32, device).unwrap());
        let gamma = Tensor::full(1.0f32, ch, device).unwrap();

        ConvNeXtBlock {
            layer_norm,
            dwconv,
            pwconv1_weight: p1w,
            pwconv1_bias: p1b,
            pwconv2_weight: p2w,
            pwconv2_bias: p2b,
            gamma,
        }
    }

    #[test]
    fn test_convnext_block_preserves_shape() {
        let device = cpu();
        let ch = 4usize;
        let block = make_convnext_block(ch, &device);

        let x = Tensor::zeros((1, ch, 8), DType::F32, &device).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(
            y.dims(),
            &[1, ch, 8],
            "ConvNeXtBlock must preserve [B, C, T]"
        );
    }

    #[test]
    fn test_upsample_stage_doubles_time() {
        let device = cpu();
        let ch = 4usize;
        let seq = 4usize;

        // ConvTranspose1d: [ch, ch, 2], stride 2
        let ctw = Tensor::zeros((ch, ch, 2), DType::F32, &device).unwrap();
        let ctb = Some(Tensor::zeros(ch, DType::F32, &device).unwrap());
        let conv_transpose = CausalConvTranspose1dLayer::from_weights(ctw, ctb, 2).unwrap();

        let convnext = make_convnext_block(ch, &device);
        let stage = UpsampleStage {
            conv_transpose,
            convnext,
        };

        let x = Tensor::zeros((1, ch, seq), DType::F32, &device).unwrap();
        let y = stage.forward(&x).unwrap();
        // Transpose stride 2 doubles time, then ConvNeXt preserves shape
        assert_eq!(y.dims(), &[1, ch, seq * 2], "UpsampleStage must double T");
    }

    #[test]
    fn test_codec_decoder_from_weights_quantizer_only() {
        // Build the quantizer portion from minimal constructed weights
        // To verify the build_rvq helper and quantizer loading work.
        let device = cpu();
        let proj_dim = 4usize;
        let hidden_dim = 8usize;
        let vocab = 5usize;
        let batch = 1usize;
        let seq = 3usize;

        // Setup
        let ip_w = Tensor::zeros((proj_dim, hidden_dim, 1), DType::F32, &device).unwrap();
        let op_w = Tensor::zeros((hidden_dim, proj_dim, 1), DType::F32, &device).unwrap();
        let cb = Tensor::zeros((vocab, proj_dim), DType::F32, &device).unwrap();

        let rvq_first =
            ResidualVectorQuantizer::from_weights(ip_w.clone(), op_w.clone(), vec![cb.clone()])
                .unwrap();
        let rvq_rest = ResidualVectorQuantizer::from_weights(ip_w, op_w, vec![cb; 15]).unwrap();
        let quantizer = SplitResidualVectorQuantizer::from_weights(rvq_first, rvq_rest).unwrap();

        // Decode dummy codes
        let mut codes = Vec::with_capacity(16);
        for _ in 0..16 {
            codes.push(Tensor::zeros((batch, seq), DType::U32, &device).unwrap());
        }

        let out = quantizer.decode(&codes).unwrap();
        assert_eq!(
            out.dims(),
            &[batch, hidden_dim, seq],
            "Quantizer output shape"
        );
    }

    #[test]
    fn test_full_pipeline_shape_propagation() {
        // Verify shape propagation through the entire CodecDecoder pipeline
        // by constructing with minimal weights.
        //
        // Pipeline: quantizer (512 hidden) → pre_conv (1024) → pre_transformer (hidden=512)
        //         → upsample×2 → decoder (→ 1 ch)
        //
        // Input: 16 codes × [1, 2] → quantizer → [1, 512, 2]
        // pre_conv (k=3): [1, 512, 2] → [1, 1024, 2]  (causal pad left 2, stride 1)
        // pre_transformer: input_proj [512,1024], hidden=512, output_proj [1024,512]
        //   → [1, 2, 1024] → ip → [1, 2, 512] → 8 blocks → norm → op → [1, 2, 1024] → [1, 1024, 2]
        // upsample 0: [1, 1024, 2] → [1, 1024, 4]  (stride 2, ConvNeXt preserves)
        // upsample 1: [1, 1024, 4] → [1, 1024, 8]  (stride 2)
        // decoder: pre_conv [1536,1024,7] → [1, 1536, 8]
        //          block 1: [1536] → transpose [768] stride8 → [1, 768, 64]
        //          block 2: [768] → transpose [384] stride5 → [1, 384, 320]
        //          block 3: [384] → transpose [192] stride4 → [1, 192, 1280]
        //          block 4: [192] → transpose [96] stride3 → [1, 96, 3840]
        //          snake + final [1,96,7] → [1, 1, 3840]
        //
        // This test constructs a reduced decoder pipeline layer by layer and verifies
        // shape propagation without needing actual weight files.

        let device = Device::Cpu;
        let batch = 1usize;
        let seq = 2usize;

        // ── Build CustomCodecDecoder directly (skip from_weights) ──
        // This is a self-contained pipeline build with matched dimensions
        // matching the Qwen3-TTS tokenizer decoder architecture.

        use crate::conv_decoder::{DecoderBlock, ResidualUnit};

        // 1. Quantizer
        let proj_dim = 256usize;
        let hidden_dim = 512usize;
        let vocab = 2048usize;

        let ip_w = Tensor::randn(0f32, 0.01, (proj_dim, hidden_dim, 1), &device).unwrap();
        let op_w = Tensor::randn(0f32, 0.01, (hidden_dim, proj_dim, 1), &device).unwrap();
        let cb = Tensor::randn(0f32, 0.01, (vocab, proj_dim), &device).unwrap();

        let rvq_first =
            ResidualVectorQuantizer::from_weights(ip_w.clone(), op_w.clone(), vec![cb.clone()])
                .unwrap();
        let rvq_rest = ResidualVectorQuantizer::from_weights(ip_w, op_w, vec![cb; 15]).unwrap();
        let quantizer = SplitResidualVectorQuantizer::from_weights(rvq_first, rvq_rest).unwrap();

        // 2. Pre-conv
        let pc_w = Tensor::randn(0f32, 0.01, (1024, 512, 3), &device).unwrap();
        let pc_b = Some(Tensor::zeros(1024, DType::F32, &device).unwrap());
        let pre_conv = CausalConv1dLayer::from_weights(pc_w, pc_b, 1, 1, 1).unwrap();

        // 3. Pre-transformer
        // input_proj: 1024 → 512
        let in_w = Tensor::randn(0f32, 0.01, (512, 1024), &device).unwrap();
        let in_b = Some(Tensor::zeros(512, DType::F32, &device).unwrap());

        let mut blocks = Vec::with_capacity(2); // 2 blocks for speed
        for _ in 0..2 {
            let q = Tensor::randn(0f32, 0.01, (1024, 512), &device).unwrap();
            let k = Tensor::randn(0f32, 0.01, (1024, 512), &device).unwrap();
            let v = Tensor::randn(0f32, 0.01, (1024, 512), &device).unwrap();
            let o = Tensor::randn(0f32, 0.01, (512, 1024), &device).unwrap();
            let gate = Tensor::randn(0f32, 0.01, (1024, 512), &device).unwrap();
            let up = Tensor::randn(0f32, 0.01, (1024, 512), &device).unwrap();
            let down = Tensor::randn(0f32, 0.01, (512, 1024), &device).unwrap();
            let ln1 = Tensor::ones(512, DType::F32, &device).unwrap();
            let ln2 = Tensor::ones(512, DType::F32, &device).unwrap();
            let als = Tensor::full(0.01f32, 512, &device).unwrap();
            let mls = Tensor::full(0.01f32, 512, &device).unwrap();

            let block = TransformerBlock::from_weights(
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                ln1,
                ln2,
                Some(als),
                Some(mls),
                None,
                None,
                8,
                8,
                1e-5,
            )
            .unwrap();
            blocks.push(block);
        }

        let norm_w = Tensor::ones(512, DType::F32, &device).unwrap();
        let norm = RmsNorm::from_weight(norm_w, 1e-5);
        let out_w = Tensor::randn(0f32, 0.01, (1024, 512), &device).unwrap();
        let out_b = Some(Tensor::zeros(1024, DType::F32, &device).unwrap());

        let pre_transformer = TransformerStack::from_blocks(
            blocks,
            Some(norm),
            Some((in_w, in_b)),
            Some((out_w, out_b)),
        );

        // 4. Upsample (1 stage for speed; 2 stages verified in test_upsample_stage_doubles_time)
        let ctw = Tensor::randn(0f32, 0.01, (1024, 1024, 2), &device).unwrap();
        let ctb = Some(Tensor::zeros(1024, DType::F32, &device).unwrap());
        let ct = CausalConvTranspose1dLayer::from_weights(ctw, ctb, 2).unwrap();
        let convnext = {
            let nw = Tensor::ones(1024, DType::F32, &device).unwrap();
            let nb = Tensor::zeros(1024, DType::F32, &device).unwrap();
            let layer_norm = candle_nn::LayerNorm::new(nw, nb, 1e-6);
            let dw_w = Tensor::zeros((1024, 1, 3), DType::F32, &device).unwrap();
            let dw_b = Some(Tensor::zeros(1024, DType::F32, &device).unwrap());
            let dwconv = CausalConv1dLayer::from_weights(dw_w, dw_b, 1, 1024, 1).unwrap();
            let p1w = Tensor::zeros((4096, 1024), DType::F32, &device).unwrap();
            let p1b = Some(Tensor::zeros(4096, DType::F32, &device).unwrap());
            let p2w = Tensor::zeros((1024, 4096), DType::F32, &device).unwrap();
            let p2b = Some(Tensor::zeros(1024, DType::F32, &device).unwrap());
            let gamma = Tensor::full(1.0f32, 1024, &device).unwrap();
            ConvNeXtBlock {
                layer_norm,
                dwconv,
                pwconv1_weight: p1w,
                pwconv1_bias: p1b,
                pwconv2_weight: p2w,
                pwconv2_bias: p2b,
                gamma,
            }
        };
        let upsample = vec![UpsampleStage {
            conv_transpose: ct,
            convnext,
        }];

        // 5. TokenizerDecoder (minimal 1-block version)
        let dec_pre_w = Tensor::randn(0f32, 0.01, (1536, 1024, 7), &device).unwrap();
        let dec_pre_b = Some(Tensor::zeros(1536, DType::F32, &device).unwrap());
        let dec_pre = CausalConv1dLayer::from_weights(dec_pre_w, dec_pre_b, 1, 1, 1).unwrap();

        // Build 1 decoder block (in_ch=1536, out_ch=768)
        let block_ch_in = 1536usize;
        let block_ch_out = 768usize;
        let sa = Tensor::ones(block_ch_in, DType::F32, &device).unwrap();
        let sb = Tensor::ones(block_ch_in, DType::F32, &device).unwrap();
        let ctw = Tensor::zeros((block_ch_in, block_ch_out, 16), DType::F32, &device).unwrap();
        let ctb = Some(Tensor::zeros(block_ch_out, DType::F32, &device).unwrap());
        let cm = CausalConvTranspose1dLayer::from_weights(ctw, ctb, 8).unwrap();
        let mut ress = vec![];
        for &d in &[1, 3, 9] {
            let aa = Tensor::ones(block_ch_out, DType::F32, &device).unwrap();
            let ab = Tensor::ones(block_ch_out, DType::F32, &device).unwrap();
            let ba = Tensor::ones(block_ch_out, DType::F32, &device).unwrap();
            let bb = Tensor::ones(block_ch_out, DType::F32, &device).unwrap();
            let c1w = Tensor::zeros((block_ch_out, block_ch_out, 7), DType::F32, &device).unwrap();
            let c1b = Some(Tensor::zeros(block_ch_out, DType::F32, &device).unwrap());
            let c2w = Tensor::zeros((block_ch_out, block_ch_out, 1), DType::F32, &device).unwrap();
            let c2b = Some(Tensor::zeros(block_ch_out, DType::F32, &device).unwrap());
            ress.push(ResidualUnit::from_weights(aa, ab, ba, bb, c1w, c1b, c2w, c2b, d).unwrap());
        }
        let dec_block = DecoderBlock {
            snake_alpha: sa,
            snake_beta: sb,
            conv_transpose: cm,
            residuals: ress,
        };

        let fsa = Tensor::ones(block_ch_out, DType::F32, &device).unwrap();
        let fsb = Tensor::ones(block_ch_out, DType::F32, &device).unwrap();
        let fcw = Tensor::zeros((1, block_ch_out, 7), DType::F32, &device).unwrap();
        let fcb = Some(Tensor::zeros(1, DType::F32, &device).unwrap());
        let fc = CausalConv1dLayer::from_weights(fcw, fcb, 1, 1, 1).unwrap();

        let tokenizer_decoder = TokenizerDecoder {
            pre_conv: dec_pre,
            decoder_blocks: vec![dec_block],
            final_snake_alpha: fsa,
            final_snake_beta: fsb,
            final_conv: fc,
        };

        // ── Final pipeline ──
        let pipeline = CodecDecoder {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            tokenizer_decoder,
        };

        // Generate codes and run
        let mut codes: Vec<Tensor> = Vec::with_capacity(16);
        for _ in 0..16 {
            codes.push(Tensor::zeros((batch, seq), DType::U32, &device).unwrap());
        }

        let waveform = pipeline.decode(&codes).unwrap();

        // quantizer: [1, 512, 2]
        // pre_conv:  [1, 1024, 2] (k=3 causal pad: 2→2+2=4, conv1d p=0 s=1 → 4-3+1=2 → length=2)
        //           Wait — let me recalculate: causal_pad_left(x, 3) pads 2 zeros
        //           After pad: [1, 1024, 4], conv1d(k=3, p=0, s=1): output = 4-3+1=2 ✓
        // pre_transformer: [1, 2, 1024] → ip [512,1024] → [1,2,512] → 2 blocks → norm → op [1024,512] → [1,2,1024] → [1,1024,2]
        // upsample: [1, 1024, 2] → ConvTranspose stride2 → [1, 1024, 4] → ConvNeXt preserves
        // decoder: pre_conv [1536,1024,7]: [1,1024,4] → [1,1536,4]
        //          one block stride8: [1,1536,4] → [1,768,32]
        //          final conv: [1,768,32] → [1,1,32]
        assert_eq!(
            waveform.dims(),
            &[1, 1, 32],
            "waveform shape with 1 decoder block, seq=2"
        );

        // Verify clamping to [-1, 1]
        let flat: Vec<f32> = waveform.flatten_all().unwrap().to_vec1().unwrap();
        for v in flat {
            assert!(
                (-1.0..=1.0).contains(&v),
                "waveform sample {v} outside [-1, 1]"
            );
        }
    }
}
