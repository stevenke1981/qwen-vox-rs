//! Talker: the autoregressive acoustic model for Qwen3-TTS 12 Hz mode.
//!
//! Architecture (from weight inspection of 1.7B-CustomVoice):
//!
//! ┌─ Text Embedding ─────────────────────────────────────────────────┐
//! │  talker.model.embed_tokens.weight: [151936, 2048]               │
//! │  talker.model.text_projection: fc1→SiLU→fc2 (each [2048,2048]) │
//! └──────────────────────────────────────────────────────────────────┘
//!
//! ┌─ Audio Code Embedding ───────────────────────────────────────────┐
//! │  talker.model.codec_embedding.weight: [3072, 2048]              │
//! └──────────────────────────────────────────────────────────────────┘
//!
//! ┌─ Backbone ────────────────────────────────────────────────────────┐
//! │  28 × TransformerBlock(hidden=2048, 16Q/8KV GQA, head_dim=128, │
//! │         FFN=6144 SwiGLU, Q/K norms [128], LayerScale [2048])    │
//! │  + final RMSNorm [2048]                                         │
//! └──────────────────────────────────────────────────────────────────┘
//!
//! ┌─ Code Prediction ────────────────────────────────────────────────┐
//! │  codec_head [3072,2048] → argmax(0:2048) → q0                  │
//! │  small_to_mtp [1024,2048]+bias → hidden 2048→1024              │
//! │  5 × TransformerBlock(hidden=1024, 16Q/8KV, FFN=3072)          │
//! │  15 × lm_head [2048,1024] → argmax → q1..q15                   │
//! └──────────────────────────────────────────────────────────────────┘
//!
//! Autoregressive loop: text → backbone → predict 16 codes → append q0
//! embedding → repeat until EOS or max frames.

use crate::error::{VoxError, VoxResult};
use crate::transformer::{RmsNorm, TransformerBlock, TransformerStack};
use crate::weights::WeightStore;
use candle_core::{Device, Tensor};

// ── Architecture Constants ──────────────────────────────────────────────────────

const TEXT_VOCAB: usize = 151936;
const CODEC_VOCAB: usize = 3072;
const CODEBOOK_SIZE: usize = 2048;
const HIDDEN: usize = 2048;
const CP_HIDDEN: usize = 1024;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 8;
const NUM_BACKBONE_LAYERS: usize = 28;
const NUM_CP_LAYERS: usize = 5;
const NUM_RESIDUAL_CODES: usize = 15;
const EPS: f64 = 1e-5;

#[allow(dead_code)]
const HEAD_DIM: usize = 128;
#[allow(dead_code)]
const BACKBONE_FFN: usize = 6144;
#[allow(dead_code)]
const CP_FFN: usize = 3072;

// ── Special Token IDs ─────────────────────────────────────────────────────────

/// BOS token in the text embedding vocabulary.
#[allow(dead_code)]
const TTS_BOS: u32 = 151672;
/// EOS token in the text embedding vocabulary.
#[allow(dead_code)]
const TTS_EOS: u32 = 151673;
/// PAD token.
#[allow(dead_code)]
const TTS_PAD: u32 = 151671;
/// BOS token in the codec embedding vocabulary.
const _CODEC_BOS: u32 = 2149;
/// EOS token in the codec embedding vocabulary (stop generation).
const CODEC_EOS: u32 = 2150;
/// PAD token in the codec vocabulary.
const _CODEC_PAD: u32 = 2148;

// ── Talker ─────────────────────────────────────────────────────────────────────

/// The autoregressive acoustic model that predicts RVQ codes from phone tokens.
pub struct Talker {
    // ── Embeddings ──
    text_embedding: Tensor,  // [151936, 2048]
    codec_embedding: Tensor, // [3072, 2048]

    // ── Text projection (fc1 → SiLU → fc2) ──
    text_proj_fc1_w: Tensor, // [2048, 2048]
    text_proj_fc1_b: Tensor, // [2048]
    text_proj_fc2_w: Tensor, // [2048, 2048]
    text_proj_fc2_b: Tensor, // [2048]

    // ── Backbone ──
    backbone: TransformerStack, // 28 layers (GQA+QK norms+LayerScale)
    final_norm: RmsNorm,        // final RMSNorm after backbone

    // ── Codec head ──
    codec_head: Tensor, // [3072, 2048]

    // ── Code predictor ──
    small_to_mtp_w: Tensor,           // [1024, 2048]
    small_to_mtp_b: Tensor,           // [1024]
    cp_transformer: TransformerStack, // 5 layers (no QK norms, no LayerScale)
    cp_lm_heads: Vec<Tensor>,         // 15 × [2048, 1024]
}

impl Talker {
    /// Load the full Talker from a WeightStore containing all `talker.*` keys.
    ///
    /// # Errors
    /// Returns `VoxError::WeightLoad` if any required tensor is missing or
    /// shape-mismatched.
    pub fn from_store(store: &WeightStore) -> VoxResult<Self> {
        let device = store.device().clone();

        // ── 1. Embeddings ──
        let text_embedding = store.require("talker.model.embed_tokens.weight")?.clone();
        Self::check_shape(&text_embedding, &[TEXT_VOCAB, HIDDEN], "text_embedding")?;

        let codec_embedding = store
            .require("talker.model.codec_embedding.weight")?
            .clone();
        Self::check_shape(&codec_embedding, &[CODEC_VOCAB, HIDDEN], "codec_embedding")?;

        // ── 2. Text projection ──
        let text_proj_fc1_w = store
            .require("talker.model.text_projection.fc1.weight")?
            .clone();
        Self::check_shape(&text_proj_fc1_w, &[HIDDEN, HIDDEN], "text_proj.fc1.w")?;
        let text_proj_fc1_b = store
            .require("talker.model.text_projection.fc1.bias")?
            .clone();
        let text_proj_fc2_w = store
            .require("talker.model.text_projection.fc2.weight")?
            .clone();
        Self::check_shape(&text_proj_fc2_w, &[HIDDEN, HIDDEN], "text_proj.fc2.w")?;
        let text_proj_fc2_b = store
            .require("talker.model.text_projection.fc2.bias")?
            .clone();

        // ── 3. Backbone layers ──
        let mut backbone_blocks = Vec::with_capacity(NUM_BACKBONE_LAYERS);
        for i in 0..NUM_BACKBONE_LAYERS {
            let lp = format!("talker.model.layers.{i}");
            let block = Self::load_backbone_block(store, &lp, &device)?;
            backbone_blocks.push(block);
        }

        // Final RMSNorm
        let norm_w = store.require("talker.model.norm.weight")?.clone();
        let final_norm = RmsNorm::from_weight(norm_w, EPS);

        // Backbone stack: no input/output projections, no final norm (handled separately)
        let backbone = TransformerStack::from_blocks(backbone_blocks, None, None, None);

        // ── 4. Codec head ──
        let codec_head = store.require("talker.codec_head.weight")?.clone();
        Self::check_shape(&codec_head, &[CODEC_VOCAB, HIDDEN], "codec_head")?;

        // ── 5. Code predictor ──
        let smtp_w = store
            .require("talker.code_predictor.small_to_mtp_projection.weight")?
            .clone();
        Self::check_shape(&smtp_w, &[CP_HIDDEN, HIDDEN], "small_to_mtp.w")?;
        let smtp_b = store
            .require("talker.code_predictor.small_to_mtp_projection.bias")?
            .clone();

        let mut cp_blocks = Vec::with_capacity(NUM_CP_LAYERS);
        for i in 0..NUM_CP_LAYERS {
            let lp = format!("talker.code_predictor.model.layers.{i}");
            let block = Self::load_code_predictor_block(store, &lp, &device)?;
            cp_blocks.push(block);
        }

        // Code predictor final RMSNorm
        let cp_norm_w = store
            .require("talker.code_predictor.model.norm.weight")?
            .clone();
        let cp_norm = RmsNorm::from_weight(cp_norm_w, EPS);

        let cp_transformer = TransformerStack::from_blocks(cp_blocks, Some(cp_norm), None, None);

        // 15 lm_heads
        let mut cp_lm_heads = Vec::with_capacity(NUM_RESIDUAL_CODES);
        for i in 0..NUM_RESIDUAL_CODES {
            let key = format!("talker.code_predictor.lm_heads.{i}.weight");
            let head = store.require(&key)?.clone();
            Self::check_shape(&head, &[CODEBOOK_SIZE, CP_HIDDEN], &key)?;
            cp_lm_heads.push(head);
        }

        Ok(Self {
            text_embedding,
            codec_embedding,
            text_proj_fc1_w,
            text_proj_fc1_b,
            text_proj_fc2_w,
            text_proj_fc2_b,
            backbone,
            final_norm,
            codec_head,
            small_to_mtp_w: smtp_w,
            small_to_mtp_b: smtp_b,
            cp_transformer,
            cp_lm_heads,
        })
    }

    // ── Text Processing ────────────────────────────────────────────────────────

    /// Encode phone tokens into the backbone's hidden space.
    ///
    /// Steps: token IDs → text_embedding → fc1 → SiLU → fc2
    ///
    /// Returns `[B, T, 2048]`.
    pub fn encode_text(&self, tokens: &[u32]) -> VoxResult<Tensor> {
        let device = self.text_embedding.device();

        // Gather embeddings: [1, T, 2048]
        let token_t = Tensor::new(tokens, device)?; // [T]
        let emb = self.text_embedding.index_select(&token_t, 0)?; // [T, 2048]
        let emb = emb.unsqueeze(0)?; // [1, T, 2048]

        // Text projection: fc1 → SiLU → fc2
        let h = emb;
        let h = h.broadcast_matmul(&self.text_proj_fc1_w.t()?)?;
        let h = h.broadcast_add(&self.text_proj_fc1_b)?;
        let h = h.silu()?;
        let h = h.broadcast_matmul(&self.text_proj_fc2_w.t()?)?;
        let h = h.broadcast_add(&self.text_proj_fc2_b)?;

        // Sanity: [B, T, 2048]
        let _dims = h
            .dims3()
            .map_err(|e| VoxError::Inference(format!("encode_text output shape: {e}")))?;
        Ok(h)
    }

    /// Embed a single q0 code token for the autoregressive loop.
    ///
    /// Looks up the codec_embedding table, returns `[1, 1, 2048]`.
    fn embed_code_token(&self, token: u32) -> VoxResult<Tensor> {
        let device = self.codec_embedding.device();
        let t = Tensor::new(&[token], device)?; // [1]
        let emb = self.codec_embedding.index_select(&t, 0)?; // [1, 2048]
        let emb = emb.unsqueeze(0)?; // [1, 1, 2048]
        Ok(emb)
    }

    // ── Backbone Forward ───────────────────────────────────────────────────────

    /// Run the full backbone: no input/output projections, just blocks + final_norm.
    ///
    /// Input: `[B, T, 2048]` (concatenated text + code embeddings)
    /// Output: `[B, T, 2048]`
    fn backbone_forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.backbone.forward(x, None)?;
        self.final_norm.forward(&h)
    }

    // ── Code Prediction ────────────────────────────────────────────────────────

    /// Predict one frame of 16 codes (q0..q15) from a backbone hidden state.
    ///
    /// `hidden`: `[B, T, 2048]` — entire sequence hidden states.
    /// Uses `hidden[last]` to predict the next frame.
    ///
    /// Returns `(q0, residual_codes)` where residual_codes has 15 entries.
    fn predict_codes(&self, hidden: &Tensor) -> VoxResult<(u16, Vec<u16>)> {
        // Last position: [B, 2048]
        let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?;

        // ── q0 via codec_head ──
        // last: [B, 2048], codec_head: [3072, 2048]
        let q0_logits = last.broadcast_matmul(&self.codec_head.t()?)?; // [B, 3072]
        let q0_logits = q0_logits.narrow(1, 0, CODEBOOK_SIZE)?; // [B, 2048]
        let q0_idx = q0_logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()? as u16;

        // ── q1..q15 via code predictor ──
        // small_to_mtp: project from 2048 → 1024
        let cp_h = last.broadcast_matmul(&self.small_to_mtp_w.t()?)?; // [B, 1024]
        let cp_h = cp_h.broadcast_add(&self.small_to_mtp_b)?;

        // Unsqueeze to [B, 1, 1024] for transformer
        let cp_h = cp_h.unsqueeze(1)?;

        // 5-layer code predictor transformer
        let cp_out = self.cp_transformer.forward(&cp_h, None)?; // [B, 1, 1024]
        let cp_last = cp_out.squeeze(1)?; // [B, 1024]

        // 15 lm_heads → argmax
        let mut codes = Vec::with_capacity(NUM_RESIDUAL_CODES);
        for head in &self.cp_lm_heads {
            // head: [2048, 1024]; cp_last: [B, 1024]
            let logits = cp_last.broadcast_matmul(&head.t()?)?; // [B, 2048]
            let idx = logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()? as u16;
            codes.push(idx);
        }

        Ok((q0_idx, codes))
    }

    // ── Autoregressive Generation ──────────────────────────────────────────────

    /// Generate a sequence of RVQ code frames from phone tokens.
    ///
    /// # Arguments
    /// * `phone_tokens` — phone/semantic token IDs (BOS should already be prepended)
    /// * `max_frames` — maximum number of code frames to generate
    ///
    /// # Returns
    /// `Vec<[u16; 16]>` — sequence of (q0, q1, ..., q15) code frames.
    pub fn generate(&self, phone_tokens: &[u32], max_frames: usize) -> VoxResult<Vec<[u16; 16]>> {
        if phone_tokens.is_empty() {
            return Err(VoxError::Inference("phone_tokens must not be empty".into()));
        }

        // Encode text: [1, T, 2048]
        let text_hidden = self.encode_text(phone_tokens)?;

        // The autoregressive input starts as just the text hidden states.
        // As we generate, we append q0 code embeddings.
        let mut input_hidden = text_hidden;
        let mut frames: Vec<[u16; 16]> = Vec::with_capacity(max_frames.min(512));

        for _step in 0..max_frames {
            // Backbone forward: [1, T, 2048] → [1, T, 2048]
            let hidden = self
                .backbone_forward(&input_hidden)
                .map_err(|e| VoxError::Inference(format!("backbone forward: {e}")))?;

            // Predict codes from last position
            let (q0, residual_codes) = self.predict_codes(&hidden)?;

            // Assemble the 16-code frame
            let mut frame = [0u16; 16];
            frame[0] = q0;
            for (i, &rc) in residual_codes.iter().enumerate() {
                frame[i + 1] = rc;
            }
            frames.push(frame);

            // Check for EOS
            if q0 as u32 == CODEC_EOS {
                break;
            }

            // Embed q0 token and append to input
            let code_emb = self.embed_code_token(q0 as u32)?; // [1, 1, 2048]
            input_hidden = Tensor::cat(&[&input_hidden, &code_emb], 1)
                .map_err(|e| VoxError::Inference(format!("cat input_hidden + code_emb: {e}")))?;
        }

        Ok(frames)
    }

    // ── Weight Loading Helpers ─────────────────────────────────────────────────

    /// Load one backbone transformer block (28 total).
    fn load_backbone_block(
        store: &WeightStore,
        prefix: &str,
        _device: &Device,
    ) -> VoxResult<TransformerBlock> {
        let q = store
            .require(&format!("{prefix}.self_attn.q_proj.weight"))?
            .clone();
        let k = store
            .require(&format!("{prefix}.self_attn.k_proj.weight"))?
            .clone();
        let v = store
            .require(&format!("{prefix}.self_attn.v_proj.weight"))?
            .clone();
        let o = store
            .require(&format!("{prefix}.self_attn.o_proj.weight"))?
            .clone();

        let gate = store
            .require(&format!("{prefix}.mlp.gate_proj.weight"))?
            .clone();
        let up = store
            .require(&format!("{prefix}.mlp.up_proj.weight"))?
            .clone();
        let down = store
            .require(&format!("{prefix}.mlp.down_proj.weight"))?
            .clone();

        let ln1 = store
            .require(&format!("{prefix}.input_layernorm.weight"))?
            .clone();
        let ln2 = store
            .require(&format!("{prefix}.post_attention_layernorm.weight"))?
            .clone();

        // Q/K per-head norms
        let q_norm = Some(
            store
                .require(&format!("{prefix}.self_attn.q_norm.weight"))?
                .clone(),
        );
        let k_norm = Some(
            store
                .require(&format!("{prefix}.self_attn.k_norm.weight"))?
                .clone(),
        );

        // LayerScale
        let als = Some(
            store
                .require(&format!("{prefix}.self_attn_layer_scale.scale"))?
                .clone(),
        );
        let mls = Some(
            store
                .require(&format!("{prefix}.mlp_layer_scale.scale"))?
                .clone(),
        );

        TransformerBlock::from_weights(
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
            ln1,
            ln2,
            als,
            mls,
            q_norm,
            k_norm,
            NUM_HEADS,
            NUM_KV_HEADS,
            EPS,
        )
    }

    /// Load one code predictor transformer block (5 total — no QK norms, no LayerScale).
    fn load_code_predictor_block(
        store: &WeightStore,
        prefix: &str,
        _device: &Device,
    ) -> VoxResult<TransformerBlock> {
        let q = store
            .require(&format!("{prefix}.self_attn.q_proj.weight"))?
            .clone();
        let k = store
            .require(&format!("{prefix}.self_attn.k_proj.weight"))?
            .clone();
        let v = store
            .require(&format!("{prefix}.self_attn.v_proj.weight"))?
            .clone();
        let o = store
            .require(&format!("{prefix}.self_attn.o_proj.weight"))?
            .clone();

        let gate = store
            .require(&format!("{prefix}.mlp.gate_proj.weight"))?
            .clone();
        let up = store
            .require(&format!("{prefix}.mlp.up_proj.weight"))?
            .clone();
        let down = store
            .require(&format!("{prefix}.mlp.down_proj.weight"))?
            .clone();

        let ln1 = store
            .require(&format!("{prefix}.input_layernorm.weight"))?
            .clone();
        let ln2 = store
            .require(&format!("{prefix}.post_attention_layernorm.weight"))?
            .clone();

        // No Q/K norms, no LayerScale in code predictor
        let q_norm: Option<Tensor> = None;
        let k_norm: Option<Tensor> = None;
        let als: Option<Tensor> = None;
        let mls: Option<Tensor> = None;

        TransformerBlock::from_weights(
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
            ln1,
            ln2,
            als,
            mls,
            q_norm,
            k_norm,
            NUM_HEADS,
            NUM_KV_HEADS,
            EPS,
        )
    }

    /// Validate tensor shape.
    fn check_shape(t: &Tensor, expected: &[usize], _name: &str) -> VoxResult<()> {
        let actual: Vec<usize> = t.dims().to_vec();
        if actual.len() != expected.len() {
            return Err(VoxError::ShapeMismatch {
                expected: expected.to_vec(),
                actual,
            });
        }
        for (a, e) in actual.iter().zip(expected.iter()) {
            if a != e {
                return Err(VoxError::ShapeMismatch {
                    expected: expected.to_vec(),
                    actual: t.dims().to_vec(),
                });
            }
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::WeightStore;
    use candle_core::{DType, Device};

    fn cpu() -> Device {
        Device::Cpu
    }

    // ── Structure loading tests ──

    #[test]
    fn test_talker_construction_fails_without_weights() {
        let store = WeightStore::new(Device::Cpu);
        let result = Talker::from_store(&store);
        assert!(result.is_err());
    }

    /// Build a minimal weight store that contains the expected 404 tensors
    /// with correct shapes, then verify that Talker::from_store succeeds.
    #[test]
    fn test_talker_from_store_with_minimal_weights() {
        let device = cpu();

        let mut store = WeightStore::new(device.clone());

        // Embeddings
        store.insert_tensor(
            "talker.model.embed_tokens.weight",
            Tensor::zeros((TEXT_VOCAB, HIDDEN), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.codec_embedding.weight",
            Tensor::zeros((CODEC_VOCAB, HIDDEN), DType::F32, &device).unwrap(),
        );

        // Text projection
        store.insert_tensor(
            "talker.model.text_projection.fc1.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.text_projection.fc1.bias",
            Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.text_projection.fc2.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.text_projection.fc2.bias",
            Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
        );

        // Backbone 28 layers
        for i in 0..NUM_BACKBONE_LAYERS {
            let prefix = format!("talker.model.layers.{i}");
            store.insert_tensor(
                format!("{prefix}.self_attn.q_proj.weight"),
                Tensor::zeros((HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.k_proj.weight"),
                Tensor::zeros((HIDDEN / 2, HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.v_proj.weight"),
                Tensor::zeros((HIDDEN / 2, HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.o_proj.weight"),
                Tensor::zeros((HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.mlp.gate_proj.weight"),
                Tensor::zeros((BACKBONE_FFN, HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.mlp.up_proj.weight"),
                Tensor::zeros((BACKBONE_FFN, HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.mlp.down_proj.weight"),
                Tensor::zeros((HIDDEN, BACKBONE_FFN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.q_norm.weight"),
                Tensor::zeros(HEAD_DIM, DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.k_norm.weight"),
                Tensor::zeros(HEAD_DIM, DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn_layer_scale.scale"),
                Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.mlp_layer_scale.scale"),
                Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
            );
        }

        // Backbone final norm
        store.insert_tensor(
            "talker.model.norm.weight",
            Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
        );

        // Codec head
        store.insert_tensor(
            "talker.codec_head.weight",
            Tensor::zeros((CODEC_VOCAB, HIDDEN), DType::F32, &device).unwrap(),
        );

        // Code predictor
        store.insert_tensor(
            "talker.code_predictor.small_to_mtp_projection.weight",
            Tensor::zeros((CP_HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.code_predictor.small_to_mtp_projection.bias",
            Tensor::zeros(CP_HIDDEN, DType::F32, &device).unwrap(),
        );

        for i in 0..NUM_CP_LAYERS {
            let prefix = format!("talker.code_predictor.model.layers.{i}");
            store.insert_tensor(
                format!("{prefix}.self_attn.q_proj.weight"),
                Tensor::zeros((HIDDEN, CP_HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.k_proj.weight"),
                Tensor::zeros((HIDDEN / 2, CP_HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.v_proj.weight"),
                Tensor::zeros((HIDDEN / 2, CP_HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.self_attn.o_proj.weight"),
                Tensor::zeros((CP_HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.mlp.gate_proj.weight"),
                Tensor::zeros((CP_FFN, CP_HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.mlp.up_proj.weight"),
                Tensor::zeros((CP_FFN, CP_HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.mlp.down_proj.weight"),
                Tensor::zeros((CP_HIDDEN, CP_FFN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::zeros(CP_HIDDEN, DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::zeros(CP_HIDDEN, DType::F32, &device).unwrap(),
            );
        }

        store.insert_tensor(
            "talker.code_predictor.model.norm.weight",
            Tensor::zeros(CP_HIDDEN, DType::F32, &device).unwrap(),
        );

        for i in 0..NUM_RESIDUAL_CODES {
            store.insert_tensor(
                format!("talker.code_predictor.lm_heads.{i}.weight"),
                Tensor::zeros((CODEBOOK_SIZE, CP_HIDDEN), DType::F32, &device).unwrap(),
            );
        }

        // ── Now test ──
        let talker = Talker::from_store(&store).unwrap_or_else(|e| {
            panic!("Talker::from_store should succeed with all required weights: {e}")
        });

        // Quick verify we can encode text
        let tokens = vec![TTS_BOS, 100, 200, TTS_EOS];
        let encoded = talker.encode_text(&tokens).unwrap();
        assert_eq!(
            encoded.dims(),
            &[1, tokens.len(), HIDDEN],
            "encode_text should produce [1, T, 2048]"
        );
    }

    // ── Text encoding tests ──

    #[test]
    fn test_encode_text_shape() {
        let device = cpu();
        let talker = build_minimal_talker(&device);

        let tokens = vec![TTS_BOS, 42, 99, TTS_EOS];
        let encoded = talker.encode_text(&tokens).unwrap();
        assert_eq!(encoded.dims(), &[1, 4, HIDDEN]);
    }

    #[test]
    fn test_encode_text_empty_succeeds() {
        let device = cpu();
        let talker = build_minimal_talker(&device);

        // Empty token list: index_select returns [0, 2048] → unsqueeze → [1, 0, 2048]
        let result = talker.encode_text(&[]).unwrap();
        assert_eq!(
            result.dims(),
            &[1, 0, HIDDEN],
            "empty tokens should give zero-length sequence"
        );
    }

    // ── Backbone / code prediction tests ──

    #[test]
    fn test_backbone_forward_preserves_shape() {
        let device = cpu();
        let talker = build_minimal_talker(&device);

        let x = Tensor::zeros((1, 4, HIDDEN), DType::F32, &device).unwrap();
        let y = talker.backbone_forward(&x).unwrap();
        assert_eq!(
            y.dims(),
            &[1, 4, HIDDEN],
            "backbone must preserve [B, T, D]"
        );
    }

    #[test]
    fn test_predict_codes_returns_16_codes() {
        let device = cpu();
        let talker = build_minimal_talker(&device);

        // Create dummy hidden state [1, 5, 2048]
        let hidden = Tensor::zeros((1, 5, HIDDEN), DType::F32, &device).unwrap();
        let (q0, residual) = talker.predict_codes(&hidden).unwrap();

        // With all-zero weights, argmax returns 0
        assert_eq!(q0, 0, "q0 should be 0 with zero-initialized weights");
        assert_eq!(residual.len(), NUM_RESIDUAL_CODES);
        for &rc in &residual {
            assert_eq!(rc, 0, "residual codes should be 0 with zero weights");
        }
    }

    #[test]
    #[ignore = "CPU-bound: 28-layer transformer autoregressive loop takes >5 min on CPU"]
    fn test_generate_returns_frames() {
        let device = cpu();
        let talker = build_minimal_talker(&device);

        let tokens = vec![TTS_BOS, 42, TTS_EOS];
        let frames = talker.generate(&tokens, 5).unwrap();

        // Should produce at least 1 frame
        assert!(!frames.is_empty(), "should generate at least one frame");
        assert!(frames.len() <= 5, "should not exceed max_frames");

        // Each frame has 16 codes
        for frame in &frames {
            assert_eq!(frame.len(), 16, "each frame must have 16 code indices");
        }
    }

    // ── Helper: build minimal talker with all required tensors ──

    fn build_minimal_talker(device: &Device) -> Talker {
        let mut store = WeightStore::new(device.clone());

        // Embeddings
        store.insert_tensor(
            "talker.model.embed_tokens.weight",
            Tensor::zeros((TEXT_VOCAB, HIDDEN), DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.codec_embedding.weight",
            Tensor::zeros((CODEC_VOCAB, HIDDEN), DType::F32, device).unwrap(),
        );

        // Text projection
        store.insert_tensor(
            "talker.model.text_projection.fc1.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.text_projection.fc1.bias",
            Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.text_projection.fc2.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.text_projection.fc2.bias",
            Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
        );

        // 28 backbone layers
        for i in 0..NUM_BACKBONE_LAYERS {
            let p = format!("talker.model.layers.{i}");
            store.insert_tensor(
                format!("{p}.self_attn.q_proj.weight"),
                Tensor::zeros((HIDDEN, HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.k_proj.weight"),
                Tensor::zeros((HIDDEN / 2, HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.v_proj.weight"),
                Tensor::zeros((HIDDEN / 2, HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.o_proj.weight"),
                Tensor::zeros((HIDDEN, HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.mlp.gate_proj.weight"),
                Tensor::zeros((BACKBONE_FFN, HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.mlp.up_proj.weight"),
                Tensor::zeros((BACKBONE_FFN, HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.mlp.down_proj.weight"),
                Tensor::zeros((HIDDEN, BACKBONE_FFN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.input_layernorm.weight"),
                Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.q_norm.weight"),
                Tensor::zeros(HEAD_DIM, DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.k_norm.weight"),
                Tensor::zeros(HEAD_DIM, DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn_layer_scale.scale"),
                Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.mlp_layer_scale.scale"),
                Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
            );
        }

        store.insert_tensor(
            "talker.model.norm.weight",
            Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
        );

        store.insert_tensor(
            "talker.codec_head.weight",
            Tensor::zeros((CODEC_VOCAB, HIDDEN), DType::F32, device).unwrap(),
        );

        // Code predictor
        store.insert_tensor(
            "talker.code_predictor.small_to_mtp_projection.weight",
            Tensor::zeros((CP_HIDDEN, HIDDEN), DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.code_predictor.small_to_mtp_projection.bias",
            Tensor::zeros(CP_HIDDEN, DType::F32, device).unwrap(),
        );

        for i in 0..NUM_CP_LAYERS {
            let p = format!("talker.code_predictor.model.layers.{i}");
            store.insert_tensor(
                format!("{p}.self_attn.q_proj.weight"),
                Tensor::zeros((HIDDEN, CP_HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.k_proj.weight"),
                Tensor::zeros((HIDDEN / 2, CP_HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.v_proj.weight"),
                Tensor::zeros((HIDDEN / 2, CP_HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.self_attn.o_proj.weight"),
                Tensor::zeros((CP_HIDDEN, HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.mlp.gate_proj.weight"),
                Tensor::zeros((CP_FFN, CP_HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.mlp.up_proj.weight"),
                Tensor::zeros((CP_FFN, CP_HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.mlp.down_proj.weight"),
                Tensor::zeros((CP_HIDDEN, CP_FFN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.input_layernorm.weight"),
                Tensor::zeros(CP_HIDDEN, DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::zeros(CP_HIDDEN, DType::F32, device).unwrap(),
            );
        }

        store.insert_tensor(
            "talker.code_predictor.model.norm.weight",
            Tensor::zeros(CP_HIDDEN, DType::F32, device).unwrap(),
        );

        for i in 0..NUM_RESIDUAL_CODES {
            store.insert_tensor(
                format!("talker.code_predictor.lm_heads.{i}.weight"),
                Tensor::zeros((CODEBOOK_SIZE, CP_HIDDEN), DType::F32, device).unwrap(),
            );
        }

        Talker::from_store(&store).unwrap()
    }
}
