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

use crate::custom_ops::causal_mask;
use crate::error::{VoxError, VoxResult};
use crate::sampling::{argmax, sample_token, SamplingConfig};
use crate::transformer::{RmsNorm, TransformerBlock, TransformerCache, TransformerStack};
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
const EPS: f64 = 1e-6;
const TALKER_ROPE_THETA: f64 = 1_000_000.0;

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
    cp_codec_embeddings: Vec<Tensor>, // 15 × [2048, 2048]
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
        let text_embedding = store.require("talker.model.text_embedding.weight")?.clone();
        Self::check_shape(&text_embedding, &[TEXT_VOCAB, HIDDEN], "text_embedding")?;

        let codec_embedding = store
            .require("talker.model.codec_embedding.weight")?
            .clone();
        Self::check_shape(&codec_embedding, &[CODEC_VOCAB, HIDDEN], "codec_embedding")?;

        // ── 2. Text projection ──
        let text_proj_fc1_w = store
            .require("talker.text_projection.linear_fc1.weight")?
            .clone();
        Self::check_shape(&text_proj_fc1_w, &[HIDDEN, HIDDEN], "text_proj.fc1.w")?;
        let text_proj_fc1_b = store
            .require("talker.text_projection.linear_fc1.bias")?
            .clone();
        let text_proj_fc2_w = store
            .require("talker.text_projection.linear_fc2.weight")?
            .clone();
        Self::check_shape(&text_proj_fc2_w, &[HIDDEN, HIDDEN], "text_proj.fc2.w")?;
        let text_proj_fc2_b = store
            .require("talker.text_projection.linear_fc2.bias")?
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
        let mut cp_codec_embeddings = Vec::with_capacity(NUM_RESIDUAL_CODES);
        for i in 0..NUM_RESIDUAL_CODES {
            let key = format!("talker.code_predictor.lm_head.{i}.weight");
            let head = store.require(&key)?.clone();
            Self::check_shape(&head, &[CODEBOOK_SIZE, CP_HIDDEN], &key)?;
            cp_lm_heads.push(head);

            let emb_key = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
            let emb = store.require(&emb_key)?.clone();
            Self::check_shape(&emb, &[CODEBOOK_SIZE, HIDDEN], &emb_key)?;
            cp_codec_embeddings.push(emb);
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
            cp_codec_embeddings,
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

    fn embed_residual_code(&self, level: usize, token: u32) -> VoxResult<Tensor> {
        let device = self.cp_codec_embeddings[level].device();
        let t = Tensor::new(&[token], device)?;
        let emb = self.cp_codec_embeddings[level].index_select(&t, 0)?;
        Ok(emb.unsqueeze(0)?)
    }

    fn embed_codec_frame(&self, frame: &[u16; 16]) -> VoxResult<Tensor> {
        let mut emb = self.embed_code_token(frame[0] as u32)?;
        for (level, &code) in frame.iter().enumerate().skip(1) {
            let residual = self.embed_residual_code(level - 1, code as u32)?;
            emb = emb.add(&residual)?;
        }
        Ok(emb)
    }

    fn embed_codec_prefill(&self, ids: &[u32]) -> VoxResult<Tensor> {
        let mut parts = Vec::with_capacity(ids.len());
        for &id in ids {
            parts.push(self.embed_code_token(id)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
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

    fn backbone_forward_cached(
        &self,
        x: &Tensor,
        cache: &mut TransformerCache,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let h = self.backbone.forward_with_cache(x, mask, cache)?;
        self.final_norm.forward(&h)
    }

    // ── Code Prediction ────────────────────────────────────────────────────────

    /// Predict one frame of 16 codes (q0..q15) from a backbone hidden state.
    ///
    /// `hidden`: `[B, T, 2048]` — entire sequence hidden states.
    /// Uses `hidden[last]` to predict the next frame.
    ///
    /// `config`: sampling configuration (temperature, top-k, top-p, repetition penalty).
    /// `q0_history`: previously generated q0 tokens (for repetition penalty).
    ///
    /// Returns `(q0, residual_codes)` where residual_codes has 15 entries.
    ///
    /// Code Predictor architecture (autoregressive, 15 steps):
    /// - Prefill pos=0: `cp_h = small_to_mtp(last_hidden)` (2048 -> 1024)
    /// - Prefill pos=1: `cp_h = small_to_mtp(codec_emb[q0])` (2048 -> 1024)
    /// - For g = 1..15:
    ///   - input = `small_to_mtp(cp_codec_emb[g-1][q_{g-1}])` (2048 -> 1024)
    ///   - output = cp_transformer(input) at last position
    ///   - q_g = argmax(cp_lm_heads[g-1] @ output)
    fn predict_codes(
        &self,
        hidden: &Tensor,
        config: &SamplingConfig,
        q0_history: &[u16],
    ) -> VoxResult<(u16, Vec<u16>)> {
        // Last position: [B, 2048]
        let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?;

        // ── q0 via codec_head ──
        // last: [B, 2048], codec_head: [3072, 2048]
        let q0_logits = last.broadcast_matmul(&self.codec_head.t()?)?; // [B, 3072]
        let q0_logits = q0_logits.narrow(1, 0, CODEBOOK_SIZE)?; // [B, 2048]
                                                                // Convert to F32 (candle matmul on CUDA may keep BF16); sampling needs f32 slice.
        let q0_logits_f32 = q0_logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
        let q0_logits_vec = q0_logits_f32.to_vec1::<f32>()?;
        let q0_idx = sample_token(&q0_logits_vec, config, q0_history);

        // ── q1..q15 via code predictor (autoregressive) ──
        // Helper: small_to_mtp 2048 → 1024, returns [B, 1, 1024] for transformer input
        let project = |x: &Tensor| -> VoxResult<Tensor> {
            let y = x.broadcast_matmul(&self.small_to_mtp_w.t()?)?;
            let y = y.broadcast_add(&self.small_to_mtp_b)?;
            Ok(y.unsqueeze(0)?) // [1, 1, 1024] (squeeze later when needed)
        };

        // ── Prefill 2 positions ──
        // pos=0: talker hidden (projected)
        let cp_pre_0 = project(&last)?; // [1, 1, 1024]
                                        // pos=1: q0 embedding from TALKER codec_embedding (also 2048-dim, projected)
        let q0_emb = self.embed_code_token(q0_idx as u32)?; // [1, 1, 2048]
        let q0_emb_2d = q0_emb.squeeze(0)?; // [1, 2048]
        let cp_pre_1 = project(&q0_emb_2d)?; // [1, 1, 1024]

        // Autoregressive loop: at iteration g (0..14), build seq of length (g+2):
        //   [cp_pre_0, cp_pre_1, emb(q1), emb(q2), ..., emb(q_g)]
        // Run CP transformer, read out last position, apply lm_head[g] → q_{g+1}.
        let mut codes: Vec<u16> = Vec::with_capacity(NUM_RESIDUAL_CODES);
        let mut history: Vec<u16> = vec![q0_idx]; // history[0] = q0, history[1] = q1, ...

        for g in 0..NUM_RESIDUAL_CODES {
            let mut seq_parts: Vec<Tensor> = Vec::with_capacity(2 + g);
            seq_parts.push(cp_pre_0.clone());
            seq_parts.push(cp_pre_1.clone());
            for (i, &q_i) in history.iter().enumerate().take(g + 1).skip(1) {
                let emb = self.embed_residual_code(i - 1, q_i as u32)?; // [1, 1, 2048]
                let emb_2d = emb.squeeze(0)?; // [1, 2048]
                seq_parts.push(project(&emb_2d)?); // [1, 1, 1024]
            }
            let seq_refs: Vec<&Tensor> = seq_parts.iter().collect();
            let seq = Tensor::cat(&seq_refs, 1)?; // [1, g+2, 1024]

            // Run CP transformer (no cache — just full forward; max 17 tokens is cheap)
            let cp_out = self
                .cp_transformer
                .forward(&seq, None)
                .map_err(|e| VoxError::Inference(format!("cp forward g={g}: {e}")))?;
            let cp_last = cp_out.narrow(1, g + 1, 1)?.squeeze(1)?; // [1, 1024]

            // Apply lm_head[g]: [2048, 1024]
            let head = &self.cp_lm_heads[g];
            let logits = cp_last.broadcast_matmul(&head.t()?)?; // [1, 2048]
            let logits_f32 = logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
            let logits_vec = logits_f32.to_vec1::<f32>()?;
            let idx = argmax(&logits_vec) as u16;
            codes.push(idx);
            history.push(idx);
        }

        Ok((q0_idx, codes))
    }

    // ── Autoregressive Generation ──────────────────────────────────────────────

    /// Generate a sequence of RVQ code frames from phone tokens.
    ///
    /// # Arguments
    /// * `phone_tokens` — phone/semantic token IDs (BOS should already be prepended)
    /// * `max_frames` — maximum number of code frames to generate
    /// * `config` — sampling configuration (use `SamplingConfig::argmax()` for deterministic)
    ///
    /// # Returns
    /// `Vec<[u16; 16]>` — sequence of (q0, q1, ..., q15) code frames.
    pub fn generate(
        &self,
        phone_tokens: &[u32],
        max_frames: usize,
        config: &SamplingConfig,
    ) -> VoxResult<Vec<[u16; 16]>> {
        if phone_tokens.is_empty() {
            return Err(VoxError::Inference("phone_tokens must not be empty".into()));
        }

        // Encode text: [1, T, 2048]
        let text_hidden = self.encode_text(phone_tokens)?;

        // The autoregressive input starts as just the text hidden states.
        // As we generate, we append q0 code embeddings.
        let mut input_hidden = text_hidden;
        let mut frames: Vec<[u16; 16]> = Vec::with_capacity(max_frames.min(512));
        let mut q0_history: Vec<u16> = Vec::with_capacity(max_frames.min(512));

        for _step in 0..max_frames {
            // Backbone forward: [1, T, 2048] → [1, T, 2048]
            let hidden = self
                .backbone_forward(&input_hidden)
                .map_err(|e| VoxError::Inference(format!("backbone forward: {e}")))?;

            // Predict codes from last position
            let (q0, residual_codes) = self.predict_codes(&hidden, config, &q0_history)?;

            // Assemble the 16-code frame
            let mut frame = [0u16; 16];
            frame[0] = q0;
            for (i, &rc) in residual_codes.iter().enumerate() {
                frame[i + 1] = rc;
            }
            frames.push(frame);
            q0_history.push(q0);

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

    /// Generate codec frames using the Qwen3-TTS base prompt layout.
    ///
    /// `input_tokens` should be the ChatML-style text prompt used by the
    /// official pipeline:
    /// `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
    ///
    /// `config` controls sampling behavior (temperature, top-k, top-p, repetition penalty).
    pub fn generate_qwen3_base(
        &self,
        input_tokens: &[u32],
        language_id: Option<u32>,
        speaker_id: Option<u32>,
        max_frames: usize,
        config: &SamplingConfig,
    ) -> VoxResult<Vec<[u16; 16]>> {
        if input_tokens.len() < 9 {
            return Err(VoxError::Inference(
                "Qwen3-TTS prompt must contain role/text/end tokens".into(),
            ));
        }

        let tts_bos_embed = self.encode_text(&[TTS_BOS])?;
        let tts_eos_embed = self.encode_text(&[TTS_EOS])?;
        let tts_pad_embed = self.encode_text(&[TTS_PAD])?;

        // Codec prefill (matches mlx-audio reference):
        //   [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id, <speaker_id?>, codec_pad_id, codec_bos_id]
        // For the CustomVoice model the speaker token is REQUIRED between
        // codec_think_eos_id and codec_pad_id; without it the model has no voice
        // identity and produces high-frequency noise instead of speech.
        let mut codec_prefill = match language_id {
            Some(id) => vec![2154, 2156, id, 2157],
            None => vec![2155, 2156, 2157],
        };
        if let Some(spk) = speaker_id {
            codec_prefill.push(spk);
        }
        codec_prefill.push(2148);
        codec_prefill.push(2149);

        let role_embed = self.encode_text(&input_tokens[..3])?;
        let codec_embed = self.embed_codec_prefill(&codec_prefill)?;
        let codec_len = codec_prefill.len();

        let pad_count = codec_len.saturating_sub(2);
        let mut prefill_text_parts = Vec::with_capacity(pad_count + 1);
        for _ in 0..pad_count {
            prefill_text_parts.push(&tts_pad_embed);
        }
        prefill_text_parts.push(&tts_bos_embed);
        let text_prefill = Tensor::cat(&prefill_text_parts, 1)?;
        let codec_without_last = codec_embed.narrow(1, 0, codec_len - 1)?;
        let prefill_embed = text_prefill.add(&codec_without_last)?;

        // Official CustomVoice generation defaults to non_streaming_mode=true:
        // all target text plus TTS_EOS is prefed with codec_pad, then the final
        // position is tts_pad + codec_bos. The autoregressive loop therefore
        // adds only tts_pad for subsequent codec frames.
        let text_parts = [
            self.encode_text(&input_tokens[3..input_tokens.len() - 5])?,
            tts_eos_embed,
        ];
        let text_refs: Vec<&Tensor> = text_parts.iter().collect();
        let text_with_eos = Tensor::cat(&text_refs, 1)?;

        let text_codec_pad_ids = vec![2148u32; text_with_eos.dim(1)?];
        let text_codec_pad = self.embed_codec_prefill(&text_codec_pad_ids)?;
        let text_with_codec_pad = text_with_eos.add(&text_codec_pad)?;

        let final_bos = tts_pad_embed.add(&codec_embed.narrow(1, codec_len - 1, 1)?)?;
        let input_hidden = Tensor::cat(
            &[
                &role_embed,
                &prefill_embed,
                &text_with_codec_pad,
                &final_bos,
            ],
            1,
        )?;

        let trailing_text_hidden = tts_pad_embed.clone();

        let mut cache = self.backbone.empty_cache();
        let prefill_mask = causal_mask(input_hidden.dim(1)?, input_hidden.device())?;
        let mut hidden = self
            .backbone_forward_cached(&input_hidden, &mut cache, Some(&prefill_mask))
            .map_err(|e| VoxError::Inference(format!("backbone prefill: {e}")))?;

        let mut frames = Vec::with_capacity(max_frames.min(512));
        let mut q0_history: Vec<u16> = Vec::with_capacity(max_frames.min(512));
        for step in 0..max_frames {
            let (q0, residual_codes) = self.predict_codes(&hidden, config, &q0_history)?;

            let mut frame = [0u16; 16];
            frame[0] = q0;
            for (i, &rc) in residual_codes.iter().enumerate() {
                frame[i + 1] = rc;
            }
            frames.push(frame);
            q0_history.push(q0);

            if q0 as u32 == CODEC_EOS {
                break;
            }

            let mut next_embed = self.embed_codec_frame(&frame)?;
            let text_add = if step < trailing_text_hidden.dim(1)? {
                trailing_text_hidden.narrow(1, step, 1)?
            } else {
                tts_pad_embed.clone()
            };
            next_embed = next_embed.add(&text_add)?;
            hidden = self
                .backbone_forward_cached(&next_embed, &mut cache, None)
                .map_err(|e| VoxError::Inference(format!("backbone incremental: {e}")))?;
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

        // LayerScale is present in some exported alignment fixtures, but the
        // official Qwen3-TTS talker weights omit it.
        let als = store
            .get(&format!("{prefix}.self_attn_layer_scale.scale"))
            .cloned();
        let mls = store
            .get(&format!("{prefix}.mlp_layer_scale.scale"))
            .cloned();

        Ok(TransformerBlock::from_weights(
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
        )?
        .with_rope_theta(TALKER_ROPE_THETA))
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

        Ok(TransformerBlock::from_weights(
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
        )?
        .with_rope_theta(TALKER_ROPE_THETA))
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
            "talker.model.text_embedding.weight",
            Tensor::zeros((TEXT_VOCAB, HIDDEN), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.codec_embedding.weight",
            Tensor::zeros((CODEC_VOCAB, HIDDEN), DType::F32, &device).unwrap(),
        );

        // Text projection
        store.insert_tensor(
            "talker.text_projection.linear_fc1.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.text_projection.linear_fc1.bias",
            Tensor::zeros(HIDDEN, DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.text_projection.linear_fc2.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            "talker.text_projection.linear_fc2.bias",
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
                format!("talker.code_predictor.lm_head.{i}.weight"),
                Tensor::zeros((CODEBOOK_SIZE, CP_HIDDEN), DType::F32, &device).unwrap(),
            );
            store.insert_tensor(
                format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
                Tensor::zeros((CODEBOOK_SIZE, HIDDEN), DType::F32, &device).unwrap(),
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
        let config = SamplingConfig::argmax();
        let (q0, residual) = talker.predict_codes(&hidden, &config, &[]).unwrap();

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
        let config = SamplingConfig::argmax();
        let frames = talker.generate(&tokens, 5, &config).unwrap();

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
            "talker.model.text_embedding.weight",
            Tensor::zeros((TEXT_VOCAB, HIDDEN), DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.model.codec_embedding.weight",
            Tensor::zeros((CODEC_VOCAB, HIDDEN), DType::F32, device).unwrap(),
        );

        // Text projection
        store.insert_tensor(
            "talker.text_projection.linear_fc1.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.text_projection.linear_fc1.bias",
            Tensor::zeros(HIDDEN, DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.text_projection.linear_fc2.weight",
            Tensor::zeros((HIDDEN, HIDDEN), DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            "talker.text_projection.linear_fc2.bias",
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
                format!("talker.code_predictor.lm_head.{i}.weight"),
                Tensor::zeros((CODEBOOK_SIZE, CP_HIDDEN), DType::F32, device).unwrap(),
            );
            store.insert_tensor(
                format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
                Tensor::zeros((CODEBOOK_SIZE, HIDDEN), DType::F32, device).unwrap(),
            );
        }

        Talker::from_store(&store).unwrap()
    }
}
