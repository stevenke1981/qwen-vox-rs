//! Quantizer for Qwen3-TTS decoder (Candle 0.9+).
//!
//! Provides:
//! - `EuclideanCodebook`: simple vector lookup by index
//! - `ResidualVectorQuantizer`: multi-layer RVQ with input/output projections (conv1d k=1)
//! - `SplitResidualVectorQuantizer`: semantic (1-layer) + acoustic (15-layer) RVQ
//! - `CodePredictor`: MTP head with 15 codec embeddings + 15 lm_heads for residual code prediction
//!
//! The decoder quantizer reconstructs hidden features from discrete codes.

use crate::error::{VoxError, VoxResult};
use crate::weights::WeightStore;
use candle_core::{DType, Result, Tensor};

/// Vector quantization codebook using Euclidean distance (nearest neighbor lookup).
pub struct EuclideanCodebook {
    /// Embedding table: [vocab_size, dim]
    embedding: Tensor,
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    dim: usize,
}

impl EuclideanCodebook {
    /// Create from embedding tensor of shape [vocab_size, dim].
    pub fn from_embedding(embedding: Tensor) -> VoxResult<Self> {
        let dims = embedding.dims2().map_err(|e| {
            VoxError::WeightLoad(format!("codebook embedding must be 2-D [vocab, dim]: {e}"))
        })?;
        let (vocab_size, dim) = dims;
        if vocab_size == 0 || dim == 0 {
            return Err(VoxError::WeightLoad(
                "codebook vocab_size and dim must be >0".into(),
            ));
        }
        Ok(Self {
            embedding,
            vocab_size,
            dim,
        })
    }

    /// Decode indices to embeddings.
    ///
    /// # Arguments
    /// * `indices` - [batch, seq_len] integer tensor (u32 or i64)
    ///
    /// # Returns
    /// Embeddings of shape [batch, seq_len, dim]
    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let batch = indices.dim(0)?;
        let seq_len = indices.dim(1)?;
        let flat = indices.flatten_all()?;
        // Support common index dtypes used by safetensors / token loading
        let flat = if flat.dtype() == DType::U32 || flat.dtype() == DType::I64 {
            flat
        } else {
            flat.to_dtype(DType::I64)?
        };
        let selected = self.embedding.index_select(&flat, 0)?;
        selected.reshape((batch, seq_len, self.dim))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Single RVQ stage with input/output 1x1 conv projections and N codebook layers.
pub struct ResidualVectorQuantizer {
    /// Input projection (Conv1d weight): [proj_dim, hidden_dim, 1]
    /// (used for encoding path; stored for completeness)
    #[allow(dead_code)]
    input_proj: Tensor,
    /// Output projection (Conv1d weight): [hidden_dim, proj_dim, 1]
    output_proj: Tensor,
    /// Per-layer Euclidean codebooks (embeddings live in proj_dim space)
    codebooks: Vec<EuclideanCodebook>,
    /// Number of RVQ layers in this quantizer
    num_layers: usize,
}

impl ResidualVectorQuantizer {
    /// Create from pre-split weights.
    ///
    /// * `input_proj` - [proj_dim, hidden_dim, 1]
    /// * `output_proj` - [hidden_dim, proj_dim, 1]
    /// * `codebook_embeddings` - one [vocab, proj_dim] per layer
    pub fn from_weights(
        input_proj: Tensor,
        output_proj: Tensor,
        codebook_embeddings: Vec<Tensor>,
    ) -> VoxResult<Self> {
        let in_dims = input_proj
            .dims3()
            .map_err(|e| VoxError::WeightLoad(format!("input_proj must be 3-D [out,in,1]: {e}")))?;
        let out_dims = output_proj.dims3().map_err(|e| {
            VoxError::WeightLoad(format!("output_proj must be 3-D [out,in,1]: {e}"))
        })?;
        if in_dims.2 != 1 || out_dims.2 != 1 {
            return Err(VoxError::WeightLoad(
                "projection weights must have kernel size 1 (last dim)".into(),
            ));
        }
        let proj_dim = in_dims.0; // for input_proj [proj, hidden, 1] -> proj_dim = 256
        let num_layers = codebook_embeddings.len();
        if num_layers == 0 {
            return Err(VoxError::WeightLoad(
                "RVQ requires at least one codebook layer".into(),
            ));
        }
        let mut codebooks = Vec::with_capacity(num_layers);
        for (i, emb) in codebook_embeddings.into_iter().enumerate() {
            let cb = EuclideanCodebook::from_embedding(emb).map_err(|e| {
                VoxError::WeightLoad(format!("failed to load codebook layer {i}: {e}"))
            })?;
            if cb.dim() != proj_dim {
                return Err(VoxError::WeightLoad(format!(
                    "codebook layer {i} dim {} != proj_dim {}",
                    cb.dim(),
                    proj_dim
                )));
            }
            codebooks.push(cb);
        }
        Ok(Self {
            input_proj,
            output_proj,
            codebooks,
            num_layers,
        })
    }

    /// Decode a list of per-layer code indices into reconstructed features.
    ///
    /// codes: slice of [batch, seq_len] index tensors, length == num_layers
    /// Returns: [batch, hidden_dim, seq_len]
    pub fn decode(&self, codes: &[Tensor]) -> Result<Tensor> {
        if codes.len() != self.num_layers {
            return Err(candle_core::Error::Msg(format!(
                "RVQ expected {} code tensors, got {}",
                self.num_layers,
                codes.len()
            )));
        }
        if codes.is_empty() {
            return Err(candle_core::Error::Msg(
                "RVQ decode called with empty codes".into(),
            ));
        }
        let batch = codes[0].dim(0)?;
        let seq_len = codes[0].dim(1)?;
        let proj_dim = self.codebooks[0].dim();
        let device = codes[0].device();
        let _dtype = codes[0].dtype(); // usually i64/u32 but we use f32 for accum

        // Start with zero residual in embedding (proj) space: [batch, seq_len, proj_dim]
        let mut residual = Tensor::zeros((batch, seq_len, proj_dim), DType::F32, device)?;

        for (layer_idx, code_tensor) in codes.iter().enumerate() {
            let emb = self.codebooks[layer_idx].decode(code_tensor)?; // [b, s, proj]
            residual = residual.add(&emb)?;
        }

        // Apply output projection using matmul (k=1 conv1d equivalent, more reliable)
        // residual: [b, s, proj]
        // output_proj: [hidden, proj, 1] -> squeeze to [hidden, proj]
        let w = self.output_proj.squeeze(2)?;
        let w_t = w.t()?; // [proj, hidden]
        let hidden_dim = w_t.dim(1)?;
        let b = residual.dim(0)?;
        let s = residual.dim(1)?;
        let p = residual.dim(2)?;
        let res2 = residual.reshape((b * s, p))?;
        let out2 = res2.matmul(&w_t)?; // [b*s, hidden]
        let out = out2.reshape((b, s, hidden_dim))?;

        // Return in conv convention: [batch, hidden_dim, seq_len]
        out.transpose(1, 2)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// Split RVQ: first codebook group is semantic (rvq_first, 1 layer),
/// remaining are acoustic (rvq_rest, 15 layers).
pub struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
}

impl SplitResidualVectorQuantizer {
    pub fn from_weights(
        rvq_first: ResidualVectorQuantizer,
        rvq_rest: ResidualVectorQuantizer,
    ) -> VoxResult<Self> {
        if rvq_first.num_layers() != 1 {
            return Err(VoxError::WeightLoad(
                "rvq_first must contain exactly 1 layer (semantic)".into(),
            ));
        }
        if rvq_rest.num_layers() != 15 {
            return Err(VoxError::WeightLoad(
                "rvq_rest must contain exactly 15 layers (acoustic)".into(),
            ));
        }
        Ok(Self {
            rvq_first,
            rvq_rest,
        })
    }

    /// Decode using split: codes[0] -> rvq_first, codes[1..] -> rvq_rest
    /// Sum the two reconstructed tensors.
    /// codes: total 16 tensors for full 16-layer codes
    /// Returns: [batch, hidden_dim, seq_len]
    pub fn decode(&self, codes: &[Tensor]) -> Result<Tensor> {
        if codes.is_empty() {
            return Err(candle_core::Error::Msg(
                "SplitRVQ requires at least 1 code tensor".into(),
            ));
        }
        let first = self.rvq_first.decode(&codes[0..1])?;
        let rest = self.rvq_rest.decode(&codes[1..])?;
        first.add(&rest)
    }
}

/// MTP (Multi-Token Prediction) head used by the talker to predict the 15 residual
/// acoustic codebook tokens (in addition to the main semantic token).
pub struct CodePredictor {
    #[allow(dead_code)]
    /// 15 embedding tables (one per code group), each [vocab_size, hidden_dim]
    codec_embeddings: Vec<Tensor>,
    /// 15 linear heads (one per code group), each [vocab_size, hidden_dim]
    lm_heads: Vec<Tensor>,
    /// Number of codebook groups (typically 15)
    num_code_groups: usize,
    /// Vocabulary size per group (typically 2048)
    vocab_size: usize,
}

impl CodePredictor {
    pub fn from_weights(codec_embeddings: Vec<Tensor>, lm_heads: Vec<Tensor>) -> VoxResult<Self> {
        if codec_embeddings.len() != lm_heads.len() {
            return Err(VoxError::WeightLoad(
                "codec_embeddings and lm_heads must have identical length".into(),
            ));
        }
        let num_code_groups = codec_embeddings.len();
        if num_code_groups == 0 {
            return Err(VoxError::WeightLoad(
                "CodePredictor requires at least one code group".into(),
            ));
        }
        // Infer vocab from first lm_head: [vocab, hidden]
        let first_head_dims = lm_heads[0].dims2().map_err(|e| {
            VoxError::WeightLoad(format!("lm_head[0] must be 2-D [vocab, hidden]: {e}"))
        })?;
        let vocab_size = first_head_dims.0;
        // Basic consistency checks
        for (i, head) in lm_heads.iter().enumerate() {
            let d = head
                .dims2()
                .map_err(|e| VoxError::WeightLoad(format!("lm_head[{i}] must be 2-D: {e}")))?;
            if d.0 != vocab_size {
                return Err(VoxError::WeightLoad(format!(
                    "lm_head[{i}] vocab_size {} != {}",
                    d.0, vocab_size
                )));
            }
        }
        for (i, emb) in codec_embeddings.iter().enumerate() {
            let d = emb.dims2().map_err(|e| {
                VoxError::WeightLoad(format!("codec_embedding[{i}] must be 2-D: {e}"))
            })?;
            if d.0 != vocab_size {
                return Err(VoxError::WeightLoad(format!(
                    "codec_embedding[{i}] vocab_size {} != {}",
                    d.0, vocab_size
                )));
            }
        }
        Ok(Self {
            codec_embeddings,
            lm_heads,
            num_code_groups,
            vocab_size,
        })
    }

    /// Predict logits for all 15 residual code groups.
    ///
    /// hidden_states: [batch, seq_len, hidden_dim] (talker output style)
    /// Returns: Vec of 15 tensors each [batch, seq_len, vocab_size]
    pub fn predict(&self, hidden_states: &Tensor) -> Result<Vec<Tensor>> {
        let mut all_logits = Vec::with_capacity(self.num_code_groups);
        for head in self.lm_heads.iter() {
            // [b, s, d] @ [d, v]  (head is [v, d] so transpose)
            // Use reshape for reliable batched matmul (candle 3D@2D can be picky on broadcast)
            let b = hidden_states.dim(0)?;
            let s = hidden_states.dim(1)?;
            let d = hidden_states.dim(2)?;
            let hs2 = hidden_states.reshape((b * s, d))?;
            let head_t = head.t()?;
            let logits2 = hs2.matmul(&head_t)?;
            let logits = logits2.reshape((b, s, head.dim(0)?))?;
            all_logits.push(logits);
        }
        Ok(all_logits)
    }

    pub fn num_code_groups(&self) -> usize {
        self.num_code_groups
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Load codebook embeddings from weight store for the decoder quantizer.
/// The decoder quantizer uses keys like:
///   decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum
///   decoder.quantizer.rvq_rest.vq.layers.{0-14}._codebook.embedding_sum
///
/// Returns Vec of 16 tensors (1 for rvq_first + 15 for rvq_rest), each [2048, 256]
pub fn load_decoder_codebooks(store: &WeightStore) -> VoxResult<Vec<Tensor>> {
    let mut codebooks = Vec::with_capacity(16);

    // rvq_first (semantic, 1 layer)
    let first_key = "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum";
    let first_emb = store.require(first_key)?.clone();
    codebooks.push(first_emb);

    // rvq_rest (acoustic, 15 layers)
    for i in 0..15 {
        let key = format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum");
        let emb = store.require(&key)?.clone();
        codebooks.push(emb);
    }

    Ok(codebooks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_euclidean_codebook_decode() {
        let device = Device::Cpu;
        // embedding: [3 vocab, 2 dim]
        let emb_data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let embedding = Tensor::from_vec(emb_data, (3, 2), &device).unwrap();
        let cb = EuclideanCodebook::from_embedding(embedding).unwrap();
        assert_eq!(cb.vocab_size(), 3);
        assert_eq!(cb.dim(), 2);

        // indices: [batch=2, seq=2]
        // row0: code 0 then 2
        // row1: code 1 then 0
        let idx_data: Vec<u32> = vec![0, 2, 1, 0];
        let indices = Tensor::from_vec(idx_data, (2, 2), &device).unwrap();

        let decoded = cb.decode(&indices).unwrap();
        assert_eq!(decoded.dims(), &[2, 2, 2]);

        let flat: Vec<f32> = decoded.flatten_all().unwrap().to_vec1().unwrap();
        // expected:
        // [0,1]  (idx0)
        // [4,5]  (idx2)
        // [2,3]  (idx1)
        // [0,1]  (idx0)
        assert_eq!(flat, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn test_code_predictor_predict_shape() {
        let device = Device::Cpu;
        // Use small numbers for fast test (real model uses 15 groups, 2048 vocab, 1024 hidden)
        let num_groups = 2usize;
        let vocab_size = 5usize;
        let hidden_dim = 3usize;

        let mut codec_embeddings = Vec::new();
        let mut lm_heads = Vec::new();
        for _ in 0..num_groups {
            let emb = Tensor::zeros((vocab_size, hidden_dim), DType::F32, &device).unwrap();
            let head = Tensor::zeros((vocab_size, hidden_dim), DType::F32, &device).unwrap();
            codec_embeddings.push(emb);
            lm_heads.push(head);
        }

        let predictor = CodePredictor::from_weights(codec_embeddings, lm_heads).unwrap();
        assert_eq!(predictor.num_code_groups(), num_groups);
        assert_eq!(predictor.vocab_size(), vocab_size);

        // hidden_states from talker: [batch, seq_len, hidden]
        let batch = 2usize;
        let seq = 7usize;
        let hidden_states = Tensor::zeros((batch, seq, hidden_dim), DType::F32, &device).unwrap();

        let logits_vec = predictor.predict(&hidden_states).unwrap();
        assert_eq!(logits_vec.len(), num_groups);
        for logits in &logits_vec {
            assert_eq!(logits.dims(), &[batch, seq, vocab_size]);
        }
    }

    #[test]
    fn test_rvq_and_split_shapes() {
        let device = Device::Cpu;
        let proj_dim = 4usize;
        let hidden_dim = 8usize;
        let vocab = 3usize;
        let batch = 1usize;
        let seq = 2usize;

        // Build dummy projs [out, in, 1]
        let input_p = Tensor::zeros((proj_dim, hidden_dim, 1), DType::F32, &device).unwrap();
        let output_p = Tensor::zeros((hidden_dim, proj_dim, 1), DType::F32, &device).unwrap();

        // One codebook for first
        let emb0 = Tensor::zeros((vocab, proj_dim), DType::F32, &device).unwrap();
        let rvq1 =
            ResidualVectorQuantizer::from_weights(input_p.clone(), output_p.clone(), vec![emb0])
                .unwrap();
        assert_eq!(rvq1.num_layers(), 1);

        // 15 dummy for rest (use same emb for simplicity)
        let mut rest_embs = vec![];
        for _ in 0..15 {
            rest_embs.push(Tensor::zeros((vocab, proj_dim), DType::F32, &device).unwrap());
        }
        let rvq15 = ResidualVectorQuantizer::from_weights(input_p, output_p, rest_embs).unwrap();
        assert_eq!(rvq15.num_layers(), 15);

        let split = SplitResidualVectorQuantizer::from_weights(rvq1, rvq15).unwrap();

        // Build 16 dummy code index tensors [b, s] of u32
        let mut codes: Vec<Tensor> = vec![];
        for _ in 0..16 {
            let c = Tensor::zeros((batch, seq), DType::U32, &device).unwrap();
            codes.push(c);
        }

        let out = split.decode(&codes).unwrap();
        assert_eq!(out.dims(), &[batch, hidden_dim, seq]);
    }
}
