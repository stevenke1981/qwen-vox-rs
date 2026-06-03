//! Speaker Encoder (ECAPA-TDNN + Attentive Statistics Pooling)
//!
//! Extracts speaker embeddings from reference audio (mel spectrogram).
//! Architecture: ECAPA-TDNN with SE-Res2Blocks + ASP pooling.
//!
//! Weight prefixes observed:
//! - `asp.*` — Attentive Statistics Pooling parameters
//! - `blocks.*` — TDNN / Res2Net blocks

use crate::error::{VoxError, VoxResult};
use crate::weights::WeightStore;
use candle_core::{Result, Tensor};

const RES2NET_SCALE: usize = 8;
const SE_RES2NET_BLOCKS: usize = 3;
const ASP_EPS: f64 = 1e-12;

/// Speaker Encoder producing fixed-dimensional speaker embeddings.
/// Input: mel spectrogram [B, T, n_mels], matching official Qwen3-TTS.
/// Output: speaker embedding [B, embed_dim]
pub struct SpeakerEncoder {
    prefix: String,
    mel_dim: usize,
    embed_dim: usize,
    initial: TdnnBlock,
    se_blocks: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatsPool,
    fc: Conv1dSame,
}

impl SpeakerEncoder {
    /// Load from WeightStore.
    ///
    /// Accepts either a component store with `asp.*` / `blocks.*` keys or a
    /// whole Qwen3-TTS model store with `speaker_encoder.*` keys.
    pub fn from_store(store: &WeightStore) -> VoxResult<Self> {
        let prefix = if Self::has_component(store, "speaker_encoder.") {
            "speaker_encoder."
        } else if Self::has_component(store, "") {
            ""
        } else {
            return Err(VoxError::WeightLoad(
                "Speaker encoder weights must contain 'speaker_encoder.asp'/'speaker_encoder.blocks' or bare 'asp'/'blocks' prefixes".to_string(),
            ));
        };

        let initial = TdnnBlock::load(store, &format!("{prefix}blocks.0"))?;
        let first_conv = &initial.conv.weight;
        let first_conv_dims = first_conv.dims();
        let mel_dim = first_conv_dims[1];

        let mut se_blocks = Vec::with_capacity(SE_RES2NET_BLOCKS);
        for i in 1..=SE_RES2NET_BLOCKS {
            se_blocks.push(SeRes2NetBlock::load(store, &format!("{prefix}blocks.{i}"))?);
        }

        let mfa = TdnnBlock::load(store, &format!("{prefix}mfa"))?;
        let asp = AttentiveStatsPool::load(store, &format!("{prefix}asp"))?;
        let fc = Conv1dSame::load(store, &format!("{prefix}fc"))?;
        let fc_dims = fc.weight.dims();

        Ok(Self {
            prefix: prefix.to_string(),
            mel_dim,
            embed_dim: fc_dims[0],
            initial,
            se_blocks,
            mfa,
            asp,
            fc,
        })
    }

    /// Forward pass: mel `[B, T, n_mels]` -> embedding `[B, embed_dim]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (_, _, n_mels) = mel.dims3()?;
        if n_mels != self.mel_dim {
            return Err(candle_core::Error::Msg(format!(
                "speaker encoder expected mel_dim {}, got {n_mels}",
                self.mel_dim
            )));
        }

        let mut h = mel
            .to_dtype(self.initial.conv.weight.dtype())?
            .transpose(1, 2)?;

        let mut hidden_states = Vec::with_capacity(self.se_blocks.len() + 1);
        h = self.initial.forward(&h)?;
        hidden_states.push(h.clone());

        for block in &self.se_blocks {
            h = block.forward(&h)?;
            hidden_states.push(h.clone());
        }

        let mfa_inputs: Vec<&Tensor> = hidden_states.iter().skip(1).collect();
        h = Tensor::cat(&mfa_inputs, 1)?;
        h = self.mfa.forward(&h)?;
        h = self.asp.forward(&h)?;
        h = self.fc.forward(&h)?;
        h.squeeze(2)
    }

    /// Prefix detected in the loaded weight store.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Number of mel bins expected by the speaker encoder.
    pub fn mel_dim(&self) -> usize {
        self.mel_dim
    }

    /// Get expected embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn has_component(store: &WeightStore, prefix: &str) -> bool {
        store.get(&format!("{prefix}asp.conv.weight")).is_some()
            && store
                .get(&format!("{prefix}blocks.0.conv.weight"))
                .is_some()
            && store.get(&format!("{prefix}fc.weight")).is_some()
    }
}

#[derive(Clone)]
struct Conv1dSame {
    weight: Tensor,
    bias: Tensor,
    dilation: usize,
}

impl Conv1dSame {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        let weight = store.require(&format!("{prefix}.weight"))?.clone();
        let bias = store.require(&format!("{prefix}.bias"))?.clone();
        Ok(Self {
            weight,
            bias,
            dilation: 1,
        })
    }

    fn load_nested(store: &WeightStore, prefix: &str, dilation: usize) -> VoxResult<Self> {
        let weight = store.require(&format!("{prefix}.conv.weight"))?.clone();
        let bias = store.require(&format!("{prefix}.conv.bias"))?.clone();
        Ok(Self {
            weight,
            bias,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let kernel = self.weight.dim(2)?;
        let total_pad = self.dilation * kernel.saturating_sub(1);
        let left = total_pad / 2;
        let right = total_pad - left;
        let padded = reflect_pad_1d(x, left, right)?;
        let mut y = padded.conv1d(&self.weight, 0, 1, self.dilation, 1)?;
        let bias = self.bias.unsqueeze(0)?.unsqueeze(2)?;
        y = y.broadcast_add(&bias)?;
        Ok(y)
    }
}

#[derive(Clone)]
struct TdnnBlock {
    conv: Conv1dSame,
}

impl TdnnBlock {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        Ok(Self {
            conv: Conv1dSame::load_nested(store, prefix, infer_tdnn_dilation(prefix))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)?.relu()
    }
}

#[derive(Clone)]
struct SeBlock {
    conv1: Conv1dSame,
    conv2: Conv1dSame,
}

impl SeBlock {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        Ok(Self {
            conv1: Conv1dSame::load(store, &format!("{prefix}.conv1"))?,
            conv2: Conv1dSame::load(store, &format!("{prefix}.conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.mean_keepdim(2)?;
        let h = self.conv1.forward(&h)?.relu()?;
        let h = candle_nn::ops::sigmoid(&self.conv2.forward(&h)?)?;
        x.broadcast_mul(&h)
    }
}

#[derive(Clone)]
struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
}

impl Res2NetBlock {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        let mut blocks = Vec::with_capacity(RES2NET_SCALE - 1);
        for i in 0..(RES2NET_SCALE - 1) {
            blocks.push(TdnnBlock {
                conv: Conv1dSame::load_nested(
                    store,
                    &format!("{prefix}.blocks.{i}"),
                    infer_res2net_dilation(prefix),
                )?,
            });
        }
        Ok(Self { blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, channels, _) = x.dims3()?;
        let chunk = channels / RES2NET_SCALE;
        if chunk * RES2NET_SCALE != channels {
            return Err(candle_core::Error::Msg(format!(
                "Res2Net channels {channels} not divisible by scale {RES2NET_SCALE}"
            )));
        }

        let mut outputs = Vec::with_capacity(RES2NET_SCALE);
        let mut prev: Option<Tensor> = None;
        for i in 0..RES2NET_SCALE {
            let part = x.narrow(1, i * chunk, chunk)?;
            let out = if i == 0 {
                part
            } else if i == 1 {
                self.blocks[i - 1].forward(&part)?
            } else {
                let prev = prev.as_ref().expect("previous Res2Net output");
                self.blocks[i - 1].forward(&part.add(prev)?)?
            };
            prev = Some(out.clone());
            outputs.push(out);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 1)
    }
}

#[derive(Clone)]
struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Res2NetBlock,
    tdnn2: TdnnBlock,
    se: SeBlock,
}

impl SeRes2NetBlock {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        Ok(Self {
            tdnn1: TdnnBlock {
                conv: Conv1dSame::load_nested(store, &format!("{prefix}.tdnn1"), 1)?,
            },
            res2net: Res2NetBlock::load(store, &format!("{prefix}.res2net_block"))?,
            tdnn2: TdnnBlock {
                conv: Conv1dSame::load_nested(store, &format!("{prefix}.tdnn2"), 1)?,
            },
            se: SeBlock::load(store, &format!("{prefix}.se_block"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.tdnn1.forward(x)?;
        let h = self.res2net.forward(&h)?;
        let h = self.tdnn2.forward(&h)?;
        let h = self.se.forward(&h)?;
        h.add(x)
    }
}

#[derive(Clone)]
struct AttentiveStatsPool {
    tdnn: TdnnBlock,
    conv: Conv1dSame,
}

impl AttentiveStatsPool {
    fn load(store: &WeightStore, prefix: &str) -> VoxResult<Self> {
        Ok(Self {
            tdnn: TdnnBlock {
                conv: Conv1dSame::load_nested(store, &format!("{prefix}.tdnn"), 1)?,
            },
            conv: Conv1dSame::load(store, &format!("{prefix}.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, seq_len) = x.dims3()?;
        let mean = x.mean_keepdim(2)?;
        let centered = x.broadcast_sub(&mean)?;
        let std = centered
            .sqr()?
            .mean_keepdim(2)?
            .clamp(ASP_EPS, f64::INFINITY)?
            .sqrt()?;

        let mean_rep = mean.broadcast_as((batch, channels, seq_len))?;
        let std_rep = std.broadcast_as((batch, channels, seq_len))?;
        let attention = Tensor::cat(&[x, &mean_rep, &std_rep], 1)?;
        let attention = self.tdnn.forward(&attention)?.tanh()?;
        let attention = self.conv.forward(&attention)?;
        let attention = candle_nn::ops::softmax(&attention, 2)?;

        let mean = attention.mul(x)?.sum_keepdim(2)?;
        let centered = x.broadcast_sub(&mean)?;
        let std = attention
            .mul(&centered.sqr()?)?
            .sum_keepdim(2)?
            .clamp(ASP_EPS, f64::INFINITY)?
            .sqrt()?;
        Tensor::cat(&[&mean, &std], 1)
    }
}

fn reflect_pad_1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let (_, _, len) = x.dims3()?;
    if len <= 1 {
        return Err(candle_core::Error::Msg(
            "reflect padding requires temporal length > 1".into(),
        ));
    }
    if left >= len || right >= len {
        return Err(candle_core::Error::Msg(format!(
            "reflect padding left={left}, right={right} must be smaller than length {len}"
        )));
    }

    let mut indices = Vec::with_capacity(left + len + right);
    for i in (1..=left).rev() {
        indices.push(i as u32);
    }
    for i in 0..len {
        indices.push(i as u32);
    }
    for i in 0..right {
        indices.push((len - 2 - i) as u32);
    }
    let index = Tensor::new(indices.as_slice(), x.device())?;
    x.contiguous()?.index_select(&index, 2)
}

fn infer_tdnn_dilation(prefix: &str) -> usize {
    if prefix.ends_with("blocks.0") {
        1
    } else {
        infer_res2net_dilation(prefix)
    }
}

fn infer_res2net_dilation(prefix: &str) -> usize {
    if prefix.contains("blocks.1") {
        2
    } else if prefix.contains("blocks.2") {
        3
    } else if prefix.contains("blocks.3") {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceManager;
    use candle_core::{DType, Device, Tensor};
    use std::path::Path;

    #[test]
    fn test_speaker_encoder_construction_without_weights() {
        // This test verifies the module compiles and from_store fails gracefully
        // when no relevant prefixes are present.
        let store = WeightStore::new(Device::Cpu);
        let result = SpeakerEncoder::from_store(&store);
        assert!(result.is_err());
    }

    #[test]
    fn test_speaker_encoder_loads_whole_model_prefix() {
        let store = minimal_store("speaker_encoder.");
        let encoder = SpeakerEncoder::from_store(&store).unwrap();
        assert_eq!(encoder.prefix(), "speaker_encoder.");
        assert_eq!(encoder.mel_dim(), 128);
        assert_eq!(encoder.embed_dim(), 1024);
    }

    #[test]
    fn test_speaker_encoder_loads_bare_component_prefix() {
        let store = minimal_store("");
        let encoder = SpeakerEncoder::from_store(&store).unwrap();
        assert_eq!(encoder.prefix(), "");
        assert_eq!(encoder.mel_dim(), 128);
        assert_eq!(encoder.embed_dim(), 1024);
    }

    #[test]
    fn test_speaker_encoder_forward_shape() {
        let store = minimal_store("");
        let encoder = SpeakerEncoder::from_store(&store).unwrap();
        let mel = Tensor::zeros((1, 12, 128), DType::F32, &Device::Cpu).unwrap();
        let embedding = encoder.forward(&mel).unwrap();
        assert_eq!(embedding.dims(), &[1, 1024]);
    }

    #[test]
    #[ignore = "loads local 0.6B model weights; run manually for speaker encoder smoke"]
    fn test_speaker_encoder_real_base_weights_smoke() {
        let path = Path::new("weights/model-0.6b/model.safetensors");
        if !path.exists() {
            eprintln!("Skipping: {} not found", path.display());
            return;
        }

        let dev_mgr = DeviceManager::from_str("cuda").unwrap();
        let store = WeightStore::from_file(path, dev_mgr.device()).unwrap();
        let encoder = SpeakerEncoder::from_store(&store).unwrap();
        let mel = Tensor::zeros((1, 12, encoder.mel_dim()), DType::F32, dev_mgr.device()).unwrap();
        let embedding = encoder.forward(&mel).unwrap();
        assert_eq!(embedding.dims(), &[1, encoder.embed_dim()]);
    }

    fn minimal_store(prefix: &str) -> WeightStore {
        let device = Device::Cpu;
        let mut store = WeightStore::new(device.clone());
        store.insert_tensor(
            format!("{prefix}asp.conv.weight"),
            Tensor::zeros((1536, 128, 1), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            format!("{prefix}asp.conv.bias"),
            Tensor::zeros(1536, DType::F32, &device).unwrap(),
        );
        insert_conv(
            &mut store,
            &format!("{prefix}asp.tdnn.conv"),
            (128, 4608, 1),
            &device,
        );

        insert_conv(
            &mut store,
            &format!("{prefix}blocks.0.conv"),
            (512, 128, 5),
            &device,
        );
        for i in 1..=SE_RES2NET_BLOCKS {
            let block = format!("{prefix}blocks.{i}");
            insert_conv(
                &mut store,
                &format!("{block}.tdnn1.conv"),
                (512, 512, 1),
                &device,
            );
            insert_conv(
                &mut store,
                &format!("{block}.tdnn2.conv"),
                (512, 512, 1),
                &device,
            );
            for j in 0..(RES2NET_SCALE - 1) {
                insert_conv(
                    &mut store,
                    &format!("{block}.res2net_block.blocks.{j}.conv"),
                    (64, 64, 3),
                    &device,
                );
            }
            insert_conv(
                &mut store,
                &format!("{block}.se_block.conv1"),
                (128, 512, 1),
                &device,
            );
            insert_conv(
                &mut store,
                &format!("{block}.se_block.conv2"),
                (512, 128, 1),
                &device,
            );
        }

        insert_conv(
            &mut store,
            &format!("{prefix}mfa.conv"),
            (1536, 1536, 1),
            &device,
        );
        store.insert_tensor(
            format!("{prefix}fc.weight"),
            Tensor::zeros((1024, 3072, 1), DType::F32, &device).unwrap(),
        );
        store.insert_tensor(
            format!("{prefix}fc.bias"),
            Tensor::zeros(1024, DType::F32, &device).unwrap(),
        );
        store
    }

    fn insert_conv(
        store: &mut WeightStore,
        prefix: &str,
        shape: (usize, usize, usize),
        device: &Device,
    ) {
        store.insert_tensor(
            format!("{prefix}.weight"),
            Tensor::zeros(shape, DType::F32, device).unwrap(),
        );
        store.insert_tensor(
            format!("{prefix}.bias"),
            Tensor::zeros(shape.0, DType::F32, device).unwrap(),
        );
    }
}
