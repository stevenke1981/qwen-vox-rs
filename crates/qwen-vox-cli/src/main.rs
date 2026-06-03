//! qwen-vox — Qwen3-TTS speech generation CLI.
//!
//! Usage:
//!   qwen-vox generate --text "Hello world" --output hello.wav
//!   qwen-vox generate --text "你好世界" --mode 25hz --output hello.wav

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use qwen_vox_core::audio_features::qwen3_speaker_mel;
use qwen_vox_core::device::DeviceManager;
use qwen_vox_core::pipeline::{TtsPipeline, TOKENIZER_FRAME_RATE_HZ, TOKENIZER_SAMPLE_RATE};
use qwen_vox_core::sampling::SamplingConfig;
use qwen_vox_core::speaker_encoder::SpeakerEncoder;
use qwen_vox_core::speech_tokenizer_encoder::SpeechTokenizerEncoder;
use qwen_vox_core::talker::Talker;
use qwen_vox_core::tokenizer::Tokenizer;
use qwen_vox_core::weights::WeightStore;
use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const AUTO_MAX_FRAMES_SENTINEL: usize = 0;
const AUTO_MIN_SECONDS: f32 = 3.0;
const AUTO_MAX_FRAMES: usize = 512;

#[derive(Parser)]
#[command(
    name = "qwen-vox",
    version,
    about = "Qwen3-TTS speech generation (Rust)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Generate speech from text.
    Generate {
        /// Input text to synthesize.
        #[arg(short, long)]
        text: String,

        /// Output WAV file path.
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// Decoder mode: "12hz" (real-time) or "25hz" (high-quality).
        #[arg(short, long, default_value = "12hz")]
        mode: String,

        /// Path to model weights (SafeTensors).
        #[arg(long, default_value = "weights/hf_original/model.safetensors")]
        weights: PathBuf,

        /// Path to speech tokenizer decoder weights (SafeTensors).
        #[arg(
            long,
            default_value = "weights/hf_original/speech_tokenizer/model.safetensors"
        )]
        decoder_weights: PathBuf,

        /// Path to tokenizer.json or an official HF tokenizer directory.
        #[arg(long, default_value = "weights/hf_original")]
        tokenizer: PathBuf,

        /// Compute device: "cpu", "cuda", or "metal".
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Language control for Qwen3-TTS: auto, english, chinese, german, italian, or portuguese.
        #[arg(long, default_value = "english")]
        language: String,

        /// Maximum codec frames to generate. Use 0 to auto-estimate from text length.
        #[arg(long, default_value_t = AUTO_MAX_FRAMES_SENTINEL)]
        max_frames: usize,

        /// Print first N codec frames for debugging generation quality.
        #[arg(long, default_value_t = 0)]
        debug_frames: usize,

        /// Write generated codec frames to a JSON file before waveform decoding.
        #[arg(long)]
        dump_codec_frames: Option<PathBuf>,

        /// Write first-frame q0 logits top-k diagnostics to a JSON file.
        #[arg(long)]
        dump_q0_topk: Option<PathBuf>,

        /// Write first N frames of q1..q15 residual logits top-k diagnostics as JSONL.
        #[arg(long)]
        dump_residual_topk: Option<PathBuf>,

        /// Sampling temperature (lower = more deterministic). Use 0 for argmax.
        #[arg(long, default_value_t = 0.9)]
        temperature: f32,

        /// Top-k sampling: keep only top-k tokens. Use 0 to disable.
        #[arg(long, default_value_t = 50)]
        top_k: usize,

        /// Top-p (nucleus) sampling: keep tokens with cumulative prob <= top_p.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Repetition penalty (>1.0 penalizes repeated tokens). 1.0 = disabled.
        #[arg(long, default_value_t = 1.05)]
        repetition_penalty: f32,

        /// RNG seed for reproducible sampling. Omit for non-deterministic sampling.
        #[arg(long)]
        seed: Option<u64>,

        /// Output speech speed multiplier. 1.0 = original, 1.2 = faster, 0.8 = slower.
        #[arg(long, default_value_t = 1.0)]
        speed: f32,

        /// Speaker for Qwen3-TTS CustomVoice model (e.g. vivian, serena, uncle_fu,
        /// dylan, eric for Chinese; ryan, aiden for English; ono_anna for Japanese;
        /// sohee for Korean). Default: vivian.
        #[arg(long, default_value = "vivian")]
        speaker: String,
    },

    /// Show decoder information.
    Info,

    /// Decode pre-generated q0..q15 codec frames from JSON into a WAV.
    DecodeFrames {
        /// Input JSON file. Accepts either {"frames": [[...]]} or a raw [[...]] array.
        #[arg(short, long)]
        input: PathBuf,

        /// Output WAV file path.
        #[arg(short, long)]
        output: PathBuf,

        /// Path to speech tokenizer decoder weights (SafeTensors).
        #[arg(
            long,
            default_value = "weights/hf_original/speech_tokenizer/model.safetensors"
        )]
        decoder_weights: PathBuf,

        /// Compute device: "cpu", "cuda", or "metal".
        #[arg(long, default_value = "cpu")]
        device: String,
    },

    /// Voice-clone speech from reference audio using Qwen3-TTS Base weights.
    Clone {
        /// Input text to synthesize.
        #[arg(short, long)]
        text: String,

        /// Reference 24 kHz WAV used for cloning.
        #[arg(long)]
        ref_audio: PathBuf,

        /// Reference transcript. Required for official clone bookkeeping and future ICL mode.
        #[arg(long, required = true)]
        ref_text: Option<String>,

        /// Output WAV file path.
        #[arg(short, long, default_value = "clone.wav")]
        output: PathBuf,

        /// Path to Qwen3-TTS Base model directory.
        #[arg(long, default_value = "weights/model-0.6b")]
        model_dir: PathBuf,

        /// Path to tokenizer.json or an official HF tokenizer directory.
        #[arg(long, default_value = "weights/hf_original")]
        tokenizer: PathBuf,

        /// Compute device: "cpu", "cuda", or "metal".
        #[arg(long, default_value = "cuda")]
        device: String,

        /// Language control for Qwen3-TTS: auto, english, chinese, german, italian, or portuguese.
        #[arg(long, default_value = "chinese")]
        language: String,

        /// Maximum codec frames to generate. Use 0 to auto-estimate from text length.
        #[arg(long, default_value_t = AUTO_MAX_FRAMES_SENTINEL)]
        max_frames: usize,

        /// RNG seed for reproducible sampling. Omit for non-deterministic sampling.
        #[arg(long)]
        seed: Option<u64>,

        /// Write reference audio codec frames to JSON for ICL parity debugging.
        #[arg(long)]
        dump_ref_codec_frames: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("qwen_vox=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            text,
            output,
            mode,
            weights,
            decoder_weights,
            tokenizer,
            device,
            language,
            max_frames,
            debug_frames,
            dump_codec_frames,
            dump_q0_topk,
            dump_residual_topk,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            speed,
            speaker,
        } => {
            // Parse and validate device early.
            let dev_mgr = DeviceManager::from_str(&device)
                .with_context(|| format!("invalid device '{device}'"))?;
            tracing::info!("Generating speech: mode={mode}, device={device}");
            tracing::info!("Active device: {:?}", dev_mgr.device());
            tracing::info!("Text: {text}");
            tracing::info!("Output: {}", output.display());
            tracing::info!("Weights: {}", weights.display());
            tracing::info!("Decoder weights: {}", decoder_weights.display());
            tracing::info!("Tokenizer: {}", tokenizer.display());
            let effective_max_frames = resolve_max_frames(&text, max_frames);
            tracing::info!(
                "Language: {language}, max_frames={max_frames}, effective_max_frames={effective_max_frames}"
            );

            let sampling_config = if temperature <= 0.0 {
                let mut config = SamplingConfig::argmax();
                config.seed = seed;
                config
            } else {
                SamplingConfig {
                    do_sample: true,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    seed,
                }
            };
            tracing::info!(
                "Sampling: do_sample={}, temperature={}, top_k={}, top_p={}, repetition_penalty={}, seed={}",
                sampling_config.do_sample,
                sampling_config.temperature,
                sampling_config.top_k,
                sampling_config.top_p,
                sampling_config.repetition_penalty,
                sampling_config.seed.map(|v| v.to_string()).unwrap_or_else(|| "none".into())
            );
            if !(0.25..=4.0).contains(&speed) {
                anyhow::bail!("--speed must be between 0.25 and 4.0, got {speed}");
            }
            tracing::info!("Output speed multiplier: {speed}");

            let speaker_id = speaker_id(&speaker, &language)
                .with_context(|| format!("failed to resolve speaker '{speaker}'"))?;
            tracing::info!(
                "Speaker: {} (token {})",
                speaker,
                speaker_id
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "auto".into())
            );

            if let Some(path) = &dump_q0_topk {
                std::env::set_var("QWEN_VOX_DUMP_Q0_TOPK", path);
            }
            if let Some(path) = &dump_residual_topk {
                std::env::set_var("QWEN_VOX_DUMP_RESIDUAL_TOPK", path);
            }

            generate_qwen3_tts(
                &text,
                &output,
                &weights,
                &decoder_weights,
                &tokenizer,
                dev_mgr.device(),
                &language,
                effective_max_frames,
                debug_frames,
                dump_codec_frames.as_ref(),
                &sampling_config,
                speaker_id,
                speed,
            )
            .with_context(|| {
                format!(
                    "failed to generate Qwen3-TTS speech at {}",
                    output.display()
                )
            })?;

            tracing::info!("Wrote speech to {}", output.display());

            Ok(())
        }
        Commands::Info => {
            fn device_avail(name: &str, kind: qwen_vox_core::config::DeviceKind) -> String {
                let mgr = DeviceManager::new(kind);
                let status =
                    if mgr.is_cpu() && !matches!(kind, qwen_vox_core::config::DeviceKind::Cpu) {
                        "UNAVAILABLE"
                    } else {
                        "AVAILABLE"
                    };
                format!("  {name:6}  -> {status}")
            }

            println!("qwen-vox v{}", env!("CARGO_PKG_VERSION"));
            println!("Qwen3-TTS Rust implementation");
            println!();
            println!("Supported modes:");
            println!("  12hz  — Real-time interactive (Causal ConvNet + MTP)");
            println!("  25hz  — High-quality synthesis (Flow Matching DiT)");
            println!();
            println!("Devices:");
            println!(
                "{}",
                device_avail("cpu", qwen_vox_core::config::DeviceKind::Cpu)
            );
            println!(
                "{}",
                device_avail("cuda", qwen_vox_core::config::DeviceKind::Cuda)
            );
            println!(
                "{}",
                device_avail("metal", qwen_vox_core::config::DeviceKind::Metal)
            );
            Ok(())
        }
        Commands::DecodeFrames {
            input,
            output,
            decoder_weights,
            device,
        } => decode_frames_json(&input, &output, &decoder_weights, &device),
        Commands::Clone {
            text,
            ref_audio,
            ref_text,
            output,
            model_dir,
            tokenizer,
            device,
            language,
            max_frames,
            seed,
            dump_ref_codec_frames,
        } => clone_voice(
            &text,
            &ref_audio,
            ref_text.as_deref(),
            &output,
            &model_dir,
            &tokenizer,
            &device,
            &language,
            max_frames,
            seed,
            dump_ref_codec_frames.as_ref(),
        ),
    }
}

fn clone_voice(
    text: &str,
    ref_audio: &PathBuf,
    ref_text: Option<&str>,
    output: &PathBuf,
    model_dir: &PathBuf,
    tokenizer_path: &PathBuf,
    device: &str,
    language: &str,
    max_frames: usize,
    seed: Option<u64>,
    dump_ref_codec_frames: Option<&PathBuf>,
) -> Result<()> {
    let started = Instant::now();
    let config_path = model_dir.join("config.json");
    let config: Value = read_json_file(&config_path)?;
    let tts_model_type = config
        .get("tts_model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if tts_model_type != "base" {
        anyhow::bail!(
            "voice clone requires Qwen3-TTS Base weights, but {} has tts_model_type={tts_model_type}. Official Qwen3-TTS also rejects voice cloning for CustomVoice weights.",
            config_path.display()
        );
    }
    if !ref_audio.exists() {
        anyhow::bail!("reference audio does not exist: {}", ref_audio.display());
    }
    let ref_text = ref_text
        .filter(|s| !s.trim().is_empty())
        .context("ref_text is required for Qwen3-TTS ICL voice clone")?;
    if text.trim().is_empty() {
        anyhow::bail!("text must not be empty");
    }

    let dev_mgr =
        DeviceManager::from_str(device).with_context(|| format!("invalid device '{device}'"))?;
    tracing::info!(
        "Voice clone: model_dir={}, device={device}",
        model_dir.display()
    );

    let (ref_samples, ref_sample_rate) = read_wav_mono_f32(ref_audio)
        .with_context(|| format!("failed to read reference WAV {}", ref_audio.display()))?;
    tracing::info!(
        "Loaded reference audio: {} samples at {} Hz in {:.2?}",
        ref_samples.len(),
        ref_sample_rate,
        started.elapsed()
    );
    let mel = qwen3_speaker_mel(&ref_samples, ref_sample_rate, dev_mgr.device())
        .context("failed to compute Qwen3-TTS speaker mel")?;

    let model_weights = model_dir.join("model.safetensors");
    let model_store =
        WeightStore::from_file(&model_weights, dev_mgr.device()).with_context(|| {
            format!(
                "failed to load Base model weights {}",
                model_weights.display()
            )
        })?;
    let speaker_encoder =
        SpeakerEncoder::from_store(&model_store).context("failed to build speaker encoder")?;
    let speaker_embedding = speaker_encoder
        .forward(&mel)
        .context("speaker encoder failed")?;
    tracing::info!(
        "Extracted speaker embedding {:?} in {:.2?}",
        speaker_embedding.dims(),
        started.elapsed()
    );

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
    let prompt = qwen3_prompt(text);
    let prompt_tokens = tokenizer
        .encode(&prompt)
        .with_context(|| "failed to tokenize Qwen3-TTS ChatML prompt")?;
    let ref_prompt = qwen3_ref_prompt(ref_text);
    let ref_tokens = tokenizer
        .encode(&ref_prompt)
        .with_context(|| "failed to tokenize Qwen3-TTS reference prompt")?;

    let decoder_weights = model_dir.join("speech_tokenizer").join("model.safetensors");
    let decoder_store =
        WeightStore::from_file(&decoder_weights, dev_mgr.device()).with_context(|| {
            format!(
                "failed to load speech tokenizer weights {}",
                decoder_weights.display()
            )
        })?;
    let speech_encoder = SpeechTokenizerEncoder::from_store(&decoder_store)
        .context("failed to build Qwen3-TTS speech tokenizer encoder")?;
    let ref_audio_tensor = candle_core::Tensor::from_vec(
        ref_samples.clone(),
        (1, ref_samples.len()),
        dev_mgr.device(),
    )
    .context("failed to create reference audio tensor")?;
    let ref_codes = speech_encoder
        .encode_audio_codes(&ref_audio_tensor, Some(ref_samples.len()))
        .context("failed to encode reference audio into Qwen3-TTS codec frames")?;
    let ref_frames = tensor_to_codec_frames(&ref_codes)
        .context("failed to convert reference codec tensor to frames")?;
    if ref_frames.is_empty() {
        anyhow::bail!("reference audio encoded to zero codec frames");
    }
    tracing::info!(
        "Encoded reference audio to {} ICL codec frames in {:.2?}",
        ref_frames.len(),
        started.elapsed()
    );
    if let Some(path) = dump_ref_codec_frames {
        write_codec_frames_json(path, &ref_frames)?;
        tracing::info!("Wrote reference codec frame dump to {}", path.display());
    }

    let talker = Talker::from_store(&model_store).context("failed to build Base talker")?;
    let mut sampling_config = SamplingConfig::default();
    sampling_config.seed = seed;
    let effective_max_frames = resolve_max_frames(text, max_frames);
    let frames = talker
        .generate_qwen3_base_icl_clone(
            &prompt_tokens,
            &ref_tokens,
            &ref_frames,
            language_id(language)?,
            &speaker_embedding,
            effective_max_frames,
            &sampling_config,
        )
        .context("Qwen3-TTS Base talker failed to generate ICL clone codec frames")?;
    if frames.is_empty() {
        anyhow::bail!("Qwen3-TTS clone generated zero codec frames");
    }
    tracing::info!(
        "Generated {} clone codec frames in {:.2?}",
        frames.len(),
        started.elapsed()
    );

    let pipeline = TtsPipeline::from_tokenizer_weights(decoder_store)
        .context("failed to build Qwen3-TTS codec decoder")?;
    let mut decode_frames = Vec::with_capacity(ref_frames.len() + frames.len());
    decode_frames.extend_from_slice(&ref_frames);
    decode_frames.extend_from_slice(&frames);
    let waveform = pipeline
        .decode_frame_codes(&decode_frames)
        .context("Qwen3-TTS codec decoder failed")?;
    write_tensor_wav_skip_prefix(
        output,
        TOKENIZER_SAMPLE_RATE,
        &waveform,
        ref_frames.len() * qwen_vox_core::pipeline::TOKENIZER_DECODE_UPSAMPLE_RATE,
    )?;
    tracing::info!(
        "Wrote cloned speech to {} in {:.2?}",
        output.display(),
        started.elapsed()
    );
    Ok(())
}

fn decode_frames_json(
    input: &PathBuf,
    output: &PathBuf,
    decoder_weights: &PathBuf,
    device: &str,
) -> Result<()> {
    let started = Instant::now();
    let dev_mgr =
        DeviceManager::from_str(device).with_context(|| format!("invalid device '{device}'"))?;
    let frames = read_codec_frames_json(input)
        .with_context(|| format!("failed to read codec frames from {}", input.display()))?;
    if frames.is_empty() {
        anyhow::bail!("codec frame JSON contains zero frames");
    }
    tracing::info!(
        "Decoding {} codec frames from {} on device={device}",
        frames.len(),
        input.display()
    );

    let decoder_store =
        WeightStore::from_file(decoder_weights, dev_mgr.device()).with_context(|| {
            format!(
                "failed to load decoder weights {}",
                decoder_weights.display()
            )
        })?;
    let pipeline = TtsPipeline::from_tokenizer_weights(decoder_store)
        .context("failed to build Qwen3-TTS codec decoder")?;
    let waveform = pipeline
        .decode_frame_codes(&frames)
        .context("Qwen3-TTS codec decoder failed")?;
    write_tensor_wav(output, TOKENIZER_SAMPLE_RATE, &waveform)?;
    tracing::info!(
        "Decoded frames to {} in {:.2?}",
        output.display(),
        started.elapsed()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_qwen3_tts(
    text: &str,
    output: &PathBuf,
    weights: &PathBuf,
    decoder_weights: &PathBuf,
    tokenizer_path: &PathBuf,
    device: &candle_core::Device,
    language: &str,
    max_frames: usize,
    debug_frames: usize,
    dump_codec_frames: Option<&PathBuf>,
    sampling_config: &SamplingConfig,
    speaker_id: Option<u32>,
    speed: f32,
) -> Result<()> {
    let started = Instant::now();
    if text.trim().is_empty() {
        anyhow::bail!("text must not be empty");
    }

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
    tracing::info!("Loaded tokenizer in {:.2?}", started.elapsed());

    let prompt = qwen3_prompt(text);
    let prompt_tokens = tokenizer
        .encode(&prompt)
        .with_context(|| "failed to tokenize Qwen3-TTS ChatML prompt")?;
    tracing::info!(
        "Tokenized prompt into {} tokens in {:.2?}",
        prompt_tokens.len(),
        started.elapsed()
    );

    let talker_store = WeightStore::from_file(weights, device)
        .with_context(|| format!("failed to load talker weights {}", weights.display()))?;
    tracing::info!("Loaded talker weights in {:.2?}", started.elapsed());

    let talker = Talker::from_store(&talker_store).context("failed to build Qwen3-TTS talker")?;
    tracing::info!("Built Qwen3-TTS talker in {:.2?}", started.elapsed());

    let decoder_store = WeightStore::from_file(decoder_weights, device).with_context(|| {
        format!(
            "failed to load decoder weights {}",
            decoder_weights.display()
        )
    })?;
    tracing::info!("Loaded decoder weights in {:.2?}", started.elapsed());

    let pipeline = TtsPipeline::from_tokenizer_weights(decoder_store)
        .context("failed to build Qwen3-TTS codec decoder")?
        .with_talker(talker);
    tracing::info!("Built codec decoder in {:.2?}", started.elapsed());

    let language_id = language_id(language)?;
    let frames = pipeline
        .talker()
        .ok_or_else(|| anyhow::anyhow!("talker is not attached"))?
        .generate_qwen3_base(
            &prompt_tokens,
            language_id,
            speaker_id,
            max_frames,
            sampling_config,
        )
        .context("Qwen3-TTS talker failed to generate codec frames")?;
    tracing::info!(
        "Generated {} codec frames in {:.2?}",
        frames.len(),
        started.elapsed()
    );
    if frames.is_empty() {
        anyhow::bail!("Qwen3-TTS generated zero codec frames");
    }

    if debug_frames > 0 {
        log_codec_frames(&frames, debug_frames);
    }

    if let Some(path) = dump_codec_frames {
        write_codec_frames_json(path, &frames)?;
        tracing::info!("Wrote codec frame dump to {}", path.display());
    }

    let waveform = pipeline
        .decode_frame_codes(&frames)
        .context("Qwen3-TTS codec decoder failed")?;
    tracing::info!("Decoded waveform in {:.2?}", started.elapsed());
    write_tensor_wav_with_speed(output, TOKENIZER_SAMPLE_RATE, &waveform, speed)?;
    Ok(())
}

fn resolve_max_frames(text: &str, requested: usize) -> usize {
    if requested != AUTO_MAX_FRAMES_SENTINEL {
        return requested;
    }
    auto_max_frames(text)
}

fn auto_max_frames(text: &str) -> usize {
    let seconds = estimate_speech_seconds(text).clamp(AUTO_MIN_SECONDS, 60.0);
    ((seconds * TOKENIZER_FRAME_RATE_HZ).ceil() as usize + 8).min(AUTO_MAX_FRAMES)
}

fn estimate_speech_seconds(text: &str) -> f32 {
    let cjk_chars = text.chars().filter(|&ch| is_cjk(ch)).count() as f32;
    let words = text
        .split_whitespace()
        .filter(|part| part.chars().any(|ch| ch.is_ascii_alphanumeric()))
        .count() as f32;
    let punctuation_pauses = text
        .chars()
        .filter(|ch| {
            matches!(
                ch,
                '.' | ',' | ';' | ':' | '?' | '!' | '。' | '，' | '；' | '：' | '？' | '！'
            )
        })
        .count() as f32
        * 0.12;

    0.8 + (cjk_chars / 4.5) + (words / 2.5) + punctuation_pauses
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
    )
}

fn qwen3_prompt(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

fn qwen3_ref_prompt(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n")
}

fn language_id(language: &str) -> Result<Option<u32>> {
    match language.to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "english" | "en" => Ok(Some(2050)),
        "chinese" | "zh" | "zh-tw" | "zh-cn" => Ok(Some(2055)),
        "german" | "de" => Ok(Some(2053)),
        "italian" | "it" => Ok(Some(2070)),
        "portuguese" | "pt" => Ok(Some(2071)),
        other => anyhow::bail!("unsupported Qwen3-TTS language: {other}"),
    }
}

/// Map speaker name to talker token id for Qwen3-TTS CustomVoice model.
///
/// Source: `weights/hf_original/config.json` `talker_config.spk_id`.
/// Without a speaker token the model has no voice identity and produces noise.
fn speaker_id(speaker: &str, _language: &str) -> Result<Option<u32>> {
    if speaker.is_empty()
        || speaker.eq_ignore_ascii_case("none")
        || speaker.eq_ignore_ascii_case("auto")
    {
        return Ok(None);
    }
    let token = match speaker.to_ascii_lowercase().as_str() {
        // Chinese speakers
        "vivian" => 3065,
        "serena" => 3066,
        "uncle_fu" | "uncle-fu" | "unclefu" => 3010,
        "dylan" => 2878,
        "eric" => 2875,
        // English speakers
        "ryan" => 3061,
        "aiden" => 2861,
        // Japanese / Korean
        "ono_anna" | "ono-anna" | "onoanna" => 2873,
        "sohee" => 2864,
        other => anyhow::bail!("unsupported Qwen3-TTS speaker: {other}"),
    };
    Ok(Some(token))
}

/// Post-process raw f32 waveform samples for safe WAV output.
///
/// Steps:
/// 1. Replace non-finite samples (NaN, ±Inf) with 0.0.
/// 2. Remove DC offset (subtract mean).
/// 3. Peak-normalize to `target_peak` (default 0.90).
/// 4. Apply a smooth soft-limiter above the threshold to avoid hard clipping.
fn normalize_waveform_for_wav(samples: &mut [f32]) {
    normalize_waveform_for_wav_with_target(samples, 0.90);
}

fn normalize_waveform_for_wav_with_target(samples: &mut [f32], target_peak: f32) {
    if samples.is_empty() {
        return;
    }

    // 1. Replace non-finite samples with 0.0
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
    }

    // 2. Remove DC offset
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    if mean.abs() > 1e-8 {
        for s in samples.iter_mut() {
            *s -= mean;
        }
    }

    // 3. Peak normalize to target_peak
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak < 1e-8 {
        return; // silence, nothing to normalize
    }
    let gain = target_peak / peak;
    for s in samples.iter_mut() {
        *s *= gain;
    }

    // 4. Soft limiter — cubic knee above threshold.
    //    Below `threshold`: linear pass-through.
    //    Above `threshold`: smooth compression toward 1.0.
    let threshold = target_peak;
    let headroom = 1.0 - threshold;
    if headroom > 0.0 {
        for s in samples.iter_mut() {
            let abs_s = s.abs();
            if abs_s > threshold {
                let x = (abs_s - threshold) / headroom;
                let soft = threshold + headroom * (1.0 - (1.0 - x).powi(2));
                *s = s.signum() * soft.min(1.0);
            }
        }
    }
}

fn write_tensor_wav(
    path: &PathBuf,
    sample_rate: u32,
    waveform: &candle_core::Tensor,
) -> Result<()> {
    write_tensor_wav_with_speed(path, sample_rate, waveform, 1.0)
}

fn write_tensor_wav_skip_prefix(
    path: &PathBuf,
    sample_rate: u32,
    waveform: &candle_core::Tensor,
    skip_samples: usize,
) -> Result<()> {
    let flat = waveform
        .flatten_all()
        .context("failed to flatten waveform tensor")?
        .to_vec1::<f32>()
        .context("failed to extract waveform samples")?;
    let start = skip_samples.min(flat.len());
    let mut output = flat[start..].to_vec();
    normalize_waveform_for_wav(&mut output);
    write_wav(path, sample_rate, &output)
}

fn write_tensor_wav_with_speed(
    path: &PathBuf,
    sample_rate: u32,
    waveform: &candle_core::Tensor,
    speed: f32,
) -> Result<()> {
    let mut flat = waveform
        .flatten_all()
        .context("failed to flatten waveform tensor")?
        .to_vec1::<f32>()
        .context("failed to extract waveform samples")?;
    if (speed - 1.0).abs() > 1e-4 {
        flat = stretch_waveform_speed(&flat, speed);
    }
    normalize_waveform_for_wav(&mut flat);
    write_wav(path, sample_rate, &flat)
}

fn stretch_waveform_speed(samples: &[f32], speed: f32) -> Vec<f32> {
    if samples.len() < 2 || (speed - 1.0).abs() <= 1e-4 {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as f32) / speed).round().max(1.0) as usize;
    if out_len == 1 {
        return vec![samples[0]];
    }

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f32) * speed;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = src_pos - left as f32;
        let value = samples[left.min(samples.len() - 1)] * (1.0 - frac) + samples[right] * frac;
        out.push(value);
    }
    out
}

fn read_wav_mono_f32(path: &PathBuf) -> Result<(Vec<f32>, usize)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let raw = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.clamp(1, 32);
            let scale = (1_i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f32 / scale).clamp(-1.0, 1.0)))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    if raw.is_empty() {
        anyhow::bail!("reference WAV contains no samples: {}", path.display());
    }
    if channels == 1 {
        return Ok((raw, spec.sample_rate as usize));
    }

    let mut mono = Vec::with_capacity(raw.len() / channels);
    for frame in raw.chunks_exact(channels) {
        mono.push(frame.iter().sum::<f32>() / channels as f32);
    }
    Ok((mono, spec.sample_rate as usize))
}

fn write_wav(path: &PathBuf, sample_rate: u32, samples: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(pcm)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Log codec frame diagnostics for debugging generation quality.
fn log_codec_frames(frames: &[[u16; 16]], max_log: usize) {
    let total = frames.len();
    tracing::info!("── Codec Frame Diagnostics ({total} frames) ──");

    // Check for repeated consecutive frames
    let mut repeated_pairs = 0usize;
    for w in frames.windows(2) {
        if w[0] == w[1] {
            repeated_pairs += 1;
        }
    }

    // Check code range
    let mut max_code: u16 = 0;
    let mut min_code: u16 = u16::MAX;
    for frame in frames {
        for &c in frame {
            max_code = max_code.max(c);
            min_code = min_code.min(c);
        }
    }

    // Check q0 distribution (first code level is most important)
    let q0_values: Vec<u16> = frames.iter().map(|f| f[0]).collect();
    let unique_q0 = {
        let mut v = q0_values.clone();
        v.sort();
        v.dedup();
        v.len()
    };

    tracing::info!(
        "  code range: [{min_code}, {max_code}], unique q0 values: {unique_q0}/{total}, repeated consecutive frames: {repeated_pairs}"
    );

    // Log first N frames
    let log_count = max_log.min(total);
    for (i, frame) in frames.iter().take(log_count).enumerate() {
        let codes: Vec<String> = frame.iter().map(|c| format!("{c:4}")).collect();
        tracing::info!("  frame[{i:3}]: [{}]", codes.join(", "));
    }
    if total > log_count {
        tracing::info!("  ... ({total} total frames, showing first {log_count})");
    }

    // Warn about potential issues
    if repeated_pairs > total / 4 {
        tracing::warn!(
            "  ⚠ {repeated_pairs}/{total} consecutive frame pairs are identical — possible argmax collapse"
        );
    }
    if unique_q0 <= 2 && total > 4 {
        tracing::warn!(
            "  ⚠ Only {unique_q0} unique q0 values across {total} frames — generation may be stuck"
        );
    }
}

fn write_codec_frames_json(path: &PathBuf, frames: &[[u16; 16]]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"format\": \"qwen-vox-codec-frames-v1\",")?;
    writeln!(writer, "  \"frame_rate_hz\": {TOKENIZER_FRAME_RATE_HZ},")?;
    writeln!(writer, "  \"codebooks\": 16,")?;
    writeln!(writer, "  \"frame_count\": {},", frames.len())?;
    writeln!(writer, "  \"frames\": [")?;
    for (i, frame) in frames.iter().enumerate() {
        write!(writer, "    [")?;
        for (j, code) in frame.iter().enumerate() {
            if j > 0 {
                write!(writer, ", ")?;
            }
            write!(writer, "{code}")?;
        }
        if i + 1 == frames.len() {
            writeln!(writer, "]")?;
        } else {
            writeln!(writer, "],")?;
        }
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")?;
    Ok(())
}

fn tensor_to_codec_frames(tensor: &candle_core::Tensor) -> Result<Vec<[u16; 16]>> {
    let dims = tensor.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[2] != 16 {
        anyhow::bail!("expected codec tensor shape [1, frames, 16], got {dims:?}");
    }
    let frame_count = dims[1];
    let values: Vec<u32> = tensor
        .flatten_all()
        .context("failed to flatten codec tensor")?
        .to_vec1()
        .context("failed to read codec tensor")?;
    let mut frames = Vec::with_capacity(frame_count);
    for chunk in values.chunks_exact(16) {
        let mut frame = [0u16; 16];
        for (dst, &src) in frame.iter_mut().zip(chunk) {
            if src > u16::MAX as u32 {
                anyhow::bail!("codec id {src} exceeds u16 range");
            }
            *dst = src as u16;
        }
        frames.push(frame);
    }
    Ok(frames)
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CodecFramesJson {
    Wrapped { frames: Vec<[u16; 16]> },
    Raw(Vec<[u16; 16]>),
}

fn read_codec_frames_json(path: &PathBuf) -> Result<Vec<[u16; 16]>> {
    let parsed: CodecFramesJson = read_json_file(path)?;
    let frames = match parsed {
        CodecFramesJson::Wrapped { frames } => frames,
        CodecFramesJson::Raw(frames) => frames,
    };
    Ok(frames)
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<T> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("failed to parse {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_max_frames_is_preserved() {
        assert_eq!(resolve_max_frames("hello", 12), 12);
    }

    #[test]
    fn auto_max_frames_gives_audible_minimum() {
        assert!(resolve_max_frames("hello", 0) >= 45);
    }

    #[test]
    fn auto_max_frames_scales_for_long_text() {
        let short = resolve_max_frames("hello", 0);
        let long = resolve_max_frames(
            "Hello from Qwen three TTS. This sentence is deliberately longer and should get more codec frames.",
            0,
        );
        assert!(long > short);
    }

    // ── normalize_waveform_for_wav tests ──────────────────────────────────────

    #[test]
    fn normalize_handles_nan_and_inf() {
        let mut samples = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, -0.5];
        normalize_waveform_for_wav(&mut samples);
        for s in &samples {
            assert!(s.is_finite(), "non-finite sample after normalization: {s}");
        }
    }

    #[test]
    fn normalize_removes_dc_offset() {
        let mut samples = vec![1.0, 1.0, 1.0, 1.0, -3.0];
        // mean = 0.2, after DC removal: [0.8, 0.8, 0.8, 0.8, -3.2]
        normalize_waveform_for_wav_with_target(&mut samples, 0.90);
        let mean_after: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            mean_after.abs() < 0.05,
            "DC offset not removed, mean = {mean_after}"
        );
    }

    #[test]
    fn normalize_peak_within_target() {
        let mut samples = vec![0.0, 0.5, -2.0, 1.5, 0.0];
        normalize_waveform_for_wav_with_target(&mut samples, 0.90);
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak <= 1.0, "peak {peak} exceeds 1.0 after normalization");
        assert!(
            peak >= 0.85,
            "peak {peak} too low — normalization should reach ~0.90"
        );
    }

    #[test]
    fn normalize_preserves_silence() {
        let mut samples = vec![0.0; 100];
        normalize_waveform_for_wav(&mut samples);
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn normalize_empty_is_noop() {
        let mut samples: Vec<f32> = vec![];
        normalize_waveform_for_wav(&mut samples);
        assert!(samples.is_empty());
    }

    #[test]
    fn speed_stretch_changes_length() {
        let samples = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        assert_eq!(stretch_waveform_speed(&samples, 2.0).len(), 4);
        assert_eq!(stretch_waveform_speed(&samples, 0.5).len(), 16);
        assert_eq!(stretch_waveform_speed(&samples, 1.0).len(), samples.len());
    }

    #[test]
    fn normalize_hot_signal_gets_limited() {
        // Simulate the RMS=0.82, peak≈1.0 scenario from the bug report
        let mut samples: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 24000.0;
                0.95 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();
        normalize_waveform_for_wav(&mut samples);
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak <= 1.0, "peak {peak} exceeds 1.0");
        // All samples should be finite and within [-1, 1]
        for s in &samples {
            assert!(s.is_finite());
            assert!(*s >= -1.0 && *s <= 1.0);
        }
    }

    #[test]
    fn normalize_soft_limiter_preserves_below_threshold() {
        // Use zero-mean samples so DC removal doesn't shift values.
        let mut samples = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.3];
        let original = samples.clone();
        normalize_waveform_for_wav_with_target(&mut samples, 0.90);
        // After DC removal (mean≈0) and peak normalization, relative
        // proportions should be preserved (linear scaling).
        let peak_orig = original.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let gain = 0.90 / peak_orig;
        for (orig, norm) in original.iter().zip(samples.iter()) {
            let expected = orig * gain;
            assert!(
                (norm - expected).abs() < 1e-4,
                "below-threshold sample distorted: {orig} -> {norm} (expected {expected})"
            );
        }
    }
}
