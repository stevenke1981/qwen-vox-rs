//! qwen-vox — Qwen3-TTS speech generation CLI.
//!
//! Usage:
//!   qwen-vox generate --text "Hello world" --output hello.wav
//!   qwen-vox generate --text "你好世界" --mode 25hz --output hello.wav

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use qwen_vox_core::device::DeviceManager;
use qwen_vox_core::pipeline::{TtsPipeline, TOKENIZER_FRAME_RATE_HZ, TOKENIZER_SAMPLE_RATE};
use qwen_vox_core::talker::Talker;
use qwen_vox_core::tokenizer::Tokenizer;
use qwen_vox_core::weights::WeightStore;
use std::path::PathBuf;

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
        #[arg(long, default_value = "weights/converted/model.safetensors")]
        weights: PathBuf,

        /// Path to speech tokenizer decoder weights (SafeTensors).
        #[arg(
            long,
            default_value = "weights/alignments/tokenizer_decoder.safetensors"
        )]
        decoder_weights: PathBuf,

        /// Path to tokenizer.json.
        #[arg(long, default_value = "tokenizer.json")]
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
    },

    /// Show decoder information.
    Info,
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

            generate_qwen3_tts(
                &text,
                &output,
                &weights,
                &decoder_weights,
                &tokenizer,
                dev_mgr.device(),
                &language,
                effective_max_frames,
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
    }
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
) -> Result<()> {
    if text.trim().is_empty() {
        anyhow::bail!("text must not be empty");
    }

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
    let prompt = qwen3_prompt(text);
    let prompt_tokens = tokenizer
        .encode(&prompt)
        .with_context(|| "failed to tokenize Qwen3-TTS ChatML prompt")?;

    let talker_store = WeightStore::from_file(weights, device)
        .with_context(|| format!("failed to load talker weights {}", weights.display()))?;
    let talker = Talker::from_store(&talker_store).context("failed to build Qwen3-TTS talker")?;

    let decoder_store = WeightStore::from_file(decoder_weights, device).with_context(|| {
        format!(
            "failed to load decoder weights {}",
            decoder_weights.display()
        )
    })?;
    let pipeline = TtsPipeline::from_tokenizer_weights(decoder_store)
        .context("failed to build Qwen3-TTS codec decoder")?
        .with_talker(talker);

    let language_id = language_id(language)?;
    let frames = pipeline
        .talker()
        .ok_or_else(|| anyhow::anyhow!("talker is not attached"))?
        .generate_qwen3_base(&prompt_tokens, language_id, max_frames)
        .context("Qwen3-TTS talker failed to generate codec frames")?;
    if frames.is_empty() {
        anyhow::bail!("Qwen3-TTS generated zero codec frames");
    }

    let waveform = pipeline
        .decode_frame_codes(&frames)
        .context("Qwen3-TTS codec decoder failed")?;
    write_tensor_wav(output, TOKENIZER_SAMPLE_RATE, &waveform)?;
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

fn write_tensor_wav(
    path: &PathBuf,
    sample_rate: u32,
    waveform: &candle_core::Tensor,
) -> Result<()> {
    let flat = waveform
        .flatten_all()
        .context("failed to flatten waveform tensor")?
        .to_vec1::<f32>()
        .context("failed to extract waveform samples")?;
    write_wav(path, sample_rate, &flat)
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
}
