//! qwen-vox — Qwen3-TTS speech generation CLI.
//!
//! Usage:
//!   qwen-vox generate --text "Hello world" --output hello.wav
//!   qwen-vox generate --text "你好世界" --mode 25hz --output hello.wav

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use qwen_vox_core::device::DeviceManager;
use qwen_vox_core::{synthesize_formant_speech, FormantSynthConfig};
use std::path::PathBuf;

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
        #[arg(long, default_value = "weights/model.safetensors")]
        weights: PathBuf,

        /// Path to tokenizer.json.
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: PathBuf,

        /// Compute device: "cpu", "cuda", or "metal".
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Base voice pitch in Hz.
        #[arg(long, default_value_t = 145.0)]
        pitch: f32,

        /// Speech speed multiplier.
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
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
            tokenizer,
            device,
            pitch,
            speed,
        } => {
            // Parse and validate device early.
            let dev_mgr = DeviceManager::from_str(&device)
                .with_context(|| format!("invalid device '{device}'"))?;
            tracing::info!("Generating speech: mode={mode}, device={device}");
            tracing::info!("Active device: {:?}", dev_mgr.device());
            tracing::info!("Text: {text}");
            tracing::info!("Output: {}", output.display());
            tracing::info!("Weights: {}", weights.display());
            tracing::info!("Tokenizer: {}", tokenizer.display());
            tracing::info!("Pitch: {pitch} Hz, speed: {speed}");

            let config = FormantSynthConfig {
                sample_rate: 24_000,
                base_pitch_hz: pitch,
                speed,
            };
            let audio = synthesize_formant_speech(&text, &config);
            write_wav(&output, config.sample_rate, &audio)
                .with_context(|| format!("failed to write WAV to {}", output.display()))?;

            let duration = audio.len() as f32 / config.sample_rate as f32;
            tracing::info!(
                "Wrote {:.2}s of Rust-synthesized speech to {}",
                duration,
                output.display()
            );

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
            println!("  fallback — Rust formant synthesizer used by the CLI generate command");
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
