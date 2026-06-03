//! Audio feature extraction for Qwen3-TTS speaker cloning.
//!
//! The speaker encoder path follows the official Hugging Face preprocessing:
//! 24 kHz audio, reflect padding, Hann-window STFT, Slaney-normalized mel
//! filterbank, and log dynamic range compression.

use crate::error::{VoxError, VoxResult};
use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::PI;

pub const QWEN3_SPEAKER_SAMPLE_RATE: usize = 24_000;
pub const QWEN3_SPEAKER_N_FFT: usize = 1024;
pub const QWEN3_SPEAKER_NUM_MELS: usize = 128;
pub const QWEN3_SPEAKER_HOP_SIZE: usize = 256;
pub const QWEN3_SPEAKER_WIN_SIZE: usize = 1024;
pub const QWEN3_SPEAKER_FMIN: f32 = 0.0;
pub const QWEN3_SPEAKER_FMAX: f32 = 12_000.0;

/// Compute official-style Qwen3-TTS speaker mel features.
///
/// Returns `[1, frames, 128]`, ready for `SpeakerEncoder::forward`.
pub fn qwen3_speaker_mel(audio: &[f32], sample_rate: usize, device: &Device) -> VoxResult<Tensor> {
    if sample_rate != QWEN3_SPEAKER_SAMPLE_RATE {
        return Err(VoxError::Inference(format!(
            "Qwen3-TTS speaker encoder requires 24 kHz reference audio, got {sample_rate} Hz"
        )));
    }
    if audio.len() < QWEN3_SPEAKER_N_FFT {
        return Err(VoxError::Inference(format!(
            "reference audio is too short: need at least {} samples, got {}",
            QWEN3_SPEAKER_N_FFT,
            audio.len()
        )));
    }

    let mel = mel_spectrogram(
        audio,
        QWEN3_SPEAKER_N_FFT,
        QWEN3_SPEAKER_NUM_MELS,
        sample_rate,
        QWEN3_SPEAKER_HOP_SIZE,
        QWEN3_SPEAKER_WIN_SIZE,
        QWEN3_SPEAKER_FMIN,
        QWEN3_SPEAKER_FMAX,
    )?;
    let frames = mel.len() / QWEN3_SPEAKER_NUM_MELS;
    let tensor = Tensor::from_vec(mel, (1, QWEN3_SPEAKER_NUM_MELS, frames), device)?;
    Ok(tensor.transpose(1, 2)?)
}

fn mel_spectrogram(
    audio: &[f32],
    n_fft: usize,
    num_mels: usize,
    sample_rate: usize,
    hop_size: usize,
    win_size: usize,
    fmin: f32,
    fmax: f32,
) -> VoxResult<Vec<f32>> {
    let padding = (n_fft - hop_size) / 2;
    let padded = reflect_pad_audio(audio, padding)?;
    if padded.len() < n_fft {
        return Err(VoxError::Inference(
            "padded audio shorter than n_fft".into(),
        ));
    }
    let frames = 1 + (padded.len() - n_fft) / hop_size;
    let bins = n_fft / 2 + 1;
    let window = hann_window(win_size);
    let mel_basis = mel_filterbank(sample_rate, n_fft, num_mels, fmin, fmax);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut spectrum = vec![0.0f32; bins * frames];
    let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];

    for frame in 0..frames {
        let offset = frame * hop_size;
        for i in 0..n_fft {
            let sample = if i < win_size {
                padded[offset + i] * window[i]
            } else {
                0.0
            };
            buffer[i] = Complex32::new(sample, 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..bins {
            let value = buffer[bin];
            spectrum[bin * frames + frame] = (value.norm_sqr() + 1e-9).sqrt();
        }
    }

    let mut mel = vec![0.0f32; num_mels * frames];
    for m in 0..num_mels {
        for frame in 0..frames {
            let mut sum = 0.0f32;
            for bin in 0..bins {
                sum += mel_basis[m * bins + bin] * spectrum[bin * frames + frame];
            }
            mel[m * frames + frame] = sum.max(1e-5).ln();
        }
    }
    Ok(mel)
}

fn reflect_pad_audio(audio: &[f32], pad: usize) -> VoxResult<Vec<f32>> {
    if pad == 0 {
        return Ok(audio.to_vec());
    }
    if audio.len() <= pad {
        return Err(VoxError::Inference(format!(
            "reference audio length {} must be greater than reflect padding {pad}",
            audio.len()
        )));
    }
    let mut padded = Vec::with_capacity(audio.len() + pad * 2);
    for i in (1..=pad).rev() {
        padded.push(audio[i]);
    }
    padded.extend_from_slice(audio);
    for i in 0..pad {
        padded.push(audio[audio.len() - 2 - i]);
    }
    Ok(padded)
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / size as f32).cos())
        .collect()
}

fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
    let fft_freqs: Vec<f32> = (0..bins)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut weights = vec![0.0f32; n_mels * bins];
    for m in 0..n_mels {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];
        let enorm = 2.0 / (upper - lower);
        for (bin, &freq) in fft_freqs.iter().enumerate() {
            let lower_slope = (freq - lower) / (center - lower);
            let upper_slope = (upper - freq) / (upper - center);
            weights[m * bins + bin] = lower_slope.min(upper_slope).max(0.0) * enorm;
        }
    }
    weights
}

fn hz_to_mel(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * f_sp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_qwen3_speaker_mel_shape() {
        let samples = vec![0.0f32; QWEN3_SPEAKER_SAMPLE_RATE];
        let mel = qwen3_speaker_mel(&samples, QWEN3_SPEAKER_SAMPLE_RATE, &Device::Cpu).unwrap();
        assert_eq!(mel.dim(0).unwrap(), 1);
        assert_eq!(mel.dim(2).unwrap(), QWEN3_SPEAKER_NUM_MELS);
        assert!(mel.dim(1).unwrap() > 0);
    }

    #[test]
    fn test_qwen3_speaker_mel_rejects_wrong_sample_rate() {
        let samples = vec![0.0f32; QWEN3_SPEAKER_SAMPLE_RATE];
        assert!(qwen3_speaker_mel(&samples, 16_000, &Device::Cpu).is_err());
    }
}
