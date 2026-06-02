//! Lightweight Rust-only speech synthesis fallback.
//!
//! The neural Qwen3-TTS path is still represented by the Candle modules in this
//! crate. This module gives the CLI a complete, dependency-light path that
//! produces audible speech-shaped WAV output today instead of a scaffold.

use std::f32::consts::TAU;

const DEFAULT_SAMPLE_RATE: u32 = 24_000;

#[derive(Debug, Clone)]
pub struct FormantSynthConfig {
    pub sample_rate: u32,
    pub base_pitch_hz: f32,
    pub speed: f32,
}

impl Default for FormantSynthConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            base_pitch_hz: 145.0,
            speed: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    formants: [(f32, f32); 3],
    duration_s: f32,
    voiced: bool,
    noise: f32,
}

#[derive(Debug, Clone, Copy)]
struct SynthState {
    pitch_phase: f32,
    phases: [f32; 3],
    rng: u32,
}

impl Default for SynthState {
    fn default() -> Self {
        Self {
            pitch_phase: 0.0,
            phases: [0.0; 3],
            rng: 0x1234_abcd,
        }
    }
}

pub fn synthesize_formant_speech(text: &str, config: &FormantSynthConfig) -> Vec<f32> {
    let sample_rate = config.sample_rate.max(8_000);
    let speed = config.speed.clamp(0.45, 2.2);
    let mut segments = Vec::new();

    for ch in text.chars() {
        append_char_segments(ch, speed, &mut segments);
    }

    if segments.is_empty() {
        append_pause(0.25, speed, &mut segments);
    }

    let mut state = SynthState::default();
    let mut out = Vec::new();
    let mut syllable_index = 0usize;
    for segment in segments {
        render_segment(
            segment,
            sample_rate,
            config.base_pitch_hz,
            syllable_index,
            &mut state,
            &mut out,
        );
        if segment.voiced {
            syllable_index += 1;
        }
    }

    smooth_edges(&mut out, sample_rate);
    normalize(&mut out, 0.82);
    out
}

fn append_char_segments(ch: char, speed: f32, segments: &mut Vec<Segment>) {
    if ch.is_whitespace() {
        append_pause(0.09, speed, segments);
        return;
    }

    if matches!(ch, '.' | '!' | '?' | ';' | ':' | '。' | '！' | '？' | '；') {
        append_pause(0.24, speed, segments);
        return;
    }

    if matches!(ch, ',' | '，' | '、' | '；') {
        append_pause(0.15, speed, segments);
        return;
    }

    let lower = ch.to_ascii_lowercase();
    match lower {
        'a' => push_vowel(730.0, 1_090.0, 2_440.0, speed, segments),
        'e' => push_vowel(530.0, 1_840.0, 2_480.0, speed, segments),
        'i' | 'y' => push_vowel(300.0, 2_200.0, 3_000.0, speed, segments),
        'o' => push_vowel(570.0, 840.0, 2_410.0, speed, segments),
        'u' | 'w' => push_vowel(330.0, 870.0, 2_240.0, speed, segments),
        'm' | 'n' => {
            segments.push(consonant(
                280.0,
                1_000.0,
                2_200.0,
                0.045 / speed,
                true,
                0.03,
            ));
            push_vowel(500.0, 1_300.0, 2_500.0, speed * 1.1, segments);
        }
        'l' | 'r' => {
            segments.push(consonant(420.0, 1_250.0, 2_400.0, 0.04 / speed, true, 0.02));
            push_vowel(530.0, 1_450.0, 2_500.0, speed * 1.08, segments);
        }
        's' | 'z' | 'x' => {
            segments.push(consonant(
                800.0,
                2_800.0,
                4_200.0,
                0.045 / speed,
                false,
                0.72,
            ));
            push_vowel(450.0, 1_650.0, 2_700.0, speed * 1.18, segments);
        }
        'f' | 'v' | 'h' => {
            segments.push(consonant(
                650.0,
                1_900.0,
                3_300.0,
                0.04 / speed,
                false,
                0.48,
            ));
            push_vowel(500.0, 1_250.0, 2_450.0, speed * 1.16, segments);
        }
        'b' | 'p' | 'd' | 't' | 'g' | 'k' | 'q' | 'c' | 'j' => {
            segments.push(consonant(
                500.0,
                1_600.0,
                2_800.0,
                0.025 / speed,
                false,
                0.35,
            ));
            push_vowel(620.0, 1_250.0, 2_500.0, speed * 1.12, segments);
        }
        _ if is_cjk(ch) => {
            segments.push(consonant(
                450.0,
                1_700.0,
                3_000.0,
                0.026 / speed,
                false,
                0.22,
            ));
            let bucket = ch as u32 % 5;
            let (f1, f2, f3) = match bucket {
                0 => (730.0, 1_090.0, 2_440.0),
                1 => (530.0, 1_840.0, 2_480.0),
                2 => (300.0, 2_200.0, 3_000.0),
                3 => (570.0, 840.0, 2_410.0),
                _ => (420.0, 1_450.0, 2_650.0),
            };
            push_vowel(f1, f2, f3, speed * 0.8, segments);
        }
        _ => push_vowel(500.0, 1_300.0, 2_500.0, speed, segments),
    }
}

fn push_vowel(f1: f32, f2: f32, f3: f32, speed: f32, segments: &mut Vec<Segment>) {
    segments.push(Segment {
        formants: [(f1, 0.95), (f2, 0.38), (f3, 0.18)],
        duration_s: 0.095 / speed.clamp(0.45, 2.2),
        voiced: true,
        noise: 0.015,
    });
}

fn consonant(f1: f32, f2: f32, f3: f32, duration_s: f32, voiced: bool, noise: f32) -> Segment {
    Segment {
        formants: [(f1, 0.28), (f2, 0.45), (f3, 0.3)],
        duration_s,
        voiced,
        noise,
    }
}

fn append_pause(duration_s: f32, speed: f32, segments: &mut Vec<Segment>) {
    segments.push(Segment {
        formants: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        duration_s: duration_s / speed.clamp(0.45, 2.2),
        voiced: false,
        noise: 0.0,
    });
}

fn render_segment(
    segment: Segment,
    sample_rate: u32,
    base_pitch_hz: f32,
    syllable_index: usize,
    state: &mut SynthState,
    out: &mut Vec<f32>,
) {
    let samples = (segment.duration_s * sample_rate as f32).round().max(1.0) as usize;
    let inv_samples = 1.0 / samples as f32;
    let pitch = base_pitch_hz * (1.0 + 0.045 * ((syllable_index as f32) * 0.73).sin());

    for i in 0..samples {
        let t = i as f32 * inv_samples;
        let envelope = raised_cosine_envelope(t);
        let excitation = if segment.voiced {
            glottal_source(state, pitch, sample_rate)
        } else {
            0.0
        };

        let mut sample = 0.0;
        for (idx, (freq, gain)) in segment.formants.iter().copied().enumerate() {
            if freq > 0.0 && gain > 0.0 {
                state.phases[idx] = advance_phase(state.phases[idx], freq, sample_rate);
                sample += state.phases[idx].sin() * gain * excitation;
            }
        }

        if segment.noise > 0.0 {
            sample += white_noise(state) * segment.noise;
        }

        out.push(sample * envelope);
    }
}

fn glottal_source(state: &mut SynthState, pitch: f32, sample_rate: u32) -> f32 {
    state.pitch_phase = advance_phase(state.pitch_phase, pitch, sample_rate);
    let phase = state.pitch_phase / TAU;
    let pulse = if phase < 0.45 {
        (phase / 0.45 * TAU).sin().max(0.0)
    } else {
        -0.18 * ((phase - 0.45) / 0.55 * TAU).sin().abs()
    };
    pulse + 0.06 * (state.pitch_phase * 2.0).sin()
}

fn advance_phase(phase: f32, freq: f32, sample_rate: u32) -> f32 {
    (phase + TAU * freq / sample_rate as f32) % TAU
}

fn white_noise(state: &mut SynthState) -> f32 {
    state.rng = state
        .rng
        .wrapping_mul(1_664_525)
        .wrapping_add(1_013_904_223);
    let unit = ((state.rng >> 8) as f32) / ((u32::MAX >> 8) as f32);
    unit * 2.0 - 1.0
}

fn raised_cosine_envelope(t: f32) -> f32 {
    let attack = (t / 0.12).clamp(0.0, 1.0);
    let release = ((1.0 - t) / 0.16).clamp(0.0, 1.0);
    attack.min(release).powf(0.65)
}

fn smooth_edges(samples: &mut [f32], sample_rate: u32) {
    let fade = (sample_rate as usize / 200).min(samples.len() / 2);
    if fade == 0 {
        return;
    }
    for i in 0..fade {
        let gain = i as f32 / fade as f32;
        samples[i] *= gain;
        let end = samples.len() - 1 - i;
        samples[end] *= gain;
    }
}

fn normalize(samples: &mut [f32], peak: f32) {
    let max_abs = samples
        .iter()
        .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
    if max_abs <= f32::EPSILON {
        return;
    }
    let gain = peak / max_abs;
    for sample in samples {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesizes_non_silent_audio() {
        let audio =
            synthesize_formant_speech("Hello from Rust speech.", &FormantSynthConfig::default());
        assert!(audio.len() > DEFAULT_SAMPLE_RATE as usize / 2);
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!(audio.iter().any(|sample| sample.abs() > 0.05));
    }

    #[test]
    fn synthesizes_cjk_text() {
        let audio = synthesize_formant_speech("你好，世界。", &FormantSynthConfig::default());
        assert!(audio.len() > DEFAULT_SAMPLE_RATE as usize / 3);
        assert!(audio.iter().any(|sample| sample.abs() > 0.05));
    }
}
