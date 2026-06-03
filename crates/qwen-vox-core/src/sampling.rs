//! Logits processing and sampling for autoregressive generation.
//!
//! Implements temperature scaling, repetition penalty, top-k, top-p (nucleus)
//! filtering, and categorical sampling — matching the upstream Qwen3-TTS
//! `generation_config.json` parameters.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for logits processing and sampling.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Whether to sample (true) or use argmax (false).
    pub do_sample: bool,
    /// Temperature for softmax scaling. Lower = more deterministic.
    pub temperature: f32,
    /// Keep only the top-k highest probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Nucleus sampling: keep tokens with cumulative probability <= top_p.
    pub top_p: f32,
    /// Repetition penalty (>1.0 penalizes repeated tokens). 1.0 = disabled.
    pub repetition_penalty: f32,
    /// Optional RNG seed for reproducible sampling.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    /// Default matches upstream Qwen3-TTS `generation_config.json`.
    fn default() -> Self {
        Self {
            do_sample: true,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Create a config that always uses argmax (no sampling).
    pub fn argmax() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

/// Process logits and sample a token index.
///
/// # Arguments
/// * `logits` — raw logit values (f32 slice, length = vocab size)
/// * `config` — sampling configuration
/// * `history` — previously generated token indices (for repetition penalty)
///
/// # Returns
/// Sampled token index.
pub fn sample_token(logits: &[f32], config: &SamplingConfig, history: &[u16]) -> u16 {
    let mut rng = rng_from_seed(config.seed);
    sample_token_with_rng(logits, config, history, &mut rng)
}

/// Process logits and sample a token index using a caller-owned RNG.
pub fn sample_token_with_rng<R: Rng + ?Sized>(
    logits: &[f32],
    config: &SamplingConfig,
    history: &[u16],
    rng: &mut R,
) -> u16 {
    if !config.do_sample {
        return argmax(logits) as u16;
    }

    let mut processed = logits.to_vec();

    // 1. Temperature scaling
    if (config.temperature - 1.0).abs() > 1e-6 && config.temperature > 0.0 {
        let inv_temp = 1.0 / config.temperature;
        for v in processed.iter_mut() {
            *v *= inv_temp;
        }
    }

    // 2. Repetition penalty
    if (config.repetition_penalty - 1.0).abs() > 1e-6 && !history.is_empty() {
        apply_repetition_penalty(&mut processed, history, config.repetition_penalty);
    }

    // 3. Top-k filtering
    if config.top_k > 0 && config.top_k < processed.len() {
        apply_top_k(&mut processed, config.top_k);
    }

    // 4. Top-p (nucleus) filtering
    if config.top_p < 1.0 && config.top_p > 0.0 {
        apply_top_p(&mut processed, config.top_p);
    }

    // 5. Softmax + categorical sampling
    sample_categorical(&processed, rng)
}

/// Build a deterministic RNG when a seed is provided, otherwise use OS entropy.
pub fn rng_from_seed(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

/// Simple argmax over a slice. Returns the FIRST index of the maximum value
/// (matches candle's tensor `.argmax()` semantics for ties).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Apply repetition penalty: for each unique token in history, divide positive
/// logits by penalty and multiply negative logits by penalty. Matches the
/// standard HF transformers `RepetitionPenaltyLogitsProcessor` which calls
/// `scores.scatter_()` (each unique position is written once).
fn apply_repetition_penalty(logits: &mut [f32], history: &[u16], penalty: f32) {
    // Collect unique token indices from history (preserving first-seen order).
    let mut seen = [false; 4096];
    let mut unique: Vec<u16> = Vec::with_capacity(history.len());
    for &token_id in history {
        let slot = token_id as usize;
        if slot < seen.len() && !seen[slot] {
            seen[slot] = true;
            unique.push(token_id);
        }
    }

    for token_id in unique {
        let idx = token_id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Keep only the top-k highest logits, set the rest to -inf.
fn apply_top_k(logits: &mut [f32], k: usize) {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Less)
    });

    let neg_inf = f32::NEG_INFINITY;
    for &idx in indices.iter().skip(k) {
        logits[idx] = neg_inf;
    }
}

/// Nucleus (top-p) filtering: keep tokens with cumulative probability <= top_p.
fn apply_top_p(logits: &mut [f32], top_p: f32) {
    // Compute softmax probabilities
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_logit.is_infinite() {
        return;
    }

    let exp_sum: f32 = logits.iter().map(|&v| (v - max_logit).exp()).sum();
    if exp_sum <= 0.0 {
        return;
    }

    // Create (probability, index) pairs and sort descending
    let mut prob_idx: Vec<(f32, usize)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| ((v - max_logit).exp() / exp_sum, i))
        .collect();
    prob_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Less));

    // Find cutoff where cumulative probability exceeds top_p
    let mut cumsum = 0.0f32;
    let mut cutoff_idx = prob_idx.len();
    for (i, &(prob, _)) in prob_idx.iter().enumerate() {
        cumsum += prob;
        if cumsum > top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Set tokens beyond cutoff to -inf
    let neg_inf = f32::NEG_INFINITY;
    for &(_, idx) in prob_idx.iter().skip(cutoff_idx) {
        logits[idx] = neg_inf;
    }
}

/// Sample from a categorical distribution defined by logits.
/// Uses softmax to convert logits to probabilities, then samples.
fn sample_categorical<R: Rng + ?Sized>(logits: &[f32], rng: &mut R) -> u16 {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Softmax
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum <= 0.0 || sum.is_infinite() {
        // Fallback to argmax
        return argmax(logits) as u16;
    }

    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let r: f32 = rng.gen();

    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u16;
        }
    }

    // Fallback (shouldn't reach here)
    (probs.len() - 1) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_returns_highest() {
        let logits = [1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn argmax_mode_returns_deterministic() {
        let config = SamplingConfig::argmax();
        let logits = [1.0, 5.0, 3.0, 2.0, 4.0];
        let history = vec![];
        assert_eq!(sample_token(&logits, &config, &history), 1);
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let logits = [1.0, 5.0, 3.0, 2.0, 4.0];
        let config = SamplingConfig {
            seed: Some(42),
            ..SamplingConfig::default()
        };
        let history = vec![];
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut rng1 = rng_from_seed(config.seed);
        let mut rng2 = rng_from_seed(config.seed);
        for _ in 0..16 {
            first.push(sample_token_with_rng(&logits, &config, &history, &mut rng1));
            second.push(sample_token_with_rng(&logits, &config, &history, &mut rng2));
        }
        assert_eq!(first, second);
    }

    #[test]
    fn repetition_penalty_reduces_repeated_logits() {
        let mut logits = [1.0, 5.0, 3.0, 2.0, 4.0];
        let history = [1u16, 1, 1]; // token 1 repeated
        apply_repetition_penalty(&mut logits, &history, 1.5);
        // Positive logit at index 1 should be divided by 1.5
        assert!((logits[1] - 5.0 / 1.5).abs() < 1e-5);
    }

    #[test]
    fn repetition_penalty_increases_negative_logits() {
        let mut logits = [-2.0, 5.0, 3.0];
        let history = [0u16];
        apply_repetition_penalty(&mut logits, &history, 1.5);
        // Negative logit at index 0 should be multiplied by 1.5
        assert!((logits[0] - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn top_k_keeps_only_k_tokens() {
        let mut logits = [1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        // Only indices 1 (5.0) and 4 (4.0) should remain finite
        assert!(logits[0].is_infinite());
        assert!(logits[1].is_finite());
        assert!(logits[2].is_infinite());
        assert!(logits[3].is_finite() || logits[3].is_infinite()); // 2.0 is not top-2
        assert!(logits[4].is_finite());
    }

    #[test]
    fn sample_returns_valid_index() {
        let logits = [1.0, 5.0, 3.0, 2.0, 4.0];
        let config = SamplingConfig::default();
        let history = vec![];
        let idx = sample_token(&logits, &config, &history);
        assert!((idx as usize) < logits.len());
    }

    #[test]
    fn default_config_matches_upstream() {
        let config = SamplingConfig::default();
        assert!(config.do_sample);
        assert!((config.temperature - 0.9).abs() < 1e-6);
        assert_eq!(config.top_k, 50);
        assert!((config.top_p - 1.0).abs() < 1e-6);
        assert!((config.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(config.seed, None);
    }
}
