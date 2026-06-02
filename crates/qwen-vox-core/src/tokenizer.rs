//! Tokenizer — loads Qwen3-TTS text tokenizer from `tokenizer.json`.
//!
//! The Qwen3-TTS text tokenizer is a BPE tokenizer with:
//! - ~151,936 text tokens (standard Qwen vocabulary)
//! - GPT-2-style byte-level encoding (all text → UTF-8 bytes → Unicode chars)
//! - Codec special tokens (codec_bos_id, codec_eos_id, codec_think_id, etc.)
//! - Merges-based BPE encoding
//!
//! All tokenization logic is driven by the loaded configuration file
//! plus the standard GPT-2 byte-to-unicode mapping.

use crate::error::{VoxError, VoxResult};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Special token IDs used by Qwen3-TTS.
#[derive(Debug, Clone, Deserialize)]
pub struct SpecialTokens {
    /// Beginning of codec sequence.
    #[serde(default)]
    pub codec_bos_id: Option<u32>,
    /// End of codec sequence.
    #[serde(default)]
    pub codec_eos_id: Option<u32>,
    /// "Think" token for 12Hz streaming.
    #[serde(default)]
    pub codec_think_id: Option<u32>,
    /// "No-think" token for 12Hz streaming.
    #[serde(default)]
    pub codec_nothink_id: Option<u32>,
    /// Padding token.
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

/// BPE merge rule: (pair, merged_token).
#[derive(Debug, Clone)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
    pub rank: usize,
}

/// Loaded tokenizer configuration.
#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    /// Model type identifier (e.g., "BPE", "WordPiece").
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// Vocabulary mapping: token string → ID.
    #[serde(default)]
    pub vocab: HashMap<String, u32>,

    /// Merge rules (for BPE), space-separated pairs.
    #[serde(default)]
    pub merges: Vec<String>,

    /// Special tokens configuration.
    #[serde(default)]
    pub special_tokens: Option<SpecialTokens>,
}

fn default_model_type() -> String {
    "BPE".to_string()
}

// ── GPT-2 byte-to-unicode mapping ──────────────────────────────────────────

/// Build the standard GPT-2 byte-to-character mapping.
///
/// Each UTF-8 byte (0–255) is mapped to a unique Unicode character so that BPE
/// can operate on a character-level sequence.  The mapping is bijective and
/// matches the `tokenizers` crate (HuggingFace).
fn byte_to_char() -> HashMap<u8, char> {
    // "Nice" bytes that stay at their own code point
    let mut bs: Vec<u8> = Vec::new();
    bs.extend(33..=126); // '!' .. '~'  (printable ASCII)
    bs.extend(161..=172); // '¡' .. '¬'  (Latin-1 Supplement)
    bs.extend(174..=255); // '®' .. 'ÿ'  (Latin-1 Supplement, minus 173)

    let mut byte_to_char: HashMap<u8, char> = HashMap::new();
    for &b in &bs {
        byte_to_char.insert(b, char::from_u32(b as u32).unwrap());
    }

    // Remaining bytes (0–32, 127–160, 173) are mapped to chr(256), chr(257), …
    let mut n: u32 = 0;
    for b in 0..=255u8 {
        if let std::collections::hash_map::Entry::Vacant(entry) = byte_to_char.entry(b) {
            entry.insert(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    byte_to_char
}

/// Inverse mapping: byte-level Unicode character → original byte.
fn char_to_byte() -> HashMap<char, u8> {
    byte_to_char().into_iter().map(|(b, c)| (c, b)).collect()
}

// ── Tokenizer ──────────────────────────────────────────────────────────────

/// Qwen3-TTS text tokenizer.
///
/// Wraps a loaded `tokenizer.json` and provides encode/decode
/// functionality for the text input pipeline.
pub struct Tokenizer {
    config: TokenizerConfig,
    /// Reverse vocabulary: ID → token string.
    id_to_token: HashMap<u32, String>,
    /// Parsed merge rules with rank ordering.
    merge_rules: Vec<MergeRule>,
    /// Merge rank lookup: (left, right) → rank.
    merge_rank: HashMap<(String, String), usize>,
    /// GPT-2 byte-to-character mapping for encode pre-tokenization.
    byte_to_char: HashMap<u8, char>,
    /// Inverse: character → byte for decode post-processing.
    char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Load tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> VoxResult<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| VoxError::Tokenizer(format!("failed to read {}: {e}", path.display())))?;

        let config: TokenizerConfig = serde_json::from_str(&content)
            .map_err(|e| VoxError::Tokenizer(format!("failed to parse tokenizer.json: {e}")))?;

        // Build reverse vocabulary
        let id_to_token: HashMap<u32, String> =
            config.vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        // Parse merge rules
        let merge_rules: Vec<MergeRule> = config
            .merges
            .iter()
            .enumerate()
            .filter_map(|(rank, rule)| {
                let parts: Vec<&str> = rule.split_whitespace().collect();
                if parts.len() == 2 {
                    let merged = format!("{}{}", parts[0], parts[1]);
                    Some(MergeRule {
                        left: parts[0].to_string(),
                        right: parts[1].to_string(),
                        merged,
                        rank,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Build merge rank lookup
        let merge_rank: HashMap<(String, String), usize> = merge_rules
            .iter()
            .map(|r| ((r.left.clone(), r.right.clone()), r.rank))
            .collect();

        let byte_to_char = byte_to_char();
        let char_to_byte = char_to_byte();

        Ok(Self {
            config,
            id_to_token,
            merge_rules,
            merge_rank,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Encode text into token IDs using GPT-2 byte-level BPE.
    ///
    /// 1. Convert input to UTF-8 bytes.
    /// 2. Map each byte to a byte-level Unicode character (GPT-2 mapping).
    /// 3. Look up each character in the vocabulary to form initial tokens.
    /// 4. Apply BPE merges until no more applicable.
    pub fn encode(&self, text: &str) -> VoxResult<Vec<u32>> {
        // Step 1 & 2: convert text to UTF-8 bytes, then to byte-level characters
        let initial_chars: String = text
            .as_bytes()
            .iter()
            .map(|b| self.byte_to_char.get(b).copied().unwrap_or(*b as char))
            .collect();

        // Step 3: split into individual tokens (each character is an initial token)
        let mut tokens: Vec<String> = initial_chars.chars().map(|c| c.to_string()).collect();

        // Step 4: apply BPE merges iteratively
        loop {
            let mut best_rank = usize::MAX;
            let mut best_pos = None;

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = Some(i);
                    }
                }
            }

            match best_pos {
                Some(pos) => {
                    let merged = format!("{}{}", tokens[pos], tokens[pos + 1]);
                    tokens[pos] = merged;
                    tokens.remove(pos + 1);
                }
                None => break,
            }
        }

        // Convert tokens to IDs
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|t| self.config.vocab.get(t).copied())
            .collect();

        if ids.is_empty() && !text.is_empty() {
            return Err(VoxError::Tokenizer(format!(
                "no tokens found for input: '{text}'"
            )));
        }

        Ok(ids)
    }

    /// Decode token IDs back to text using the inverse byte-level mapping.
    ///
    /// Each byte-level character in every token string is mapped back to a byte.
    /// The bytes are then interpreted as UTF-8.
    pub fn decode(&self, ids: &[u32]) -> VoxResult<String> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                for c in token.chars() {
                    if let Some(&b) = self.char_to_byte.get(&c) {
                        bytes.push(b);
                    }
                    // If a character is not in our byte mapping (e.g. special
                    // tokens like "codec_bos") we skip it during text decode.
                }
            }
        }
        let text = String::from_utf8(bytes)
            .map_err(|e| VoxError::Tokenizer(format!("invalid UTF-8 from decoded bytes: {e}")))?;
        Ok(text)
    }

    /// Return vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab.len()
    }

    /// Return the model type string.
    pub fn model_type(&self) -> &str {
        &self.config.model_type
    }

    /// Get a special token ID by name.
    pub fn special_token(&self, name: &str) -> Option<u32> {
        self.config
            .special_tokens
            .as_ref()
            .and_then(|st| match name {
                "codec_bos" => st.codec_bos_id,
                "codec_eos" => st.codec_eos_id,
                "codec_think" => st.codec_think_id,
                "codec_nothink" => st.codec_nothink_id,
                "pad" => st.pad_token_id,
                _ => None,
            })
    }

    /// Return the number of merge rules.
    pub fn num_merges(&self) -> usize {
        self.merge_rules.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_deserialize() {
        let json = r#"{
            "model_type": "BPE",
            "vocab": {"hello": 0, "world": 1, "Ġ": 2},
            "merges": ["h e", "he l", "hel lo"],
            "special_tokens": {
                "codec_bos_id": 100,
                "codec_eos_id": 101
            }
        }"#;

        let config: TokenizerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "BPE");
        assert_eq!(config.vocab.len(), 3);
        assert_eq!(config.merges.len(), 3);
        assert_eq!(
            config.special_tokens.as_ref().unwrap().codec_bos_id,
            Some(100)
        );
    }

    #[test]
    fn test_special_tokens() {
        let json = r#"{
            "model_type": "BPE",
            "vocab": {},
            "merges": [],
            "special_tokens": {
                "codec_bos_id": 100,
                "codec_eos_id": 101,
                "codec_think_id": 102,
                "codec_nothink_id": 103
            }
        }"#;

        let config: TokenizerConfig = serde_json::from_str(json).unwrap();
        let (byte_to_char, char_to_byte) = (byte_to_char(), char_to_byte());
        let tokenizer = Tokenizer {
            config,
            id_to_token: HashMap::new(),
            merge_rules: Vec::new(),
            merge_rank: HashMap::new(),
            byte_to_char,
            char_to_byte,
        };

        assert_eq!(tokenizer.special_token("codec_bos"), Some(100));
        assert_eq!(tokenizer.special_token("codec_eos"), Some(101));
        assert_eq!(tokenizer.special_token("codec_think"), Some(102));
        assert_eq!(tokenizer.special_token("codec_nothink"), Some(103));
        assert_eq!(tokenizer.special_token("unknown"), None);
    }
}
