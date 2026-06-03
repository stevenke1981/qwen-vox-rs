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
use serde_json::Value;
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
    /// Added special tokens that must be matched before byte-level BPE.
    added_tokens: Vec<(String, u32)>,
    /// GPT-2 byte-to-character mapping for encode pre-tokenization.
    byte_to_char: HashMap<u8, char>,
    /// Inverse: character → byte for decode post-processing.
    char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Load tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> VoxResult<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::from_hf_dir(path);
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| VoxError::Tokenizer(format!("failed to read {}: {e}", path.display())))?;

        let mut config: TokenizerConfig = serde_json::from_str(&content)
            .map_err(|e| VoxError::Tokenizer(format!("failed to parse tokenizer.json: {e}")))?;
        let added_tokens = Self::load_added_tokens_from_json(&content)?;
        for (token, id) in &added_tokens {
            config.vocab.entry(token.clone()).or_insert(*id);
        }
        Self::from_config(config, added_tokens)
    }

    fn from_hf_dir(path: &Path) -> VoxResult<Self> {
        let vocab_path = path.join("vocab.json");
        let merges_path = path.join("merges.txt");
        let tokenizer_config_path = path.join("tokenizer_config.json");

        let vocab_content = std::fs::read_to_string(&vocab_path).map_err(|e| {
            VoxError::Tokenizer(format!("failed to read {}: {e}", vocab_path.display()))
        })?;
        let mut vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content)
            .map_err(|e| VoxError::Tokenizer(format!("failed to parse vocab.json: {e}")))?;

        let merges_content = std::fs::read_to_string(&merges_path).map_err(|e| {
            VoxError::Tokenizer(format!("failed to read {}: {e}", merges_path.display()))
        })?;
        let merges = merges_content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_string)
            .collect();

        let added_tokens = if tokenizer_config_path.exists() {
            let tokenizer_config =
                std::fs::read_to_string(&tokenizer_config_path).map_err(|e| {
                    VoxError::Tokenizer(format!(
                        "failed to read {}: {e}",
                        tokenizer_config_path.display()
                    ))
                })?;
            Self::load_added_tokens_from_json(&tokenizer_config)?
        } else {
            Vec::new()
        };
        for (token, id) in &added_tokens {
            vocab.entry(token.clone()).or_insert(*id);
        }

        Self::from_config(
            TokenizerConfig {
                model_type: "BPE".to_string(),
                vocab,
                merges,
                special_tokens: None,
            },
            added_tokens,
        )
    }

    fn load_added_tokens_from_json(content: &str) -> VoxResult<Vec<(String, u32)>> {
        let value: Value = serde_json::from_str(content)
            .map_err(|e| VoxError::Tokenizer(format!("failed to parse tokenizer metadata: {e}")))?;
        let Some(decoder) = value
            .get("added_tokens_decoder")
            .and_then(|v| v.as_object())
        else {
            return Ok(Vec::new());
        };

        let mut tokens = Vec::with_capacity(decoder.len());
        for (id, token) in decoder {
            let id = id
                .parse::<u32>()
                .map_err(|e| VoxError::Tokenizer(format!("invalid added token id {id}: {e}")))?;
            let content = token
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| VoxError::Tokenizer(format!("missing content for token {id}")))?;
            tokens.push((content.to_string(), id));
        }
        tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        Ok(tokens)
    }

    fn from_config(config: TokenizerConfig, added_tokens: Vec<(String, u32)>) -> VoxResult<Self> {
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
            added_tokens,
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
        let mut ids = Vec::new();
        let mut pos = 0usize;

        while pos < text.len() {
            if let Some((token, id)) = self.match_added_token_at(text, pos) {
                ids.push(id);
                pos += token.len();
                continue;
            }

            let next_special = self.find_next_added_token(text, pos).unwrap_or(text.len());
            ids.extend(self.encode_regular(&text[pos..next_special])?);
            pos = next_special;
        }

        if ids.is_empty() && !text.is_empty() {
            return Err(VoxError::Tokenizer(format!(
                "no tokens found for input: '{text}'"
            )));
        }

        Ok(ids)
    }

    fn match_added_token_at<'a>(&'a self, text: &str, pos: usize) -> Option<(&'a str, u32)> {
        self.added_tokens
            .iter()
            .find(|(token, _)| text[pos..].starts_with(token))
            .map(|(token, id)| (token.as_str(), *id))
    }

    fn find_next_added_token(&self, text: &str, pos: usize) -> Option<usize> {
        self.added_tokens
            .iter()
            .filter_map(|(token, _)| text[pos..].find(token).map(|offset| pos + offset))
            .min()
    }

    fn encode_regular(&self, text: &str) -> VoxResult<Vec<u32>> {
        let mut ids = Vec::new();
        for piece in pretokenize(text) {
            ids.extend(self.encode_bpe_piece(&piece)?);
        }
        Ok(ids)
    }

    fn encode_bpe_piece(&self, text: &str) -> VoxResult<Vec<u32>> {
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
        let mut ids = Vec::with_capacity(tokens.len());
        for token in &tokens {
            let id = self.config.vocab.get(token).copied().ok_or_else(|| {
                VoxError::Tokenizer(format!("token '{token}' not found for input: '{text}'"))
            })?;
            ids.push(id);
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum PretokenKind {
    Alpha,
    Digit,
    Cjk,
    Other,
}

fn pretoken_kind(ch: char) -> PretokenKind {
    if ch.is_ascii_alphabetic() {
        PretokenKind::Alpha
    } else if ch.is_ascii_digit() {
        PretokenKind::Digit
    } else if is_cjk(ch) {
        PretokenKind::Cjk
    } else {
        PretokenKind::Other
    }
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

fn pretokenize(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut pending_space = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == ' ' || ch == '\t' {
            pending_space.push(ch);
            continue;
        }

        if ch.is_whitespace() {
            if !pending_space.is_empty() {
                pieces.push(std::mem::take(&mut pending_space));
            }
            pieces.push(ch.to_string());
            continue;
        }

        let kind = pretoken_kind(ch);
        let mut piece = std::mem::take(&mut pending_space);
        piece.push(ch);

        if kind != PretokenKind::Other {
            while let Some(&next) = chars.peek() {
                if next.is_whitespace() || pretoken_kind(next) != kind {
                    break;
                }
                piece.push(next);
                chars.next();
            }
        }

        pieces.push(piece);
    }

    if !pending_space.is_empty() {
        pieces.push(pending_space);
    }

    pieces
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
            added_tokens: Vec::new(),
            byte_to_char,
            char_to_byte,
        };

        assert_eq!(tokenizer.special_token("codec_bos"), Some(100));
        assert_eq!(tokenizer.special_token("codec_eos"), Some(101));
        assert_eq!(tokenizer.special_token("codec_think"), Some(102));
        assert_eq!(tokenizer.special_token("codec_nothink"), Some(103));
        assert_eq!(tokenizer.special_token("unknown"), None);
    }

    #[test]
    fn test_qwen3_official_prompt_token_ids() {
        let weights_dir = Path::new("weights/hf_original");
        if !weights_dir.exists() {
            return;
        }

        let tokenizer = Tokenizer::from_file(weights_dir).unwrap();
        let prompt = "<|im_start|>assistant\n你好，這是官方 Qwen3 TTS 參考語音。<|im_end|>\n<|im_start|>assistant\n";
        let ids = tokenizer.encode(prompt).unwrap();
        assert_eq!(
            ids,
            vec![
                151644, 77091, 198, 108386, 3837, 107304, 100777, 1207, 16948, 18, 350, 9951,
                26853, 225, 77598, 102819, 78685, 1773, 151645, 198, 151644, 77091, 198,
            ]
        );
    }
}
