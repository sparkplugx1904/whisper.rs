#![forbid(unsafe_code)]
//! Tokenizer (GPT-2 BPE & multilingual special tokens) for whisper.rs.

use thiserror::Error;
use whisper_model::Vocab;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Token not found: {0}")]
    TokenNotFound(String),
    #[error("Invalid token id: {0}")]
    InvalidTokenId(i32),
    #[error("Decode error: {0}")]
    DecodeError(String),
}

#[derive(Clone)]
pub struct Tokenizer {
    pub vocab: Vocab,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    /// Decode a sequence of token IDs to a string.
    /// If `skip_special` is true, tokens >= token_eot are skipped.
    pub fn decode(&self, tokens: &[i32], skip_special: bool) -> String {
        let mut out = String::new();
        let eot = self.vocab.token_eot;

        for &tok in tokens {
            if skip_special && tok >= eot {
                continue;
            }
            if let Some(s) = self.vocab.id_to_token(tok) {
                out.push_str(s);
            }
        }
        out
    }

    /// Encode input text into a sequence of BPE token IDs using greedy longest-match.
    /// Handles CJK characters, multi-byte UTF-8, punctuation, and leading spaces.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut tokens = Vec::new();

        // Split text into word / whitespace chunks
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut chunk = String::new();
            if chars[i].is_whitespace() {
                // Collect whitespace and the word following it
                chunk.push(chars[i]);
                i += 1;
                while i < chars.len() && !chars[i].is_whitespace() {
                    chunk.push(chars[i]);
                    i += 1;
                }
            } else {
                while i < chars.len() && !chars[i].is_whitespace() {
                    chunk.push(chars[i]);
                    i += 1;
                }
            }
            chunks.push(chunk);
        }

        for chunk in chunks {
            let bytes = chunk.as_bytes();
            let mut pos = 0;

            while pos < bytes.len() {
                let mut best_len = 0;
                let mut best_id = None;

                // Search for longest matching token from current byte position
                for len in (1..=bytes.len() - pos).rev() {
                    if let Ok(sub_str) = std::str::from_utf8(&bytes[pos..pos + len]) {
                        if let Some(&id) = self.vocab.token_to_id.get(sub_str) {
                            best_len = len;
                            best_id = Some(id);
                            break;
                        }
                    }
                }

                if let (Some(id), len) = (best_id, best_len) {
                    tokens.push(id);
                    pos += len;
                } else {
                    // Byte-level fallback
                    let b = bytes[pos];
                    let byte_str = format!("{}", b as char);
                    if let Some(&id) = self.vocab.token_to_id.get(&byte_str) {
                        tokens.push(id);
                    } else {
                        // Raw byte token lookup
                        let byte_token = std::str::from_utf8(&[b])
                            .ok()
                            .and_then(|s| self.vocab.token_to_id.get(s).copied());
                        if let Some(id) = byte_token {
                            tokens.push(id);
                        }
                    }
                    pos += 1;
                }
            }
        }

        tokens
    }

    /// Find token id for an exact token string.
    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.vocab.token_to_id(token)
    }

    /// Get token string by id.
    pub fn id_to_token(&self, id: i32) -> Option<&str> {
        self.vocab.id_to_token(id)
    }

    pub fn eot(&self) -> i32 {
        self.vocab.token_eot
    }

    pub fn sot(&self) -> i32 {
        self.vocab.token_sot
    }

    pub fn translate(&self) -> i32 {
        self.vocab.token_translate
    }

    pub fn transcribe(&self) -> i32 {
        self.vocab.token_transcribe
    }

    pub fn not(&self) -> i32 {
        self.vocab.token_not
    }

    pub fn beg(&self) -> i32 {
        self.vocab.token_beg
    }
}
