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
