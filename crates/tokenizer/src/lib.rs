//! Tokenizer (GPT-2 BPE & multilingual special tokens) for whisper.rs.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Token not found: {0}")]
    TokenNotFound(String),
    #[error("Invalid token id: {0}")]
    InvalidTokenId(i32),
    #[error("Decode error: {0}")]
    DecodeError(String),
}
