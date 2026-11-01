//! Model loader for whisper.rs supporting whisper.cpp ggml format (.bin).

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number: {0:#x}")]
    InvalidMagic(u32),
    #[error("Unsupported model version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid model format: {0}")]
    FormatError(String),
}
