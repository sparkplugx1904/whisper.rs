//! Audio loading, resampling, and log-mel spectrogram extraction for whisper.rs.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid WAV header: {0}")]
    InvalidWav(String),
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),
}
