//! Audio loading, resampling, and log-mel spectrogram extraction for whisper.rs.

pub mod fft;
pub mod mel;
pub mod resample;
pub mod wav;

pub use fft::Fft400;
pub use mel::{compute_log_mel_spectrogram, hann_window, MelSpectrogram, HOP_LENGTH, N_MELS};
pub use resample::resample_to_16k;
pub use wav::{WavAudio, WHISPER_SAMPLE_RATE};

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
