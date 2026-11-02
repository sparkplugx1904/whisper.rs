#![forbid(unsafe_code)]
//! Audio loading, resampling, and log-mel spectrogram extraction for whisper.rs.

pub mod decode;
pub mod fft;
pub mod mel;
pub mod resample;
pub mod wav;

pub use decode::{load_audio_from_bytes, load_audio_from_file};
pub use fft::Fft400;
pub use mel::{compute_log_mel_spectrogram, hann_window, MelSpectrogram, HOP_LENGTH, N_MELS};
pub use resample::resample_to_16k;
use thiserror::Error;
pub use wav::{WavAudio, WHISPER_SAMPLE_RATE};

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid WAV header: {0}")]
    InvalidWav(String),
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),
}
