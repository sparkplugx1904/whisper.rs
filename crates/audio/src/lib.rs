#![forbid(unsafe_code)]
//! Audio loading, resampling, and log-mel spectrogram extraction for whisper.rs.

pub mod decode;
pub mod fft;
pub mod mel;
pub mod resample;
pub mod wav;

pub use decode::{load_audio_from_bytes, load_audio_from_file};
pub use fft::Fft400;
pub use mel::{
    compute_log_mel_spectrogram, generate_slaney_mel_filters, hann_window, MelSpectrogram,
    HOP_LENGTH, N_MELS,
};
pub use resample::resample_to_16k;
use thiserror::Error;
pub use wav::{
    WavAudio, KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, KSDATAFORMAT_SUBTYPE_PCM, WAVE_FORMAT_EXTENSIBLE,
    WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM, WHISPER_SAMPLE_RATE,
};

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid WAV header: {0}")]
    InvalidWav(String),
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),
}
