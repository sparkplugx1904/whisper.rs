//! Audio loading, resampling, and log-mel spectrogram extraction for whisper.rs.

pub mod decode;
pub mod fft;
pub mod mel;
pub mod resample;
pub mod wav;

pub use decode::{load_audio_from_bytes, load_audio_from_file};
pub use fft::Fft400;
pub use mel::{
    compute_log_mel_spectrogram, dot_f32, dot_f32_scalar, generate_slaney_mel_filters, hann_window,
    hann_window_400, project_mel_frame, MelSpectrogram, HANN_WINDOW_400, HOP_LENGTH, N_MELS,
};
pub use resample::{
    high_pass_filter, high_pass_filtered, resample_to_16k, PolyphaseSincFilter,
    DEFAULT_HIGH_PASS_CUTOFF_HZ,
};
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
