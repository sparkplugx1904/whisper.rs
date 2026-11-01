#![forbid(unsafe_code)]
//! whisper-rs-pure: 100% Pure Rust implementation of OpenAI Whisper inference (whisper.cpp port).

pub mod decoder;
pub mod encoder;
pub mod params;
pub mod segment;
pub mod transcribe;

pub use decoder::{DecoderBlockWeights, DecoderKvCache, DecoderWeights, WhisperDecoder};
pub use encoder::{EncoderBlockWeights, EncoderWeights, WhisperEncoder};
pub use params::WhisperParams;
pub use segment::Segment;
pub use transcribe::WhisperContext;

pub use whisper_audio as audio;
pub use whisper_model as model;
pub use whisper_tensor as tensor;
pub use whisper_tokenizer as tokenizer;
