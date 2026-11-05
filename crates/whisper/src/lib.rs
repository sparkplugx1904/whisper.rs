#![forbid(unsafe_code)]
//! whisper.rs: Rust implementation of OpenAI Whisper inference (whisper.cpp port).

pub mod beam;
pub mod decoder;
pub mod dtw;
pub mod encoder;
pub mod filter;
pub mod params;
pub mod segment;
pub mod state;
pub mod transcribe;
pub use beam::{BeamCandidate, BeamSearch};
pub use decoder::{DecoderBlockWeights, DecoderKvCache, DecoderWeights, WhisperDecoder};
pub use dtw::{
    align_dtw, get_alignment_heads, merge_tokens_to_words, AlignmentHeadsPreset, WordTiming,
};
pub use encoder::{EncoderBlockWeights, EncoderWeights, WhisperEncoder};
pub use filter::{
    apply_repetition_penalty, apply_suppress_blank, apply_suppress_non_speech, compression_ratio,
    compute_distribution_entropy, compute_token_logprob, GrammarLogitFilter, LogitFilter,
    LogitFilterContext, LogitFilterPipeline, RepetitionPenaltyFilter, SuppressBlankFilter,
    SuppressNonSpeechFilter, TimestampLogitFilter,
};
pub use params::{FullParamsBuilder, SamplingStrategy, WhisperParams};
pub use segment::{
    format_json, format_karaoke_srt, format_karaoke_vtt, format_srt, format_vtt, Segment,
};
pub use state::WhisperState;
pub use transcribe::WhisperContext;
pub use transcribe::{FallbackEvaluation, WindowPromptBuilder};

pub use whisper_audio as audio;
pub use whisper_model as model;
pub use whisper_tensor as tensor;
pub use whisper_tokenizer as tokenizer;
