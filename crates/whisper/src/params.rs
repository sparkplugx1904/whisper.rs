#[derive(Debug, Clone)]
pub struct WhisperParams {
    /// Language code (e.g. "en", "id") or "auto" for auto-detection
    pub language: String,
    /// Translate from source language to English
    pub translate: bool,
    /// Sampling temperature (0.0 for greedy / default beam search)
    pub temperature: f32,
    /// Temperature increment for fallback
    pub temperature_inc: f32,
    /// Beam size for beam search (1 = greedy, >1 = beam search)
    pub beam_size: usize,
    /// Disable timestamp tokens
    pub no_timestamps: bool,
    /// Suppress blank token at beginning
    pub suppress_blank: bool,
    /// Suppress non-speech tokens
    pub suppress_nst: bool,
    /// Gzip/LZ77 compression ratio threshold to trigger fallback
    pub compression_ratio_threshold: f64,
    /// Average logprob threshold to trigger fallback
    pub logprob_threshold: f64,
    /// No-speech probability threshold
    pub no_speech_threshold: f64,
    /// Initial prompt text
    pub initial_prompt: Option<String>,
    /// Optional GBNF grammar specification for constrained decoding
    pub grammar: Option<String>,
    /// Maximum number of tokens to generate per 30s chunk (0 = default 224)
    pub max_tokens: usize,
    /// Number of threads for computation
    pub n_threads: usize,
}

impl Default for WhisperParams {
    fn default() -> Self {
        Self {
            language: "auto".to_string(),
            translate: false,
            temperature: 0.0,
            temperature_inc: 0.2,
            beam_size: 1,
            no_timestamps: false,
            suppress_blank: true,
            suppress_nst: true,
            compression_ratio_threshold: 2.4,
            logprob_threshold: -1.0,
            no_speech_threshold: 0.6,
            initial_prompt: None,
            grammar: None,
            max_tokens: 224,
            n_threads: 4,
        }
    }
}
