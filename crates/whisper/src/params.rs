use std::sync::Arc;

pub type NewSegmentCallback = Arc<dyn Fn(&crate::segment::Segment) + Send + Sync>;
pub type ProgressCallback = Arc<dyn Fn(i32) + Send + Sync>;

#[derive(Clone)]
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
    /// Enable Flash Attention 2 (Tiled Online Softmax)
    pub flash_attn: bool,
    /// Enable speaker diarization turn token detection (tinydiarize)
    pub tdrz_enable: bool,
    /// Native Rust closure callback invoked whenever a new segment is decoded
    pub on_new_segment: Option<NewSegmentCallback>,
    /// Native Rust closure callback invoked on transcription progress (0-100 percentage)
    pub on_progress: Option<ProgressCallback>,
}

impl std::fmt::Debug for WhisperParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperParams")
            .field("language", &self.language)
            .field("translate", &self.translate)
            .field("temperature", &self.temperature)
            .field("beam_size", &self.beam_size)
            .field("no_timestamps", &self.no_timestamps)
            .field("flash_attn", &self.flash_attn)
            .field("tdrz_enable", &self.tdrz_enable)
            .finish()
    }
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
            flash_attn: false,
            tdrz_enable: false,
            on_new_segment: None,
            on_progress: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    Greedy { best_of: usize },
    BeamSearch { beam_size: usize },
}

/// Type-safe fluent builder for transcription parameters.
#[derive(Clone)]
pub struct FullParamsBuilder {
    params: WhisperParams,
}

impl FullParamsBuilder {
    pub fn new(strategy: SamplingStrategy) -> Self {
        let mut params = WhisperParams::default();
        match strategy {
            SamplingStrategy::Greedy { .. } => {
                params.beam_size = 1;
            }
            SamplingStrategy::BeamSearch { beam_size } => {
                params.beam_size = beam_size.max(1);
            }
        }
        Self { params }
    }

    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.params.language = lang.into();
        self
    }

    pub fn translate(mut self, translate: bool) -> Self {
        self.params.translate = translate;
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.params.temperature = temp;
        self
    }

    pub fn no_timestamps(mut self, no_ts: bool) -> Self {
        self.params.no_timestamps = no_ts;
        self
    }

    pub fn initial_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.params.initial_prompt = Some(prompt.into());
        self
    }

    pub fn grammar(mut self, grammar: impl Into<String>) -> Self {
        self.params.grammar = Some(grammar.into());
        self
    }

    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.params.n_threads = n_threads;
        self
    }

    pub fn flash_attn(mut self, flash: bool) -> Self {
        self.params.flash_attn = flash;
        self
    }

    pub fn tdrz_enable(mut self, tdrz: bool) -> Self {
        self.params.tdrz_enable = tdrz;
        self
    }

    pub fn on_new_segment<F>(mut self, cb: F) -> Self
    where
        F: Fn(&crate::segment::Segment) + Send + Sync + 'static,
    {
        self.params.on_new_segment = Some(Arc::new(cb));
        self
    }

    pub fn on_progress<F>(mut self, cb: F) -> Self
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        self.params.on_progress = Some(Arc::new(cb));
        self
    }

    pub fn build(self) -> WhisperParams {
        self.params
    }
}
