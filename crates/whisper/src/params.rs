use std::sync::Arc;

pub type NewSegmentCallback = Arc<dyn Fn(&crate::segment::Segment) + Send + Sync>;
pub type ProgressCallback = Arc<dyn Fn(i32) + Send + Sync>;
pub type AbortCallback = Arc<dyn Fn() -> bool + Send + Sync>;

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
    /// Pre-calibrated alignment heads preset for DTW word-level timestamps
    pub dtw_aheads_preset: crate::dtw::AlignmentHeadsPreset,
    /// Repetition penalty (> 1.0 penalizes already generated tokens, default 1.0)
    pub repetition_penalty: f32,
    /// Maximum character length per subtitle segment (0 = disabled)
    pub max_len: usize,
    /// Split long segments on word boundaries when exceeding max_len
    pub split_on_word: bool,
    /// Length penalty exponent for beam search (default: 0.7)
    pub length_penalty: f32,
    /// Pre-computed raw token ID array for context priming
    pub prompt_tokens: Vec<i32>,
    /// Disable context carryover between 30-second audio windows (default: false)
    pub no_context: bool,
    /// Carry over initial text prompt across subsequent 30s windows (default: false)
    pub carry_initial_prompt: bool,
    /// Entropy threshold for dynamic temperature fallback (default: 2.40)
    pub entropy_thold: f32,
    /// Callback to abort processing early (return true to abort)
    pub abort_callback: Option<AbortCallback>,
    /// Start time offset in audio processing (in ms, default: 0)
    pub offset_ms: i32,
    /// Duration of audio to process (in ms, default: 0 = full audio)
    pub duration_ms: i32,
    /// Timestamp token limit at step 0 (default: 1.0s)
    pub max_initial_ts: f32,
    /// Force single segment output, useful for streaming (default: false)
    pub single_segment: bool,
    /// Regular expression matching tokens to suppress (default: None)
    pub suppress_regex: Option<String>,
    /// Override audio context length (default: 0 = 1500)
    pub audio_ctx: usize,
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
            .field("offset_ms", &self.offset_ms)
            .field("duration_ms", &self.duration_ms)
            .field("max_initial_ts", &self.max_initial_ts)
            .field("single_segment", &self.single_segment)
            .field("suppress_regex", &self.suppress_regex)
            .field("audio_ctx", &self.audio_ctx)
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
            dtw_aheads_preset: crate::dtw::AlignmentHeadsPreset::TinyEn,
            max_len: 0,
            split_on_word: false,
            repetition_penalty: 1.0,
            length_penalty: 0.7,
            prompt_tokens: Vec::new(),
            no_context: false,
            carry_initial_prompt: false,
            entropy_thold: 2.40,
            on_new_segment: None,
            on_progress: None,
            abort_callback: None,
            offset_ms: 0,
            duration_ms: 0,
            max_initial_ts: 1.0,
            single_segment: false,
            suppress_regex: None,
            audio_ctx: 0,
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

    pub fn dtw_aheads_preset(mut self, preset: crate::dtw::AlignmentHeadsPreset) -> Self {
        self.params.dtw_aheads_preset = preset;
        self
    }

    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.params.repetition_penalty = penalty;
        self
    }

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.params.max_len = max_len;
        self
    }

    pub fn split_on_word(mut self, split: bool) -> Self {
        self.params.split_on_word = split;
        self
    }

    pub fn length_penalty(mut self, penalty: f32) -> Self {
        self.params.length_penalty = penalty;
        self
    }

    pub fn prompt_tokens(mut self, tokens: Vec<i32>) -> Self {
        self.params.prompt_tokens = tokens;
        self
    }

    pub fn no_context(mut self, no_context: bool) -> Self {
        self.params.no_context = no_context;
        self
    }

    pub fn carry_initial_prompt(mut self, carry: bool) -> Self {
        self.params.carry_initial_prompt = carry;
        self
    }

    pub fn entropy_thold(mut self, thold: f32) -> Self {
        self.params.entropy_thold = thold;
        self
    }
    pub fn abort_callback<F>(mut self, cb: F) -> Self
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        self.params.abort_callback = Some(Arc::new(cb));
        self
    }
    pub fn offset_ms(mut self, offset_ms: i32) -> Self {
        self.params.offset_ms = offset_ms;
        self
    }
    pub fn duration_ms(mut self, duration_ms: i32) -> Self {
        self.params.duration_ms = duration_ms;
        self
    }
    pub fn max_initial_ts(mut self, max_initial_ts: f32) -> Self {
        self.params.max_initial_ts = max_initial_ts;
        self
    }
    pub fn single_segment(mut self, single_segment: bool) -> Self {
        self.params.single_segment = single_segment;
        self
    }
    pub fn suppress_regex(mut self, regex: impl Into<String>) -> Self {
        self.params.suppress_regex = Some(regex.into());
        self
    }
    pub fn audio_ctx(mut self, audio_ctx: usize) -> Self {
        self.params.audio_ctx = audio_ctx;
        self
    }
    pub fn build(self) -> WhisperParams {
        self.params
    }
}
