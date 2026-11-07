use crate::decoder::{DecoderKvCache, DecoderWorkspace};
use crate::segment::Segment;

pub struct WhisperState {
    pub cache: DecoderKvCache,
    /// Scratch buffers for the decoder forward pass. Kept here so a reused state
    /// never reallocates the intermediate activations between windows or tokens.
    pub workspace: DecoderWorkspace,
    pub encoder_output: Vec<f32>,
    pub logits: Vec<f32>,
    pub mel: Option<whisper_audio::MelSpectrogram>,
    pub segments: Vec<Segment>,
    pub vad_segments: Vec<(usize, usize)>,
    pub lang_id: usize,
    pub lang_prob: f32,
}

impl WhisperState {
    pub fn new(n_layers: usize, d_model: usize) -> Self {
        Self {
            cache: DecoderKvCache::new(n_layers, 1, d_model),
            workspace: DecoderWorkspace::new(),
            encoder_output: Vec::new(),
            logits: Vec::new(),
            mel: None,
            segments: Vec::new(),
            vad_segments: Vec::new(),
            lang_id: 0,
            lang_prob: 0.0,
        }
    }

    /// Reset state between transcriptions to reuse allocated buffers.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.workspace.clear();
        self.encoder_output.clear();
        self.logits.clear();
        self.mel = None;
        self.segments.clear();
        self.vad_segments.clear();
        self.lang_id = 0;
        self.lang_prob = 0.0;
    }

    /// Clear VAD state when VAD is disabled (whisper.cpp PR #4122)
    pub fn clear_vad(&mut self) {
        self.vad_segments.clear();
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }
}
