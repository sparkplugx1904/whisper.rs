//! Accumulated wall-clock telemetry for the transcription pipeline.
//!
//! The counters mirror `whisper_state` in whisper.cpp: each phase stores the
//! total time it consumed together with the number of times it ran, so callers
//! can report an average per run (`total / n`). Values are cumulative across
//! `WhisperContext::full` calls and are only cleared by
//! [`WhisperState::reset_timings`](crate::WhisperState::reset_timings), which is
//! what `whisper_full` does at the end of its run.

/// Phase timings in microseconds plus the matching run counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timings {
    /// Log-mel spectrogram computation.
    pub mel_us: u64,
    /// Token selection and logit post-processing inside the decoding loops.
    pub sample_us: u64,
    /// Encoder forward passes, including language-detection prefill.
    pub encode_us: u64,
    /// Single-sequence decoder forward passes (prompt and token steps).
    pub decode_us: u64,
    /// Batched decoder steps used by beam search.
    pub batchd_us: u64,
    /// The prompt prefill subset of `decode_us`.
    ///
    /// A prompt decode also advances the KV cache for that token, so its wall
    /// time is part of `decode_us`; this field isolates it for reporting the
    /// way whisper.cpp separates prompt evaluation from token generation.
    pub prompt_us: u64,
    /// Number of token selection passes.
    pub n_sample: u32,
    /// Number of encoder forward passes.
    pub n_encode: u32,
    /// Number of decoder forward passes.
    pub n_decode: u32,
    /// Number of batched beam decoder steps.
    pub n_batchd: u32,
    /// Number of prompt prefill decodes.
    pub n_prompt: u32,
    /// Windows rejected by the average logprob threshold.
    pub n_fail_p: u32,
    /// Windows rejected by the entropy threshold.
    pub n_fail_h: u32,
}

impl Timings {
    /// Accumulate one encoder forward pass of `elapsed_us` microseconds.
    pub fn add_encode(&mut self, elapsed_us: u64) {
        self.encode_us += elapsed_us;
        self.n_encode += 1;
    }

    /// Accumulate one decoder forward pass of `elapsed_us` microseconds.
    pub fn add_decode(&mut self, elapsed_us: u64) {
        self.decode_us += elapsed_us;
        self.n_decode += 1;
    }

    /// Accumulate one prompt prefill decode of `elapsed_us` microseconds.
    pub fn add_prompt(&mut self, elapsed_us: u64) {
        self.prompt_us += elapsed_us;
        self.n_prompt += 1;
    }

    /// Accumulate one batched beam step of `elapsed_us` microseconds.
    pub fn add_batch_decode(&mut self, elapsed_us: u64) {
        self.batchd_us += elapsed_us;
        self.n_batchd += 1;
    }

    /// Accumulate one token selection pass of `elapsed_us` microseconds.
    pub fn add_sample(&mut self, elapsed_us: u64) {
        self.sample_us += elapsed_us;
        self.n_sample += 1;
    }

    /// Zero every counter.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulators_increment_counts() {
        let mut t = Timings::default();
        t.add_encode(100);
        t.add_encode(50);
        t.add_decode(30);
        t.add_prompt(20);
        t.add_batch_decode(10);
        t.add_sample(5);

        assert_eq!(t.encode_us, 150);
        assert_eq!(t.n_encode, 2);
        assert_eq!(t.decode_us, 30);
        assert_eq!(t.n_decode, 1);
        assert_eq!(t.prompt_us, 20);
        assert_eq!(t.n_prompt, 1);
        assert_eq!(t.batchd_us, 10);
        assert_eq!(t.n_batchd, 1);
        assert_eq!(t.sample_us, 5);
        assert_eq!(t.n_sample, 1);
    }

    #[test]
    fn test_reset_zeroes_every_field() {
        let mut t = Timings {
            mel_us: 1,
            sample_us: 2,
            encode_us: 3,
            decode_us: 4,
            batchd_us: 5,
            prompt_us: 6,
            n_sample: 7,
            n_encode: 8,
            n_decode: 9,
            n_batchd: 10,
            n_prompt: 11,
            n_fail_p: 12,
            n_fail_h: 13,
        };
        t.reset();
        assert_eq!(t, Timings::default());
    }
}
