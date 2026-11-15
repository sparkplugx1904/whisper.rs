#![forbid(unsafe_code)]
//! Voice Activity Detection (VAD) for whisper.rs.
//!
//! Two detectors are available:
//! - [`silero::SileroVad`], the GGML Silero model used by whisper.cpp, with the
//!   same STFT/conv/LSTM graph and speech-segment post-processing.
//! - [`detect_speech_segments`], a dependency-free adaptive energy detector for
//!   callers that do not ship a model file.

pub mod silero;

pub use silero::{segments_from_probs, SileroVad, SileroVadError};

#[derive(Debug, Clone)]
pub struct VadParams {
    pub threshold: f32,
    pub min_speech_duration_ms: usize,
    pub min_silence_duration_ms: usize,
    pub max_speech_duration_s: f32,
    pub speech_pad_ms: usize,
    pub samples_overlap: f32,
    pub sample_rate: usize,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            max_speech_duration_s: f32::MAX,
            speech_pad_ms: 30,
            samples_overlap: 0.1,
            sample_rate: 16000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    pub t0: f64,
    pub t1: f64,
}

impl SpeechSegment {
    pub fn new(t0: f64, t1: f64) -> Self {
        Self { t0, t1 }
    }
}

/// Detect speech segments in 16 kHz mono float PCM samples using adaptive energy thresholding.
/// If no speech is present (pure silence), returns an empty vector (`Vec::new()`).
pub fn detect_speech_segments(samples: &[f32], params: &VadParams) -> Vec<SpeechSegment> {
    if samples.is_empty() {
        return Vec::new();
    }

    let frame_len = (params.sample_rate * 30) / 1000; // 30ms window
    let frame_hop = (params.sample_rate * 10) / 1000; // 10ms hop
    let total_sec = samples.len() as f64 / params.sample_rate as f64;

    if samples.len() < frame_len {
        return Vec::new();
    }

    let n_frames = (samples.len() - frame_len) / frame_hop + 1;
    let mut energies = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let start = i * frame_hop;
        let frame = &samples[start..start + frame_len];
        let mut sum_sq = 0.0f32;
        for &s in frame {
            sum_sq += s * s;
        }
        let rms = (sum_sq / frame_len as f32).sqrt();
        energies.push(rms);
    }

    // Adaptive noise floor: median energy of all frames
    let mut sorted_energies = energies.clone();
    sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_energy = sorted_energies[sorted_energies.len() / 2];
    let dyn_threshold = (params.threshold * 0.02).max(median_energy * 2.0);

    let mut is_speech = vec![false; n_frames];
    for (i, &e) in energies.iter().enumerate() {
        if e > dyn_threshold {
            is_speech[i] = true;
        }
    }

    // 1. Merge short silence gaps (< min_silence_duration_ms)
    let frame_hop_ms = ((frame_hop * 1000) / params.sample_rate).max(1);
    let min_silence_frames = params.min_silence_duration_ms.div_ceil(frame_hop_ms);
    let mut silence_len = 0;
    for i in 0..n_frames {
        if !is_speech[i] {
            silence_len += 1;
        } else {
            if silence_len < min_silence_frames && silence_len > 0 && i >= silence_len {
                is_speech[(i - silence_len)..i].fill(true);
            }
            silence_len = 0;
        }
    }

    // 2. Extract speech segments with min duration filter
    let min_speech_frames = params.min_speech_duration_ms.div_ceil(frame_hop_ms);
    let pad_sec = params.speech_pad_ms as f64 / 1000.0;
    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut seg_start = 0;

    for (k, &speech) in is_speech.iter().enumerate() {
        if speech && !in_speech {
            in_speech = true;
            seg_start = k;
        } else if !speech && in_speech {
            in_speech = false;
            let seg_len = k - seg_start;
            if seg_len >= min_speech_frames {
                let t0 =
                    ((seg_start * frame_hop) as f64 / params.sample_rate as f64 - pad_sec).max(0.0);
                let t1 = ((k * frame_hop + frame_len) as f64 / params.sample_rate as f64 + pad_sec)
                    .min(total_sec);
                segments.push(SpeechSegment::new(t0, t1));
            }
        }
    }

    if in_speech {
        let seg_len = n_frames - seg_start;
        if seg_len >= min_speech_frames {
            let t0 =
                ((seg_start * frame_hop) as f64 / params.sample_rate as f64 - pad_sec).max(0.0);
            let t1 = total_sec;
            segments.push(SpeechSegment::new(t0, t1));
        }
    }

    // If no voice segment was detected (pure silence), returns empty list
    split_long_segments(segments, params)
}

/// Split speech segments longer than `max_speech_duration_s`.
///
/// The Silero path splits while scanning probabilities; the energy detector has
/// no probability trace, so the split is applied afterwards. The overlap is
/// clamped below the chunk length so the loop always advances.
fn split_long_segments(segments: Vec<SpeechSegment>, params: &VadParams) -> Vec<SpeechSegment> {
    let max_s = params.max_speech_duration_s;
    if !max_s.is_finite() || max_s <= 0.0 {
        return segments;
    }
    // The Silero path subtracts one analysis window and two padding margins
    // before splitting; a limit that small means "do not split" (the upstream
    // code falls back to INT_MAX/2). The energy detector has no analysis window,
    // so only the padding term applies. The comparison stays in f32 because the
    // parameter is f32: widening one side would make an exact boundary such as
    // 0.6 s with a 300 ms pad fall on the wrong side of the test.
    let pad_s = params.speech_pad_ms as f32 / 1000.0;
    if max_s - 2.0 * pad_s <= 1e-6 {
        return segments;
    }
    let max_s = max_s as f64;
    let overlap = (params.samples_overlap.max(0.0) as f64).min(max_s * 0.5);
    let mut split = Vec::with_capacity(segments.len());
    for segment in segments {
        let mut t0 = segment.t0;
        while segment.t1 - t0 > max_s + 1e-9 {
            let t1 = (t0 + max_s).min(segment.t1);
            split.push(SpeechSegment::new(t0, t1));
            t0 = (t1 - overlap).max(t0 + 1e-9);
        }
        split.push(SpeechSegment::new(t0, segment.t1));
    }
    split
}

#[cfg(test)]
mod tests {
    use super::{split_long_segments, SpeechSegment, VadParams};

    #[test]
    fn splits_long_speech_with_overlap() {
        let params = VadParams {
            max_speech_duration_s: 1.0,
            samples_overlap: 0.1,
            ..VadParams::default()
        };
        let split = split_long_segments(vec![SpeechSegment::new(0.0, 3.0)], &params);
        assert_eq!(split.len(), 4);
        assert!((split[0].t1 - split[0].t0 - 1.0).abs() < 1e-6);
        assert!((split[1].t0 - 0.9).abs() < 1e-6);
        assert!((split[3].t1 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn limit_below_two_padding_margins_does_not_split() {
        let params = VadParams {
            max_speech_duration_s: 0.05,
            speech_pad_ms: 30,
            ..VadParams::default()
        };
        let segments = vec![SpeechSegment::new(0.0, 3.0)];
        assert_eq!(split_long_segments(segments.clone(), &params), segments);
    }

    #[test]
    fn exact_two_padding_margins_does_not_split() {
        let params = VadParams {
            max_speech_duration_s: 0.6,
            speech_pad_ms: 300,
            ..VadParams::default()
        };
        let segments = vec![SpeechSegment::new(0.0, 3.0)];
        assert_eq!(split_long_segments(segments.clone(), &params), segments);
    }

    #[test]
    fn unbounded_max_duration_is_a_noop() {
        let params = VadParams::default();
        let segments = vec![SpeechSegment::new(0.0, 30.0)];
        assert_eq!(split_long_segments(segments.clone(), &params), segments);
    }

    #[test]
    fn overlap_longer_than_chunk_still_advances() {
        let params = VadParams {
            max_speech_duration_s: 1.0,
            samples_overlap: 5.0,
            ..VadParams::default()
        };
        let split = split_long_segments(vec![SpeechSegment::new(0.0, 3.0)], &params);
        assert!(split.len() >= 3);
        assert!(split.windows(2).all(|pair| pair[1].t0 > pair[0].t0));
    }
}
