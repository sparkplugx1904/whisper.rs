#![forbid(unsafe_code)]
//! Voice Activity Detection (VAD) for whisper.rs.
//!
//! Implementation: Adaptive Energy-based Voice Activity Detection with Dynamic Noise Floor Tracking.
//! Analyzes short-time frame root-mean-square (RMS) energy, applies noise floor percentile estimation,
//! and performs temporal smoothing with speech/silence duration constraints and padding.

#[derive(Debug, Clone)]
pub struct VadParams {
    pub threshold: f32,
    pub min_speech_duration_ms: usize,
    pub min_silence_duration_ms: usize,
    pub speech_pad_ms: usize,
    pub sample_rate: usize,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
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
    segments
}
