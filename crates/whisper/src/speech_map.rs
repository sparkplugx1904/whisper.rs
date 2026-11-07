//! Speech activity map used to skip silent audio windows.
//!
//! Whisper always processes fixed 30-second windows, so a window that contains
//! only silence still pays the full encoder and decoder cost. Mapping the VAD
//! output onto mel-frame indices lets the transcription loop jump directly to
//! the next region that actually contains speech instead of encoding and
//! decoding 30 seconds of silence.
//!
//! Frame indices are kept absolute with respect to the start of the input
//! audio (10 ms per mel frame, matching the encoder's hop length). Because the
//! map never rebases indices, timestamps produced from a skipped window need no
//! remapping: a frame index always refers to the same position in the original
//! audio.

use crate::params::WhisperParams;
use whisper_vad::{detect_speech_segments, VadParams};

/// Mel frames per second; the encoder consumes one frame per 10 ms of audio.
const FRAMES_PER_SECOND: f64 = 100.0;

/// Absolute mel-frame ranges that contain speech.
///
/// Ranges are sorted, non-empty, and non-overlapping. Adjacent ranges are
/// merged so a caller can treat each entry as a maximal contiguous speech
/// region.
#[derive(Debug, Clone, Default)]
pub struct SpeechMap {
    ranges: Vec<(usize, usize)>,
}

impl SpeechMap {
    /// Builds a speech map from 16 kHz mono samples.
    ///
    /// Returns `None` when VAD is disabled, which signals the caller to keep
    /// processing every window. When VAD is enabled an empty map is returned
    /// for audio that contains no speech at all.
    pub fn from_samples(samples: &[f32], params: &WhisperParams) -> Option<Self> {
        if !params.vad_enable {
            return None;
        }

        let vad_params = VadParams {
            threshold: params.vad_threshold,
            min_speech_duration_ms: params.vad_min_speech_duration_ms,
            min_silence_duration_ms: params.vad_min_silence_duration_ms,
            speech_pad_ms: params.vad_speech_pad_ms,
            sample_rate: 16_000,
        };

        let segments = detect_speech_segments(samples, &vad_params);
        let mut ranges: Vec<(usize, usize)> = segments
            .iter()
            .map(|segment| {
                let start = (segment.t0 * FRAMES_PER_SECOND).floor() as usize;
                let mut end = (segment.t1 * FRAMES_PER_SECOND).ceil() as usize;
                // A segment shorter than one frame still has to cover a frame,
                // otherwise the range would be empty and never match.
                if end <= start {
                    end = start + 1;
                }
                (start, end)
            })
            .collect();

        ranges.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            match merged.last_mut() {
                // Overlapping or touching ranges describe one continuous
                // speech region, so fold them together.
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }

        Some(Self { ranges: merged })
    }

    /// Returns `true` when no speech was detected.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Returns the sorted, merged speech ranges in mel frames.
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    /// Returns the start of the first speech range at or after `frame`.
    ///
    /// If `frame` already lies inside a speech range, that range's start is
    /// returned so the caller can rewind to the beginning of the utterance.
    pub fn next_speech_start(&self, frame: usize) -> Option<usize> {
        for &(start, end) in &self.ranges {
            // Ranges are sorted and disjoint, so the first range that ends
            // after `frame` either contains it or starts after it.
            if frame < end {
                return Some(start);
            }
        }
        None
    }

    /// Returns `true` when the half-open frame range `[start, end)` intersects
    /// any speech range. Touching ranges do not count as an intersection.
    pub fn overlaps(&self, start: usize, end: usize) -> bool {
        if start >= end {
            return false;
        }
        self.ranges
            .iter()
            .any(|&(range_start, range_end)| range_start < end && start < range_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: usize = 16_000;

    fn params_with_vad() -> WhisperParams {
        WhisperParams {
            vad_enable: true,
            vad_threshold: 0.5,
            vad_min_speech_duration_ms: 250,
            vad_min_silence_duration_ms: 100,
            vad_speech_pad_ms: 300,
            ..WhisperParams::default()
        }
    }

    fn silence(seconds: f64) -> Vec<f32> {
        vec![0.0; (seconds * SAMPLE_RATE as f64) as usize]
    }

    fn tone(seconds: f64, amplitude: f32) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn disabled_vad_returns_none() {
        let params = WhisperParams::default();
        assert!(!params.vad_enable);
        assert!(SpeechMap::from_samples(&silence(1.0), &params).is_none());
    }

    #[test]
    fn pure_silence_yields_empty_map() {
        let params = params_with_vad();
        let map = SpeechMap::from_samples(&silence(3.0), &params).expect("vad enabled");
        assert!(map.is_empty());
        assert!(map.ranges().is_empty());
        assert_eq!(map.next_speech_start(0), None);
        assert!(!map.overlaps(0, 300));
    }

    #[test]
    fn tone_burst_produces_padded_range() {
        let params = params_with_vad();
        let burst_start_sec = 1.0;
        let mut samples = silence(burst_start_sec);
        samples.extend(tone(0.5, 0.5));
        samples.extend(silence(1.5));

        let map = SpeechMap::from_samples(&samples, &params).expect("vad enabled");
        assert_eq!(map.ranges().len(), 1, "one contiguous burst expected");

        let (start, end) = map.ranges()[0];
        let burst_start_frame = (burst_start_sec * FRAMES_PER_SECOND) as usize;
        let pad_frames = params.vad_speech_pad_ms / 10;
        assert!(
            start + pad_frames <= burst_start_frame,
            "range start {start} should be at least {pad_frames} frames before the burst at {burst_start_frame}"
        );
        assert!(end > start);
        assert!(map.overlaps(burst_start_frame, burst_start_frame + 10));
    }

    #[test]
    fn next_speech_start_inside_and_outside_speech() {
        let map = SpeechMap {
            ranges: vec![(100, 200), (300, 400)],
        };

        assert_eq!(map.next_speech_start(0), Some(100));
        assert_eq!(map.next_speech_start(99), Some(100));
        assert_eq!(map.next_speech_start(100), Some(100));
        assert_eq!(map.next_speech_start(150), Some(100));
        assert_eq!(map.next_speech_start(200), Some(300));
        assert_eq!(map.next_speech_start(250), Some(300));
        assert_eq!(map.next_speech_start(399), Some(300));
        assert_eq!(map.next_speech_start(400), None);
        assert_eq!(map.next_speech_start(10_000), None);
    }

    #[test]
    fn overlaps_boundary_cases() {
        let map = SpeechMap {
            ranges: vec![(10, 20), (20, 30)],
        };

        assert!(map.overlaps(0, 11));
        assert!(map.overlaps(10, 20));
        assert!(map.overlaps(15, 16));
        assert!(map.overlaps(19, 21));
        assert!(map.overlaps(20, 30));
        assert!(map.overlaps(29, 100));

        assert!(!map.overlaps(0, 10));
        assert!(!map.overlaps(30, 40));
        assert!(!map.overlaps(20, 20));
        assert!(!map.overlaps(25, 25));
        assert!(!map.overlaps(5, 5));
    }
}
