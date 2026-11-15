//! Silence-heavy VAD fixture: the speech map must keep the bursts and skip the
//! deep silence, so enabling VAD cannot drop speech or force silence through the
//! encoder.

use whisper_rs::params::WhisperParams;
use whisper_rs::SpeechMap;
use whisper_vad::{detect_speech_segments, VadParams};

const SAMPLE_RATE: usize = 16_000;

fn tone_amp(seconds: f32, amplitude: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * seconds) as usize;
    (0..n)
        .map(|i| {
            (2.0 * std::f32::consts::PI * 440.0 * (i as f32 / SAMPLE_RATE as f32)).sin() * amplitude
        })
        .collect()
}

fn silence(seconds: f32) -> Vec<f32> {
    vec![0.0; (SAMPLE_RATE as f32 * seconds) as usize]
}

/// 2 s quiet, 2 s tone, 4 s quiet, 2 s tone, 2 s quiet.
fn silence_heavy_corpus() -> Vec<f32> {
    let mut audio = silence(2.0);
    audio.extend(tone_amp(2.0, 0.5));
    audio.extend(silence(4.0));
    audio.extend(tone_amp(2.0, 0.5));
    audio.extend(silence(2.0));
    audio
}

fn energy_params() -> WhisperParams {
    WhisperParams {
        vad_enable: true,
        // Pin the dependency-free detector: a Silero model path would make the
        // expected ranges depend on a downloadable model.
        vad_model_path: None,
        vad_min_speech_duration_ms: 100,
        vad_min_silence_duration_ms: 200,
        vad_speech_pad_ms: 100,
        ..WhisperParams::default()
    }
}

#[test]
fn speech_map_keeps_bursts_and_skips_deep_silence() {
    let audio = silence_heavy_corpus();
    let params = energy_params();
    let map = SpeechMap::from_samples(&audio, &params).expect("VAD is enabled");

    let ranges = map.ranges();
    assert_eq!(ranges.len(), 2, "one range per burst: {ranges:?}");

    // The whole burst must be mapped, not just its first second: the encoder
    // skips every frame the map does not cover.
    assert!(
        map.overlaps(200, 400),
        "first burst must be mapped: {ranges:?}"
    );
    assert!(
        map.overlaps(800, 1000),
        "second burst must be mapped: {ranges:?}"
    );

    // The 4 s of silence sits between the ranges. Each edge loses 100 ms to the
    // speech pad and gains up to 20 ms of analysis-window pre-roll, so the hole
    // is 3.65 s rather than the full 4 s; anything outside that band means the
    // gap leaked into the map or speech was dropped.
    let gap = ranges[1].0 - ranges[0].1;
    assert!(
        (350..=380).contains(&gap),
        "the 4 s silence gap must remain outside the map, got {gap} frames: {ranges:?}"
    );

    // Guards the outer silence as well: the map must not cover the leading 2 s
    // or the trailing 2 s.
    let covered: usize = ranges.iter().map(|(start, end)| end - start).sum();
    let total_frames = audio.len().div_ceil(SAMPLE_RATE / 100);
    assert!(
        covered * 2 < total_frames,
        "VAD should skip most of the 8 s of silence: {covered}/{total_frames}"
    );
}

/// A speech map can never expose the segment split: `segments_to_ranges` merges
/// adjacent ranges, so the split pieces of one burst collapse back into a single
/// range. The parameter therefore has to be observed on the detector output.
#[test]
fn unlimited_segment_duration_does_not_split_the_burst() {
    let mut audio = silence(2.0);
    audio.extend(tone_amp(3.0, 0.5));
    audio.extend(silence(2.0));

    let unlimited = VadParams {
        max_speech_duration_s: f32::MAX,
        ..VadParams::default()
    };
    let finite = VadParams {
        max_speech_duration_s: 1.0,
        ..VadParams::default()
    };

    let unlimited_segments = detect_speech_segments(&audio, &unlimited);
    let finite_segments = detect_speech_segments(&audio, &finite);
    assert_eq!(
        unlimited_segments.len(),
        1,
        "an unbounded limit must keep the 3 s burst whole: {unlimited_segments:?}"
    );
    assert!(
        finite_segments.len() >= 3,
        "a 1 s limit must split the 3 s burst: {finite_segments:?}"
    );
    assert!(finite_segments
        .iter()
        .all(|segment| segment.t1 - segment.t0 <= 1.0 + 1e-6));

    let params = WhisperParams {
        vad_max_speech_duration_s: f32::MAX,
        ..energy_params()
    };
    let map = SpeechMap::from_samples(&audio, &params).expect("VAD is enabled");
    assert_eq!(map.ranges().len(), 1);
    assert!(
        map.overlaps(200, 500),
        "burst must stay mapped: {:?}",
        map.ranges()
    );
}

/// The energy detector's threshold floor is `vad_threshold * 0.02`. A burst at
/// amplitude 0.05 (RMS 0.0354) clears the default 0.01 floor but not the 0.04
/// floor of a 2.0 threshold, so this fixture proves the parameter reaches the
/// detector in both directions.
#[test]
fn threshold_floor_decides_borderline_bursts() {
    let mut audio = silence(2.0);
    audio.extend(tone_amp(2.0, 0.05));
    audio.extend(silence(2.0));

    let detected = SpeechMap::from_samples(&audio, &energy_params()).expect("VAD is enabled");
    assert!(
        !detected.is_empty(),
        "0.0354 RMS must clear the default 0.01 floor"
    );

    let strict = WhisperParams {
        vad_threshold: 2.0,
        ..energy_params()
    };
    let rejected = SpeechMap::from_samples(&audio, &strict).expect("VAD is enabled");
    assert!(
        rejected.is_empty(),
        "0.0354 RMS must fall under a 0.04 floor: {:?}",
        rejected.ranges()
    );
}
