use std::path::{Path, PathBuf};
use whisper_audio::WavAudio;
use whisper_vad::{detect_speech_segments, VadParams};

fn workspace_path(rel: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().unwrap().parent().unwrap();
    ws_root.join(rel)
}

#[test]
fn test_detect_speech_jfk() {
    let wav_path = workspace_path("samples/jfk.wav");
    assert!(wav_path.exists());

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let params = VadParams::default();

    let segments = detect_speech_segments(&audio.samples, &params);
    eprintln!("Detected {} speech segments in JFK audio:", segments.len());
    for s in &segments {
        eprintln!("  [{:.2}s --> {:.2}s]", s.t0, s.t1);
    }

    assert!(!segments.is_empty(), "Speech segments must not be empty");
    assert!(segments[0].t0 < 1.0, "JFK speech starts within 1.0s");
    assert!(
        segments.last().unwrap().t1 > 9.0,
        "JFK speech extends past 9.0s"
    );
}

#[test]
fn test_detect_silence_returns_empty() {
    let silence = vec![0.0f32; 16000 * 5]; // 5 seconds of absolute silence
    let params = VadParams::default();
    let segments = detect_speech_segments(&silence, &params);
    assert!(
        segments.is_empty(),
        "Pure silence must produce zero speech segments"
    );
}

#[test]
fn test_detect_short_audio() {
    let short_audio = vec![0.1f32; 100]; // Shorter than frame length (480 samples)
    let params = VadParams::default();
    let segments = detect_speech_segments(&short_audio, &params);
    assert!(
        segments.is_empty(),
        "Audio shorter than window must return empty segments"
    );
}

fn tone_amp(seconds: f32, amplitude: f32) -> Vec<f32> {
    let n = (16_000.0 * seconds) as usize;
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32 / 16_000.0)).sin() * amplitude)
        .collect()
}

fn tone(seconds: f32) -> Vec<f32> {
    tone_amp(seconds, 0.5)
}

fn silence(seconds: f32) -> Vec<f32> {
    vec![0.0; (16_000.0 * seconds) as usize]
}

/// Deterministic white noise from a fixed-seed xorshift, uniform in
/// `[-amplitude, amplitude]` (RMS ≈ `amplitude / sqrt(3)`).
///
/// A noise bed keeps the median frame energy above zero, which is what makes the
/// adaptive `median * 2` term of the threshold bind; digital-zero silence always
/// collapses the threshold to its fixed floor.
fn noise(seconds: f32, amplitude: f32) -> Vec<f32> {
    let n = (16_000.0 * seconds) as usize;
    let mut x: u32 = 0x1234_5678;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x as f32 / u32::MAX as f32) * 2.0 - 1.0) * amplitude
        })
        .collect()
}

/// Silence-heavy fixture: 2 s quiet, 2 s speech, 4 s quiet, 2 s speech, 2 s quiet.
fn silence_heavy_corpus() -> Vec<f32> {
    let mut audio = silence(2.0);
    audio.extend(tone(2.0));
    audio.extend(silence(4.0));
    audio.extend(tone(2.0));
    audio.extend(silence(2.0));
    audio
}

#[test]
fn test_silence_heavy_corpus_detects_only_the_speech_bursts() {
    let params = VadParams {
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 200,
        speech_pad_ms: 100,
        ..VadParams::default()
    };
    let segments = detect_speech_segments(&silence_heavy_corpus(), &params);
    assert_eq!(segments.len(), 2, "two tone bursts expected: {segments:?}");
    assert!(segments[0].t0 < 2.2 && segments[0].t1 > 3.8);
    assert!(segments[1].t0 < 8.2 && segments[1].t1 > 9.8);
    // The 4 s of silence loses one 100 ms pad at each edge and gains up to 20 ms
    // of analysis-window pre-roll, so the hole is 3.65 s, not the full 4 s.
    let gap = segments[1].t0 - segments[0].t1;
    assert!(
        (3.5..=3.8).contains(&gap),
        "the 4 s silence gap must not be joined, got {gap:.3}s: {segments:?}"
    );
}

#[test]
fn test_max_duration_splits_detected_speech() {
    let params = VadParams {
        max_speech_duration_s: 1.0,
        speech_pad_ms: 30,
        samples_overlap: 0.1,
        ..VadParams::default()
    };
    let mut audio = silence(2.0);
    audio.extend(tone(3.0));
    audio.extend(silence(2.0));
    let segments = detect_speech_segments(&audio, &params);
    assert!(
        segments.len() >= 3,
        "3 s of speech with a 1 s limit: {segments:?}"
    );
    assert!(segments.windows(2).all(|pair| pair[1].t0 > pair[0].t0));
    assert!(segments
        .iter()
        .all(|segment| segment.t1 - segment.t0 <= 1.0 + 1e-6));
    assert!(segments.len() <= 5, "split loop ran away: {segments:?}");
}

/// The fixed part of the threshold is `threshold * 0.02`. A burst at amplitude
/// 0.05 (RMS 0.0354) clears the default 0.01 floor and falls under the 0.04 floor
/// of a 2.0 threshold, so both outcomes are visible with a ~2x margin on each
/// side. Absolute-zero silence keeps the median at zero, which is why the
/// parameter is only observable with a burst near the floor.
#[test]
fn test_threshold_floor_decides_borderline_bursts() {
    let mut audio = silence(2.0);
    audio.extend(tone_amp(2.0, 0.05));
    audio.extend(silence(2.0));

    let detected = detect_speech_segments(&audio, &VadParams::default());
    assert_eq!(
        detected.len(),
        1,
        "0.0354 RMS must clear the default 0.01 floor"
    );

    let strict = VadParams {
        threshold: 2.0,
        ..VadParams::default()
    };
    assert!(
        detect_speech_segments(&audio, &strict).is_empty(),
        "0.0354 RMS must fall under a 0.04 floor"
    );
}

/// A 0.02-amplitude noise bed has RMS ≈ 0.0115, which lifts the median-based
/// threshold to ≈ 0.023 — above the fixed 0.01 floor. Neither the bed itself nor
/// a 0.02-amplitude tone (RMS 0.0141) may be reported as speech, while a loud
/// burst still must be. Without the `median * 2` term the bed alone clears the
/// 0.01 floor, so this fixture pins the adaptive half of the threshold.
#[test]
fn test_noise_bed_is_not_reported_as_speech() {
    let mut bed = noise(6.0, 0.02);
    assert!(
        detect_speech_segments(&bed, &VadParams::default()).is_empty(),
        "a stationary noise bed must not produce speech"
    );

    bed = noise(3.0, 0.02);
    bed.extend(tone_amp(2.0, 0.02));
    bed.extend(noise(3.0, 0.02));
    assert!(
        detect_speech_segments(&bed, &VadParams::default()).is_empty(),
        "a tone below the noise-driven threshold must not produce speech"
    );

    let mut loud = noise(3.0, 0.02);
    loud.extend(tone_amp(2.0, 0.5));
    loud.extend(noise(3.0, 0.02));
    let segments = detect_speech_segments(&loud, &VadParams::default());
    assert_eq!(segments.len(), 1, "loud burst over the bed: {segments:?}");
    assert!(segments[0].t0 < 3.2 && segments[0].t1 > 4.8);
}
