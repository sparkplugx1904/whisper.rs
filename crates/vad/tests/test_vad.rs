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
