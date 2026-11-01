use std::path::{Path, PathBuf};
use whisper_audio::WavAudio;
use whisper_vad::{detect_speech_segments, SileroVadModel, VadParams};

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
fn test_load_silero_vad_model() {
    let path = Path::new("/tmp/whisper_ref/models/for-tests-silero-v6.2.0-ggml.bin");
    if !path.exists() {
        eprintln!("Silero test model not found, skipping");
        return;
    }

    let model = SileroVadModel::load_from_file(path).expect("load silero model");
    assert_eq!(model.model_type, "silero-16k");
    assert_eq!(model.major_version, 6);
    assert_eq!(model.minor_version, 2);
    assert_eq!(model.patch_version, 0);
}
