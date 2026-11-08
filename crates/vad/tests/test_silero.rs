use std::path::{Path, PathBuf};

use whisper_vad::{segments_from_probs, SileroVad, VadParams};

fn workspace_path(rel: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().unwrap().parent().unwrap();
    ws_root.join(rel)
}

fn resolve_model_path(name: &str) -> PathBuf {
    let path = workspace_path(&format!("models/{name}"));
    if path.exists() {
        return path;
    }
    PathBuf::from(format!("/tmp/models/{name}"))
}

#[test]
fn test_silero_rejects_invalid_model() {
    let error = SileroVad::from_bytes(b"not a ggml model").unwrap_err();
    assert!(error.to_string().contains("bad magic"));
}

#[test]
fn test_silero_matches_whisper_cpp_probabilities() {
    let model_path = resolve_model_path("for-tests-silero-v6.2.0-ggml.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("VAD model or sample not found, skipping Silero parity test");
        return;
    }

    let bytes = std::fs::read(&model_path).expect("read VAD model");
    let mut vad = SileroVad::from_bytes(&bytes).expect("load VAD model");
    let audio = whisper_audio::WavAudio::load_from_file(&wav_path).expect("load wav");

    let n_window = vad.n_window();
    let probs = vad.detect_probs(&audio.samples);
    assert_eq!(probs.len(), 344);

    // Reference values from whisper.cpp's whisper_vad_detect_speech().
    assert!((probs[0] - 0.001695).abs() < 1e-3);
    assert!((probs[1] - 0.083180).abs() < 1e-3);
    assert!((probs[50] - 0.985183).abs() < 1e-3);

    let params = VadParams::default();
    let segments = segments_from_probs(probs, &params, n_window);
    assert!(!segments.is_empty());
    let padded_audio_sec = (probs.len() * n_window) as f64 / params.sample_rate as f64;
    for segment in &segments {
        assert!(segment.t1 >= segment.t0);
        assert!(segment.t1 <= padded_audio_sec);
    }
}

#[test]
fn test_silero_reset_clears_recurrent_state() {
    let model_path = resolve_model_path("for-tests-silero-v6.2.0-ggml.bin");
    if !model_path.exists() {
        eprintln!("VAD model not found, skipping Silero reset test");
        return;
    }

    let bytes = std::fs::read(&model_path).expect("read VAD model");
    let mut vad = SileroVad::from_bytes(&bytes).expect("load VAD model");
    let samples = vec![0.0f32; 16_000];

    let first = vad.detect_probs(&samples).to_vec();
    vad.reset_state();
    let second = vad.detect_probs(&samples).to_vec();

    assert_eq!(first, second);
}
