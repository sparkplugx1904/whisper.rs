use std::path::{Path, PathBuf};
use whisper_rs::audio::WavAudio;
use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams};

fn workspace_path(rel: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().unwrap().parent().unwrap();
    ws_root.join(rel)
}

fn resolve_model_path(name: &str) -> PathBuf {
    let p = workspace_path(&format!("models/{}", name));
    if p.exists() {
        return p;
    }
    PathBuf::from(format!("/tmp/models/{}", name))
}

#[test]
fn test_segment_probability_and_fallback_stats() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping fallback criteria test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");

    let params = WhisperParams::default();
    let segments = ctx.transcribe_pcm(&audio.samples, &params);
    assert!(!segments.is_empty());

    for seg in &segments {
        assert!(seg.avg_logprob.is_finite(), "avg_logprob must be finite");
        assert!(seg.avg_logprob <= 0.0, "avg_logprob must be <= 0.0");
        assert!(
            seg.no_speech_prob.is_finite(),
            "no_speech_prob must be finite"
        );
        assert!(
            (0.0..=1.0).contains(&seg.no_speech_prob),
            "no_speech_prob must be between 0 and 1"
        );

        let json = seg.to_json();
        assert!(json.contains("avg_logprob"));
        assert!(json.contains("no_speech_prob"));
    }
}
