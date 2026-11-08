use std::path::{Path, PathBuf};
use whisper_rs::audio::WavAudio;
use whisper_rs::{FullParamsBuilder, SamplingStrategy, WhisperContext};

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
fn test_silero_vad_full_pipeline_on_jfk() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let vad_path = resolve_model_path("for-tests-silero-v6.2.0-ggml.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !vad_path.exists() || !wav_path.exists() {
        eprintln!("Model, VAD model, or sample not found, skipping Silero pipeline test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let ctx = WhisperContext::load_model(&model_path).expect("load model");

    let params = FullParamsBuilder::new(SamplingStrategy::Greedy { best_of: 1 })
        .vad_enable(true)
        .vad_model_path(vad_path.to_string_lossy().into_owned())
        .build();

    let segments = ctx.transcribe_pcm(&audio.samples, &params);
    assert!(
        !segments.is_empty(),
        "Silero VAD must keep the speech in jfk.wav"
    );

    let text: String = segments.iter().map(|s| s.text.as_str()).collect();
    assert!(
        !text.trim().is_empty(),
        "transcript produced through Silero VAD must not be blank"
    );

    // jfk.wav is 11.0 s long; padded VAD ranges must never push a timestamp
    // past the end of the audio by more than the 0.5 s tolerance.
    for seg in &segments {
        assert!(
            (0.0..=11.5).contains(&seg.t0),
            "segment start {} outside audio bounds",
            seg.t0
        );
        assert!(
            (0.0..=11.5).contains(&seg.t1),
            "segment end {} outside audio bounds",
            seg.t1
        );
        assert!(seg.t1 >= seg.t0, "segment end must not precede its start");
    }
}
