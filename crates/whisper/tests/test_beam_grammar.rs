use std::path::{Path, PathBuf};
use whisper_rs::audio::WavAudio;
use whisper_rs::model::WhisperModel;
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
fn test_beam_search_with_grammar() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping beam grammar test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("create context");

    let grammar_str = r#"
root ::= " And so my fellow Americans" (" ask" | " say")
"#;

    let params = FullParamsBuilder::new(SamplingStrategy::BeamSearch { beam_size: 2 })
        .grammar(grammar_str)
        .no_timestamps(true)
        .build();
    let debug_params = params.clone();
    eprintln!("Running transcribe_pcm in test_beam_grammar...");
    let segments = ctx.transcribe_pcm(&audio.samples, &debug_params);
    eprintln!("Done transcribe_pcm, segments count: {}", segments.len());
    for (i, s) in segments.iter().enumerate() {
        eprintln!("  seg {}: text='{}', tokens={:?}", i, s.text, s.tokens);
    }
    assert!(!segments.is_empty(), "Segments must not be empty");
    let text = segments[0].text.trim();
    eprintln!("Beam Search + Grammar output: '{}'", text);
    assert!(
        text.starts_with("And so my fellow Americans ask"),
        "Beam search output must follow grammar rule"
    );
}
