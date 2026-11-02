use std::path::{Path, PathBuf};
use whisper_rs::audio::WavAudio;
use whisper_rs::filter::compression_ratio;
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
fn test_compression_ratio_filter() {
    let repetitive_text = "And so my fellow Americans and so my fellow Americans and so my fellow Americans and so my fellow Americans and so my fellow Americans";
    let cr_rep = compression_ratio(repetitive_text);
    eprintln!("Repetitive text CR: {:.2}", cr_rep);
    assert!(
        cr_rep > 2.0,
        "Repetitive text should have high compression ratio"
    );

    let natural_text =
        "Ask not what your country can do for you, ask what you can do for your country.";
    let cr_nat = compression_ratio(natural_text);
    eprintln!("Natural text CR: {:.2}", cr_nat);
    assert!(
        cr_nat < 1.8,
        "Natural text should have low compression ratio"
    );
}

#[test]
fn test_beam_search_transcription() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");

    if !model_path.exists() {
        eprintln!(
            "Model not found at {:?}, skipping beam search test",
            model_path
        );
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");

    let params = WhisperParams {
        beam_size: 3,
        no_timestamps: true,
        ..Default::default()
    };

    let segments = ctx.transcribe_pcm(&audio.samples, &params);
    assert!(!segments.is_empty());

    let text: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    eprintln!("Beam search text: '{}'", text);

    assert!(
        text.contains("And so my fellow Americans")
            && text.contains("ask what you can do for your country"),
        "Beam search output mismatch: '{}'",
        text
    );
}

#[test]
fn test_state_reuse() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");

    if !model_path.exists() {
        eprintln!(
            "Model not found at {:?}, skipping state reuse test",
            model_path
        );
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");

    let mut state = ctx.create_state();
    let params = WhisperParams {
        no_timestamps: true,
        ..Default::default()
    };

    // First run
    ctx.full(&mut state, &params, &audio.samples)
        .expect("first run");
    let text1 = state.segments[0].text.clone();

    // Second run with reused state
    ctx.full(&mut state, &params, &audio.samples)
        .expect("second run");
    let text2 = state.segments[0].text.clone();

    assert_eq!(text1, text2, "Reused state must yield deterministic result");
}
