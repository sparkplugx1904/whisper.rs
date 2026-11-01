use std::path::{Path, PathBuf};
use whisper_rs_pure::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs_pure::model::WhisperModel;
use whisper_rs_pure::{WhisperContext, WhisperParams};

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
fn test_language_detection() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Multilingual model not found, skipping language detection test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");

    let mel = compute_log_mel_spectrogram(
        &audio.samples,
        &ctx.model.filters.data,
        ctx.model.filters.n_mel,
    );

    let (lang, prob) = ctx.detect_language(&mel);
    eprintln!("Detected language: '{}' with probability {:.4}", lang, prob);

    assert_eq!(
        lang, "en",
        "Expected English language to be detected for jfk.wav"
    );
    assert!(
        prob > 0.5,
        "Expected high confidence probability for English"
    );
}

#[test]
fn test_transcribe_with_timestamps_and_srt() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");

    if !model_path.exists() {
        eprintln!("Model not found, skipping timestamp transcription test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");

    let mel = compute_log_mel_spectrogram(
        &audio.samples,
        &ctx.model.filters.data,
        ctx.model.filters.n_mel,
    );

    let params = WhisperParams {
        no_timestamps: false,
        ..Default::default()
    };

    let segments = ctx.transcribe(&mel, &params);
    eprintln!("Generated {} segments:", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        eprintln!("[{:.3} --> {:.3}] {}", seg.t0, seg.t1, seg.text);
        let srt = seg.to_srt(i + 1);
        let vtt = seg.to_vtt();
        assert!(!srt.is_empty());
        assert!(!vtt.is_empty());
    }

    assert!(!segments.is_empty(), "Segments should not be empty");
    assert!(segments[0].t0 < 1.0, "First segment should start near 0.0s");

    let full_text: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    assert!(
        full_text.contains("And so my fellow Americans")
            && full_text.contains("ask what you can do for your country"),
        "Full text does not match JFK speech: '{}'",
        full_text
    );
}
