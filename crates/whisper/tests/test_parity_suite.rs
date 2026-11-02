use std::fs::File;
use std::io::Read;
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
fn test_zero_samples_handling() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping zero samples test");
        return;
    }

    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");
    let params = WhisperParams::default();

    // 1. Completely empty slice: should return empty segments without panicking
    let segments_empty = ctx.transcribe_pcm(&[], &params);
    assert!(
        segments_empty.is_empty(),
        "Empty input must return empty segments"
    );

    // 2. Ultra-short slice (< 100ms / < 1600 samples): must return gracefully without panic
    let short_samples = vec![0.0f32; 800]; // 50ms
    let segments_short = ctx.transcribe_pcm(&short_samples, &params);
    assert!(segments_short.is_empty() || segments_short[0].text.trim().is_empty());
}

#[test]
fn test_in_memory_buffer_loader() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");

    if !model_path.exists() {
        eprintln!("Model not found, skipping buffer loader test");
        return;
    }

    // Read full model file into in-memory bytes
    let mut model_file = File::open(&model_path).expect("open model file");
    let mut buffer = Vec::new();
    model_file
        .read_to_end(&mut buffer)
        .expect("read model to buffer");
    assert!(!buffer.is_empty());

    // Load from in-memory byte buffer
    let model = WhisperModel::load_from_bytes(buffer).expect("load from buffer");
    assert_eq!(model.hparams.n_vocab, 51864);

    let ctx = WhisperContext::new(model).expect("init context");
    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");

    let params = WhisperParams {
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
    assert!(
        text.contains("And so my fellow Americans")
            && text.contains("ask what you can do for your country"),
        "Buffer-loaded model failed to transcribe accurately: '{}'",
        text
    );
}

#[test]
fn test_vad_full_speech_and_silence() {
    let wav_path = workspace_path("samples/jfk.wav");
    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let params = whisper_vad::VadParams::default();

    // Active speech test on JFK
    let segments_speech = whisper_vad::detect_speech_segments(&audio.samples, &params);
    assert!(
        !segments_speech.is_empty(),
        "Speech segments must be detected on JFK speech"
    );
    assert!(
        segments_speech[0].t0 < 1.5,
        "Speech should begin near start of audio"
    );

    // Complete silence test (zeros)
    let silence = vec![0.0f32; 16000 * 5]; // 5 seconds of total silence
    let segments_silence = whisper_vad::detect_speech_segments(&silence, &params);
    assert!(
        segments_silence.is_empty(),
        "Pure silence must produce 0 speech segments"
    );
}
