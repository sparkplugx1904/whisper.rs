use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use whisper_rs::audio::WavAudio;
use whisper_rs::{FullParamsBuilder, SamplingStrategy, WhisperContext, WhisperParams};

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
fn test_encoder_and_logits_callbacks_fire() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping engine callback test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let ctx = WhisperContext::load_model(&model_path).expect("load model");
    let n_vocab = ctx.model.hparams.n_vocab as usize;

    let encoder_calls = Arc::new(AtomicUsize::new(0));
    let logits_calls = Arc::new(AtomicUsize::new(0));

    let encoder_calls_cb = Arc::clone(&encoder_calls);
    let logits_calls_cb = Arc::clone(&logits_calls);

    let params = FullParamsBuilder::new(SamplingStrategy::Greedy { best_of: 1 })
        .on_encoder_begin(move || {
            encoder_calls_cb.fetch_add(1, Ordering::SeqCst);
            true
        })
        .on_logits(move |_tokens, logits| {
            assert_eq!(
                logits.len(),
                n_vocab,
                "logits callback must receive one full vocabulary row"
            );
            logits_calls_cb.fetch_add(1, Ordering::SeqCst);
        })
        .build();

    let segments = ctx.transcribe_pcm(&audio.samples, &params);
    assert!(!segments.is_empty(), "transcript must not be empty");
    assert!(
        encoder_calls.load(Ordering::SeqCst) >= 1,
        "on_encoder_begin must be called before each encoder pass"
    );
    assert!(
        logits_calls.load(Ordering::SeqCst) >= 1,
        "on_logits must be called at least once per decoding step"
    );
}

#[test]
fn test_encoder_callback_abort_yields_no_segments() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping encoder abort test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let ctx = WhisperContext::load_model(&model_path).expect("load model");

    let params = WhisperParams {
        on_encoder_begin: Some(Arc::new(|| false)),
        ..WhisperParams::default()
    };

    let mut state = ctx.create_state();
    ctx.full(&mut state, &params, &audio.samples)
        .expect("aborted run must still return cleanly");

    assert!(
        state.segments.is_empty(),
        "returning false from on_encoder_begin must stop before any window is encoded"
    );
    assert_eq!(
        state.timings.n_encode, 0,
        "no encoder pass may run once the callback refuses"
    );
}
