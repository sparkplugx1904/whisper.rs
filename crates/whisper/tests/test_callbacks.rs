use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
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
fn test_native_closure_callbacks() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping callback test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("create context");

    let segment_called = Arc::new(AtomicBool::new(false));
    let segment_called_clone = Arc::clone(&segment_called);

    let progress_count = Arc::new(AtomicUsize::new(0));
    let progress_count_clone = Arc::clone(&progress_count);

    let params = FullParamsBuilder::new(SamplingStrategy::Greedy { best_of: 1 })
        .on_new_segment(move |seg| {
            assert!(!seg.text.is_empty());
            segment_called_clone.store(true, Ordering::SeqCst);
        })
        .on_progress(move |progress| {
            assert!((0..=100).contains(&progress));
            progress_count_clone.fetch_add(1, Ordering::SeqCst);
        })
        .build();

    let segments = ctx.transcribe_pcm(&audio.samples, &params);
    assert!(!segments.is_empty());
    assert!(
        segment_called.load(Ordering::SeqCst),
        "on_new_segment closure must be called"
    );
    assert!(
        progress_count.load(Ordering::SeqCst) > 0,
        "on_progress closure must be called"
    );
}
