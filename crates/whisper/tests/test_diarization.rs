use std::path::{Path, PathBuf};
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
fn test_tinydiarize_speaker_turn_flag() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping diarization test");
        return;
    }

    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");

    let params = FullParamsBuilder::new(SamplingStrategy::Greedy { best_of: 1 })
        .tdrz_enable(true)
        .build();

    assert!(params.tdrz_enable, "tdrz_enable must be active");

    // Test token decoder handling speaker turn token 50363
    let tokens = vec![50363, 15496, 11, 995]; // [SPEAKER_TURN], "Hello", ",", "world"
    let text = ctx.tokenizer.decode(&tokens, true);
    assert!(!text.is_empty());
}
