use std::path::{Path, PathBuf};
use whisper_audio::generate_slaney_mel_filters;
use whisper_model::WhisperModel;

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
fn test_slaney_mel_filters_matches_model_reference() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping test", model_path);
        return;
    }

    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let ref_filters = &model.filters.data;

    let dynamic_filters = generate_slaney_mel_filters(80, 400, 16000);
    assert_eq!(dynamic_filters.len(), ref_filters.len());

    let mut max_diff = 0.0f32;
    for (i, (&dyn_val, &ref_val)) in dynamic_filters.iter().zip(ref_filters.iter()).enumerate() {
        let diff = (dyn_val - ref_val).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff <= 1e-4,
            "Filter difference too large at index {}: dynamic {} vs reference {}, diff {}",
            i,
            dyn_val,
            ref_val,
            diff
        );
    }
    eprintln!(
        "Maximum Slaney filterbank diff vs reference: {:.6e}",
        max_diff
    );
    assert!(max_diff <= 1e-4);

    // Also test 128 mel bins generation for large-v3-turbo
    let turbo_filters = generate_slaney_mel_filters(128, 400, 16000);
    assert_eq!(turbo_filters.len(), 128 * 201);
    for &val in &turbo_filters {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }
}
