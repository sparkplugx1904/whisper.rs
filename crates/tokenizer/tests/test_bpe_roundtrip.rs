use std::path::{Path, PathBuf};
use whisper_model::WhisperModel;
use whisper_tokenizer::Tokenizer;

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
fn test_bpe_encode_decode_roundtrip() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping bpe roundtrip test");
        return;
    }

    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let tokenizer = Tokenizer::new(model.vocab);

    let test_phrases = [
        "And so my fellow Americans ask not what your country can do for you",
        "Hello, world! 12345.",
        "Special chars: @#$%^&*()_+=-~`",
        "こんにちは世界",
        "你好世界",
        "Jakarta, Indonesia",
    ];

    for &phrase in &test_phrases {
        let tokens = tokenizer.encode(phrase);
        assert!(
            !tokens.is_empty(),
            "Tokens should not be empty for '{}'",
            phrase
        );

        let decoded = tokenizer.decode(&tokens, false);
        assert_eq!(
            decoded, phrase,
            "BPE encode/decode mismatch for phrase '{}': got '{}'",
            phrase, decoded
        );
        eprintln!(
            "BPE verified: '{}' -> {} tokens -> '{}'",
            phrase,
            tokens.len(),
            decoded
        );
    }
}
