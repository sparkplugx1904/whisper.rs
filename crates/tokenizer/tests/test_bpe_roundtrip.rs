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

#[test]
fn test_bpe_matches_whisper_cpp_token_ids() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping BPE parity test");
        return;
    }

    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let tokenizer = Tokenizer::new(model.vocab);

    // Generated with whisper.cpp's whisper_tokenize() on ggml-tiny.bin.
    let cases: &[(&str, &[i32])] = &[
        ("Hello, world!", &[15947, 11, 1002, 0]),
        ("don't stop", &[13966, 380, 1590]),
        ("I'll be there", &[40, 603, 312, 456]),
        ("  leading spaces", &[220, 5775, 7673]),
        ("abc123def", &[455, 66, 4762, 18, 20595]),
        ("你好，世界", &[26410, 171, 120, 234, 24486]),
        ("ça va très bien", &[1138, 64, 2773, 504, 1462, 82, 3610]),
        ("unbelievable", &[409, 13729]),
        (
            "the quick brown fox jumps over the lazy dog",
            &[3322, 1702, 6292, 21026, 16704, 670, 264, 14847, 3000],
        ),
        ("TESTING 123 !!!", &[13639, 6840, 3017, 34466, 25901]),
        ("a\tb\nc", &[64, 197, 65, 198, 66]),
        ("word—dash", &[7462, 2958, 27476, 71]),
        ("こんにちは世界", &[38088, 24486]),
    ];

    for (text, expected) in cases {
        let actual = tokenizer.encode(text);
        assert_eq!(
            &actual, expected,
            "token mismatch for '{}': got {:?}",
            text, actual
        );
    }
}
