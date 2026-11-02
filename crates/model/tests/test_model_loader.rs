use std::path::{Path, PathBuf};
use whisper_model::{ModelType, WhisperModel};

fn resolve_model_path(name: &str) -> PathBuf {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(format!("models/{}", name));
    if ws.exists() {
        return ws;
    }
    PathBuf::from(format!("/tmp/models/{}", name))
}

#[test]
fn test_load_tiny_en() {
    let path = resolve_model_path("ggml-tiny.en.bin");
    if !path.exists() {
        eprintln!("Model file {:?} not found, skipping integration test", path);
        return;
    }

    let model = WhisperModel::load_from_file(path).expect("Failed to load ggml-tiny.en.bin");

    // Hyperparameters check
    assert_eq!(model.hparams.n_vocab, 51864);
    assert_eq!(model.hparams.n_audio_ctx, 1500);
    assert_eq!(model.hparams.n_audio_state, 384);
    assert_eq!(model.hparams.n_audio_head, 6);
    assert_eq!(model.hparams.n_audio_layer, 4);
    assert_eq!(model.hparams.n_text_ctx, 448);
    assert_eq!(model.hparams.n_text_state, 384);
    assert_eq!(model.hparams.n_text_head, 6);
    assert_eq!(model.hparams.n_text_layer, 4);
    assert_eq!(model.hparams.n_mels, 80);
    assert_eq!(model.hparams.ftype, 1);
    assert_eq!(model.hparams.qntvr, 0);
    assert_eq!(model.hparams.model_type, ModelType::Tiny);
    assert!(!model.hparams.is_multilingual());

    // Mel filters check
    assert_eq!(model.filters.n_mel, 80);
    assert_eq!(model.filters.n_fft, 201);
    assert_eq!(model.filters.data.len(), 80 * 201);

    // Vocab check
    assert_eq!(model.vocab.n_vocab, 51864);
    assert_eq!(model.vocab.token_eot, 50256);
    assert_eq!(model.vocab.token_sot, 50257);
    assert_eq!(model.vocab.token_beg, 50363);

    // Tensors check
    assert_eq!(model.tensors.len(), 167);
    assert_eq!(model.total_weight_bytes, 77110272);

    // Sample tensor check
    let conv1 = model
        .tensor_info("encoder.conv1.weight")
        .expect("encoder.conv1.weight must exist");
    assert_eq!(conv1.ne, vec![3, 80, 384]);

    let conv1_f32 = model
        .get_tensor_f32("encoder.conv1.weight")
        .expect("dequant conv1 f32");
    assert_eq!(conv1_f32.len(), 3 * 80 * 384);
}

#[test]
fn test_large_v3_turbo_header_detection() {
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Cursor;
    use whisper_model::header::{Hyperparameters, ModelType, GGML_FILE_MAGIC};

    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(GGML_FILE_MAGIC).unwrap();
    buf.write_i32::<LittleEndian>(51866).unwrap(); // n_vocab
    buf.write_i32::<LittleEndian>(1500).unwrap(); // n_audio_ctx
    buf.write_i32::<LittleEndian>(1280).unwrap(); // n_audio_state
    buf.write_i32::<LittleEndian>(20).unwrap(); // n_audio_head
    buf.write_i32::<LittleEndian>(32).unwrap(); // n_audio_layer (32 encoder layers)
    buf.write_i32::<LittleEndian>(448).unwrap(); // n_text_ctx
    buf.write_i32::<LittleEndian>(1280).unwrap(); // n_text_state
    buf.write_i32::<LittleEndian>(20).unwrap(); // n_text_head
    buf.write_i32::<LittleEndian>(4).unwrap(); // n_text_layer (4 decoder layers!)
    buf.write_i32::<LittleEndian>(128).unwrap(); // n_mels (128 mel filters)
    buf.write_i32::<LittleEndian>(1).unwrap(); // raw_ftype

    let mut cursor = Cursor::new(buf);
    let hparams = Hyperparameters::read(&mut cursor).expect("read turbo header");
    assert_eq!(hparams.model_type, ModelType::LargeV3Turbo);
    assert_eq!(hparams.model_type.to_string(), "large-v3-turbo");
    assert_eq!(hparams.n_mels, 128);
    assert_eq!(hparams.n_text_layer, 4);
    assert_eq!(hparams.n_audio_layer, 32);
}
#[test]
fn test_load_tiny_multilingual() {
    let path = resolve_model_path("ggml-tiny.bin");
    if !path.exists() {
        eprintln!("Model file {:?} not found, skipping integration test", path);
        return;
    }

    let model = WhisperModel::load_from_file(path).expect("Failed to load ggml-tiny.bin");

    // Hyperparameters check
    assert_eq!(model.hparams.n_vocab, 51865);
    assert_eq!(model.hparams.n_audio_layer, 4);
    assert_eq!(model.hparams.n_audio_state, 384);
    assert_eq!(model.hparams.n_text_layer, 4);
    assert_eq!(model.hparams.model_type, ModelType::Tiny);
    assert!(model.hparams.is_multilingual());

    // Vocab check
    assert_eq!(model.vocab.n_vocab, 51865);
    assert_eq!(model.vocab.token_eot, 50257);
    assert_eq!(model.vocab.token_sot, 50258);
    assert_eq!(model.vocab.token_translate, 50358);
    assert_eq!(model.vocab.token_transcribe, 50359);
    assert_eq!(model.vocab.token_beg, 50364);

    // Tensors check
    assert_eq!(model.tensors.len(), 167);
    assert_eq!(model.total_weight_bytes, 77111040);
}
