use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::path::{Path, PathBuf};
use whisper_rs::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs::model::WhisperModel;
use whisper_rs::tokenizer::Tokenizer;
use whisper_rs::{DecoderKvCache, WhisperDecoder, WhisperEncoder};

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
fn test_decoder_first_logits_vs_reference() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let ref_path = workspace_path("tests/fixtures/jfk_first_logits.bin");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!("Missing model or fixture, skipping decoder logits test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");

    let mel = compute_log_mel_spectrogram(&audio.samples, &model.filters.data, model.filters.n_mel);
    let mel_chunk = mel.get_chunk(0, model.hparams.n_audio_ctx as usize);

    let encoder = WhisperEncoder::new(&model).expect("init encoder");
    let enc_out = encoder.forward(&mel_chunk);

    let decoder = WhisperDecoder::new(&model).expect("init decoder");
    let mut cache = DecoderKvCache::new(decoder.n_layers);

    let prompt = [model.vocab.token_sot];
    let logits = decoder.decode(&prompt, &enc_out, &mut cache);

    // Read reference fixture
    let mut ref_file = File::open(ref_path).expect("open ref");
    let n_vocab = ref_file.read_u32::<LittleEndian>().unwrap() as usize;
    assert_eq!(n_vocab, model.hparams.n_vocab as usize);

    let mut ref_logits = Vec::with_capacity(n_vocab);
    for _ in 0..n_vocab {
        ref_logits.push(ref_file.read_f32::<LittleEndian>().unwrap());
    }

    assert_eq!(logits.len(), ref_logits.len());

    let mut dot_prod = 0.0f64;
    let mut norm_ours = 0.0f64;
    let mut norm_ref = 0.0f64;
    let mut max_diff = 0.0f32;

    for (&a, &b) in logits.iter().zip(&ref_logits) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        dot_prod += (a as f64) * (b as f64);
        norm_ours += (a as f64) * (a as f64);
        norm_ref += (b as f64) * (b as f64);
    }

    let cosine_sim = dot_prod / (norm_ours.sqrt() * norm_ref.sqrt());
    eprintln!(
        "Decoder step 0 verification: cosine_similarity = {:.6}, max_abs_diff = {:e}",
        cosine_sim, max_diff
    );

    // Check top token
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > max_val {
            max_val = l;
            max_idx = i;
        }
    }

    let mut ref_max_idx = 0;
    let mut ref_max_val = f32::NEG_INFINITY;
    for (i, &l) in ref_logits.iter().enumerate() {
        if l > ref_max_val {
            ref_max_val = l;
            ref_max_idx = i;
        }
    }

    eprintln!(
        "Top token: ours = {} ({}), ref = {} ({})",
        max_idx,
        model.vocab.id_to_token(max_idx as i32).unwrap_or("?"),
        ref_max_idx,
        model.vocab.id_to_token(ref_max_idx as i32).unwrap_or("?")
    );

    assert_eq!(max_idx, ref_max_idx, "Top predicted token must match");
    assert!(cosine_sim >= 0.9999, "Cosine similarity must be >= 0.9999");
}

#[test]
fn test_greedy_transcription_jfk() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");

    if !model_path.exists() {
        eprintln!("Missing model, skipping greedy transcription test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let tokenizer = Tokenizer::new(model.vocab.clone());

    let mel = compute_log_mel_spectrogram(&audio.samples, &model.filters.data, model.filters.n_mel);
    let mel_chunk = mel.get_chunk(0, model.hparams.n_audio_ctx as usize);

    let encoder = WhisperEncoder::new(&model).expect("init encoder");
    let enc_out = encoder.forward(&mel_chunk);

    let decoder = WhisperDecoder::new(&model).expect("init decoder");
    let mut cache = DecoderKvCache::new(decoder.n_layers);

    let mut tokens = vec![tokenizer.sot()];
    let mut generated = Vec::new();

    for _ in 0..100 {
        let logits = decoder.decode(&tokens, &enc_out, &mut cache);
        assert_eq!(logits.len(), decoder.n_vocab);

        // Greedy argmax
        let mut best_id = 0;
        let mut best_logit = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > best_logit {
                best_logit = l;
                best_id = i as i32;
            }
        }

        if best_id == tokenizer.eot() {
            break;
        }

        generated.push(best_id);
        tokens = vec![best_id];
    }

    let text = tokenizer.decode(&generated, true);
    eprintln!("Decoded text: '{}'", text);

    assert!(
        text.contains("And so my fellow Americans")
            && text.contains("ask not what your country can do for you")
            && text.contains("ask what you can do for your country"),
        "Decoded text does not match expected JFK speech: '{}'",
        text
    );
}
