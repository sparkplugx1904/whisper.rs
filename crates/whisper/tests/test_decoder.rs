use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::path::{Path, PathBuf};
use whisper_rs::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs::model::WhisperModel;
use whisper_rs::tokenizer::Tokenizer;
use whisper_rs::{DecodeSlot, DecoderKvCache, DecoderWorkspace, WhisperDecoder, WhisperEncoder};

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
    let mut cache = DecoderKvCache::new(decoder.n_layers, 1, decoder.d_model);
    let mut workspace = DecoderWorkspace::new();

    let prompt = [model.vocab.token_sot];
    let logits = decoder.decode(&prompt, &enc_out, &mut cache, &mut workspace);

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
    let mut cache = DecoderKvCache::new(decoder.n_layers, 1, decoder.d_model);
    let mut workspace = DecoderWorkspace::new();

    let mut tokens = vec![tokenizer.sot()];
    let mut generated = Vec::new();

    for _ in 0..100 {
        let logits = decoder.decode(&tokens, &enc_out, &mut cache, &mut workspace);
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

#[test]
fn test_batched_decode_matches_separate_calls() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");

    if !model_path.exists() {
        eprintln!("Missing model, skipping batched decode test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let mel = compute_log_mel_spectrogram(&audio.samples, &model.filters.data, model.filters.n_mel);
    let mel_chunk = mel.get_chunk(0, model.hparams.n_audio_ctx as usize);

    let encoder = WhisperEncoder::new(&model).expect("init encoder");
    let enc_out = encoder.forward(&mel_chunk);
    let decoder = WhisperDecoder::new(&model).expect("init decoder");
    let n_vocab = decoder.n_vocab;
    let d_model = decoder.d_model;

    let prompt_a = [model.vocab.token_prev, model.vocab.token_sot];
    let prompt_b = [model.vocab.token_prev];

    // Reference: each prompt decodes in its own cache, one row per token.
    let mut ref_a = Vec::new();
    {
        let mut cache = DecoderKvCache::new(decoder.n_layers, 1, d_model);
        let mut ws = DecoderWorkspace::new();
        ref_a.extend_from_slice(decoder.decode(&prompt_a, &enc_out, &mut cache, &mut ws));
    }
    let mut ref_b = Vec::new();
    {
        let mut cache = DecoderKvCache::new(decoder.n_layers, 1, d_model);
        let mut ws = DecoderWorkspace::new();
        ref_b.extend_from_slice(decoder.decode(&prompt_b, &enc_out, &mut cache, &mut ws));
    }

    // Batched: both prompts share one forward pass over the weight matrices.
    let mut cache = DecoderKvCache::new(decoder.n_layers, 2, d_model);
    let mut ws = DecoderWorkspace::new();
    let batched = decoder.decode_batch(
        &[
            DecodeSlot {
                slot: 0,
                tokens: &prompt_a,
            },
            DecodeSlot {
                slot: 1,
                tokens: &prompt_b,
            },
        ],
        &enc_out,
        &mut cache,
        &mut ws,
    );
    assert_eq!(batched.len(), (prompt_a.len() + prompt_b.len()) * n_vocab);
    assert_eq!(cache.t_past, vec![prompt_a.len(), prompt_b.len()]);

    let rows_a = &batched[..prompt_a.len() * n_vocab];
    let rows_b = &batched[prompt_a.len() * n_vocab..];

    // Rows from different sequences are independent, so the only expected
    // difference between the batched GEMM and the single-row GEMV path is
    // floating-point accumulation order.
    let max_diff = |got: &[f32], expected: &[f32]| {
        got.iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let diff_a = max_diff(rows_a, &ref_a);
    let diff_b = max_diff(rows_b, &ref_b);
    assert!(
        diff_a < 1e-3,
        "batched logits for sequence A differ from the sequential decode by {diff_a}"
    );
    assert!(
        diff_b < 1e-3,
        "batched logits for sequence B differ from the sequential decode by {diff_b}"
    );

    // Forking a slot must reproduce the source history exactly; beam search
    // relies on this instead of cloning a whole cache.
    cache.ensure_slots(3);
    cache.copy_slot(2, 0);
    assert_eq!(cache.slot_len(2), prompt_a.len());
    for (layer_idx, layer) in cache.k_self.iter().enumerate() {
        assert_eq!(layer[0], layer[2], "layer {layer_idx} key copy diverged");
    }
    for (layer_idx, layer) in cache.v_self.iter().enumerate() {
        assert_eq!(layer[0], layer[2], "layer {layer_idx} value copy diverged");
    }
}
