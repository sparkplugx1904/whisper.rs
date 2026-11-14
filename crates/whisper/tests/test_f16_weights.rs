//! Checks that an f16 checkpoint keeps its projections packed in half
//! precision instead of being widened to `f32` at load time, and that the
//! packed path produces the same transcription as the expanded one.

use std::path::{Path, PathBuf};

use whisper_rs::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs::model::WhisperModel;
use whisper_rs::tensor::{DType, LinearWeight};
use whisper_rs::tokenizer::Tokenizer;
use whisper_rs::{DecoderKvCache, DecoderWorkspace, WhisperDecoder, WhisperEncoder};

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

/// Greedy decode of the first window of `jfk.wav`, returning the token ids.
fn greedy_tokens(decoder: &WhisperDecoder, enc: &[f32], tokenizer: &Tokenizer) -> Vec<i32> {
    let mut cache = DecoderKvCache::new(decoder.n_layers, 1, decoder.d_model);
    let mut workspace = DecoderWorkspace::new();

    let mut tokens = vec![tokenizer.sot()];
    let mut generated = Vec::new();
    for _ in 0..100 {
        let logits = decoder.decode(&tokens, enc, &mut cache, &mut workspace);
        let mut best_id = 0i32;
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
    generated
}

#[test]
fn test_f16_weights_stay_packed_and_match_expanded_path() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    if !model_path.exists() {
        eprintln!("Missing tiny.en model, skipping f16 weight test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");

    // A default ggml checkpoint stores the projection matrices as f16. The
    // loader must hand them over packed: `rows * cols` halves, not floats.
    let native = model
        .get_linear_weight("decoder.blocks.0.attn.query.weight")
        .expect("query weight");
    assert_eq!(native.dtype(), DType::F16);
    assert_eq!(native.resident_bytes(), native.rows() * native.cols() * 2);

    let encoder = WhisperEncoder::new(&model).expect("init encoder");
    for (l, block) in encoder.weights.blocks.iter().enumerate() {
        assert_eq!(
            block.attn_qkv_w.dtype(),
            DType::F16,
            "encoder block {l} qkv"
        );
        assert_eq!(
            block.attn_qkv_w.resident_bytes(),
            block.attn_qkv_w.rows() * block.attn_qkv_w.cols() * 2,
            "encoder block {l} qkv must stay packed"
        );
    }

    let decoder = WhisperDecoder::new(&model).expect("init decoder");
    assert_eq!(decoder.weights.token_embedding.dtype(), DType::F16);

    let mut packed = 0usize;
    for block in &encoder.weights.blocks {
        for w in [
            &block.attn_qkv_w,
            &block.attn_out_w,
            &block.mlp0_w,
            &block.mlp2_w,
        ] {
            packed += w.resident_bytes();
        }
    }
    for block in &decoder.weights.blocks {
        for w in [
            &block.attn_q_w,
            &block.attn_k_w,
            &block.attn_v_w,
            &block.attn_out_w,
            &block.cross_attn_q_w,
            &block.cross_attn_k_w,
            &block.cross_attn_v_w,
            &block.cross_attn_out_w,
            &block.mlp0_w,
            &block.mlp2_w,
        ] {
            packed += w.resident_bytes();
        }
    }
    packed += decoder.weights.token_embedding.resident_bytes();
    let rss_kib = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).map(str::to_string))
        })
        .unwrap_or_default();
    eprintln!(
        "packed matrix weights: {:.1} MiB (f32-expanded would be {:.1} MiB), process RSS {rss_kib} KiB",
        packed as f64 / (1024.0 * 1024.0),
        (packed * 2) as f64 / (1024.0 * 1024.0),
    );

    // Same decoder with the token embedding expanded to f32, i.e. the layout
    // the loader produced before f16 weights were kept packed.
    let mut expanded = WhisperDecoder {
        weights: decoder.weights.clone(),
        n_vocab: decoder.n_vocab,
        d_model: decoder.d_model,
        n_heads: decoder.n_heads,
        n_layers: decoder.n_layers,
        max_ctx: decoder.max_ctx,
    };
    let embedding = &decoder.weights.token_embedding;
    expanded.weights.token_embedding =
        LinearWeight::from_f32(embedding.rows(), embedding.cols(), embedding.to_f32());

    let tokenizer = Tokenizer::new(model.vocab.clone());
    let mel = compute_log_mel_spectrogram(&audio.samples, &model.filters.data, model.filters.n_mel);
    let mel_chunk = mel.get_chunk(0, model.hparams.n_audio_ctx as usize);
    let enc_out = encoder.forward(&mel_chunk);

    // First-step logits: the embedding lookup, the f16 GEMM/GEMV blocks and
    // the vocabulary projection must agree with the widened f32 weights.
    let mut cache_a = DecoderKvCache::new(decoder.n_layers, 1, decoder.d_model);
    let mut cache_b = DecoderKvCache::new(expanded.n_layers, 1, expanded.d_model);
    let mut ws_a = DecoderWorkspace::new();
    let mut ws_b = DecoderWorkspace::new();
    let prompt = [tokenizer.sot()];
    let logits_a = decoder.decode(&prompt, &enc_out, &mut cache_a, &mut ws_a);
    let logits_b = expanded.decode(&prompt, &enc_out, &mut cache_b, &mut ws_b);

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in logits_a.iter().zip(logits_b.iter()) {
        dot += (a as f64) * (b as f64);
        norm_a += (a as f64) * (a as f64);
        norm_b += (b as f64) * (b as f64);
    }
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    eprintln!("f16 vs f32-expanded step-0 cosine similarity: {cosine:.7}");
    assert!(cosine >= 0.999999, "f16 logits diverged: cosine {cosine}");

    let mut best_a = 0usize;
    let mut best_b = 0usize;
    for i in 1..logits_a.len() {
        if logits_a[i] > logits_a[best_a] {
            best_a = i;
        }
        if logits_b[i] > logits_b[best_b] {
            best_b = i;
        }
    }
    assert_eq!(
        best_a, best_b,
        "packed and expanded paths disagree on argmax"
    );

    let tokens_a = greedy_tokens(&decoder, &enc_out, &tokenizer);
    let tokens_b = greedy_tokens(&expanded, &enc_out, &tokenizer);
    let text_a = tokenizer.decode(&tokens_a, true);
    let text_b = tokenizer.decode(&tokens_b, true);
    eprintln!("packed f16 transcription: '{}'", text_a);
    assert_eq!(
        text_a, text_b,
        "f16-packed transcription differs from the f32-expanded one"
    );
    assert!(
        text_a.contains("And so my fellow Americans")
            && text_a.contains("ask not what your country can do for you")
            && text_a.contains("ask what you can do for your country"),
        "Decoded text does not match expected JFK speech: '{text_a}'"
    );
}
