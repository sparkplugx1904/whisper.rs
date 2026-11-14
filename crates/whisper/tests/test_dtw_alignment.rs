//! Real cross-attention DTW word timestamps.
//!
//! The transcriptions here are intentionally short (`samples/jfk.wav`) and use
//! `ggml-tiny.en.bin`, matching the repository's fast end-to-end test model.

use std::path::{Path, PathBuf};
use whisper_rs::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs::model::WhisperModel;
use whisper_rs::{
    get_alignment_heads, AlignmentHeadsPreset, DecoderKvCache, DecoderWorkspace, WhisperContext,
    WhisperDecoder, WhisperEncoder, WhisperParams,
};

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
fn dtw_capture_is_cross_attention_softmax() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Missing tiny model or sample, skipping capture test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(&model_path).expect("load model");
    let mel = compute_log_mel_spectrogram(&audio.samples, &model.filters.data, model.filters.n_mel);
    let enc = WhisperEncoder::new(&model)
        .expect("encoder")
        .forward(&mel.get_chunk(0, model.hparams.n_audio_ctx as usize));

    let decoder = WhisperDecoder::new(&model).expect("decoder");
    let heads = get_alignment_heads(AlignmentHeadsPreset::TinyEn);
    assert!(!heads.is_empty());

    let tokens = [
        model.vocab.token_sot,
        model.vocab.token_not,
        220,
        11,
        model.vocab.token_eot,
    ];
    let mut cache = DecoderKvCache::new(decoder.n_layers, 1, decoder.d_model);
    let mut ws = DecoderWorkspace::new();
    let _ = decoder.decode_with_alignment_heads(&tokens, &enc, &mut cache, &mut ws, heads);

    let n_audio = enc.len() / decoder.d_model;
    let attention = ws.alignment_attention();
    assert_eq!(attention.len(), tokens.len() * heads.len() * n_audio);
    assert!(attention.iter().all(|v| v.is_finite() && *v >= 0.0));

    // Every captured (token, head) row is a softmax distribution over audio.
    let stride = heads.len() * n_audio;
    for row in attention.chunks_exact(stride) {
        for head in row.chunks_exact(n_audio) {
            let sum: f32 = head.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "attention row is not normalized: {sum}"
            );
        }
    }
}

#[test]
fn word_timestamps_are_monotonic_and_inside_the_segment() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Missing tiny model or sample, skipping word timestamp test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let ctx = WhisperContext::load_model(&model_path).expect("load model");
    let params = WhisperParams {
        dtw_aheads_preset: AlignmentHeadsPreset::TinyEn,
        ..Default::default()
    };

    let segments = ctx.transcribe_pcm(&audio.samples, &params);
    assert!(!segments.is_empty(), "no segments produced");

    let mut total_words = 0usize;
    for segment in &segments {
        assert!(
            !segment.words.is_empty(),
            "segment without word timings: {:?}",
            segment.text
        );

        let mut prev_t1 = segment.t0;
        for word in &segment.words {
            assert!(!word.word.trim().is_empty());
            assert!(
                word.t0 >= segment.t0 - 1e-9 && word.t1 <= segment.t1 + 1e-9,
                "word {:?} [{}, {}] outside segment [{}, {}]",
                word.word,
                word.t0,
                word.t1,
                segment.t0,
                segment.t1
            );
            assert!(
                word.t0 <= word.t1 + 1e-9,
                "word {:?} has negative duration",
                word.word
            );
            assert!(
                word.t0 + 1e-9 >= prev_t1,
                "word start {:?} moved before the previous word ended",
                word.word
            );
            assert!(
                word.t1 + 1e-9 >= prev_t1,
                "word end {:?} moved backwards",
                word.word
            );
            prev_t1 = word.t1;
            total_words += 1;
        }
    }
    assert!(total_words > 0, "segments contained no words");

    for segment in &segments {
        eprintln!(
            "segment [{:.2}, {:.2}] {:?}",
            segment.t0, segment.t1, segment.text
        );
        for word in &segment.words {
            eprintln!("  word [{:.2}, {:.2}] {:?}", word.t0, word.t1, word.word);
        }
    }
}

#[test]
fn approximation_is_explicit_and_segment_bounded() {
    let tokens = [(" Hello", 0.9), (" world", 0.9)];
    let words = whisper_rs::dtw::approximate_word_timings(&tokens, 2.0, 3.0, 0.02);
    assert_eq!(words.len(), 2);
    for word in &words {
        assert!(word.t0 >= 2.0 && word.t1 <= 3.0);
        assert!(word.t0 <= word.t1);
    }
}
