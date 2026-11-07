//! Word-error-rate harness over the bundled samples.
//!
//! whisper.cpp has an open request to "start measuring the word error rate
//! (WER) across some representative dataset" (ggml-org/whisper.cpp#2454). The
//! reference transcripts in `tests/transcripts/whisper_cpp/` were produced by
//! upstream from the same audio, so comparing against them measures the port
//! rather than the model.
//!
//! These tests decode several minutes of audio each, so they are `#[ignore]`d.
//! Run them explicitly:
//!
//! ```text
//! cargo test -p whisper-rs --test test_wer_samples -- --ignored --nocapture
//! ```
//!
//! A sample is skipped when its audio or reference transcript is missing; the
//! printed table always states how many samples actually ran.

use std::path::{Path, PathBuf};
use whisper_rs::{WhisperContext, WhisperParams};

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

/// Lowercase alphanumeric tokens, which is how the archived transcripts are
/// compared elsewhere in this repository.
fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Levenshtein distance over word tokens, divided by the reference length.
fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r = normalize(reference);
    let h = normalize(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0usize; h.len() + 1];
    for (i, rw) in r.iter().enumerate() {
        cur[0] = i + 1;
        for (j, hw) in h.iter().enumerate() {
            let cost = usize::from(rw != hw);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()] as f64 / r.len() as f64
}

fn run_sample(model: &WhisperContext, stem: &str, params: &WhisperParams) -> Option<f64> {
    let wav = workspace_path(&format!("samples/{stem}.wav"));
    let reference_path = workspace_path(&format!("tests/transcripts/whisper_cpp/{stem}.txt"));
    if !wav.exists() || !reference_path.exists() {
        eprintln!("skip {stem}: missing audio or reference transcript");
        return None;
    }

    let reference = std::fs::read_to_string(&reference_path).ok()?;
    // Samples are not all 16 kHz mono; `transcribe_pcm` expects the decoder's
    // output rate, so resample exactly like `transcribe_file` does.
    let audio = whisper_rs::audio::load_audio_from_file(&wav).ok()?;
    let samples = if audio.sample_rate != 16_000 {
        whisper_rs::audio::resample_to_16k(&audio.samples, audio.sample_rate)
    } else {
        audio.samples.clone()
    };
    let segments = model.transcribe_pcm(&samples, params);
    let hypothesis: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let wer = word_error_rate(&reference, &hypothesis);
    eprintln!(
        "{stem:16} WER = {wer:.4}  ({} words)",
        normalize(&reference).len()
    );
    Some(wer)
}

#[test]
#[ignore = "decodes several minutes of audio; run explicitly"]
fn test_wer_against_whisper_cpp_transcripts() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    if !model_path.exists() {
        eprintln!("Missing ggml-tiny.en.bin, skipping WER harness");
        return;
    }
    let ctx = WhisperContext::load_model(&model_path).expect("load model");
    let params = WhisperParams {
        no_timestamps: true,
        ..Default::default()
    };

    let mut measured = 0usize;
    let mut worst = 0.0f64;
    for stem in ["jfk", "mm1", "diffusion_60s"] {
        if let Some(wer) = run_sample(&ctx, stem, &params) {
            measured += 1;
            worst = worst.max(wer);
        }
    }

    assert!(measured > 0, "no sample could be measured");
    eprintln!("measured {measured} sample(s), worst WER {worst:.4}");
    // The port has to stay in the same accuracy class as the reference
    // implementation it is compared against; 15% leaves room for the tiny
    // model's own variance on noisy samples without hiding a regression.
    assert!(
        worst <= 0.15,
        "worst WER {worst:.4} exceeded 0.15 across {measured} sample(s)"
    );
}
