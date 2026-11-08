use std::path::{Path, PathBuf};
use whisper_rs::audio::WavAudio;
use whisper_rs::{Timings, WhisperContext, WhisperParams};

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
fn test_timing_parity_accumulates_and_resets() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping timing parity test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let ctx = WhisperContext::load_model(&model_path).expect("load model");
    let params = WhisperParams::default();

    let mut state = ctx.create_state();
    ctx.full(&mut state, &params, &audio.samples)
        .expect("full transcription");

    let timings = state.timings;
    assert!(timings.n_encode >= 1, "encoder must run at least once");
    assert!(timings.n_decode >= 1, "decoder must run at least once");
    assert!(timings.encode_us > 0, "encoder time must be recorded");
    assert!(timings.decode_us > 0, "decoder time must be recorded");
    assert!(timings.mel_us > 0, "mel time must be recorded");
    assert!(timings.sample_us > 0, "sampling time must be recorded");
    assert!(timings.n_sample >= 1);
    assert!(timings.n_prompt >= 1);
    assert!(timings.prompt_us > 0);
    assert!(
        timings.prompt_us <= timings.decode_us,
        "prompt time is the prefill subset of decoder time"
    );

    // whisper.cpp keeps timing counters across transcriptions; `clear` must
    // not touch them.
    state.clear();
    assert_eq!(state.timings, timings, "clear() must preserve timings");

    state.reset_timings();
    assert_eq!(
        state.timings,
        Timings::default(),
        "reset_timings() must zero every counter"
    );
}
