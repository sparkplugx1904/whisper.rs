use std::path::{Path, PathBuf};
use whisper_rs::audio::WavAudio;
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

fn jfk_samples() -> Option<Vec<f32>> {
    let wav = WavAudio::load_from_file(workspace_path("samples/jfk.wav")).ok()?;
    if wav.sample_rate != 16_000 {
        return None;
    }
    Some(wav.samples)
}

#[test]
fn test_vad_returns_no_segments_for_digital_silence() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    if !model_path.exists() {
        eprintln!("Missing model, skipping VAD silence test");
        return;
    }

    let ctx = WhisperContext::load_model(&model_path).expect("load model");
    let silence = vec![0.0f32; 16_000 * 20];

    let vad_params = WhisperParams {
        vad_enable: true,
        ..Default::default()
    };
    let segments = ctx.transcribe_pcm(&silence, &vad_params);
    assert!(
        segments.is_empty(),
        "VAD must skip silent audio instead of letting the decoder hallucinate {} segment(s)",
        segments.len()
    );
}

#[test]
fn test_vad_keeps_absolute_timestamps() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let Some(speech) = jfk_samples() else {
        eprintln!("Missing sample audio or model, skipping VAD timestamp test");
        return;
    };
    if !model_path.exists() {
        eprintln!("Missing model, skipping VAD timestamp test");
        return;
    }

    let ctx = WhisperContext::load_model(&model_path).expect("load model");

    // 40 s of silence, the 11 s JFK excerpt, then 30 s of silence. VAD has to
    // jump over the opening silence and still report the speech at ~40 s.
    let mut samples = vec![0.0f32; 16_000 * 40];
    samples.extend_from_slice(&speech);
    samples.extend_from_slice(&vec![0.0f32; 16_000 * 30]);

    let params = WhisperParams {
        vad_enable: true,
        ..Default::default()
    };
    let segments = ctx.transcribe_pcm(&samples, &params);
    assert!(
        !segments.is_empty(),
        "VAD must not drop the only speech region in the file"
    );

    let first = segments.first().unwrap();
    assert!(
        (39.0..=41.5).contains(&first.t0),
        "first segment starts at {:.2}s but speech starts at 40s; frame indices must stay absolute",
        first.t0
    );

    let text: String = segments.iter().map(|s| s.text.as_str()).collect();
    assert!(
        text.contains("fellow Americans"),
        "VAD path lost the transcript: '{text}'"
    );
}

#[test]
fn test_no_speech_gate_drops_flagged_windows() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    if !model_path.exists() {
        eprintln!("Missing model, skipping no-speech gate test");
        return;
    }

    let ctx = WhisperContext::load_model(&model_path).expect("load model");

    // Digital silence is where the decoder invents a word ("you" with
    // tiny.en). It does not trip the default gate because this checkpoint
    // reports a near-zero no-speech probability on synthetic silence, so the
    // thresholds are tightened here to cross both conditions and exercise the
    // rule: no_speech_prob > no_speech_threshold AND avg_logprob < logprob_threshold.
    let silence = vec![0.0f32; 16_000 * 20];
    let gated = WhisperParams {
        no_timestamps: true,
        no_speech_threshold: 1e-4,
        logprob_threshold: -0.1,
        ..Default::default()
    };
    let segments = ctx.transcribe_pcm(&silence, &gated);
    let text: String = segments.iter().map(|s| s.text.as_str()).collect();
    assert!(
        text.trim().is_empty(),
        "window flagged as non-speech must be dropped, got '{text}'"
    );

    // Same audio, gate disabled: the phantom tokens come back, which is what
    // the gate exists to stop.
    let ungated = WhisperParams {
        no_speech_threshold: 0.0,
        ..gated
    };
    let ungated_segments = ctx.transcribe_pcm(&silence, &ungated);
    assert!(
        !ungated_segments.is_empty(),
        "with the gate disabled the silence should still be decoded"
    );

    // Ordinary speech must survive the default thresholds.
    let speech = jfk_samples().expect("jfk sample");
    let speech_params = WhisperParams {
        no_timestamps: true,
        ..Default::default()
    };
    let speech_segments = ctx.transcribe_pcm(&speech, &speech_params);
    let speech_text: String = speech_segments.iter().map(|s| s.text.as_str()).collect();
    assert!(
        speech_text.contains("fellow Americans"),
        "speech must not be classified as silence: '{speech_text}'"
    );
}
