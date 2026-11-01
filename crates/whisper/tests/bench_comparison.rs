use std::path::{Path, PathBuf};
use std::time::Instant;
use whisper_rs_pure::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs_pure::model::WhisperModel;
use whisper_rs_pure::{WhisperContext, WhisperParams};

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
fn bench_whisper_pure_vs_cpp() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");

    if !model_path.exists() {
        eprintln!("Model not found, skipping benchmark");
        return;
    }

    let t0 = Instant::now();
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let ctx = WhisperContext::new(model).expect("init context");
    let load_time = t0.elapsed();

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let audio_duration = audio.samples.len() as f64 / 16000.0;

    let t_mel_start = Instant::now();
    let mel = compute_log_mel_spectrogram(
        &audio.samples,
        &ctx.model.filters.data,
        ctx.model.filters.n_mel,
    );
    let mel_time = t_mel_start.elapsed();

    let params = WhisperParams {
        no_timestamps: false,
        n_threads: 4,
        ..Default::default()
    };

    let t_transcribe_start = Instant::now();
    let segments = ctx.transcribe(&mel, &params);
    let transcribe_time = t_transcribe_start.elapsed();

    let total_time = mel_time + transcribe_time;
    let rtf = total_time.as_secs_f64() / audio_duration;

    eprintln!("\n=== BENCHMARK REPORT: whisper.rs (Pure Rust) ===");
    eprintln!("Audio duration:    {:.2} s", audio_duration);
    eprintln!(
        "Model load time:   {:.2} ms",
        load_time.as_secs_f64() * 1000.0
    );
    eprintln!(
        "Mel extraction:    {:.2} ms",
        mel_time.as_secs_f64() * 1000.0
    );
    eprintln!(
        "Transcription:     {:.2} ms",
        transcribe_time.as_secs_f64() * 1000.0
    );
    eprintln!(
        "Total processing:  {:.2} ms",
        total_time.as_secs_f64() * 1000.0
    );
    eprintln!(
        "Real-Time Factor:  {:.3}x (faster than real-time if < 1.0)",
        rtf
    );
    eprintln!("Output segments:   {}", segments.len());
    for seg in &segments {
        eprintln!("  [{:.2} --> {:.2}] {}", seg.t0, seg.t1, seg.text.trim());
    }
    eprintln!("================================================\n");

    assert!(!segments.is_empty());
    if !cfg!(debug_assertions) {
        assert!(
            rtf < 1.0,
            "Pure Rust engine must be faster than real-time (RTF < 1.0 in release mode)"
        );
    }
}
