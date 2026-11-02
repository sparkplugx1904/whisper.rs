use std::path::{Path, PathBuf};
use whisper_rs::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs::model::WhisperModel;
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

fn word_error_rate(ref_text: &str, hyp_text: &str) -> f64 {
    let clean = |s: &str| {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
    };

    let ref_clean = clean(ref_text);
    let hyp_clean = clean(hyp_text);
    let ref_words: Vec<&str> = ref_clean.split_whitespace().collect();
    let hyp_words: Vec<&str> = hyp_clean.split_whitespace().collect();

    let n = ref_words.len();
    let m = hyp_words.len();
    if n == 0 {
        return if m == 0 { 0.0 } else { 1.0 };
    }

    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        row[0] = i;
    }
    for (j, item) in dp[0].iter_mut().enumerate() {
        *item = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if ref_words[i - 1] == hyp_words[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    (dp[n][m] as f64) / (n as f64)
}

#[test]
fn test_q8_0_transcription() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en-q8_0.bin");

    if !model_path.exists() {
        eprintln!("q8_0 model not found at {:?}, skipping test", model_path);
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load q8_0 model");
    assert_eq!(model.hparams.ftype, 7); // ggml_ftype MOSTLY_Q8_0 is 7

    let ctx = WhisperContext::new(model).expect("init context");
    let mel = compute_log_mel_spectrogram(
        &audio.samples,
        &ctx.model.filters.data,
        ctx.model.filters.n_mel,
    );

    let params = WhisperParams {
        no_timestamps: true,
        ..Default::default()
    };

    let segments = ctx.transcribe(&mel, &params);
    let full_text: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    eprintln!("Q8_0 text: '{}'", full_text);

    let ref_text = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country";
    let wer = word_error_rate(ref_text, &full_text);
    eprintln!("Q8_0 WER: {:.4}", wer);

    assert!(wer <= 0.05, "WER {} exceeded tolerance 0.05 for Q8_0", wer);
}

#[test]
fn test_q5_1_transcription() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en-q5_1.bin");

    if !model_path.exists() {
        eprintln!("q5_1 model not found at {:?}, skipping test", model_path);
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load q5_1 model");
    assert_eq!(model.hparams.ftype, 9); // ggml_ftype MOSTLY_Q5_1 is 9

    let ctx = WhisperContext::new(model).expect("init context");
    let mel = compute_log_mel_spectrogram(
        &audio.samples,
        &ctx.model.filters.data,
        ctx.model.filters.n_mel,
    );

    let params = WhisperParams {
        no_timestamps: true,
        ..Default::default()
    };

    let segments = ctx.transcribe(&mel, &params);
    let full_text: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    eprintln!("Q5_1 text: '{}'", full_text);

    let ref_text = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country";
    let wer = word_error_rate(ref_text, &full_text);
    eprintln!("Q5_1 WER: {:.4}", wer);

    assert!(wer <= 0.05, "WER {} exceeded tolerance 0.05 for Q5_1", wer);
}
