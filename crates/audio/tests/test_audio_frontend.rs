use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;

use std::path::Path;
use whisper_audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_model::WhisperModel;
fn workspace_path(rel: &str) -> std::path::PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().unwrap().parent().unwrap();
    ws_root.join(rel)
}

#[test]
fn test_wav_loader() {
    let wav_path = workspace_path("samples/jfk.wav");
    assert!(
        wav_path.exists(),
        "samples/jfk.wav must exist at {:?}",
        wav_path
    );
    let audio = WavAudio::load_from_file(&wav_path).expect("Failed to load jfk.wav");
    assert_eq!(audio.sample_rate, 16000);
    assert_eq!(audio.samples.len(), 176000);

    // Audio values should be in range [-1.0, 1.0]
    for &s in &audio.samples {
        assert!((-1.05..=1.05).contains(&s));
    }
}

#[test]
fn test_mel_spectrogram_vs_reference() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = Path::new("/tmp/models/ggml-tiny.en.bin");
    let ref_path = workspace_path("tests/fixtures/jfk_mel_reference.bin");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!("Missing model or fixture, skipping mel test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");
    let mel = compute_log_mel_spectrogram(&audio.samples, &model.filters.data, model.filters.n_mel);

    // Read reference fixture
    let mut ref_file = File::open(ref_path).expect("open ref");
    let ref_n_mel = ref_file.read_u32::<LittleEndian>().unwrap() as usize;
    let ref_n_len = ref_file.read_u32::<LittleEndian>().unwrap() as usize;

    assert_eq!(mel.n_mel, ref_n_mel);
    assert_eq!(mel.n_len, ref_n_len);

    let total_elements = ref_n_mel * ref_n_len;
    let mut ref_data = Vec::with_capacity(total_elements);
    for _ in 0..total_elements {
        ref_data.push(ref_file.read_f32::<LittleEndian>().unwrap());
    }

    assert_eq!(mel.data.len(), ref_data.len());

    let mut max_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;

    for (&m, &r) in mel.data.iter().zip(&ref_data) {
        let diff = (m - r).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_sq_diff += (diff as f64) * (diff as f64);
    }

    let rmse = (sum_sq_diff / total_elements as f64).sqrt();
    eprintln!(
        "Mel spectrogram verification: max_abs_diff = {:e}, RMSE = {:e}",
        max_diff, rmse
    );
    let mut count = 0;
    for t in 0..mel.n_len {
        for m in 0..mel.n_mel {
            let idx = m * mel.n_len + t;
            let diff = (mel.data[idx] - ref_data[idx]).abs();
            if diff > 1e-4 && count < 10 {
                eprintln!(
                    "Diff at frame {} mel {}: ours={}, ref={}, diff={:e}",
                    t, m, mel.data[idx], ref_data[idx], diff
                );
                count += 1;
            }
        }
    }

    assert!(
        max_diff <= 1e-4,
        "Max difference {:e} exceeded tolerance 1e-4",
        max_diff
    );
}
