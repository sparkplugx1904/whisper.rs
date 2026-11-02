use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::path::{Path, PathBuf};
use whisper_rs::audio::{compute_log_mel_spectrogram, WavAudio};
use whisper_rs::model::WhisperModel;
use whisper_rs::WhisperEncoder;

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
fn test_encoder_output_vs_reference() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let ref_path = workspace_path("tests/fixtures/jfk_encoder_output.bin");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!("Missing model or fixture, skipping encoder test");
        return;
    }

    let audio = WavAudio::load_from_file(&wav_path).expect("load wav");
    let model = WhisperModel::load_from_file(model_path).expect("load model");

    let mel = compute_log_mel_spectrogram(&audio.samples, &model.filters.data, model.filters.n_mel);
    let mel_chunk = mel.get_chunk(0, model.hparams.n_audio_ctx as usize);

    let encoder = WhisperEncoder::new(&model).expect("init encoder");
    let enc_out = encoder.forward(&mel_chunk);

    // Read reference fixture
    let mut ref_file = File::open(ref_path).expect("open ref");
    let ne0 = ref_file.read_u32::<LittleEndian>().unwrap() as usize;
    let ne1 = ref_file.read_u32::<LittleEndian>().unwrap() as usize;

    assert_eq!(ne0, model.hparams.n_audio_state as usize);
    assert_eq!(ne1, model.hparams.n_audio_ctx as usize);

    let total_elements = ne0 * ne1;
    let mut ref_data = Vec::with_capacity(total_elements);
    for _ in 0..total_elements {
        ref_data.push(ref_file.read_f32::<LittleEndian>().unwrap());
    }

    assert_eq!(enc_out.len(), ref_data.len());

    let mut dot_prod = 0.0f64;
    let mut norm_ours = 0.0f64;
    let mut norm_ref = 0.0f64;
    let mut max_diff = 0.0f32;

    for (&a, &b) in enc_out.iter().zip(&ref_data) {
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
        "Encoder verification: cosine_similarity = {:.6}, max_abs_diff = {:e}",
        cosine_sim, max_diff
    );

    // Criterion: cosine similarity >= 0.9999 or max diff <= 1e-3
    assert!(
        cosine_sim >= 0.9999 || max_diff <= 1e-3,
        "Encoder verification failed: cosine_sim = {}, max_diff = {:e}",
        cosine_sim,
        max_diff
    );
}
