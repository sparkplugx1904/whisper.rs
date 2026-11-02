use std::path::{Path, PathBuf};
use whisper_audio::{load_audio_from_bytes, load_audio_from_file, WHISPER_SAMPLE_RATE};

fn workspace_path(rel: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().unwrap().parent().unwrap();
    ws_root.join(rel)
}

#[test]
fn test_universal_decoder_jfk_wav() {
    let wav_path = workspace_path("samples/jfk.wav");
    assert!(wav_path.exists(), "samples/jfk.wav must exist");

    let audio = load_audio_from_file(&wav_path).expect("load_audio_from_file jfk.wav");
    assert_eq!(audio.sample_rate, WHISPER_SAMPLE_RATE);
    assert_eq!(audio.channels, 1);
    assert!(
        audio.samples.len() >= 170000,
        "JFK samples count must match ~11 seconds (got {})",
        audio.samples.len()
    );

    // Test in-memory bytes loading
    let bytes = std::fs::read(&wav_path).expect("read jfk.wav bytes");
    let audio_mem = load_audio_from_bytes(&bytes).expect("load_audio_from_bytes jfk.wav");
    assert_eq!(audio_mem.sample_rate, WHISPER_SAMPLE_RATE);
    assert_eq!(audio_mem.samples.len(), audio.samples.len());
}
