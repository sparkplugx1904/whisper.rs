use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use whisper::*;

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
fn test_c_abi_end_to_end() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");

    if !model_path.exists() {
        eprintln!("Model not found, skipping C-ABI test");
        return;
    }

    let audio = whisper_rs_pure::audio::WavAudio::load_from_file(&wav_path).expect("load wav");

    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(
            !ctx.is_null(),
            "whisper_init_from_file must return non-null context"
        );

        let n_vocab = whisper_n_vocab(ctx);
        assert_eq!(n_vocab, 51864);

        let eot = whisper_token_eot(ctx);
        assert_eq!(eot, 50256);

        let params =
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

        let ret = whisper_full(
            ctx,
            params,
            audio.samples.as_ptr(),
            audio.samples.len() as i32,
        );
        assert_eq!(ret, 0, "whisper_full must return 0 on success");

        let n_segments = whisper_full_n_segments(ctx);
        assert!(n_segments >= 1, "Must have at least 1 segment");

        let seg_text_ptr = whisper_full_get_segment_text(ctx, 0);
        assert!(!seg_text_ptr.is_null());
        let text = CStr::from_ptr(seg_text_ptr).to_str().unwrap();
        eprintln!("C-ABI Transcribed Text: '{}'", text);

        assert!(
            text.contains("And so my fellow Americans")
                && text.contains("ask what you can do for your country"),
            "C-ABI text mismatch: '{}'",
            text
        );

        let t0 = whisper_full_get_segment_t0(ctx, 0);
        let t1 = whisper_full_get_segment_t1(ctx, 0);
        eprintln!("C-ABI Timestamps (centiseconds): [{} --> {}]", t0, t1);
        assert_eq!(t0, 0);
        assert!(t1 > 900);

        whisper_free(ctx);
    }
}
