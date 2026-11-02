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

    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();

    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null());

        let params =
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        let ret = whisper_full(
            ctx,
            params,
            audio.samples.as_ptr(),
            audio.samples.len() as i32,
        );
        assert_eq!(ret, 0);

        let n_segments = whisper_full_n_segments(ctx);
        assert!(n_segments >= 1);

        let seg_text_ptr = whisper_full_get_segment_text(ctx, 0);
        let text = CStr::from_ptr(seg_text_ptr).to_str().unwrap();
        assert!(text.contains("And so my fellow Americans"));

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_tokenize_bpe() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping C-ABI tokenize test");
        return;
    }

    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null());

        let text = CString::new("Hello, world!").unwrap();
        let mut tokens = vec![0i32; 16];
        let n_tokens =
            whisper_tokenize(ctx, text.as_ptr(), tokens.as_mut_ptr(), tokens.len() as i32);
        assert!(
            n_tokens > 0,
            "Tokenize must return positive number of tokens"
        );
        eprintln!("C-ABI tokenized 'Hello, world!' into {} tokens", n_tokens);

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_step_by_step_pipeline() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() {
        eprintln!("Model not found, skipping step-by-step test");
        return;
    }

    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();

    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null());

        // 1. PCM to Mel
        let ret_mel =
            whisper_pcm_to_mel(ctx, audio.samples.as_ptr(), audio.samples.len() as i32, 1);
        assert_eq!(ret_mel, 0, "whisper_pcm_to_mel must return 0");

        // 2. Language Auto Detect with probabilities array
        let max_langs = whisper_lang_max_id() as usize + 1;
        let mut lang_probs = vec![0.0f32; max_langs];
        let lang_id =
            whisper_lang_auto_detect(ctx, std::ptr::null_mut(), 0, 1, lang_probs.as_mut_ptr());
        assert!(
            lang_id >= 0,
            "Language detection must return non-negative ID"
        );
        eprintln!(
            "Detected lang ID: {}, prob en: {:.4}",
            lang_id, lang_probs[0]
        );
        assert!(lang_probs[0] > 0.5, "English prob must be > 0.5");

        // 3. Encode
        let ret_enc = whisper_encode(ctx, 0, 1);
        assert_eq!(ret_enc, 0, "whisper_encode must return 0");

        // 4. Decode
        let prompt = [whisper_token_sot(ctx)];
        let ret_dec = whisper_decode(ctx, prompt.as_ptr(), 1, 0, 1);
        assert_eq!(ret_dec, 0, "whisper_decode must return 0");

        // 5. Get Logits
        let logits_ptr = whisper_get_logits(ctx);
        assert!(!logits_ptr.is_null(), "Logits pointer must not be null");
        let n_vocab = whisper_n_vocab(ctx) as usize;
        let logits_slice = std::slice::from_raw_parts(logits_ptr, n_vocab);
        assert!(
            logits_slice.iter().any(|&l| l != 0.0),
            "Logits must not be all zero"
        );

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_logits_null_when_uninitialized() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping test", model_path);
        return;
    }
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null(), "whisper_init_from_file failed");

        let logits_ptr = whisper_get_logits(ctx);
        assert_eq!(
            logits_ptr,
            std::ptr::null_mut(),
            "whisper_get_logits before decode must return null pointer"
        );

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_set_mel_integer_overflow() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping test", model_path);
        return;
    }
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null(), "whisper_init_from_file failed");

        let dummy_data = [0.0f32; 10];
        let ret = whisper_set_mel(ctx, dummy_data.as_ptr(), i32::MAX, 10);
        assert_eq!(
            ret, -1,
            "whisper_set_mel must return -1 on integer overflow"
        );

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_decode_invalid_token() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping test", model_path);
        return;
    }
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null(), "whisper_init_from_file failed");

        let invalid_token_neg = [-1];
        let ret_neg = whisper_decode(ctx, invalid_token_neg.as_ptr(), 1, 0, 1);
        assert_eq!(
            ret_neg, -1,
            "whisper_decode must return -1 for negative token ID"
        );

        let invalid_token_high = [999999];
        let ret_high = whisper_decode(ctx, invalid_token_high.as_ptr(), 1, 0, 1);
        assert_eq!(
            ret_high, -1,
            "whisper_decode must return -1 for out-of-vocab token ID"
        );

        whisper_free(ctx);
    }
}
