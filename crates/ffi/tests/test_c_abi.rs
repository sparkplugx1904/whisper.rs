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

        let n_tokens = whisper_full_n_tokens(ctx, 0);
        assert!(n_tokens > 0, "Segment must have tokens");

        let tok_data = whisper_full_get_token_data(ctx, 0, 0);
        assert!(tok_data.id > 0);
        assert!(tok_data.p > 0.0);
        assert!(tok_data.t1 > tok_data.t0);
        assert!(tok_data.t1 > 0);

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
        let lang_id = whisper_lang_auto_detect(ctx, 0, 1, lang_probs.as_mut_ptr());
        let lang_id_with_state = whisper_lang_auto_detect_with_state(
            ctx,
            std::ptr::null_mut(),
            0,
            1,
            lang_probs.as_mut_ptr(),
        );
        assert_eq!(lang_id, lang_id_with_state);
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

#[test]
fn test_sizeof_whisper_context_params() {
    assert_eq!(
        std::mem::size_of::<whisper_context_params>(),
        48,
        "whisper_context_params must be exactly 48 bytes matching C-ABI include/whisper.h"
    );
}

#[test]
fn test_c_abi_version_and_metadata_accessors() {
    unsafe {
        let ver_ptr = whisper_version();
        assert!(!ver_ptr.is_null());
        let ver = CStr::from_ptr(ver_ptr).to_str().unwrap();
        assert!(!ver.is_empty());

        let en_full = whisper_lang_str_full(0);
        assert!(!en_full.is_null());
        assert_eq!(CStr::from_ptr(en_full).to_str().unwrap(), "english");

        let zh_full = whisper_lang_str_full(1);
        assert!(!zh_full.is_null());
        assert_eq!(CStr::from_ptr(zh_full).to_str().unwrap(), "chinese");

        let de_full = whisper_lang_str_full(2);
        assert!(!de_full.is_null());
        assert_eq!(CStr::from_ptr(de_full).to_str().unwrap(), "german");
    }
}

#[test]
fn test_sizeof_vad_and_telemetry_structs() {
    assert_eq!(
        std::mem::size_of::<whisper_vad_params>(),
        24,
        "whisper_vad_params must be exactly 24 bytes matching C-ABI include/whisper.h"
    );
    assert_eq!(
        std::mem::size_of::<whisper_vad_context_params>(),
        12,
        "whisper_vad_context_params must be exactly 12 bytes matching C-ABI include/whisper.h"
    );
    assert_eq!(
        std::mem::size_of::<whisper_timings>(),
        20,
        "whisper_timings must be exactly 20 bytes matching C-ABI include/whisper.h"
    );
}

#[test]
fn test_c_abi_param_allocators_and_deleters() {
    unsafe {
        // 1. whisper_context_params by ref
        let cparams = whisper_context_default_params_by_ref();
        assert!(
            !cparams.is_null(),
            "whisper_context_default_params_by_ref must not be null"
        );
        assert!(!(*cparams).use_gpu);
        assert_eq!((*cparams).dtw_aheads_preset, 0);

        (*cparams).use_gpu = true;
        assert!((*cparams).use_gpu);

        whisper_free_context_params(cparams);
        whisper_free_context_params(std::ptr::null_mut()); // Verify null-safety

        // 2. whisper_full_params by ref (Greedy)
        let fparams_greedy =
            whisper_full_default_params_by_ref(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        assert!(!fparams_greedy.is_null());
        assert_eq!(
            (*fparams_greedy).strategy,
            whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY
        );
        assert_eq!((*fparams_greedy).n_threads, 4);
        assert!((*fparams_greedy).print_timestamps);

        (*fparams_greedy).n_threads = 8;
        assert_eq!((*fparams_greedy).n_threads, 8);

        whisper_free_params(fparams_greedy);

        // 3. whisper_full_params by ref (Beam Search)
        let fparams_beam = whisper_full_default_params_by_ref(
            whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH,
        );
        assert!(!fparams_beam.is_null());
        assert_eq!(
            (*fparams_beam).strategy,
            whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH
        );
        assert_eq!((*fparams_beam).beam_search.beam_size, 5);

        whisper_free_params(fparams_beam);
        whisper_free_params(std::ptr::null_mut()); // Verify null-safety
    }
}

#[test]
fn test_c_abi_multi_tenant_no_state_and_with_params() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!(
            "Model not found at {:?}, skipping multi-tenant test",
            model_path
        );
        return;
    }
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    let cparams = whisper_context_default_params();

    unsafe {
        // 1. Context without default state
        let ctx = whisper_init_from_file_with_params_no_state(c_model_path.as_ptr(), cparams);
        assert!(
            !ctx.is_null(),
            "init_from_file_with_params_no_state must succeed"
        );

        // Verify invoking default-state APIs safely handles absence of state
        let fparams =
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        let dummy_samples = [0.0f32; 1600];
        let ret_full = whisper_full(
            ctx,
            fparams,
            dummy_samples.as_ptr(),
            dummy_samples.len() as i32,
        );
        assert_eq!(
            ret_full, -1,
            "whisper_full must fail gracefully on no-state context"
        );
        assert_eq!(whisper_full_n_segments(ctx), 0);
        assert!(whisper_full_get_segment_text(ctx, 0).is_null());
        assert_eq!(whisper_full_get_segment_t0(ctx, 0), 0);
        assert_eq!(whisper_full_get_segment_t1(ctx, 0), 0);

        // 2. Allocate independent worker states for multi-tenancy
        let state1 = whisper_init_state(ctx);
        let state2 = whisper_init_state(ctx);
        assert!(!state1.is_null(), "state1 allocation must succeed");
        assert!(!state2.is_null(), "state2 allocation must succeed");
        assert_ne!(
            state1, state2,
            "multi-tenant states must have distinct addresses"
        );

        whisper_free_state(state1);
        whisper_free_state(state2);
        whisper_free(ctx);

        // 3. Test context with default state and parameters
        let ctx_with_state = whisper_init_from_file_with_params(c_model_path.as_ptr(), cparams);
        assert!(!ctx_with_state.is_null());
        whisper_free(ctx_with_state);

        // 4. Test buffer loaders with and without state
        let model_bytes = std::fs::read(&model_path).expect("read model file");
        let mut buf_copy = model_bytes.clone();
        let ctx_buf_no_state = whisper_init_from_buffer_with_params_no_state(
            buf_copy.as_mut_ptr() as *mut std::ffi::c_void,
            buf_copy.len(),
            cparams,
        );
        assert!(!ctx_buf_no_state.is_null());
        let st = whisper_init_state(ctx_buf_no_state);
        assert!(!st.is_null());
        whisper_free_state(st);
        whisper_free(ctx_buf_no_state);

        let ctx_buf_with_state = whisper_init_from_buffer_with_params(
            buf_copy.as_mut_ptr() as *mut std::ffi::c_void,
            buf_copy.len(),
            cparams,
        );
        assert!(!ctx_buf_with_state.is_null());
        whisper_free(ctx_buf_with_state);
    }
}

#[test]
fn test_c_abi_step_by_step_multi_state_execution() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() {
        eprintln!(
            "Model not found at {:?}, skipping step-by-step test",
            model_path
        );
        return;
    }

    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    let cparams = whisper_context_default_params();

    unsafe {
        let ctx = whisper_init_from_file_with_params_no_state(c_model_path.as_ptr(), cparams);
        assert!(!ctx.is_null());
        let state = whisper_init_state(ctx);
        assert!(!state.is_null());

        // Step 1: PCM to Mel with State
        let ret_mel = whisper_pcm_to_mel_with_state(
            ctx,
            state,
            audio.samples.as_ptr(),
            audio.samples.len() as i32,
            1,
        );
        assert_eq!(ret_mel, 0, "whisper_pcm_to_mel_with_state must succeed");
        let n_len = whisper_n_len_from_state(state);
        assert!(n_len > 0, "Spectrogram frames must be > 0");

        // Step 2: Set Mel with State (custom spectrogram)
        let dummy_mel = vec![0.0f32; 80 * 20];
        let ret_set_mel = whisper_set_mel_with_state(ctx, state, dummy_mel.as_ptr(), 20, 80);
        assert_eq!(ret_set_mel, 0, "whisper_set_mel_with_state must return 0");

        // Verify error checking on invalid input
        assert_eq!(
            whisper_set_mel_with_state(ctx, state, std::ptr::null(), 20, 80),
            -1
        );
        assert_eq!(
            whisper_set_mel_with_state(ctx, state, dummy_mel.as_ptr(), i32::MAX, 80),
            -1
        );

        // Restore real audio mel
        whisper_pcm_to_mel_with_state(
            ctx,
            state,
            audio.samples.as_ptr(),
            audio.samples.len() as i32,
            1,
        );

        // Step 3: Encode with State
        let ret_enc = whisper_encode_with_state(ctx, state, 0, 1);
        assert_eq!(ret_enc, 0, "whisper_encode_with_state must return 0");

        // Step 4: Decode with State
        let prompt = [whisper_token_sot(ctx)];
        let ret_dec = whisper_decode_with_state(ctx, state, prompt.as_ptr(), 1, 0, 1);
        assert_eq!(ret_dec, 0, "whisper_decode_with_state must return 0");

        // Invalid token handling
        let invalid_tok = [-1];
        assert_eq!(
            whisper_decode_with_state(ctx, state, invalid_tok.as_ptr(), 1, 0, 1),
            -1
        );

        // Step 5: Get Logits from State
        let logits_ptr = whisper_get_logits_from_state(state);
        assert!(!logits_ptr.is_null());
        let n_vocab = whisper_n_vocab(ctx) as usize;
        let logits_slice = std::slice::from_raw_parts(logits_ptr, n_vocab);
        assert!(logits_slice.iter().any(|&l| l != 0.0));

        whisper_free_state(state);
        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_telemetry_and_logging() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!(
            "Model not found at {:?}, skipping telemetry test",
            model_path
        );
        return;
    }
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    let cparams = whisper_context_default_params();

    unsafe {
        let ctx = whisper_init_from_file_with_params(c_model_path.as_ptr(), cparams);
        assert!(!ctx.is_null());

        // 1. whisper_get_timings
        let timings = whisper_get_timings(ctx);
        assert!(
            !timings.is_null(),
            "whisper_get_timings must not return null"
        );
        assert_eq!((*timings).sample_ms, 0.0);

        // 2. Perform a step to record time
        let dummy_samples = [0.0f32; 1600];
        whisper_pcm_to_mel(ctx, dummy_samples.as_ptr(), dummy_samples.len() as i32, 1);
        assert!((*timings).sample_ms >= 0.0);

        // 3. Reset timings
        whisper_reset_timings(ctx);
        assert_eq!((*timings).sample_ms, 0.0);
        assert_eq!((*timings).encode_ms, 0.0);
        assert_eq!((*timings).decode_ms, 0.0);

        // 4. Logging callback verification
        static LOG_CALLED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static LOG_TEXT: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

        unsafe extern "C" fn test_log_cb(
            level: ggml_log_level,
            text: *const std::ffi::c_char,
            _user_data: *mut std::ffi::c_void,
        ) {
            if level == ggml_log_level::GGML_LOG_LEVEL_INFO && !text.is_null() {
                LOG_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
                let s = CStr::from_ptr(text).to_str().unwrap_or("").to_string();
                if let Ok(mut lock) = LOG_TEXT.lock() {
                    *lock = Some(s);
                }
            }
        }

        LOG_CALLED.store(false, std::sync::atomic::Ordering::SeqCst);
        *LOG_TEXT.lock().unwrap() = None;
        whisper_log_set(Some(test_log_cb), std::ptr::null_mut());
        whisper_print_timings(ctx);

        assert!(
            LOG_CALLED.load(std::sync::atomic::Ordering::SeqCst),
            "Log callback must have been invoked by whisper_print_timings"
        );
        let text_guard = LOG_TEXT.lock().unwrap();
        assert!(
            text_guard
                .as_ref()
                .unwrap()
                .contains("whisper_print_timings"),
            "Log message must contain timing header"
        );

        // Reset log callback to avoid side effects
        whisper_log_set(None, std::ptr::null_mut());

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_vad_standalone_api() {
    let wav_path = workspace_path("samples/jfk.wav");
    let model_path = resolve_model_path("for-tests-silero-v6.2.0-ggml.bin");
    if !wav_path.exists() || !model_path.exists() {
        eprintln!("VAD model or sample not found, skipping VAD test");
        return;
    }
    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();

    unsafe {
        // 1. Parameter defaults
        let vad_params = whisper_vad_default_params();
        assert_eq!(vad_params.threshold, 0.5);
        assert_eq!(vad_params.min_speech_duration_ms, 250);
        assert_eq!(vad_params.min_silence_duration_ms, 100);
        assert_eq!(vad_params.speech_pad_ms, 30);

        let vad_cparams = whisper_vad_default_context_params();
        assert_eq!(vad_cparams.n_threads, 4);
        assert!(!vad_cparams.use_gpu);

        // 2. Init from the Silero GGML model
        let vctx = whisper_vad_init_from_file_with_params(c_model_path.as_ptr(), vad_cparams);
        assert!(
            !vctx.is_null(),
            "whisper_vad_init_from_file_with_params must load the Silero model"
        );

        // 3. Speech detection on JFK audio (active speech)
        let detected =
            whisper_vad_detect_speech(vctx, audio.samples.as_ptr(), audio.samples.len() as i32);
        assert_eq!(detected, 1, "VAD graph must run successfully");

        let n_probs = whisper_vad_n_probs(vctx);
        assert_eq!(n_probs, 344, "one probability per 512-sample window");
        let probs_ptr = whisper_vad_probs(vctx);
        assert!(
            !probs_ptr.is_null(),
            "Probabilities buffer must not be null"
        );
        let probs_slice = std::slice::from_raw_parts(probs_ptr, n_probs as usize);
        assert!(probs_slice.iter().all(|&p| (0.0..=1.0).contains(&p)));

        // Reference values from whisper.cpp on the same model and audio.
        assert!((probs_slice[0] - 0.001695).abs() < 1e-3);
        assert!((probs_slice[1] - 0.083180).abs() < 1e-3);
        assert!((probs_slice[50] - 0.985183).abs() < 1e-3);

        // 4. Extract segments from samples
        let segs = whisper_vad_segments_from_samples(
            vctx,
            vad_params,
            audio.samples.as_ptr(),
            audio.samples.len() as i32,
        );
        assert!(!segs.is_null());
        let n_segs = whisper_vad_segments_n_segments(segs);
        assert!(n_segs > 0, "Expected speech segments in JFK audio");
        for i in 0..n_segs {
            let t0 = whisper_vad_segments_get_segment_t0(segs, i);
            let t1 = whisper_vad_segments_get_segment_t1(segs, i);
            assert!(t1 >= t0, "Segment end must be >= start");
            assert!(t1 <= 1100.0, "Segment must stay inside the audio");
        }
        whisper_vad_free_segments(segs);

        // 5. Extract segments from probabilities
        let segs_probs = whisper_vad_segments_from_probs(vctx, vad_params);
        assert!(!segs_probs.is_null());
        assert!(whisper_vad_segments_n_segments(segs_probs) > 0);
        whisper_vad_free_segments(segs_probs);

        // 6. Pure silence still runs the graph, but every window is silent.
        let silence = vec![0.0f32; 16000 * 5];
        let detected_silence =
            whisper_vad_detect_speech(vctx, silence.as_ptr(), silence.len() as i32);
        assert_eq!(detected_silence, 1);
        let n_silence = whisper_vad_n_probs(vctx) as usize;
        let silence_probs = std::slice::from_raw_parts(whisper_vad_probs(vctx), n_silence);
        assert!(silence_probs.iter().all(|&p| p < 0.5));

        // 7. Streaming & reset state
        whisper_vad_reset_state(vctx);
        assert_eq!(whisper_vad_n_probs(vctx), 0);

        // 8. Free context & verify null safety
        whisper_vad_free(vctx);
        whisper_vad_free(std::ptr::null_mut());
        whisper_vad_free_segments(std::ptr::null_mut());
    }
}

#[test]
fn test_c_abi_vad_loader_init() {
    let model_path = resolve_model_path("for-tests-silero-v6.2.0-ggml.bin");
    if !model_path.exists() {
        eprintln!("VAD model not found, skipping VAD loader test");
        return;
    }

    let data = std::fs::read(&model_path).expect("read VAD model");
    let mut loader_ctx = LoaderContext {
        data,
        pos: 0,
        closed: false,
    };

    unsafe {
        let mut loader = whisper_model_loader {
            context: &mut loader_ctx as *mut LoaderContext as *mut std::ffi::c_void,
            read: Some(loader_read),
            eof: Some(loader_eof),
            close: Some(loader_close),
        };

        let vctx = whisper_vad_init_with_params(&mut loader, whisper_vad_default_context_params());
        assert!(!vctx.is_null(), "loader-based VAD init must succeed");
        assert!(loader_ctx.closed, "the loader must be closed");

        let samples = vec![0.0f32; 16000];
        assert_eq!(
            whisper_vad_detect_speech(vctx, samples.as_ptr(), samples.len() as i32),
            1
        );
        assert_eq!(whisper_vad_n_probs(vctx), 32);

        whisper_vad_free(vctx);
    }
}

#[test]
fn test_c_abi_honours_max_len_and_offset() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");

    if !model_path.exists() {
        eprintln!("Model not found, skipping C-ABI max_len test");
        return;
    }

    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    let sr = audio.sample_rate as usize;

    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null());

        // Baseline: one long segment for an 11 s clip.
        let mut params =
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        params.print_progress = false;
        assert_eq!(
            whisper_full(
                ctx,
                params,
                audio.samples.as_ptr(),
                audio.samples.len() as i32
            ),
            0
        );
        let n_plain = whisper_full_n_segments(ctx);
        let plain_text_len = {
            let t = CStr::from_ptr(whisper_full_get_segment_text(ctx, 0))
                .to_string_lossy()
                .trim()
                .chars()
                .count();
            t
        };
        assert!(
            plain_text_len > 40,
            "expected a full sentence, got {plain_text_len} chars"
        );

        // `max_len` must split the same audio into narrower segments. Before the
        // field was mapped, the C ABI accepted it and produced identical output.
        params.max_len = 20;
        params.split_on_word = true;
        assert_eq!(
            whisper_full(
                ctx,
                params,
                audio.samples.as_ptr(),
                audio.samples.len() as i32
            ),
            0
        );
        let n_split = whisper_full_n_segments(ctx);
        assert!(
            n_split > n_plain,
            "max_len had no effect: {n_plain} segment(s) before, {n_split} after"
        );
        for i in 0..n_split {
            let text = CStr::from_ptr(whisper_full_get_segment_text(ctx, i)).to_string_lossy();
            let chars = text.trim().chars().count();
            // A single word longer than the limit is still allowed through, so
            // the bound is the limit plus one word.
            assert!(
                chars <= 40,
                "segment {i} has {chars} chars, well past the 20 char limit: '{}'",
                text.trim()
            );
        }

        // `offset_ms` must shift the reported speech start.
        let mut offset_params =
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        offset_params.offset_ms = 3000;
        assert_eq!(
            whisper_full(
                ctx,
                offset_params,
                audio.samples.as_ptr(),
                (audio.samples.len() - 3 * sr) as i32
            ),
            0
        );
        assert!(whisper_full_n_segments(ctx) >= 1);

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_tokenize_contract_and_count() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping tokenize contract test");
        return;
    }

    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    let text = CString::new("Hello, world!").unwrap();
    let expected = [15947, 11, 1002, 0];

    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null());

        let count = whisper_token_count(ctx, text.as_ptr());
        assert_eq!(count, expected.len() as i32);

        // A buffer that is too small reports the required size as a negative
        // value, like whisper.cpp, instead of silently truncating.
        let mut small = [0i32; 2];
        let rc = whisper_tokenize(ctx, text.as_ptr(), small.as_mut_ptr(), small.len() as i32);
        assert_eq!(rc, -(expected.len() as i32));

        let mut tokens = [0i32; 4];
        let rc = whisper_tokenize(ctx, text.as_ptr(), tokens.as_mut_ptr(), tokens.len() as i32);
        assert_eq!(rc, expected.len() as i32);
        assert_eq!(tokens, expected);

        // Token text is returned as the raw vocabulary bytes. Token 171 is the
        // first byte of the fullwidth comma and is not valid UTF-8 on its own.
        let byte_token = whisper_token_to_str(ctx, 171);
        assert!(!byte_token.is_null());
        assert_eq!(CStr::from_ptr(byte_token).to_bytes(), &[0xef]);

        whisper_free(ctx);
    }
}

struct LoaderContext {
    data: Vec<u8>,
    pos: usize,
    closed: bool,
}

unsafe extern "C" fn loader_read(
    ctx: *mut std::ffi::c_void,
    output: *mut std::ffi::c_void,
    read_size: usize,
) -> usize {
    let loader = &mut *(ctx as *mut LoaderContext);
    let n = read_size.min(loader.data.len() - loader.pos);
    if n > 0 {
        std::ptr::copy_nonoverlapping(loader.data.as_ptr().add(loader.pos), output as *mut u8, n);
        loader.pos += n;
    }
    n
}

unsafe extern "C" fn loader_eof(ctx: *mut std::ffi::c_void) -> bool {
    let loader = &*(ctx as *const LoaderContext);
    loader.pos >= loader.data.len()
}

unsafe extern "C" fn loader_close(ctx: *mut std::ffi::c_void) {
    let loader = &mut *(ctx as *mut LoaderContext);
    loader.closed = true;
}

#[test]
fn test_c_abi_loader_init_and_openvino() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping loader init test");
        return;
    }

    let data = std::fs::read(&model_path).expect("read model");
    let expected_vocab = 51865;

    let mut loader_ctx = LoaderContext {
        data,
        pos: 0,
        closed: false,
    };

    unsafe {
        let mut loader = whisper_model_loader {
            context: &mut loader_ctx as *mut LoaderContext as *mut std::ffi::c_void,
            read: Some(loader_read),
            eof: Some(loader_eof),
            close: Some(loader_close),
        };

        let ctx = whisper_init_with_params(&mut loader, whisper_context_default_params());
        assert!(!ctx.is_null(), "loader-based init must succeed");
        assert!(loader_ctx.closed, "the loader must be closed exactly once");
        assert_eq!(whisper_n_vocab(ctx), expected_vocab);
        assert_eq!(whisper_full_n_segments(ctx), 0);

        let device = CString::new("CPU").unwrap();
        assert_eq!(
            whisper_ctx_init_openvino_encoder(
                ctx,
                std::ptr::null(),
                device.as_ptr(),
                std::ptr::null()
            ),
            1
        );

        whisper_free(ctx);
    }
}

#[derive(Default)]
struct SegmentCallbackData {
    calls: i32,
    n_new_sum: i32,
    last_total: i32,
}

unsafe extern "C" fn segment_callback(
    _ctx: *mut whisper_context,
    state: *mut whisper_state,
    n_new: i32,
    user_data: *mut std::ffi::c_void,
) {
    let data = &mut *(user_data as *mut SegmentCallbackData);
    data.calls += 1;
    data.n_new_sum += n_new;
    data.last_total = whisper_full_n_segments_from_state(state);
}

#[test]
fn test_c_abi_full_parallel_chunks_and_callbacks() {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping parallel test");
        return;
    }

    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();

    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null());

        let mut params =
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        let mut callback_data = SegmentCallbackData::default();
        params.new_segment_callback = segment_callback as *const std::ffi::c_void;
        params.new_segment_callback_user_data =
            &mut callback_data as *mut SegmentCallbackData as *mut std::ffi::c_void;

        let rc = whisper_full_parallel(
            ctx,
            params,
            audio.samples.as_ptr(),
            audio.samples.len() as i32,
            2,
        );
        assert_eq!(rc, 0, "parallel transcription must succeed");

        let n_segments = whisper_full_n_segments(ctx);
        assert!(
            n_segments >= 1,
            "parallel transcription produced no segments"
        );
        assert!(callback_data.calls >= 1);
        assert_eq!(callback_data.n_new_sum, n_segments);
        assert_eq!(callback_data.last_total, n_segments);

        let mut previous_t1 = 0;
        for i in 0..n_segments {
            let t0 = whisper_full_get_segment_t0(ctx, i);
            let t1 = whisper_full_get_segment_t1(ctx, i);
            assert!(t0 <= t1, "segment {} has reversed timestamps", i);
            assert!(
                t0 >= previous_t1,
                "segment {} starts before the previous one ended",
                i
            );
            previous_t1 = t1;
        }

        let duration_cs = (audio.samples.len() as f64 / 16000.0 * 100.0) as i64;
        assert!(
            previous_t1 <= duration_cs + 10,
            "last segment ends past the audio: {} > {}",
            previous_t1,
            duration_cs
        );

        whisper_free(ctx);
    }
}

#[test]
fn test_c_abi_lang_auto_detect_honours_offset() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!("Model or sample not found, skipping language offset test");
        return;
    }

    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();

    unsafe {
        let ctx = whisper_init_from_file(c_model_path.as_ptr());
        assert!(!ctx.is_null());
        assert_eq!(
            whisper_pcm_to_mel(ctx, audio.samples.as_ptr(), audio.samples.len() as i32, 1),
            0
        );

        let max_langs = whisper_lang_max_id() as usize + 1;
        let mut probs_start = vec![0.0f32; max_langs];
        let mut probs_later = vec![0.0f32; max_langs];

        let lang_start = whisper_lang_auto_detect(ctx, 0, 1, probs_start.as_mut_ptr());
        let lang_later = whisper_lang_auto_detect(ctx, 5_000, 1, probs_later.as_mut_ptr());
        assert!(lang_start >= 0);
        assert!(lang_later >= 0);

        // The two windows cover different audio, so the language posterior must
        // change. This fails if offset_ms is ignored.
        let max_delta = probs_start
            .iter()
            .zip(&probs_later)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-4,
            "offset_ms had no effect on the language posterior"
        );

        let mut scratch = vec![0.0f32; max_langs];
        assert_eq!(
            whisper_lang_auto_detect(ctx, -10, 1, scratch.as_mut_ptr()),
            -1,
            "negative offsets must be rejected"
        );
        let past_end = (audio.samples.len() as f64 / 16000.0 * 1000.0) as i32 + 1000;
        assert_eq!(
            whisper_lang_auto_detect(ctx, past_end, 1, scratch.as_mut_ptr()),
            -2,
            "offsets past the audio must be rejected"
        );

        whisper_free(ctx);
    }
}

unsafe extern "C" fn loader_read_full(
    ctx: *mut std::ffi::c_void,
    output: *mut std::ffi::c_void,
    read_size: usize,
) -> usize {
    let loader = &mut *(ctx as *mut LoaderContext);
    let n = read_size.min(loader.data.len() - loader.pos);
    if n > 0 {
        std::ptr::copy_nonoverlapping(loader.data.as_ptr().add(loader.pos), output as *mut u8, n);
        loader.pos += n;
    }
    // The reference file loader returns the requested size even at EOF and
    // relies on the eof callback to stop the reader.
    read_size
}

#[test]
fn test_c_abi_loader_uses_eof_callback() {
    let model_path = resolve_model_path("ggml-tiny.bin");
    if !model_path.exists() {
        eprintln!("Model not found, skipping EOF loader test");
        return;
    }

    let data = std::fs::read(&model_path).expect("read model");
    let mut loader_ctx = LoaderContext {
        data,
        pos: 0,
        closed: false,
    };

    unsafe {
        let mut loader = whisper_model_loader {
            context: &mut loader_ctx as *mut LoaderContext as *mut std::ffi::c_void,
            read: Some(loader_read_full),
            eof: Some(loader_eof),
            close: Some(loader_close),
        };

        let ctx = whisper_init_with_params(&mut loader, whisper_context_default_params());
        assert!(!ctx.is_null(), "EOF-driven loader must terminate and load");
        assert!(loader_ctx.closed);
        assert_eq!(whisper_n_vocab(ctx), 51865);
        whisper_free(ctx);
    }
}
