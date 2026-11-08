//! Integration coverage for the C callback slots and timing telemetry exposed
//! through `include/whisper.h`.
//!
//! Every test builds its own context so callback state and accumulated timings
//! cannot leak between cases. The model and audio are optional test assets: a
//! missing file skips the run instead of failing.

use std::ffi::{c_int, c_void, CString};
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

/// Loads the tiny.en model and the JFK sample, or returns `None` when either
/// asset is unavailable so the test can skip cleanly.
fn setup() -> Option<(*mut whisper_context, Vec<f32>)> {
    let model_path = resolve_model_path("ggml-tiny.en.bin");
    let wav_path = workspace_path("samples/jfk.wav");
    if !model_path.exists() || !wav_path.exists() {
        eprintln!(
            "skipping: model ({}) or audio ({}) missing",
            model_path.display(),
            wav_path.display()
        );
        return None;
    }

    let audio = whisper_rs::audio::WavAudio::load_from_file(&wav_path).expect("load wav");
    let c_model_path = CString::new(model_path.to_str().unwrap()).unwrap();
    let ctx = unsafe { whisper_init_from_file(c_model_path.as_ptr()) };
    assert!(!ctx.is_null(), "whisper_init_from_file returned null");
    Some((ctx, audio.samples))
}

fn default_params() -> whisper_full_params {
    whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY)
}

// ---------------------------------------------------------------------------
// 1. Progress callback
// ---------------------------------------------------------------------------

struct ProgressProbe {
    calls: i32,
    out_of_range: i32,
    last: c_int,
    ctx_ok: bool,
    expected_ctx: *mut whisper_context,
}

extern "C" fn on_progress(
    ctx: *mut whisper_context,
    _state: *mut whisper_state,
    progress: c_int,
    user_data: *mut c_void,
) {
    let probe = unsafe { &mut *(user_data as *mut ProgressProbe) };
    probe.calls += 1;
    probe.last = progress;
    if !(0..=100).contains(&progress) {
        probe.out_of_range += 1;
    }
    probe.ctx_ok &= ctx == probe.expected_ctx;
}

#[test]
fn test_progress_callback() {
    let Some((ctx, samples)) = setup() else {
        return;
    };

    let mut probe = ProgressProbe {
        calls: 0,
        out_of_range: 0,
        last: -1,
        ctx_ok: true,
        expected_ctx: ctx,
    };

    let mut params = default_params();
    params.progress_callback = on_progress as *const c_void;
    params.progress_callback_user_data = &mut probe as *mut ProgressProbe as *mut c_void;

    let ret = unsafe { whisper_full(ctx, params, samples.as_ptr(), samples.len() as i32) };
    assert_eq!(ret, 0, "whisper_full must succeed");

    assert!(probe.calls >= 1, "progress callback never fired");
    assert_eq!(probe.out_of_range, 0, "progress escaped 0..=100");
    assert!(
        probe.ctx_ok,
        "progress callback received a different context"
    );
    assert_eq!(probe.last, 100, "final progress report must be 100");

    unsafe { whisper_free(ctx) };
}

// ---------------------------------------------------------------------------
// 2. Encoder-begin callback
// ---------------------------------------------------------------------------

struct EncoderProbe {
    calls: i32,
    allow: bool,
    ctx_ok: bool,
    expected_ctx: *mut whisper_context,
}

extern "C" fn on_encoder_begin(
    ctx: *mut whisper_context,
    _state: *mut whisper_state,
    user_data: *mut c_void,
) -> bool {
    let probe = unsafe { &mut *(user_data as *mut EncoderProbe) };
    probe.calls += 1;
    probe.ctx_ok &= ctx == probe.expected_ctx;
    probe.allow
}

#[test]
fn test_encoder_begin_callback() {
    let Some((ctx, samples)) = setup() else {
        return;
    };

    let mut probe = EncoderProbe {
        calls: 0,
        allow: true,
        ctx_ok: true,
        expected_ctx: ctx,
    };

    let mut params = default_params();
    params.encoder_begin_callback = on_encoder_begin as *const c_void;
    params.encoder_begin_callback_user_data = &mut probe as *mut EncoderProbe as *mut c_void;

    let ret = unsafe { whisper_full(ctx, params, samples.as_ptr(), samples.len() as i32) };
    assert_eq!(ret, 0, "normal run must succeed");
    assert!(probe.calls >= 1, "encoder-begin callback never fired");
    assert!(probe.ctx_ok, "encoder-begin callback context mismatch");

    let normal_calls = probe.calls;
    let normal_segments = unsafe { whisper_full_n_segments(ctx) };
    assert!(
        normal_segments >= 1,
        "normal run must produce at least one segment"
    );

    // Second pass on the same context: refuse every encoder pass. The engine
    // clears the state at the start of `whisper_full`, so the vetoed run must
    // leave no segments behind and still report success.
    probe.allow = false;
    let ret = unsafe { whisper_full(ctx, params, samples.as_ptr(), samples.len() as i32) };
    assert_eq!(ret, 0, "vetoed run must still return 0");
    assert!(
        probe.calls > normal_calls,
        "encoder-begin callback did not fire on the vetoed run"
    );

    let aborted_segments = unsafe { whisper_full_n_segments(ctx) };
    assert_eq!(
        aborted_segments, 0,
        "encoder-begin veto must produce zero segments (normal run produced {normal_segments})"
    );
    assert!(aborted_segments < normal_segments);

    unsafe { whisper_free(ctx) };
}

// ---------------------------------------------------------------------------
// 3. Abort callback
// ---------------------------------------------------------------------------

struct AbortProbe {
    calls: i32,
}

extern "C" fn on_abort(user_data: *mut c_void) -> bool {
    let probe = unsafe { &mut *(user_data as *mut AbortProbe) };
    probe.calls += 1;
    // Always abort. The engine checks the callback before the first window and
    // at every window boundary.
    true
}

#[test]
fn test_abort_callback() {
    let Some((ctx, samples)) = setup() else {
        return;
    };

    let mut probe = AbortProbe { calls: 0 };

    let mut params = default_params();
    params.abort_callback = on_abort as *const c_void;
    params.abort_callback_user_data = &mut probe as *mut AbortProbe as *mut c_void;

    let ret = unsafe { whisper_full(ctx, params, samples.as_ptr(), samples.len() as i32) };
    // Aborting before the first window is a clean stop, not an error: the
    // current implementation returns 0 and keeps whatever segments existed.
    // Because the abort fires at the very start, that is zero segments.
    assert_eq!(ret, 0, "aborted run must return a valid code");
    assert!(probe.calls >= 1, "abort callback never fired");

    let n_segments = unsafe { whisper_full_n_segments(ctx) };
    assert_eq!(
        n_segments, 0,
        "abort before the first window must not emit segments"
    );

    unsafe { whisper_free(ctx) };
}

// ---------------------------------------------------------------------------
// 4. Logits filter callback
// ---------------------------------------------------------------------------

struct LogitsProbe {
    calls: i32,
    expected_vocab: i32,
    bad_vocab: i32,
    bad_n_tokens: i32,
    bad_token_ids: i32,
    clipped: i32,
}

extern "C" fn on_logits(
    ctx: *mut whisper_context,
    _state: *mut whisper_state,
    tokens: *const whisper_token_data,
    n_tokens: c_int,
    logits: *mut f32,
    user_data: *mut c_void,
) {
    let probe = unsafe { &mut *(user_data as *mut LogitsProbe) };
    probe.calls += 1;

    let n_vocab = unsafe { whisper_n_vocab(ctx) };
    if n_vocab != probe.expected_vocab {
        probe.bad_vocab += 1;
    }
    if n_tokens < 0 {
        probe.bad_n_tokens += 1;
    }

    if !tokens.is_null() && n_tokens > 0 {
        let history = unsafe { std::slice::from_raw_parts(tokens, n_tokens as usize) };
        for t in history {
            if t.id < 0 || t.id >= n_vocab {
                probe.bad_token_ids += 1;
            }
        }
    }

    // The buffer is documented to hold `whisper_n_vocab(ctx)` floats. Read and
    // write both ends so a short buffer would trap under ASan instead of
    // silently corrupting a neighbour, while leaving the logits unchanged.
    if !logits.is_null() && n_vocab > 0 {
        let last = (n_vocab - 1) as usize;
        unsafe {
            let head = std::ptr::read(logits);
            std::ptr::write(logits, head);
            let tail = std::ptr::read(logits.add(last));
            std::ptr::write(logits.add(last), tail);
        }
        probe.clipped += 1;
    }
}

#[test]
fn test_logits_filter_callback() {
    let Some((ctx, samples)) = setup() else {
        return;
    };

    let n_vocab = unsafe { whisper_n_vocab(ctx) };
    assert!(n_vocab > 0, "model must expose a vocabulary");

    let mut probe = LogitsProbe {
        calls: 0,
        expected_vocab: n_vocab,
        bad_vocab: 0,
        bad_n_tokens: 0,
        bad_token_ids: 0,
        clipped: 0,
    };

    let mut params = default_params();
    params.logits_filter_callback = on_logits as *const c_void;
    params.logits_filter_callback_user_data = &mut probe as *mut LogitsProbe as *mut c_void;

    let ret = unsafe { whisper_full(ctx, params, samples.as_ptr(), samples.len() as i32) };
    assert_eq!(ret, 0, "whisper_full must succeed");
    assert!(probe.calls >= 1, "logits callback never fired");
    assert_eq!(probe.bad_vocab, 0, "callback observed a different n_vocab");
    assert_eq!(probe.bad_n_tokens, 0, "callback received negative n_tokens");
    assert_eq!(
        probe.bad_token_ids, 0,
        "token history contained ids outside the vocabulary"
    );
    assert_eq!(
        probe.clipped, probe.calls,
        "every call must have a writable logits buffer"
    );

    unsafe { whisper_free(ctx) };
}

// ---------------------------------------------------------------------------
// 5. New-segment callback
// ---------------------------------------------------------------------------

struct SegmentProbe {
    calls: i32,
    sum_n_new: i32,
    bad_n_new: i32,
}

extern "C" fn on_new_segment(
    _ctx: *mut whisper_context,
    state: *mut whisper_state,
    n_new: c_int,
    user_data: *mut c_void,
) {
    let probe = unsafe { &mut *(user_data as *mut SegmentProbe) };
    let total = unsafe { whisper_full_n_segments_from_state(state) };
    probe.calls += 1;
    probe.sum_n_new += n_new;
    // The canonical index is `total - n_new`; it must address a real segment.
    if n_new <= 0 || n_new > total {
        probe.bad_n_new += 1;
    }
}

#[test]
fn test_new_segment_callback() {
    let Some((ctx, samples)) = setup() else {
        return;
    };

    let mut probe = SegmentProbe {
        calls: 0,
        sum_n_new: 0,
        bad_n_new: 0,
    };

    let mut params = default_params();
    params.new_segment_callback = on_new_segment as *const c_void;
    params.new_segment_callback_user_data = &mut probe as *mut SegmentProbe as *mut c_void;

    let ret = unsafe { whisper_full(ctx, params, samples.as_ptr(), samples.len() as i32) };
    assert_eq!(ret, 0, "whisper_full must succeed");

    let n_segments = unsafe { whisper_full_n_segments(ctx) };
    assert!(n_segments >= 1, "run must produce at least one segment");
    assert!(probe.calls >= 1, "new-segment callback must fire");
    assert_eq!(
        probe.sum_n_new, n_segments,
        "reported n_new values must sum to the segment count"
    );
    assert_eq!(
        probe.bad_n_new, 0,
        "every call must report a valid n_new for the visible segment list"
    );

    unsafe { whisper_free(ctx) };
}

// ---------------------------------------------------------------------------
// 6. Timings
// ---------------------------------------------------------------------------

#[test]
fn test_timings() {
    let Some((ctx, samples)) = setup() else {
        return;
    };

    let params = default_params();
    let ret = unsafe { whisper_full(ctx, params, samples.as_ptr(), samples.len() as i32) };
    assert_eq!(ret, 0, "whisper_full must succeed");

    let timings = unsafe { whisper_get_timings(ctx) };
    assert!(!timings.is_null(), "whisper_get_timings returned null");
    let t = unsafe { &*timings };
    assert!(
        t.encode_ms > 0.0,
        "encode_ms must be positive, got {}",
        t.encode_ms
    );
    assert!(
        t.decode_ms > 0.0,
        "decode_ms must be positive, got {}",
        t.decode_ms
    );
    assert!(
        t.sample_ms > 0.0,
        "sample_ms must be positive, got {}",
        t.sample_ms
    );

    // Must be safe to call with a live context (output goes to the log sink).
    unsafe { whisper_print_timings(ctx) };

    unsafe { whisper_reset_timings(ctx) };

    let timings = unsafe { whisper_get_timings(ctx) };
    assert!(
        !timings.is_null(),
        "whisper_get_timings returned null after reset"
    );
    let t = unsafe { &*timings };
    assert_eq!(t.encode_ms, 0.0, "reset must zero encode_ms");
    assert_eq!(t.decode_ms, 0.0, "reset must zero decode_ms");
    assert_eq!(t.sample_ms, 0.0, "reset must zero sample_ms");

    unsafe { whisper_free(ctx) };
}
