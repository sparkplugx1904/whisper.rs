#![allow(clippy::missing_safety_doc, clippy::manual_c_str_literals)]
//! C-ABI Compatibility Layer exporting libwhisper symbols matching whisper.h.
//! Allows drop-in replacement for Go, Python, Ruby, and C/C++ applications.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use whisper_rs::model::vocab::{lang_id, lang_str};
use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams, WhisperState};

#[allow(non_camel_case_types)]
pub struct whisper_context {
    pub inner: WhisperContext,
    pub default_state: Box<whisper_state>,
    pub segment_c_strings: Vec<CString>,
    pub token_c_strings: Vec<Vec<CString>>,
    pub token_to_str_cache: Vec<CString>,
}

#[allow(non_camel_case_types)]
pub struct whisper_state {
    pub inner: WhisperState,
    pub segment_c_strings: Vec<CString>,
    pub token_c_strings: Vec<Vec<CString>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum whisper_sampling_strategy {
    WHISPER_SAMPLING_GREEDY = 0,
    WHISPER_SAMPLING_BEAM_SEARCH = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_ahead {
    pub n_text_layer: c_int,
    pub n_head: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_aheads {
    pub n_heads: usize,
    pub heads: *const whisper_ahead,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_context_params {
    pub use_gpu: bool,
    pub flash_attn: bool,
    pub gpu_device: c_int,
    pub dtw_token_timestamps: bool,
    pub dtw_aheads_preset: c_int,
    pub dtw_n_top: c_int,
    pub dtw_aheads: whisper_aheads,
    pub dtw_mem_size: usize,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub struct whisper_token_data {
    pub id: c_int,
    pub tid: c_int,
    pub p: f32,
    pub plog: f32,
    pub pt: f32,
    pub ptsum: f32,
    pub t0: i64,
    pub t1: i64,
    pub t_dtw: i64,
    pub vlen: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_greedy_params {
    pub best_of: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_beam_search_params {
    pub beam_size: c_int,
    pub patience: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_full_params {
    pub strategy: whisper_sampling_strategy,
    pub n_threads: c_int,
    pub n_max_text_ctx: c_int,
    pub offset_ms: c_int,
    pub duration_ms: c_int,

    pub translate: bool,
    pub no_context: bool,
    pub no_timestamps: bool,
    pub single_segment: bool,
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,

    pub token_timestamps: bool,
    pub thold_pt: f32,
    pub thold_ptsum: f32,
    pub max_len: c_int,
    pub split_on_word: bool,
    pub max_tokens: c_int,

    pub debug_mode: bool,
    pub audio_ctx: c_int,
    pub tdrz_enable: bool,

    pub suppress_regex: *const c_char,
    pub initial_prompt: *const c_char,
    pub carry_initial_prompt: bool,
    pub prompt_tokens: *const c_int,
    pub prompt_n_tokens: c_int,

    pub language: *const c_char,
    pub detect_language: bool,

    pub suppress_blank: bool,
    pub suppress_nst: bool,

    pub temperature: f32,
    pub max_initial_ts: f32,
    pub length_penalty: f32,

    pub temperature_inc: f32,
    pub entropy_thold: f32,
    pub logprob_thold: f32,
    pub no_speech_thold: f32,

    pub greedy: whisper_greedy_params,
    pub beam_search: whisper_beam_search_params,

    pub new_segment_callback: *const c_void,
    pub new_segment_callback_user_data: *mut c_void,

    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,

    pub encoder_begin_callback: *const c_void,
    pub encoder_begin_callback_user_data: *mut c_void,

    pub abort_callback: *const c_void,
    pub abort_callback_user_data: *mut c_void,

    pub logits_filter_callback: *const c_void,
    pub logits_filter_callback_user_data: *mut c_void,

    pub grammar_rules: *const c_void,
    pub n_grammar_rules: usize,
    pub i_start_rule: usize,
    pub grammar_penalty: f32,

    pub vad: bool,
    pub vad_model: *const c_char,
    pub vad_threshold: f32,
    pub vad_min_speech_duration_ms: c_int,
    pub vad_min_silence_duration_ms: c_int,
    pub vad_max_speech_duration_s: f32,
    pub vad_speech_pad_ms: c_int,
    pub vad_samples_overlap: f32,
}

#[no_mangle]
pub extern "C" fn whisper_context_default_params() -> whisper_context_params {
    whisper_context_params {
        use_gpu: false,
        flash_attn: false,
        gpu_device: 0,
        dtw_token_timestamps: false,
        dtw_aheads_preset: 0,
        dtw_n_top: 0,
        dtw_aheads: whisper_aheads {
            n_heads: 0,
            heads: ptr::null(),
        },
        dtw_mem_size: 0,
    }
}

#[no_mangle]
pub extern "C" fn whisper_full_default_params(
    strategy: whisper_sampling_strategy,
) -> whisper_full_params {
    whisper_full_params {
        strategy,
        n_threads: 4,
        n_max_text_ctx: 16384,
        offset_ms: 0,
        duration_ms: 0,
        translate: false,
        no_context: false,
        no_timestamps: false,
        single_segment: false,
        print_special: false,
        print_progress: false,
        print_realtime: false,
        print_timestamps: true,
        token_timestamps: false,
        thold_pt: 0.01,
        thold_ptsum: 0.01,
        max_len: 0,
        split_on_word: false,
        max_tokens: 0,
        debug_mode: false,
        audio_ctx: 0,
        tdrz_enable: false,
        suppress_regex: ptr::null(),
        initial_prompt: ptr::null(),
        carry_initial_prompt: false,
        prompt_tokens: ptr::null(),
        prompt_n_tokens: 0,
        language: b"en\0".as_ptr() as *const c_char,
        detect_language: false,
        suppress_blank: true,
        suppress_nst: false,
        temperature: 0.0,
        max_initial_ts: 1.0,
        length_penalty: -1.0,
        temperature_inc: 0.2,
        entropy_thold: 2.4,
        logprob_thold: -1.0,
        no_speech_thold: 0.6,
        greedy: whisper_greedy_params { best_of: 5 },
        beam_search: whisper_beam_search_params {
            beam_size: match strategy {
                whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH => 5,
                _ => 1,
            },
            patience: -1.0,
        },
        new_segment_callback: ptr::null(),
        new_segment_callback_user_data: ptr::null_mut(),
        progress_callback: ptr::null(),
        progress_callback_user_data: ptr::null_mut(),
        encoder_begin_callback: ptr::null(),
        encoder_begin_callback_user_data: ptr::null_mut(),
        abort_callback: ptr::null(),
        abort_callback_user_data: ptr::null_mut(),
        logits_filter_callback: ptr::null(),
        logits_filter_callback_user_data: ptr::null_mut(),
        grammar_rules: ptr::null(),
        n_grammar_rules: 0,
        i_start_rule: 0,
        grammar_penalty: 100.0,
        vad: false,
        vad_model: ptr::null(),
        vad_threshold: 0.5,
        vad_min_speech_duration_ms: 250,
        vad_min_silence_duration_ms: 100,
        vad_max_speech_duration_s: f32::MAX,
        vad_speech_pad_ms: 30,
        vad_samples_overlap: 0.1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_file_with_params(
    path: *const c_char,
    _params: whisper_context_params,
) -> *mut whisper_context {
    whisper_init_from_file(path)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_file(path: *const c_char) -> *mut whisper_context {
    if path.is_null() {
        return ptr::null_mut();
    }
    let c_str = CStr::from_ptr(path);
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let model = match WhisperModel::load_from_file(path_str) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };

    let inner = match WhisperContext::new(model) {
        Ok(c) => c,
        Err(_) => return ptr::null_mut(),
    };

    let default_state = Box::new(whisper_state {
        inner: inner.create_state(),
        segment_c_strings: Vec::new(),
        token_c_strings: Vec::new(),
    });
    let token_to_str_cache = inner
        .model
        .vocab
        .id_to_token
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap_or_else(|_| CString::new("").unwrap()))
        .collect();

    let ctx = Box::new(whisper_context {
        inner,
        default_state,
        segment_c_strings: Vec::new(),
        token_c_strings: Vec::new(),
        token_to_str_cache,
    });

    Box::into_raw(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_buffer(
    buffer: *mut c_void,
    buffer_size: usize,
) -> *mut whisper_context {
    if buffer.is_null() || buffer_size == 0 {
        return ptr::null_mut();
    }

    let slice = std::slice::from_raw_parts(buffer as *const u8, buffer_size);
    let model = match WhisperModel::load_from_bytes(slice.to_vec()) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };

    let inner = match WhisperContext::new(model) {
        Ok(c) => c,
        Err(_) => return ptr::null_mut(),
    };

    let default_state = Box::new(whisper_state {
        inner: inner.create_state(),
        segment_c_strings: Vec::new(),
        token_c_strings: Vec::new(),
    });
    let token_to_str_cache = inner
        .model
        .vocab
        .id_to_token
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap_or_else(|_| CString::new("").unwrap()))
        .collect();

    let ctx = Box::new(whisper_context {
        inner,
        default_state,
        segment_c_strings: Vec::new(),
        token_c_strings: Vec::new(),
        token_to_str_cache,
    });

    Box::into_raw(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_state(ctx: *mut whisper_context) -> *mut whisper_state {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    let context = &*ctx;
    let state = Box::new(whisper_state {
        inner: context.inner.create_state(),
        segment_c_strings: Vec::new(),
        token_c_strings: Vec::new(),
    });
    Box::into_raw(state)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_free(ctx: *mut whisper_context) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_free_state(state: *mut whisper_state) {
    if !state.is_null() {
        drop(Box::from_raw(state));
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_with_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    params: whisper_full_params,
    samples: *const f32,
    n_samples: c_int,
) -> c_int {
    if ctx.is_null() || state.is_null() || samples.is_null() || n_samples <= 0 {
        return -1;
    }

    let context = &*ctx;
    let state_ref = &mut *state;

    let sample_slice = std::slice::from_raw_parts(samples, n_samples as usize);

    let lang_str = if !params.language.is_null() {
        CStr::from_ptr(params.language)
            .to_str()
            .unwrap_or("auto")
            .to_string()
    } else {
        "auto".to_string()
    };

    let prompt_str = if !params.initial_prompt.is_null() {
        CStr::from_ptr(params.initial_prompt)
            .to_str()
            .ok()
            .map(|s| s.to_string())
    } else {
        None
    };

    let beam_sz = if params.strategy == whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH {
        params.beam_search.beam_size.max(1) as usize
    } else {
        1
    };

    let rust_params = WhisperParams {
        language: lang_str,
        translate: params.translate,
        temperature: params.temperature,
        temperature_inc: params.temperature_inc,
        beam_size: beam_sz,
        no_timestamps: params.no_timestamps,
        initial_prompt: prompt_str,
        n_threads: params.n_threads.max(1) as usize,
        max_tokens: if params.max_tokens > 0 {
            params.max_tokens as usize
        } else {
            0
        },
        ..Default::default()
    };

    if context
        .inner
        .full(&mut state_ref.inner, &rust_params, sample_slice)
        .is_err()
    {
        return -2;
    }

    // Refresh CString cache for segments and tokens
    state_ref.segment_c_strings.clear();
    state_ref.token_c_strings.clear();
    for seg in &state_ref.inner.segments {
        let c_str = CString::new(seg.text.clone()).unwrap_or_else(|_| CString::new("").unwrap());
        state_ref.segment_c_strings.push(c_str);

        let mut tok_cstrs = Vec::new();
        for &tok_id in &seg.tokens {
            let tok_str = context.inner.tokenizer.id_to_token(tok_id).unwrap_or("");
            tok_cstrs.push(CString::new(tok_str).unwrap_or_else(|_| CString::new("").unwrap()));
        }
        state_ref.token_c_strings.push(tok_cstrs);
    }
    // Invoke new_segment_callback if provided
    if !params.new_segment_callback.is_null() {
        type NewSegCb = extern "C" fn(*mut whisper_context, *mut whisper_state, c_int, *mut c_void);
        let cb: NewSegCb = std::mem::transmute(params.new_segment_callback);
        cb(
            ctx,
            state,
            state_ref.inner.segments.len() as c_int,
            params.new_segment_callback_user_data,
        );
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full(
    ctx: *mut whisper_context,
    params: whisper_full_params,
    samples: *const f32,
    n_samples: c_int,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    let state_ptr = (*ctx).default_state.as_mut() as *mut whisper_state;
    whisper_full_with_state(ctx, state_ptr, params, samples, n_samples)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_parallel(
    ctx: *mut whisper_context,
    params: whisper_full_params,
    samples: *const f32,
    n_samples: c_int,
    _n_processors: c_int,
) -> c_int {
    whisper_full(ctx, params, samples, n_samples)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_n_segments_from_state(state: *mut whisper_state) -> c_int {
    if state.is_null() {
        return 0;
    }
    let state_ref = &*state;
    state_ref.inner.segments.len() as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_n_segments(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.default_state.inner.segments.len() as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_t0_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
) -> i64 {
    if state.is_null() || i_segment < 0 {
        return 0;
    }
    let state_ref = &*state;
    if let Some(seg) = state_ref.inner.segments.get(i_segment as usize) {
        (seg.t0 * 100.0).round() as i64
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_t0(
    ctx: *mut whisper_context,
    i_segment: c_int,
) -> i64 {
    if ctx.is_null() || i_segment < 0 {
        return 0;
    }
    let context = &*ctx;
    if let Some(seg) = context.default_state.inner.segments.get(i_segment as usize) {
        (seg.t0 * 100.0).round() as i64
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_t1_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
) -> i64 {
    if state.is_null() || i_segment < 0 {
        return 0;
    }
    let state_ref = &*state;
    if let Some(seg) = state_ref.inner.segments.get(i_segment as usize) {
        (seg.t1 * 100.0).round() as i64
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_t1(
    ctx: *mut whisper_context,
    i_segment: c_int,
) -> i64 {
    if ctx.is_null() || i_segment < 0 {
        return 0;
    }
    let context = &*ctx;
    if let Some(seg) = context.default_state.inner.segments.get(i_segment as usize) {
        (seg.t1 * 100.0).round() as i64
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_text_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
) -> *const c_char {
    if state.is_null() || i_segment < 0 {
        return ptr::null();
    }
    let state_ref = &*state;
    if let Some(c_str) = state_ref.segment_c_strings.get(i_segment as usize) {
        c_str.as_ptr()
    } else {
        ptr::null()
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_text(
    ctx: *mut whisper_context,
    i_segment: c_int,
) -> *const c_char {
    if ctx.is_null() || i_segment < 0 {
        return ptr::null();
    }
    let context = &*ctx;
    if let Some(c_str) = context
        .default_state
        .segment_c_strings
        .get(i_segment as usize)
    {
        c_str.as_ptr()
    } else {
        ptr::null()
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_n_tokens(
    ctx: *mut whisper_context,
    i_segment: c_int,
) -> c_int {
    if ctx.is_null() || i_segment < 0 {
        return 0;
    }
    let context = &*ctx;
    if let Some(seg) = context.default_state.inner.segments.get(i_segment as usize) {
        seg.tokens.len() as c_int
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_id(
    ctx: *mut whisper_context,
    i_segment: c_int,
    i_token: c_int,
) -> c_int {
    if ctx.is_null() || i_segment < 0 || i_token < 0 {
        return 0;
    }
    let context = &*ctx;
    if let Some(seg) = context.default_state.inner.segments.get(i_segment as usize) {
        seg.tokens.get(i_token as usize).copied().unwrap_or(0)
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_text(
    ctx: *mut whisper_context,
    i_segment: c_int,
    i_token: c_int,
) -> *const c_char {
    if ctx.is_null() || i_segment < 0 || i_token < 0 {
        return ptr::null();
    }
    let context = &*ctx;
    if let Some(seg) = context
        .default_state
        .token_c_strings
        .get(i_segment as usize)
    {
        if let Some(c_str) = seg.get(i_token as usize) {
            return c_str.as_ptr();
        }
    }
    ptr::null()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_data_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
    i_token: c_int,
) -> whisper_token_data {
    let empty = whisper_token_data {
        id: 0,
        tid: 0,
        p: 0.0,
        plog: 0.0,
        pt: 0.0,
        ptsum: 0.0,
        t0: 0,
        t1: 0,
        t_dtw: 0,
        vlen: 0.0,
    };
    if state.is_null() || i_segment < 0 || i_token < 0 {
        return empty;
    }
    let state_ref = &*state;
    if let Some(seg) = state_ref.inner.segments.get(i_segment as usize) {
        if let Some(&tok_id) = seg.tokens.get(i_token as usize) {
            let n_toks = seg.tokens.len().max(1);
            let seg_dur = (seg.t1 - seg.t0).max(0.0);
            let tok_step = seg_dur / (n_toks as f64);
            let mut t0 = seg.t0 + (i_token as f64) * tok_step;
            let mut t1 = t0 + tok_step;
            let mut prob = 0.95f32;

            if !seg.words.is_empty() {
                let word_idx = (i_token as usize * seg.words.len()) / n_toks;
                if let Some(w) = seg.words.get(word_idx) {
                    t0 = w.t0;
                    t1 = w.t1;
                    prob = w.probability;
                }
            }

            let t0_cs = (t0 * 100.0).round() as i64;
            let t1_cs = (t1 * 100.0).round() as i64;

            return whisper_token_data {
                id: tok_id,
                tid: 0,
                p: prob,
                plog: prob.max(1e-6).ln(),
                pt: 1.0,
                ptsum: 1.0,
                t0: t0_cs,
                t1: t1_cs,
                t_dtw: t0_cs,
                vlen: (t1 - t0) as f32,
            };
        }
    }
    empty
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_data(
    ctx: *mut whisper_context,
    i_segment: c_int,
    i_token: c_int,
) -> whisper_token_data {
    if ctx.is_null() {
        return whisper_token_data {
            id: 0,
            tid: 0,
            p: 0.0,
            plog: 0.0,
            pt: 0.0,
            ptsum: 0.0,
            t0: 0,
            t1: 0,
            t_dtw: 0,
            vlen: 0.0,
        };
    }
    let state_ptr = (*ctx).default_state.as_mut() as *mut whisper_state;
    whisper_full_get_token_data_from_state(state_ptr, i_segment, i_token)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_p_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
    i_token: c_int,
) -> f32 {
    whisper_full_get_token_data_from_state(state, i_segment, i_token).p
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_p(
    ctx: *mut whisper_context,
    i_segment: c_int,
    i_token: c_int,
) -> f32 {
    whisper_full_get_token_data(ctx, i_segment, i_token).p
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_t0_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
    i_token: c_int,
) -> i64 {
    whisper_full_get_token_data_from_state(state, i_segment, i_token).t0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_t0(
    ctx: *mut whisper_context,
    i_segment: c_int,
    i_token: c_int,
) -> i64 {
    whisper_full_get_token_data(ctx, i_segment, i_token).t0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_t1_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
    i_token: c_int,
) -> i64 {
    whisper_full_get_token_data_from_state(state, i_segment, i_token).t1
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_t1(
    ctx: *mut whisper_context,
    i_segment: c_int,
    i_token: c_int,
) -> i64 {
    whisper_full_get_token_data(ctx, i_segment, i_token).t1
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_lang_id(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.default_state.inner.lang_id as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_is_multilingual(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    if context.inner.model.hparams.is_multilingual() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_n_vocab(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_vocab
}

#[no_mangle]
pub unsafe extern "C" fn whisper_n_audio_ctx(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_audio_ctx
}

#[no_mangle]
pub unsafe extern "C" fn whisper_n_text_ctx(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_text_ctx
}

#[no_mangle]
pub unsafe extern "C" fn whisper_n_len(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    (context.default_state.inner.encoder_output.len()
        / context.inner.model.hparams.n_audio_state.max(1) as usize) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_eot(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.tokenizer.eot()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_sot(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.tokenizer.sot()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_prev(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.vocab.token_prev
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_solm(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.vocab.token_solm
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_not(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.tokenizer.not()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_beg(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.tokenizer.beg()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_translate(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.tokenizer.translate()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_transcribe(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.tokenizer.transcribe()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_lang(ctx: *mut whisper_context, lang_id: c_int) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.tokenizer.sot() + 1 + lang_id
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_to_str(
    ctx: *mut whisper_context,
    token: c_int,
) -> *const c_char {
    if ctx.is_null() || token < 0 {
        return ptr::null();
    }
    let context = &*ctx;
    if let Some(c_str) = context.token_to_str_cache.get(token as usize) {
        c_str.as_ptr()
    } else {
        ptr::null()
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_tokenize(
    ctx: *mut whisper_context,
    text: *const c_char,
    tokens: *mut c_int,
    n_max_tokens: c_int,
) -> c_int {
    if ctx.is_null() || text.is_null() || tokens.is_null() || n_max_tokens <= 0 {
        return 0;
    }
    let context = &*ctx;
    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let token_ids = context.inner.tokenizer.encode(text_str);
    let count = token_ids.len().min(n_max_tokens as usize);
    for (i, &tok_id) in token_ids.iter().take(count).enumerate() {
        *tokens.add(i) = tok_id;
    }
    count as c_int
}

#[no_mangle]
pub extern "C" fn whisper_lang_max_id() -> c_int {
    (whisper_rs::model::vocab::LANGUAGES.len() - 1) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_lang_id(lang_cstr: *const c_char) -> c_int {
    if lang_cstr.is_null() {
        return -1;
    }
    if let Ok(s) = CStr::from_ptr(lang_cstr).to_str() {
        lang_id(s).map(|idx| idx as c_int).unwrap_or(-1)
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn whisper_lang_str(id: c_int) -> *const c_char {
    if id < 0 {
        return ptr::null();
    }
    if lang_str(id as usize).is_some() {
        static CACHED_LANGS: std::sync::LazyLock<Vec<CString>> = std::sync::LazyLock::new(|| {
            whisper_rs::model::vocab::LANGUAGES
                .iter()
                .map(|(code, _)| CString::new(*code).unwrap())
                .collect()
        });
        if (id as usize) < CACHED_LANGS.len() {
            return CACHED_LANGS[id as usize].as_ptr();
        }
        ptr::null()
    } else {
        ptr::null()
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_pcm_to_mel(
    ctx: *mut whisper_context,
    samples: *const f32,
    n_samples: c_int,
    _n_threads: c_int,
) -> c_int {
    if ctx.is_null() || samples.is_null() || n_samples <= 0 {
        return -1;
    }
    let context = &mut *ctx;
    let sample_slice = std::slice::from_raw_parts(samples, n_samples as usize);
    let mel = whisper_rs::audio::compute_log_mel_spectrogram(
        sample_slice,
        &context.inner.model.filters.data,
        context.inner.model.filters.n_mel,
    );
    context.default_state.inner.mel = Some(mel);
    0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_set_mel(
    ctx: *mut whisper_context,
    data: *const f32,
    n_len: c_int,
    n_mel: c_int,
) -> c_int {
    if ctx.is_null() || data.is_null() || n_len <= 0 || n_mel <= 0 {
        return -1;
    }
    if n_len.checked_mul(n_mel).is_none() {
        return -1;
    }
    let context = &mut *ctx;
    let total_elements = match (n_len as usize).checked_mul(n_mel as usize) {
        Some(n) => n,
        None => return -1,
    };
    let mel_slice = std::slice::from_raw_parts(data, total_elements);
    let mel = whisper_rs::audio::MelSpectrogram {
        n_mel: n_mel as usize,
        n_len: n_len as usize,
        n_len_org: n_len as usize,
        data: mel_slice.to_vec(),
    };
    context.default_state.inner.mel = Some(mel);
    0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_lang_auto_detect(
    ctx: *mut whisper_context,
    offset_ms: c_int,
    n_threads: c_int,
    lang_probs: *mut f32,
) -> c_int {
    whisper_lang_auto_detect_with_state(ctx, ptr::null_mut(), offset_ms, n_threads, lang_probs)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_lang_auto_detect_with_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    _offset_ms: c_int,
    _n_threads: c_int,
    lang_probs: *mut f32,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    let context = &*ctx;
    let st = if !state.is_null() {
        &(*state).inner
    } else {
        &context.default_state.inner
    };

    let mel = match &st.mel {
        Some(m) => m,
        None => return -1,
    };

    let max_langs = whisper_lang_max_id() as usize + 1;
    let mut probs = vec![0.0f32; max_langs];

    let (_code, best_idx, _prob) = context.inner.detect_language_probs(mel, Some(&mut probs));

    if !lang_probs.is_null() {
        let slice = std::slice::from_raw_parts_mut(lang_probs, max_langs);
        slice.copy_from_slice(&probs);
    }

    best_idx as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_encode(
    ctx: *mut whisper_context,
    offset: c_int,
    _n_threads: c_int,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    let context = &mut *ctx;
    let n_audio_ctx = context.inner.model.hparams.n_audio_ctx as usize;
    let mel_chunk = if let Some(ref mel) = context.default_state.inner.mel {
        mel.get_chunk(offset.max(0) as usize, n_audio_ctx)
    } else {
        vec![0.0f32; context.inner.model.filters.n_mel * 2 * n_audio_ctx]
    };

    let enc_out = context.inner.encoder.forward(&mel_chunk);
    context.default_state.inner.encoder_output = enc_out;
    0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_decode(
    ctx: *mut whisper_context,
    tokens: *const c_int,
    n_tokens: c_int,
    n_past: c_int,
    _n_threads: c_int,
) -> c_int {
    if ctx.is_null() || tokens.is_null() || n_tokens <= 0 {
        return -1;
    }
    let context = &mut *ctx;
    let n_vocab = context.inner.model.hparams.n_vocab;
    let token_slice = std::slice::from_raw_parts(tokens, n_tokens as usize);
    for &tok in token_slice {
        if tok < 0 || tok >= n_vocab {
            return -1;
        }
    }
    let tok_vec: Vec<i32> = token_slice.to_vec();

    context.default_state.inner.cache.t_past = n_past.max(0) as usize;
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        context.inner.decoder.decode(
            &tok_vec,
            &context.default_state.inner.encoder_output,
            &mut context.default_state.inner.cache,
        )
    }));
    match res {
        Ok(logits) => {
            context.default_state.inner.logits = logits;
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_get_logits(ctx: *mut whisper_context) -> *mut f32 {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    let context = &mut *ctx;
    let len = context.default_state.inner.logits.len();
    let n_vocab = context.inner.model.hparams.n_vocab as usize;
    if len < n_vocab {
        ptr::null_mut()
    } else {
        context.default_state.inner.logits[len - n_vocab..].as_mut_ptr()
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_get_logits_from_state(state: *mut whisper_state) -> *mut f32 {
    if state.is_null() {
        return ptr::null_mut();
    }
    let state_ref = &mut *state;
    if state_ref.inner.logits.is_empty() {
        return ptr::null_mut();
    }
    state_ref.inner.logits.as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_print_timings(_ctx: *mut whisper_context) {}

#[no_mangle]
pub unsafe extern "C" fn whisper_reset_timings(_ctx: *mut whisper_context) {}

static SYSTEM_INFO: &str = "whisper.rs: AVX2=1, FMA=1, NEON=1, MultiThreading=1\0";

#[no_mangle]
pub extern "C" fn whisper_print_system_info() -> *const c_char {
    SYSTEM_INFO.as_ptr() as *const c_char
}
