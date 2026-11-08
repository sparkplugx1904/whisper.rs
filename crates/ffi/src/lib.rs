#![allow(clippy::missing_safety_doc, clippy::manual_c_str_literals)]
//! C-ABI Compatibility Layer exporting libwhisper symbols matching whisper.h.
//! Allows drop-in replacement for Go, Python, Ruby, and C/C++ applications.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use std::sync::Arc;
use whisper_rs::model::header::ModelType;
use whisper_rs::model::vocab::{lang_id, lang_str};
use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams, WhisperState};

/// Raw pointer wrapper for C callbacks stored in `Send + Sync` closures.
struct SendPtr<T>(*mut T);

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtr<T> {}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }

    unsafe fn as_ref(&self) -> &T {
        &*self.0
    }
}

fn merge_timings(dst: &mut whisper_rs::Timings, src: &whisper_rs::Timings) {
    dst.mel_us += src.mel_us;
    dst.sample_us += src.sample_us;
    dst.encode_us += src.encode_us;
    dst.decode_us += src.decode_us;
    dst.batchd_us += src.batchd_us;
    dst.prompt_us += src.prompt_us;
    dst.n_sample += src.n_sample;
    dst.n_encode += src.n_encode;
    dst.n_decode += src.n_decode;
    dst.n_batchd += src.n_batchd;
    dst.n_prompt += src.n_prompt;
    dst.n_fail_p += src.n_fail_p;
    dst.n_fail_h += src.n_fail_h;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct whisper_timings {
    pub sample_ms: f32,
    pub encode_ms: f32,
    pub decode_ms: f32,
    pub batchd_ms: f32,
    pub prompt_ms: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_vad_params {
    pub threshold: f32,
    pub min_speech_duration_ms: c_int,
    pub min_silence_duration_ms: c_int,
    pub max_speech_duration_s: f32,
    pub speech_pad_ms: c_int,
    pub samples_overlap: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct whisper_vad_context_params {
    pub n_threads: c_int,
    pub use_gpu: bool,
    pub gpu_device: c_int,
}

#[allow(non_camel_case_types)]
pub struct whisper_vad_context {
    pub params: whisper_vad_context_params,
    pub probs: Vec<f32>,
    pub model_path: Option<String>,
    pub samples_cache: Vec<f32>,
    pub silero: Option<whisper_vad::SileroVad>,
}

#[allow(non_camel_case_types)]
pub struct whisper_vad_segments {
    pub segments: Vec<whisper_vad::SpeechSegment>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ggml_log_level {
    GGML_LOG_LEVEL_NONE = 0,
    GGML_LOG_LEVEL_INFO = 1,
    GGML_LOG_LEVEL_WARN = 2,
    GGML_LOG_LEVEL_ERROR = 3,
    GGML_LOG_LEVEL_DEBUG = 4,
    GGML_LOG_LEVEL_CONT = 5,
}

#[allow(non_camel_case_types)]
pub type ggml_log_callback = Option<
    unsafe extern "C" fn(level: ggml_log_level, text: *const c_char, user_data: *mut c_void),
>;

struct LogCallbackState {
    callback: ggml_log_callback,
    user_data: *mut c_void,
}

unsafe impl Send for LogCallbackState {}
unsafe impl Sync for LogCallbackState {}

static LOG_CALLBACK: std::sync::RwLock<LogCallbackState> =
    std::sync::RwLock::new(LogCallbackState {
        callback: None,
        user_data: ptr::null_mut(),
    });

pub fn whisper_log(level: ggml_log_level, msg: &str) {
    if let Ok(state) = LOG_CALLBACK.read() {
        if let Some(cb) = state.callback {
            if let Ok(c_str) = CString::new(msg) {
                unsafe {
                    cb(level, c_str.as_ptr(), state.user_data);
                }
                return;
            }
        }
    }
    if level != ggml_log_level::GGML_LOG_LEVEL_NONE {
        eprint!("{}", msg);
    }
}

#[allow(non_camel_case_types)]
pub struct whisper_context {
    pub inner: WhisperContext,
    pub default_state: Option<Box<whisper_state>>,
    pub segment_c_strings: Vec<CString>,
    pub token_c_strings: Vec<Vec<CString>>,
    pub token_to_str_cache: Vec<CString>,
    pub timings: whisper_timings,
    pub params: whisper_context_params,
}

unsafe impl Send for whisper_context {}
unsafe impl Sync for whisper_context {}

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
    pub vad_params: whisper_vad_params,
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
        vad_params: whisper_vad_default_params(),
    }
}

#[no_mangle]
pub extern "C" fn whisper_context_default_params_by_ref() -> *mut whisper_context_params {
    let params = Box::new(whisper_context_default_params());
    Box::into_raw(params)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_free_context_params(params: *mut whisper_context_params) {
    if !params.is_null() {
        drop(Box::from_raw(params));
    }
}

#[no_mangle]
pub extern "C" fn whisper_full_default_params_by_ref(
    strategy: whisper_sampling_strategy,
) -> *mut whisper_full_params {
    let params = Box::new(whisper_full_default_params(strategy));
    Box::into_raw(params)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_free_params(params: *mut whisper_full_params) {
    if !params.is_null() {
        drop(Box::from_raw(params));
    }
}

unsafe fn init_context_from_model(
    model: WhisperModel,
    with_state: bool,
    params: whisper_context_params,
) -> *mut whisper_context {
    let inner = match WhisperContext::new(model) {
        Ok(c) => c,
        Err(_) => return ptr::null_mut(),
    };

    let default_state = if with_state {
        Some(Box::new(whisper_state {
            inner: inner.create_state(),
            segment_c_strings: Vec::new(),
            token_c_strings: Vec::new(),
        }))
    } else {
        None
    };

    let token_to_str_cache = inner
        .model
        .vocab
        .token_bytes
        .iter()
        .map(|bytes| CString::new(bytes.clone()).unwrap_or_else(|_| CString::new("").unwrap()))
        .collect();

    let ctx = Box::new(whisper_context {
        inner,
        default_state,
        segment_c_strings: Vec::new(),
        token_c_strings: Vec::new(),
        token_to_str_cache,
        timings: whisper_timings::default(),
        params,
    });

    Box::into_raw(ctx)
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub struct whisper_model_loader {
    pub context: *mut c_void,
    pub read: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> usize>,
    pub eof: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
    pub close: Option<unsafe extern "C" fn(*mut c_void)>,
}

/// Drain a `whisper_model_loader` into memory. The ABI streams the GGML file,
/// but the Rust loader indexes the whole buffer, so the bytes are collected
/// first and the loader is closed once, including on a short or failed read.
unsafe fn read_loader_bytes(loader: *mut whisper_model_loader) -> Option<Vec<u8>> {
    if loader.is_null() {
        return None;
    }
    let loader = &*loader;
    let read = loader.read?;

    let mut bytes = Vec::new();
    let mut chunk = vec![0u8; 1 << 20];

    loop {
        // The reference file/buffer loaders can return the requested size even
        // on a short read; `eof` is the authoritative end marker. Checking it
        // also prevents a loader that never returns 0 from spinning forever.
        if let Some(eof) = loader.eof {
            if eof(loader.context) {
                break;
            }
        }

        let n = read(loader.context, chunk.as_mut_ptr().cast(), chunk.len());
        if n == 0 {
            break;
        }
        if n > chunk.len() {
            if let Some(close) = loader.close {
                close(loader.context);
            }
            return None;
        }
        bytes.extend_from_slice(&chunk[..n]);
    }

    if let Some(close) = loader.close {
        close(loader.context);
    }

    Some(bytes)
}

struct LoaderReader<'a> {
    loader: &'a whisper_model_loader,
}

impl std::io::Read for LoaderReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.loader.read.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "loader has no read callback",
            )
        })?;
        let n = unsafe { read(self.loader.context, buf.as_mut_ptr().cast(), buf.len()) };
        if n > buf.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "loader returned more bytes than requested",
            ));
        }
        Ok(n)
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_with_params_no_state(
    loader: *mut whisper_model_loader,
    params: whisper_context_params,
) -> *mut whisper_context {
    if loader.is_null() {
        return ptr::null_mut();
    }

    let loader_ref = &*loader;
    let mut reader = LoaderReader { loader: loader_ref };
    let result = WhisperModel::load_from_reader(&mut reader, || {
        loader_ref.eof.is_some_and(|eof| eof(loader_ref.context))
    });

    if let Some(close) = loader_ref.close {
        close(loader_ref.context);
    }

    let model = match result {
        Ok(model) => model,
        Err(_) => return ptr::null_mut(),
    };

    init_context_from_model(model, false, params)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_with_params(
    loader: *mut whisper_model_loader,
    params: whisper_context_params,
) -> *mut whisper_context {
    let ctx = whisper_init_with_params_no_state(loader, params);
    if ctx.is_null() {
        return ctx;
    }

    let state = whisper_init_state(ctx);
    if state.is_null() {
        whisper_free(ctx);
        return ptr::null_mut();
    }
    (*ctx).default_state = Some(Box::from_raw(state));

    ctx
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init(loader: *mut whisper_model_loader) -> *mut whisper_context {
    whisper_init_with_params(loader, whisper_context_default_params())
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_no_state(
    loader: *mut whisper_model_loader,
) -> *mut whisper_context {
    whisper_init_with_params_no_state(loader, whisper_context_default_params())
}

#[no_mangle]
pub unsafe extern "C" fn whisper_ctx_init_openvino_encoder_with_state(
    _ctx: *mut whisper_context,
    _state: *mut whisper_state,
    _model_path: *const c_char,
    _device: *const c_char,
    _cache_dir: *const c_char,
) -> c_int {
    // whisper.cpp returns 1 when built without OpenVINO, and this port never
    // links OpenVINO. Callers use a non-zero return to keep the CPU path.
    1
}

#[no_mangle]
pub unsafe extern "C" fn whisper_ctx_init_openvino_encoder(
    _ctx: *mut whisper_context,
    _model_path: *const c_char,
    _device: *const c_char,
    _cache_dir: *const c_char,
) -> c_int {
    1
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_file_with_params(
    path: *const c_char,
    params: whisper_context_params,
) -> *mut whisper_context {
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

    init_context_from_model(model, true, params)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_file_with_params_no_state(
    path: *const c_char,
    params: whisper_context_params,
) -> *mut whisper_context {
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

    init_context_from_model(model, false, params)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_file(path: *const c_char) -> *mut whisper_context {
    whisper_init_from_file_with_params(path, whisper_context_default_params())
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_file_no_state(
    path: *const c_char,
) -> *mut whisper_context {
    whisper_init_from_file_with_params_no_state(path, whisper_context_default_params())
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_buffer_with_params(
    buffer: *mut c_void,
    buffer_size: usize,
    params: whisper_context_params,
) -> *mut whisper_context {
    if buffer.is_null() || buffer_size == 0 {
        return ptr::null_mut();
    }

    let slice = std::slice::from_raw_parts(buffer as *const u8, buffer_size);
    let model = match WhisperModel::load_from_bytes(slice.to_vec()) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };

    init_context_from_model(model, true, params)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_buffer_with_params_no_state(
    buffer: *mut c_void,
    buffer_size: usize,
    params: whisper_context_params,
) -> *mut whisper_context {
    if buffer.is_null() || buffer_size == 0 {
        return ptr::null_mut();
    }

    let slice = std::slice::from_raw_parts(buffer as *const u8, buffer_size);
    let model = match WhisperModel::load_from_bytes(slice.to_vec()) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };

    init_context_from_model(model, false, params)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_buffer(
    buffer: *mut c_void,
    buffer_size: usize,
) -> *mut whisper_context {
    whisper_init_from_buffer_with_params(buffer, buffer_size, whisper_context_default_params())
}

#[no_mangle]
pub unsafe extern "C" fn whisper_init_from_buffer_no_state(
    buffer: *mut c_void,
    buffer_size: usize,
) -> *mut whisper_context {
    whisper_init_from_buffer_with_params_no_state(
        buffer,
        buffer_size,
        whisper_context_default_params(),
    )
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

unsafe fn wrap_progress_callback(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    params: &whisper_full_params,
) -> Option<whisper_rs::params::ProgressCallback> {
    if params.progress_callback.is_null() {
        return None;
    }
    let callback = SendPtr(params.progress_callback as *mut c_void);
    let ctx_ptr = SendPtr(ctx);
    let state_ptr = SendPtr(state);
    let user_data = SendPtr(params.progress_callback_user_data);

    Some(Arc::new(move |progress: i32| {
        type Cb = extern "C" fn(*mut whisper_context, *mut whisper_state, c_int, *mut c_void);
        let cb: Cb = std::mem::transmute(callback.get());
        cb(ctx_ptr.get(), state_ptr.get(), progress, user_data.get());
    }))
}

unsafe fn wrap_abort_callback(params: &whisper_full_params) -> Option<whisper_rs::AbortCallback> {
    if params.abort_callback.is_null() {
        return None;
    }
    let callback = SendPtr(params.abort_callback as *mut c_void);
    let user_data = SendPtr(params.abort_callback_user_data);

    Some(Arc::new(move || {
        type Cb = extern "C" fn(*mut c_void) -> bool;
        let cb: Cb = std::mem::transmute(callback.get());
        cb(user_data.get())
    }))
}

unsafe fn wrap_encoder_begin_callback(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    params: &whisper_full_params,
) -> Option<whisper_rs::EncoderBeginCallback> {
    if params.encoder_begin_callback.is_null() {
        return None;
    }
    let callback = SendPtr(params.encoder_begin_callback as *mut c_void);
    let ctx_ptr = SendPtr(ctx);
    let state_ptr = SendPtr(state);
    let user_data = SendPtr(params.encoder_begin_callback_user_data);

    Some(Arc::new(move || {
        type Cb = extern "C" fn(*mut whisper_context, *mut whisper_state, *mut c_void) -> bool;
        let cb: Cb = std::mem::transmute(callback.get());
        cb(ctx_ptr.get(), state_ptr.get(), user_data.get())
    }))
}

unsafe fn wrap_logits_callback(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    params: &whisper_full_params,
) -> Option<whisper_rs::LogitsCallback> {
    if params.logits_filter_callback.is_null() {
        return None;
    }
    let callback = SendPtr(params.logits_filter_callback as *mut c_void);
    let ctx_ptr = SendPtr(ctx);
    let state_ptr = SendPtr(state);
    let user_data = SendPtr(params.logits_filter_callback_user_data);

    Some(Arc::new(move |tokens: &[i32], logits: &mut [f32]| {
        type Cb = extern "C" fn(
            *mut whisper_context,
            *mut whisper_state,
            *const whisper_token_data,
            c_int,
            *mut f32,
            *mut c_void,
        );
        let cb: Cb = std::mem::transmute(callback.get());
        let token_data: Vec<whisper_token_data> = tokens
            .iter()
            .map(|&id| whisper_token_data {
                id,
                tid: id,
                p: 0.0,
                plog: 0.0,
                pt: 0.0,
                ptsum: 0.0,
                t0: 0,
                t1: 0,
                t_dtw: 0,
                vlen: 0.0,
            })
            .collect();
        cb(
            ctx_ptr.get(),
            state_ptr.get(),
            token_data.as_ptr(),
            token_data.len() as c_int,
            logits.as_mut_ptr(),
            user_data.get(),
        );
    }))
}

/// Translate the C parameter struct into the engine representation.
///
/// Kept separate from `whisper_full_with_state` so `whisper_full_parallel` can
/// apply the same mapping to every chunk instead of hand-picking fields.
unsafe fn build_rust_params(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    params: &whisper_full_params,
) -> WhisperParams {
    let on_progress = wrap_progress_callback(ctx, state, params);
    let abort_callback = wrap_abort_callback(params);
    let on_encoder_begin = wrap_encoder_begin_callback(ctx, state, params);
    let on_logits = wrap_logits_callback(ctx, state, params);
    let vad_model_path = if params.vad_model.is_null() {
        None
    } else {
        CStr::from_ptr(params.vad_model)
            .to_str()
            .ok()
            .map(|path| path.to_string())
    };

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

    let raw_prompt_tokens = if !params.prompt_tokens.is_null() && params.prompt_n_tokens > 0 {
        let slice =
            std::slice::from_raw_parts(params.prompt_tokens, params.prompt_n_tokens as usize);
        slice.to_vec()
    } else {
        Vec::new()
    };

    let beam_sz = if params.strategy == whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH {
        params.beam_search.beam_size.max(1) as usize
    } else {
        1
    };

    // `detect_language` overrides an explicit language, matching the C API
    // contract where it forces auto-detection for this call.
    let lang_str = if params.detect_language {
        "auto".to_string()
    } else {
        lang_str
    };

    let suppress_regex = if !params.suppress_regex.is_null() {
        CStr::from_ptr(params.suppress_regex)
            .to_str()
            .ok()
            .map(|s| s.to_string())
    } else {
        None
    };

    // Every field the C struct declares has to reach the engine, otherwise the
    // caller sees a flag "accepted" and silently ignored. `max_len` and
    // `split_on_word` used to be dropped here, which is exactly the complaint in
    // ggml-org/whisper.cpp#460 for the Go bindings.
    WhisperParams {
        language: lang_str,
        translate: params.translate,
        temperature: params.temperature,
        temperature_inc: params.temperature_inc,
        beam_size: beam_sz,
        no_timestamps: params.no_timestamps,
        suppress_blank: params.suppress_blank,
        suppress_nst: params.suppress_nst,
        compression_ratio_threshold: params.entropy_thold as f64,
        logprob_threshold: params.logprob_thold as f64,
        no_speech_threshold: params.no_speech_thold as f64,
        entropy_thold: params.entropy_thold,
        initial_prompt: prompt_str,
        prompt_tokens: raw_prompt_tokens,
        no_context: params.no_context,
        carry_initial_prompt: params.carry_initial_prompt,
        max_len: params.max_len.max(0) as usize,
        split_on_word: params.split_on_word,
        max_initial_ts: params.max_initial_ts,
        length_penalty: params.length_penalty,
        single_segment: params.single_segment,
        suppress_regex,
        audio_ctx: params.audio_ctx.max(0) as usize,
        offset_ms: params.offset_ms,
        duration_ms: params.duration_ms,
        tdrz_enable: params.tdrz_enable,
        n_threads: params.n_threads.max(1) as usize,
        max_tokens: if params.max_tokens > 0 {
            params.max_tokens as usize
        } else {
            0
        },
        vad_enable: params.vad,
        vad_threshold: params.vad_params.threshold,
        vad_min_speech_duration_ms: params.vad_params.min_speech_duration_ms.max(0) as usize,
        vad_min_silence_duration_ms: params.vad_params.min_silence_duration_ms.max(0) as usize,
        vad_speech_pad_ms: params.vad_params.speech_pad_ms.max(0) as usize,
        vad_model_path,
        on_progress,
        abort_callback,
        on_encoder_begin,
        on_logits,
        // `grammar_rules` is a parsed C rule table, which this port does not
        // consume; callers pass GBNF source through the Rust API instead.
        ..Default::default()
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

    let rust_params = build_rust_params(ctx, state, &params);

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
            let bytes = context
                .inner
                .model
                .vocab
                .id_to_token_bytes(tok_id)
                .unwrap_or(&[]);
            tok_cstrs
                .push(CString::new(bytes.to_vec()).unwrap_or_else(|_| CString::new("").unwrap()));
        }
        state_ref.token_c_strings.push(tok_cstrs);
    }
    // The callback contract is `n_new` new segments since the previous call.
    // This path delivers after decoding, so all segments are new at once; a
    // single call lets the canonical `n_segments - n_new` index address them.
    let n_segments = state_ref.inner.segments.len();
    if !params.new_segment_callback.is_null() && n_segments > 0 {
        type NewSegCb = extern "C" fn(*mut whisper_context, *mut whisper_state, c_int, *mut c_void);
        let cb: NewSegCb = std::mem::transmute(params.new_segment_callback);
        cb(
            ctx,
            state,
            n_segments as c_int,
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
    let state_ptr = match (*ctx).default_state.as_mut() {
        Some(st) => st.as_mut() as *mut whisper_state,
        None => return -1,
    };
    whisper_full_with_state(ctx, state_ptr, params, samples, n_samples)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_parallel(
    ctx: *mut whisper_context,
    params: whisper_full_params,
    samples: *const f32,
    n_samples: c_int,
    n_processors: c_int,
) -> c_int {
    if ctx.is_null() || samples.is_null() || n_samples <= 0 {
        return -1;
    }

    // Resource guard: each processor owns a KV cache, workspace, and segment
    // buffer. Upstream leaves this caller-controlled; cap it like the previous
    // implementation so a bad value cannot spawn thousands of threads.
    let n_procs = n_processors.clamp(1, 16) as usize;
    if n_procs == 1 {
        return whisper_full(ctx, params, samples, n_samples);
    }

    // VAD rewrites the sample stream before chunking in whisper.cpp. The engine
    // keeps VAD internal to `full()`, so falling back to the serial path keeps
    // the transcript correct instead of splitting a stream the engine has not
    // produced.
    if params.vad {
        return whisper_full(ctx, params, samples, n_samples);
    }

    let context = &*ctx;
    let sample_slice = std::slice::from_raw_parts(samples, n_samples as usize);
    let offset_samples = (16000i64 * params.offset_ms.max(0) as i64 / 1000) as usize;

    if offset_samples >= sample_slice.len() {
        return -1;
    }

    // whisper.cpp divides the post-offset audio into equal chunks and gives the
    // final worker every remaining sample. The main thread takes the first
    // chunk (including the offset samples) so its timestamps are absolute.
    let n_samples_per_processor = (sample_slice.len() - offset_samples) / n_procs;
    // The engine ignores inputs shorter than 100 ms; splitting below that
    // threshold would silently return an empty transcript for every worker.
    if n_samples_per_processor < 1600 {
        return whisper_full(ctx, params, samples, n_samples);
    }

    let default_state_ref = match (*ctx).default_state.as_mut() {
        Some(st) => st.as_mut(),
        None => return -1,
    };

    let main_params = build_rust_params(ctx, default_state_ref as *mut whisper_state, &params);
    let params_ptr = SendPtr(&params as *const whisper_full_params as *mut whisper_full_params);
    let ctx_ptr = SendPtr(ctx);

    let main_len = offset_samples + n_samples_per_processor;
    let main_samples = &sample_slice[..main_len];

    let (main_result, worker_results) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n_procs.saturating_sub(1));

        for i in 0..n_procs - 1 {
            let start = offset_samples + (i + 1) * n_samples_per_processor;
            let end = if i == n_procs - 2 {
                sample_slice.len()
            } else {
                start + n_samples_per_processor
            };
            let chunk = &sample_slice[start..end];
            let inner = &context.inner;

            handles.push(scope.spawn(move || {
                let mut state = whisper_state {
                    inner: inner.create_state(),
                    segment_c_strings: Vec::new(),
                    token_c_strings: Vec::new(),
                };
                let state_ptr = &mut state as *mut whisper_state;
                let mut params_cur =
                    unsafe { build_rust_params(ctx_ptr.get(), state_ptr, params_ptr.as_ref()) };
                params_cur.offset_ms = 0;
                params_cur.on_new_segment = None;
                params_cur.on_progress = None;

                inner
                    .full(&mut state.inner, &params_cur, chunk)
                    .map(|_| {
                        let segments = state.inner.segments;
                        let timings = state.inner.timings;
                        (segments, timings)
                    })
                    .map_err(|_| ())
            }));
        }

        let main_result = context
            .inner
            .full(&mut default_state_ref.inner, &main_params, main_samples)
            .map_err(|_| ());

        let worker_results: Vec<Result<(Vec<whisper_rs::Segment>, whisper_rs::Timings), ()>> =
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or(Err(())))
                .collect();

        (main_result, worker_results)
    });

    if main_result.is_err() {
        return -2;
    }

    for (i, result) in worker_results.into_iter().enumerate() {
        let (segments, timings) = match result {
            Ok(result) => result,
            Err(_) => continue,
        };
        merge_timings(&mut default_state_ref.inner.timings, &timings);

        // Worker i (0-based here) processed the chunk starting at
        // offset + (i + 1) * n_samples_per_processor.
        let shift = (offset_samples + (i + 1) * n_samples_per_processor) as f64 / 16000.0;
        for mut segment in segments {
            segment.t0 += shift;
            segment.t1 += shift;
            for word in &mut segment.words {
                word.t0 += shift;
                word.t1 += shift;
            }

            // Keep the merged timeline monotonic, as the reference does.
            if let Some(last) = default_state_ref.inner.segments.last() {
                segment.t0 = segment.t0.max(last.t1);
            }
            default_state_ref.inner.segments.push(segment);
        }
    }

    // Rebuild the C string views over the merged segment list.
    default_state_ref.segment_c_strings.clear();
    default_state_ref.token_c_strings.clear();
    for seg in &default_state_ref.inner.segments {
        let c_str = CString::new(seg.text.clone()).unwrap_or_else(|_| CString::new("").unwrap());
        default_state_ref.segment_c_strings.push(c_str);

        let mut tok_cstrs = Vec::new();
        for &tok_id in &seg.tokens {
            let bytes = context
                .inner
                .model
                .vocab
                .id_to_token_bytes(tok_id)
                .unwrap_or(&[]);
            tok_cstrs
                .push(CString::new(bytes.to_vec()).unwrap_or_else(|_| CString::new("").unwrap()));
        }
        default_state_ref.token_c_strings.push(tok_cstrs);
    }

    // The main chunk bypasses `whisper_full_with_state`; all merged segments are
    // new when this post-run callback fires, so report them in one call. The
    // canonical `n_segments - n_new` index then covers the whole list.
    let n_segments = default_state_ref.inner.segments.len();
    if !params.new_segment_callback.is_null() && n_segments > 0 {
        type NewSegCb = extern "C" fn(*mut whisper_context, *mut whisper_state, c_int, *mut c_void);
        let cb: NewSegCb = std::mem::transmute(params.new_segment_callback);
        cb(
            ctx,
            default_state_ref as *mut whisper_state,
            n_segments as c_int,
            params.new_segment_callback_user_data,
        );
    }

    0
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
    match context.default_state.as_ref() {
        Some(st) => st.inner.segments.len() as c_int,
        None => 0,
    }
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
    if let Some(state) = context.default_state.as_ref() {
        if let Some(seg) = state.inner.segments.get(i_segment as usize) {
            return (seg.t0 * 100.0).round() as i64;
        }
    }
    0
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
    if let Some(state) = context.default_state.as_ref() {
        if let Some(seg) = state.inner.segments.get(i_segment as usize) {
            return (seg.t1 * 100.0).round() as i64;
        }
    }
    0
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
    if let Some(state) = context.default_state.as_ref() {
        if let Some(c_str) = state.segment_c_strings.get(i_segment as usize) {
            return c_str.as_ptr();
        }
    }
    ptr::null()
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
    if let Some(state) = context.default_state.as_ref() {
        if let Some(seg) = state.inner.segments.get(i_segment as usize) {
            return seg.tokens.len() as c_int;
        }
    }
    0
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
    if let Some(state) = context.default_state.as_ref() {
        if let Some(seg) = state.inner.segments.get(i_segment as usize) {
            return seg.tokens.get(i_token as usize).copied().unwrap_or(0);
        }
    }
    0
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
    if let Some(state) = context.default_state.as_ref() {
        if let Some(seg) = state.token_c_strings.get(i_segment as usize) {
            if let Some(c_str) = seg.get(i_token as usize) {
                return c_str.as_ptr();
            }
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
    let state_ptr = match (*ctx).default_state.as_mut() {
        Some(s) => s.as_mut() as *mut whisper_state,
        None => {
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
            }
        }
    };
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
    match context.default_state.as_ref() {
        Some(s) => s.inner.lang_id as c_int,
        None => 0,
    }
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
    match context.default_state.as_ref() {
        Some(s) => {
            (s.inner.encoder_output.len()
                / context.inner.model.hparams.n_audio_state.max(1) as usize) as c_int
        }
        None => 0,
    }
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
    if ctx.is_null() || text.is_null() {
        return 0;
    }
    let context = &*ctx;
    let text_bytes = CStr::from_ptr(text).to_bytes();
    let token_ids = context.inner.tokenizer.encode_bytes(text_bytes);
    let count = token_ids.len() as c_int;

    // whisper.h promises a negative result carrying the required size instead
    // of silently truncating. A null destination is only valid in count mode.
    if n_max_tokens < 0 || tokens.is_null() || count > n_max_tokens {
        return -count;
    }

    for (i, &tok_id) in token_ids.iter().enumerate() {
        *tokens.add(i) = tok_id;
    }
    count
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_count(
    ctx: *mut whisper_context,
    text: *const c_char,
) -> c_int {
    -whisper_tokenize(ctx, text, ptr::null_mut(), 0)
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
pub unsafe extern "C" fn whisper_pcm_to_mel_with_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    samples: *const f32,
    n_samples: c_int,
    _n_threads: c_int,
) -> c_int {
    if ctx.is_null() || state.is_null() || samples.is_null() || n_samples <= 0 {
        return -1;
    }
    let t_start = std::time::Instant::now();
    let context = &*ctx;
    let state_ref = &mut *state;
    let sample_slice = std::slice::from_raw_parts(samples, n_samples as usize);
    let mel = whisper_rs::audio::compute_log_mel_spectrogram(
        sample_slice,
        &context.inner.model.filters.data,
        context.inner.model.filters.n_mel,
    );
    state_ref.inner.mel = Some(mel);
    state_ref.inner.timings.mel_us += t_start.elapsed().as_micros() as u64;
    0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_pcm_to_mel(
    ctx: *mut whisper_context,
    samples: *const f32,
    n_samples: c_int,
    n_threads: c_int,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    match (*ctx).default_state.as_mut() {
        Some(st) => whisper_pcm_to_mel_with_state(ctx, st.as_mut(), samples, n_samples, n_threads),
        None => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_set_mel_with_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    data: *const f32,
    n_len: c_int,
    n_mel: c_int,
) -> c_int {
    if ctx.is_null() || state.is_null() || data.is_null() || n_len <= 0 || n_mel <= 0 {
        return -1;
    }
    if n_len.checked_mul(n_mel).is_none() {
        return -1;
    }
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
    let state_ref = &mut *state;
    state_ref.inner.mel = Some(mel);
    0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_set_mel(
    ctx: *mut whisper_context,
    data: *const f32,
    n_len: c_int,
    n_mel: c_int,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    match (*ctx).default_state.as_mut() {
        Some(st) => whisper_set_mel_with_state(ctx, st.as_mut(), data, n_len, n_mel),
        None => -1,
    }
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
    offset_ms: c_int,
    _n_threads: c_int,
    lang_probs: *mut f32,
) -> c_int {
    if ctx.is_null() || offset_ms < 0 {
        return -1;
    }
    let state_ptr = if !state.is_null() {
        state
    } else {
        match (*ctx).default_state.as_mut() {
            Some(st) => &mut **st as *mut whisper_state,
            None => return -1,
        }
    };
    let context = &*ctx;
    let st = &mut *state_ptr;

    let seek = (offset_ms / 10) as usize;
    let mel = match &st.inner.mel {
        Some(m) => m,
        None => return -1,
    };

    if seek >= mel.n_len_org {
        return -2;
    }

    let n_audio_ctx = context.inner.model.hparams.n_audio_ctx as usize;
    let mel_chunk = mel.get_chunk(seek, n_audio_ctx);
    st.inner.encoder_output = context.inner.encoder.forward(&mel_chunk);
    // The encoder output changed, so cross-attention projections cached from a
    // previous window must not be reused.
    st.inner.cache.invalidate_cross();

    let max_langs = whisper_lang_max_id() as usize + 1;
    let mut probs = vec![0.0f32; max_langs];
    let (_, best_idx, _) = context
        .inner
        .detect_language_probs_with_enc(&st.inner.encoder_output, Some(&mut probs));

    if !lang_probs.is_null() {
        let slice = std::slice::from_raw_parts_mut(lang_probs, max_langs);
        slice.copy_from_slice(&probs);
    }

    best_idx as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_encode_with_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    offset: c_int,
    _n_threads: c_int,
) -> c_int {
    if ctx.is_null() || state.is_null() {
        return -1;
    }
    let t_start = std::time::Instant::now();
    let context = &*ctx;
    let state_ref = &mut *state;
    let n_audio_ctx = context.inner.model.hparams.n_audio_ctx as usize;
    let mel_chunk = if let Some(ref mel) = state_ref.inner.mel {
        mel.get_chunk(offset.max(0) as usize, n_audio_ctx)
    } else {
        vec![0.0f32; context.inner.model.filters.n_mel * 2 * n_audio_ctx]
    };

    let enc_out = context.inner.encoder.forward(&mel_chunk);
    state_ref.inner.encoder_output = enc_out;
    // New audio invalidates the cross-attention projections that the previous
    // window left in the cache, even if both windows have the same length.
    state_ref.inner.cache.invalidate_cross();
    state_ref
        .inner
        .timings
        .add_encode(t_start.elapsed().as_micros() as u64);
    0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_encode(
    ctx: *mut whisper_context,
    offset: c_int,
    n_threads: c_int,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    match (*ctx).default_state.as_mut() {
        Some(st) => whisper_encode_with_state(ctx, st.as_mut(), offset, n_threads),
        None => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_decode_with_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    tokens: *const c_int,
    n_tokens: c_int,
    n_past: c_int,
    _n_threads: c_int,
) -> c_int {
    if ctx.is_null() || state.is_null() || tokens.is_null() || n_tokens <= 0 {
        return -1;
    }
    let t_start = std::time::Instant::now();
    let context = &*ctx;
    let state_ref = &mut *state;
    let n_vocab = context.inner.model.hparams.n_vocab;
    let token_slice = std::slice::from_raw_parts(tokens, n_tokens as usize);
    for &tok in token_slice {
        if tok < 0 || tok >= n_vocab {
            return -1;
        }
    }
    let tok_vec: Vec<i32> = token_slice.to_vec();

    // The KV cache is a slot arena; the C API drives a single sequence, so the
    // caller-supplied history length applies to slot 0.
    state_ref.inner.cache.t_past[0] = n_past.max(0) as usize;
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let logits = context.inner.decoder.decode(
            &tok_vec,
            &state_ref.inner.encoder_output,
            &mut state_ref.inner.cache,
            &mut state_ref.inner.workspace,
        );
        // Copy out of the reusable workspace so the C getter keeps returning a
        // stable pointer for the lifetime of the state.
        state_ref.inner.logits.clear();
        state_ref.inner.logits.extend_from_slice(logits);
    }));
    match res {
        Ok(()) => {
            state_ref
                .inner
                .timings
                .add_decode(t_start.elapsed().as_micros() as u64);
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_decode(
    ctx: *mut whisper_context,
    tokens: *const c_int,
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    match (*ctx).default_state.as_mut() {
        Some(st) => {
            whisper_decode_with_state(ctx, st.as_mut(), tokens, n_tokens, n_past, n_threads)
        }
        None => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_get_logits(ctx: *mut whisper_context) -> *mut f32 {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    match (*ctx).default_state.as_mut() {
        Some(st) => whisper_get_logits_from_state(st.as_mut()),
        None => ptr::null_mut(),
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
pub unsafe extern "C" fn whisper_get_timings(ctx: *mut whisper_context) -> *mut whisper_timings {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    let t = (*ctx)
        .default_state
        .as_ref()
        .map(|state| state.inner.timings)
        .unwrap_or_default();
    let per_run = |total_us: u64, runs: u32| 1e-3f32 * total_us as f32 / runs.max(1) as f32;

    (*ctx).timings = whisper_timings {
        sample_ms: per_run(t.sample_us, t.n_sample),
        encode_ms: per_run(t.encode_us, t.n_encode),
        decode_ms: per_run(t.decode_us, t.n_decode),
        batchd_ms: per_run(t.batchd_us, t.n_batchd),
        prompt_ms: per_run(t.prompt_us, t.n_prompt),
    };
    &mut (*ctx).timings as *mut whisper_timings
}

#[no_mangle]
pub unsafe extern "C" fn whisper_print_timings(ctx: *mut whisper_context) {
    if ctx.is_null() {
        return;
    }
    let t = (*ctx)
        .default_state
        .as_ref()
        .map(|state| state.inner.timings)
        .unwrap_or_default();
    let ms = |total_us: u64| 1e-3f32 * total_us as f32;
    let per_run = |total_us: u64, runs: u32| ms(total_us) / runs.max(1) as f32;

    let msg = format!(
        "\nwhisper_print_timings:     load time =      0.00 ms\n\
         whisper_print_timings:     fallbacks = {:5} p / {:5} h\n\
         whisper_print_timings:      mel time = {:9.2} ms\n\
         whisper_print_timings:   sample time = {:9.2} ms / {:5} runs ( {:8.2} ms per run)\n\
         whisper_print_timings:   encode time = {:9.2} ms / {:5} runs ( {:8.2} ms per run)\n\
         whisper_print_timings:   decode time = {:9.2} ms / {:5} runs ( {:8.2} ms per run)\n\
         whisper_print_timings:   batchd time = {:9.2} ms / {:5} runs ( {:8.2} ms per run)\n\
         whisper_print_timings:   prompt time = {:9.2} ms / {:5} runs ( {:8.2} ms per run)\n",
        t.n_fail_p,
        t.n_fail_h,
        ms(t.mel_us),
        ms(t.sample_us),
        t.n_sample,
        per_run(t.sample_us, t.n_sample),
        ms(t.encode_us),
        t.n_encode,
        per_run(t.encode_us, t.n_encode),
        ms(t.decode_us),
        t.n_decode,
        per_run(t.decode_us, t.n_decode),
        ms(t.batchd_us),
        t.n_batchd,
        per_run(t.batchd_us, t.n_batchd),
        ms(t.prompt_us),
        t.n_prompt,
        per_run(t.prompt_us, t.n_prompt),
    );
    whisper_log(ggml_log_level::GGML_LOG_LEVEL_INFO, &msg);
}

#[no_mangle]
pub unsafe extern "C" fn whisper_reset_timings(ctx: *mut whisper_context) {
    if !ctx.is_null() {
        (*ctx).timings = whisper_timings::default();
        if let Some(state) = (*ctx).default_state.as_mut() {
            state.inner.reset_timings();
        }
    }
}

#[no_mangle]
pub extern "C" fn whisper_log_set(log_callback: ggml_log_callback, user_data: *mut c_void) {
    if let Ok(mut state) = LOG_CALLBACK.write() {
        state.callback = log_callback;
        state.user_data = user_data;
    }
}

fn system_info_string() -> String {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let features = format!(
        "AVX = {} | AVX2 = {} | FMA = {} | NEON = 0",
        std::arch::is_x86_feature_detected!("avx") as u8,
        std::arch::is_x86_feature_detected!("avx2") as u8,
        std::arch::is_x86_feature_detected!("fma") as u8
    );

    #[cfg(target_arch = "aarch64")]
    let features = "AVX = 0 | AVX2 = 0 | FMA = 0 | NEON = 1".to_string();

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    let features = "AVX = 0 | AVX2 = 0 | FMA = 0 | NEON = 0".to_string();

    format!("WHISPER : {}", features)
}

static SYSTEM_INFO: std::sync::LazyLock<CString> = std::sync::LazyLock::new(|| {
    CString::new(system_info_string()).expect("system info must not contain NUL bytes")
});

#[no_mangle]
pub extern "C" fn whisper_print_system_info() -> *const c_char {
    SYSTEM_INFO.as_ptr()
}

#[no_mangle]
pub extern "C" fn whisper_version() -> *const c_char {
    static VERSION: std::sync::LazyLock<CString> =
        std::sync::LazyLock::new(|| CString::new(env!("CARGO_PKG_VERSION")).unwrap());
    VERSION.as_ptr()
}

#[no_mangle]
pub extern "C" fn whisper_lang_str_full(id: c_int) -> *const c_char {
    if id < 0 {
        return ptr::null();
    }
    static CACHED_FULL_LANGS: std::sync::LazyLock<Vec<CString>> = std::sync::LazyLock::new(|| {
        whisper_rs::model::vocab::LANGUAGES
            .iter()
            .map(|(_, name)| CString::new(*name).unwrap())
            .collect()
    });
    if (id as usize) < CACHED_FULL_LANGS.len() {
        return CACHED_FULL_LANGS[id as usize].as_ptr();
    }
    ptr::null()
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_lang_id_from_state(state: *mut whisper_state) -> c_int {
    if state.is_null() {
        return -1;
    }
    let state_ref = &*state;
    state_ref.inner.lang_id as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_no_speech_prob(
    ctx: *mut whisper_context,
    i_segment: c_int,
) -> f32 {
    if ctx.is_null() || i_segment < 0 {
        return 0.0;
    }
    let context = &*ctx;
    if let Some(state) = &context.default_state {
        if let Some(seg) = state.inner.segments.get(i_segment as usize) {
            return seg.no_speech_prob;
        }
    }
    0.0
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_no_speech_prob_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
) -> f32 {
    if state.is_null() || i_segment < 0 {
        return 0.0;
    }
    let state_ref = &*state;
    if let Some(seg) = state_ref.inner.segments.get(i_segment as usize) {
        seg.no_speech_prob
    } else {
        0.0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_speaker_turn_next_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
) -> bool {
    if state.is_null() || i_segment < 0 {
        return false;
    }
    let state_ref = &*state;
    if let Some(seg) = state_ref.inner.segments.get(i_segment as usize) {
        seg.text.contains("[SPEAKER_TURN]")
    } else {
        false
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_segment_speaker_turn_next(
    ctx: *mut whisper_context,
    i_segment: c_int,
) -> bool {
    if ctx.is_null() {
        return false;
    }
    let context = &*ctx;
    match &context.default_state {
        Some(s) => {
            let state_ptr = (s.as_ref() as *const whisper_state) as *mut whisper_state;
            whisper_full_get_segment_speaker_turn_next_from_state(state_ptr, i_segment)
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_n_tokens_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
) -> c_int {
    if state.is_null() || i_segment < 0 {
        return 0;
    }
    let state_ref = &*state;
    if let Some(seg) = state_ref.inner.segments.get(i_segment as usize) {
        seg.tokens.len() as c_int
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_id_from_state(
    state: *mut whisper_state,
    i_segment: c_int,
    i_token: c_int,
) -> c_int {
    if state.is_null() || i_segment < 0 || i_token < 0 {
        return -1;
    }
    let state_ref = &*state;
    if let Some(seg) = state_ref.inner.segments.get(i_segment as usize) {
        seg.tokens.get(i_token as usize).copied().unwrap_or(-1)
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_token_text_from_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    i_segment: c_int,
    i_token: c_int,
) -> *const c_char {
    let tok_id = whisper_full_get_token_id_from_state(state, i_segment, i_token);
    if tok_id < 0 {
        return ptr::null();
    }
    whisper_token_to_str(ctx, tok_id)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_n_vad_segments_from_state(
    state: *mut whisper_state,
) -> c_int {
    if state.is_null() {
        return 0;
    }
    let state_ref = &*state;
    state_ref.inner.vad_segments.len() as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_n_vad_segments(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    match &context.default_state {
        Some(s) => s.inner.vad_segments.len() as c_int,
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_vad_segment_t0_from_state(
    state: *mut whisper_state,
    i: c_int,
) -> i64 {
    if state.is_null() || i < 0 {
        return 0;
    }
    let state_ref = &*state;
    if let Some(&(t0, _)) = state_ref.inner.vad_segments.get(i as usize) {
        (t0 as f64 * 0.02 * 100.0).round() as i64
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_vad_segment_t0(
    ctx: *mut whisper_context,
    i: c_int,
) -> i64 {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    match &context.default_state {
        Some(s) => {
            let state_ptr = (s.as_ref() as *const whisper_state) as *mut whisper_state;
            whisper_full_get_vad_segment_t0_from_state(state_ptr, i)
        }
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_vad_segment_t1_from_state(
    state: *mut whisper_state,
    i: c_int,
) -> i64 {
    if state.is_null() || i < 0 {
        return 0;
    }
    let state_ref = &*state;
    if let Some(&(_, t1)) = state_ref.inner.vad_segments.get(i as usize) {
        (t1 as f64 * 0.02 * 100.0).round() as i64
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_full_get_vad_segment_t1(
    ctx: *mut whisper_context,
    i: c_int,
) -> i64 {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    match &context.default_state {
        Some(s) => {
            let state_ptr = (s.as_ref() as *const whisper_state) as *mut whisper_state;
            whisper_full_get_vad_segment_t1_from_state(state_ptr, i)
        }
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_vocab(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_vocab
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_audio_ctx(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_audio_ctx
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_audio_state(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_audio_state
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_audio_head(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_audio_head
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_audio_layer(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_audio_layer
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_text_ctx(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_text_ctx
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_text_state(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_text_state
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_text_head(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_text_head
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_text_layer(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_text_layer
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_n_mels(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.n_mels
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_ftype(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.inner.model.hparams.ftype
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_type(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    match context.inner.model.hparams.model_type {
        ModelType::Tiny => 1,
        ModelType::Base => 2,
        ModelType::Small => 3,
        ModelType::Medium => 4,
        ModelType::Large | ModelType::LargeV3 | ModelType::LargeV3Turbo => 5,
        ModelType::Unknown => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_model_type_readable(ctx: *mut whisper_context) -> *const c_char {
    if ctx.is_null() {
        return ptr::null();
    }
    let context = &*ctx;
    static STR_TINY: &[u8] = b"tiny\0";
    static STR_BASE: &[u8] = b"base\0";
    static STR_SMALL: &[u8] = b"small\0";
    static STR_MEDIUM: &[u8] = b"medium\0";
    static STR_LARGE: &[u8] = b"large\0";
    static STR_LARGE_V3: &[u8] = b"large-v3\0";
    static STR_LARGE_TURBO: &[u8] = b"large-v3-turbo\0";
    static STR_UNKNOWN: &[u8] = b"unknown\0";

    match context.inner.model.hparams.model_type {
        ModelType::Tiny => STR_TINY.as_ptr() as *const c_char,
        ModelType::Base => STR_BASE.as_ptr() as *const c_char,
        ModelType::Small => STR_SMALL.as_ptr() as *const c_char,
        ModelType::Medium => STR_MEDIUM.as_ptr() as *const c_char,
        ModelType::Large => STR_LARGE.as_ptr() as *const c_char,
        ModelType::LargeV3 => STR_LARGE_V3.as_ptr() as *const c_char,
        ModelType::LargeV3Turbo => STR_LARGE_TURBO.as_ptr() as *const c_char,
        ModelType::Unknown => STR_UNKNOWN.as_ptr() as *const c_char,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_token_nosp(ctx: *mut whisper_context) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    let context = &*ctx;
    context.inner.model.vocab.token_nosp
}

#[no_mangle]
pub unsafe extern "C" fn whisper_n_len_from_state(state: *mut whisper_state) -> c_int {
    if state.is_null() {
        return 0;
    }
    let state_ref = &*state;
    if let Some(m) = &state_ref.inner.mel {
        m.n_len as c_int
    } else {
        0
    }
}

// -----------------------------------------------------------------------------
// Voice Activity Detection (VAD) C-API
// -----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn whisper_vad_default_params() -> whisper_vad_params {
    whisper_vad_params {
        threshold: 0.5,
        min_speech_duration_ms: 250,
        min_silence_duration_ms: 100,
        max_speech_duration_s: f32::MAX,
        speech_pad_ms: 30,
        samples_overlap: 0.1,
    }
}

#[no_mangle]
pub extern "C" fn whisper_vad_default_context_params() -> whisper_vad_context_params {
    whisper_vad_context_params {
        n_threads: 4,
        use_gpu: false,
        gpu_device: 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_init_from_file_with_params(
    path: *const c_char,
    params: whisper_vad_context_params,
) -> *mut whisper_vad_context {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(path) => path,
        Err(_) => return ptr::null_mut(),
    };
    let bytes = match std::fs::read(path_str) {
        Ok(bytes) => bytes,
        Err(_) => return ptr::null_mut(),
    };
    let silero = match whisper_vad::SileroVad::from_bytes(&bytes) {
        Ok(silero) => silero,
        Err(_) => return ptr::null_mut(),
    };

    let ctx = Box::new(whisper_vad_context {
        params,
        probs: Vec::new(),
        model_path: Some(path_str.to_string()),
        samples_cache: Vec::new(),
        silero: Some(silero),
    });

    Box::into_raw(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_init_with_params(
    loader: *mut whisper_model_loader,
    params: whisper_vad_context_params,
) -> *mut whisper_vad_context {
    let bytes = match read_loader_bytes(loader) {
        Some(bytes) if !bytes.is_empty() => bytes,
        _ => return ptr::null_mut(),
    };
    let silero = match whisper_vad::SileroVad::from_bytes(&bytes) {
        Ok(silero) => silero,
        Err(_) => return ptr::null_mut(),
    };

    let ctx = Box::new(whisper_vad_context {
        params,
        probs: Vec::new(),
        model_path: None,
        samples_cache: Vec::new(),
        silero: Some(silero),
    });

    Box::into_raw(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_free(ctx: *mut whisper_vad_context) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_detect_speech(
    ctx: *mut whisper_vad_context,
    samples: *const f32,
    n_samples: c_int,
) -> c_int {
    whisper_vad_reset_state(ctx);
    whisper_vad_detect_speech_no_reset(ctx, samples, n_samples)
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_detect_speech_no_reset(
    ctx: *mut whisper_vad_context,
    samples: *const f32,
    n_samples: c_int,
) -> c_int {
    if ctx.is_null() || samples.is_null() || n_samples <= 0 {
        return 0;
    }
    let context = &mut *ctx;
    let sample_slice = std::slice::from_raw_parts(samples, n_samples as usize);

    let probs = match context.silero.as_mut() {
        Some(silero) => silero.detect_probs(sample_slice).to_vec(),
        None => return 0,
    };

    context.probs = probs;
    context.samples_cache = sample_slice.to_vec();

    // The reference returns whether the graph ran, not whether speech was
    // present. Callers inspect `probs` for the actual decision.
    1
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_reset_state(ctx: *mut whisper_vad_context) {
    if !ctx.is_null() {
        let context = &mut *ctx;
        if let Some(silero) = context.silero.as_mut() {
            silero.reset_state();
        }
        context.probs.clear();
        context.samples_cache.clear();
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_n_probs(ctx: *mut whisper_vad_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    (*ctx).probs.len() as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_probs(ctx: *mut whisper_vad_context) -> *mut f32 {
    if ctx.is_null() || (*ctx).probs.is_empty() {
        return ptr::null_mut();
    }
    (*ctx).probs.as_mut_ptr()
}

unsafe fn c_vad_params(params: &whisper_vad_params) -> whisper_vad::VadParams {
    whisper_vad::VadParams {
        threshold: params.threshold,
        min_speech_duration_ms: params.min_speech_duration_ms.max(0) as usize,
        min_silence_duration_ms: params.min_silence_duration_ms.max(0) as usize,
        max_speech_duration_s: params.max_speech_duration_s,
        speech_pad_ms: params.speech_pad_ms.max(0) as usize,
        samples_overlap: params.samples_overlap,
        sample_rate: 16000,
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_segments_from_samples(
    ctx: *mut whisper_vad_context,
    params: whisper_vad_params,
    samples: *const f32,
    n_samples: c_int,
) -> *mut whisper_vad_segments {
    if ctx.is_null() || samples.is_null() || n_samples <= 0 {
        return ptr::null_mut();
    }

    let context = &*ctx;
    if context.silero.is_some() {
        if whisper_vad_detect_speech(ctx, samples, n_samples) == 0 {
            return ptr::null_mut();
        }
        return whisper_vad_segments_from_probs(ctx, params);
    }

    // The energy detector remains available for contexts that do not carry a
    // Silero model; it is not reachable through the file/loader constructors.
    let sample_slice = std::slice::from_raw_parts(samples, n_samples as usize);
    let rust_params = c_vad_params(&params);
    let segments = whisper_vad::detect_speech_segments(sample_slice, &rust_params);
    Box::into_raw(Box::new(whisper_vad_segments { segments }))
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_segments_from_probs(
    ctx: *mut whisper_vad_context,
    params: whisper_vad_params,
) -> *mut whisper_vad_segments {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    let context = &*ctx;
    let rust_params = c_vad_params(&params);

    if let Some(silero) = &context.silero {
        let segments =
            whisper_vad::segments_from_probs(&context.probs, &rust_params, silero.n_window());
        return Box::into_raw(Box::new(whisper_vad_segments { segments }));
    }

    if !context.samples_cache.is_empty() {
        return whisper_vad_segments_from_samples(
            ctx,
            params,
            context.samples_cache.as_ptr(),
            context.samples_cache.len() as c_int,
        );
    }

    Box::into_raw(Box::new(whisper_vad_segments {
        segments: Vec::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_segments_n_segments(
    segments: *mut whisper_vad_segments,
) -> c_int {
    if segments.is_null() {
        return 0;
    }
    let segs = &*segments;
    segs.segments.len() as c_int
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_segments_get_segment_t0(
    segments: *mut whisper_vad_segments,
    i_segment: c_int,
) -> f32 {
    if segments.is_null() || i_segment < 0 {
        return 0.0;
    }
    let segs = &*segments;
    if let Some(seg) = segs.segments.get(i_segment as usize) {
        (seg.t0 * 100.0) as f32
    } else {
        0.0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_segments_get_segment_t1(
    segments: *mut whisper_vad_segments,
    i_segment: c_int,
) -> f32 {
    if segments.is_null() || i_segment < 0 {
        return 0.0;
    }
    let segs = &*segments;
    if let Some(seg) = segs.segments.get(i_segment as usize) {
        (seg.t1 * 100.0) as f32
    } else {
        0.0
    }
}

#[no_mangle]
pub unsafe extern "C" fn whisper_vad_free_segments(segments: *mut whisper_vad_segments) {
    if !segments.is_null() {
        drop(Box::from_raw(segments));
    }
}

// =================================================================================================
// Development benchmarks matching whisper.cpp's ggml helpers.
// =================================================================================================

static BENCH_MEMCPY_STRINGS: std::sync::LazyLock<std::sync::Mutex<Vec<CString>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

static BENCH_MUL_MAT_STRINGS: std::sync::LazyLock<std::sync::Mutex<Vec<CString>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

/// Timed iterations per memcpy phase, matching `whisper.cpp`.
const BENCH_MEMCPY_N: usize = 20;
/// Bytes copied per iteration when at least one thread is requested (1 GB).
const BENCH_MEMCPY_ARR: usize = 1024;

/// Matrix sizes reported by the quantized matmul benchmark, matching `whisper.cpp`.
const BENCH_MUL_MAT_SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];
/// Upper bound on timed runs per type and size, matching `whisper.cpp`.
const BENCH_MUL_MAT_N_MAX: usize = 128;
/// Wall-clock budget after which a type stops being timed, matching `whisper.cpp`.
const BENCH_MUL_MAT_TIME_BUDGET: f64 = 1.0;

const BENCH_MUL_MAT_TYPES: [whisper_tensor::DType; 7] = [
    whisper_tensor::DType::Q4_0,
    whisper_tensor::DType::Q4_1,
    whisper_tensor::DType::Q5_0,
    whisper_tensor::DType::Q5_1,
    whisper_tensor::DType::Q8_0,
    whisper_tensor::DType::F16,
    whisper_tensor::DType::F32,
];

/// Retain a report and return a pointer that stays valid for the process lifetime.
///
/// `CString` owns its heap buffer, so growing the vector only moves the owning
/// structs and never invalidates pointers handed out by earlier calls.
fn store_c_string(
    cache: &'static std::sync::LazyLock<std::sync::Mutex<Vec<CString>>>,
    text: String,
) -> *const c_char {
    let c_text = CString::new(text).unwrap_or_else(|_| CString::new("").unwrap());
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.push(c_text);
    cache.last().unwrap().as_ptr()
}

fn allocate_memcpy_buffers(size: usize) -> (Vec<u8>, Vec<u8>) {
    let mut src = vec![0u8; size];
    for (i, byte) in src.iter_mut().enumerate() {
        *byte = i as u8;
    }
    (src, vec![0u8; size])
}

/// Mutate one source byte so the compiler cannot hoist the copy out of the loop.
fn perturb_source(buf: &mut [u8], seed: u64) {
    if buf.is_empty() {
        return;
    }
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let index = ((state >> 33) as usize) % buf.len();
    buf[index] = (state >> 17) as u8;
}

fn memcpy_checksum(bytes: &[u8]) -> f64 {
    bytes.iter().map(|&byte| byte as f64).sum()
}

fn memcpy_gbps(n: usize, size: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        (n as f64 * size as f64) / (seconds * 1e9)
    } else {
        0.0
    }
}

fn bench_memcpy_report(n_threads: c_int) -> String {
    let n = BENCH_MEMCPY_N;
    // Upstream's trick: a non-positive thread count collapses the array to zero
    // bytes, which keeps the report cheap while still exercising the code path.
    let arr = if n_threads > 0 {
        BENCH_MEMCPY_ARR
    } else {
        n_threads.max(0) as usize
    };
    let size = arr * 1_000_000;

    let mut report = String::new();
    let mut sum = 0.0f64;

    // Heat-up pass.
    {
        let (mut src, mut dst) = allocate_memcpy_buffers(size);
        dst.copy_from_slice(&src);

        let mut tsum = 0.0f64;
        for i in 0..n {
            let t0 = std::time::Instant::now();
            dst.copy_from_slice(&src);
            tsum += t0.elapsed().as_secs_f64();
            perturb_source(&mut src, i as u64);
        }

        report.push_str(&format!(
            "memcpy: {:7.2} GB/s (heat-up)\n",
            memcpy_gbps(n, size, tsum)
        ));

        std::hint::black_box(dst.as_slice());
        sum += memcpy_checksum(&dst);
    }

    // Single-thread pass.
    {
        let (mut src, mut dst) = allocate_memcpy_buffers(size);
        dst.copy_from_slice(&src);

        let mut tsum = 0.0f64;
        for i in 0..n {
            let t0 = std::time::Instant::now();
            dst.copy_from_slice(&src);
            tsum += t0.elapsed().as_secs_f64();
            perturb_source(&mut src, i as u64);
        }

        report.push_str(&format!(
            "memcpy: {:7.2} GB/s ( 1 thread)\n",
            memcpy_gbps(n, size, tsum)
        ));

        std::hint::black_box(dst.as_slice());
        sum += memcpy_checksum(&dst);
    }

    // Multi-thread passes, one report line per requested thread count.
    let k_max = n_threads.max(0) as usize;
    for k in 1..=k_max {
        let (mut src, mut dst) = allocate_memcpy_buffers(size);
        dst.copy_from_slice(&src);

        let mut src_rest: &mut [u8] = &mut src;
        let mut dst_rest: &mut [u8] = &mut dst;
        let mut parts = Vec::with_capacity(k);
        for th in 0..k {
            let start = (th as u128 * size as u128 / k as u128) as usize;
            let end = ((th + 1) as u128 * size as u128 / k as u128) as usize;
            let (src_chunk, src_tail) = src_rest.split_at_mut(end - start);
            let (dst_chunk, dst_tail) = dst_rest.split_at_mut(end - start);
            src_rest = src_tail;
            dst_rest = dst_tail;
            parts.push((src_chunk, dst_chunk));
        }

        let t0 = std::time::Instant::now();
        std::thread::scope(|scope| {
            let (last_src, last_dst) = parts.pop().expect("k is at least one");
            let last_th = parts.len();

            for (th, (src_chunk, dst_chunk)) in parts.into_iter().enumerate() {
                scope.spawn(move || {
                    for i in 0..n {
                        dst_chunk.copy_from_slice(src_chunk);
                        perturb_source(src_chunk, ((th as u64) << 32) | i as u64);
                    }
                });
            }

            for i in 0..n {
                last_dst.copy_from_slice(last_src);
                perturb_source(last_src, ((last_th as u64) << 32) | i as u64);
            }
        });
        let elapsed = t0.elapsed().as_secs_f64();

        report.push_str(&format!(
            "memcpy: {:7.2} GB/s ({:2} thread)\n",
            memcpy_gbps(n, size, elapsed),
            k
        ));

        std::hint::black_box(dst.as_slice());
        sum += memcpy_checksum(&dst);
    }

    report.push_str(&format!("sum:    {:.6}\n", sum));
    report
}

#[allow(clippy::too_many_arguments)]
fn run_quantized_matmul(
    weights: &[u8],
    dtype: whisper_tensor::DType,
    n: usize,
    b: &[f32],
    out: &mut [f32],
    n_threads: usize,
    pool: Option<&rayon::ThreadPool>,
) {
    let mut run = || {
        whisper_tensor::ops::quantized_matmul(weights, dtype, n, n, b, n, out, n_threads);
    };
    match pool {
        Some(pool) => pool.install(run),
        None => run(),
    }
}

fn bench_mul_mat_report(
    sizes: &[usize],
    n_max: usize,
    n_threads: c_int,
    time_budget: f64,
) -> String {
    let n_threads = if n_threads > 0 {
        n_threads as usize
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
    .clamp(1, 16);

    // Rayon's global pool ignores the requested thread count, so install a
    // dedicated pool for the duration of the benchmark.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .ok();

    let mut report = String::new();

    for &n in sizes {
        let mut gflops = [0.0f64; BENCH_MUL_MAT_TYPES.len()];
        let mut runs = [0usize; BENCH_MUL_MAT_TYPES.len()];

        for (k, dtype) in BENCH_MUL_MAT_TYPES.iter().enumerate() {
            let weight_values: Vec<f32> = (0..n * n)
                .map(|i| (((i * 37) % 23) as f32 - 11.0) * 0.03125)
                .collect();
            let weights = dtype
                .quantize_from_f32(&weight_values)
                .expect("benchmark weights must quantize");
            let b: Vec<f32> = (0..n * n)
                .map(|i| (((i * 13) % 17) as f32 - 8.0) * 0.0625)
                .collect();
            let mut out = vec![0.0f32; n * n];

            // Heat-up.
            run_quantized_matmul(&weights, *dtype, n, &b, &mut out, n_threads, pool.as_ref());

            let mut tsum = 0.0f64;
            let mut count = 0usize;
            for _ in 0..n_max {
                let t0 = std::time::Instant::now();
                run_quantized_matmul(&weights, *dtype, n, &b, &mut out, n_threads, pool.as_ref());
                tsum += t0.elapsed().as_secs_f64();
                count += 1;

                std::hint::black_box(out.as_slice());

                if tsum > time_budget && count >= 3 {
                    break;
                }
            }

            runs[k] = count;
            gflops[k] = if tsum > 0.0 {
                (2.0 * (n * n * n) as f64 * count as f64) / tsum * 1e-9
            } else {
                0.0
            };
        }

        report.push_str(&format!(
            "{:4} x {:4}: Q4_0 {:7.1} GFLOPS ({:3} runs) | Q4_1 {:7.1} GFLOPS ({:3} runs)\n",
            n, n, gflops[0], runs[0], gflops[1], runs[1]
        ));
        report.push_str(&format!(
            "{:4} x {:4}: Q5_0 {:7.1} GFLOPS ({:3} runs) | Q5_1 {:7.1} GFLOPS ({:3} runs) | Q8_0 {:7.1} GFLOPS ({:3} runs)\n",
            n, n, gflops[2], runs[2], gflops[3], runs[3], gflops[4], runs[4]
        ));
        report.push_str(&format!(
            "{:4} x {:4}: F16  {:7.1} GFLOPS ({:3} runs) | F32  {:7.1} GFLOPS ({:3} runs)\n",
            n, n, gflops[5], runs[5], gflops[6], runs[6]
        ));
    }

    report
}

#[no_mangle]
pub extern "C" fn whisper_bench_memcpy(n_threads: c_int) -> c_int {
    eprint!("{}", bench_memcpy_report(n_threads));
    0
}

#[no_mangle]
pub extern "C" fn whisper_bench_memcpy_str(n_threads: c_int) -> *const c_char {
    store_c_string(&BENCH_MEMCPY_STRINGS, bench_memcpy_report(n_threads))
}

#[no_mangle]
pub extern "C" fn whisper_bench_ggml_mul_mat(n_threads: c_int) -> c_int {
    eprint!(
        "{}",
        bench_mul_mat_report(
            BENCH_MUL_MAT_SIZES,
            BENCH_MUL_MAT_N_MAX,
            n_threads,
            BENCH_MUL_MAT_TIME_BUDGET
        )
    );
    0
}

#[no_mangle]
pub extern "C" fn whisper_bench_ggml_mul_mat_str(n_threads: c_int) -> *const c_char {
    store_c_string(
        &BENCH_MUL_MAT_STRINGS,
        bench_mul_mat_report(
            BENCH_MUL_MAT_SIZES,
            BENCH_MUL_MAT_N_MAX,
            n_threads,
            BENCH_MUL_MAT_TIME_BUDGET,
        ),
    )
}

#[cfg(test)]
mod bench_tests {
    use super::*;

    #[test]
    fn memcpy_report_zero_threads_is_well_formed() {
        let report = bench_memcpy_report(0);
        assert!(report.contains("memcpy:"));
        assert!(report.contains("sum:"));
        assert_eq!(report.lines().count(), 3);
    }

    #[test]
    fn mul_mat_report_formats_every_type() {
        let report = bench_mul_mat_report(&[32], 3, 1, 0.0);
        let lines: Vec<&str> = report.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("Q4_0") && lines[0].contains("Q4_1"));
        assert!(lines[1].contains("Q5_0") && lines[1].contains("Q5_1"));
        assert!(lines[1].contains("Q8_0"));
        assert!(lines[2].contains("F16") && lines[2].contains("F32"));
        for line in lines {
            assert!(line.contains("GFLOPS"));
            assert!(line.contains("runs)"));
        }
    }
}
