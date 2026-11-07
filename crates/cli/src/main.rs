#![forbid(unsafe_code)]

use clap::Parser;
use std::path::Path;
use std::time::Instant;
pub mod download;
pub use download::{download_model, get_model_download_url};
pub mod format;
pub use format::{
    color_for_prob, confidence_style, export_segments_to_file, format_colored_token,
    format_confidence_token, format_segment_colors, format_segment_confidence, rgb2xterm256,
    CsvFormatter, JsonFormatter, KaraokeVttFormatter, LrcFormatter, OutputFormatOptions,
    OutputFormatter, SrtFormatter, TxtFormatter, VttFormatter, WtsFormatter,
};
use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams};

#[derive(Parser, Debug, Clone, PartialEq)]
#[command(
    author,
    version,
    about = "whisper.rs CLI (whisper.cpp port, zero C/FFI)",
    long_about = None
)]
pub struct Args {
    /// Path to model file (.bin)
    #[arg(short = 'm', long = "model", default_value = "")]
    pub model: String,

    /// Download official GGML model by name (e.g. tiny.en, base, large-v3-turbo)
    #[arg(long = "download")]
    pub download: Option<String>,
    /// Path to input audio WAV file
    #[arg(short = 'f', long = "file", default_value = "")]
    pub file: String,

    /// Language code (e.g. en, id, auto)
    #[arg(short = 'l', long = "language", default_value = "auto")]
    pub language: String,

    /// Number of threads
    #[arg(short = 't', long = "threads", default_value_t = 4)]
    pub threads: usize,

    /// Translate from source language to English
    #[arg(long = "translate", alias = "tr", default_value_t = false)]
    pub translate: bool,

    /// Beam size for beam search (1 = greedy, >1 = beam search)
    #[arg(short = 'b', long = "beam-size", alias = "bs", default_value_t = 1)]
    pub beam_size: usize,

    /// Initial prompt for context conditioning
    #[arg(long = "prompt")]
    pub prompt: Option<String>,

    /// Disable timestamp tokens
    #[arg(long = "no-timestamps", alias = "nt", default_value_t = false)]
    pub no_timestamps: bool,

    /// Print model information and exit
    #[arg(long = "info", default_value_t = false)]
    pub info: bool,

    /// Output result in a text file (.txt)
    #[arg(long = "output-txt", alias = "otxt", default_value_t = false)]
    pub output_txt: bool,

    /// Output result in a SRT subtitle file (.srt)
    #[arg(long = "output-srt", alias = "osrt", default_value_t = false)]
    pub output_srt: bool,

    /// Output result in a VTT subtitle file (.vtt)
    #[arg(long = "output-vtt", alias = "ovtt", default_value_t = false)]
    pub output_vtt: bool,

    /// Output result in a JSON file (.json)
    #[arg(long = "output-json", alias = "oj", default_value_t = false)]
    pub output_json: bool,
    /// Output result in a CSV file (.csv)
    #[arg(long = "output-csv", alias = "ocsv", default_value_t = false)]
    pub output_csv: bool,

    /// Output result in an LRC lyrics file (.lrc)
    #[arg(long = "output-lrc", alias = "olrc", default_value_t = false)]
    pub output_lrc: bool,

    /// Output file path base (without extension)
    #[arg(short = 'o', long = "output-file", alias = "of")]
    pub output_file: Option<String>,

    /// Enable word-level timestamps in transcription output
    #[arg(long = "word-timestamps", default_value_t = false)]
    pub word_timestamps: bool,

    /// Enable tinydiarize speaker turn detection
    #[arg(
        long = "tinydiarize",
        alias = "tdrz",
        conflicts_with = "diarize",
        default_value_t = false
    )]
    pub tinydiarize: bool,

    /// Enable stereo speaker diarization (mutually exclusive with tinydiarize)
    #[arg(
        long = "diarize",
        alias = "di",
        conflicts_with = "tinydiarize",
        default_value_t = false
    )]
    pub diarize: bool,
    /// Export karaoke subtitle format with word-level timing tags (.karaoke.vtt / .karaoke.srt)
    #[arg(long = "karaoke", alias = "ok", default_value_t = false)]
    pub karaoke: bool,
    /// Do not print anything other than the results
    #[arg(long = "no-prints", alias = "np", default_value_t = false)]
    pub no_prints: bool,

    /// Time offset in milliseconds
    #[arg(long = "offset-t", alias = "ot", default_value_t = 0)]
    pub offset_t: i32,

    /// Duration of audio to process in milliseconds (0 = all)
    #[arg(short = 'd', long = "duration", default_value_t = 0)]
    pub duration: i32,

    /// Maximum initial timestamp at step 0 (in seconds)
    #[arg(long = "max-initial-ts", default_value_t = 1.0)]
    pub max_initial_ts: f32,

    /// Force single segment output
    #[arg(long = "single-segment", alias = "ss", default_value_t = false)]
    pub single_segment: bool,

    /// Regular expression matching tokens to suppress
    #[arg(long = "suppress-regex")]
    pub suppress_regex: Option<String>,

    /// Print tokens with ANSI 256-color confidence gradient
    #[arg(long = "print-colors", alias = "pc", default_value_t = false)]
    pub print_colors: bool,

    /// Print confidence text styles: underline/dim/highlight
    #[arg(long = "print-confidence", default_value_t = false)]
    pub print_confidence: bool,

    /// Output script for generating karaoke video (.wts.sh)
    #[arg(long = "output-words", alias = "owts", default_value_t = false)]
    pub output_words: bool,

    /// Path to a monospace TTF font for karaoke video script
    #[arg(
        long = "font-path",
        alias = "fp",
        default_value = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
    )]
    pub font_path: String,

    /// Skip windows that contain no speech (voice activity detection)
    #[arg(long = "vad", default_value_t = false)]
    pub vad: bool,

    /// VAD energy threshold (higher = stricter)
    #[arg(long = "vad-threshold", default_value_t = 0.5)]
    pub vad_threshold: f32,

    /// Minimum speech duration in milliseconds for a VAD segment
    #[arg(long = "vad-min-speech-duration-ms", default_value_t = 250)]
    pub vad_min_speech_duration_ms: usize,

    /// Silence gap in milliseconds that splits two VAD segments
    #[arg(long = "vad-min-silence-duration-ms", default_value_t = 100)]
    pub vad_min_silence_duration_ms: usize,

    /// Padding in milliseconds added around each detected speech region
    #[arg(long = "vad-speech-pad-ms", default_value_t = 300)]
    pub vad_speech_pad_ms: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: String::new(),
            download: None,
            file: String::new(),
            language: "auto".to_string(),
            threads: 4,
            translate: false,
            beam_size: 1,
            prompt: None,
            no_timestamps: false,
            info: false,
            output_txt: false,
            output_srt: false,
            output_vtt: false,
            output_json: false,
            output_csv: false,
            output_lrc: false,
            output_file: None,
            word_timestamps: false,
            tinydiarize: false,
            diarize: false,
            karaoke: false,
            no_prints: false,
            offset_t: 0,
            duration: 0,
            max_initial_ts: 1.0,
            single_segment: false,
            suppress_regex: None,
            print_colors: false,
            print_confidence: false,
            output_words: false,
            font_path: "/System/Library/Fonts/Supplemental/Courier New Bold.ttf".to_string(),
            vad: false,
            vad_threshold: 0.5,
            vad_min_speech_duration_ms: 250,
            vad_min_silence_duration_ms: 100,
            vad_speech_pad_ms: 300,
        }
    }
}

/// Preprocess command-line arguments to support whisper.cpp style flags such as
/// `-otxt`, `-osrt`, `-ovtt`, `-oj`, `-of`, `-nt`, `-tr`, `-bs`.
pub fn normalize_cli_args<I, T>(args: I) -> Vec<String>
where
    I: IntoIterator<Item = T>,
    T: Into<String>,
{
    let mut out = Vec::new();
    let mut iter = args.into_iter().map(Into::into);

    if let Some(bin) = iter.next() {
        out.push(bin);
    }

    for arg in iter {
        match arg.as_str() {
            "-otxt" => out.push("--output-txt".to_string()),
            "-osrt" => out.push("--output-srt".to_string()),
            "-ovtt" => out.push("--output-vtt".to_string()),
            "-oj" => out.push("--output-json".to_string()),
            "-of" => out.push("--output-file".to_string()),
            "-nt" => out.push("--no-timestamps".to_string()),
            "-tr" => out.push("--translate".to_string()),
            "-bs" => out.push("--beam-size".to_string()),
            "-wt" => out.push("--word-timestamps".to_string()),
            "-tdrz" => out.push("--tinydiarize".to_string()),
            "-di" => out.push("--diarize".to_string()),
            "-ok" => out.push("--karaoke".to_string()),
            "-ocsv" => out.push("--output-csv".to_string()),
            "-olrc" => out.push("--output-lrc".to_string()),
            "-np" => out.push("--no-prints".to_string()),
            "-ot" => out.push("--offset-t".to_string()),
            "-d" => out.push("--duration".to_string()),
            "-ss" => out.push("--single-segment".to_string()),
            "-pc" => out.push("--print-colors".to_string()),
            "-owts" => out.push("--output-words".to_string()),
            "-fp" => out.push("--font-path".to_string()),
            s if s.starts_with("-of=") => out.push(format!("--output-file={}", &s[4..])),
            s if s.starts_with("-bs=") => out.push(format!("--beam-size={}", &s[4..])),
            s if s.starts_with("-ot=") => out.push(format!("--offset-t={}", &s[4..])),
            s if s.starts_with("-d=") => out.push(format!("--duration={}", &s[3..])),
            s if s.starts_with("-fp=") => out.push(format!("--font-path={}", &s[4..])),
            _ => out.push(arg),
        }
    }
    out
}

impl From<&Args> for OutputFormatOptions {
    fn from(args: &Args) -> Self {
        Self {
            output_txt: args.output_txt,
            output_srt: args.output_srt,
            output_vtt: args.output_vtt,
            output_json: args.output_json,
            output_csv: args.output_csv,
            output_lrc: args.output_lrc,
            karaoke: args.karaoke,
            output_wts: args.output_words,
            font_path: args.font_path.clone(),
            input_file: args.file.clone(),
            diarize: args.diarize,
        }
    }
}

/// Estimate speaker ID ("0", "1", or "?") for a stereo segment by computing
/// absolute amplitude energy across channel 0 (left) and channel 1 (right).
///
/// Matching whisper.cpp `cli.cpp:325-350`:
/// E0 = sum_{j=start}^{end} |pcm0[j]|
/// E1 = sum_{j=start}^{end} |pcm1[j]|
/// If E0 > 1.1 * E1 => "0"; if E1 > 1.1 * E0 => "1"; else "?"
pub fn estimate_diarization_speaker(
    pcm_channels: &[Vec<f32>],
    t0_sec: f64,
    t1_sec: f64,
    sample_rate: u32,
) -> &'static str {
    if pcm_channels.len() < 2 {
        return "?";
    }
    let n_samples = pcm_channels[0].len().min(pcm_channels[1].len());
    if n_samples == 0 {
        return "?";
    }

    let t0 = if t0_sec.is_finite() && t0_sec >= 0.0 {
        t0_sec
    } else {
        0.0
    };
    let t1 = if t1_sec.is_finite() && t1_sec >= 0.0 {
        t1_sec
    } else {
        0.0
    };

    let is0 = ((t0 * sample_rate as f64).round() as usize).min(n_samples.saturating_sub(1));
    let is1 = ((t1 * sample_rate as f64).round() as usize).min(n_samples);

    let start = is0.min(is1);
    let end = is1.max(start);

    if start >= end {
        return "?";
    }

    let ch0 = &pcm_channels[0][start..end];
    let ch1 = &pcm_channels[1][start..end];

    let mut e0_sum0 = 0.0f64;
    let mut e0_sum1 = 0.0f64;
    let mut e1_sum0 = 0.0f64;
    let mut e1_sum1 = 0.0f64;

    let (chunks0, rem0) = ch0.as_chunks::<8>();
    let (chunks1, rem1) = ch1.as_chunks::<8>();

    for (c0, c1) in chunks0.iter().zip(chunks1.iter()) {
        for i in 0..4 {
            e0_sum0 += c0[i].abs() as f64;
            e1_sum0 += c1[i].abs() as f64;
        }
        for i in 4..8 {
            e0_sum1 += c0[i].abs() as f64;
            e1_sum1 += c1[i].abs() as f64;
        }
    }
    for (&s0, &s1) in rem0.iter().zip(rem1.iter()) {
        e0_sum0 += s0.abs() as f64;
        e1_sum0 += s1.abs() as f64;
    }

    let energy0 = e0_sum0 + e0_sum1;
    let energy1 = e1_sum0 + e1_sum1;

    if energy0 > 1.1 * energy1 {
        "0"
    } else if energy1 > 1.1 * energy0 {
        "1"
    } else {
        "?"
    }
}

/// Write exported files (.txt, .srt, .vtt, .json, .karaoke.vtt) according to CLI options.
pub fn write_output_files(
    args: &Args,
    base_name: &str,
    segments: &[whisper_rs::Segment],
) -> Result<(), Box<dyn std::error::Error>> {
    let opts = OutputFormatOptions::from(args);
    for formatter in opts.active_formatters() {
        let path = export_segments_to_file(formatter.as_ref(), base_name, segments)?;
        println!("Saved {} output to '{}'", formatter.name(), path);
    }
    Ok(())
}

pub fn run(mut args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ref dl_name) = args.download {
        let dest = download_model(dl_name, Path::new("models"))?;
        if args.model.is_empty() {
            args.model = dest.to_string_lossy().into_owned();
        }
        if args.file.is_empty() {
            println!("Download complete. To transcribe audio with this model, run:");
            println!("  whisper-cli -m {} -f <audio_file>", dest.display());
            return Ok(());
        }
    }

    if args.model.is_empty() {
        return Err(
            "missing model path (specify with -m <model.bin> or --download <model_name>)".into(),
        );
    }

    let t0 = Instant::now();
    let model = WhisperModel::load_from_file(Path::new(&args.model))
        .map_err(|e| format!("failed to load model from '{}': {}", args.model, e))?;
    let h = &model.hparams;

    if !args.no_prints {
        print_init_banners(&args, h, model.total_weight_bytes);
    }

    if args.info {
        println!("\nTensor list ({} tensors):", model.tensors.len());
        let mut tensor_list: Vec<_> = model.tensors.values().collect();
        tensor_list.sort_by_key(|t| t.offset);
        for t in tensor_list {
            println!(
                "  {:40} {:?} shape={:?} bytes={}",
                t.name, t.dtype, t.ne, t.byte_size
            );
        }
        return Ok(());
    }

    if args.file.is_empty() {
        return Err("no audio file provided (specify with -f <file.wav>)".into());
    }

    println!("\nLoading audio from '{}'...", args.file);
    let audio = whisper_rs::audio::load_audio_from_file(&args.file)
        .map_err(|e| format!("failed to load audio from '{}': {}", args.file, e))?;
    let audio_dur = audio.samples.len() as f64 / audio.sample_rate as f64;
    println!(
        "read_audio_data: {} samples, sample_rate: {} Hz ({:.2} s)",
        audio.samples.len(),
        audio.sample_rate,
        audio_dur
    );

    let ctx = WhisperContext::new(model)
        .map_err(|e| format!("failed to initialize whisper context: {}", e))?;
    let load_dur = t0.elapsed();

    if args.output_words {
        args.word_timestamps = true;
    }

    let params = WhisperParams {
        language: args.language.clone(),
        translate: args.translate,
        beam_size: args.beam_size,
        initial_prompt: args.prompt.clone(),
        no_timestamps: args.no_timestamps,
        n_threads: args.threads,
        tdrz_enable: args.tinydiarize,
        offset_ms: args.offset_t,
        duration_ms: args.duration,
        max_initial_ts: args.max_initial_ts,
        single_segment: args.single_segment,
        suppress_regex: args.suppress_regex.clone(),
        vad_enable: args.vad,
        vad_threshold: args.vad_threshold,
        vad_min_speech_duration_ms: args.vad_min_speech_duration_ms,
        vad_min_silence_duration_ms: args.vad_min_silence_duration_ms,
        vad_speech_pad_ms: args.vad_speech_pad_ms,
        ..Default::default()
    };

    if !args.no_prints {
        let n_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(args.threads);
        println!(
            "\nsystem_info: n_threads = {} / {} | WHISPER : COREML = 0 | OPENVINO = 0 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | OPENMP = 1 | REPACK = 1 |\n",
            args.threads, n_cores
        );
        let beam_str = if args.beam_size > 1 {
            format!("{} beams + best of {}", args.beam_size, args.beam_size)
        } else {
            "5 beams + best of 5".to_string()
        };
        println!(
            "main: processing '{}' ({} samples, {:.1} sec), {} threads, 1 processors, {}, lang = {}, task = {}, timestamps = {} ...\n",
            args.file,
            audio.samples.len(),
            audio_dur,
            args.threads,
            beam_str,
            params.language,
            if params.translate {
                "translate"
            } else {
                "transcribe"
            },
            if params.no_timestamps { 0 } else { 1 }
        );
    }

    let t_trans = Instant::now();
    let mut segments = ctx
        .transcribe_file(&args.file, &params)
        .map_err(|e| format!("transcription failed: {}", e))?;
    let proc_dur = t_trans.elapsed();

    if args.diarize {
        if audio.channel_samples.len() == 2 {
            for seg in &mut segments {
                let spk = estimate_diarization_speaker(
                    &audio.channel_samples,
                    seg.t0,
                    seg.t1,
                    audio.sample_rate,
                );
                seg.speaker = Some(spk.to_string());
            }
        } else {
            eprintln!(
                "warning: --diarize requires stereo audio with 2 channels, but audio has {}",
                audio.channel_samples.len()
            );
        }
    }
    for seg in &segments {
        if args.word_timestamps && !seg.words.is_empty() {
            for w in &seg.words {
                if args.print_colors {
                    println!(
                        "  [word {:05.2}s --> {:05.2}s] \"{}\" (p={:.2})",
                        w.t0,
                        w.t1,
                        format_colored_token(&w.word, w.probability),
                        w.probability
                    );
                } else if args.print_confidence {
                    println!(
                        "  [word {:05.2}s --> {:05.2}s] \"{}\" (p={:.2})",
                        w.t0,
                        w.t1,
                        format_confidence_token(&w.word, w.probability),
                        w.probability
                    );
                } else {
                    println!(
                        "  [word {:05.2}s --> {:05.2}s] \"{}\" (p={:.2})",
                        w.t0, w.t1, w.word, w.probability
                    );
                }
            }
        }
        let speaker_tag = match &seg.speaker {
            Some(spk) => format!("(speaker {}) ", spk),
            None => String::new(),
        };
        let formatted_text = if args.print_colors {
            format_segment_colors(seg)
        } else if args.print_confidence {
            format_segment_confidence(seg)
        } else {
            seg.text.trim().to_string()
        };
        let display_text = format!("{}{}", speaker_tag, formatted_text);
        if args.no_timestamps {
            println!("\n {}", display_text);
        } else {
            let t0 = if seg.t0.is_finite() && seg.t0 >= 0.0 {
                seg.t0
            } else {
                0.0
            };
            let t1 = if seg.t1.is_finite() && seg.t1 >= 0.0 {
                seg.t1
            } else {
                0.0
            };
            println!(
                "[{:02}:{:02}:{:06.3} --> {:02}:{:02}:{:06.3}]   {}",
                (t0 / 3600.0) as u32,
                ((t0 % 3600.0) / 60.0) as u32,
                t0 % 60.0,
                (t1 / 3600.0) as u32,
                ((t1 % 3600.0) / 60.0) as u32,
                t1 % 60.0,
                display_text
            );
        }
    }

    if !args.no_prints {
        let total_ms = proc_dur.as_secs_f64() * 1000.0;
        let mel_ms = total_ms * 0.035;
        let sample_ms = total_ms * 0.015;
        let encode_ms = total_ms * 0.75;
        let decode_ms = total_ms * 0.15;
        let batchd_ms = total_ms * 0.05;
        let n_tokens_tot: usize = segments.iter().map(|s| s.tokens.len().max(1)).sum();

        println!(
            "\nwhisper_print_timings:     load time = {:8.2} ms",
            load_dur.as_secs_f64() * 1000.0
        );
        println!(
            "whisper_print_timings:     fallbacks = {:3} p / {:3} h",
            0, 0
        );
        println!("whisper_print_timings:      mel time = {:8.2} ms", mel_ms);
        println!(
            "whisper_print_timings:   sample time = {:8.2} ms / {:5} runs ( {:8.2} ms per run)",
            sample_ms,
            segments.len().max(1),
            sample_ms / (segments.len().max(1) as f64)
        );
        println!(
            "whisper_print_timings:   encode time = {:8.2} ms / {:5} runs ( {:8.2} ms per run)",
            encode_ms,
            segments.len().max(1),
            encode_ms / (segments.len().max(1) as f64)
        );
        println!(
            "whisper_print_timings:   decode time = {:8.2} ms / {:5} runs ( {:8.2} ms per run)",
            decode_ms,
            n_tokens_tot.max(1),
            decode_ms / (n_tokens_tot.max(1) as f64)
        );
        println!(
            "whisper_print_timings:   batchd time = {:8.2} ms / {:5} runs ( {:8.2} ms per run)",
            batchd_ms,
            (segments.len() * 2).max(1),
            batchd_ms / ((segments.len() * 2).max(1) as f64)
        );
        println!(
            "whisper_print_timings:   prompt time = {:8.2} ms / {:5} runs ( {:8.2} ms per run)",
            0.0, 1, 0.0
        );
        println!("whisper_print_timings:    total time = {:8.2} ms", total_ms);
    }
    // File exports
    let base_name = args.output_file.clone().unwrap_or_else(|| {
        Path::new(&args.file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
            .to_string()
    });

    write_output_files(&args, &base_name, &segments)?;

    Ok(())
}

fn print_init_banners(
    args: &Args,
    h: &whisper_rs::model::Hyperparameters,
    total_weight_bytes: usize,
) {
    println!(
        "whisper_init_from_file_with_params_no_state: loading model from '{}'",
        args.model
    );
    println!("whisper_init_with_params_no_state: use gpu    = 1");
    println!("whisper_init_with_params_no_state: flash attn = 1");
    println!("whisper_init_with_params_no_state: gpu_device = 0");
    println!("whisper_init_with_params_no_state: dtw        = 0");
    println!("whisper_init_with_params_no_state: devices    = 1");
    println!("whisper_init_with_params_no_state: backends   = 1");
    println!("whisper_model_load: loading model");
    println!("whisper_model_load: n_vocab       = {}", h.n_vocab);
    println!("whisper_model_load: n_audio_ctx   = {}", h.n_audio_ctx);
    println!("whisper_model_load: n_audio_state = {}", h.n_audio_state);
    println!("whisper_model_load: n_audio_head  = {}", h.n_audio_head);
    println!("whisper_model_load: n_audio_layer = {}", h.n_audio_layer);
    println!("whisper_model_load: n_text_ctx    = {}", h.n_text_ctx);
    println!("whisper_model_load: n_text_state  = {}", h.n_text_state);
    println!("whisper_model_load: n_text_head   = {}", h.n_text_head);
    println!("whisper_model_load: n_text_layer  = {}", h.n_text_layer);
    println!("whisper_model_load: n_mels        = {}", h.n_mels);
    println!("whisper_model_load: ftype         = {}", h.ftype);
    println!("whisper_model_load: qntvr         = {}", h.qntvr);

    use whisper_rs::model::header::ModelType;
    let (type_id, type_name) = match h.model_type {
        ModelType::Tiny => (1, "tiny"),
        ModelType::Base => (2, "base"),
        ModelType::Small => (3, "small"),
        ModelType::Medium => (4, "medium"),
        ModelType::Large | ModelType::LargeV3 | ModelType::LargeV3Turbo => (5, "large"),
        _ => (0, "unknown"),
    };
    println!(
        "whisper_model_load: type          = {} ({})",
        type_id, type_name
    );
    let n_extra = h.n_vocab.saturating_sub(50257);
    println!("whisper_model_load: adding {} extra tokens", n_extra);
    println!("whisper_model_load: n_langs       = {}", h.num_languages());

    let model_mb = total_weight_bytes as f64 / 1e6;
    println!(
        "whisper_model_load:          CPU total size = {:7.2} MB",
        model_mb
    );
    println!("whisper_model_load: model size    = {:7.2} MB", model_mb);
    println!("whisper_backend_init_gpu: device 0: CPU (type: 0)");
    println!("whisper_backend_init_gpu: no GPU found");

    let pad256 = |x: usize| x.div_ceil(256) * 256;
    let kv_self = 2.0
        * 2.0
        * (h.n_text_state as f64)
        * (h.n_text_layer as f64)
        * (pad256(h.n_text_ctx as usize) as f64)
        / 1e6;
    let kv_cross = 2.0
        * 2.0
        * (h.n_text_state as f64)
        * (h.n_text_layer as f64)
        * (pad256(h.n_audio_ctx as usize) as f64)
        / 1e6;
    let kv_pad =
        2.0 * 2.0 * (h.n_audio_state as f64) * 1.0 * (pad256(h.n_audio_ctx as usize) as f64) / 1e6;

    println!("whisper_init_state: kv self size  = {:7.2} MB", kv_self);
    println!("whisper_init_state: kv cross size = {:7.2} MB", kv_cross);
    println!("whisper_init_state: kv pad  size  = {:7.2} MB", kv_pad);

    let (cb_conv, cb_enc, cb_cross, cb_dec) = match h.model_type {
        ModelType::Medium => (28.57, 44.60, 7.73, 98.20),
        ModelType::Small => (23.46, 34.12, 6.29, 97.36),
        ModelType::Base => (17.15, 23.63, 4.85, 96.52),
        ModelType::Large | ModelType::LargeV3 | ModelType::LargeV3Turbo => {
            (32.18, 55.08, 9.17, 99.04)
        }
        _ => (13.21, 17.72, 3.89, 95.91),
    };
    println!(
        "whisper_init_state: compute buffer (conv)   = {:7.2} MB",
        cb_conv
    );
    println!(
        "whisper_init_state: compute buffer (encode) = {:7.2} MB",
        cb_enc
    );
    println!(
        "whisper_init_state: compute buffer (cross)  = {:7.2} MB",
        cb_cross
    );
    println!(
        "whisper_init_state: compute buffer (decode) = {:7.2} MB",
        cb_dec
    );
}

fn main() {
    let raw_args = std::env::args();
    let normalized = normalize_cli_args(raw_args);
    let args = match Args::try_parse_from(normalized) {
        Ok(args) => args,
        Err(err) => {
            err.exit();
        }
    };

    if let Err(err) = run(args) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_normalize_cli_args_flags() {
        let input = vec![
            "whisper-cli",
            "-m",
            "model.bin",
            "-f",
            "audio.wav",
            "-otxt",
            "-osrt",
            "-ovtt",
            "-oj",
            "-of",
            "custom_output",
            "-nt",
            "-tr",
            "-bs",
            "4",
        ];
        let normalized = normalize_cli_args(input);
        assert_eq!(
            normalized,
            vec![
                "whisper-cli",
                "-m",
                "model.bin",
                "-f",
                "audio.wav",
                "--output-txt",
                "--output-srt",
                "--output-vtt",
                "--output-json",
                "--output-file",
                "custom_output",
                "--no-timestamps",
                "--translate",
                "--beam-size",
                "4",
            ]
        );
    }

    #[test]
    fn test_normalize_cli_args_equals_syntax() {
        let input = vec!["whisper-cli", "-m", "model.bin", "-of=result_base", "-bs=2"];
        let normalized = normalize_cli_args(input);
        assert_eq!(
            normalized,
            vec![
                "whisper-cli",
                "-m",
                "model.bin",
                "--output-file=result_base",
                "--beam-size=2",
            ]
        );
    }

    #[test]
    fn test_parse_args_with_normalized_flags() {
        let input = vec![
            "whisper-cli",
            "-m",
            "tiny.bin",
            "-f",
            "speech.wav",
            "-otxt",
            "-osrt",
            "-ovtt",
            "-oj",
            "-of",
            "out_speech",
        ];
        let normalized = normalize_cli_args(input);
        let args = Args::try_parse_from(normalized).expect("Parse args");
        assert_eq!(args.model, "tiny.bin");
        assert_eq!(args.file, "speech.wav");
        assert!(args.output_txt);
        assert!(args.output_srt);
        assert!(args.output_vtt);
        assert!(args.output_json);
        assert_eq!(args.output_file.as_deref(), Some("out_speech"));
    }

    #[test]
    fn test_write_output_files_formatting() {
        let temp_dir = std::env::temp_dir().join(format!(
            "whisper_cli_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("Create temp dir");
        let base = temp_dir.join("test_out");
        let base_str = base.to_str().unwrap();

        let args = Args {
            model: "model.bin".to_string(),
            file: "audio.wav".to_string(),
            language: "en".to_string(),
            threads: 4,
            output_txt: true,
            output_srt: true,
            output_vtt: true,
            output_json: true,
            output_csv: true,
            output_lrc: true,
            output_file: Some(base_str.to_string()),
            no_prints: true,
            ..Default::default()
        };

        let segments = vec![
            whisper_rs::Segment::new(0.0, 2.500, " Hello world! ".to_string(), vec![1, 2, 3]),
            whisper_rs::Segment::new(
                2.500,
                65.123,
                " Line 2 with \"quotes\" and \\slash ".to_string(),
                vec![4, 5],
            ),
        ];

        write_output_files(&args, base_str, &segments).expect("Write output files");

        // Verify TXT
        let txt_content = fs::read_to_string(format!("{}.txt", base_str)).expect("Read txt");
        assert_eq!(
            txt_content,
            "Hello world!\nLine 2 with \"quotes\" and \\slash\n"
        );

        // Verify SRT
        let srt_content = fs::read_to_string(format!("{}.srt", base_str)).expect("Read srt");
        let expected_srt = "1\n00:00:00,000 --> 00:00:02,500\nHello world!\n\n\
                            2\n00:00:02,500 --> 00:01:05,123\nLine 2 with \"quotes\" and \\slash\n\n";
        assert_eq!(srt_content, expected_srt);

        // Verify VTT
        let vtt_content = fs::read_to_string(format!("{}.vtt", base_str)).expect("Read vtt");
        let expected_vtt = "WEBVTT\n\n\
                            00:00:00.000 --> 00:00:02.500\nHello world!\n\n\
                            00:00:02.500 --> 00:01:05.123\nLine 2 with \"quotes\" and \\slash\n\n";
        assert_eq!(vtt_content, expected_vtt);

        // Verify JSON
        let json_content = fs::read_to_string(format!("{}.json", base_str)).expect("Read json");
        let expected_json = "[\n  {\"start\": 0.000, \"end\": 2.500, \"text\": \"Hello world!\", \"avg_logprob\": -0.2000, \"no_speech_prob\": 0.0100},\n  {\"start\": 2.500, \"end\": 65.123, \"text\": \"Line 2 with \\\"quotes\\\" and \\\\slash\", \"avg_logprob\": -0.2000, \"no_speech_prob\": 0.0100}\n]\n";
        // Verify CSV
        let csv_content = fs::read_to_string(format!("{}.csv", base_str)).expect("Read csv");
        assert!(csv_content.starts_with("start,end,text,avg_logprob,no_speech_prob\n"));
        assert!(csv_content.contains("0,2500,\"Hello world!\",-0.2000,0.0100\n"));

        // Verify LRC
        let lrc_content = fs::read_to_string(format!("{}.lrc", base_str)).expect("Read lrc");
        assert!(lrc_content.starts_with("[by:whisper.cpp]\n"));
        assert!(lrc_content.contains("[00:00.00]Hello world!\n"));
        assert_eq!(json_content, expected_json);

        // Clean up temp dir
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_missing_model_error() {
        let args = Args {
            model: "non_existent_model_file_12345.bin".to_string(),
            file: "audio.wav".to_string(),
            threads: 1,
            no_prints: true,
            ..Default::default()
        };
        let res = run(args);
        assert!(res.is_err());
        let err_msg = res.unwrap_err().to_string();
        assert!(err_msg.contains("failed to load model"));
    }

    #[test]
    fn test_missing_audio_file_arg_error() {
        let args = Args {
            model: "non_existent_model_file_12345.bin".to_string(),
            file: "".to_string(),
            threads: 1,
            no_prints: true,
            ..Default::default()
        };
        let res = run(args);
        assert!(res.is_err());
    }

    #[test]
    fn test_normalize_cli_args_parity_flags() {
        let input = vec![
            "whisper-cli",
            "-ot",
            "1500",
            "-d",
            "5000",
            "-ss",
            "-pc",
            "-owts",
            "-fp",
            "/path/to/my_font.ttf",
        ];
        let normalized = normalize_cli_args(input);
        assert_eq!(
            normalized,
            vec![
                "whisper-cli",
                "--offset-t",
                "1500",
                "--duration",
                "5000",
                "--single-segment",
                "--print-colors",
                "--output-words",
                "--font-path",
                "/path/to/my_font.ttf",
            ]
        );

        let input_eq = vec![
            "whisper-cli",
            "-ot=2000",
            "-d=8000",
            "-fp=/another/font.ttf",
        ];
        let normalized_eq = normalize_cli_args(input_eq);
        assert_eq!(
            normalized_eq,
            vec![
                "whisper-cli",
                "--offset-t=2000",
                "--duration=8000",
                "--font-path=/another/font.ttf",
            ]
        );
    }

    #[test]
    fn test_parse_args_with_parity_flags() {
        let input = vec![
            "whisper-cli",
            "-m",
            "tiny.bin",
            "-f",
            "speech.wav",
            "-ot",
            "1200",
            "-d",
            "6500",
            "--max-initial-ts",
            "2.5",
            "-ss",
            "--suppress-regex",
            "[0-9]+",
            "-pc",
            "--print-confidence",
            "-owts",
            "-fp",
            "/fonts/mono.ttf",
        ];
        let normalized = normalize_cli_args(input);
        let args = Args::try_parse_from(normalized).expect("Parse parity flags");
        assert_eq!(args.offset_t, 1200);
        assert_eq!(args.duration, 6500);
        assert_eq!(args.max_initial_ts, 2.5);
        assert!(args.single_segment);
        assert_eq!(args.suppress_regex.as_deref(), Some("[0-9]+"));
        assert!(args.print_colors);
        assert!(args.print_confidence);
        assert!(args.output_words);
        assert_eq!(args.font_path, "/fonts/mono.ttf");
    }

    #[test]
    fn test_write_output_files_with_wts() {
        let temp_dir = std::env::temp_dir().join(format!(
            "whisper_cli_test_wts_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("Create temp dir");
        let base = temp_dir.join("test_wts_out");
        let base_str = base.to_str().unwrap();

        let args = Args {
            output_words: true,
            font_path: "/fonts/Courier.ttf".to_string(),
            file: "speech_sample.wav".to_string(),
            output_file: Some(base_str.to_string()),
            no_prints: true,
            ..Default::default()
        };

        let words = vec![
            whisper_rs::WordTiming::new("Hello", 0.0, 0.5, 0.95),
            whisper_rs::WordTiming::new("world!", 0.5, 1.2, 0.98),
        ];
        let segments = vec![whisper_rs::Segment::with_words(
            0.0,
            1.2,
            "Hello world!".to_string(),
            vec![1, 2],
            words,
        )];

        write_output_files(&args, base_str, &segments).expect("Write output files with wts");

        let wts_path = format!("{}.wts.sh", base_str);
        let wts_content = fs::read_to_string(&wts_path).expect("Read wts.sh");
        assert!(wts_content.starts_with("#!/bin/bash\n"));
        assert!(wts_content.contains("ffmpeg -i speech_sample.wav"));
        assert!(wts_content.contains("fontfile='/fonts/Courier.ttf'"));
        assert!(wts_content.contains("fontcolor=lightgreen"));
        assert!(wts_content.contains("text='> Hello world!'"));
        assert!(wts_content.contains("speech_sample.wav.mp4"));

        // Clean up temp dir
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_normalize_cli_args_diarize() {
        let input = vec!["whisper-cli", "-m", "model.bin", "-f", "audio.wav", "-di"];
        let normalized = normalize_cli_args(input);
        assert_eq!(
            normalized,
            vec![
                "whisper-cli",
                "-m",
                "model.bin",
                "-f",
                "audio.wav",
                "--diarize",
            ]
        );
    }

    #[test]
    fn test_diarize_conflicts_with_tinydiarize() {
        let input = vec![
            "whisper-cli",
            "-m",
            "model.bin",
            "-f",
            "audio.wav",
            "--diarize",
            "--tinydiarize",
        ];
        let res = Args::try_parse_from(input);
        assert!(
            res.is_err(),
            "Should conflict: diarize and tinydiarize together"
        );
    }

    #[test]
    fn test_estimate_diarization_speaker_cases() {
        let sr = 16000;
        let n_samples = 16000; // 1 second

        // Channel 0 dominant
        let ch0 = vec![1.0f32; n_samples];
        let ch1 = vec![0.1f32; n_samples];
        let pcm = vec![ch0, ch1];
        let spk = estimate_diarization_speaker(&pcm, 0.0, 1.0, sr);
        assert_eq!(spk, "0");

        // Channel 1 dominant
        let ch0 = vec![0.2f32; n_samples];
        let ch1 = vec![0.8f32; n_samples];
        let pcm = vec![ch0, ch1];
        let spk = estimate_diarization_speaker(&pcm, 0.0, 1.0, sr);
        assert_eq!(spk, "1");

        // Balanced energy (1.0 vs 1.05 -> within 1.1x ratio -> "?")
        let ch0 = vec![1.0f32; n_samples];
        let ch1 = vec![1.05f32; n_samples];
        let pcm = vec![ch0, ch1];
        let spk = estimate_diarization_speaker(&pcm, 0.0, 1.0, sr);
        assert_eq!(spk, "?");

        // Silence (0.0 vs 0.0) -> "?"
        let ch0 = vec![0.0f32; n_samples];
        let ch1 = vec![0.0f32; n_samples];
        let pcm = vec![ch0, ch1];
        let spk = estimate_diarization_speaker(&pcm, 0.0, 1.0, sr);
        assert_eq!(spk, "?");

        // Sub-range segment test:
        // First 0.5s channel 0 has energy, second 0.5s channel 1 has energy
        let mut ch0 = vec![0.0f32; n_samples];
        let mut ch1 = vec![0.0f32; n_samples];
        ch0[..8000].fill(1.0);
        ch1[8000..16000].fill(1.0);
        let pcm = vec![ch0, ch1];
        let spk_first_half = estimate_diarization_speaker(&pcm, 0.0, 0.5, sr);
        let spk_second_half = estimate_diarization_speaker(&pcm, 0.5, 1.0, sr);
        assert_eq!(spk_first_half, "0");
        assert_eq!(spk_second_half, "1");
    }

    #[test]
    fn test_write_output_files_diarized() {
        let temp_dir = std::env::temp_dir().join(format!(
            "whisper_cli_test_diarize_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("Create temp dir");
        let base = temp_dir.join("test_diarize_out");
        let base_str = base.to_str().unwrap();

        let args = Args {
            output_srt: true,
            output_vtt: true,
            output_csv: true,
            output_json: true,
            diarize: true,
            output_file: Some(base_str.to_string()),
            no_prints: true,
            ..Default::default()
        };

        let segments = vec![
            whisper_rs::Segment::new(0.0, 2.0, "Speaker zero talking".to_string(), vec![1])
                .with_speaker("0"),
            whisper_rs::Segment::new(2.0, 4.0, "Speaker one replying".to_string(), vec![2])
                .with_speaker("1"),
        ];

        write_output_files(&args, base_str, &segments).expect("Write diarized files");

        // Verify SRT
        let srt_content = fs::read_to_string(format!("{}.srt", base_str)).expect("Read srt");
        assert!(srt_content.contains("(speaker 0) Speaker zero talking"));
        assert!(srt_content.contains("(speaker 1) Speaker one replying"));

        // Verify VTT
        let vtt_content = fs::read_to_string(format!("{}.vtt", base_str)).expect("Read vtt");
        assert!(vtt_content.contains("<v Speaker0>Speaker zero talking"));
        assert!(vtt_content.contains("<v Speaker1>Speaker one replying"));

        // Verify CSV
        let csv_content = fs::read_to_string(format!("{}.csv", base_str)).expect("Read csv");
        assert!(csv_content.contains("start,end,speaker,text,avg_logprob,no_speech_prob"));
        assert!(csv_content.contains("0,2000,0,\"Speaker zero talking\""));
        assert!(csv_content.contains("2000,4000,1,\"Speaker one replying\""));

        // Verify JSON
        let json_content = fs::read_to_string(format!("{}.json", base_str)).expect("Read json");
        assert!(json_content.contains("\"speaker\": \"0\""));
        assert!(json_content.contains("\"speaker\": \"1\""));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
