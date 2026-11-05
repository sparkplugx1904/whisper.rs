#![forbid(unsafe_code)]

use clap::Parser;
use std::path::Path;
use std::time::Instant;
pub mod download;
pub use download::{download_model, get_model_download_url};
pub mod format;
pub use format::{
    export_segments_to_file, JsonFormatter, KaraokeVttFormatter, OutputFormatOptions,
    OutputFormatter, SrtFormatter, TxtFormatter, VttFormatter,
};
use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams};

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
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
    #[arg(long = "tinydiarize", alias = "tdrz", default_value_t = false)]
    pub tinydiarize: bool,

    /// Export karaoke subtitle format with word-level timing tags (.karaoke.vtt / .karaoke.srt)
    #[arg(long = "karaoke", alias = "ok", default_value_t = false)]
    pub karaoke: bool,
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
            "-ok" => out.push("--karaoke".to_string()),
            "-ocsv" => out.push("--output-csv".to_string()),
            "-olrc" => out.push("--output-lrc".to_string()),
            s if s.starts_with("-of=") => out.push(format!("--output-file={}", &s[4..])),
            s if s.starts_with("-bs=") => out.push(format!("--beam-size={}", &s[4..])),
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
        }
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
    println!(
        "whisper_init_from_file: loading model from '{}'...",
        args.model
    );
    let model = WhisperModel::load_from_file(Path::new(&args.model))
        .map_err(|e| format!("failed to load model from '{}': {}", args.model, e))?;

    let h = &model.hparams;
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
    println!("whisper_model_load: type          = {}", h.model_type);
    println!("whisper_model_load: n_langs       = {}", h.num_languages());
    println!(
        "whisper_model_load: total tensors = {}",
        model.tensors.len()
    );
    println!(
        "whisper_model_load: model size    = {:.2} MB ({} bytes)",
        model.total_weight_bytes as f64 / 1e6,
        model.total_weight_bytes
    );

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

    let params = WhisperParams {
        language: args.language.clone(),
        translate: args.translate,
        beam_size: args.beam_size,
        initial_prompt: args.prompt.clone(),
        no_timestamps: args.no_timestamps,
        n_threads: args.threads,
        tdrz_enable: args.tinydiarize,
        ..Default::default()
    };

    println!(
        "main: processing '{}' ({:.1} sec), {} threads, lang = {}, task = {} ...\n",
        args.file,
        audio_dur,
        args.threads,
        params.language,
        if params.translate {
            "translate"
        } else {
            "transcribe"
        }
    );

    let t_trans = Instant::now();
    let segments = ctx
        .transcribe_file(&args.file, &params)
        .map_err(|e| format!("transcription failed: {}", e))?;
    let proc_dur = t_trans.elapsed();

    for seg in &segments {
        if args.word_timestamps && !seg.words.is_empty() {
            for w in &seg.words {
                println!(
                    "  [word {:05.2}s --> {:05.2}s] \"{}\" (p={:.2})",
                    w.t0, w.t1, w.word, w.probability
                );
            }
        }
        if args.no_timestamps {
            println!("{}", seg.text.trim());
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
                "[{:02}:{:02}:{:06.3} --> {:02}:{:02}:{:06.3}] {}",
                (t0 / 3600.0) as u32,
                ((t0 % 3600.0) / 60.0) as u32,
                t0 % 60.0,
                (t1 / 3600.0) as u32,
                ((t1 % 3600.0) / 60.0) as u32,
                t1 % 60.0,
                seg.text.trim()
            );
        }
    }

    println!(
        "\nwhisper_print_timings:     load time = {:.2} ms",
        load_dur.as_secs_f64() * 1000.0
    );
    println!(
        "whisper_print_timings:    total time = {:.2} ms",
        proc_dur.as_secs_f64() * 1000.0
    );
    println!(
        "whisper_print_timings:    real time factor = {:.3}x",
        proc_dur.as_secs_f64() / audio_dur.max(0.001)
    );

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
            translate: false,
            beam_size: 1,
            prompt: None,
            no_timestamps: false,
            info: false,
            output_txt: true,
            output_srt: true,
            output_vtt: true,
            output_json: true,
            output_csv: true,
            output_lrc: true,
            output_file: Some(base_str.to_string()),
            word_timestamps: false,
            tinydiarize: false,
            karaoke: false,
            download: None,
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
        assert!(csv_content.starts_with("start,end,text\n"));
        assert!(csv_content.contains("0,2500,\"Hello world!\"\n"));

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
            language: "auto".to_string(),
            threads: 1,
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
            karaoke: false,
            download: None,
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
            language: "auto".to_string(),
            threads: 1,
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
            karaoke: false,
            download: None,
        };
        let res = run(args);
        assert!(res.is_err());
    }
}
