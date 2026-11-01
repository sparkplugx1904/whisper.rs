#![forbid(unsafe_code)]

use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use whisper_rs_pure::audio::WavAudio;
use whisper_rs_pure::model::WhisperModel;
use whisper_rs_pure::{WhisperContext, WhisperParams};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Pure-Rust Whisper CLI (whisper.cpp port, zero C/FFI)",
    long_about = None
)]
struct Args {
    /// Path to model file (.bin)
    #[arg(short = 'm', long = "model")]
    model: String,

    /// Path to input audio WAV file
    #[arg(short = 'f', long = "file", default_value = "")]
    file: String,

    /// Language code (e.g. en, id, auto)
    #[arg(short = 'l', long = "language", default_value = "auto")]
    language: String,

    /// Number of threads
    #[arg(short = 't', long = "threads", default_value_t = 4)]
    threads: usize,

    /// Translate from source language to English
    #[arg(long = "translate", default_value_t = false)]
    translate: bool,
    /// Beam size for beam search (1 = greedy, >1 = beam search)
    #[arg(short = 'b', long = "beam-size", default_value_t = 1)]
    beam_size: usize,

    /// Initial prompt for context conditioning
    #[arg(long = "prompt")]
    prompt: Option<String>,

    /// Disable timestamp tokens
    #[arg(long = "no-timestamps", default_value_t = false)]
    no_timestamps: bool,
    /// Print model information and exit
    #[arg(long = "info", default_value_t = false)]
    info: bool,

    /// Output result in a text file (.txt)
    #[arg(long = "output-txt", default_value_t = false)]
    output_txt: bool,

    /// Output result in a SRT subtitle file (.srt)
    #[arg(long = "output-srt", default_value_t = false)]
    output_srt: bool,

    /// Output result in a VTT subtitle file (.vtt)
    #[arg(long = "output-vtt", default_value_t = false)]
    output_vtt: bool,

    /// Output result in a JSON file (.json)
    #[arg(long = "output-json", default_value_t = false)]
    output_json: bool,

    /// Output file path base (without extension)
    #[arg(short = 'o', long = "output-file")]
    output_file: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let t0 = Instant::now();
    println!(
        "whisper_init_from_file: loading model from '{}'...",
        args.model
    );
    let model = WhisperModel::load_from_file(Path::new(&args.model))?;

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
        eprintln!("\nNotice: No audio file provided (-f <file.wav>). Exiting.");
        return Ok(());
    }

    println!("\nLoading audio from '{}'...", args.file);
    let audio = WavAudio::load_from_file(&args.file)?;
    let audio_dur = audio.samples.len() as f64 / audio.sample_rate as f64;
    println!(
        "read_audio_data: {} samples, sample_rate: {} Hz ({:.2} s)",
        audio.samples.len(),
        audio.sample_rate,
        audio_dur
    );

    let ctx = WhisperContext::new(model)?;
    let load_dur = t0.elapsed();

    let params = WhisperParams {
        language: args.language.clone(),
        translate: args.translate,
        beam_size: args.beam_size,
        initial_prompt: args.prompt,
        no_timestamps: args.no_timestamps,
        n_threads: args.threads,
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
    let segments = ctx.transcribe_file(&args.file, &params)?;
    let proc_dur = t_trans.elapsed();

    for seg in &segments {
        if args.no_timestamps {
            println!("{}", seg.text.trim());
        } else {
            println!(
                "[{:02}:{:02}:{:06.3} --> {:02}:{:02}:{:06.3}] {}",
                (seg.t0 / 3600.0) as u32,
                ((seg.t0 % 3600.0) / 60.0) as u32,
                seg.t0 % 60.0,
                (seg.t1 / 3600.0) as u32,
                ((seg.t1 % 3600.0) / 60.0) as u32,
                seg.t1 % 60.0,
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
    let base_name = args.output_file.unwrap_or_else(|| {
        Path::new(&args.file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
            .to_string()
    });

    if args.output_txt {
        let path = format!("{}.txt", base_name);
        let mut f = File::create(&path)?;
        for s in &segments {
            writeln!(f, "{}", s.text.trim())?;
        }
        println!("Saved TXT output to '{}'", path);
    }

    if args.output_srt {
        let path = format!("{}.srt", base_name);
        let mut f = File::create(&path)?;
        for (i, s) in segments.iter().enumerate() {
            write!(f, "{}", s.to_srt(i + 1))?;
        }
        println!("Saved SRT output to '{}'", path);
    }

    if args.output_vtt {
        let path = format!("{}.vtt", base_name);
        let mut f = File::create(&path)?;
        writeln!(f, "WEBVTT\n")?;
        for s in &segments {
            write!(f, "{}", s.to_vtt())?;
        }
        println!("Saved VTT output to '{}'", path);
    }

    if args.output_json {
        let path = format!("{}.json", base_name);
        let mut f = File::create(&path)?;
        writeln!(f, "[")?;
        for (i, s) in segments.iter().enumerate() {
            let comma = if i + 1 < segments.len() { "," } else { "" };
            let escaped = s.text.trim().replace('\\', "\\\\").replace('"', "\\\"");
            writeln!(
                f,
                "  {{\"t0\": {:.3}, \"t1\": {:.3}, \"text\": \"{}\"}}{}",
                s.t0, s.t1, escaped, comma
            )?;
        }
        writeln!(f, "]")?;
        println!("Saved JSON output to '{}'", path);
    }

    Ok(())
}
