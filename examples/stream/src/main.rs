#![forbid(unsafe_code)]

use clap::Parser;
use std::io::{Read, Write};
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use whisper_rs::audio::WavAudio;
use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Real-time streaming audio transcriber with sliding window splicing & prompt feedback"
)]
struct Args {
    /// Model file path (.bin)
    #[arg(short = 'm', long = "model", default_value = "models/ggml-tiny.en.bin")]
    model: String,

    /// Input WAV audio file to stream from (simulates real-time microphone stream)
    #[arg(short = 'f', long = "file", default_value = "samples/jfk.wav")]
    file: String,

    /// Stream live PCM directly from standard input (16kHz mono 16-bit LE, e.g. from arecord/sox/ffmpeg)
    #[arg(long = "stdin")]
    stdin: bool,

    /// Audio step size in milliseconds
    #[arg(long = "step-ms", default_value_t = 3000)]
    step_ms: usize,

    /// Length of audio window in milliseconds
    #[arg(long = "length-ms", default_value_t = 10000)]
    length_ms: usize,

    /// Length of audio window in seconds (legacy flag)
    #[arg(long = "length-s")]
    length_s: Option<usize>,

    /// Audio to keep from previous step in milliseconds
    #[arg(long = "keep-ms", default_value_t = 200)]
    keep_ms: usize,

    /// Voice energy detection threshold
    #[arg(long = "vad-thold", default_value_t = 0.005)]
    vad_thold: f32,

    /// Keep context between audio chunks (prompt token feedback)
    #[arg(short = 'k', long = "keep-context", alias = "kc")]
    keep_context: bool,

    /// Audio context size (0 - all 1500 tokens, e.g. 512, 768 for partial encoder execution)
    #[arg(short = 'a', long = "audio-ctx", alias = "ac", default_value_t = 0)]
    audio_ctx: usize,

    /// Spoken language
    #[arg(short = 'l', long = "language", default_value = "en")]
    language: String,

    /// Translate from source language to english
    #[arg(short = 't', long = "translate", alias = "tr")]
    translate: bool,

    /// Print timestamps
    #[arg(long = "print-timestamps")]
    print_timestamps: bool,
}

fn compute_audio_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / (samples.len() as f32)).sqrt()
}

fn render_vu_meter(energy: f32) -> String {
    let bars = ((energy * 200.0).clamp(0.0, 20.0) as usize).min(20);
    let filled = "|".repeat(bars);
    let empty = ".".repeat(20 - bars);
    format!("[{}{}]", filled, empty)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== REAL-TIME STREAMING AUDIO TRANSCRIBER ===");
    println!("Loading Whisper model from '{}'...", args.model);
    let model = WhisperModel::load_from_file(Path::new(&args.model))?;
    let ctx = WhisperContext::new(model)?;

    let step_ms = args.step_ms;
    let keep_ms = args.keep_ms.min(step_ms);
    let length_ms = args
        .length_s
        .map(|s| s * 1000)
        .unwrap_or(args.length_ms)
        .max(step_ms);

    let sample_rate = 16000usize;
    let n_samples_step = (step_ms * sample_rate) / 1000;
    let n_samples_len = (length_ms * sample_rate) / 1000;
    let n_samples_keep = (keep_ms * sample_rate) / 1000;

    let n_new_line = (length_ms / step_ms).saturating_sub(1).max(1);

    println!(
        "Stream parameters: step = {} ms ({} samples) | len = {} ms ({} samples) | keep = {} ms ({} samples) | n_new_line = {}",
        step_ms, n_samples_step, length_ms, n_samples_len, keep_ms, n_samples_keep, n_new_line
    );
    println!(
        "Mode = {} | keep_context = {} | audio_ctx = {} | language = {} | VAD thold = {:.4}\n",
        if args.stdin {
            "live STDIN pipe"
        } else {
            "file stream simulation"
        },
        args.keep_context,
        if args.audio_ctx == 0 {
            "full (1500)".to_string()
        } else {
            args.audio_ctx.to_string()
        },
        args.language,
        args.vad_thold
    );

    let mut pcmf32_old: Vec<f32> = Vec::new();
    let mut prompt_tokens: Vec<i32> = Vec::new();
    let mut n_iter = 0usize;

    // Helper to process a new audio slice through sliding window splicing and transcription
    let mut process_chunk = |pcmf32_new: &[f32], time_sec: f64| {
        let n_samples_new = pcmf32_new.len();
        // Splicing historical buffer pcmf32_old with incoming buffer pcmf32_new:
        // n_samples_take = min(|pcm_old|, max(0, n_samples_keep + n_samples_len - n_samples_new))
        let n_samples_take = (n_samples_keep + n_samples_len)
            .saturating_sub(n_samples_new)
            .min(pcmf32_old.len());

        let mut pcmf32 = Vec::with_capacity(n_samples_take + n_samples_new);
        if n_samples_take > 0 {
            let start = pcmf32_old.len() - n_samples_take;
            pcmf32.extend_from_slice(&pcmf32_old[start..]);
        }
        pcmf32.extend_from_slice(pcmf32_new);
        pcmf32_old = pcmf32.clone();

        let energy = compute_audio_energy(pcmf32_new);
        let vu = render_vu_meter(energy);

        if energy < args.vad_thold {
            print!("\x1B[2K\r[{:05.2} s] {} (Silence)", time_sec, vu);
            let _ = std::io::stdout().flush();
        } else {
            let params = WhisperParams {
                no_timestamps: !args.print_timestamps,
                audio_ctx: args.audio_ctx,
                language: args.language.clone(),
                translate: args.translate,
                prompt_tokens: if args.keep_context {
                    prompt_tokens.clone()
                } else {
                    Vec::new()
                },
                single_segment: true,
                ..Default::default()
            };

            let segments = ctx.transcribe_pcm(&pcmf32, &params);
            let text: String = segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ");

            if !text.is_empty() {
                if args.print_timestamps {
                    for seg in &segments {
                        println!("[{:05.2} --> {:05.2}] {}", seg.t0, seg.t1, seg.text.trim());
                    }
                } else {
                    print!(
                        "\x1B[2K\r[{:05.2} s] {} LIVE TRANSCRIPT >> \"{}\"",
                        time_sec, vu, text
                    );
                    let _ = std::io::stdout().flush();
                }
            }

            n_iter += 1;
            if n_iter.is_multiple_of(n_new_line) {
                println!();

                // Keep part of the audio for next iteration to mitigate word boundary issues
                let keep_start = pcmf32.len().saturating_sub(n_samples_keep);
                pcmf32_old = pcmf32[keep_start..].to_vec();

                // Add tokens of the last segment(s) as the autoregressive prompt
                if args.keep_context {
                    prompt_tokens.clear();
                    for seg in &segments {
                        prompt_tokens.extend_from_slice(&seg.tokens);
                    }
                    if prompt_tokens.len() > 224 {
                        let start = prompt_tokens.len() - 224;
                        prompt_tokens = prompt_tokens[start..].to_vec();
                    }
                }
            }
        }
    };

    if args.stdin {
        println!("Listening for raw 16kHz 16-bit LE PCM samples on STDIN...");
        let mut stdin = std::io::stdin();
        let mut byte_buf = vec![0u8; n_samples_step * 2];
        let mut total_samples_read = 0usize;

        loop {
            match stdin.read_exact(&mut byte_buf) {
                Ok(()) => {
                    let mut pcmf32_new = Vec::with_capacity(n_samples_step);
                    for bytes in byte_buf.as_chunks::<2>().0 {
                        let sample_i16 = i16::from_le_bytes([bytes[0], bytes[1]]);
                        pcmf32_new.push(sample_i16 as f32 / 32768.0);
                    }

                    total_samples_read += pcmf32_new.len();
                    let time_sec = total_samples_read as f64 / sample_rate as f64;
                    process_chunk(&pcmf32_new, time_sec);
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    println!("\nEnd of STDIN stream reached.");
                    break;
                }
                Err(e) => {
                    eprintln!("Error reading STDIN: {}", e);
                    break;
                }
            }
        }
    } else {
        println!("Opening stream source file: '{}'...", args.file);
        let audio = WavAudio::load_from_file(&args.file)?;
        let mut offset = 0;
        let total_samples = audio.samples.len();

        while offset < total_samples {
            let end = (offset + n_samples_step).min(total_samples);
            let pcmf32_new = &audio.samples[offset..end];
            offset = end;

            let time_sec = offset as f64 / sample_rate as f64;
            process_chunk(pcmf32_new, time_sec);

            sleep(Duration::from_millis(50));
        }
    }

    println!("\n=== STREAM COMPLETED SUCCESSFULLY ===");
    Ok(())
}
