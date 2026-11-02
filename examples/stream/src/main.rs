#![forbid(unsafe_code)]

use clap::Parser;
use std::collections::VecDeque;
use std::io::Read;
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
    about = "Real-time streaming audio transcriber with sliding window & voice energy threshold"
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

    /// Chunk step in milliseconds
    #[arg(long = "step-ms", default_value_t = 3000)]
    step_ms: usize,

    /// Length of audio window in seconds
    #[arg(long = "length-s", default_value_t = 10)]
    length_s: usize,

    /// Voice energy detection threshold
    #[arg(long = "vad-thold", default_value_t = 0.005)]
    vad_thold: f32,
}

fn compute_audio_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / (samples.len() as f32)).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== REAL-TIME STREAMING AUDIO TRANSCRIBER ===");
    println!("Loading Whisper model from '{}'...", args.model);
    let model = WhisperModel::load_from_file(Path::new(&args.model))?;
    let ctx = WhisperContext::new(model)?;

    let sample_rate = 16000usize;
    let chunk_size = (sample_rate * args.step_ms) / 1000;
    let window_capacity = sample_rate * args.length_s;

    let mut ring_buffer: VecDeque<f32> = VecDeque::with_capacity(window_capacity);

    let params = WhisperParams {
        no_timestamps: true,
        ..Default::default()
    };

    println!(
        "Stream started: mode = {}, window = {} s, step = {} ms, VAD threshold = {:.4}\n",
        if args.stdin {
            "live STDIN pipe"
        } else {
            "file stream simulation"
        },
        args.length_s,
        args.step_ms,
        args.vad_thold
    );

    if args.stdin {
        println!("Listening for raw 16kHz 16-bit LE PCM samples on STDIN...");
        let mut stdin = std::io::stdin();
        let mut byte_buf = vec![0u8; chunk_size * 2];
        let mut total_samples_read = 0usize;

        loop {
            match stdin.read_exact(&mut byte_buf) {
                Ok(()) => {
                    let mut chunk = Vec::with_capacity(chunk_size);
                    for bytes in byte_buf.as_chunks::<2>().0 {
                        let sample_i16 = i16::from_le_bytes([bytes[0], bytes[1]]);
                        chunk.push(sample_i16 as f32 / 32768.0);
                    }

                    for &s in &chunk {
                        if ring_buffer.len() == window_capacity {
                            ring_buffer.pop_front();
                        }
                        ring_buffer.push_back(s);
                    }

                    total_samples_read += chunk.len();
                    let time_sec = total_samples_read as f64 / sample_rate as f64;
                    let energy = compute_audio_energy(&chunk);

                    if energy >= args.vad_thold {
                        let window_samples: Vec<f32> = ring_buffer.iter().copied().collect();
                        let segments = ctx.transcribe_pcm(&window_samples, &params);
                        let text: String = segments
                            .iter()
                            .map(|s| s.text.trim())
                            .collect::<Vec<_>>()
                            .join(" ");

                        if !text.is_empty() {
                            println!("[{:05.2} s] LIVE TRANSCRIPT >> \"{}\"", time_sec, text);
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
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
            let end = (offset + chunk_size).min(total_samples);
            let chunk = &audio.samples[offset..end];
            offset = end;

            for &s in chunk {
                if ring_buffer.len() == window_capacity {
                    ring_buffer.pop_front();
                }
                ring_buffer.push_back(s);
            }

            let energy = compute_audio_energy(chunk);
            let time_sec = offset as f64 / sample_rate as f64;

            if energy < args.vad_thold {
                println!(
                    "[{:05.2} s] (Silence / Energy {:.4} < thold)",
                    time_sec, energy
                );
                continue;
            }

            let window_samples: Vec<f32> = ring_buffer.iter().copied().collect();
            let segments = ctx.transcribe_pcm(&window_samples, &params);
            let text: String = segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ");

            if !text.is_empty() {
                println!("[{:05.2} s] LIVE TRANSCRIPT >> \"{}\"", time_sec, text);
            }

            sleep(Duration::from_millis(50));
        }
    }

    println!("\n=== STREAM COMPLETED SUCCESSFULLY ===");
    Ok(())
}
