#![forbid(unsafe_code)]

use clap::Parser;
use std::path::Path;
use std::process::Command;
use whisper_rs::audio::WavAudio;
use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Voice command recognizer and runner for whisper.rs"
)]
struct Args {
    /// Model file path (.bin)
    #[arg(short = 'm', long = "model", default_value = "models/ggml-tiny.en.bin")]
    model: String,

    /// Input WAV audio file containing voice command
    #[arg(short = 'f', long = "file", default_value = "samples/jfk.wav")]
    file: String,

    /// Keyword/phrase to detect
    #[arg(short = 'c', long = "command", default_value = "country")]
    command: String,

    /// Shell command to execute when keyword is detected
    #[arg(
        short = 'e',
        long = "exec",
        default_value = "echo '>>> Action triggered by voice keyword'"
    )]
    exec: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== WHISPER VOICE COMMAND RUNNER ===");
    println!("Loading model from '{}'...", args.model);
    let model = WhisperModel::load_from_file(Path::new(&args.model))?;
    let ctx = WhisperContext::new(model)?;

    println!("Loading audio command from '{}'...", args.file);
    let audio = WavAudio::load_from_file(&args.file)?;

    let params = WhisperParams {
        no_timestamps: true,
        ..Default::default()
    };

    let segments = ctx.transcribe_pcm(&audio.samples, &params);
    let transcript: String = segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");

    println!("\nRecognized Speech: \"{}\"", transcript);

    let lower_speech = transcript.to_lowercase();
    let lower_target = args.command.to_lowercase();

    if lower_speech.contains(&lower_target) {
        println!(
            "\n>>> MATCH! Keyword '{}' detected in speech.",
            args.command
        );
        println!(">>> Executing command: `{}`...", args.exec);
        let parts: Vec<&str> = args.exec.split_whitespace().collect();
        if let Some((bin, cmd_args)) = parts.split_first() {
            let status = Command::new(bin).args(cmd_args).status()?;
            println!(">>> Subprocess finished with exit status: {}", status);
        } else {
            println!(">>> Empty command, nothing to execute.");
        }
    } else {
        println!(
            "\n>>> NO MATCH: Target keyword '{}' not found in recognized speech.",
            args.command
        );
    }
    println!("=====================================");

    Ok(())
}
