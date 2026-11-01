use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Pure-Rust Whisper CLI (whisper.cpp port)", long_about = None)]
struct Args {
    /// Path to model file (.bin)
    #[arg(short = 'm', long = "model")]
    model: String,

    /// Path to input audio WAV file
    #[arg(short = 'f', long = "file")]
    file: String,

    /// Language (e.g. en, id, auto)
    #[arg(short = 'l', long = "language", default_value = "auto")]
    language: String,

    /// Number of threads
    #[arg(short = 't', long = "threads", default_value_t = 4)]
    threads: usize,

    /// Translate from source language to English
    #[arg(long = "translate", default_value_t = false)]
    translate: bool,
}

fn main() {
    let args = Args::parse();
    println!(
        "whisper-cli (pure Rust) - model: {}, file: {}",
        args.model, args.file
    );
}
