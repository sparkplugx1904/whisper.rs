use clap::Parser;
use std::path::Path;
use whisper_rs_pure::model::WhisperModel;

#[derive(Parser, Debug)]
#[command(author, version, about = "Pure-Rust Whisper CLI (whisper.cpp port)", long_about = None)]
struct Args {
    /// Path to model file (.bin)
    #[arg(short = 'm', long = "model")]
    model: String,

    /// Path to input audio WAV file
    #[arg(short = 'f', long = "file", default_value = "")]
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

    /// Print model information and exit
    #[arg(long = "info", default_value_t = false)]
    info: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading model from '{}'...", args.model);
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

    if !args.file.is_empty() {
        println!("Audio transcription not yet executed (next phases).");
    }

    Ok(())
}
