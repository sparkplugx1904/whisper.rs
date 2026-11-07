#![forbid(unsafe_code)]

use clap::Parser;
use std::path::Path;
use std::time::Instant;
use whisper_rs::audio::compute_log_mel_spectrogram;
use whisper_rs::model::WhisperModel;
use whisper_rs::{DecoderKvCache, DecoderWorkspace, WhisperDecoder, WhisperEncoder};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Benchmark suite for whisper.rs (latency, throughput, RTF, thread scaling)"
)]
struct Args {
    /// Model file path (.bin)
    #[arg(short = 'm', long = "model", default_value = "models/ggml-tiny.en.bin")]
    model: String,

    /// Number of threads for computation
    #[arg(short = 't', long = "threads", default_value_t = 4)]
    threads: usize,

    /// Number of iterations for benchmark averages
    #[arg(short = 'n', long = "n-iter", default_value_t = 3)]
    n_iter: usize,

    /// Optional path to audio WAV file to benchmark end-to-end transcription
    #[arg(short = 'f', long = "file")]
    file: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Configure Rayon thread pool if specified
    if args.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global();
    }

    println!("=== WHISPER.RS BENCHMARK SUITE ===");
    println!("Threads: {}", args.threads);
    println!("Loading model from '{}'...", args.model);
    let t0 = Instant::now();
    let model = WhisperModel::load_from_file(Path::new(&args.model))?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Model load time: {:.2} ms (size: {:.2} MB)",
        load_ms,
        model.total_weight_bytes as f64 / 1e6
    );

    let encoder = WhisperEncoder::new(&model)?;
    let decoder = WhisperDecoder::new(&model)?;

    let n_audio_ctx = model.hparams.n_audio_ctx as usize;
    let n_mels = model.hparams.n_mels as usize;

    // 1. Audio / Mel Benchmark
    println!("\n[1/4] Benchmarking Mel Spectrogram Extraction...");
    let dummy_audio = vec![0.1f32; 16000 * 30]; // 30s audio
    let mut mel_times = Vec::new();
    for _ in 0..args.n_iter {
        let t = Instant::now();
        let _ = compute_log_mel_spectrogram(&dummy_audio, &model.filters.data, n_mels);
        mel_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let avg_mel_ms = mel_times.iter().sum::<f64>() / (mel_times.len() as f64);
    println!("  Avg Mel extraction (30s audio): {:.2} ms", avg_mel_ms);

    // 2. Encoder Benchmark
    println!("\n[2/4] Benchmarking Encoder Latency (1500 audio frames)...");
    let dummy_mel = vec![0.0f32; n_mels * 2 * n_audio_ctx];
    let mut enc_times = Vec::new();
    for _ in 0..args.n_iter {
        let t = Instant::now();
        let _ = encoder.forward(&dummy_mel);
        enc_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let avg_enc_ms = enc_times.iter().sum::<f64>() / (enc_times.len() as f64);
    println!("  Avg Encoder pass latency:       {:.2} ms", avg_enc_ms);

    // 3. Decoder Benchmark (per token)
    println!("\n[3/4] Benchmarking Decoder Latency (per-token generation)...");
    let enc_output = vec![0.0f32; n_audio_ctx * model.hparams.n_audio_state as usize];
    let mut cache = DecoderKvCache::new(decoder.n_layers, 1, decoder.d_model);
    let mut workspace = DecoderWorkspace::new();
    decoder.precompute_cross_attn(&enc_output, &mut cache);

    let n_tokens_bench = 16;
    let mut dec_times = Vec::new();
    for i in 0..n_tokens_bench {
        let t = Instant::now();
        let _ = decoder.decode(
            &[model.vocab.token_sot + i],
            &enc_output,
            &mut cache,
            &mut workspace,
        );
        dec_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let avg_dec_ms = dec_times.iter().sum::<f64>() / (dec_times.len() as f64);
    println!(
        "  Avg Decoder latency:            {:.2} ms / token",
        avg_dec_ms
    );

    // 4. Summary & RTF
    println!("\n[4/4] Summary Performance & Throughput:");
    let est_30s_decode_tokens = 50.0;
    let est_total_30s_ms = avg_mel_ms + avg_enc_ms + (avg_dec_ms * est_30s_decode_tokens);
    let rtf = (est_total_30s_ms / 1000.0) / 30.0;

    println!(
        "  Estimated 30s processing time:  {:.2} ms",
        est_total_30s_ms
    );
    println!("  Real-Time Factor (RTF):         {:.3}x", rtf);
    println!(
        "  Throughput:                     {:.2} tokens / sec",
        1000.0 / avg_dec_ms
    );
    println!("==================================");

    // 5. Optional full WAV transcription benchmark
    if let Some(wav_file) = &args.file {
        println!("\n[5/5] Benchmarking Real Audio File: '{}'...", wav_file);
        let audio = whisper_rs::audio::WavAudio::load_from_file(wav_file)?;
        let audio_dur = audio.samples.len() as f64 / audio.sample_rate as f64;
        let ctx = whisper_rs::WhisperContext::new(model)?;
        let params = whisper_rs::WhisperParams {
            n_threads: args.threads,
            ..Default::default()
        };

        let mut e2e_times = Vec::new();
        let mut total_tokens = 0;
        for _ in 0..args.n_iter {
            let t = Instant::now();
            let segments = ctx.transcribe_pcm(&audio.samples, &params);
            e2e_times.push(t.elapsed().as_secs_f64() * 1000.0);
            total_tokens = segments
                .iter()
                .map(|s| s.text.split_whitespace().count())
                .sum();
        }
        let avg_e2e_ms = e2e_times.iter().sum::<f64>() / (e2e_times.len() as f64);
        let e2e_rtf = (avg_e2e_ms / 1000.0) / audio_dur;
        println!("  Audio duration:                 {:.2} s", audio_dur);
        println!("  Avg End-to-End processing:      {:.2} ms", avg_e2e_ms);
        println!("  End-to-End RTF:                 {:.3}x", e2e_rtf);
        if total_tokens > 0 {
            println!(
                "  End-to-End Throughput:          {:.2} words / sec",
                (total_tokens as f64) / (avg_e2e_ms / 1000.0)
            );
        }
        println!("==================================");
    }

    Ok(())
}
