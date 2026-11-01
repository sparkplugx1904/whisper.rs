use clap::Parser;
use std::path::Path;
use std::time::Instant;
use whisper_rs_pure::audio::compute_log_mel_spectrogram;
use whisper_rs_pure::model::WhisperModel;
use whisper_rs_pure::{DecoderKvCache, WhisperDecoder, WhisperEncoder};

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

    /// Number of iterations for benchmark averages
    #[arg(short = 'n', long = "n-iter", default_value_t = 3)]
    n_iter: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== WHISPER.RS BENCHMARK SUITE ===");
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
    let mut cache = DecoderKvCache::new(decoder.n_layers);
    decoder.precompute_cross_attn(&enc_output, &mut cache);

    let n_tokens_bench = 16;
    let mut dec_times = Vec::new();
    for i in 0..n_tokens_bench {
        let t = Instant::now();
        let _ = decoder.decode(&[model.vocab.token_sot + i], &enc_output, &mut cache);
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

    Ok(())
}
