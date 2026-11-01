use clap::Parser;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;
use whisper_rs_pure::audio::WavAudio;
use whisper_rs_pure::model::WhisperModel;
use whisper_rs_pure::{WhisperContext, WhisperParams};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "OpenAI-compatible HTTP REST server for whisper.rs"
)]
struct Args {
    /// Model file path (.bin)
    #[arg(short = 'm', long = "model", default_value = "models/ggml-tiny.en.bin")]
    model: String,

    /// Port to listen on
    #[arg(short = 'p', long = "port", default_value_t = 8080)]
    port: u16,

    /// Host IP address
    #[arg(long = "host", default_value = "127.0.0.1")]
    host: String,
}

fn handle_client(mut stream: TcpStream, ctx: Arc<WhisperContext>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return;
    }

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_length = 0usize;
    let mut is_multipart = false;
    let mut boundary = String::new();

    loop {
        let mut header_line = String::new();
        if reader.read_line(&mut header_line).is_err() || header_line.trim().is_empty() {
            break;
        }

        let line_lower = header_line.to_lowercase();
        if line_lower.starts_with("content-length:") {
            if let Some(val) = header_line.split(':').nth(1) {
                content_length = val.trim().parse::<usize>().unwrap_or(0);
            }
        }
        if line_lower.starts_with("content-type:") && line_lower.contains("multipart/form-data") {
            is_multipart = true;
            if let Some(pos) = header_line.find("boundary=") {
                boundary = header_line[pos + 9..].trim().to_string();
            }
        }
    }

    if method == "GET" && path == "/health" {
        let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
        let _ = stream.write_all(response.as_bytes());
        return;
    }

    if method == "POST" && (path == "/v1/audio/transcriptions" || path == "/v1/audio/translations")
    {
        let is_translation = path.contains("translations");

        let mut body = vec![0u8; content_length];
        if reader.read_exact(&mut body).is_err() {
            let response = "HTTP/1.1 400 Bad Request\r\n\r\nFailed to read body";
            let _ = stream.write_all(response.as_bytes());
            return;
        }

        // Extract WAV samples either from multipart or raw body
        let wav_data = if is_multipart && !boundary.is_empty() {
            extract_wav_from_multipart(&body)
        } else {
            &body[..]
        };

        match WavAudio::load_from_bytes(wav_data) {
            Ok(audio) => {
                let params = WhisperParams {
                    translate: is_translation,
                    no_timestamps: true,
                    ..Default::default()
                };

                let segments = ctx.transcribe_pcm(&audio.samples, &params);
                let full_text: String = segments
                    .iter()
                    .map(|s| s.text.trim())
                    .collect::<Vec<_>>()
                    .join(" ");

                let escaped = full_text.replace('\\', "\\\\").replace('"', "\\\"");
                let json_resp = format!("{{\"text\":\"{}\"}}\n", escaped);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    json_resp.len(),
                    json_resp
                );
                let _ = stream.write_all(response.as_bytes());
            }
            Err(e) => {
                let err_msg = format!("Failed to parse WAV audio: {}\n", e);
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                    err_msg.len(),
                    err_msg
                );
                let _ = stream.write_all(response.as_bytes());
            }
        }
        return;
    }

    let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
    let _ = stream.write_all(response.as_bytes());
}

fn extract_wav_from_multipart(body: &[u8]) -> &[u8] {
    // Search for RIFF header in multipart payload
    if let Some(pos) = body.windows(4).position(|w| w == b"RIFF") {
        return &body[pos..];
    }
    body
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading Whisper model from '{}'...", args.model);
    let model = WhisperModel::load_from_file(Path::new(&args.model))?;
    let ctx = Arc::new(WhisperContext::new(model)?);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr)?;
    println!("Whisper HTTP server listening on http://{}", addr);
    println!("Endpoints:");
    println!("  GET  /health");
    println!("  POST /v1/audio/transcriptions");
    println!("  POST /v1/audio/translations");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let ctx_clone = Arc::clone(&ctx);
                std::thread::spawn(move || {
                    handle_client(s, ctx_clone);
                });
            }
            Err(e) => {
                eprintln!("Connection failed: {}", e);
            }
        }
    }

    Ok(())
}
