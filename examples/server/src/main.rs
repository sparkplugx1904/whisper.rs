#![forbid(unsafe_code)]
use clap::Parser;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;

use whisper_rs::model::WhisperModel;
use whisper_rs::{WhisperContext, WhisperParams};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "OpenAI-compatible HTTP REST audio transcription server for whisper.rs"
)]
struct Args {
    /// Model file path (.bin)
    #[arg(short = 'm', long = "model", default_value = "models/ggml-tiny.en.bin")]
    model: String,

    /// Host address to bind
    #[arg(long = "host", default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short = 'p', long = "port", default_value_t = 8080)]
    port: u16,
}

fn extract_field_from_multipart<'a>(body: &'a [u8], field_name: &str) -> Option<&'a str> {
    let needle = format!("name=\"{}\"", field_name);
    let needle_bytes = needle.as_bytes();
    if let Some(pos) = body
        .windows(needle_bytes.len())
        .position(|w| w == needle_bytes)
    {
        let rem = &body[pos + needle_bytes.len()..];
        if let Some(header_end) = rem.windows(4).position(|w| w == b"\r\n\r\n") {
            let val_part = &rem[header_end + 4..];
            if let Some(val_end) = val_part.windows(2).position(|w| w == b"\r\n") {
                return std::str::from_utf8(&val_part[..val_end]).ok();
            }
        }
    }
    None
}

fn extract_wav_from_multipart(body: &[u8]) -> &[u8] {
    if let Some(pos) = body.windows(4).position(|w| w == b"RIFF") {
        if pos + 8 <= body.len() {
            let riff_len =
                u32::from_le_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]])
                    as usize;
            let total_wav_len = (riff_len + 8).min(body.len() - pos);
            return &body[pos..pos + total_wav_len];
        }
        return &body[pos..];
    }
    body
}

fn handle_client(mut stream: TcpStream, ctx: Arc<WhisperContext>) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });

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

    let mut content_length: usize = 0;
    let mut is_multipart = false;
    let mut boundary = String::new();

    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
            break;
        }
        let lower = line.to_lowercase();
        if lower.starts_with("content-length:") {
            if let Some(val) = line.split(':').nth(1) {
                content_length = val.trim().parse().unwrap_or(0);
            }
        } else if lower.starts_with("content-type:") && lower.contains("multipart/form-data") {
            is_multipart = true;
            if let Some(b_pos) = line.find("boundary=") {
                boundary = line[b_pos + 9..].trim().to_string();
            }
        }
    }

    const MAX_BODY_SIZE: usize = 64 * 1024 * 1024; // 64 MB
    if content_length > MAX_BODY_SIZE {
        let response = "HTTP/1.1 413 Payload Too Large\r\n\r\nPayload Too Large";
        let _ = stream.write_all(response.as_bytes());
        return;
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

        let wav_data = if is_multipart && !boundary.is_empty() {
            extract_wav_from_multipart(&body)
        } else {
            &body[..]
        };

        // Extract optional OpenAI parameters
        let language = extract_field_from_multipart(&body, "language").map(|s| s.to_string());
        let prompt = extract_field_from_multipart(&body, "prompt").map(|s| s.to_string());
        let temperature: f32 = extract_field_from_multipart(&body, "temperature")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let response_format =
            extract_field_from_multipart(&body, "response_format").unwrap_or("json");

        match whisper_rs::audio::load_audio_from_bytes(wav_data) {
            Ok(audio) => {
                let params = WhisperParams {
                    translate: is_translation,
                    language: language.unwrap_or_else(|| "en".to_string()),
                    initial_prompt: prompt,
                    temperature,
                    no_timestamps: response_format == "json" || response_format == "text",
                    ..Default::default()
                };

                let segments = ctx.transcribe_pcm(&audio.samples, &params);

                let (content_type, body_str) = match response_format {
                    "text" => {
                        let text = segments
                            .iter()
                            .map(|s| s.text.trim())
                            .collect::<Vec<_>>()
                            .join(" ");
                        ("text/plain", text)
                    }
                    "srt" => {
                        let mut srt_out = String::new();
                        for (i, seg) in segments.iter().enumerate() {
                            srt_out.push_str(&seg.to_srt(i + 1));
                        }
                        ("text/plain", srt_out)
                    }
                    "vtt" => {
                        let mut vtt_out = String::from("WEBVTT\n\n");
                        for seg in &segments {
                            vtt_out.push_str(&seg.to_vtt());
                        }
                        ("text/vtt", vtt_out)
                    }
                    _ => {
                        let full_text: String = segments
                            .iter()
                            .map(|s| s.text.trim())
                            .collect::<Vec<_>>()
                            .join(" ");
                        let escaped = full_text.replace('\\', "\\\\").replace('"', "\\\"");
                        (
                            "application/json",
                            format!("{{\"text\":\"{}\"}}\n", escaped),
                        )
                    }
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n{}",
                    content_type,
                    body_str.len(),
                    body_str
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
