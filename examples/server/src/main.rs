#![forbid(unsafe_code)]
use clap::Parser;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use whisper_rs::model::WhisperModel;
use whisper_rs::segment::{escape_json, Segment};
use whisper_rs::{WhisperContext, WhisperParams};

/// Shared server context supporting atomic hot-reloading.
pub type SharedContext = Arc<RwLock<Arc<WhisperContext>>>;

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "OpenAI-compatible HTTP REST audio transcription server for whisper.rs"
)]
pub struct Args {
    /// Model file path (.bin)
    #[arg(short = 'm', long = "model", default_value = "models/ggml-tiny.en.bin")]
    pub model: String,

    /// Host address to bind (e.g. 127.0.0.1 or a Unix domain socket path ending in .sock)
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on
    #[arg(short = 'p', long = "port", default_value_t = 8080)]
    pub port: u16,

    /// Request path prefix for all endpoints (e.g. /api/v1)
    #[arg(long = "request-path", default_value = "")]
    pub request_path: String,

    /// Path to public directory for serving static assets (HTML/CSS/JS)
    #[arg(long = "public")]
    pub public: Option<String>,
}

/// Server routing and static configuration.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub request_path: String,
    pub public_dir: Option<PathBuf>,
    pub port: u16,
}

impl ServerConfig {
    pub fn new(request_path: &str, public_dir: Option<PathBuf>, port: u16) -> Self {
        Self {
            request_path: normalize_request_path(request_path),
            public_dir,
            port,
        }
    }
}

/// Normalizes the request path prefix (e.g. `api/v1` -> `/api/v1`, `/` -> `""`).
pub fn normalize_request_path(path: &str) -> String {
    let trimmed = path.trim().trim_end_matches('/');
    if trimmed.is_empty() || trimmed == "/" {
        String::new()
    } else if !trimmed.starts_with('/') {
        format!("/{}", trimmed)
    } else {
        trimmed.to_string()
    }
}

/// Wrapped client connection enabling disconnect detection during audio transcription.
pub struct ClientConnection<R, W> {
    pub reader: BufReader<R>,
    pub writer: W,
    pub is_disconnected: Arc<dyn Fn() -> bool + Send + Sync>,
    pub on_start_transcription: Option<Arc<dyn Fn() + Send + Sync>>,
    pub on_finish_transcription: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl ClientConnection<TcpStream, TcpStream> {
    pub fn from_tcp_stream(stream: TcpStream) -> std::io::Result<Self> {
        let reader_stream = stream.try_clone()?;
        let writer_stream = stream;
        let peek_stream = writer_stream.try_clone()?;

        let is_disconnected: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
            let mut buf = [0u8; 1];
            match peek_stream.peek(&mut buf) {
                Ok(0) => true, // Client cleanly closed connection
                Ok(_) => false,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => false,
                Err(_) => true, // Connection reset / broken pipe
            }
        });

        let s_start = writer_stream.try_clone()?;
        let on_start_transcription: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let _ = s_start.set_nonblocking(true);
        });

        let s_finish = writer_stream.try_clone()?;
        let on_finish_transcription: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let _ = s_finish.set_nonblocking(false);
        });

        Ok(Self {
            reader: BufReader::new(reader_stream),
            writer: writer_stream,
            is_disconnected,
            on_start_transcription: Some(on_start_transcription),
            on_finish_transcription: Some(on_finish_transcription),
        })
    }
}

#[cfg(unix)]
impl ClientConnection<std::os::unix::net::UnixStream, std::os::unix::net::UnixStream> {
    pub fn from_unix_stream(stream: std::os::unix::net::UnixStream) -> std::io::Result<Self> {
        let reader_stream = stream.try_clone()?;
        let writer_stream = stream;
        let check_stream = writer_stream.try_clone()?;

        let is_disconnected: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
            let mut buf = [0u8; 1];
            let mut stream_ref = &check_stream;
            match stream_ref.read(&mut buf) {
                Ok(0) => true,
                Ok(_) => false,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => false,
                Err(_) => true,
            }
        });

        let s_start = writer_stream.try_clone()?;
        let on_start_transcription: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let _ = s_start.set_nonblocking(true);
        });

        let s_finish = writer_stream.try_clone()?;
        let on_finish_transcription: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let _ = s_finish.set_nonblocking(false);
        });

        Ok(Self {
            reader: BufReader::new(reader_stream),
            writer: writer_stream,
            is_disconnected,
            on_start_transcription: Some(on_start_transcription),
            on_finish_transcription: Some(on_finish_transcription),
        })
    }
}

impl<R: Read, W: Write> ClientConnection<R, W> {
    pub fn new_mock(
        reader: R,
        writer: W,
        is_disconnected: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> Self {
        Self {
            reader: BufReader::new(reader),
            writer,
            is_disconnected,
            on_start_transcription: None,
            on_finish_transcription: None,
        }
    }
}

/// Send an HTTP response with standard global CORS headers.
pub fn send_response<W: Write>(writer: &mut W, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: content-type, authorization\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n",
        status,
        content_type,
        body.len()
    );
    let _ = writer.write_all(header.as_bytes());
    let _ = writer.write_all(body);
}

/// Extract a textual field from a multipart/form-data body.
pub fn extract_field_from_multipart<'a>(body: &'a [u8], field_name: &str) -> Option<&'a str> {
    let needle1 = format!("name=\"{}\"", field_name);
    let needle2 = format!("name='{}'", field_name);
    let needle3 = format!("name={}", field_name);

    let pos = body
        .windows(needle1.len())
        .position(|w| w == needle1.as_bytes())
        .or_else(|| {
            body.windows(needle2.len())
                .position(|w| w == needle2.as_bytes())
        })
        .or_else(|| {
            body.windows(needle3.len())
                .position(|w| w == needle3.as_bytes())
        })?;

    let rem = &body[pos..];
    let header_end_offset = if let Some(p) = rem.windows(4).position(|w| w == b"\r\n\r\n") {
        p + 4
    } else {
        let p = rem.windows(2).position(|w| w == b"\n\n")?;
        p + 2
    };

    let val_part = &rem[header_end_offset..];
    let end_pos = if let Some(b_pos) = val_part.windows(4).position(|w| w == b"\r\n--") {
        b_pos
    } else if let Some(b_pos) = val_part.windows(3).position(|w| w == b"\n--") {
        b_pos
    } else if let Some(nl_pos) = val_part.windows(2).position(|w| w == b"\r\n") {
        nl_pos
    } else if let Some(nl_pos) = val_part.iter().position(|&b| b == b'\n') {
        nl_pos
    } else {
        val_part.len()
    };

    let raw_str = std::str::from_utf8(&val_part[..end_pos]).ok()?.trim();
    let cleaned = raw_str.trim_matches('"').trim_matches('\'').trim();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Extract audio byte payload from multipart/form-data or raw request body.
pub fn extract_wav_from_multipart(body: &[u8]) -> &[u8] {
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

    let needle = b"name=\"file\"";
    if let Some(pos) = body.windows(needle.len()).position(|w| w == needle) {
        let rem = &body[pos + needle.len()..];
        if let Some(header_end) = rem.windows(4).position(|w| w == b"\r\n\r\n") {
            let file_data = &rem[header_end + 4..];
            if let Some(boundary_pos) = file_data.windows(4).position(|w| w == b"\r\n--") {
                return &file_data[..boundary_pos];
            }
            return file_data;
        }
    }

    body
}

/// Determine MIME type for static files based on extension.
pub fn mime_type_from_path(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()).unwrap_or("") {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "json" => "application/json",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Serve a static asset from the public directory with path-traversal prevention.
pub fn serve_static_file<W: Write>(writer: &mut W, public_dir: &Path, subpath: &str) -> bool {
    let subpath_clean = subpath.trim_start_matches('/');
    if subpath_clean.contains("..") {
        return false;
    }

    let target_path = if subpath_clean.is_empty() {
        public_dir.join("index.html")
    } else {
        public_dir.join(subpath_clean)
    };

    let Ok(canonical_public) = public_dir.canonicalize() else {
        return false;
    };
    let Ok(canonical_target) = target_path.canonicalize() else {
        return false;
    };

    if !canonical_target.starts_with(&canonical_public) || !canonical_target.is_file() {
        return false;
    }

    let Ok(bytes) = std::fs::read(&canonical_target) else {
        return false;
    };

    let content_type = mime_type_from_path(&canonical_target);
    send_response(writer, "200 OK", content_type, &bytes);
    true
}

/// Generate default embedded HTML test form matching whisper.cpp server.
pub fn get_default_html_form(port: u16, prefix: &str) -> String {
    let action_path = if prefix.is_empty() {
        "/v1/audio/transcriptions".to_string()
    } else {
        format!("{}/v1/audio/transcriptions", prefix)
    };
    let load_path = if prefix.is_empty() {
        "/load".to_string()
    } else {
        format!("{}/load", prefix)
    };

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Whisper.cpp Server</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 2rem auto;
            max-width: 800px;
            padding: 0 1rem;
            line-height: 1.5;
            color: #333;
        }}
        h1 {{ border-bottom: 2px solid #eaeaea; padding-bottom: 0.5rem; }}
        h2 {{ margin-top: 1.5rem; }}
        form {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            background: #fdfdfd;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 1.5rem;
        }}
        label {{ font-weight: 600; }}
        input, select, button {{
            padding: 0.5rem;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
        button {{
            background: #0366d6;
            color: white;
            font-weight: 600;
            border: none;
            cursor: pointer;
            padding: 0.75rem 1.5rem;
            align-self: flex-start;
        }}
        button:hover {{ background: #0255b3; }}
        pre {{
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
            font-family: monospace;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <h1>Whisper Server</h1>

    <h2>Transcriptions Endpoint</h2>
    <pre>
curl 127.0.0.1:{port}{action_path} \
  -H "Content-Type: multipart/form-data" \
  -F file="@&lt;audio-file.wav&gt;" \
  -F temperature="0.0" \
  -F response_format="verbose_json"
    </pre>

    <h2>Model Hot-Reloading (/load)</h2>
    <pre>
curl 127.0.0.1:{port}{load_path} \
  -H "Content-Type: multipart/form-data" \
  -F model="&lt;path-to-model-file&gt;"
    </pre>

    <div>
        <h2>Try it out</h2>
        <form action="{action_path}" method="POST" enctype="multipart/form-data">
            <label for="file">Choose an audio file (WAV):</label>
            <input type="file" id="file" name="file" accept="audio/*" required>

            <label for="temperature">Temperature:</label>
            <input type="number" id="temperature" name="temperature" value="0.0" step="0.01" placeholder="e.g. 0.0">

            <label for="response_format">Response Format:</label>
            <select id="response_format" name="response_format">
                <option value="verbose_json" selected>Verbose JSON</option>
                <option value="json">JSON</option>
                <option value="text">Text</option>
                <option value="srt">SRT</option>
                <option value="vtt">VTT</option>
            </select>

            <button type="submit">Transcribe</button>
        </form>
    </div>
</body>
</html>
"#
    )
}

/// Format transcription result matching full OpenAI-compatible `verbose_json` schema.
pub fn format_verbose_json(
    segments: &[Segment],
    duration: f64,
    language: &str,
    is_translation: bool,
) -> String {
    let mut segs_json = Vec::with_capacity(segments.len());
    let mut all_words = Vec::new();

    for (i, seg) in segments.iter().enumerate() {
        let mut words_json = Vec::with_capacity(seg.words.len());
        for w in &seg.words {
            let w_str = format!(
                "{{\"word\":\"{}\",\"start\":{:.3},\"end\":{:.3},\"probability\":{:.4}}}",
                escape_json(&w.word),
                w.t0,
                w.t1,
                w.probability
            );
            all_words.push(w_str.clone());
            words_json.push(w_str);
        }

        let tokens_str = seg
            .tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let seg_str = format!(
            "{{\"id\":{},\"start\":{:.3},\"end\":{:.3},\"text\":\"{}\",\"tokens\":[{}],\"words\":[{}],\"avg_logprob\":{:.4},\"no_speech_prob\":{:.4}}}",
            i,
            seg.t0,
            seg.t1,
            escape_json(seg.text.trim()),
            tokens_str,
            words_json.join(","),
            seg.avg_logprob,
            seg.no_speech_prob
        );
        segs_json.push(seg_str);
    }

    let full_text: String = segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");

    let task_str = if is_translation {
        "translate"
    } else {
        "transcribe"
    };

    format!(
        "{{\"task\":\"{}\",\"language\":\"{}\",\"duration\":{:.2},\"text\":\"{}\",\"words\":[{}],\"segments\":[{}]}}\n",
        task_str,
        escape_json(language),
        duration,
        escape_json(&full_text),
        all_words.join(","),
        segs_json.join(",")
    )
}

/// Handle a single incoming HTTP connection.
pub fn handle_client<R: Read, W: Write>(
    mut conn: ClientConnection<R, W>,
    ctx_holder: SharedContext,
    config: &ServerConfig,
) {
    let mut request_line = String::new();
    if conn.reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return;
    }

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let full_path = parts[1];
    let path = full_path.split('?').next().unwrap_or(full_path);

    let mut content_length: usize = 0;
    let mut is_multipart = false;
    let mut boundary = String::new();

    loop {
        let mut line = String::new();
        if conn.reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
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

    // Handle CORS preflight
    if method == "OPTIONS" {
        send_response(&mut conn.writer, "204 No Content", "text/plain", b"");
        return;
    }

    const MAX_BODY_SIZE: usize = 64 * 1024 * 1024; // 64 MB
    if content_length > MAX_BODY_SIZE {
        send_response(
            &mut conn.writer,
            "413 Payload Too Large",
            "text/plain; charset=utf-8",
            b"Payload Too Large\n",
        );
        return;
    }

    let prefix = &config.request_path;

    // Health check endpoint
    let is_health =
        path == "/health" || (!prefix.is_empty() && path == format!("{}/health", prefix));
    if method == "GET" && is_health {
        send_response(
            &mut conn.writer,
            "200 OK",
            "application/json",
            b"{\"status\":\"ok\"}\n",
        );
        return;
    }

    // Dynamic model hot-reloading (POST /load or POST ${request_path}/load)
    let is_load = path == "/load" || (!prefix.is_empty() && path == format!("{}/load", prefix));
    if method == "POST" && is_load {
        let mut body = vec![0u8; content_length];
        if conn.reader.read_exact(&mut body).is_err() {
            send_response(
                &mut conn.writer,
                "400 Bad Request",
                "application/json",
                b"{\"error\":\"Failed to read body\"}\n",
            );
            return;
        }

        let model_field = extract_field_from_multipart(&body, "model")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                // Optional fallback for JSON / urlencoded payloads
                if let Ok(s) = std::str::from_utf8(&body) {
                    let s = s.trim();
                    if let Some(pos) = s.find("\"model\"") {
                        let rem = &s[pos + 7..];
                        if let Some(colon) = rem.find(':') {
                            let val_rem = rem[colon + 1..].trim();
                            if let Some(stripped) = val_rem.strip_prefix('"') {
                                if let Some(end_quote) = stripped.find('"') {
                                    return Some(&stripped[..end_quote]);
                                }
                            }
                        }
                    } else if let Some(pos) = s.find("model=") {
                        let val_rem = &s[pos + 6..];
                        let end = val_rem.find('&').unwrap_or(val_rem.len());
                        return Some(&val_rem[..end]);
                    }
                }
                None
            });

        let model_path_str = match model_field {
            Some(p) => p,
            None => {
                eprintln!("error: no 'model' field in the request");
                send_response(
                    &mut conn.writer,
                    "400 Bad Request",
                    "application/json",
                    b"{\"error\":\"no 'model' field in the request\"}\n",
                );
                return;
            }
        };

        let model_path = Path::new(model_path_str);
        if !model_path.exists() {
            eprintln!("error: 'model': {} not found!", model_path_str);
            send_response(
                &mut conn.writer,
                "400 Bad Request",
                "application/json",
                b"{\"error\":\"model not found!\"}\n",
            );
            return;
        }

        let mut write_guard = match ctx_holder.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        match WhisperModel::load_from_file(model_path) {
            Ok(new_model) => match WhisperContext::new(new_model) {
                Ok(new_ctx) => {
                    *write_guard = Arc::new(new_ctx);
                    drop(write_guard);
                    send_response(
                        &mut conn.writer,
                        "200 OK",
                        "text/plain; charset=utf-8",
                        b"Load was successful!\n",
                    );
                }
                Err(e) => {
                    eprintln!("error: model context init failed: {}", e);
                    let err = format!(
                        "{{\"error\":\"failed to create whisper context: {}\"}}\n",
                        e
                    );
                    send_response(
                        &mut conn.writer,
                        "500 Internal Server Error",
                        "application/json",
                        err.as_bytes(),
                    );
                }
            },
            Err(e) => {
                eprintln!("error: model load failed: {}", e);
                let err = format!("{{\"error\":\"failed to load model from file: {}\"}}\n", e);
                send_response(
                    &mut conn.writer,
                    "500 Internal Server Error",
                    "application/json",
                    err.as_bytes(),
                );
            }
        }
        return;
    }

    // Audio transcription / translation endpoints
    let is_transcription = path == "/v1/audio/transcriptions"
        || path == "/inference"
        || (!prefix.is_empty()
            && (path == format!("{}/v1/audio/transcriptions", prefix)
                || path == format!("{}/inference", prefix)));
    let is_translation = path == "/v1/audio/translations"
        || (!prefix.is_empty() && path == format!("{}/v1/audio/translations", prefix));

    if method == "POST" && (is_transcription || is_translation) {
        let mut body = vec![0u8; content_length];
        if conn.reader.read_exact(&mut body).is_err() {
            send_response(
                &mut conn.writer,
                "400 Bad Request",
                "application/json",
                b"{\"error\":\"Failed to read body\"}\n",
            );
            return;
        }

        let wav_data = if is_multipart && !boundary.is_empty() {
            extract_wav_from_multipart(&body)
        } else {
            &body[..]
        };

        let language = extract_field_from_multipart(&body, "language").map(|s| s.to_string());
        let prompt = extract_field_from_multipart(&body, "prompt").map(|s| s.to_string());
        let temperature: f32 = extract_field_from_multipart(&body, "temperature")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let response_format =
            extract_field_from_multipart(&body, "response_format").unwrap_or("json");

        match whisper_rs::audio::load_audio_from_bytes(wav_data) {
            Ok(audio) => {
                let current_ctx = {
                    let guard = match ctx_holder.read() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    Arc::clone(&*guard)
                };

                if let Some(on_start) = &conn.on_start_transcription {
                    on_start();
                }

                let aborted_by_client = Arc::new(AtomicBool::new(false));
                let aborted_flag = Arc::clone(&aborted_by_client);
                let is_disconnected_fn = Arc::clone(&conn.is_disconnected);

                let abort_cb: whisper_rs::params::AbortCallback = Arc::new(move || {
                    if is_disconnected_fn() {
                        aborted_flag.store(true, Ordering::SeqCst);
                        true
                    } else {
                        false
                    }
                });

                let params = WhisperParams {
                    translate: is_translation,
                    language: language.unwrap_or_else(|| "en".to_string()),
                    initial_prompt: prompt,
                    temperature,
                    no_timestamps: response_format == "json" || response_format == "text",
                    abort_callback: Some(abort_cb),
                    ..Default::default()
                };

                let segments = current_ctx.transcribe_pcm(&audio.samples, &params);

                if let Some(on_finish) = &conn.on_finish_transcription {
                    on_finish();
                }

                // Check for client disconnect during transcription
                if aborted_by_client.load(Ordering::SeqCst) {
                    eprintln!("client disconnected, aborted processing");
                    send_response(
                        &mut conn.writer,
                        "499 Client Closed Request",
                        "application/json",
                        b"{\"error\":\"client disconnected\"}\n",
                    );
                    return;
                }

                let duration = audio.samples.len() as f64 / audio.sample_rate as f64;
                let (content_type, body_str) = match response_format {
                    "text" => {
                        let text = segments
                            .iter()
                            .map(|s| s.text.trim())
                            .collect::<Vec<_>>()
                            .join(" ");
                        ("text/plain; charset=utf-8", text)
                    }
                    "srt" => {
                        let mut srt_out = String::new();
                        for (i, seg) in segments.iter().enumerate() {
                            srt_out.push_str(&seg.to_srt(i + 1));
                        }
                        ("text/plain; charset=utf-8", srt_out)
                    }
                    "vtt" => {
                        let mut vtt_out = String::from("WEBVTT\n\n");
                        for seg in &segments {
                            vtt_out.push_str(&seg.to_vtt());
                        }
                        ("text/vtt; charset=utf-8", vtt_out)
                    }
                    "verbose_json" => {
                        let vjson = format_verbose_json(
                            &segments,
                            duration,
                            &params.language,
                            is_translation,
                        );
                        ("application/json", vjson)
                    }
                    _ => {
                        let full_text: String = segments
                            .iter()
                            .map(|s| s.text.trim())
                            .collect::<Vec<_>>()
                            .join(" ");
                        (
                            "application/json",
                            format!("{{\"text\":\"{}\"}}\n", escape_json(&full_text)),
                        )
                    }
                };

                send_response(
                    &mut conn.writer,
                    "200 OK",
                    content_type,
                    body_str.as_bytes(),
                );
            }
            Err(e) => {
                let err_msg = format!("Failed to parse WAV audio: {}\n", e);
                send_response(
                    &mut conn.writer,
                    "400 Bad Request",
                    "text/plain; charset=utf-8",
                    err_msg.as_bytes(),
                );
            }
        }
        return;
    }

    // Static assets & Root HTML test form
    if method == "GET" {
        let is_root = path == "/"
            || path.is_empty()
            || (!prefix.is_empty() && (path == prefix || path == format!("{}/", prefix)));

        if is_root {
            if let Some(pub_dir) = &config.public_dir {
                if serve_static_file(&mut conn.writer, pub_dir, "index.html") {
                    return;
                }
            }
            // Serve embedded default HTML test form when no public index exists
            let html = get_default_html_form(config.port, prefix);
            send_response(
                &mut conn.writer,
                "200 OK",
                "text/html; charset=utf-8",
                html.as_bytes(),
            );
            return;
        }

        if let Some(pub_dir) = &config.public_dir {
            let rel_path = if !prefix.is_empty() && path.starts_with(prefix) {
                path.strip_prefix(prefix).unwrap_or(path)
            } else {
                path
            };
            if serve_static_file(&mut conn.writer, pub_dir, rel_path) {
                return;
            }
        }
    }

    // 404 Not Found fallback
    let not_found_body = format!("File Not Found ({})\n", path);
    send_response(
        &mut conn.writer,
        "404 Not Found",
        "text/plain; charset=utf-8",
        not_found_body.as_bytes(),
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading Whisper model from '{}'...", args.model);
    let model = WhisperModel::load_from_file(Path::new(&args.model))?;
    let ctx = Arc::new(WhisperContext::new(model)?);
    let shared_ctx: SharedContext = Arc::new(RwLock::new(ctx));

    let public_dir = args.public.map(PathBuf::from);
    let config = Arc::new(ServerConfig::new(&args.request_path, public_dir, args.port));

    let prefix = &config.request_path;
    let is_unix_sock = args.host.ends_with(".sock");

    #[cfg(unix)]
    if is_unix_sock {
        let sock_path = &args.host;
        if std::path::Path::new(sock_path).exists() {
            let _ = std::fs::remove_file(sock_path);
        }
        let listener = std::os::unix::net::UnixListener::bind(sock_path)?;
        println!("Whisper HTTP server listening on unix://{}", sock_path);
        println!("Endpoints:");
        println!("  GET  {}/health", prefix);
        println!("  POST {}/load", prefix);
        println!("  POST {}/v1/audio/transcriptions", prefix);
        println!("  POST {}/v1/audio/translations", prefix);
        println!("  POST {}/inference", prefix);
        println!("  GET  {}/", prefix);
        if let Some(pub_dir) = &config.public_dir {
            println!("  Static assets served from: {}", pub_dir.display());
        }

        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let ctx_clone = Arc::clone(&shared_ctx);
                    let cfg_clone = Arc::clone(&config);
                    std::thread::spawn(move || {
                        if let Ok(conn) = ClientConnection::from_unix_stream(s) {
                            handle_client(conn, ctx_clone, &cfg_clone);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Connection failed: {}", e);
                }
            }
        }
        return Ok(());
    }

    #[cfg(not(unix))]
    if is_unix_sock {
        return Err("Unix domain sockets (.sock) are only supported on Unix targets".into());
    }

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr)?;
    println!("Whisper HTTP server listening on http://{}", addr);
    println!("Endpoints:");
    println!("  GET  {}/health", prefix);
    println!("  POST {}/load", prefix);
    println!("  POST {}/v1/audio/transcriptions", prefix);
    println!("  POST {}/v1/audio/translations", prefix);
    println!("  POST {}/inference", prefix);
    println!("  GET  {}/", prefix);
    if let Some(pub_dir) = &config.public_dir {
        println!("  Static assets served from: {}", pub_dir.display());
    }

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let ctx_clone = Arc::clone(&shared_ctx);
                let cfg_clone = Arc::clone(&config);
                std::thread::spawn(move || {
                    if let Ok(conn) = ClientConnection::from_tcp_stream(s) {
                        handle_client(conn, ctx_clone, &cfg_clone);
                    }
                });
            }
            Err(e) => {
                eprintln!("Connection failed: {}", e);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use whisper_rs::dtw::WordTiming;

    fn workspace_path(rel: &str) -> PathBuf {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let ws_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(manifest_dir);
        ws_root.join(rel)
    }

    fn make_test_context() -> Option<SharedContext> {
        let candidates = [
            workspace_path("models/ggml-tiny.en.bin"),
            workspace_path("models/ggml-tiny.bin"),
            PathBuf::from("models/ggml-tiny.en.bin"),
            PathBuf::from("models/ggml-tiny.bin"),
        ];

        for candidate in &candidates {
            if candidate.exists() {
                if let Ok(model) = WhisperModel::load_from_file(candidate) {
                    if let Ok(ctx) = WhisperContext::new(model) {
                        return Some(Arc::new(RwLock::new(Arc::new(ctx))));
                    }
                }
            }
        }
        None
    }

    #[test]
    fn test_normalize_request_path() {
        assert_eq!(normalize_request_path(""), "");
        assert_eq!(normalize_request_path("/"), "");
        assert_eq!(normalize_request_path("api/v1"), "/api/v1");
        assert_eq!(normalize_request_path("/api/v1/"), "/api/v1");
        assert_eq!(normalize_request_path("/api/v1"), "/api/v1");
    }

    #[test]
    fn test_global_cors_and_options_preflight() {
        let Some(ctx) = make_test_context() else {
            eprintln!("No test model found, skipping test");
            return;
        };
        let config = ServerConfig::new("", None, 8080);

        let req = b"OPTIONS /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1:8080\r\n\r\n";
        let mut response = Vec::new();
        let conn = ClientConnection::new_mock(Cursor::new(req), &mut response, Arc::new(|| false));

        handle_client(conn, ctx, &config);

        let resp_str = String::from_utf8(response).unwrap();
        assert!(resp_str.starts_with("HTTP/1.1 204 No Content"));
        assert!(resp_str.contains("Access-Control-Allow-Origin: *"));
        assert!(resp_str.contains("Access-Control-Allow-Headers: content-type, authorization"));
        assert!(resp_str.contains("Access-Control-Allow-Methods: GET, POST, OPTIONS"));
    }

    #[test]
    fn test_health_endpoint() {
        let Some(ctx) = make_test_context() else {
            eprintln!("No test model found, skipping test");
            return;
        };
        let config = ServerConfig::new("/api/v1", None, 8080);

        // Test with prefix
        let req = b"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1:8080\r\n\r\n";
        let mut response = Vec::new();
        let conn = ClientConnection::new_mock(Cursor::new(req), &mut response, Arc::new(|| false));

        handle_client(conn, Arc::clone(&ctx), &config);

        let resp_str = String::from_utf8(response).unwrap();
        assert!(resp_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp_str.contains("Access-Control-Allow-Origin: *"));
        assert!(resp_str.contains("{\"status\":\"ok\"}"));

        // Test base /health
        let req_base = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:8080\r\n\r\n";
        let mut response_base = Vec::new();
        let conn_base = ClientConnection::new_mock(
            Cursor::new(req_base),
            &mut response_base,
            Arc::new(|| false),
        );

        handle_client(conn_base, ctx, &config);

        let resp_base_str = String::from_utf8(response_base).unwrap();
        assert!(resp_base_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp_base_str.contains("{\"status\":\"ok\"}"));
    }

    #[test]
    fn test_embedded_default_html_test_form() {
        let Some(ctx) = make_test_context() else {
            eprintln!("No test model found, skipping test");
            return;
        };
        let config = ServerConfig::new("/my-api", None, 9090);

        let req = b"GET /my-api/ HTTP/1.1\r\nHost: 127.0.0.1:9090\r\n\r\n";
        let mut response = Vec::new();
        let conn = ClientConnection::new_mock(Cursor::new(req), &mut response, Arc::new(|| false));

        handle_client(conn, ctx, &config);

        let resp_str = String::from_utf8(response).unwrap();
        assert!(resp_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp_str.contains("Content-Type: text/html; charset=utf-8"));
        assert!(resp_str.contains("Access-Control-Allow-Origin: *"));
        assert!(resp_str.contains("<title>Whisper.cpp Server</title>"));
        assert!(resp_str.contains("action=\"/my-api/v1/audio/transcriptions\""));
        assert!(resp_str.contains("<option value=\"verbose_json\""));
    }

    #[test]
    fn test_static_file_serving_and_traversal_prevention() {
        let temp_dir =
            std::env::temp_dir().join(format!("whisper_test_public_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let index_file = temp_dir.join("index.html");
        std::fs::write(&index_file, "<html><body>Hello Static</body></html>").unwrap();
        let css_file = temp_dir.join("style.css");
        std::fs::write(&css_file, "body { color: red; }").unwrap();

        let Some(ctx) = make_test_context() else {
            eprintln!("No test model found, skipping test");
            return;
        };
        let config = ServerConfig::new("", Some(temp_dir.clone()), 8080);

        // 1. GET / should serve index.html
        let req1 = b"GET / HTTP/1.1\r\nHost: 127.0.0.1:8080\r\n\r\n";
        let mut resp1 = Vec::new();
        let conn1 = ClientConnection::new_mock(Cursor::new(req1), &mut resp1, Arc::new(|| false));
        handle_client(conn1, Arc::clone(&ctx), &config);
        let resp1_str = String::from_utf8(resp1).unwrap();
        assert!(resp1_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp1_str.contains("Content-Type: text/html; charset=utf-8"));
        assert!(resp1_str.contains("Hello Static"));

        // 2. GET /style.css should serve style.css
        let req2 = b"GET /style.css HTTP/1.1\r\nHost: 127.0.0.1:8080\r\n\r\n";
        let mut resp2 = Vec::new();
        let conn2 = ClientConnection::new_mock(Cursor::new(req2), &mut resp2, Arc::new(|| false));
        handle_client(conn2, Arc::clone(&ctx), &config);
        let resp2_str = String::from_utf8(resp2).unwrap();
        assert!(resp2_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp2_str.contains("Content-Type: text/css; charset=utf-8"));
        assert!(resp2_str.contains("body { color: red; }"));

        // 3. Path traversal attempt must return 404
        let req3 = b"GET /../../etc/passwd HTTP/1.1\r\nHost: 127.0.0.1:8080\r\n\r\n";
        let mut resp3 = Vec::new();
        let conn3 = ClientConnection::new_mock(Cursor::new(req3), &mut resp3, Arc::new(|| false));
        handle_client(conn3, ctx, &config);
        let resp3_str = String::from_utf8(resp3).unwrap();
        assert!(resp3_str.starts_with("HTTP/1.1 404 Not Found"));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_format_verbose_json_schema() {
        let words = vec![
            WordTiming::new("Hello", 0.0, 0.45, 0.98),
            WordTiming::new("world", 0.45, 0.95, 0.95),
        ];
        let seg1 = Segment::with_stats(
            0.0,
            1.0,
            " Hello world".to_string(),
            vec![50364, 15496, 995, 50365],
            words,
            -0.15,
            0.005,
            0.32,
        );
        let seg2 = Segment::new(
            1.0,
            2.0,
            " Testing 1 2 3".to_string(),
            vec![50365, 2345, 50366],
        );

        let segments = vec![seg1, seg2];
        let json_str = format_verbose_json(&segments, 2.0, "en", false);

        assert!(json_str.contains("\"task\":\"transcribe\""));
        assert!(json_str.contains("\"language\":\"en\""));
        assert!(json_str.contains("\"duration\":2.00"));
        assert!(json_str.contains("\"text\":\"Hello world Testing 1 2 3\""));
        assert!(json_str.contains("\"segments\":["));

        // Segment 0 fields
        assert!(json_str.contains("\"id\":0"));
        assert!(json_str.contains("\"start\":0.000"));
        assert!(json_str.contains("\"end\":1.000"));
        assert!(json_str.contains("\"text\":\"Hello world\""));
        assert!(json_str.contains("\"tokens\":[50364,15496,995,50365]"));
        assert!(json_str.contains(
            "\"words\":[{\"word\":\"Hello\",\"start\":0.000,\"end\":0.450,\"probability\":0.9800}"
        ));
        assert!(json_str.contains("\"avg_logprob\":-0.1500"));
        assert!(json_str.contains("\"no_speech_prob\":0.0050"));

        // Segment 1 fields
        assert!(json_str.contains("\"id\":1"));
        assert!(json_str.contains("\"tokens\":[50365,2345,50366]"));
        assert!(json_str.contains("\"words\":[]"));

        // Translation task
        let json_tr = format_verbose_json(&segments, 2.0, "es", true);
        assert!(json_tr.contains("\"task\":\"translate\""));
        assert!(json_tr.contains("\"language\":\"es\""));
    }

    #[test]
    fn test_model_hot_reloading_validation_and_load() {
        let Some(ctx) = make_test_context() else {
            eprintln!("No test model found, skipping test");
            return;
        };
        let config = ServerConfig::new("/api/v1", None, 8080);

        // 1. Missing model field -> 400 Bad Request
        let req_no_field = b"POST /api/v1/load HTTP/1.1\r\n\
Host: 127.0.0.1:8080\r\n\
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary\r\n\
Content-Length: 68\r\n\r\n\
------WebKitFormBoundary\r\n\
Content-Disposition: form-data; name=\"other\"\r\n\r\n\
foo\r\n\
------WebKitFormBoundary--\r\n";

        let mut resp1 = Vec::new();
        let conn1 =
            ClientConnection::new_mock(Cursor::new(req_no_field), &mut resp1, Arc::new(|| false));
        handle_client(conn1, Arc::clone(&ctx), &config);
        let resp1_str = String::from_utf8(resp1).unwrap();
        assert!(resp1_str.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(resp1_str.contains("no 'model' field in the request"));

        // 2. Non-existent model path -> 400 Bad Request
        let req_missing_file = b"POST /api/v1/load HTTP/1.1\r\n\
Host: 127.0.0.1:8080\r\n\
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary\r\n\
Content-Length: 104\r\n\r\n\
------WebKitFormBoundary\r\n\
Content-Disposition: form-data; name=\"model\"\r\n\r\n\
/nonexistent/model.bin\r\n\
------WebKitFormBoundary--\r\n";

        let mut resp2 = Vec::new();
        let conn2 = ClientConnection::new_mock(
            Cursor::new(req_missing_file),
            &mut resp2,
            Arc::new(|| false),
        );
        handle_client(conn2, Arc::clone(&ctx), &config);
        let resp2_str = String::from_utf8(resp2).unwrap();
        assert!(resp2_str.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(resp2_str.contains("model not found!"));

        // 3. Valid model file -> HTTP 200 "Load was successful!\n"
        let model_path = [
            workspace_path("models/ggml-tiny.en.bin"),
            workspace_path("models/ggml-tiny.bin"),
            PathBuf::from("models/ggml-tiny.en.bin"),
            PathBuf::from("models/ggml-tiny.bin"),
        ]
        .into_iter()
        .find(|p| p.exists());
        if let Some(model_path) = model_path {
            let body_content = format!(
                "------WebKitFormBoundary\r\n\
Content-Disposition: form-data; name=\"model\"\r\n\r\n\
{}\r\n\
------WebKitFormBoundary--\r\n",
                model_path.to_str().unwrap()
            );
            let req_valid = format!(
                "POST /api/v1/load HTTP/1.1\r\n\
Host: 127.0.0.1:8080\r\n\
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary\r\n\
Content-Length: {}\r\n\r\n\
{}",
                body_content.len(),
                body_content
            );

            let mut resp3 = Vec::new();
            let conn3 = ClientConnection::new_mock(
                Cursor::new(req_valid.as_bytes()),
                &mut resp3,
                Arc::new(|| false),
            );
            handle_client(conn3, ctx, &config);
            let resp3_str = String::from_utf8(resp3).unwrap();
            assert!(resp3_str.starts_with("HTTP/1.1 200 OK"));
            assert!(resp3_str.contains("Load was successful!\n"));
            assert!(resp3_str.contains("Access-Control-Allow-Origin: *"));
        }
    }

    #[test]
    fn test_client_disconnect_abort_emits_499() {
        let Some(ctx) = make_test_context() else {
            eprintln!("No test model found, skipping test");
            return;
        };
        let config = ServerConfig::new("", None, 8080);

        // Construct minimal valid WAV (44-byte header + 16000 samples of silence = 1 second)
        let sample_rate = 16000u32;
        let num_samples = 16000usize;
        let data_size = (num_samples * 2) as u32;
        let file_size = 36 + data_size;

        let mut wav_bytes = Vec::new();
        wav_bytes.extend_from_slice(b"RIFF");
        wav_bytes.extend_from_slice(&file_size.to_le_bytes());
        wav_bytes.extend_from_slice(b"WAVEfmt ");
        wav_bytes.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
        wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav_bytes.extend_from_slice(&sample_rate.to_le_bytes());
        wav_bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav_bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav_bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav_bytes.extend_from_slice(b"data");
        wav_bytes.extend_from_slice(&data_size.to_le_bytes());
        wav_bytes.resize(44 + data_size as usize, 0u8);

        let req_header = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\n\
Host: 127.0.0.1:8080\r\n\
Content-Length: {}\r\n\r\n",
            wav_bytes.len()
        );

        let mut full_req = req_header.into_bytes();
        full_req.extend_from_slice(&wav_bytes);

        // Disconnect checker returning true simulates client early disconnect during inference
        let disconnect_checker = Arc::new(|| true);
        let mut response = Vec::new();
        let conn =
            ClientConnection::new_mock(Cursor::new(full_req), &mut response, disconnect_checker);

        handle_client(conn, ctx, &config);

        let resp_str = String::from_utf8(response).unwrap();
        assert!(
            resp_str.starts_with("HTTP/1.1 499 Client Closed Request"),
            "Expected HTTP 499 on client abort, got: {}",
            resp_str
        );
        assert!(resp_str.contains("client disconnected"));
        assert!(resp_str.contains("Access-Control-Allow-Origin: *"));
    }
}
