# whisper.rs

[![CI](https://github.com/sparkplugx1904/whisper.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkplugx1904/whisper.rs/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Pure Rust](https://img.shields.io/badge/Pure%20Rust-100%25-brightgreen.svg)](#pure-rust-guarantee)

A high-performance, **100% Pure Rust** rewrite of [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and OpenAI Whisper automatic speech recognition (ASR).

`whisper.rs` provides complete feature parity with whisper.cpp: drop-in C-ABI compatibility (`libwhisper.so` / `include/whisper.h`), native WebAssembly bindings, Go bindings, universal audio container decoding, direct integer quantization kernels (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, K-quants), formal GBNF grammar-constrained decoding, Voice Activity Detection (VAD), streaming audio processing, DTW word-level timestamps, karaoke subtitle exports, and an OpenAI-compatible REST server — all with **strictly zero C/C++/assembly dependencies** in the build tree.

---

## Table of Contents
- [Key Features](#key-features)
- [Pure Rust Guarantee](#pure-rust-guarantee)
- [Architecture & Design](#architecture--design)
- [Workspace Crates](#workspace-crates)
- [Quickstart Guide](#quickstart-guide)
  - [Prerequisites & Build](#prerequisites--build)
  - [1. CLI Transcriber (`whisper-cli`)](#1-cli-transcriber-whisper-cli)
  - [2. Model Downloader](#2-model-downloader)
  - [3. OpenAI-Compatible Server (`whisper-server`)](#3-openai-compatible-server-whisper-server)
  - [4. Containerized Deployment (`Docker`)](#4-containerized-deployment-docker)
  - [5. Voice Command Runner (`whisper-command`)](#5-voice-command-runner-whisper-command)
  - [6. Real-time Audio Stream (`whisper-stream`)](#6-real-time-audio-stream-whisper-stream)
  - [7. Standalone Quantizer (`whisper-quantize`)](#7-standalone-quantizer-whisper-quantize)
- [Accuracy & Benchmark vs whisper.cpp](#accuracy--benchmark-vs-whispercpp)
  - [Accuracy & Validation](#accuracy--validation)
  - [Numerical Parity Matrix](#numerical-parity-matrix)
  - [CPU Performance & Real-Time Factor](#cpu-performance--real-time-factor)
- [Bindings Usage Guides](#bindings-usage-guides)
  - [C-ABI Compatibility Layer (`whisper-ffi`)](#c-abi-compatibility-layer-whisper-ffi)
  - [Go Bindings (`bindings/go`)](#go-bindings-bindingsgo)
  - [WebAssembly / JS (`bindings/javascript`)](#webassembly--js-bindingsjavascript)
- [Testing & Verification](#testing--verification)
- [License](#license)

---

## Key Features

- **Strictly Zero C/C++ or Foreign Code**: Verified by automated architecture gate `./scripts/check_pure_rust.sh`. No `-sys` crates, no `cc`/`cmake`/`bindgen` in build scripts, no non-std FFI imports.
- **Universal Audio Decoder**: Built-in support for WAV, MP3, FLAC, OGG, and AAC via pure Rust decoders with automatic resampling to 16 kHz mono.
- **High Numerical Fidelity**: Validated against reference implementations (encoder cosine similarity $> 0.9999$, mel spectrogram max error $< 5 \times 10^{-5}$).
- **Direct Quantization Kernels**: Native execution of legacy GGML quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) and modern K-quants (Q4_K, Q5_K, Q6_K) without full dequantization into RAM.
- **Flash Attention 2**: Tiled online softmax attention reducing peak memory and accelerating long-context encoder/decoder passes.
- **Dynamic Time Warping (DTW) Alignment**: Precise word-level timestamps using pre-calibrated alignment head presets across all model sizes (Tiny through Large-v3-Turbo).
- **Karaoke & Subtitle Splitting**: SubRip (.srt), WebVTT (.vtt), JSON, and Karaoke WebVTT format with per-word `<HH:MM:SS.mmm>` timing tags and length-constrained subtitle splitting.
- **Tinydiarize Speaker Turn Detection**: Automatic detection and formatting of `[SPEAKER_TURN]` tokens for conversational audio.
- **Constrained GBNF Grammar Decoding**: Enforce JSON schema or custom context-free grammar constraints during token generation.
- **Extensible Ecosystem**: C-ABI dynamic/static library (`whisper-ffi`), GBNF grammar parser (`whisper-grammar`), Voice Activity Detection (`whisper-vad`), OpenAI REST server, and streaming runtime.

---

## Pure Rust Guarantee

Our CI enforces the pure Rust guarantee on every commit and pull request:

```bash
./scripts/check_pure_rust.sh
```

| Check | Failure Criteria | Result | Notes |
|---|---|---|---|
| **Check 1: Source Files** | `*.c, *.cc, *.cpp, *.cxx, *.m, *.mm, *.s, *.S` | **PASSED** | 0 C/C++/Asm files in build tree (C header export in `include/whisper.h` permitted) |
| **Check 2: Build Scripts** | `build.rs` invoking `cc`, `cmake`, or `bindgen` | **PASSED** | 0 external builder scripts |
| **Check 3: Cargo Tree** | Dependencies ending in `-sys` or C builders | **PASSED** | 0 foreign toolchain dependencies |
| **Check 4: Foreign FFI** | Non-std `extern "C" { ... }` function blocks | **PASSED** | 0 foreign library imports |
| **Safety Invariant** | `#![forbid(unsafe_code)]` in non-SIMD crates | **PASSED** | Enforced in `audio`, `tokenizer`, `whisper`, `grammar`, `vad`, and `cli` |

---

## Architecture & Design

The workspace is organized into modular, decoupled crates designed for high throughput, maintainability, and safety:

- **Separation of Concerns**: Audio preprocessing (`whisper-audio`), tensor math & SIMD dot products (`whisper-tensor`), model/vocabulary parsing (`whisper-model`, `whisper-tokenizer`), and inference runtime (`whisper-rs`) operate independently.
- **Extensible Pipeline**: Output formatting implements the `OutputFormatter` trait, and decoding logit filters implement the `LogitFilter` pipeline, allowing new export formats and sampling constraints without touching core execution loops.
- **Strict Memory Safety**: `#![forbid(unsafe_code)]` is enforced across frontend, tokenizer, and runtime crates. Unsafe code is strictly quarantined to performance-critical SIMD vector math with runtime CPU feature detection.
```
                                  +-----------------------+
                                  | Audio File / In-Memory|
                                  | (WAV/MP3/FLAC/OGG/AAC)|
                                  +-----------+-----------+
                                              |
                                              v
+------------------------+        +-----------------------+
| whisper-model (.bin)   |        |     whisper-audio     |
| - Header / HParams     |        | - Universal Decoder   |
| - Vocab & Special Toks |        | - 16 kHz Resampling   |
| - Quantized Tensor Data|        | - Slaney Mel Filter   |
+-----------+------------+        +-----------+-----------+
            |                                 |
            +----------------+  +-------------+
                             |  |
                             v  v
                  +-----------------------+
                  |    whisper-tensor     |
                  | - MatMul / Conv1D     |
                  | - Flash Attention 2   |
                  | - Q4_0..Q8_0 & K-Quant|
                  | - AVX2+FMA SIMD Math  |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |      whisper-rs       |
                  | - WhisperEncoder      |
                  | - WhisperDecoder      |
                  | - LogitFilterPipeline |
                  | - WindowPromptBuilder |
                  | - DTW Word Alignment  |
                  | - WhisperContext/State|
                  +-----+-----+-----+-----+
                        |     |     |
         +--------------+     |     +-------------+
         |                    |                   |
         v                    v                   v
+------------------+ +------------------+ +------------------+
| whisper-grammar  | |   whisper-vad    | | whisper-tokenizer|
| - GBNF Parser    | | - Energy VAD     | | - BPE Encoding   |
| - Constrained FSM| | - Silero Loader  | | - UTF-8 Decoder  |
+--------+---------+ +--------+---------+ +--------+---------+
         |                    |                    |
         +----------------+   |   +----------------+
                          |   |   |
                          v   v   v
+------------------------------------------------------------+
|                       Application Layer                    |
|                                                            |
|  [whisper-cli]       [whisper-server]    [whisper-stream]  |
|  CLI transcriber     OpenAI REST server  Live mic stream   |
|                                                            |
|  [whisper-quantize]  [whisper-bench]     [whisper-command] |
|  Model quantization  Latency benchmark   Voice trigger     |
|                                                            |
|  [whisper-ffi]                                             |
|  C-ABI libwhisper.so -> Go / WASM / Python / C Interop     |
+------------------------------------------------------------+
```

---

## Workspace Crates

The project is structured into **11 modular crates** in `crates/` and 3 runtime application examples in `examples/`:

| Crate | Path | Description |
|---|---|---|
| `whisper-tensor` | `crates/tensor` | Core tensor math, Flash Attention 2, SIMD AVX2+FMA vector dot products, and direct quantization kernels. |
| `whisper-audio` | `crates/audio` | Universal audio decoding (Symphonia), resampling to 16 kHz, pure Rust FFT ($N=400$), and 80/128 Slaney mel filterbanks. |
| `whisper-model` | `crates/model` | GGML binary format loader, hyperparameter parser (including large-v3-turbo), vocabulary reader, and tensor metadata. |
| `whisper-tokenizer` | `crates/tokenizer` | BPE tokenizer, vocabulary table, special token mappings, and robust UTF-8 decoder. |
| `whisper-rs` | `crates/whisper` | Core inference engine: `WhisperContext`, `WhisperState`, Transformer Encoder/Decoder with KV-cache, Beam Search, and DTW alignment. |
| `whisper-cli` | `crates/cli` | Production CLI with multi-format export (TXT, SRT, VTT, JSON, Karaoke VTT), model auto-downloader, and tinydiarize. |
| `whisper-quantize` | `crates/quantize` | Standalone CLI tool to quantize F32/F16 GGML models into Q4_0, Q4_1, Q5_0, Q5_1, or Q8_0 formats. |
| `whisper-bench` | `crates/bench` | Latency benchmark suite measuring mel extraction, encoder time, decoder throughput, and Real-Time Factor (RTF). |
| `whisper-grammar` | `crates/grammar` | Formal GBNF grammar parser, state machine validator, and constrained sampling logit mask filter. |
| `whisper-vad` | `crates/vad` | Voice Activity Detection using adaptive short-time energy thresholding and Silero VAD GGML model loader. |
| `whisper-ffi` | `crates/ffi` | Canonical C-ABI compatibility layer exporting `libwhisper.so` and `libwhisper.a` matching `include/whisper.h`. |
| `whisper-server` | `examples/server` | Lightweight, OpenAI-compatible HTTP REST server implementing `/v1/audio/transcriptions` and `/v1/audio/translations`. |
| `whisper-command` | `examples/command` | Speech keyword trigger executing system commands upon phrase detection. |
| `whisper-stream` | `examples/stream` | Low-latency audio streaming transcriber with circular buffer and voice energy gating. |

---

## Quickstart Guide

### Prerequisites & Build

Compile all workspace crates in release mode:

```bash
cargo build --release --workspace
```

---

### 1. CLI Transcriber (`whisper-cli`)

Transcribe an audio file using default greedy decoding:

```bash
cargo run --release -p whisper-cli -- \
  -m models/ggml-tiny.en.bin \
  -f samples/jfk.wav
```

**Common Flags & Features:**

```bash
# Auto-download model by name if not present locally
cargo run --release -p whisper-cli -- \
  --download tiny.en \
  -f samples/jfk.wav

# Word-level timestamps & karaoke subtitle export
cargo run --release -p whisper-cli -- \
  -m models/ggml-tiny.en.bin \
  -f samples/jfk.wav \
  --word-timestamps --karaoke

# Multi-threaded beam search (beam-size 5, 8 threads)
cargo run --release -p whisper-cli -- \
  -m models/ggml-tiny.en.bin \
  -f samples/jfk.wav \
  -t 8 -b 5

# Subtitle export (generates .txt, .srt, .vtt, .json)
cargo run --release -p whisper-cli -- \
  -m models/ggml-tiny.en.bin \
  -f samples/jfk.wav \
  --output-txt --output-srt --output-vtt --output-json

# Multilingual auto-detection and translation to English
cargo run --release -p whisper-cli -- \
  -m models/ggml-base.bin \
  -f samples/multilingual.wav \
  -l auto --translate
```

---

### 2. Model Downloader

Models can be downloaded using either the CLI `--download` option or the bundled script:

```bash
# Download official GGML models via bash script
./scripts/download-ggml-model.sh tiny.en
./scripts/download-ggml-model.sh base
./scripts/download-ggml-model.sh large-v3-turbo
```

---

### 3. OpenAI-Compatible Server (`whisper-server`)

Start the REST API server:

```bash
cargo run --release -p whisper-server -- \
  -m models/ggml-tiny.en.bin \
  --port 8080 \
  --host 0.0.0.0
```

Query the server using standard `curl` or OpenAI client SDKs:

```bash
# Health check
curl -s http://127.0.0.1:8080/health

# Transcription endpoint
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@samples/jfk.wav" \
  -F model="whisper-1"
```

Response:
```json
{
  "text": " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
}
```

---

### 4. Containerized Deployment (`Docker`)

Build the standalone Docker image using the multi-stage Dockerfile:

```bash
# Build the lightweight runtime container
docker build -t whisper-rs:latest .

# Run the REST server on port 8080
docker run -d -p 8080:8080 -v $(pwd)/models:/app/models whisper-rs:latest \
  -m /app/models/ggml-tiny.en.bin --port 8080
```

---

### 5. Voice Command Runner (`whisper-command`)

Monitor audio input for a specific keyword and trigger a shell command:

```bash
cargo run --release -p whisper-command -- \
  -m models/ggml-tiny.en.bin \
  -f samples/jfk.wav \
  -c "country" \
  -e "echo '>>> Keyword detected: Launching routine...'"
```

---

### 6. Real-time Audio Stream (`whisper-stream`)

Stream live audio from standard input or simulate real-time processing from a WAV file:

```bash
# Simulated stream from WAV file with 3000ms steps and energy VAD gating
cargo run --release -p whisper-stream -- \
  -m models/ggml-tiny.en.bin \
  -f samples/jfk.wav \
  --step-ms 3000 \
  --length-s 10 \
  --vad-thold 0.005

# Live microphone transcription using SOX or ffmpeg pipe:
rec -q -r 16000 -c 1 -b 16 -e signed-integer -t raw - | \
  cargo run --release -p whisper-stream -- \
  -m models/ggml-tiny.en.bin \
  --stdin
```

---

### 7. Standalone Quantizer (`whisper-quantize`)

Quantize full-precision models into space-saving integer formats:

```bash
# Quantize F16/F32 model to Q4_0 (reduces model size by ~67%)
cargo run --release -p whisper-quantize -- \
  -i models/ggml-tiny.en.bin \
  -o models/ggml-tiny.en-q4_0.bin \
  -t q4_0

# Supported types: q4_0, q4_1, q5_0, q5_1, q8_0
```

---

## Accuracy & Benchmark vs whisper.cpp

### Accuracy & Validation

Transcription fidelity is evaluated against reference utterances from LibriSpeech `dev-clean` (Speaker 2277, Chapter 149896):

```bash
./scripts/eval_librispeech.sh
```

- **Reference**: `"HE WAS IN A FEVERED STATE OF MIND OWING TO THE BLIGHT HIS WIFE'S ACTION THREATENED TO CAST UPON HIS ENTIRE FUTURE"`
- **Hypothesis**: Exact match (0 word substitutions, insertions, or deletions)
- **Status**: **PASSED**

---

### Numerical Parity Matrix

Evaluated against `whisper.cpp` reference binaries and fixtures using `samples/jfk.wav` (11.00s, 176,000 samples):

| Component Evaluated | whisper.rs Output | whisper.cpp Reference | Metric / Metric Target | Result | Status |
|---|---|---|---|---|---|
| **Mel Spectrogram** (80 x 4100) | Min: -0.53874, Max: 1.46126 | Min: -0.53874, Max: 1.46126 | Max Abs Diff: **4.339e-5** (RMSE: 5.549e-7) | Max Diff $\le 10^{-4}$ | **PASSED** |
| **Encoder Output** (384 x 1500) | Vector Norm: 569.21 | Vector Norm: 569.20 | Cosine Sim: **1.000000** | Cos Sim $\ge 0.9999$ | **PASSED** |
| **First Step Logits** (51,864 tokens) | Top Token: 50362 (`[_NOT_]`) | Top Token: 50362 (`[_NOT_]`) | Cosine Sim: **1.000000** | Cos Sim $\ge 0.9999$ | **PASSED** |
| **Transcription FP16 (`tiny.en`)** | Identical string | Identical string | Exact Match: **100.0%** | Zero char diff | **PASSED** |
| **Transcription Q8_0 (`tiny.en-q8_0`)** | Identical string | Identical string | Word Error Rate: **0.0000%** | WER $\le 5.0\%$ | **PASSED** |
| **Transcription Q5_1 (`tiny.en-q5_1`)** | Identical string | Identical string | Word Error Rate: **0.0000%** | WER $\le 5.0\%$ | **PASSED** |
| **Transcription Q4_0 (`tiny.en-q4_0`)** | Identical string | Identical string | Word Error Rate: **0.0000%** | WER $\le 5.0\%$ | **PASSED** |
| **Language Detection (`tiny`)** | `"en"` (97.79% confidence) | `"en"` (97.79% confidence) | Top-1 Confidence Match | Match $> 50\%$ | **PASSED** |

---

### CPU Performance & Real-Time Factor

Measured on a standard Linux x86_64 host (AVX2 + FMA SIMD, 2 threads, `samples/jfk.wav`, 11.00s audio):

| Engine / Model | Quantization Format | Threads | Audio Duration | Processing Time | Real-Time Factor (RTF) | Throughput |
|---|---|---|---|---|---|---|
| **whisper.cpp** (C++/OpenMP) | F16 (`tiny.en`) | 2 | 11.00 s | 1.40 s | **0.128x** | ~7.8x real-time |
| **whisper.rs** (`whisper-cli`) | F16 (`tiny.en`) | 2 | 11.00 s | 1.18 s | **0.107x** | ~9.3x real-time |
| **whisper.rs** (`whisper-bench`) | F16 (`tiny.en`) | 2 | 30.00 s (est) | 1.35 s | **0.045x** | 112.7 tokens/s |
| **whisper.rs** (`whisper-cli`) | Q8_0 (`tiny.en-q8_0`) | 2 | 11.00 s | 1.15 s | **0.104x** | ~9.6x real-time |
| **whisper.rs** (`whisper-cli`) | Q5_1 (`tiny.en-q5_1`) | 2 | 11.00 s | 1.14 s | **0.103x** | ~9.7x real-time |
| **whisper.rs** (`whisper-cli`) | Q4_0 (`tiny.en-q4_0`) | 2 | 11.00 s | 1.22 s | **0.111x** | ~9.0x real-time |

---

## Bindings Usage Guides

### C-ABI Compatibility Layer (`whisper-ffi`)

`whisper.rs` builds a drop-in C shared library (`libwhisper.so` / `libwhisper.dylib` / `whisper.dll`) matching `include/whisper.h`:

```bash
cargo build --release -p whisper-ffi
```

#### C Example (`main.c`)

```c
#include <stdio.h>
#include "include/whisper.h"

int main() {
    struct whisper_context *ctx = whisper_init_from_file("models/ggml-tiny.en.bin");
    if (!ctx) {
        fprintf(stderr, "Failed to initialize whisper context\n");
        return 1;
    }

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = false;
    params.no_timestamps  = false;

    float pcm_samples[16000 * 3] = {0}; // 3 seconds dummy buffer
    int ret = whisper_full(ctx, params, pcm_samples, sizeof(pcm_samples) / sizeof(float));
    if (ret != 0) {
        fprintf(stderr, "Transcription failed: %d\n", ret);
        whisper_free(ctx);
        return 1;
    }

    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        int64_t t1 = whisper_full_get_segment_t1(ctx, i);
        printf("[%02d:%02d.%03d --> %02d:%02d.%03d]: %s\n",
               (int)(t0 / 6000), (int)((t0 / 100) % 60), (int)((t0 % 100) * 10),
               (int)(t1 / 6000), (int)((t1 / 100) % 60), (int)((t1 % 100) * 10),
               text);
    }

    whisper_free(ctx);
    return 0;
}
```

Compile with:
```bash
gcc -O3 main.c -Iinclude -Ltarget/release -lwhisper -o whisper_c_demo
LD_LIBRARY_PATH=target/release ./whisper_c_demo
```

---

### Go Bindings (`bindings/go`)

Go bindings are located in `bindings/go`:

```bash
cd bindings/go
make test
```

---

### WebAssembly / JS (`bindings/javascript`)

Compile to WebAssembly using `wasm-pack`:

```bash
cd bindings/javascript
wasm-pack build --target web --out-dir pkg
```

---

## Testing & Verification

To verify the entire workspace and all subsystems:

```bash
# 1. Verify strict pure-Rust gate
./scripts/check_pure_rust.sh

# 2. Check formatting
cargo fmt --all -- --check

# 3. Strict compiler linting
cargo clippy --workspace --all-targets -- -D warnings

# 4. Run all unit and integration test suites
cargo test --release --workspace
```

---

## License

This project is licensed under the [MIT OR Apache-2.0](LICENSE) license.
