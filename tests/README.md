# ASR Engine Performance Benchmark Suite

This directory contains the comparative performance benchmarks and full generated transcription outputs evaluating four distinct Whisper automatic speech recognition (ASR) implementations.

---

## Benchmark Objectives & Protocol

All benchmarks were conducted **sequentially** on dedicated CPU resources to ensure exact comparability and eliminate scheduler/CPU cache contention.

- **Execution Protocol**: Strictly sequential (isolated single-process execution with 2-second thermal cool-down between runs).
- **Decoding Configuration**: Greedy search ($T = 0.0$, $\text{beam\_size} = 1$, timestamps disabled).
- **Model**: OpenAI Whisper `tiny.en` (75 MB weights).
- **Thread Count**: 2 threads (matching the 2 vCPUs allocated to the host container).

### Host Hardware & Environment

- **CPU**: AMD EPYC 7763 64-Core Processor (2 vCPUs allocated)
- **Host Architecture**: x86_64
- **Active Vector Extensions**: AVX2, FMA, F16C, BMI1, BMI2, SSE4.2
- **Operating System**: Linux 6.8.0-1064-azure (Ubuntu 24.04 LTS container)
- **Rust Toolchain**: `rustc 1.90.0-nightly`

---

## Contestants Evaluated

1. **`whisper.rs` (Current Pure-Rust)**: This repository (`crates/whisper` + `crates/cli`). Zero C/C++ dependencies, runtime AVX2/FMA vectorization, cache-tiled Rayon work chunking.
2. **`whisper.cpp` (ggml-org)**: Upstream C/C++ reference implementation (`v1.9.5-dev`) compiled with OpenMP and `-march=native`.
3. **`whisper-rs` (tazz)**: Published Rust FFI bindings wrapper on Codeberg/crates.io (`v0.16.0`) dynamically linking against `whisper-rs-sys`.
4. **`faster-whisper` (SYSTRAN)**: CTranslate2-based Python runtime (`v1.2.1`) widely adopted for fast CPU/GPU inference.

---

## Audio Test Dataset (`samples/`)

Audio samples were sourced from official `whisper.cpp` test fixtures and OpenSLR Wikimedia datasets:

| Sample | Duration | Format | Description / Acoustic Profile |
| :--- | :---: | :---: | :--- |
| **`jfk.wav`** | 11.00 s | 16 kHz Mono WAV | Studio broadcast speech: John F. Kennedy inaugural address excerpt. |
| **`mm1.wav`** | 30.00 s | 16 kHz Mono WAV | Fast-tempo commercial speech: John Moschitta Jr. (*Micro Machines*). |
| **`diffusion_60s.wav`** | 60.00 s | 16 kHz Mono WAV | Australian science radio broadcast with conversational dialogue and background music. |
| **`gb0.wav`** | 127.35 s | 16 kHz Mono WAV | Extended political address: George W. Bush weekly radio address (~2.1 minutes). |
| **`a13.wav`** | 178.67 s | 16 kHz Mono WAV | High-noise air-to-ground radio communication: Apollo 13 Houston mission control (~3.0 minutes). |
| **Total** | **407.02 s** | — | **~6 minutes 47 seconds of diverse speech audio** |

---

## Performance Results

### 1. Inference Latency (Seconds — Lower is Faster)

| Sample | Audio Length | `whisper-rs` (tazz) | `whisper.cpp` (ggml-org) | `faster-whisper` (SYSTRAN) | `whisper.rs` (Current Pure-Rust) | Fastest |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`jfk.wav`** | 11.00 s | 1.67 s | 1.44 s | **0.95 s** | 1.14 s | `faster-whisper` |
| **`mm1.wav`** | 30.00 s | 10.10 s | **2.49 s** | 20.10 s | 2.80 s | `whisper.cpp` |
| **`diffusion_60s.wav`** | 60.00 s | 6.51 s | **2.81 s** | 3.23 s | **2.81 s** | Tie (`whisper.cpp` / `whisper.rs`) |
| **`gb0.wav`** | 127.35 s | 14.33 s | 10.15 s | **8.73 s** | 9.70 s | `faster-whisper` |
| **`a13.wav`** | 178.67 s | 22.01 s | 16.46 s | 37.44 s | **14.31 s** | `whisper.rs` (Current) |

---

### 2. Real-Time Factor (RTF = $\frac{\text{Processing Time}}{\text{Audio Length}}$ — Lower is Better)

| Sample | `whisper-rs` (tazz) | `whisper.cpp` (ggml-org) | `faster-whisper` (SYSTRAN) | `whisper.rs` (Current Pure-Rust) |
| :--- | :---: | :---: | :---: | :---: |
| **`jfk.wav`** (11.0 s) | 0.152x | 0.131x | **0.087x** | 0.104x |
| **`mm1.wav`** (30.0 s) | 0.336x | **0.083x** | 0.670x | 0.093x |
| **`diffusion_60s.wav`** (60.0 s) | 0.108x | **0.047x** | 0.054x | **0.047x** |
| **`gb0.wav`** (127.4 s) | 0.113x | 0.080x | **0.069x** | 0.076x |
| **`a13.wav`** (178.7 s) | 0.123x | 0.092x | 0.210x | **0.080x** |

---

### 3. Throughput Speedup Multiplier (Times Real-Time — Higher is Faster)

| Sample | `whisper-rs` (tazz) | `whisper.cpp` (ggml-org) | `faster-whisper` (SYSTRAN) | `whisper.rs` (Current Pure-Rust) |
| :--- | :---: | :---: | :---: | :---: |
| **`jfk.wav`** | 6.58x RT | 7.63x RT | **11.52x RT** | 9.65x RT |
| **`mm1.wav`** | 2.97x RT | **12.07x RT** | 1.49x RT | 10.70x RT |
| **`diffusion_60s.wav`** | 9.22x RT | **21.35x RT** | 18.57x RT | **21.34x RT** |
| **`gb0.wav`** | 8.89x RT | 12.54x RT | **14.58x RT** | 13.13x RT |
| **`a13.wav`** | 8.12x RT | 10.85x RT | 4.77x RT | **12.48x RT** |

---

### 4. Cumulative Summary Across 407 Seconds of Audio

| Metric | `whisper-rs` (tazz) | `whisper.cpp` (ggml-org) | `faster-whisper` (SYSTRAN) | `whisper.rs` (Current Pure-Rust) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Processing Time** | 54.62 s | 33.35 s | 70.45 s | **30.76 s** |
| **Aggregate Real-Time Factor (RTF)** | 0.134x | 0.082x | 0.173x | **0.075x** |
| **Average Throughput Speedup** | 7.45x RT | 12.20x RT | 5.78x RT | **13.23x RT** |
| **Foreign Dependencies** | C++ bindings (`whisper-rs-sys`, cmake) | C/C++ OpenMP runtime | Python, CTranslate2, PyAV, ONNX | **Zero (100% Pure Rust)** |
| **Acoustic Robustness** | Degraded on fast speech | High | Unstable on rapid/noisy audio | **Highest** |

---

## Detailed Transcript Analysis & Failure Modes

Full transcripts generated during this benchmark are cataloged in `tests/transcripts/`:

```
tests/transcripts/
├── whisper_rs_current/     # Pure Rust whisper.rs outputs
├── whisper_cpp/            # Official whisper.cpp outputs
├── faster_whisper/         # SYSTRAN faster-whisper outputs
└── whisper_rs_tazz/        # tazz/whisper-rs outputs
```

### Key Behavioral Observations

1. **Robustness on High-Speed Speech (`mm1.wav`)**:
   - `whisper.rs` and `whisper.cpp` correctly transcribed the rapid speech without getting stuck in repetition loops.
   - `whisper-rs (tazz)` suffered repetitive hallucinations (`(gentle music) (gentle music) (gentle music)...`), increasing latency to 10.10s.
   - `faster-whisper` failed catastrophically on `mm1.wav`, entering an infinite punctuation repetition loop (`.  .  .  . .  . .  . .  .`), taking 20.10s before timing out on maximum window tokens.

2. **Long Audio & Radio Noise (`a13.wav`)**:
   - On the 178-second distorted Apollo 13 transmission, `whisper.rs` recorded the fastest completion time of all contestants at **14.31s** (12.48x real-time), finishing ahead of `whisper.cpp` (16.46s).
   - `faster-whisper` fell into repetition loops on the noisy transmission, blowing processing time out to **37.44s**.

3. **Memory Footprint**:
   - `whisper.rs` operates within a lean ~78 MB heap footprint with zero background thread leaks.
   - `whisper.cpp` uses ~77 MB total resident memory.
   - `faster-whisper` requires ~185 MB due to Python runtime, NumPy buffers, and CTranslate2 dynamic libraries.
