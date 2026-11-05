# Contributing to whisper.rs

Thank you for your interest in contributing to `whisper.rs`. We welcome contributions from bug fixes and documentation improvements to performance optimizations and new features.

---

## Getting Started

### Prerequisites

- **Rust**: Latest stable toolchain (edition 2021).
- **Cargo**: Standard package manager included with Rust.
- Optional: `wasm-pack` for WebAssembly bindings or `go` (1.20+) for Go bindings.

### Building the Workspace

Clone the repository and build all crates:

```bash
cargo build --workspace
```

To build in release mode with optimizations enabled:

```bash
cargo build --release --workspace
```

---

## Development Workflow

### 1. Code Formatting

All code must adhere to standard Rust formatting guidelines:

```bash
cargo fmt --all -- --check
```

To automatically format modified files:

```bash
cargo fmt --all
```

### 2. Linting & Static Analysis

We run strict compiler checks with Clippy. All warnings are treated as errors in CI:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### 3. Architecture & Purity Gate

`whisper.rs` is committed to being 100% pure Rust with zero C/C++ or foreign build dependencies:

```bash
./scripts/check_pure_rust.sh
```

Ensure that:
- No foreign C/C++/assembly files are added to the build tree.
- Build scripts (`build.rs`) do not invoke `cc`, `cmake`, or `bindgen`.
- No `-sys` crate dependencies are introduced.
- `#![forbid(unsafe_code)]` remains intact in crates outside `whisper-tensor`.

### 4. Running Tests

Run the full suite of unit and integration tests across all workspace crates:

```bash
cargo test --workspace
```

To run tests with release optimizations:

```bash
cargo test --release --workspace
```

---

## Coding Guidelines

- **Safety**: Keep unsafe code isolated strictly to performance-critical vectorized math in `whisper-tensor`. All unsafe blocks must be documented with a `// SAFETY:` invariant.
- **Dependencies**: Keep the dependency footprint minimal. Prefer standard library functionality where feasible.
- **Modularity**: Respect crate boundaries:
  - `whisper-audio`: Audio decoders, resampling, FFT, and mel spectrogram extraction.
  - `whisper-tensor`: SIMD matrix operations, quantizations, and attention kernels.
  - `whisper-model`: GGML model binary loader and metadata parser.
  - `whisper-tokenizer`: BPE tokenizer and vocabulary handling.
  - `whisper-rs`: Core Whisper context, decoder state, and inference pipelines.
  - `whisper-cli`: Command-line interface and file export formatters.

---

## Pull Request Guidelines

1. **Focused Scope**: Keep PRs focused on a single feature, bug fix, or optimization.
2. **Test Coverage**: Accompany bug fixes and new features with unit or integration tests.
3. **Verify Locally**: Before opening a pull request, verify that formatting, linter, tests, and architecture gates pass:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --release --workspace
   ./scripts/check_pure_rust.sh
   ```
4. **Clear Description**: Include a concise description of the motivation, changes, and verification steps in the pull request.
