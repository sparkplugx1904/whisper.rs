#!/usr/bin/env bash
# Transcribe the merged clip audio with this repository's CLI.
set -euo pipefail

WAV_PATH="${1:?merged wav}"
WORK="${2:?work directory}"
MODEL_URL="${3:-https://huggingface.co/Sparkplugx1904/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$WORK/models" "$WORK/out"
MODEL_PATH="$WORK/models/ggml-large-v3-turbo.bin"
if [[ ! -s "$MODEL_PATH" ]]; then
    echo "downloading large-v3-turbo model"
    curl -L --fail --silent --show-error --retry 5 --retry-delay 3 \
        "$MODEL_URL" -o "$MODEL_PATH"
fi

cargo build --release -p whisper-cli
git rev-parse --short=12 HEAD > "$WORK/out/whisper_rs.version.txt"

OUT_BASE="$WORK/out/merged"
/usr/bin/time -v -o "$WORK/out/transcribe.time.txt" \
    "$REPO_ROOT/target/release/whisper-cli" \
    -m "$MODEL_PATH" \
    -f "$WAV_PATH" \
    -l id \
    -t 4 \
    -b 5 \
    --output-json \
    --output-txt \
    -o "$OUT_BASE" \
    --no-prints

test -s "$OUT_BASE.json"
test -s "$OUT_BASE.txt"