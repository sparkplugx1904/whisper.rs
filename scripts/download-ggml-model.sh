#!/usr/bin/env bash
# Script to download official Whisper GGML models
set -euo pipefail

MODEL="${1:-tiny.en}"
MODELS_DIR="${2:-models}"

SRC="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

mkdir -p "$MODELS_DIR"

echo "Downloading ggml-$MODEL.bin from $SRC..."
FILE="ggml-$MODEL.bin"
TARGET="$MODELS_DIR/$FILE"

if [ -f "$TARGET" ]; then
    echo "Model '$TARGET' already exists. Skipping."
    exit 0
fi

curl -L --fail --retry 3 --retry-delay 3 -o "$TARGET" "$SRC/$FILE"
echo "Done! Model saved to '$TARGET'."
