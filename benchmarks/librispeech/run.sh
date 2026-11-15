#!/usr/bin/env bash
# Run the LibriSpeech acceptance gate.
#
# Builds nothing. The binary and model must already exist. The transcription is
# serialized through replica/coordination/bench.sh, which takes the exclusive
# /tmp/vot-solo/bench.lock, so this script is safe to queue behind another run.
#
#   WHISPER_MODEL=/tmp/libri/models/ggml-large-v3-turbo.bin \
#       benchmarks/librispeech/run.sh full
#   WHISPER_MODEL=... benchmarks/librispeech/run.sh fast
#
# Environment:
#   WHISPER_MODEL   ggml model path (required)
#   WHISPER_BIN     whisper-cli path (default target/release/whisper-cli)
#   LIBRI_ROOT      staged corpus directory (default /tmp/libri)
#   BOUNDARIES      boundaries.json (default $LIBRI_ROOT/boundaries.json)
#   BASELINE        baseline JSON (default benchmarks/librispeech/baseline.json)
#   LABEL           run label (default librispeech-<mode>)
#   OUT             run directory (default $LIBRI_ROOT/runs/$LABEL)
#   SAVE_BASELINE   set to 1 to write the baseline instead of gating against it
#   BASELINE_NOTE   required with SAVE_BASELINE=1; recorded inside the baseline
#   OVERWRITE_BASELINE  set to 1 to replace an existing baseline file
set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
    fast|full) ;;
    *) echo "usage: run.sh [fast|full]" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WHISPER_BIN="${WHISPER_BIN:-target/release/whisper-cli}"
: "${WHISPER_MODEL:?set WHISPER_MODEL to the ggml model path}"
LIBRI_ROOT="${LIBRI_ROOT:-/tmp/libri}"
BOUNDARIES="${BOUNDARIES:-$LIBRI_ROOT/boundaries.json}"
BASELINE="${BASELINE:-benchmarks/librispeech/baseline.json}"
LABEL="${LABEL:-librispeech-$MODE}"
OUT="${OUT:-$LIBRI_ROOT/runs/$LABEL}"
SAVE_BASELINE="${SAVE_BASELINE:-0}"
FAST_UTTERANCES=10

for path in "$WHISPER_BIN" "$WHISPER_MODEL" "$LIBRI_ROOT/chapter.wav" "$BOUNDARIES"; do
    [ -e "$path" ] || { echo "missing: $path" >&2; exit 2; }
done

WAV="$LIBRI_ROOT/chapter.wav"
if [ "$MODE" = "fast" ]; then
    WAV="$LIBRI_ROOT/fast.wav"
    python3 - "$LIBRI_ROOT/chapter.wav" "$BOUNDARIES" "$WAV" "$FAST_UTTERANCES" <<'PY'
import json
import sys
import wave

source, boundaries_path, target, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(boundaries_path, encoding="utf-8") as handle:
    boundaries = json.load(handle)
clips = boundaries["clips"][:limit]
if len(clips) < limit:
    raise SystemExit(f"only {len(clips)} utterances staged, need {limit}")
end = int(clips[-1]["end_sample"]) + round(float(boundaries["gap_s"]) * int(boundaries["sample_rate"]))
with wave.open(source, "rb") as reader:
    frames = reader.readframes(end)
with wave.open(target, "wb") as writer:
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(int(boundaries["sample_rate"]))
    writer.writeframes(frames)
print(f"fast audio: {end / int(boundaries['sample_rate']):.2f} s ({len(clips)} utterances) -> {target}")
PY
fi

LANG_CODE=en MODEL="$WHISPER_MODEL" \
    replica/coordination/bench.sh "$LABEL" "$WHISPER_BIN" "$WAV" "$OUT" 5

GATE_ARGS=("$OUT" --mode "$MODE" --boundaries "$BOUNDARIES" --json "$OUT/gate.json")
if [ "$SAVE_BASELINE" = "1" ]; then
    : "${BASELINE_NOTE:?set BASELINE_NOTE (source commit, review, binary sha) when saving a baseline}"
    if [ -f "$BASELINE" ] && [ "${OVERWRITE_BASELINE:-0}" != "1" ]; then
        echo "$BASELINE already exists; set OVERWRITE_BASELINE=1 to replace it" >&2
        exit 3
    fi
    GATE_ARGS+=(--no-baseline --max-wer 1.0 --save-baseline "$BASELINE" --baseline-note "$BASELINE_NOTE")
elif [ -f "$BASELINE" ] && python3 -c \
    "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2] in d.get('modes',{}) else 1)" \
    "$BASELINE" "$MODE"; then
    GATE_ARGS+=(--baseline "$BASELINE")
else
    echo "no baseline for mode $MODE at $BASELINE;" >&2
    echo "bootstrap it with SAVE_BASELINE=1 from a reviewed run" >&2
    exit 3
fi

python3 benchmarks/librispeech/gate.py "${GATE_ARGS[@]}"