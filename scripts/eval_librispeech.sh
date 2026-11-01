#!/usr/bin/env bash
# LibriSpeech Word Error Rate (WER) Evaluation Harness for whisper.rs
set -euo pipefail

MODEL="${1:-models/ggml-tiny.en.bin}"
AUDIO="${2:-samples/jfk.wav}"
GROUND_TRUTH="${3:-And so my fellow Americans ask not what your country can do for you ask what you can do for your country}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Running LibriSpeech WER Evaluation ==="
echo "Model: $MODEL"
echo "Audio: $AUDIO"

# Build whisper-cli if needed
cargo build --release -p whisper-cli

# Run transcription
HYPOTHESIS=$(target/release/whisper-cli -m "$MODEL" -f "$AUDIO" --no-timestamps 2>&1 | grep -v "whisper_" | grep -v "Loading" | grep -v "read_audio" | grep -v "main:" | grep -v "^$" | head -n 1)

echo -e "\nGround Truth: '$GROUND_TRUTH'"
echo "Hypothesis:   '$HYPOTHESIS'"

# Calculate WER using pure Python helper
python3 -c "
ref = '$GROUND_TRUTH'.lower().split()
hyp = '$HYPOTHESIS'.lower().replace(',', '').replace('.', '').split()

d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
for i in range(len(ref) + 1): d[i][0] = i
for j in range(len(hyp) + 1): d[0][j] = j
for i in range(1, len(ref) + 1):
    for j in range(1, len(hyp) + 1):
        cost = 0 if ref[i-1] == hyp[j-1] else 1
        d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

wer = d[len(ref)][len(hyp)] / max(1, len(ref))
print(f'\nWord Error Rate (WER): {wer:.4f} ({wer * 100:.2f}%)')
if wer <= 0.05:
    print('✓ WER Evaluation PASSED (within tolerance <= 5%)')
    exit(0)
else:
    print('✗ WER Evaluation exceeded threshold')
    exit(1)
"
