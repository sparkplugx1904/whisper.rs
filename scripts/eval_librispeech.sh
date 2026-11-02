#!/usr/bin/env bash
# Authentic LibriSpeech Word Error Rate (WER) Evaluation Harness for whisper.rs
# Evaluates against genuine utterances from the OpenSLR LibriSpeech dev-clean corpus (Speaker 2277, Chapter 149896).
set -euo pipefail

MODEL="${1:-models/ggml-tiny.en.bin}"
AUDIO="${2:-samples/librispeech/2277-149896-0000.wav}"
if [ -z "${3:-}" ]; then
    GROUND_TRUTH="HE WAS IN A FEVERED STATE OF MIND OWING TO THE BLIGHT HIS WIFE'S ACTION THREATENED TO CAST UPON HIS ENTIRE FUTURE"
else
    GROUND_TRUTH="$3"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Running Authentic LibriSpeech WER Evaluation ==="
echo "Model: $MODEL"
echo "Audio: $AUDIO"

# Build whisper-cli if needed
cargo build --release -p whisper-cli

# Run transcription
HYPOTHESIS=$(target/release/whisper-cli -m "$MODEL" -f "$AUDIO" --no-timestamps 2>&1 | grep -v "whisper_" | grep -v "Loading" | grep -v "read_audio" | grep -v "main:" | grep -v "^$" | head -n 1)

echo -e "\nGround Truth: '$GROUND_TRUTH'"
echo "Hypothesis:   '$HYPOTHESIS'"

# Calculate WER using Python Levenshtein helper
python3 - "$GROUND_TRUTH" "$HYPOTHESIS" << 'EOF'
import sys
import string

ref_str = sys.argv[1]
hyp_str = sys.argv[2]

def clean_tokens(s):
    # Standard ASR text normalization: strip punctuation, lowercase
    for ch in string.punctuation:
        s = s.replace(ch, ' ')
    return [w.strip() for w in s.lower().split() if w.strip()]

ref = clean_tokens(ref_str)
hyp = clean_tokens(hyp_str)

d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
for i in range(len(ref) + 1): d[i][0] = i
for j in range(len(hyp) + 1): d[0][j] = j
for i in range(1, len(ref) + 1):
    for j in range(1, len(hyp) + 1):
        cost = 0 if ref[i-1] == hyp[j-1] else 1
        d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

distance = d[len(ref)][len(hyp)]
wer = distance / max(1, len(ref))
print(f'\nReference word count: {len(ref)}')
print(f'Hypothesis word count: {len(hyp)}')
print(f'Levenshtein word edits: {distance}')
print(f'Word Error Rate (WER): {wer:.4f} ({wer * 100:.2f}%)')

if wer <= 0.05:
    print('✓ LibriSpeech WER Evaluation PASSED (within tolerance <= 5%)')
    sys.exit(0)
else:
    print('✗ LibriSpeech WER Evaluation exceeded tolerance threshold')
    sys.exit(1)
EOF
