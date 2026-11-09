# Solo whisper.rs evaluation

Evaluate this repository's engine on the curated `VOT-Denpasar_18-12-25-0`
clips, concatenated in metadata order without inserted silence.

## Pipeline

1. `download_dataset.py` downloads `metadata.tsv` and all 282 clips from
   `Sparkplugx1904/voiceoftrisma-voice-datasets`.
2. `merge_clips.py` decodes every clip to 16 kHz mono signed 16-bit PCM with
   ffmpeg and appends the raw frames. The output is 33.27 minutes of audio with
   no gaps; `boundaries.json` records the exact sample range of every clip.
3. `run_whisper_rs.sh` builds `whisper-cli` and transcribes the merged WAV with
   `ggml-large-v3-turbo.bin` (F16, the model published under
   `Sparkplugx1904/whisper.cpp`), language `id`, four threads, beam size 5.
   Timestamps stay enabled so ASR segments can be mapped back to clips.
4. `evaluate.py` computes overall WER/CER, assigns each ASR segment to a clip by
   midpoint, and reports per-clip WER, RMS, average log probability, no-speech
   probability, error type, frequent substitutions/insertions/deletions, and
   the worst clips with reference/hypothesis pairs.

## Running

The workflow is manual:

```bash
gh workflow run benchmark-vot-solo.yml
gh run watch
```

The report and all intermediate transcripts are uploaded as the
`vot-solo-report` artifact. `report.md` is also written to the workflow step
summary.

## Scoring method

Reference and hypothesis both go through the same normalization: remove
bracketed annotations such as `[SPEAKER_TURN]` or `(music)`, lowercase, replace
Unicode punctuation with spaces, collapse whitespace. WER and CER are computed
with `jiwer`.

Per-clip assignment is diagnostic: an ASR segment is counted for the clip that
contains its midpoint. Segments spanning a boundary are not split, so per-clip
numbers will not sum exactly to the overall error count. The overall score is
computed on the full concatenated transcript and is the primary metric.

## Interpreting the report

- High `deletions` on quiet clips points to missed low-energy speech.
- High `insertions` or a large hypothesis/reference word ratio on music or noise
  points to hallucination.
- Low `avg_logprob` with high WER points to decoding uncertainty rather than
  segmentation error.
- `short_clips_median_wer` compared with `long_clips_median_wer` shows whether
  short utterances lose context.
- `worst_error_share` shows how concentrated the errors are: a small set of
  clips dominating the error mass is usually a content problem (names, numbers,
  music), not a uniform model weakness.