# LibriSpeech acceptance gate

Quality gate for `whisper.rs`, replacing the retired VOT-Denpasar benchmark. The
corpus is LibriSpeech `dev-clean`, speaker 2277, chapter 149896: 35 utterances,
166.38 s after concatenation, five engine windows. A single 6 s utterance cannot
exercise the seek, the prompt history or a window boundary, which is why the
chapter is concatenated instead of scored one file at a time.

## Corpus and staging

`prepare_chapter.py` decodes every chapter utterance to 16 kHz mono signed
16-bit PCM with `ffmpeg` and appends `--gap` seconds of digital silence after
each one, including the last. The result is deterministic and was verified
against the staged files: 2,662,080 frames (166.38 s) and a 537-word reference.

```bash
# from an extracted LibriSpeech tree
python3 benchmarks/librispeech/prepare_chapter.py \
    --corpus-root /tmp/libri/LibriSpeech \
    --chapter dev-clean/2277/149896 \
    --out /tmp/libri

# or straight from the dev-clean tarball (only the chapter is extracted)
curl -L -o dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
python3 benchmarks/librispeech/prepare_chapter.py \
    --tar dev-clean.tar.gz --chapter dev-clean/2277/149896 --out /tmp/libri
```

Outputs: `chapter.wav`, `reference.txt` (space-joined transcript in utterance
order) and `boundaries.json` (per-utterance spans and sentences). The
checked-in `boundaries.json` / `reference.txt` are the artifacts for this
chapter; regenerate them only if the concatenation changes.

## Running

`run.sh` expects a built `whisper-cli` and a model; it never builds and never
starts a second engine process. Transcription is serialized through
`replica/coordination/bench.sh` (exclusive `/tmp/vot-solo/bench.lock`), with
`-l en -t 4 -b 5 --no-prints` and the model from the environment.

```bash
WHISPER_MODEL=/tmp/libri/models/ggml-large-v3-turbo.bin \
    benchmarks/librispeech/run.sh full

WHISPER_MODEL=/tmp/libri/models/ggml-large-v3-turbo.bin \
    benchmarks/librispeech/run.sh fast
```

| variable | default |
| --- | --- |
| `WHISPER_MODEL` | required |
| `WHISPER_BIN` | `target/release/whisper-cli` |
| `LIBRI_ROOT` | `/tmp/libri` |
| `BOUNDARIES` | `$LIBRI_ROOT/boundaries.json` |
| `BASELINE` | `benchmarks/librispeech/baseline.json` |
| `LABEL` / `OUT` | `librispeech-<mode>` / `$LIBRI_ROOT/runs/$LABEL` |

`fast` truncates the staged chapter to the first ten utterances (54.07 s, two
windows) into `$LIBRI_ROOT/fast.wav`, so a quick loop is possible without a
second corpus. `full` uses the whole chapter.

The transcription command comes from `bench.sh`; `run.sh` adds nothing to the
model or decoding parameters, so the numbers are comparable between modes and
across builds.

## Baseline and acceptance

A baseline must be produced from a reviewed build, once:

```bash
WHISPER_MODEL=... SAVE_BASELINE=1 benchmarks/librispeech/run.sh full
WHISPER_MODEL=... SAVE_BASELINE=1 benchmarks/librispeech/run.sh fast
```

That writes `benchmarks/librispeech/baseline.json` with one entry per mode.
The committed baseline has the `full` entry only (the clean-tree artifact
described below). The `fast` entry needs one fast run of the reviewed build;
until then `run.sh fast` exits 3 with that instruction rather than scoring
against the wrong baseline.
A baseline can also be saved from an existing scored run without re-running the
engine:

```bash
python3 benchmarks/librispeech/gate.py /tmp/libri/runs/<label> --mode full \
    --no-baseline --max-wer 1.0 --save-baseline benchmarks/librispeech/baseline.json
```

The gate records `binary_sha256` from the run's `binary` path (review item 1:
the commit alone does not identify the working tree that was built).

Writing a baseline requires `BASELINE_NOTE` and, if the file already exists,
`OVERWRITE_BASELINE=1`. Both guards exist because a baseline is only meaningful
for a frozen, reviewed source tree:

```bash
BASELINE_NOTE="reviewed <source-sha>, binary <sha256>, accepted <date>" \
OVERWRITE_BASELINE=1 WHISPER_MODEL=... SAVE_BASELINE=1 \
    benchmarks/librispeech/run.sh full
```

The committed `full` baseline is the clean-tree artifact from commit
`fe63be3af196` (engine commit `c638f07`), source `transcribe.rs 5c30644a`,
`beam.rs 3a6b0899`, `filter.rs 8f841311`, binary sha `878cae19…`, WER 0.56 %
(2 sub / 1 ins / 0 del) against 1.48 % for the accepted pass-2 build. It
supersedes the two artifacts accepted before the commit: `fix3` (`6e099736…`,
source `122cba5c`) and `fix5` (`c9471857…`, source `1237ba1f`), whose
transcripts are byte-identical. The note in the file records the measurement:
local RTF is 5.247 versus 4.109 for pass-2 (single runs on a contended box;
`fix3` 4.918 and `fix5` 5.546 - no mechanism is established and no performance
claim is made; median-of-3 is required before one). The frozen rule truncates
the sequence to `result_len` before the single-timestamp test, which is what
upstream (`whisper.cpp:7729` then `:7940`) does; the remaining behavioural
delta is that a window ending `[.., timestamp, text]` skips the tail instead of
re-decoding it. The continuity and timestamp-bounds checks are the guard for
that delta and must stay enabled. Do not regenerate the baseline while the
tree is changing: a baseline saved from an unfrozen tree certifies an untested
grammar. The gate prints the baseline note and the run's binary sha so a
mismatch is visible in every report. A run is only covered by the accepted
baseline if its binary sha matches the note; check the source hashes first
(`sha256sum crates/whisper/src/{transcribe,beam,filter}.rs`), because the
working tree can move after a baseline is frozen and a fresh build will not
carry the accepted provenance.
From then on `run.sh` fails with exit 3 if the baseline is missing rather than
silently running with no ceiling.

`gate.py` charges the run with:

| check | threshold |
| --- | --- |
| WER | `<=` baseline WER for the same mode, zero tolerance by default |
| duplicate segments | `<=` baseline count (identical normalized text over the same or overlapping audio) |
| 8-gram loop | fail when an 8-gram repeats at least 3 times and the reference does not contain it |
| word inflation | fail above 1.25x the reference word count when the hypothesis emits words the reference never contains |
| block inflation | same 1.25x rule applied per 30 s block, so one duplicated window cannot hide in the chapter total |
| block coverage | fail when a 30 s block with >= 20 reference words emits less than half of what the baseline emitted there |
| timestamp bounds | fail when the last segment ends more than 1 s past the audio (the reverted pass-3 build emitted 196 s on 166 s) |
| segment continuity | fail when consecutive segments skip more than 5 s of audio; the largest legitimate gap observed is 0.92 s (the 1 s seek floor), the early-stop signature is 20+ s. The 5 s default assumes read speech with short pauses; for other corpora derive it from the reference clip gaps |

Block coverage is the deletion-side guard: an early stop deletes 20+ s inside
one window, and the chapter WER can still beat the baseline when the rest of
the audio improves. Both block checks use the same 30 s grid aligned to the
start of the file; a block with fewer than 20 reference words is not gated.

Nothing here may be relaxed to make a run pass. A failing gate means either the
engine regressed or the baseline was produced by a build that should not have
been trusted; regenerate the baseline only from a reviewed build and record its
binary hash.

## Self-test

`selftest.py` proves the checks without an engine. It builds synthetic
transcripts from the boundaries against a 60 %-WER baseline, so every attack
stays below the baseline WER and only the anomaly checks can reject it:

```bash
python3 benchmarks/librispeech/selftest.py --mode full   # 5 passed
python3 benchmarks/librispeech/selftest.py --mode fast   # 5 passed
```

Covered: clean (PASS), 30 % longer hypothesis, a duplicated 30 s block, a
30 s block replaced by 9 words, a 30 s block deleted.

## CI

`.github/workflows/benchmark-librispeech.yml` runs the full gate manually and
nightly. It downloads `dev-clean`, stages the chapter, downloads
`ggml-large-v3-turbo`, builds the CLI in release mode and uploads the run JSON,
the gate report and both audio files as artifacts. The VOT workflow is manual
only and is not part of the acceptance path.
