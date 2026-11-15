#!/usr/bin/env python3
"""Self-test the LibriSpeech gate without an engine.

Builds synthetic transcripts from the chapter boundaries and checks that the
gate accepts a clean transcript and rejects each known decoding defect. The
synthetic baseline has 60 % word substitutions, so every attack stays *below*
the baseline WER and only the anomaly checks can reject it:

    clean            reference text, unchanged                    PASS
    long_hypothesis  reference + 30 % distinct words              FAIL (1.25x)
    dup_block        the first 30 s block emitted again at +30 s  FAIL (block 1.25x)
    truncated_block  the 30-60 s block replaced by 9 words        FAIL (coverage)
    deleted_block    the 30-60 s block removed                    FAIL (coverage)

Usage:
    python3 benchmarks/librispeech/selftest.py --mode full
    python3 benchmarks/librispeech/selftest.py --mode fast
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "gate.py"
TIME_FILE = (
    '\tCommand being timed: "selftest"\n'
    "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.00\n"
    "\tMaximum resident set size (kbytes): 0\n"
)
FAST_UTTERANCES = 10


def segments(clips: list[dict], shift_s: float = 0.0) -> list[dict]:
    return [
        {
            "start": round(float(clip["start_s"]) + shift_s, 3),
            "end": round(float(clip["end_s"]) + shift_s, 3),
            "text": clip["sentence"],
        }
        for clip in clips
    ]


def bad_baseline(clips: list[dict], ratio: float = 0.6) -> list[dict]:
    out = []
    counter = 0
    for clip in clips:
        tokens = clip["sentence"].split()
        replaced = [
            f"sub{counter + index:04d}" if index / max(1, len(tokens)) < ratio else token
            for index, token in enumerate(tokens)
        ]
        counter += len(tokens)
        out.append(
            {
                "start": round(float(clip["start_s"]), 3),
                "end": round(float(clip["end_s"]), 3),
                "text": " ".join(replaced),
            }
        )
    return out


def write_run(root: Path, name: str, run_segments: list[dict]) -> Path:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "merged.json").write_text(json.dumps(run_segments), encoding="utf-8")
    (target / "transcribe.time.txt").write_text(TIME_FILE, encoding="utf-8")
    return target


def gate(run: Path, boundaries: Path, mode: str, baseline: Path | None) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(GATE),
        str(run),
        "--boundaries",
        str(boundaries),
        "--mode",
        mode,
    ]
    command += ["--baseline", str(baseline)] if baseline else ["--no-baseline", "--max-wer", "1.0"]
    return subprocess.run(command, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, default=HERE / "boundaries.json")
    parser.add_argument("--mode", choices=("fast", "full"), default="full")
    parser.add_argument("--out", type=Path, default=Path("/tmp/librispeech-gate-selftest"))
    args = parser.parse_args()

    data = json.loads(args.boundaries.read_text(encoding="utf-8"))
    clips = data["clips"]
    if args.mode == "fast":
        clips = clips[:FAST_UTTERANCES]
    if len(clips) < 3:
        raise SystemExit("need at least three utterances")

    duration = float(clips[-1]["end_s"]) + float(data.get("gap_s", 0.0))
    if duration < 90.0 and args.mode == "full":
        raise SystemExit("full mode needs at least 90 s for the coverage attack; use --mode fast")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    write_run(args.out, "bad_baseline", bad_baseline(clips))
    baseline = args.out / "baseline.json"
    saved = subprocess.run(
        [
            sys.executable,
            str(GATE),
            str(args.out / "bad_baseline"),
            "--boundaries",
            str(args.boundaries),
            "--mode",
            args.mode,
            "--no-baseline",
            "--max-wer",
            "1.0",
            "--save-baseline",
            str(baseline),
            "--baseline-note",
            "selftest synthetic baseline (60% substitutions); not an acceptance baseline",
        ],
        capture_output=True,
        text=True,
    )
    if saved.returncode != 0:
        print(saved.stdout, saved.stderr)
        raise SystemExit("could not save the synthetic baseline")

    clean = segments(clips)
    first_block = [clip for clip in clips if float(clip["end_s"]) <= 30.0 + 1e-3]
    middle_midpoints = {index for index, clip in enumerate(clips)
                        if 30.0 <= (float(clip["start_s"]) + float(clip["end_s"])) / 2.0 < 60.0}
    kept = segments([clip for index, clip in enumerate(clips) if index not in middle_midpoints])
    reference_words = sum(len(clip["sentence"].split()) for clip in clips)

    fixtures = {
        "clean": clean,
        "long_hypothesis": clean
        + [
            {
                "start": round(duration - 2.0, 3),
                "end": round(duration - 1.0, 3),
                "text": " ".join(f"extra{index:03d}" for index in range(round(0.30 * reference_words))),
            }
        ],
        "dup_block": clean + segments(first_block, shift_s=30.0),
        "truncated_block": kept
        + [{"start": 30.0, "end": 37.0, "text": "di dalam tubuh yang sehat terdapat jiwa yang kuat"}],
        "deleted_block": kept,
    }
    expectations = {
        "clean": (0, None),
        "long_hypothesis": (1, "beyond the reference word counts"),
        "dup_block": (1, "words vs"),
        "truncated_block": (1, "emitted 9 words against"),
        "deleted_block": (1, "emitted 0 words against"),
    }

    failures = 0
    print(f"mode {args.mode}, {len(clips)} utterances, {duration:.2f} s, {reference_words} reference words")
    for name, (want_rc, needle) in expectations.items():
        run = write_run(args.out, name, fixtures[name])
        result = gate(run, args.boundaries, args.mode, baseline)
        ok = result.returncode == want_rc and (needle is None or needle in result.stdout)
        failures += 0 if ok else 1
        first_line = next(
            (line for line in result.stdout.splitlines() if line.startswith("- ")), "PASS"
        )
        print(f"{'OK  ' if ok else 'FAIL'} {name:16} rc={result.returncode} {first_line}")
        if not ok:
            print(result.stdout)
            print(result.stderr)

    print(f"{len(expectations) - failures} passed, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())