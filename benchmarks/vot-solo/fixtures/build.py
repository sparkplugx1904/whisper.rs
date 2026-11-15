#!/usr/bin/env python3
"""Build the adversarial fixtures used to prove the fast gate cannot be bypassed.

Fixtures are derived from the reference clips so the expected verdict is known
in advance. All but ``early_block_truncation`` must fail; that one documents a
limit of the baseline and must pass until the baseline is ratcheted.

``dup_overlap``
    The same block duplicated at identical timestamps. ``duplicate_segments``
    flags it directly, which proves the duplicate detector itself works.

``dup_window``
    All reference clips of the 1480-1580 s window, plus a second copy of the
    1481-1508 s block shifted 60 s later. The copy shares no timestamp with its
    source, so ``duplicate_segments`` cannot see it; only the word-count
    inflation check can. WER stays below that window's 46.63 % baseline.

``long_hypothesis``
    All reference clips of the 0-120 s window, plus 30 % extra distinct words
    at the end. No 8-gram repeats, so only the word-count check can catch it.
    WER (30.14 %) stays below that window's 35.82 % baseline.

``truncated_window``
    All reference clips of the 0-120 s window, except that the 30-60 s block is
    replaced by the 9-word `babcf00` output ending at 37.38 s. Aggregate WER is
    about 13.8 % (below the 35.82 % baseline) and the word ratio is 0.86, so
    only the per-30 s block coverage check can catch the 22.6 s deletion.

``early_block_truncation``
    All reference clips of the 0-120 s window except the 0-30 s block, which
    keeps only the first station-ID clip. The baseline `236d89d` emitted exactly
    that station ID in 0-30 s (9 of 67 reference words), so no baseline-relative
    rule can separate this transcript from the baseline. It is expected to PASS
    against `baseline_windows.json` and to FAIL once the baseline is regenerated
    from a reviewed passing run (0-30 s then covers 75 of 67 words).

Usage:
    python3 benchmarks/vot-solo/fixtures/build.py \
        --boundaries /tmp/vot-solo/boundaries.json \
        --out benchmarks/vot-solo/fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import clips_inside_window, load_boundaries, normalize_text

TIME_FILE = (
    '\tCommand being timed: "fixture"\n'
    "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.00\n"
    "\tMaximum resident set size (kbytes): 0\n"
)


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def segments_from_clips(clips: list[dict], indexes: list[int], shift_s: float = 0.0) -> list[dict]:
    segments = []
    for index in indexes:
        clip = clips[index]
        start = float(clip["start_s"]) + shift_s
        end = float(clip["end_s"]) + shift_s
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": clip["sentence"],
                "avg_logprob": -0.2,
                "no_speech_prob": 0.0,
            }
        )
    return segments


def write_fixture(out: Path, name: str, segments: list[dict]) -> None:
    target = out / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "merged.json").write_text(
        json.dumps(segments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (target / "transcribe.time.txt").write_text(TIME_FILE, encoding="utf-8")
    print(f"{name}: {len(segments)} segments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    clips = load_boundaries(args.boundaries)["clips"]
    args.out.mkdir(parents=True, exist_ok=True)

    late = clips_inside_window(clips, 1480.0, 1580.0)
    block = [index for index in late if float(clips[index]["start_s"]) < 1508.0]
    duplicate = segments_from_clips(clips, late) + segments_from_clips(clips, block, shift_s=60.0)
    write_fixture(args.out, "dup_window", duplicate)

    overlapping = segments_from_clips(clips, late) + segments_from_clips(clips, block)
    write_fixture(args.out, "dup_overlap", overlapping)

    early = clips_inside_window(clips, 0.0, 120.0)
    reference_words = sum(word_count(clips[index]["sentence"]) for index in early)
    extra_words = round(0.30 * reference_words)
    filler = " ".join(f"tambahan{i:02d}" for i in range(extra_words))
    long_hypothesis = segments_from_clips(clips, early) + [
        {
            "start": 118.0,
            "end": 120.0,
            "text": filler,
            "avg_logprob": -0.2,
            "no_speech_prob": 0.0,
        }
    ]
    write_fixture(args.out, "long_hypothesis", long_hypothesis)
    print(f"long_hypothesis: {extra_words} extra words on {reference_words} reference words")

    kept = [
        index
        for index in early
        if float(clips[index]["end_s"]) <= 30.0 or float(clips[index]["start_s"]) >= 60.0
    ]
    truncated = segments_from_clips(clips, kept) + [
        {
            "start": 30.0,
            "end": 37.38,
            "text": "Di dalam tubuh yang sehat terdapat jiwa yang kuat",
            "avg_logprob": -0.6615,
            "no_speech_prob": 0.0,
        }
    ]
    truncated.sort(key=lambda segment: segment["start"])
    write_fixture(args.out, "truncated_window", truncated)

    before_30 = [index for index in early if float(clips[index]["end_s"]) <= 30.0]
    after_30 = [index for index in early if float(clips[index]["start_s"]) >= 30.0]
    write_fixture(args.out, "early_block_truncation", segments_from_clips(clips, before_30[:1] + after_30))
    print(
        "truncated_window: 9 words for the 30-60 s block"
        f" ({sum(word_count(clips[i]['sentence']) for i in clips_inside_window(clips, 30.0, 60.0))} reference words)"
    )
    print(
        "early_block_truncation: station ID only before 30 s"
        f" ({sum(word_count(clips[i]['sentence']) for i in before_30[:1])} of"
        f" {sum(word_count(clips[i]['sentence']) for i in clips_inside_window(clips, 0.0, 30.0))} reference words)"
    )


if __name__ == "__main__":
    main()