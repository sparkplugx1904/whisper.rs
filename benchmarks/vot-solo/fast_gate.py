#!/usr/bin/env python3
"""Fast quality gate for the VOT-Denpasar solo benchmark.

Scores one run directory (``merged.json`` plus ``transcribe.time.txt``) over a
few short time windows instead of the full 33.27 min recording. The default
windows are the two staged slices: 0-120 s and 1480-1580 s.

Reference clips are selected only when they lie completely inside a window.
Hypothesis words are assigned by word midpoint, so a 30 s ASR segment is split
at the window boundary rather than counted whole. A window fails when its WER
is worse than the recorded baseline, when it contains duplicate segments, when
it repeats 8-grams the reference does not contain, or when it emits more than
1.25x the reference word count with words the reference itself never contains
(a genuine reference repetition is allowed because those words are counted).

The baseline is a JSON file produced by ``--save-baseline`` from a known-good
run (``baseline_windows.json`` next to this script). Every number needed for a
verdict is printed; the process exits non-zero on FAIL.

Each window is also split into 30 s blocks. A block that emits less than half
of what the baseline emitted over the same audio fails: a reintroduced early
stop deletes 20+ s inside one window, and the window aggregate hides it because
the rest of the slice improves. A block the baseline under-covered is gated at
that same (low) level, never skipped, so dropping it entirely still fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jiwer

from evaluate import (
    clips_inside_window,
    duplicate_segments,
    load_boundaries,
    load_segments,
    normalize_text,
    parse_time_file,
    parse_window_specs,
    repeated_ngram_stats,
    window_diagnostics,
    word_tokens,
    assign_tokens_to_ranges,
)

SCHEMA = "vot-solo-fast-gate/1"
BASELINE_SCHEMA = "vot-solo-fast-gate-baseline/1"
DEFAULT_BOUNDARIES = Path(os.environ.get("VOT_BOUNDARIES", "/tmp/vot-solo/boundaries.json"))
DEFAULT_BASELINE = Path(__file__).with_name("baseline_windows.json")
MAX_WORD_RATIO = 1.25
# An 8-gram repeated three times is a decoding loop. Two occurrences can be
# legitimate boilerplate (the station branding recurs) or a surface variant
# that the reference spells differently, so only the third occurrence fails.
REPEAT_LOOP_OCCURRENCES = 3
# A 30 s block that emits less than half of what the baseline emitted over the
# same audio is a deletion, not a wording difference. The aggregate window WER
# cannot see it: a run that deletes one 30 s window and improves everywhere else
# still scores below the baseline (measured at 32.62 % against 35.82 %).
COVERAGE_BLOCK_S = 30.0
COVERAGE_FLOOR = 0.5
# A block with reference speech but almost no baseline output is still gated
# at the baseline's own level (never at zero), so a run cannot silently drop a
# block the baseline did decode.
COVERAGE_MIN_REFERENCE_WORDS = 20
SLICE_NAME = re.compile(r"slice_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.wav$")


def resolve_run_file(run_dir: Path, name: str) -> Path:
    for candidate in (run_dir / name, run_dir / "out" / name):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"{run_dir}: no {name} (looked in the run directory and out/)")


def load_run_metadata(run_dir: Path) -> dict:
    for candidate in (run_dir / "run.json", run_dir / "out" / "run.json"):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                return data
    return {}


def detect_offset(run_dir: Path, metadata: dict, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    wav = metadata.get("wav")
    if wav:
        match = SLICE_NAME.search(Path(str(wav)).name)
        if match:
            return float(match.group(1))
    return 0.0


def shift_segments(segments: list[dict], offset: float) -> list[dict]:
    if not offset:
        return segments
    shifted = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        clone = dict(segment)
        clone["start"] = start + offset
        clone["end"] = end + offset
        words = segment.get("words")
        if isinstance(words, list):
            clone["words"] = [
                {
                    **word,
                    "start": float(word.get("start", start)) + offset,
                    "end": float(word.get("end", word.get("start", start))) + offset,
                }
                for word in words
                if isinstance(word, dict)
            ]
        shifted.append(clone)
    return shifted


def wav_duration_s(path: Path | None) -> float | None:
    if not path or not path.is_file():
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return round(handle.getnframes() / rate, 3) if rate else None
    except (OSError, wave.Error):
        return None


def windows_from_clips(clips: list[dict], spec: str) -> list[tuple[float, float]]:
    start_text, _, end_text = spec.partition("-")
    try:
        first = int(start_text)
        last = int(end_text) if end_text else first
    except ValueError:
        raise SystemExit(f"bad --from-clip {spec!r}; expected FIRST or FIRST-LAST clip index")
    if first < 0 or last < first or last >= len(clips):
        raise SystemExit(
            f"--from-clip {spec!r} outside the {len(clips)} reference clips"
        )
    return [(float(clips[first]["start_s"]), float(clips[last]["end_s"]))]


def clip_rows(
    clips: list[dict],
    window_indexes: list[int],
    tokens: list[tuple[str, float, float, int]],
) -> list[dict]:
    """Per-clip scores inside one window, using word-midpoint assignment."""
    ranges = [(float(clips[index]["start_s"]), float(clips[index]["end_s"])) for index in window_indexes]
    assigned, _, _ = assign_tokens_to_ranges(tokens, ranges)
    rows = []
    for clip_index, words in zip(window_indexes, assigned):
        clip = clips[clip_index]
        reference = normalize_text(clip["sentence"])
        hypothesis = normalize_text(" ".join(words))
        if not reference:
            continue
        score = jiwer.process_words(reference, hypothesis)
        rows.append(
            {
                "index": clip_index,
                "path": clip.get("path", ""),
                "start_s": round(float(clip["start_s"]), 3),
                "end_s": round(float(clip["end_s"]), 3),
                "wer": round(score.wer, 6),
                "errors": score.substitutions + score.insertions + score.deletions,
                "reference_words": len(reference.split()),
                "hypothesis_words": len(hypothesis.split()),
                "reference": clip["sentence"],
                "hypothesis": " ".join(words),
            }
        )
    return rows


def load_baseline(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"{path}: unreadable baseline ({error})")
    if not isinstance(data, dict) or not isinstance(data.get("windows"), dict):
        raise SystemExit(f"{path}: not a fast-gate baseline file")
    return data


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def excess_words(hypothesis: str, reference: str) -> int:
    """Hypothesis word occurrences beyond what the reference itself contains.

    A window may legitimately run longer than the reference when the reference
    repeats a phrase: those extra occurrences are already counted in the
    reference word counts. Words the reference never emits (or emits fewer
    times) are inflation from a duplicated window or unanchored insertions.
    """
    hypothesis_counts = Counter(hypothesis.split())
    reference_counts = Counter(reference.split())
    return sum(
        count - reference_counts[word]
        for word, count in hypothesis_counts.items()
        if count > reference_counts[word]
    )


def coverage_blocks(
    segments: list[dict],
    clips: list[dict],
    start: float,
    end: float,
    tokens: list[tuple[str, float, float, int]],
) -> list[dict]:
    """Score consecutive 30 s sub-ranges of one window with the window scorer."""
    ranges: list[tuple[float, float]] = []
    cursor = start
    while cursor < end - 1e-9:
        nxt = min(cursor + COVERAGE_BLOCK_S, end)
        ranges.append((cursor, nxt))
        cursor = nxt
    return window_diagnostics(segments, clips, ranges, tokens)


def judge_coverage(blocks: list[dict], baseline_entry: dict | None) -> list[str]:
    """Fail a block that emits far fewer words than the baseline over the same audio.

    The comparison is against the baseline's own blocks, so the floor adapts to
    what the baseline achieved and a run is never required to beat it. A block
    the baseline decoded at all is still gated at half that count, which catches
    a reintroduced early stop that the window aggregate hides.
    """
    baseline_blocks = (baseline_entry or {}).get("blocks") or {}
    failures: list[str] = []
    for block in blocks:
        entry = baseline_blocks.get(block["window"])
        if not entry:
            continue
        baseline_reference = int(entry.get("reference_words", 0))
        baseline_hypothesis = int(entry.get("hypothesis_words", 0))
        if baseline_reference < COVERAGE_MIN_REFERENCE_WORDS:
            continue
        floor = max(1, (baseline_hypothesis + 1) // 2)
        if block["hypothesis_words"] < floor:
            failures.append(
                f"{block['window']} s emitted {block['hypothesis_words']} words against"
                f" {baseline_hypothesis} in the baseline ({block['reference_words']} reference)"
            )
    return failures


def judge_window(window: dict, baseline_entry: dict | None, args) -> list[str]:
    failures: list[str] = []
    reference_words = window["reference_words"]
    if not reference_words:
        return ["no reference clips fully inside the window"]
    if baseline_entry is None:
        if args.max_wer is None:
            failures.append("no baseline for this window and no --max-wer given")
        elif window["wer"] > args.max_wer:
            failures.append(
                f"WER {pct(window['wer'])} > --max-wer {pct(args.max_wer)}"
            )
    else:
        limit = baseline_entry["wer"] + args.wer_tolerance
        if window["wer"] > limit:
            failures.append(
                f"WER {pct(window['wer'])} > baseline {pct(baseline_entry['wer'])}"
                f" +{pct(args.wer_tolerance)} tolerance"
            )
    # Duplicates, 8-gram loops and word inflation are absolute defects; the
    # baseline only raises the tolerated count, it is not required to check them.
    baseline_duplicates = int(baseline_entry.get("duplicate_segments", 0)) if baseline_entry else 0
    if window["duplicate_segments"] > baseline_duplicates:
        failures.append(
            f"{window['duplicate_segments']} duplicate segments"
            f" (baseline {baseline_duplicates})"
        )
    baseline_repeats = int(baseline_entry.get("unexplained_repeats", 0)) if baseline_entry else 0
    looped = [
        example
        for example in window.get("repeated_ngram_examples", [])
        if example["count"] >= REPEAT_LOOP_OCCURRENCES
        and example.get("unexplained", 1) > 0
    ]
    if window["unexplained_repeats"] > baseline_repeats and looped:
        failures.append(
            f"{len(looped)} 8-grams repeat at least"
            f" {REPEAT_LOOP_OCCURRENCES} times without a matching reference"
            f" (e.g. {looped[0]['ngram']!r} x{looped[0]['count']})"
        )
    if reference_words and window["hypothesis_words"] > MAX_WORD_RATIO * reference_words:
        excess = excess_words(window["hypothesis"], window["reference"])
        if excess:
            failures.append(
                f"hypothesis {window['hypothesis_words']} words vs {reference_words} reference"
                f" ({window['word_ratio']:.2f}x > {MAX_WORD_RATIO:.2f}x) with"
                f" {excess} words beyond the reference word counts"
            )
    failures.extend(judge_coverage(window.get("blocks") or [], baseline_entry))
    return failures


def render(result: dict) -> str:
    lines = []
    commit = result["commit"] or "unknown"
    lines.append(
        f"fast gate: {result['run_dir']} (commit {commit}, offset {result['offset_s']:g} s)"
    )
    lines.append(
        "window       clips  refW  hypW   ratio     WER     CER   sub   ins   del  dup  rep/unexp  verdict"
    )
    for window in result["windows"]:
        ratio = f"{window['word_ratio']:.2f}x" if window["word_ratio"] is not None else "-"
        lines.append(
            "{name:<12} {clips:>5} {rw:>5} {hw:>5} {ratio:>7} {wer:>7} {cer:>7}"
            " {sub:>5} {ins:>5} {dels:>5} {dup:>4} {rep:>4}/{unexp:<5} {verdict}".format(
                name=window["window"],
                clips=window["clips"],
                rw=window["reference_words"],
                hw=window["hypothesis_words"],
                ratio=ratio,
                wer=pct(window["wer"]),
                cer=pct(window["cer"]),
                sub=window["substitutions"],
                ins=window["insertions"],
                dels=window["deletions"],
                dup=window["duplicate_segments"],
                rep=window["repeated_ngrams"],
                unexp=window["unexplained_repeats"],
                verdict=window["verdict"],
            )
        )
        blocks = window.get("blocks") or []
        if blocks:
            lines.append(
                "  blocks "
                + " | ".join(
                    f"{block['window']}:{block['hypothesis_words']}/{block['reference_words']}"
                    for block in blocks
                )
            )
    run = result["run"]
    rtf = f"{run['rtf']:.3f}" if run["rtf"] is not None else "-"
    lines.append(
        f"run: {run['segments']} segments, wall {run['wall_s']:.1f} s,"
        f" audio {run['audio_duration_s']:.1f} s, RTF {rtf},"
        f" peak RSS {run['peak_rss_mb']:.0f} MB,"
        f" duplicate segments {run['duplicate_segments']},"
        f" repeated 8-grams {run['repeated_8grams']}"
        f" ({run['unexplained_repeats']} unexplained)"
    )
    if result["baseline"]:
        lines.append(f"baseline: {result['baseline']}")
    if result["failures"]:
        lines.append("")
        lines.append("FAILURES")
        for failure in result["failures"]:
            lines.append(f"- {failure}")
        lines.append("")
        lines.append("offending clips")
        for row in result["offending_clips"]:
            lines.append(
                f"  clip {row['index']} ({row['start_s']:.2f}-{row['end_s']:.2f} s,"
                f" window {row['window']}, WER {pct(row['wer'])},"
                f" {row['hypothesis_words']}/{row['reference_words']} words)"
            )
            lines.append(f"    REF: {row['reference']}")
            lines.append(f"    HYP: {row['hypothesis'] or '<empty>'}")
    lines.append(f"VERDICT: {result['verdict']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory with merged.json and transcribe.time.txt")
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument(
        "--windows",
        action="append",
        metavar="START-END",
        help="time window in seconds, repeatable or comma separated (default: 0-120,1480-1580)",
    )
    windows.add_argument(
        "--from-clip",
        metavar="FIRST[-LAST]",
        help="derive the window from reference clip indexes instead of seconds",
    )
    parser.add_argument("--offset", type=float, default=None, help="seconds to add to segment times")
    parser.add_argument("--baseline", type=Path, default=None, help="baseline JSON (default: baseline_windows.json)")
    parser.add_argument("--no-baseline", action="store_true", help="skip baseline comparison")
    parser.add_argument("--max-wer", type=float, default=None, help="absolute WER ceiling when no baseline exists")
    parser.add_argument("--wer-tolerance", type=float, default=0.0, help="allowed WER delta over baseline")
    parser.add_argument("--show", type=int, default=8, help="offending clips to print")
    parser.add_argument("--json", type=Path, help="write the full result as JSON")
    parser.add_argument("--markdown", type=Path, help="write the full result as a markdown report")
    parser.add_argument("--save-baseline", type=Path, help="write per-window metrics as a baseline file")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir}: not a directory")
    segments_path = resolve_run_file(run_dir, "merged.json")
    time_path = resolve_run_file(run_dir, "transcribe.time.txt")
    boundaries_path = args.boundaries
    if not boundaries_path.is_file():
        raise SystemExit(f"{boundaries_path}: reference boundaries not found (set --boundaries or VOT_BOUNDARIES)")

    boundaries = load_boundaries(boundaries_path)
    clips = boundaries["clips"]
    raw_segments = load_segments(segments_path)
    metadata = load_run_metadata(run_dir)
    offset = detect_offset(run_dir, metadata, args.offset)
    segments = shift_segments(raw_segments, offset)

    if args.from_clip:
        window_specs = windows_from_clips(clips, args.from_clip)
    else:
        window_specs = parse_window_specs(args.windows)

    baseline_path: Path | None = None
    if not args.no_baseline:
        candidate = args.baseline if args.baseline is not None else DEFAULT_BASELINE
        if candidate.is_file():
            baseline_path = candidate
        elif args.baseline is not None:
            raise SystemExit(f"{candidate}: baseline file not found")
    baseline = load_baseline(baseline_path)
    baseline_windows = baseline.get("windows", {})

    tokens, fallback_segments = word_tokens(segments)
    windows_report = window_diagnostics(segments, clips, window_specs, tokens)
    for window in windows_report:
        window["blocks"] = [
            {
                "window": block["window"],
                "reference_words": block["reference_words"],
                "hypothesis_words": block["hypothesis_words"],
                "word_ratio": block["word_ratio"],
            }
            for block in coverage_blocks(
                segments, clips, window["start_s"], window["end_s"], tokens
            )
        ]

    failures: list[str] = []
    offending: list[dict] = []
    for window in windows_report:
        window_failures = judge_window(window, baseline_windows.get(window["window"]), args)
        window["verdict"] = "PASS" if not window_failures else "FAIL"
        window["failures"] = window_failures
        if window_failures:
            failures.extend(f"{window['window']} s: {failure}" for failure in window_failures)
            indexes = clips_inside_window(clips, window["start_s"], window["end_s"])
            rows = clip_rows(clips, indexes, tokens)
            rows.sort(key=lambda row: (-row["errors"], -row["wer"]))
            for row in rows[: args.show]:
                offending.append({**row, "window": window["window"]})

    all_words = [text for text, _, _, _ in tokens]
    reference_words = [word for clip in clips for word in normalize_text(clip["sentence"]).split()]
    run_repeats = repeated_ngram_stats(all_words, reference_words)
    duplicates = duplicate_segments(segments)
    measurement = parse_time_file(time_path)

    audio_duration = wav_duration_s(Path(metadata["wav"])) if metadata.get("wav") else None
    if not audio_duration:
        audio_duration = max((float(segment.get("end", 0.0)) for segment in segments), default=0.0)

    result = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run_dir.resolve()),
        "segments_file": str(segments_path),
        "time_file": str(time_path),
        "boundaries": str(boundaries_path),
        "baseline": str(baseline_path) if baseline_path else None,
        "label": metadata.get("label"),
        "commit": metadata.get("commit"),
        "wav": metadata.get("wav"),
        "offset_s": offset,
        "params": {
            "language": "id",
            "threads": 4,
            "beam": metadata.get("beam", 5),
        },
        "run": {
            "segments": len(segments),
            "fallback_segments": fallback_segments,
            "wall_s": measurement["wall_s"],
            "audio_duration_s": round(audio_duration, 3),
            "rtf": round(measurement["wall_s"] / audio_duration, 4) if audio_duration else None,
            "peak_rss_mb": measurement["peak_rss_mb"],
            "duplicate_segments": len(duplicates),
            "duplicate_examples": duplicates[:5],
            "repeated_8grams": run_repeats["repeated_ngrams"],
            "unexplained_repeats": run_repeats["unexplained_repeats"],
        },
        "windows": windows_report,
        "failures": failures,
        "offending_clips": offending,
        "verdict": "FAIL" if failures else "PASS",
    }

    if args.save_baseline:
        saved = {
            "schema": BASELINE_SCHEMA,
            "generated_at": result["generated_at"],
            "commit": result["commit"],
            "label": result["label"],
            "wav": result["wav"],
            "offset_s": offset,
            "params": result["params"],
            "windows": {
                window["window"]: {
                    "wer": window["wer"],
                    "cer": window["cer"],
                    "reference_words": window["reference_words"],
                    "hypothesis_words": window["hypothesis_words"],
                    "word_ratio": window["word_ratio"],
                    "duplicate_segments": window["duplicate_segments"],
                    "repeated_ngrams": window["repeated_ngrams"],
                    "unexplained_repeats": window["unexplained_repeats"],
                    "blocks": {
                        block["window"]: {
                            "reference_words": block["reference_words"],
                            "hypothesis_words": block["hypothesis_words"],
                            "word_ratio": block["word_ratio"],
                        }
                        for block in window["blocks"]
                    },
                }
                for window in windows_report
            },
        }
        args.save_baseline.write_text(
            json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("```\n" + render(result) + "\n```\n", encoding="utf-8")

    print(render(result))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
