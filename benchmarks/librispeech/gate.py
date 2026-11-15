#!/usr/bin/env python3
"""Acceptance gate for the LibriSpeech benchmark.

Scores one ``whisper-cli`` run directory against the staged chapter reference and
fails on a quality regression. The checks are the ones proven on the retired VOT
dataset, without VOT inputs:

* WER above the baseline (CER, substitutions, insertions, deletions are printed),
* duplicate segments (same normalized text over the same or overlapping audio),
* an 8-gram repeated at least three times that the reference does not contain,
* more than 1.25x the reference word count with words the reference never emits,
* a 30 s block that emits less than half of what the baseline emitted there,
* a last timestamp past the end of the audio,
* a gap of more than 5 s between consecutive segments (a skipped window).

The last check is the deletion-side guard: an early stop deletes 20+ s inside one
window and the aggregate WER can still beat the baseline when the rest of the
chapter improves.

Modes:
    full  the whole staged chapter (~166 s, five engine windows)
    fast  the first ten utterances (~54 s, two windows)

A baseline is produced from a reviewed run and stored per mode:

    python3 benchmarks/librispeech/gate.py /tmp/libri/runs/rev-full \
        --mode full --save-baseline benchmarks/librispeech/baseline.json

Exit status is 0 only for PASS.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jiwer

SCHEMA = "librispeech-gate/1"
BASELINE_SCHEMA = "librispeech-gate-baseline/1"
HERE = Path(__file__).resolve().parent
DEFAULT_BOUNDARIES = HERE / "boundaries.json"
DEFAULT_BASELINE = HERE / "baseline.json"
FAST_UTTERANCES = 10
BLOCK_S = 30.0
MAX_WORD_RATIO = 1.25
REPEAT_LOOP_OCCURRENCES = 3
REPEAT_NGRAM = 8
COVERAGE_MIN_REFERENCE_WORDS = 20
DURATION_TOLERANCE_S = 1.5
# Observed on the accepted LibriSpeech runs: the largest legitimate gap between
# consecutive segments is 0.92 s (the 1 s seek floor); the early-stop regression
# skipped 20+ s. Timestamps beyond the audio are a defect on their own (the
# reverted pass-3 filter build emitted a segment ending at 196 s on 166 s of
# audio). The 5 s gap assumes read speech with short pauses; for broadcast
# audio with music beds or long pauses, derive the threshold from the corpus
# (for example from the reference clip gaps) instead of relying on the default.
MAX_SEGMENT_GAP_S = 5.0
MAX_TIMESTAMP_OVERRUN_S = 1.0

ANNOTATION = re.compile(r"[\[\(][^\]\)]*[\]\)]")
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE = re.compile(r"\s+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = ANNOTATION.sub(" ", text)
    text = text.lower().replace("\ufeff", " ")
    text = PUNCTUATION.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def load_boundaries(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "clips" not in data:
        raise SystemExit(f"{path}: not a boundaries file")
    return data


def load_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("expected the CLI JSON output to be a list of segments")
    return [
        segment
        for segment in data
        if isinstance(segment, dict) and str(segment.get("text", "")).strip()
    ]


def word_tokens(segments: list[dict]) -> tuple[list[tuple[str, float, float]], int]:
    """Normalized word tokens with absolute spans; midpoint fallback per segment."""
    tokens: list[tuple[str, float, float]] = []
    fallback = 0
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        words = segment.get("words")
        if isinstance(words, list) and words:
            for word in words:
                if not isinstance(word, dict):
                    continue
                text = normalize_text(str(word.get("word", "")))
                if not text:
                    continue
                tokens.append(
                    (text, float(word.get("start", start)), float(word.get("end", start)))
                )
        else:
            text = normalize_text(str(segment.get("text", "")))
            if not text:
                continue
            fallback += 1
            midpoint = (start + end) / 2.0
            for word in text.split():
                tokens.append((word, midpoint, midpoint))
    return tokens, fallback


def duplicate_segments(segments: list[dict], min_overlap_s: float = 1.0) -> list[dict]:
    history: list[dict] = []
    duplicates: list[dict] = []
    for index, segment in enumerate(segments):
        text = normalize_text(str(segment.get("text", "")))
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        entry = {"index": index, "start_s": round(start, 3), "end_s": round(end, 3), "text": text}
        for earlier in history:
            if earlier["text"] != text:
                continue
            overlap = min(end, earlier["end_s"]) - max(start, earlier["start_s"])
            same_span = abs(start - earlier["start_s"]) < 0.01 and abs(end - earlier["end_s"]) < 0.01
            if same_span or overlap >= min_overlap_s:
                duplicates.append({**entry, "duplicates_index": earlier["index"]})
                break
        history.append(entry)
    return duplicates


def repeated_ngram_stats(words: list[str], reference_words: list[str], n: int = REPEAT_NGRAM) -> dict:
    def counts(sequence: list[str]) -> Counter[tuple[str, ...]]:
        return Counter(tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1))

    hypothesis_counts = counts(words)
    reference_counts = counts(reference_words)
    repeated = sum(count - 1 for count in hypothesis_counts.values() if count > 1)
    unexplained = sum(
        max(0, count - max(reference_counts.get(gram, 0), 1))
        for gram, count in hypothesis_counts.items()
        if count > 1
    )
    examples = [
        {
            "ngram": " ".join(gram),
            "count": count,
            "reference_count": reference_counts.get(gram, 0),
            "unexplained": max(0, count - max(reference_counts.get(gram, 0), 1)),
        }
        for gram, count in sorted(
            hypothesis_counts.items(),
            key=lambda item: (-(item[1] - max(reference_counts.get(item[0], 0), 1)), item[1]),
        )
        if count > 1
    ][:5]
    return {"repeated_ngrams": repeated, "unexplained_repeats": unexplained, "examples": examples}


def excess_words(hypothesis: str, reference: str) -> int:
    hypothesis_counts = Counter(hypothesis.split())
    reference_counts = Counter(reference.split())
    return sum(
        count - reference_counts[word]
        for word, count in hypothesis_counts.items()
        if count > reference_counts[word]
    )


def parse_time_file(path: Path) -> dict[str, float]:
    wall_s = 0.0
    peak_rss_mb = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("\tElapsed (wall clock) time"):
            parts = line.split("):", 1)[-1].strip().split(":")
            try:
                if len(parts) == 2:
                    wall_s = float(parts[0]) * 60.0 + float(parts[1])
                elif len(parts) == 3:
                    wall_s = float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            except ValueError:
                pass
        elif line.startswith("\tMaximum resident set size"):
            try:
                peak_rss_mb = float(line.split("):", 1)[-1].strip()) / 1024.0
            except ValueError:
                pass
    return {"wall_s": round(wall_s, 3), "peak_rss_mb": round(peak_rss_mb, 1)}


def wav_duration_s(path: Path | None) -> float | None:
    if not path or not path.is_file():
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return round(handle.getnframes() / rate, 3) if rate else None
    except (OSError, wave.Error):
        return None


def resolve(run_dir: Path, name: str) -> Path:
    for candidate in (run_dir / name, run_dir / "out" / name):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"{run_dir}: no {name}")


def clips_to_limit(clips: list[dict], utterances: int | None) -> list[dict]:
    return clips if utterances is None else clips[:utterances]


def block_ranges(start: float, end: float) -> list[tuple[float, float]]:
    ranges = []
    cursor = start
    while cursor < end - 1e-9:
        nxt = min(cursor + BLOCK_S, end)
        ranges.append((cursor, nxt))
        cursor = nxt
    return ranges


def block_rows(clips: list[dict], ranges: list[tuple[float, float]], tokens) -> list[dict]:
    """Score 30 s blocks.

    Reference clips and hypothesis words are both assigned by time midpoint, so
    a straddling utterance is counted whole on both sides. Segments without word
    timestamps fall back to the segment midpoint and are then block-granular.
    """
    rows = []
    for start, end in ranges:
        inside = [
            clip
            for clip in clips
            if start <= (float(clip["start_s"]) + float(clip["end_s"])) / 2.0 < end
        ]
        reference = normalize_text(" ".join(clip["sentence"] for clip in inside))
        hypothesis = " ".join(
            text for text, t_start, t_end in tokens if start <= (t_start + t_end) / 2.0 < end
        )
        score = jiwer.process_words(reference, hypothesis) if reference else None
        rows.append(
            {
                "window": f"{start:g}-{end:g}",
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "reference_words": len(reference.split()),
                "hypothesis_words": len(hypothesis.split()),
                "wer": round(score.wer, 6) if score else None,
                "errors": (score.substitutions + score.insertions + score.deletions) if score else 0,
                "reference": reference,
                "hypothesis": hypothesis,
            }
        )
    return rows


def structural_stats(segments: list[dict], audio_duration: float) -> dict:
    """Segment continuity and timestamp bounds, independent of the reference."""
    ordered = sorted(segments, key=lambda segment: float(segment.get("start", 0.0)))
    max_end = max((float(segment.get("end", 0.0)) for segment in ordered), default=0.0)
    gaps = [
        {
            "from_s": round(float(ordered[index]["end"]), 3),
            "to_s": round(float(ordered[index + 1]["start"]), 3),
            "gap_s": round(float(ordered[index + 1]["start"]) - float(ordered[index]["end"]), 3),
        }
        for index in range(len(ordered) - 1)
    ]
    worst = max(gaps, key=lambda gap: gap["gap_s"], default=None)
    return {
        "max_end_s": round(max_end, 3),
        "overrun_s": round(max_end - audio_duration, 3),
        "max_gap_s": round(worst["gap_s"], 3) if worst else 0.0,
        "max_gap": worst,
    }


def load_baseline(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != BASELINE_SCHEMA:
        raise SystemExit(f"{path}: not a {BASELINE_SCHEMA} baseline")
    return data


def file_sha256(path: Path | None) -> str | None:
    if not path or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def judge(metrics: dict, blocks: list[dict], baseline: dict | None, args) -> list[str]:
    failures: list[str] = []
    reference_words = metrics["reference_words"]
    if reference_words == 0:
        return ["reference has no words for this mode"]

    if baseline is None:
        if args.max_wer is None:
            failures.append("no baseline for this mode and no --max-wer given")
        elif metrics["wer"] > args.max_wer:
            failures.append(f"WER {pct(metrics['wer'])} > --max-wer {pct(args.max_wer)}")
    else:
        limit = baseline["metrics"]["wer"] + args.wer_tolerance
        if metrics["wer"] > limit:
            failures.append(
                f"WER {pct(metrics['wer'])} > baseline {pct(baseline['metrics']['wer'])}"
                f" +{pct(args.wer_tolerance)} tolerance"
            )

    baseline_duplicates = int((baseline or {}).get("metrics", {}).get("duplicate_segments", 0))
    if metrics["duplicate_segments"] > baseline_duplicates:
        failures.append(
            f"{metrics['duplicate_segments']} duplicate segments (baseline {baseline_duplicates})"
        )

    baseline_repeats = int((baseline or {}).get("metrics", {}).get("unexplained_repeats", 0))
    looped = [
        example
        for example in metrics["examples"]
        if example["count"] >= REPEAT_LOOP_OCCURRENCES and example["unexplained"] > 0
    ]
    if metrics["unexplained_repeats"] > baseline_repeats and looped:
        failures.append(
            f"{len(looped)} 8-grams repeat at least {REPEAT_LOOP_OCCURRENCES} times"
            f" without a matching reference (e.g. {looped[0]['ngram']!r} x{looped[0]['count']})"
        )

    if metrics["hypothesis_words"] > MAX_WORD_RATIO * reference_words:
        excess = excess_words(metrics["hypothesis"], metrics["reference"])
        if excess:
            failures.append(
                f"hypothesis {metrics['hypothesis_words']} words vs {reference_words} reference"
                f" ({metrics['word_ratio']:.2f}x > {MAX_WORD_RATIO:.2f}x) with {excess}"
                f" words beyond the reference word counts"
            )

    for block in blocks:
        if block["reference_words"] == 0:
            continue
        if block["hypothesis_words"] > MAX_WORD_RATIO * block["reference_words"]:
            excess = excess_words(block["hypothesis"], block["reference"])
            if excess:
                failures.append(
                    f"{block['window']} s has {block['hypothesis_words']} words vs"
                    f" {block['reference_words']} reference"
                    f" ({block['hypothesis_words'] / block['reference_words']:.2f}x >"
                    f" {MAX_WORD_RATIO:.2f}x) with {excess} words beyond the reference"
                    f" word counts"
                )

    structure = metrics["structure"]
    if structure["overrun_s"] > MAX_TIMESTAMP_OVERRUN_S:
        failures.append(
            f"last timestamp {structure['max_end_s']:.2f} s exceeds the"
            f" {metrics['audio_duration_s']:.2f} s audio by {structure['overrun_s']:.2f} s"
        )
    if structure["max_gap_s"] > MAX_SEGMENT_GAP_S:
        gap = structure["max_gap"]
        failures.append(
            f"segments skip {gap['gap_s']:.2f} s of audio"
            f" ({gap['from_s']:.2f} -> {gap['to_s']:.2f} s)"
        )

    baseline_blocks = (baseline or {}).get("blocks") or {}
    for block in blocks:
        entry = baseline_blocks.get(block["window"])
        if not entry or block["reference_words"] < COVERAGE_MIN_REFERENCE_WORDS:
            continue
        floor = max(1, math.ceil(0.5 * int(entry["hypothesis_words"])))
        if block["hypothesis_words"] < floor:
            failures.append(
                f"{block['window']} s emitted {block['hypothesis_words']} words against"
                f" {entry['hypothesis_words']} in the baseline"
                f" ({block['reference_words']} reference)"
            )
    return failures


def render(result: dict) -> str:
    m = result["metrics"]
    lines = [
        f"librispeech gate: {result['run_dir']} (mode {result['mode']},"
        f" {result['utterances']} utterances, {result['audio_duration_s']:.2f} s)",
        "metric              value",
        f"reference words     {m['reference_words']}",
        f"hypothesis words    {m['hypothesis_words']} ({m['word_ratio']:.2f}x)",
        f"WER                 {pct(m['wer'])}",
        f"CER                 {pct(m['cer'])}",
        f"sub / ins / del     {m['substitutions']} / {m['insertions']} / {m['deletions']}",
        f"duplicate segments  {m['duplicate_segments']}",
        f"repeated 8-grams    {m['repeated_ngrams']} ({m['unexplained_repeats']} unexplained)",
        f"wall / RTF          {result['run']['wall_s']:.1f} s /"
        f" {result['run']['rtf']:.3f}"
        if result["run"]["rtf"] is not None
        else f"wall                {result['run']['wall_s']:.1f} s / RTF -",
        f"peak RSS            {result['run']['peak_rss_mb']:.0f} MB",
        f"segment continuity  max gap {m['structure']['max_gap_s']:.2f} s,"
        f" last timestamp {m['structure']['max_end_s']:.2f} s",
        "",
        "block            refW  hypW     wer",
    ]
    for block in result["blocks"]:
        wer = pct(block["wer"]) if block["wer"] is not None else "-"
        lines.append(
            f"{block['window']:<14} {block['reference_words']:>5} {block['hypothesis_words']:>5}"
            f"  {wer:>7}"
        )
    if result["baseline"]:
        lines.append(f"baseline: {result['baseline']}")
    if result.get("baseline_note"):
        lines.append(f"baseline note: {result['baseline_note']}")
    if result["failures"]:
        lines.append("")
        lines.append("FAILURES")
        lines.extend(f"- {failure}" for failure in result["failures"])
        lines.append("")
        lines.append("offending blocks")
        for block in result["offending_blocks"]:
            lines.append(
                f"  {block['window']} s (WER {pct(block['wer'] or 0.0)},"
                f" {block['hypothesis_words']}/{block['reference_words']} words)"
            )
            lines.append(f"    REF: {block['reference'] or '<empty>'}")
            lines.append(f"    HYP: {block['hypothesis'] or '<empty>'}")
    if result.get("binary_sha256"):
        lines.append(f"binary sha256: {result['binary_sha256']}")
    lines.append(f"VERDICT: {result['verdict']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--reference", type=Path, help="space-joined reference; defaults to the boundaries")
    parser.add_argument("--mode", choices=("fast", "full"), default="full")
    parser.add_argument("--utterances", type=int, default=None, help="limit the reference to the first N utterances")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--max-wer", type=float, default=None)
    parser.add_argument("--wer-tolerance", type=float, default=0.0)
    parser.add_argument("--save-baseline", type=Path, default=None)
    parser.add_argument(
        "--baseline-note",
        default=None,
        help="why this run is trusted as the baseline (source commit, review, binary sha)",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"{args.run_dir}: not a directory")
    boundaries = load_boundaries(args.boundaries)
    utterances = args.utterances
    if utterances is None and args.mode == "fast":
        utterances = FAST_UTTERANCES
    clips = clips_to_limit(boundaries["clips"], utterances)

    duration = float(boundaries["total_duration_s"])
    if utterances is not None:
        last = clips[-1]
        duration = float(last["end_s"]) + float(boundaries.get("gap_s", 0.0))

    segments_path = resolve(args.run_dir, "merged.json")
    time_path = resolve(args.run_dir, "transcribe.time.txt")
    segments = load_segments(segments_path)
    metadata = {}
    for candidate in (args.run_dir / "run.json", args.run_dir / "out" / "run.json"):
        if candidate.is_file():
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
            break

    audio_duration = wav_duration_s(Path(metadata["wav"])) if metadata.get("wav") else None
    if audio_duration is None:
        audio_duration = duration
    elif abs(audio_duration - duration) > DURATION_TOLERANCE_S:
        raise SystemExit(
            f"run audio {audio_duration:.2f} s does not match mode {args.mode}"
            f" ({duration:.2f} s); wrong wav or wrong --utterances"
        )

    tokens, fallback = word_tokens(segments)
    tokens = [token for token in tokens if (token[1] + token[2]) / 2.0 < audio_duration]

    if args.reference:
        reference = normalize_text(args.reference.read_text(encoding="utf-8"))
    else:
        reference = normalize_text(" ".join(clip["sentence"] for clip in clips))
    hypothesis = " ".join(text for text, _, _ in tokens)
    score = jiwer.process_words(reference, hypothesis)
    repeat_stats = repeated_ngram_stats(hypothesis.split(), reference.split())
    duplicates = duplicate_segments(segments)

    structure = structural_stats(segments, audio_duration)
    metrics = {
        "audio_duration_s": round(audio_duration, 3),
        "structure": structure,
        "reference_words": len(reference.split()),
        "hypothesis_words": len(hypothesis.split()),
        "word_ratio": round(len(hypothesis.split()) / max(1, len(reference.split())), 4),
        "wer": round(score.wer, 6),
        "cer": round(jiwer.cer(reference, hypothesis), 6),
        "substitutions": score.substitutions,
        "insertions": score.insertions,
        "deletions": score.deletions,
        "duplicate_segments": len(duplicates),
        "duplicate_examples": duplicates[:5],
        "repeated_ngrams": repeat_stats["repeated_ngrams"],
        "unexplained_repeats": repeat_stats["unexplained_repeats"],
        "examples": repeat_stats["examples"],
        "reference": reference,
        "hypothesis": hypothesis,
    }

    blocks = block_rows(clips, block_ranges(0.0, duration), tokens)

    baseline_path = None
    if not args.no_baseline:
        candidate = args.baseline if args.baseline is not None else DEFAULT_BASELINE
        if candidate.is_file():
            baseline_path = candidate
        elif args.baseline is not None:
            raise SystemExit(f"{candidate}: baseline file not found")
    baseline_data = load_baseline(baseline_path)
    baseline = baseline_data.get("modes", {}).get(args.mode)
    if baseline is not None and int(baseline.get("utterances", -1)) != len(clips):
        raise SystemExit(
            f"baseline for mode {args.mode} has {baseline.get('utterances')} utterances,"
            f" this run uses {len(clips)}; regenerate with --save-baseline"
        )

    measurement = parse_time_file(time_path)
    failures = judge(metrics, blocks, baseline, args)

    offending: list[dict] = []
    if failures:
        failed_blocks = {
            block["window"]
            for block in blocks
            if any(block["window"] in failure for failure in failures)
        }
        ranked = sorted(
            (block for block in blocks if block["reference_words"] >= COVERAGE_MIN_REFERENCE_WORDS),
            key=lambda block: (-block["errors"], -block["wer"] if block["wer"] is not None else 0.0),
        )
        chosen = [block for block in blocks if block["window"] in failed_blocks] + ranked
        seen = set()
        for block in chosen:
            if block["window"] in seen:
                continue
            seen.add(block["window"])
            offending.append(block)
            if len(offending) >= args.show:
                break

    result = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(args.run_dir.resolve()),
        "segments_file": str(segments_path),
        "boundaries": str(args.boundaries),
        "mode": args.mode,
        "utterances": len(clips),
        "audio_duration_s": round(audio_duration, 3),
        "baseline": str(baseline_path) if baseline_path else None,
        "baseline_note": baseline.get("note") if baseline else None,
        "label": metadata.get("label"),
        "commit": metadata.get("commit"),
        "binary": metadata.get("binary"),
        "binary_sha256": file_sha256(Path(metadata["binary"]) if metadata.get("binary") else None),
        "run": {
            "segments": len(segments),
            "fallback_segments": fallback,
            "wall_s": measurement["wall_s"],
            "rtf": round(measurement["wall_s"] / audio_duration, 4) if audio_duration else None,
            "peak_rss_mb": measurement["peak_rss_mb"],
        },
        "metrics": metrics,
        "blocks": blocks,
        "failures": failures,
        "offending_blocks": offending,
        "verdict": "PASS" if not failures else "FAIL",
    }

    if args.save_baseline:
        if not args.baseline_note:
            raise SystemExit("--save-baseline requires --baseline-note (source hashes, review, binary sha)")
        saved = load_baseline(args.save_baseline) if args.save_baseline.is_file() else {
            "schema": BASELINE_SCHEMA,
            "generated_at": None,
            "modes": {},
        }
        saved["generated_at"] = result["generated_at"]
        saved["modes"][args.mode] = {
            "utterances": len(clips),
            "audio_duration_s": result["audio_duration_s"],
            "commit": result["commit"],
            "binary_sha256": result["binary_sha256"],
            "note": args.baseline_note,
            "metrics": {
                key: metrics[key]
                for key in (
                    "wer",
                    "cer",
                    "reference_words",
                    "hypothesis_words",
                    "word_ratio",
                    "duplicate_segments",
                    "repeated_ngrams",
                    "unexplained_repeats",
                )
            },
            "blocks": {
                block["window"]: {
                    "reference_words": block["reference_words"],
                    "hypothesis_words": block["hypothesis_words"],
                }
                for block in blocks
            },
        }
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.save_baseline.write_text(
            json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(render(result))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())