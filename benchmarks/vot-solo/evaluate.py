#!/usr/bin/env python3
"""Evaluate a merged-clip whisper.rs transcript against the curated reference."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import wave
from array import array
from collections import Counter
from pathlib import Path

import jiwer

ANNOTATION = re.compile(r"[\[\(][^\]\)]*[\]\)]")
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE = re.compile(r"\s+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = ANNOTATION.sub(" ", text)
    text = text.lower().replace("\ufeff", " ")
    text = PUNCTUATION.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def load_boundaries(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("expected the CLI JSON output to be a list of segments")
    return [
        segment
        for segment in data
        if isinstance(segment, dict) and str(segment.get("text", "")).strip()
    ]


def parse_time_file(path: Path) -> dict[str, float]:
    wall_s = 0.0
    peak_rss_mb = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("\tElapsed (wall clock) time"):
            value = line.split("):", 1)[-1].strip()
            parts = value.split(":")
            try:
                if len(parts) == 2:
                    wall_s = float(parts[0]) * 60.0 + float(parts[1])
                elif len(parts) == 3:
                    wall_s = (
                        float(parts[0]) * 3600.0
                        + float(parts[1]) * 60.0
                        + float(parts[2])
                    )
            except ValueError:
                pass
        elif line.startswith("\tMaximum resident set size"):
            try:
                peak_rss_mb = float(line.split("):", 1)[-1].strip()) / 1024.0
            except ValueError:
                pass
    return {"wall_s": round(wall_s, 3), "peak_rss_mb": round(peak_rss_mb, 1)}


def clip_rms(audio: array, boundaries: list[dict]) -> list[dict]:
    stats = []
    for clip in boundaries:
        start = clip["start_sample"]
        end = clip["end_sample"]
        if end <= start:
            stats.append({"rms": 0.0, "peak": 0})
            continue
        window = audio[start:end]
        square_sum = sum(float(sample) * float(sample) for sample in window)
        rms = math.sqrt(square_sum / len(window)) / 32768.0
        peak = max(abs(sample) for sample in window)
        stats.append({"rms": round(rms, 6), "peak": int(peak)})
    return stats


def alignment_patterns(output) -> dict:
    references = output.references[0]
    hypotheses = output.hypotheses[0]
    substitutions: Counter[tuple[str, str]] = Counter()
    insertions: Counter[str] = Counter()
    deletions: Counter[str] = Counter()

    for chunk in output.alignments[0]:
        if chunk.type == "substitute":
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                substitutions[
                    (references[chunk.ref_start_idx + offset], hypotheses[chunk.hyp_start_idx + offset])
                ] += 1
        elif chunk.type == "insert":
            for index in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                insertions[hypotheses[index]] += 1
        elif chunk.type == "delete":
            for index in range(chunk.ref_start_idx, chunk.ref_end_idx):
                deletions[references[index]] += 1

    return {
        "top_substitutions": [
            {"reference": ref, "hypothesis": hyp, "count": count}
            for (ref, hyp), count in substitutions.most_common(30)
        ],
        "top_insertions": [
            {"word": word, "count": count} for word, count in insertions.most_common(20)
        ],
        "top_deletions": [
            {"word": word, "count": count} for word, count in deletions.most_common(20)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--time-file", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--tool-version", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = load_boundaries(args.boundaries)
    clips = boundaries["clips"]
    segments = load_segments(args.segments)

    with wave.open(str(args.audio), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit("expected 16-bit mono WAV")
        pcm = array("h")
        pcm.frombytes(handle.readframes(handle.getnframes()))
    if sys_byteorder_is_big():
        pcm.byteswap()
    audio_stats = clip_rms(pcm, clips)

    reference_full = normalize_text("\n".join(clip["sentence"] for clip in clips))
    hypothesis_full = normalize_text("\n".join(str(segment.get("text", "")) for segment in segments))
    overall = jiwer.process_words(reference_full, hypothesis_full)

    starts = [clip["start_s"] for clip in clips]
    per_clip_hyp: list[list[str]] = [[] for _ in clips]
    per_clip_segments: list[list[dict]] = [[] for _ in clips]
    unassigned: list[dict] = []

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        midpoint = (start + end) / 2.0
        index = bisect.bisect_right(starts, midpoint) - 1
        if index < 0 or index >= len(clips) or midpoint >= clips[index]["end_s"]:
            unassigned.append(segment)
            continue
        per_clip_hyp[index].append(str(segment.get("text", "")).strip())
        per_clip_segments[index].append(segment)

    per_clip = []
    total_error_mass = 0
    for index, clip in enumerate(clips):
        reference = normalize_text(clip["sentence"])
        hypothesis = normalize_text(" ".join(per_clip_hyp[index]))
        if reference:
            score = jiwer.process_words(reference, hypothesis)
            wer = score.wer
            errors = score.substitutions + score.insertions + score.deletions
            total_error_mass += errors
        else:
            score = None
            errors = 0
            wer = 0.0

        assigned = per_clip_segments[index]
        total_duration = sum(
            max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
            for segment in assigned
        )
        avg_logprob = (
            sum(
                float(segment.get("avg_logprob", 0.0))
                * max(0.001, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
                for segment in assigned
            )
            / total_duration
            if total_duration > 0
            else None
        )
        avg_no_speech = (
            sum(
                float(segment.get("no_speech_prob", 0.0))
                * max(0.001, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
                for segment in assigned
            )
            / total_duration
            if total_duration > 0
            else None
        )

        per_clip.append(
            {
                "index": index,
                "path": clip["path"],
                "start_s": clip["start_s"],
                "end_s": clip["end_s"],
                "duration_s": round(clip["end_s"] - clip["start_s"], 3),
                "reference": clip["sentence"],
                "hypothesis": " ".join(per_clip_hyp[index]),
                "wer": round(wer, 6),
                "cer": round(jiwer.cer(reference, hypothesis), 6) if reference else 0.0,
                "errors": errors,
                "reference_words": len(reference.split()),
                "hypothesis_words": len(hypothesis.split()),
                "rms": audio_stats[index]["rms"],
                "peak": audio_stats[index]["peak"],
                "avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
                "avg_no_speech_prob": round(avg_no_speech, 4)
                if avg_no_speech is not None
                else None,
                "segments_assigned": len(assigned),
                "error_type": (
                    "missing"
                    if reference and not hypothesis
                    else "insertion"
                    if hypothesis and len(hypothesis.split()) > 2 * max(1, len(reference.split()))
                    else "mixed"
                    if errors
                    else "clean"
                ),
            }
        )

    worst = sorted(
        (clip for clip in per_clip if clip["reference_words"] >= 3 and clip["errors"] > 0),
        key=lambda clip: (-clip["wer"], -clip["duration_s"]),
    )[:30]
    worst_error_share = (
        sum(clip["errors"] for clip in worst) / total_error_mass if total_error_mass else 0.0
    )

    short = [clip for clip in per_clip if clip["reference_words"] >= 3 and clip["duration_s"] < 3.0]
    long = [clip for clip in per_clip if clip["reference_words"] >= 3 and clip["duration_s"] >= 3.0]
    low_confidence = [
        clip
        for clip in per_clip
        if clip["reference_words"] >= 3
        and clip["avg_logprob"] is not None
        and clip["avg_logprob"] < -1.0
    ]

    measurement = parse_time_file(args.time_file) if args.time_file else {"wall_s": 0.0, "peak_rss_mb": 0.0}
    duration_s = boundaries["total_duration_s"]
    result = {
        "model": args.model,
        "quantization": args.quantization,
        "tool_version": args.tool_version,
        "clips": len(clips),
        "audio_duration_s": duration_s,
        "wall_s": measurement["wall_s"],
        "rtf": round(measurement["wall_s"] / duration_s, 4) if duration_s else None,
        "peak_rss_mb": measurement["peak_rss_mb"],
        "reference_words": len(reference_full.split()),
        "hypothesis_words": len(hypothesis_full.split()),
        "wer": round(overall.wer, 6),
        "cer": round(jiwer.cer(reference_full, hypothesis_full), 6),
        "hits": overall.hits,
        "substitutions": overall.substitutions,
        "insertions": overall.insertions,
        "deletions": overall.deletions,
        "worst_error_share": round(worst_error_share, 4),
        "unassigned_segments": len(unassigned),
        "short_clips_median_wer": round(median([clip["wer"] for clip in short]), 6)
        if short
        else None,
        "long_clips_median_wer": round(median([clip["wer"] for clip in long]), 6)
        if long
        else None,
        "low_confidence_clips": len(low_confidence),
        "low_confidence_median_wer": round(median([clip["wer"] for clip in low_confidence]), 6)
        if low_confidence
        else None,
        **alignment_patterns(overall),
        "worst_clips": worst,
    }

    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "per_clip.json").write_text(
        json.dumps(per_clip, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(render_report(result), encoding="utf-8")

    summary_path = os_environ("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(render_report(result) + "\n")


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def sys_byteorder_is_big() -> bool:
    import sys

    return sys.byteorder == "big"


def os_environ(name: str) -> str | None:
    import os

    return os.environ.get(name)


def render_report(result: dict) -> str:
    lines = [
        "# Solo whisper.rs evaluation",
        "",
        f"Model: `{result['model']}` ({result['quantization']}), "
        f"engine commit `{result['tool_version'] or 'unknown'}`.",
        f"Audio: {result['clips']} clips, {result['audio_duration_s'] / 60:.2f} min. "
        f"Wall: {result['wall_s']:.1f} s, RTF {result['rtf']:.3f}, "
        f"peak RSS {result['peak_rss_mb']:.0f} MB.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| WER | {result['wer']:.2%} |",
        f"| CER | {result['cer']:.2%} |",
        f"| reference words | {result['reference_words']} |",
        f"| hypothesis words | {result['hypothesis_words']} |",
        f"| substitutions | {result['substitutions']} |",
        f"| insertions | {result['insertions']} |",
        f"| deletions | {result['deletions']} |",
        f"| worst-30 share of errors | {result['worst_error_share']:.1%} |",
        f"| median WER, clips < 3 s | {fmt_pct(result['short_clips_median_wer'])} |",
        f"| median WER, clips >= 3 s | {fmt_pct(result['long_clips_median_wer'])} |",
        f"| low-confidence clips | {result['low_confidence_clips']} |",
        f"| median WER, low-confidence | {fmt_pct(result['low_confidence_median_wer'])} |",
        "",
        "## Worst clips",
        "",
        "| # | time | dur | RMS | logprob | no-speech | WER | ref words | hyp words | type |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for clip in result["worst_clips"]:
        lines.append(
            "| {index} | {start:.2f}-{end:.2f} | {dur:.2f} | {rms:.4f} | {lp} | {ns} | "
            "{wer:.2%} | {rw} | {hw} | {etype} |".format(
                index=clip["index"],
                start=clip["start_s"],
                end=clip["end_s"],
                dur=clip["duration_s"],
                rms=clip["rms"],
                lp=f"{clip['avg_logprob']:.2f}" if clip["avg_logprob"] is not None else "-",
                ns=f"{clip['avg_no_speech_prob']:.3f}"
                if clip["avg_no_speech_prob"] is not None
                else "-",
                wer=clip["wer"],
                rw=clip["reference_words"],
                hw=clip["hypothesis_words"],
                etype=clip["error_type"],
            )
        )

    lines.extend(["", "## Reference / hypothesis for worst clips", ""])
    for clip in result["worst_clips"][:15]:
        lines.extend(
            [
                f"### clip {clip['index']} ({clip['start_s']:.2f}-{clip['end_s']:.2f} s, WER {clip['wer']:.2%})",
                "",
                f"- REF: {clip['reference']}",
                f"- HYP: {clip['hypothesis'] or '<empty>'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Frequent substitutions",
            "",
            "| reference | hypothesis | count |",
            "| --- | --- | ---: |",
        ]
    )
    for item in result["top_substitutions"][:20]:
        lines.append(f"| {item['reference']} | {item['hypothesis']} | {item['count']} |")

    lines.extend(["", "## Frequent insertions", "", "| word | count |", "| --- | ---: |"])
    for item in result["top_insertions"][:15]:
        lines.append(f"| {item['word']} | {item['count']} |")

    lines.extend(["", "## Frequent deletions", "", "| word | count |", "| --- | ---: |"])
    for item in result["top_deletions"][:15]:
        lines.append(f"| {item['word']} | {item['count']} |")

    lines.extend(
        [
            "",
            "## Method",
            "",
            "Clips were decoded to 16 kHz mono PCM and concatenated in metadata order with no",
            "inserted silence. ASR segments are assigned to a clip by their midpoint; a segment",
            "that crosses a boundary is counted only for the clip containing its midpoint, so",
            "per-clip WER is diagnostic rather than a substitute for the overall score.",
            "Hypothesis bracketed annotations such as `[SPEAKER_TURN]` and `(music)` are removed",
            "before scoring; punctuation is normalized to spaces.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt_pct(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "-"


if __name__ == "__main__":
    main()