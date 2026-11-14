#!/usr/bin/env python3
"""Score per-clip greedy and beam-search hypotheses against the curated reference.

Companion to ``clip_eval`` (benchmarks/vot-solo/clip_eval). Normalization matches
``evaluate.py``: strip bracketed annotations, lowercase, replace punctuation with
spaces, collapse whitespace. WER comes from ``jiwer``.

The script also flags clips whose hypothesis is a word-prefix of the reference,
which is the signature of a decoder that stops before the end of the utterance.
"""

from __future__ import annotations

import argparse
import json
import re
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


def load_results(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON array")
    return {int(entry["index"]): entry for entry in data}


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def is_prefix(short: str, long: str) -> bool:
    short_words = normalize_text(short).split()
    long_words = normalize_text(long).split()
    return bool(short_words) and len(short_words) < len(long_words) and long_words[: len(short_words)] == short_words


def trailing_repeats(text: str) -> tuple[str, int]:
    """Longest trailing phrase repeated at least twice (3x for single words)."""
    words = normalize_text(text).split()
    best = ("", 0)
    for period in range(1, min(4, len(words) // 2 + 1)):
        block = words[-period:]
        count = 0
        for end in range(len(words), period - 1, -period):
            if words[end - period : end] != block:
                break
            count += 1
        threshold = 3 if period == 1 else 2
        if count >= threshold and count * period > best[1] * len(best[0].split() or [""]):
            best = (" ".join(block), count)
    return best


def classify(ref: str, greedy: str, beam: str) -> str:
    ref_words = word_count(ref)
    beam_words = word_count(beam)
    greedy_words = word_count(greedy)
    word, repeats = trailing_repeats(beam)
    if is_prefix(beam, ref) and not is_prefix(greedy, ref):
        return f"beam truncates: {beam_words}/{ref_words} ref words"
    if is_prefix(beam, ref):
        return f"both truncate: greedy {greedy_words}, beam {beam_words}/{ref_words} ref words"
    if is_prefix(greedy, ref):
        return f"greedy truncates: {greedy_words}/{ref_words} ref words, beam complete"
    if beam_words == 0 and ref_words:
        return "beam empty"
    if beam_words == 0:
        return "empty reference and hypothesis"
    if ref_words and beam_words <= 0.6 * ref_words:
        return f"beam short: {beam_words}/{ref_words} ref words"
    if repeats >= 3:
        return f"beam repeats '{word}' {repeats} times"
    greedy_wer = jiwer.wer(normalize_text(ref), normalize_text(greedy))
    beam_wer = jiwer.wer(normalize_text(ref), normalize_text(beam))
    if beam_wer + 0.05 < greedy_wer:
        return f"beam improves ({greedy_wer:.2f} -> {beam_wer:.2f})"
    if beam_wer > greedy_wer + 0.05:
        return f"beam worsens ({greedy_wer:.2f} -> {beam_wer:.2f})"
    return "same"


def score(entries: dict[int, dict], reference_by_index: dict[int, str]) -> dict:
    per_clip = []
    for index in sorted(entries):
        reference = reference_by_index[index]
        hypothesis = entries[index].get("text", "")
        per_clip.append(
            {
                "index": index,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": jiwer.wer(normalize_text(reference), normalize_text(hypothesis)),
                "duration_s": entries[index].get("duration_s", 0.0),
                "elapsed_s": entries[index].get("elapsed_s", 0.0),
                "ref_words": word_count(reference),
            }
        )
    references = [normalize_text(reference_by_index[index]) for index in sorted(entries)]
    hypotheses = [normalize_text(entries[index].get("text", "")) for index in sorted(entries)]
    return {
        "per_clip": per_clip,
        "aggregate_wer": jiwer.wer(references, hypotheses),
        "total_duration_s": sum(clip["duration_s"] for clip in per_clip),
        "total_elapsed_s": sum(clip["elapsed_s"] for clip in per_clip),
        "mean_clip_wer": sum(clip["wer"] for clip in per_clip) / max(len(per_clip), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", required=True, type=Path)
    parser.add_argument("--beam", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="write a markdown report here")
    parser.add_argument("--json-out", type=Path, help="write raw metrics here")
    parser.add_argument("--title", default="Per-clip beam search comparison")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--command-greedy", default="")
    parser.add_argument("--command-beam", default="")
    parser.add_argument("--wall-greedy", default="")
    parser.add_argument("--wall-beam", default="")
    args = parser.parse_args()

    greedy_entries = load_results(args.greedy)
    beam_entries = load_results(args.beam)
    if set(greedy_entries) != set(beam_entries):
        raise SystemExit("greedy and beam results cover different clip indices")

    reference_by_index = {index: beam_entries[index]["reference"] for index in beam_entries}
    greedy = score(greedy_entries, reference_by_index)
    beam = score(beam_entries, reference_by_index)

    greedy_by_index = {clip["index"]: clip for clip in greedy["per_clip"]}
    beam_by_index = {clip["index"]: clip for clip in beam["per_clip"]}

    rows = []
    truncated = []
    for index in sorted(reference_by_index):
        greedy_clip = greedy_by_index[index]
        beam_clip = beam_by_index[index]
        note = classify(
            greedy_clip["reference"],
            greedy_clip["hypothesis"],
            beam_clip["hypothesis"],
        )
        rows.append(
            {
                "index": index,
                "duration_s": greedy_clip["duration_s"],
                "ref_words": greedy_clip["ref_words"],
                "greedy_wer": greedy_clip["wer"],
                "beam_wer": beam_clip["wer"],
                "note": note,
            }
        )
        if "truncat" in note or "empty" in note or "short" in note:
            truncated.append(
                {
                    "index": index,
                    "reference": greedy_clip["reference"],
                    "greedy": greedy_clip["hypothesis"],
                    "beam": beam_clip["hypothesis"],
                    "note": note,
                }
            )

    metrics = {
        "greedy": greedy,
        "beam": beam,
        "rows": rows,
        "truncated": truncated,
        "beam_size": args.beam_size,
    }

    print(
        f"aggregate WER: greedy {greedy['aggregate_wer']:.4f} "
        f"({greedy['total_duration_s']:.1f}s audio, {greedy['total_elapsed_s']:.1f}s compute)"
    )
    print(
        f"aggregate WER: beam {beam['aggregate_wer']:.4f} "
        f"({beam['total_duration_s']:.1f}s audio, {beam['total_elapsed_s']:.1f}s compute)"
    )
    for row in rows:
        print(
            f"{row['index']:>4} {row['duration_s']:>6.2f}s {row['ref_words']:>3}w "
            f"greedy {row['greedy_wer']:.3f} beam {row['beam_wer']:.3f}  {row['note']}"
        )
    if truncated:
        print("\ntruncated clips:")
        for clip in truncated:
            print(f"  {clip['index']}: {clip['note']}")
            print(f"    ref : {clip['reference']}")
            print(f"    grdy: {clip['greedy']}")
            print(f"    beam: {clip['beam']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if args.out:
        lines = [f"# {args.title}", ""]
        lines.append(
            "Model: `ggml-large-v3-turbo.bin` (F16). Language `id`, 4 threads, "
            "temperature 0, timestamps enabled. Every clip is decoded from its MP3 "
            "with the repository decoder and transcribed independently; the model is "
            "loaded once for each run."
        )
        lines.append("")
        lines.append("## Aggregate WER")
        lines.append("")
        lines.append(
            "| mode | beam size | clips | audio (s) | compute (s) | RTF | aggregate WER | mean clip WER |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, result, size in (
            ("greedy", greedy, 1),
            ("beam", beam, args.beam_size),
        ):
            rtf = result["total_elapsed_s"] / max(result["total_duration_s"], 1e-9)
            lines.append(
                f"| {name} | {size} | {len(result['per_clip'])} | "
                f"{result['total_duration_s']:.1f} | {result['total_elapsed_s']:.1f} | "
                f"{rtf:.2f} | {result['aggregate_wer']:.4f} | {result['mean_clip_wer']:.4f} |"
            )
        lines.append("")
        if args.command_greedy or args.command_beam:
            lines.append("## Commands")
            lines.append("")
            if args.command_greedy:
                lines.append(f"- greedy: `{args.command_greedy}`")
            if args.command_beam:
                lines.append(f"- beam {args.beam_size}: `{args.command_beam}`")
            if args.wall_greedy:
                lines.append(f"- greedy wall time: {args.wall_greedy}")
            if args.wall_beam:
                lines.append(f"- beam {args.beam_size} wall time: {args.wall_beam}")
            lines.append("")
        lines.append("## Per-clip results")
        lines.append("")
        lines.append("| index | duration (s) | ref words | greedy WER | beam WER | note |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for row in rows:
            lines.append(
                f"| {row['index']} | {row['duration_s']:.2f} | {row['ref_words']} | "
                f"{row['greedy_wer']:.3f} | {row['beam_wer']:.3f} | {row['note']} |"
            )
        lines.append("")
        if truncated:
            lines.append("## Remaining truncations")
            lines.append("")
            for clip in truncated:
                lines.append(f"### Clip {clip['index']} — {clip['note']}")
                lines.append("")
                lines.append(f"- reference: {clip['reference']}")
                lines.append(f"- greedy: {clip['greedy']}")
                lines.append(f"- beam: {clip['beam']}")
                lines.append("")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()