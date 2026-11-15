#!/usr/bin/env python3
"""Stage one LibriSpeech chapter as a single multi-window audio file.

Decodes every utterance of a chapter to 16 kHz mono signed 16-bit PCM,
concatenates them with a fixed silence gap, and writes the three files the
LibriSpeech gate needs:

* ``chapter.wav``    the concatenated audio (166 s for 2277/149896, ~5.5 min of
  engine windows, so the seek and the prompt history are exercised),
* ``reference.txt``  the space-joined transcript in utterance order,
* ``boundaries.json`` per-utterance spans and sentences, used for the per-30 s
  block coverage check and for the fast mode.

The layout is deterministic: utterance *i* is followed by ``--gap`` seconds of
digital silence, including after the last utterance. ``ffmpeg`` is the only
external requirement.

Usage:
    python3 prepare_chapter.py \
        --corpus-root /tmp/libri/LibriSpeech \
        --chapter dev-clean/2277/149896 \
        --out /tmp/libri

    # or from a tarball, extracting only the chapter
    python3 prepare_chapter.py --tar dev-clean.tar.gz \
        --chapter dev-clean/2277/149896 --out /tmp/libri
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
DEFAULT_GAP_S = 0.2


def extract_chapter(tar_path: Path, corpus_root: Path, chapter: str) -> None:
    """Extract only ``LibriSpeech/<chapter>`` from a LibriSpeech tarball."""
    prefix = f"LibriSpeech/{chapter.rstrip('/')}/"
    chapter_dir = corpus_root / chapter
    if chapter_dir.is_dir() and any(chapter_dir.glob("*.flac")):
        return
    chapter_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name.startswith(prefix)]
        if not members:
            raise SystemExit(f"{tar_path}: no member under {prefix}")
        archive.extractall(corpus_root.parent, members=members, filter="data")


def decode(path: Path) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def read_transcript(chapter_dir: Path) -> list[tuple[str, str]]:
    transcripts = sorted(chapter_dir.glob("*.trans.txt"))
    if len(transcripts) != 1:
        raise SystemExit(f"{chapter_dir}: expected one *.trans.txt, found {len(transcripts)}")
    rows = []
    for line in transcripts[0].read_text(encoding="utf-8-sig").splitlines():
        name, _, sentence = line.strip().partition(" ")
        if name and sentence.strip():
            rows.append((name, sentence.strip()))
    if not rows:
        raise SystemExit(f"{transcripts[0]}: no utterance lines")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus-root", type=Path, help="extracted LibriSpeech root")
    source.add_argument("--tar", type=Path, help="LibriSpeech tarball to read the chapter from")
    parser.add_argument("--chapter", required=True, help="e.g. dev-clean/2277/149896")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP_S, help="silence between utterances")
    args = parser.parse_args()

    corpus_root = args.corpus_root or (args.out / "LibriSpeech")
    if args.tar:
        extract_chapter(args.tar, corpus_root, args.chapter)

    chapter_dir = corpus_root / args.chapter
    if not chapter_dir.is_dir():
        raise SystemExit(f"{chapter_dir}: chapter directory not found")

    gap_samples = round(args.gap * SAMPLE_RATE)
    silence = b"\x00" * (2 * gap_samples)
    args.out.mkdir(parents=True, exist_ok=True)

    references: list[str] = []
    boundaries: list[dict] = []
    cursor = 0
    missing = 0

    with wave.open(str(args.out / "chapter.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)

        for utterance, sentence in read_transcript(chapter_dir):
            source = chapter_dir / f"{utterance}.flac"
            if not source.is_file():
                source = chapter_dir / f"{utterance}.wav"
            if not source.is_file():
                missing += 1
                continue
            frames = decode(source)
            if len(frames) % 2:
                raise SystemExit(f"{source}: odd PCM byte count")
            samples = len(frames) // 2
            output.writeframesraw(frames)
            output.writeframesraw(silence)

            boundaries.append(
                {
                    "index": len(boundaries),
                    "utterance": utterance,
                    "sentence": sentence,
                    "start_sample": cursor,
                    "end_sample": cursor + samples,
                    "start_s": round(cursor / SAMPLE_RATE, 6),
                    "end_s": round((cursor + samples) / SAMPLE_RATE, 6),
                    "samples": samples,
                }
            )
            references.append(sentence)
            cursor += samples + gap_samples

    if not boundaries:
        raise SystemExit("no utterances decoded")

    (args.out / "reference.txt").write_text(" ".join(references) + "\n", encoding="utf-8")
    (args.out / "boundaries.json").write_text(
        json.dumps(
            {
                "sample_rate": SAMPLE_RATE,
                "gap_s": args.gap,
                "total_samples": cursor,
                "total_duration_s": round(cursor / SAMPLE_RATE, 6),
                "clips": boundaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    duration = cursor / SAMPLE_RATE
    print(
        f"{len(boundaries)} utterances, {duration:.2f} s ({duration / 60.0:.2f} min),"
        f" {missing} transcript entries without audio, gap {args.gap:g} s"
    )


if __name__ == "__main__":
    main()