#!/usr/bin/env python3
"""Decode curated clips and concatenate them without gaps.

The metadata rows define the order. Every clip is decoded separately to
16 kHz mono signed 16-bit PCM, then the raw frames are appended directly; no
silence or resampling gap is inserted between clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import subprocess


def decode_clip(path: Path) -> bytes:
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
            "16000",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--boundaries", required=True, type=Path)
    args = parser.parse_args()

    rows: list[tuple[str, str]] = []
    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sentence_index = header.index("sentence") if "sentence" in header else 1
        for row in reader:
            if len(row) <= sentence_index:
                continue
            name = row[0].strip()
            sentence = row[sentence_index].strip()
            if name:
                rows.append((name, sentence))

    if not rows:
        raise SystemExit("metadata has no rows")

    boundaries: list[dict] = []
    sample_cursor = 0
    total_bytes = 0

    with wave.open(str(args.wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)

        for index, (name, sentence) in enumerate(rows):
            clip_path = args.clips / name
            if not clip_path.exists():
                raise SystemExit(f"missing clip: {clip_path}")
            frames = decode_clip(clip_path)
            if len(frames) % 2:
                raise SystemExit(f"odd PCM byte count for {clip_path}")
            samples = len(frames) // 2
            if samples == 0:
                raise SystemExit(f"empty audio after decode: {clip_path}")

            output.writeframesraw(frames)
            total_bytes += len(frames)
            boundaries.append(
                {
                    "index": index,
                    "path": name,
                    "sentence": sentence,
                    "start_sample": sample_cursor,
                    "end_sample": sample_cursor + samples,
                    "start_s": round(sample_cursor / 16_000.0, 6),
                    "end_s": round((sample_cursor + samples) / 16_000.0, 6),
                    "samples": samples,
                }
            )
            sample_cursor += samples

    args.boundaries.write_text(
        json.dumps(
            {
                "sample_rate": 16_000,
                "total_samples": sample_cursor,
                "total_duration_s": round(sample_cursor / 16_000.0, 6),
                "clips": boundaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"merged {len(rows)} clips, {sample_cursor} samples, "
        f"{sample_cursor / 16_000.0 / 60.0:.2f} min, {total_bytes / 1e6:.1f} MB PCM"
    )


if __name__ == "__main__":
    main()