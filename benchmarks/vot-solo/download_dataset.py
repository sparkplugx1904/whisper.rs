#!/usr/bin/env python3
"""Download the curated clips and metadata for one VOT recording."""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "Sparkplugx1904/voiceoftrisma-voice-datasets"
RECORDING = "VOT-Denpasar_18-12-25-0"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{RECORDING}"


def download(url: str, path: Path, retries: int = 5) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
            temporary.replace(path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clips", type=int, default=12)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = args.out / "metadata.tsv"
    download(f"{BASE}/metadata.tsv", metadata)

    names: list[str] = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            if row and row[0].strip():
                names.append(row[0].strip())

    def fetch(name: str) -> tuple[str, int]:
        path = args.out / "clips" / name
        download(f"{BASE}/clips/{urllib.parse.quote(name)}", path)
        return name, path.stat().st_size

    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.clips) as pool:
        for index, (_, size) in enumerate(pool.map(fetch, names), start=1):
            total += size
            if index % 50 == 0:
                print(f"downloaded {index}/{len(names)} clips, {total / 1e6:.1f} MB")

    print(f"downloaded {len(names)} clips, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()