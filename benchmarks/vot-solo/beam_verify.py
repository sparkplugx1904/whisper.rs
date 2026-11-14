#!/usr/bin/env python3
"""Compare beam-search transcripts from whisper.rs and upstream whisper.cpp.

For each selected VOT clip the script decodes the MP3 to 16 kHz mono PCM WAV,
transcribes it once with each CLI, and scores the normalized text against the
dataset reference with WER. The two engines are always scored separately: the
hypotheses are never merged, so a regression in one cannot hide behind the
other.

Example:

    python3 benchmarks/vot-solo/beam_verify.py \
        --model /tmp/vot-dbg/models/model.bin

Clips default to the station-identification intro (where a beam used to stop
after the jingle) plus a spread of ordinary utterances.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import jiwer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/tmp/vot-dbg")
DEFAULT_MODEL = DEFAULT_DATA / "models" / "model.bin"
DEFAULT_UPSTREAM = Path("/tmp/whisper.cpp-upstream/build/bin/whisper-cli")
DEFAULT_CLIPS = [
    "01-01",
    "01-02",
    "01-03",
    "01-04",
    "01-07",
    "01-08",
    "03-01",
    "05-01",
    "06-01",
    "07-01",
]

# The pure-Rust engine peaks near 4.8 GB RSS with the large-v3-turbo F16 model;
# upstream stays around 1.7 GB. Running two engines at once in this container
# triggers a supervisor SIGTERM, so each contestant is gated on free memory.
REQUIRED_HEADROOM_GB = 4.4
UPSTREAM_HEADROOM_GB = 2.2
ENGINE_LOCK = Path("/tmp/vot-dbg/engine.lock")

SEGMENT_RE = re.compile(r"^\[[0-9:.\s>-]+\]\s*(.*)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ANNOTATION_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase and reduce punctuation to spaces, matching the VOT scoring."""
    text = unicodedata.normalize("NFKC", text)
    text = ANNOTATION_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text.lower())
    return " ".join(text.split())


def parse_transcript(stdout: str) -> str:
    """Keep only segment payloads; the CLI also prints diagnostics on stdout."""
    parts: list[str] = []
    for raw_line in stdout.splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        match = SEGMENT_RE.match(line)
        if match:
            payload = match.group(1).strip()
            if payload:
                parts.append(payload)
    return " ".join(parts)


def clip_stem(clip: str) -> str:
    return clip if not clip.endswith(".mp3") else clip[:-4]


def clip_path(clips_dir: Path, stem: str) -> Path:
    return clips_dir / f"VOT-Denpasar_18-12-25-0-{stem}.mp3"


def load_references(metadata: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            name = Path(row[0].strip()).stem
            sentence = row[1] if len(row) > 1 else ""
            references[name] = sentence
    return references


def decode(mp3: Path, wav: Path) -> None:
    if wav.exists() and wav.stat().st_size > 44:
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        check=True,
    )


def run_cli(
    command: list[str], label: str, headroom_gb: float, retries: int = 3
) -> tuple[str, str]:
    last_error = ""
    wait_for_headroom(headroom_gb, label)
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=3600
        )
        elapsed = time.perf_counter() - started
        if completed.returncode == 0:
            print(
                f"    {label:<9} {elapsed:7.2f} s  stdout={len(completed.stdout)} B"
                f" stderr={len(completed.stderr)} B",
                file=sys.stderr,
            )
            return parse_transcript(completed.stdout), completed.stdout
        last_error = completed.stderr.strip()
        if completed.returncode != -15:
            raise RuntimeError(f"{label} exited {completed.returncode}: {last_error}")
        print(
            f"    {label:<9} killed by SIGTERM after {elapsed:.1f} s "
            f"(attempt {attempt}/{retries}); waiting for memory headroom",
            file=sys.stderr,
        )
        wait_for_headroom(headroom_gb, label)
    raise RuntimeError(f"{label} kept getting killed by SIGTERM: {last_error}")


def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("inf")


def heavy_engines() -> list[tuple[int, float, str]]:
    """Processes that compete for the model memory budget."""
    found: list[tuple[int, float, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except OSError:
            continue
        if "whisper" not in cmdline and "clip-eval" not in cmdline:
            continue
        try:
            rss_kb = 0
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
        except OSError:
            continue
        if rss_kb > 1_200_000:
            found.append((pid, rss_kb / (1024 * 1024), cmdline[:120]))
    return found


def wait_for_headroom(required_gb: float, label: str, timeout_s: float = 14400.0) -> None:
    """Block until no other engine holds memory and enough RAM is free.

    The container supervisor sends SIGTERM when two multi-gigabyte engines
    overlap, so contestants are serialized across processes, not just within
    this script.
    """
    deadline = time.monotonic() + timeout_s
    announced = False
    while True:
        available = mem_available_gb()
        competitors = heavy_engines()
        locked = ENGINE_LOCK.exists()
        if available >= required_gb and not competitors and not locked:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"timed out waiting for memory headroom before {label}: "
                f"{available:.2f} GB available, competitors={competitors}"
            )
        if not announced:
            detail = ", ".join(f"pid {pid} ({rss:.1f} GB)" for pid, rss, _ in competitors)
            print(
                f"    waiting for memory: {available:.2f} GB available"
                + (f", competing: {detail}" if detail else "")
                + (", engine.lock held" if locked else ""),
                file=sys.stderr,
            )
            announced = True
        time.sleep(5.0)


def safe_wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    if not hypothesis.strip():
        return 1.0
    return float(jiwer.wer(reference, hypothesis))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--ours",
        type=Path,
        default=REPO_ROOT / "target" / "release" / "whisper-cli",
    )
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--clips",
        default=",".join(DEFAULT_CLIPS),
        help="comma separated clip stems, e.g. 01-01,03-01",
    )
    parser.add_argument("--language", default="id")
    parser.add_argument("--threads", default="4")
    parser.add_argument("--beam-size", default="5")
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--work", type=Path, default=None)
    args = parser.parse_args()

    clips_dir = args.data / "clips"
    metadata = args.data / "metadata.tsv"
    work = args.work or (args.data / "beam-verify")

    for path, what in (
        (args.model, "model"),
        (args.ours, "whisper.rs CLI"),
        (args.upstream, "upstream CLI"),
        (metadata, "metadata.tsv"),
    ):
        if not path.exists():
            print(f"error: {what} not found: {path}", file=sys.stderr)
            return 2
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 2

    work.mkdir(parents=True, exist_ok=True)
    references = load_references(metadata)

    requested = [c.strip() for c in args.clips.split(",") if c.strip()]
    selected: list[str] = []
    for stem in requested:
        if clip_path(clips_dir, stem).exists():
            selected.append(stem)
        else:
            print(f"warning: skipping missing clip {stem}", file=sys.stderr)
    if not selected:
        print("error: no clips available", file=sys.stderr)
        return 2

    print(f"model:    {args.model}")
    print(f"ours:     {args.ours}")
    print(f"upstream: {args.upstream}")
    print(f"clips:    {len(selected)} -> {work}")
    print()

    rows: list[dict[str, object]] = []
    refs: list[str] = []
    ours_hyps: list[str] = []
    up_hyps: list[str] = []

    for stem in selected:
        mp3 = clip_path(clips_dir, stem)
        wav = work / f"{stem}.wav"
        print(f"[{stem}] decoding {mp3.name}", file=sys.stderr)
        decode(mp3, wav)

        ours_cmd = [
            str(args.ours),
            "-m",
            str(args.model),
            "-f",
            str(wav),
            "-l",
            args.language,
            "-t",
            args.threads,
            "-b",
            args.beam_size,
            "--no-prints",
        ]
        upstream_cmd = [
            str(args.upstream),
            "-m",
            str(args.model),
            "-f",
            str(wav),
            "-l",
            args.language,
            "-t",
            args.threads,
            "-bs",
            args.beam_size,
            "-np",
        ]

        ours_text, ours_raw = run_cli(ours_cmd, "whisper.rs", REQUIRED_HEADROOM_GB)
        time.sleep(args.cooldown)
        upstream_text, upstream_raw = run_cli(
            upstream_cmd, "upstream", UPSTREAM_HEADROOM_GB
        )
        time.sleep(args.cooldown)

        (work / f"{stem}.ours.raw.txt").write_text(ours_raw, encoding="utf-8")
        (work / f"{stem}.upstream.raw.txt").write_text(upstream_raw, encoding="utf-8")

        (work / f"{stem}.ours.txt").write_text(ours_text + "\n", encoding="utf-8")
        (work / f"{stem}.upstream.txt").write_text(
            upstream_text + "\n", encoding="utf-8"
        )

        reference = references.get(f"VOT-Denpasar_18-12-25-0-{stem}", "")
        norm_ref = normalize(reference)
        norm_ours = normalize(ours_text)
        norm_up = normalize(upstream_text)
        refs.append(norm_ref)
        ours_hyps.append(norm_ours)
        up_hyps.append(norm_up)

        rows.append(
            {
                "clip": stem,
                "ref_words": len(norm_ref.split()),
                "ours_words": len(norm_ours.split()),
                "up_words": len(norm_up.split()),
                "ours_wer": safe_wer(norm_ref, norm_ours),
                "up_wer": safe_wer(norm_ref, norm_up),
                "ours_text": norm_ours,
                "up_text": norm_up,
            }
        )

    header = (
        f"{'clip':<7} {'ref':>4} {'ours':>5} {'up':>4} "
        f"{'ours_wer':>9} {'up_wer':>8}  {'delta':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        delta = float(row["ours_wer"]) - float(row["up_wer"])
        print(
            f"{row['clip']:<7} {row['ref_words']:>4} {row['ours_words']:>5} "
            f"{row['up_words']:>4} {row['ours_wer']:>9.4f} {row['up_wer']:>8.4f} "
            f"{delta:>+7.4f}"
        )
    print("-" * len(header))

    corpus_ours = float(jiwer.wer(refs, ours_hyps)) if any(refs) else 0.0
    corpus_up = float(jiwer.wer(refs, up_hyps)) if any(refs) else 0.0
    print(f"corpus WER  whisper.rs = {corpus_ours:.4f}   upstream = {corpus_up:.4f}")

    print()
    print("truncation check (ours vs upstream hypothesis length):")
    flagged = False
    for row in rows:
        ours_words = int(row["ours_words"])
        up_words = int(row["up_words"])
        if ours_words < 0.6 * up_words:
            flagged = True
            print(f"  {row['clip']}: ours={ours_words} words, upstream={up_words} words")
            print(f"    whisper.rs: {row['ours_text']}")
            print(f"    upstream:   {row['up_text']}")
    if not flagged:
        print("  no clip lost more than 40% of the upstream word count")

    return 0


if __name__ == "__main__":
    sys.exit(main())