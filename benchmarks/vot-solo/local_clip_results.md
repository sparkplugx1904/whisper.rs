# Per-clip beam search comparison (VOT-Denpasar, large-v3-turbo)

Model: `ggml-large-v3-turbo.bin` (F16). Language `id`, 4 threads, temperature 0, timestamps enabled. Every clip is decoded from its MP3 with the repository decoder and transcribed independently; the model is loaded once for each run.

## Provenance and environment

- Harness: `benchmarks/vot-solo/clip_eval` (standalone crate, model loaded once per run).
- Binary built 2026-11-14 12:30 UTC from the working tree at commit `8534178`
  ("keep beam hypotheses alive and enforce timestamp grammar"). It predates
  `526005c`, `3906cd9`, `971a213`, `babcf00` and `c973bba` (the later decoder
  position, speaker-turn, beam-prefix-scoring and special-token fixes), so the
  numbers below describe the beam fix at `8534178`, not current `HEAD`
  (`c973bba`).
- Host: 2 vCPU, 7.8 GB RAM, no swap. The engine holds the F16 model as ~3.2 GB
  of F32 weights plus ~1.6 GB of mmapped file pages, so a run peaks at
  4.5-4.7 GB RSS. When two engine processes overlap, the Codespaces supervisor
  sends SIGTERM to the newer one (no OOM log; exit status 143). Beam attempt 1
  was killed that way; attempt 2 completed once the engine slot was exclusive.
- The task asked for greedy and beam on 26 clips. Greedy ran on all 26. Beam
  was reduced on the main agent's instruction to the 10 worst greedy clips
  because of the memory ceiling, so the table below covers those 10 for both
  modes and the aggregate WERs are computed on the same subset.

## Full greedy run (all 26 clips)

The greedy sweep over the complete selection transcribed 151.19 s of audio in
1111.96 s (RTF 7.36, wall 1112.3 s) with an aggregate WER of **0.1662** over 26
clips. Beam was not measured on the other 16 clips.

## Aggregate WER

| mode | beam size | clips | audio (s) | compute (s) | RTF | aggregate WER | mean clip WER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | 1 | 10 | 54.9 | 469.4 | 8.55 | 0.3333 | 0.3442 |
| beam | 5 | 10 | 54.9 | 703.6 | 12.82 | 0.4048 | 0.4195 |

## Commands

- greedy: `cd benchmarks/vot-solo/clip_eval && CARGO_TARGET_DIR=/tmp/vot-clip-eval-target cargo build --release && /tmp/vot-clip-eval-target/release/clip-eval --model /tmp/vot-dbg/models/model.bin --metadata /tmp/vot-dbg/metadata.tsv --clips /tmp/vot-dbg/clips --indices 0,1,2,3,4,9,16,25,48,59,79,86,114,142,146,169,178,210,222,223,224,225,226,227,236,279 --beam-size 1 --out /tmp/vot-dbg/results_greedy.json`
- beam 5: `/tmp/vot-clip-eval-target/release/clip-eval --model /tmp/vot-dbg/models/model.bin --metadata /tmp/vot-dbg/metadata.tsv --clips /tmp/vot-dbg/clips --indices 0,2,279,9,86,114,1,4,48,16 --beam-size 5 --out /tmp/vot-dbg/results_beam.json`
- greedy wall time: 1112.3 s wall (26 clips, 151.19 s audio, RTF 7.36, peak RSS 4.74 GB)
- beam 5 wall time: 711 s wall (10 clips, 54.89 s audio, RTF 12.82, peak RSS 4.50 GB)

## Per-clip results

| index | duration (s) | ref words | greedy WER | beam WER | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4.86 | 9 | 1.000 | 1.000 | beam short: 2/9 ref words |
| 1 | 4.44 | 9 | 0.222 | 0.444 | beam worsens (0.22 -> 0.44) |
| 2 | 9.52 | 22 | 0.455 | 0.409 | same |
| 4 | 9.16 | 26 | 0.192 | 0.269 | beam worsens (0.19 -> 0.27) |
| 9 | 3.81 | 9 | 0.333 | 0.333 | same |
| 16 | 4.04 | 12 | 0.167 | 0.167 | same |
| 48 | 2.62 | 6 | 0.167 | 0.167 | same |
| 86 | 4.38 | 10 | 0.300 | 0.300 | same |
| 114 | 5.91 | 11 | 0.273 | 0.273 | same |
| 279 | 6.16 | 12 | 0.333 | 0.833 | beam repeats 'terima kasih' 4 times |

## Remaining truncations

### Clip 0 — beam short: 2/9 ref words

- reference: Voice of Trisma, the best online radio in Bali.
- greedy: [SPEAKER_TURN] Terima kasih.
- beam: [SPEAKER_TURN] Terima kasih.


## Observed effect of the beam fix at `8534178`

- On the 10 worst greedy clips, beam-5 does not improve aggregate WER
  (0.4048 vs 0.3333) and is 1.5x slower (RTF 12.82 vs 8.55 on the same subset;
  ~70 s for every 3-10 s clip).
- The one real gain is clip 2 (22 reference words): 0.455 -> 0.409, where beam
  recovers more of the sentence after the opening address.
- Regressions come from extra trailing text and repetition: clip 1 appends
  "Terima kasih" after a complete hypothesis, and clip 279 repeats
  "Terima kasih" four times after the closing line (0.333 -> 0.833).
- Clip 0 (station ID) still decodes only `[SPEAKER_TURN] Terima kasih.` under
  both modes. This is the symptom the later commit `c973bba` ("keep special
  tokens out of the beam prompt and logits") targets; the numbers here cannot
  show whether it fixes the intro, because the harness binary predates it.
  Rerunning `clip_eval` at `HEAD` is the next measurement.
