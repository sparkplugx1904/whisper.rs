//! Per-clip evaluation harness for the curated VOT-Denpasar dataset.
//!
//! Loads the model once, decodes each selected clip to 16 kHz mono f32 with the
//! shared `whisper-audio` decoder, runs a full transcription, and records the
//! reference and hypothesis as JSON so that WER can be scored offline.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use whisper_rs::params::{FullParamsBuilder, SamplingStrategy};
use whisper_rs::transcribe::WhisperContext;

const SAMPLE_RATE: f64 = 16_000.0;

struct Args {
    model: PathBuf,
    metadata: PathBuf,
    clips: PathBuf,
    indices: Option<Vec<usize>>,
    beam_size: usize,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: clip-eval --model <ggml> --metadata <metadata.tsv> --clips <dir> \
     --indices <0,1,2|all> --beam-size <n> --out <results.json>"
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut metadata = None;
    let mut clips = None;
    let mut indices = None;
    let mut beam_size = 5usize;
    let mut out = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--metadata" => metadata = Some(PathBuf::from(value()?)),
            "--clips" => clips = Some(PathBuf::from(value()?)),
            "--indices" => {
                let raw = value()?;
                if raw != "all" {
                    let parsed = raw
                        .split(',')
                        .map(|item| {
                            item.trim()
                                .parse::<usize>()
                                .map_err(|_| format!("invalid clip index: {item}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    indices = Some(parsed);
                }
            }
            "--beam-size" => {
                beam_size = value()?
                    .parse::<usize>()
                    .map_err(|_| "invalid --beam-size".to_string())?
                    .max(1);
            }
            "--out" => out = Some(PathBuf::from(value()?)),
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
    }

    Ok(Args {
        model: model.ok_or_else(|| format!("missing --model\n{}", usage()))?,
        metadata: metadata.ok_or_else(|| format!("missing --metadata\n{}", usage()))?,
        clips: clips.ok_or_else(|| format!("missing --clips\n{}", usage()))?,
        indices,
        beam_size,
        out: out.ok_or_else(|| format!("missing --out\n{}", usage()))?,
    })
}

/// Metadata rows: `path<TAB>sentence`. The header is skipped and the first
/// column is the clip file name.
fn load_metadata(path: &Path) -> Result<Vec<(String, String)>, String> {
    let content = fs::read_to_string(path).map_err(|err| format!("{path:?}: {err}"))?;
    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim_start_matches('\u{feff}');
        if line_no == 0 || line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let name = parts.next().unwrap_or("").trim();
        let sentence = parts.next().unwrap_or("").trim();
        if !name.is_empty() {
            rows.push((name.to_string(), sentence.to_string()));
        }
    }
    Ok(rows)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    let rows = match load_metadata(&args.metadata) {
        Ok(rows) => rows,
        Err(message) => {
            eprintln!("failed to read metadata: {message}");
            return ExitCode::FAILURE;
        }
    };

    let selected: Vec<usize> = match &args.indices {
        Some(indices) => {
            if let Some(bad) = indices.iter().find(|&&index| index >= rows.len()) {
                eprintln!(
                    "clip index {bad} out of range (metadata has {} rows)",
                    rows.len()
                );
                return ExitCode::FAILURE;
            }
            indices.clone()
        }
        None => (0..rows.len()).collect(),
    };

    println!(
        "loading model {} (beam_size = {})",
        args.model.display(),
        args.beam_size
    );
    let load_start = Instant::now();
    let ctx = match WhisperContext::load_model(&args.model) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("failed to load model: {err}");
            return ExitCode::FAILURE;
        }
    };
    let load_s = load_start.elapsed().as_secs_f64();
    println!("model loaded in {load_s:.2}s");

    let params = FullParamsBuilder::new(SamplingStrategy::BeamSearch {
        beam_size: args.beam_size,
    })
    .language("id")
    .n_threads(4)
    .temperature(0.0)
    .build();

    let mut state = ctx.create_state();
    let mut results = Vec::with_capacity(selected.len());
    let mut audio_s = 0.0f64;
    let mut transcribe_s = 0.0f64;
    let run_start = Instant::now();

    for index in selected {
        let (name, reference) = &rows[index];
        let clip_path = args.clips.join(name);

        let audio = match whisper_audio::decode::load_audio_from_file(&clip_path) {
            Ok(audio) => audio,
            Err(err) => {
                eprintln!("clip {index} ({name}): decode failed: {err}");
                return ExitCode::FAILURE;
            }
        };
        let samples = if audio.sample_rate != SAMPLE_RATE as u32 {
            whisper_audio::resample::resample_to_16k(&audio.samples, audio.sample_rate)
        } else {
            audio.samples
        };
        let duration_s = samples.len() as f64 / SAMPLE_RATE;
        audio_s += duration_s;

        let clip_start = Instant::now();
        if let Err(err) = ctx.full(&mut state, &params, &samples) {
            eprintln!("clip {index} ({name}): transcription failed: {err}");
            return ExitCode::FAILURE;
        }
        let elapsed_s = clip_start.elapsed().as_secs_f64();
        transcribe_s += elapsed_s;

        let text = state
            .segments
            .iter()
            .map(|segment| segment.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        let tokens: Vec<i32> = state
            .segments
            .iter()
            .flat_map(|segment| segment.tokens.iter().copied())
            .collect();
        let segments: Vec<serde_json::Value> = state
            .segments
            .iter()
            .map(|segment| {
                serde_json::json!({
                    "t0": segment.t0,
                    "t1": segment.t1,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                })
            })
            .collect();

        println!(
            "clip {index:>3} {duration_s:>6.2}s {elapsed_s:>7.2}s rtf {:.2} | {}",
            elapsed_s / duration_s.max(1e-6),
            text
        );

        results.push(serde_json::json!({
            "index": index,
            "path": clip_path.to_string_lossy(),
            "reference": reference,
            "text": text,
            "tokens": tokens,
            "duration_s": (duration_s * 1000.0).round() / 1000.0,
            "elapsed_s": (elapsed_s * 1000.0).round() / 1000.0,
            "segments": segments,
        }));
    }

    let run_s = run_start.elapsed().as_secs_f64();
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(err) = fs::create_dir_all(parent) {
                eprintln!("failed to create {}: {err}", parent.display());
                return ExitCode::FAILURE;
            }
        }
    }
    let payload = serde_json::to_string_pretty(&results).expect("results serialize");
    if let Err(err) = fs::write(&args.out, payload) {
        eprintln!("failed to write {}: {err}", args.out.display());
        return ExitCode::FAILURE;
    }

    println!(
        "done: {} clips, {:.2}s audio, {:.2}s transcribe, {:.2}s wall, model load {:.2}s, rtf {:.3}",
        results.len(),
        audio_s,
        transcribe_s,
        run_s,
        load_s,
        transcribe_s / audio_s.max(1e-9)
    );
    ExitCode::SUCCESS
}
