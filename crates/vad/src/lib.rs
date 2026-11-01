//! Voice Activity Detection (VAD) for whisper.rs.

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid VAD model format: {0}")]
    FormatError(String),
}

#[derive(Debug, Clone)]
pub struct VadParams {
    pub threshold: f32,
    pub min_speech_duration_ms: usize,
    pub min_silence_duration_ms: usize,
    pub speech_pad_ms: usize,
    pub sample_rate: usize,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            sample_rate: 16000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    pub t0: f64,
    pub t1: f64,
}

impl SpeechSegment {
    pub fn new(t0: f64, t1: f64) -> Self {
        Self { t0, t1 }
    }
}

/// Detect active speech segments from raw PCM samples (16,000 Hz).
pub fn detect_speech_segments(samples: &[f32], params: &VadParams) -> Vec<SpeechSegment> {
    if samples.is_empty() {
        return Vec::new();
    }

    let frame_len = (params.sample_rate * 30) / 1000; // 30ms = 480 samples
    let frame_hop = (params.sample_rate * 10) / 1000; // 10ms = 160 samples

    if samples.len() < frame_len {
        return vec![SpeechSegment::new(
            0.0,
            samples.len() as f64 / params.sample_rate as f64,
        )];
    }

    // 1. Calculate short-time RMS energy per frame
    let n_frames = (samples.len() - frame_len) / frame_hop + 1;
    let mut is_speech = vec![false; n_frames];

    // Compute noise floor
    let mut energies = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let offset = i * frame_hop;
        let frame = &samples[offset..offset + frame_len];
        let mut sum_sq = 0.0f32;
        for &s in frame {
            sum_sq += s * s;
        }
        let rms = (sum_sq / frame_len as f32).sqrt();
        energies.push(rms);
    }

    let mut sorted_energies = energies.clone();
    sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_energy = sorted_energies[sorted_energies.len() / 2];
    let dyn_threshold = (params.threshold * 0.02).max(median_energy * 2.0);

    for (i, &e) in energies.iter().enumerate() {
        if e > dyn_threshold {
            is_speech[i] = true;
        }
    }

    // 2. Merge silent frames shorter than min_silence_duration_ms
    let min_silence_frames = params.min_silence_duration_ms / 10;
    let mut i = 0;
    while i < n_frames {
        if !is_speech[i] {
            let start = i;
            while i < n_frames && !is_speech[i] {
                i += 1;
            }
            let silence_len = i - start;
            if silence_len < min_silence_frames && start > 0 && i < n_frames {
                is_speech[start..i].fill(true);
            }
        } else {
            i += 1;
        }
    }

    // 3. Extract continuous speech segments
    let min_speech_frames = params.min_speech_duration_ms / 10;
    let pad_sec = (params.speech_pad_ms as f64) / 1000.0;
    let total_sec = samples.len() as f64 / params.sample_rate as f64;

    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut seg_start = 0;

    for (k, &speech) in is_speech.iter().enumerate() {
        if speech && !in_speech {
            in_speech = true;
            seg_start = k;
        } else if !speech && in_speech {
            in_speech = false;
            let seg_len = k - seg_start;
            if seg_len >= min_speech_frames {
                let t0 =
                    ((seg_start * frame_hop) as f64 / params.sample_rate as f64 - pad_sec).max(0.0);
                let t1 = ((k * frame_hop + frame_len) as f64 / params.sample_rate as f64 + pad_sec)
                    .min(total_sec);
                segments.push(SpeechSegment::new(t0, t1));
            }
        }
    }

    if in_speech {
        let seg_len = n_frames - seg_start;
        if seg_len >= min_speech_frames {
            let t0 =
                ((seg_start * frame_hop) as f64 / params.sample_rate as f64 - pad_sec).max(0.0);
            let t1 = total_sec;
            segments.push(SpeechSegment::new(t0, t1));
        }
    }

    // If no voice segment was detected, fall back to entire audio
    if segments.is_empty() {
        segments.push(SpeechSegment::new(0.0, total_sec));
    }

    segments
}

/// Loader for Silero VAD GGML format model.
pub struct SileroVadModel {
    pub model_type: String,
    pub major_version: i32,
    pub minor_version: i32,
    pub patch_version: i32,
}

impl SileroVadModel {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, VadError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != 0x67676d6c {
            return Err(VadError::FormatError(format!("Invalid magic {:#x}", magic)));
        }

        let str_len = reader.read_i32::<LittleEndian>()? as usize;
        let mut type_buf = vec![0u8; str_len];
        reader.read_exact(&mut type_buf)?;
        let model_type = String::from_utf8_lossy(&type_buf).into_owned();

        let major = reader.read_i32::<LittleEndian>()?;
        let minor = reader.read_i32::<LittleEndian>()?;
        let patch = reader.read_i32::<LittleEndian>()?;

        Ok(Self {
            model_type,
            major_version: major,
            minor_version: minor,
            patch_version: patch,
        })
    }
}
