use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::resample::resample_to_16k;
use crate::wav::{WavAudio, WHISPER_SAMPLE_RATE};
use crate::AudioError;

/// Load audio from file path supporting WAV, MP3, FLAC, OGG, AAC.
/// Automatically decodes to mono 32-bit float PCM and resamples to 16 kHz.
pub fn load_audio_from_file(path: impl AsRef<Path>) -> Result<WavAudio, AudioError> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path_ref.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    decode_media_source(mss, hint)
}

/// Load audio from in-memory byte slice supporting WAV, MP3, FLAC, OGG, AAC.
/// Automatically decodes to mono 32-bit float PCM and resamples to 16 kHz.
pub fn load_audio_from_bytes(bytes: &[u8]) -> Result<WavAudio, AudioError> {
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let hint = Hint::new();
    decode_media_source(mss, hint)
}

fn decode_media_source(mss: MediaSourceStream, hint: Hint) -> Result<WavAudio, AudioError> {
    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let meta_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &meta_opts)
        .map_err(|e| AudioError::InvalidWav(format!("Audio probe failed: {}", e)))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| {
            AudioError::InvalidWav("No audio track found in media container".to_string())
        })?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.ok_or_else(|| {
        AudioError::InvalidWav("Missing sample rate in track codec parameters".to_string())
    })?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| {
            AudioError::InvalidWav(format!("Audio decoder initialization failed: {}", e))
        })?;

    let mut mono_samples = Vec::new();

    loop {
        let packet: symphonia::core::formats::Packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                append_samples_mono(&decoded, &mut mono_samples);
            }
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    if mono_samples.is_empty() {
        return Err(AudioError::InvalidWav(
            "Zero audio samples decoded from media stream".to_string(),
        ));
    }

    // Auto-resample to 16 kHz if source sample rate differs
    let final_samples = if sample_rate != WHISPER_SAMPLE_RATE {
        resample_to_16k(&mono_samples, sample_rate)
    } else {
        mono_samples
    };

    Ok(WavAudio {
        sample_rate: WHISPER_SAMPLE_RATE,
        channels: 1,
        samples: final_samples,
    })
}

fn append_samples_mono(buffer: &AudioBufferRef, out: &mut Vec<f32>) {
    match buffer {
        AudioBufferRef::F32(buf) => {
            let n_channels = buf.spec().channels.count();
            let n_frames = buf.frames();
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for ch in 0..n_channels {
                    sum += buf.chan(ch)[frame_idx];
                }
                out.push(sum / n_channels as f32);
            }
        }
        AudioBufferRef::U8(buf) => {
            let n_channels = buf.spec().channels.count();
            let n_frames = buf.frames();
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for ch in 0..n_channels {
                    let s = (buf.chan(ch)[frame_idx] as f32 - 128.0) / 128.0;
                    sum += s;
                }
                out.push(sum / n_channels as f32);
            }
        }
        AudioBufferRef::S16(buf) => {
            let n_channels = buf.spec().channels.count();
            let n_frames = buf.frames();
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for ch in 0..n_channels {
                    let s = buf.chan(ch)[frame_idx] as f32 / 32768.0;
                    sum += s;
                }
                out.push(sum / n_channels as f32);
            }
        }
        AudioBufferRef::S24(buf) => {
            let n_channels = buf.spec().channels.count();
            let n_frames = buf.frames();
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for ch in 0..n_channels {
                    let s = buf.chan(ch)[frame_idx].0 as f32 / 8388608.0;
                    sum += s;
                }
                out.push(sum / n_channels as f32);
            }
        }
        AudioBufferRef::S32(buf) => {
            let n_channels = buf.spec().channels.count();
            let n_frames = buf.frames();
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for ch in 0..n_channels {
                    let s = buf.chan(ch)[frame_idx] as f32 / 2147483648.0;
                    sum += s;
                }
                out.push(sum / n_channels as f32);
            }
        }
        AudioBufferRef::F64(buf) => {
            let n_channels = buf.spec().channels.count();
            let n_frames = buf.frames();
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for ch in 0..n_channels {
                    sum += buf.chan(ch)[frame_idx] as f32;
                }
                out.push(sum / n_channels as f32);
            }
        }
        _ => {}
    }
}
