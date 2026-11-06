use crate::AudioError;
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WAVE_FORMAT_PCM: u16 = 1;
pub const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
pub const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// GUID for PCM audio in WAVE_FORMAT_EXTENSIBLE:
/// 00000001-0000-0010-8000-00AA00389B71
pub const KSDATAFORMAT_SUBTYPE_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

/// GUID for IEEE float audio in WAVE_FORMAT_EXTENSIBLE:
/// 00000003-0000-0010-8000-00AA00389B71
pub const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];
#[derive(Debug, Clone)]
pub struct WavAudio {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
    pub channel_samples: Vec<Vec<f32>>,
}

impl WavAudio {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, AudioError> {
        let mut file = File::open(path)?;
        Self::load_from_reader(&mut file)
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, AudioError> {
        let mut cursor = Cursor::new(bytes);
        Self::load_from_reader(&mut cursor)
    }

    pub fn load_from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self, AudioError> {
        let mut riff = [0u8; 4];
        reader.read_exact(&mut riff)?;
        if &riff != b"RIFF" {
            return Err(AudioError::InvalidWav("Missing RIFF header".to_string()));
        }

        let _file_size = reader.read_u32::<LittleEndian>()?;

        let mut wave = [0u8; 4];
        reader.read_exact(&mut wave)?;
        if &wave != b"WAVE" {
            return Err(AudioError::InvalidWav("Missing WAVE chunk".to_string()));
        }

        let mut audio_format = 0u16;
        let mut effective_format = 0u16;
        let mut channels = 0u16;
        let mut sample_rate = 0u32;
        let mut bits_per_sample = 0u16;
        let mut data_found = false;
        let mut samples = Vec::new();
        let mut channel_samples: Vec<Vec<f32>> = Vec::new();
        while !data_found {
            let mut chunk_id = [0u8; 4];
            if reader.read_exact(&mut chunk_id).is_err() {
                break;
            }
            let chunk_size = reader.read_u32::<LittleEndian>()? as usize;

            match &chunk_id {
                b"fmt " => {
                    if chunk_size < 16 {
                        return Err(AudioError::InvalidWav("fmt chunk too small".to_string()));
                    }
                    audio_format = reader.read_u16::<LittleEndian>()?;
                    channels = reader.read_u16::<LittleEndian>()?;
                    sample_rate = reader.read_u32::<LittleEndian>()?;
                    let _byte_rate = reader.read_u32::<LittleEndian>()?;
                    let _block_align = reader.read_u16::<LittleEndian>()?;
                    bits_per_sample = reader.read_u16::<LittleEndian>()?;

                    if audio_format == WAVE_FORMAT_EXTENSIBLE {
                        if chunk_size < 40 {
                            return Err(AudioError::InvalidWav(
                                "fmt chunk too small for WAVE_FORMAT_EXTENSIBLE".to_string(),
                            ));
                        }
                        let cb_size = reader.read_u16::<LittleEndian>()?;
                        if cb_size < 22 {
                            return Err(AudioError::InvalidWav(
                                "cbSize too small for WAVE_FORMAT_EXTENSIBLE".to_string(),
                            ));
                        }
                        let _valid_bits_per_sample = reader.read_u16::<LittleEndian>()?;
                        let _channel_mask = reader.read_u32::<LittleEndian>()?;
                        let mut sub_format_guid = [0u8; 16];
                        reader.read_exact(&mut sub_format_guid)?;

                        if sub_format_guid == KSDATAFORMAT_SUBTYPE_PCM {
                            effective_format = WAVE_FORMAT_PCM;
                        } else if sub_format_guid == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {
                            effective_format = WAVE_FORMAT_IEEE_FLOAT;
                        } else {
                            return Err(AudioError::UnsupportedFormat(format!(
                                "Unsupported WAVE_FORMAT_EXTENSIBLE sub-format GUID: {:02x?}",
                                sub_format_guid
                            )));
                        }

                        if chunk_size > 40 {
                            reader.seek(SeekFrom::Current((chunk_size - 40) as i64))?;
                        }
                    } else {
                        effective_format = audio_format;
                        if chunk_size > 16 {
                            reader.seek(SeekFrom::Current((chunk_size - 16) as i64))?;
                        }
                    }
                }
                b"data" => {
                    if channels == 0 || sample_rate == 0 || effective_format == 0 {
                        return Err(AudioError::InvalidWav(
                            "fmt chunk missing before data".to_string(),
                        ));
                    }

                    let num_channels = channels as usize;
                    match (effective_format, bits_per_sample) {
                        (WAVE_FORMAT_PCM, 8) => {
                            // 8-bit unsigned PCM
                            let num_samples = chunk_size;
                            let total_frames = num_samples.checked_div(num_channels).unwrap_or(0);
                            samples.reserve(total_frames);
                            channel_samples = (0..num_channels)
                                .map(|_| Vec::with_capacity(total_frames))
                                .collect();

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for ch_vec in &mut channel_samples {
                                    let s = reader.read_u8()? as f32;
                                    let val = (s - 128.0) / 128.0;
                                    sum += val;
                                    ch_vec.push(val);
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        (WAVE_FORMAT_PCM, 16) => {
                            // 16-bit signed PCM
                            let num_samples = chunk_size / 2;
                            let total_frames = num_samples.checked_div(num_channels).unwrap_or(0);
                            samples.reserve(total_frames);
                            channel_samples = (0..num_channels)
                                .map(|_| Vec::with_capacity(total_frames))
                                .collect();

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for ch_vec in &mut channel_samples {
                                    let s = reader.read_i16::<LittleEndian>()? as f32;
                                    let val = s / 32768.0;
                                    sum += val;
                                    ch_vec.push(val);
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        (WAVE_FORMAT_PCM, 24) => {
                            // 24-bit signed PCM (packed 3 bytes per sample)
                            let num_samples = chunk_size / 3;
                            let total_frames = num_samples.checked_div(num_channels).unwrap_or(0);
                            samples.reserve(total_frames);
                            channel_samples = (0..num_channels)
                                .map(|_| Vec::with_capacity(total_frames))
                                .collect();

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for ch_vec in &mut channel_samples {
                                    let mut b = [0u8; 3];
                                    reader.read_exact(&mut b)?;
                                    let val_i = i32::from_le_bytes([
                                        b[0],
                                        b[1],
                                        b[2],
                                        if b[2] & 0x80 != 0 { 0xFF } else { 0x00 },
                                    ]);
                                    let val = val_i as f32 / 8388608.0;
                                    sum += val;
                                    ch_vec.push(val);
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        (WAVE_FORMAT_PCM, 32) => {
                            // 32-bit signed PCM
                            let num_samples = chunk_size / 4;
                            let total_frames = num_samples.checked_div(num_channels).unwrap_or(0);
                            samples.reserve(total_frames);
                            channel_samples = (0..num_channels)
                                .map(|_| Vec::with_capacity(total_frames))
                                .collect();

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for ch_vec in &mut channel_samples {
                                    let s = reader.read_i32::<LittleEndian>()? as f32;
                                    let val = s / 2147483648.0;
                                    sum += val;
                                    ch_vec.push(val);
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        (WAVE_FORMAT_IEEE_FLOAT, 32) => {
                            // 32-bit IEEE float
                            let num_samples = chunk_size / 4;
                            let total_frames = num_samples.checked_div(num_channels).unwrap_or(0);
                            samples.reserve(total_frames);
                            channel_samples = (0..num_channels)
                                .map(|_| Vec::with_capacity(total_frames))
                                .collect();

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for ch_vec in &mut channel_samples {
                                    let s = reader.read_f32::<LittleEndian>()?;
                                    sum += s;
                                    ch_vec.push(s);
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        (WAVE_FORMAT_IEEE_FLOAT, 64) => {
                            // 64-bit IEEE float
                            let num_samples = chunk_size / 8;
                            let total_frames = num_samples.checked_div(num_channels).unwrap_or(0);
                            samples.reserve(total_frames);
                            channel_samples = (0..num_channels)
                                .map(|_| Vec::with_capacity(total_frames))
                                .collect();

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for ch_vec in &mut channel_samples {
                                    let s = reader.read_f64::<LittleEndian>()? as f32;
                                    sum += s;
                                    ch_vec.push(s);
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        _ => {
                            return Err(AudioError::UnsupportedFormat(format!(
                                "Format {} (effective {}), {} bits per sample not supported",
                                audio_format, effective_format, bits_per_sample
                            )));
                        }
                    }
                    data_found = true;
                }
                _ => {
                    // Skip unknown chunk
                    let pad = (chunk_size % 2) as i64;
                    reader.seek(SeekFrom::Current(chunk_size as i64 + pad))?;
                }
            }
        }

        if !data_found {
            return Err(AudioError::InvalidWav("data chunk not found".to_string()));
        }

        Ok(Self {
            sample_rate,
            channels,
            samples,
            channel_samples,
        })
    }
}
