use crate::AudioError;
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

pub const WHISPER_SAMPLE_RATE: u32 = 16000;

#[derive(Debug, Clone)]
pub struct WavAudio {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
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
        let mut channels = 0u16;
        let mut sample_rate = 0u32;
        let mut bits_per_sample = 0u16;
        let mut data_found = false;
        let mut samples = Vec::new();

        while !data_found {
            let mut chunk_id = [0u8; 4];
            if reader.read_exact(&mut chunk_id).is_err() {
                break;
            }
            let chunk_size = reader.read_u32::<LittleEndian>()? as usize;

            match &chunk_id {
                b"fmt " => {
                    audio_format = reader.read_u16::<LittleEndian>()?;
                    channels = reader.read_u16::<LittleEndian>()?;
                    sample_rate = reader.read_u32::<LittleEndian>()?;
                    let _byte_rate = reader.read_u32::<LittleEndian>()?;
                    let _block_align = reader.read_u16::<LittleEndian>()?;
                    bits_per_sample = reader.read_u16::<LittleEndian>()?;

                    if chunk_size > 16 {
                        reader.seek(SeekFrom::Current((chunk_size - 16) as i64))?;
                    }
                }
                b"data" => {
                    if channels == 0 || sample_rate == 0 {
                        return Err(AudioError::InvalidWav(
                            "fmt chunk missing before data".to_string(),
                        ));
                    }

                    let num_channels = channels as usize;
                    match (audio_format, bits_per_sample) {
                        (1, 16) => {
                            // 16-bit signed PCM
                            let num_samples = chunk_size / 2;
                            let total_frames = num_samples / num_channels;
                            samples.reserve(total_frames);

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for _ in 0..num_channels {
                                    let s = reader.read_i16::<LittleEndian>()? as f32;
                                    sum += s / 32768.0;
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        (3, 32) => {
                            // 32-bit IEEE float
                            let num_samples = chunk_size / 4;
                            let total_frames = num_samples / num_channels;
                            samples.reserve(total_frames);

                            for _ in 0..total_frames {
                                let mut sum = 0.0f32;
                                for _ in 0..num_channels {
                                    let s = reader.read_f32::<LittleEndian>()?;
                                    sum += s;
                                }
                                samples.push(sum / num_channels as f32);
                            }
                        }
                        _ => {
                            return Err(AudioError::UnsupportedFormat(format!(
                                "Format {}, {} bits per sample not supported",
                                audio_format, bits_per_sample
                            )));
                        }
                    }
                    data_found = true;
                }
                _ => {
                    // Skip unknown chunk
                    reader.seek(SeekFrom::Current(chunk_size as i64))?;
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
        })
    }
}
