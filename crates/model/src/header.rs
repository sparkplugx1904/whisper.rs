use crate::ModelError;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

pub const GGML_FILE_MAGIC: u32 = 0x67676d6c; // "ggml"
pub const GGML_QNT_VERSION_FACTOR: i32 = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV3,
    LargeV3Turbo,
    Unknown,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Tiny => write!(f, "tiny"),
            ModelType::Base => write!(f, "base"),
            ModelType::Small => write!(f, "small"),
            ModelType::Medium => write!(f, "medium"),
            ModelType::Large => write!(f, "large"),
            ModelType::LargeV3 => write!(f, "large-v3"),
            ModelType::LargeV3Turbo => write!(f, "large-v3-turbo"),
            ModelType::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyperparameters {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    pub ftype: i32,
    pub qntvr: i32,
    pub model_type: ModelType,
}

impl Hyperparameters {
    pub fn is_multilingual(&self) -> bool {
        self.n_vocab >= 51865
    }

    pub fn num_languages(&self) -> usize {
        (self.n_vocab - 51765 - if self.is_multilingual() { 1 } else { 0 }) as usize
    }

    pub fn read<R: Read>(reader: &mut R) -> Result<Self, ModelError> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGML_FILE_MAGIC {
            return Err(ModelError::InvalidMagic(magic));
        }

        let n_vocab = reader.read_i32::<LittleEndian>()?;
        let n_audio_ctx = reader.read_i32::<LittleEndian>()?;
        let n_audio_state = reader.read_i32::<LittleEndian>()?;
        let n_audio_head = reader.read_i32::<LittleEndian>()?;
        let n_audio_layer = reader.read_i32::<LittleEndian>()?;
        let n_text_ctx = reader.read_i32::<LittleEndian>()?;
        let n_text_state = reader.read_i32::<LittleEndian>()?;
        let n_text_head = reader.read_i32::<LittleEndian>()?;
        let n_text_layer = reader.read_i32::<LittleEndian>()?;
        let n_mels = reader.read_i32::<LittleEndian>()?;
        let raw_ftype = reader.read_i32::<LittleEndian>()?;

        if n_text_state != n_audio_state {
            return Err(ModelError::FormatError(format!(
                "n_text_state ({}) != n_audio_state ({})",
                n_text_state, n_audio_state
            )));
        }

        let qntvr = raw_ftype / GGML_QNT_VERSION_FACTOR;
        let ftype = raw_ftype % GGML_QNT_VERSION_FACTOR;

        let model_type = match n_audio_layer {
            4 => ModelType::Tiny,
            6 => ModelType::Base,
            12 => ModelType::Small,
            24 => ModelType::Medium,
            32 => {
                if n_vocab == 51866 {
                    if n_text_layer == 4 {
                        ModelType::LargeV3Turbo
                    } else {
                        ModelType::LargeV3
                    }
                } else {
                    ModelType::Large
                }
            }
            _ => ModelType::Unknown,
        };

        Ok(Self {
            n_vocab,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
            n_mels,
            ftype,
            qntvr,
            model_type,
        })
    }
}
