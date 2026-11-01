//! Model loader for whisper.rs supporting whisper.cpp ggml format (.bin).

pub mod filters;
pub mod header;
pub mod loader;
pub mod tensor_info;
pub mod vocab;

pub use filters::MelFilters;
pub use header::{Hyperparameters, ModelType, GGML_FILE_MAGIC};
pub use loader::WhisperModel;
pub use tensor_info::TensorInfo;
pub use vocab::{lang_id, lang_str, lang_str_full, Vocab, LANGUAGES};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number: {0:#x}")]
    InvalidMagic(u32),
    #[error("Unsupported model version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid model format: {0}")]
    FormatError(String),
}
