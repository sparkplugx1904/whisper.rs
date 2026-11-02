#![forbid(unsafe_code)]
pub mod download;
pub use download::{download_model, get_model_download_url};
