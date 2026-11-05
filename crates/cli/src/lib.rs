#![forbid(unsafe_code)]
pub mod download;
pub use download::{download_model, get_model_download_url};
pub mod format;
pub use format::{
    export_segments_to_file, CsvFormatter, JsonFormatter, KaraokeVttFormatter, LrcFormatter,
    OutputFormatOptions, OutputFormatter, SrtFormatter, TxtFormatter, VttFormatter,
};
