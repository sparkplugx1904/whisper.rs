#![forbid(unsafe_code)]
pub mod download;
pub use download::{download_model, get_model_download_url};
pub mod format;
pub use format::{
    color_for_prob, confidence_style, export_segments_to_file, format_colored_token,
    format_confidence_token, format_segment_colors, format_segment_confidence, rgb2xterm256,
    CsvFormatter, JsonFormatter, KaraokeVttFormatter, LrcFormatter, OutputFormatOptions,
    OutputFormatter, SrtFormatter, TxtFormatter, VttFormatter, WtsFormatter,
};
