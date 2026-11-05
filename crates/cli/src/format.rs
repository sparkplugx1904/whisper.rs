use std::io::Write;
use whisper_rs::Segment;

/// Formatter strategy for serializing transcription segments into various subtitle/text formats.
pub trait OutputFormatter: Send + Sync {
    /// Friendly format name (e.g. "TXT", "SRT", "VTT", "JSON", "Karaoke VTT").
    fn name(&self) -> &'static str;

    /// File extension for this format (without leading dot).
    fn file_extension(&self) -> &'static str;

    /// Format and write segments into any standard stream implementing `std::io::Write`.
    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()>;
}

/// Plain text (.txt) formatter.
#[derive(Debug, Default, Clone, Copy)]
pub struct TxtFormatter;

impl OutputFormatter for TxtFormatter {
    fn name(&self) -> &'static str {
        "TXT"
    }

    fn file_extension(&self) -> &'static str {
        "txt"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        for s in segments {
            writeln!(writer, "{}", s.text.trim())?;
        }
        Ok(())
    }
}

/// SubRip (.srt) subtitle formatter.
#[derive(Debug, Default, Clone, Copy)]
pub struct SrtFormatter;

impl OutputFormatter for SrtFormatter {
    fn name(&self) -> &'static str {
        "SRT"
    }

    fn file_extension(&self) -> &'static str {
        "srt"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        for (i, s) in segments.iter().enumerate() {
            write!(writer, "{}", s.to_srt(i + 1))?;
        }
        Ok(())
    }
}

/// WebVTT (.vtt) subtitle formatter.
#[derive(Debug, Default, Clone, Copy)]
pub struct VttFormatter;

impl OutputFormatter for VttFormatter {
    fn name(&self) -> &'static str {
        "VTT"
    }

    fn file_extension(&self) -> &'static str {
        "vtt"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        writeln!(writer, "WEBVTT\n")?;
        for s in segments {
            write!(writer, "{}", s.to_vtt())?;
        }
        Ok(())
    }
}

/// JSON (.json) segment array formatter.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonFormatter;

impl OutputFormatter for JsonFormatter {
    fn name(&self) -> &'static str {
        "JSON"
    }

    fn file_extension(&self) -> &'static str {
        "json"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        writeln!(writer, "[")?;
        for (i, s) in segments.iter().enumerate() {
            let comma = if i + 1 < segments.len() { "," } else { "" };
            writeln!(writer, "  {}{}", s.to_json(), comma)?;
        }
        writeln!(writer, "]")?;
        Ok(())
    }
}

/// Karaoke WebVTT (.karaoke.vtt) subtitle formatter with word-level timing tags.
#[derive(Debug, Default, Clone, Copy)]
pub struct KaraokeVttFormatter;

impl OutputFormatter for KaraokeVttFormatter {
    fn name(&self) -> &'static str {
        "Karaoke VTT"
    }

    fn file_extension(&self) -> &'static str {
        "karaoke.vtt"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        writeln!(writer, "WEBVTT\n")?;
        for s in segments {
            write!(writer, "{}", s.to_karaoke_vtt())?;
        }
        Ok(())
    }
}

/// Comma-Separated Values (.csv) subtitle formatter matching whisper.cpp format.
#[derive(Debug, Default, Clone, Copy)]
pub struct CsvFormatter;

impl OutputFormatter for CsvFormatter {
    fn name(&self) -> &'static str {
        "CSV"
    }

    fn file_extension(&self) -> &'static str {
        "csv"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        writeln!(writer, "start,end,text")?;
        for s in segments {
            let t0_ms = (s.t0.max(0.0) * 1000.0).round() as u64;
            let t1_ms = (s.t1.max(0.0) * 1000.0).round() as u64;
            let escaped = s.text.trim().replace('"', "\"\"");
            writeln!(writer, "{},{},\"{}\"", t0_ms, t1_ms, escaped)?;
        }
        Ok(())
    }
}

/// Synchronized lyrics (.lrc) formatter matching whisper.cpp format.
#[derive(Debug, Default, Clone, Copy)]
pub struct LrcFormatter;

impl OutputFormatter for LrcFormatter {
    fn name(&self) -> &'static str {
        "LRC"
    }

    fn file_extension(&self) -> &'static str {
        "lrc"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        writeln!(writer, "[by:whisper.cpp]")?;
        for s in segments {
            let secs = s.t0.max(0.0);
            let total_cs = (secs * 100.0).round() as u64;
            let cs = total_cs % 100;
            let total_s = total_cs / 100;
            let s_rem = total_s % 60;
            let min = total_s / 60;
            writeln!(
                writer,
                "[{:02}:{:02}.{:02}]{}",
                min,
                s_rem,
                cs,
                s.text.trim()
            )?;
        }
        Ok(())
    }
}

/// Configuration for output format selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutputFormatOptions {
    pub output_txt: bool,
    pub output_srt: bool,
    pub output_vtt: bool,
    pub output_json: bool,
    pub output_csv: bool,
    pub output_lrc: bool,
    pub karaoke: bool,
}

impl OutputFormatOptions {
    /// Collect active formatters in stable order.
    pub fn active_formatters(&self) -> Vec<Box<dyn OutputFormatter>> {
        let mut list: Vec<Box<dyn OutputFormatter>> = Vec::new();
        if self.output_txt {
            list.push(Box::new(TxtFormatter));
        }
        if self.output_srt {
            list.push(Box::new(SrtFormatter));
        }
        if self.output_vtt {
            list.push(Box::new(VttFormatter));
        }
        if self.output_json {
            list.push(Box::new(JsonFormatter));
        }
        if self.output_csv {
            list.push(Box::new(CsvFormatter));
        }
        if self.output_lrc {
            list.push(Box::new(LrcFormatter));
        }
        if self.karaoke {
            list.push(Box::new(KaraokeVttFormatter));
        }
        list
    }
}

/// Export segments using a specific formatter to file on disk.
pub fn export_segments_to_file(
    formatter: &dyn OutputFormatter,
    base_name: &str,
    segments: &[Segment],
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}.{}", base_name, formatter.file_extension());
    let mut f = std::fs::File::create(&path).map_err(|e| {
        format!(
            "failed to create {} output file '{}': {}",
            formatter.name(),
            path,
            e
        )
    })?;

    formatter.write_segments(segments, &mut f).map_err(|e| {
        if formatter.name() == "VTT" && e.to_string().contains("header") {
            format!("failed to write VTT header to '{}': {}", path, e)
        } else {
            format!(
                "failed to write {} output file '{}': {}",
                formatter.name(),
                path,
                e
            )
        }
    })?;

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_txt_formatting() {
        let segments = vec![
            Segment::new(0.0, 1.0, " Hello ".to_string(), vec![]),
            Segment::new(1.0, 2.0, " World ".to_string(), vec![]),
        ];
        let mut buf = Vec::new();
        TxtFormatter.write_segments(&segments, &mut buf).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "Hello\nWorld\n");
    }

    #[test]
    fn test_in_memory_vtt_formatting() {
        let segments = vec![Segment::new(0.0, 1.0, " Hello ".to_string(), vec![])];
        let mut buf = Vec::new();
        VttFormatter.write_segments(&segments, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("WEBVTT\n\n"));
        assert!(s.contains("00:00:00.000 --> 00:00:01.000\nHello\n\n"));
    }

    #[test]
    fn test_active_formatters_selection() {
        let opts = OutputFormatOptions {
            output_txt: true,
            output_json: true,
            ..Default::default()
        };
        let active = opts.active_formatters();
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].name(), "TXT");
        assert_eq!(active[1].name(), "JSON");
    }
}
