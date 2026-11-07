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

pub(crate) fn speaker_id(s: &str) -> &str {
    let s = s.trim();
    if let Some(rest) = s
        .strip_prefix("(speaker ")
        .and_then(|r| r.strip_suffix(')'))
    {
        rest.trim()
    } else if let Some(rest) = s
        .strip_prefix("Speaker ")
        .or_else(|| s.strip_prefix("speaker "))
    {
        rest.trim()
    } else if let Some(rest) = s
        .strip_prefix("<v Speaker")
        .and_then(|r| r.strip_suffix('>'))
    {
        rest.trim()
    } else {
        s
    }
}

/// SubRip (.srt) subtitle formatter.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SrtFormatter {
    pub diarize: bool,
}

impl SrtFormatter {
    pub const fn new(diarize: bool) -> Self {
        Self { diarize }
    }
}

#[allow(non_upper_case_globals)]
pub const SrtFormatter: SrtFormatter = SrtFormatter { diarize: false };

impl OutputFormatter for SrtFormatter {
    fn name(&self) -> &'static str {
        "SRT"
    }

    fn file_extension(&self) -> &'static str {
        "srt"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        for (i, s) in segments.iter().enumerate() {
            let speaker_prefix = match &s.speaker {
                Some(spk) if self.diarize || !spk.is_empty() => {
                    format!("(speaker {}) ", speaker_id(spk))
                }
                None if self.diarize => "(speaker ?) ".to_string(),
                _ => String::new(),
            };
            write!(
                writer,
                "{}\n{} --> {}\n{}{}\n\n",
                i + 1,
                whisper_rs::segment::fmt_time_srt(s.t0),
                whisper_rs::segment::fmt_time_srt(s.t1),
                speaker_prefix,
                s.text.trim()
            )?;
        }
        Ok(())
    }
}
/// WebVTT (.vtt) subtitle formatter.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VttFormatter {
    pub diarize: bool,
}

impl VttFormatter {
    pub const fn new(diarize: bool) -> Self {
        Self { diarize }
    }
}

#[allow(non_upper_case_globals)]
pub const VttFormatter: VttFormatter = VttFormatter { diarize: false };

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
            let speaker_prefix = match &s.speaker {
                Some(spk) if self.diarize || !spk.is_empty() => {
                    format!("<v Speaker{}>", speaker_id(spk))
                }
                None if self.diarize => "<v Speaker?>".to_string(),
                _ => String::new(),
            };
            write!(
                writer,
                "{} --> {}\n{}{}\n\n",
                whisper_rs::segment::fmt_time_vtt(s.t0),
                whisper_rs::segment::fmt_time_vtt(s.t1),
                speaker_prefix,
                s.text.trim()
            )?;
        }
        Ok(())
    }
}

/// JSON (.json) segment array formatter.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JsonFormatter {
    pub diarize: bool,
}

impl JsonFormatter {
    pub const fn new(diarize: bool) -> Self {
        Self { diarize }
    }
}

#[allow(non_upper_case_globals)]
pub const JsonFormatter: JsonFormatter = JsonFormatter { diarize: false };

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
            let t0 = if s.t0.is_finite() && s.t0 >= 0.0 {
                s.t0
            } else {
                0.0
            };
            let t1 = if s.t1.is_finite() && s.t1 >= 0.0 {
                s.t1
            } else {
                0.0
            };
            let spk = s.speaker.as_deref().map(speaker_id).or(if self.diarize {
                Some("?")
            } else {
                None
            });
            let speaker_field = match spk {
                Some(id) => format!(
                    ", \"speaker\": \"{}\"",
                    whisper_rs::segment::escape_json(id)
                ),
                None => String::new(),
            };
            let json_body = if s.words.is_empty() {
                format!(
                    "{{\"start\": {:.3}, \"end\": {:.3}, \"text\": \"{}\", \"avg_logprob\": {:.4}, \"no_speech_prob\": {:.4}{}}}",
                    t0,
                    t1,
                    whisper_rs::segment::escape_json(s.text.trim()),
                    s.avg_logprob,
                    s.no_speech_prob,
                    speaker_field
                )
            } else {
                let words_json = s
                    .words
                    .iter()
                    .map(|w| {
                        format!(
                            "{{\"word\": \"{}\", \"start\": {:.3}, \"end\": {:.3}, \"probability\": {:.4}}}",
                            whisper_rs::segment::escape_json(&w.word),
                            w.t0,
                            w.t1,
                            w.probability
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{{\"start\": {:.3}, \"end\": {:.3}, \"text\": \"{}\", \"avg_logprob\": {:.4}, \"no_speech_prob\": {:.4}, \"words\": [{}]{}}}",
                    t0,
                    t1,
                    whisper_rs::segment::escape_json(s.text.trim()),
                    s.avg_logprob,
                    s.no_speech_prob,
                    words_json,
                    speaker_field
                )
            };
            writeln!(writer, "  {}{}", json_body, comma)?;
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CsvFormatter {
    pub diarize: bool,
}

impl CsvFormatter {
    pub const fn new(diarize: bool) -> Self {
        Self { diarize }
    }
}

#[allow(non_upper_case_globals)]
pub const CsvFormatter: CsvFormatter = CsvFormatter { diarize: false };

impl OutputFormatter for CsvFormatter {
    fn name(&self) -> &'static str {
        "CSV"
    }

    fn file_extension(&self) -> &'static str {
        "csv"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        let diarize_active = self.diarize || segments.iter().any(|s| s.speaker.is_some());
        // Confidence travels with the text: a subtitle line a human is about to
        // edit is only useful if the file says how sure the decoder was. The
        // terminal already color-codes this, but exported files carried none of
        // it (ggml-org/whisper.cpp#427).
        if diarize_active {
            writeln!(writer, "start,end,speaker,text,avg_logprob,no_speech_prob")?;
        } else {
            writeln!(writer, "start,end,text,avg_logprob,no_speech_prob")?;
        }
        for s in segments {
            let t0_ms = (s.t0.max(0.0) * 1000.0).round() as u64;
            let t1_ms = (s.t1.max(0.0) * 1000.0).round() as u64;
            let escaped = s.text.trim().replace('"', "\"\"");
            if diarize_active {
                let spk = s.speaker.as_deref().map(speaker_id).unwrap_or("?");
                writeln!(
                    writer,
                    "{},{},{},\"{}\",{:.4},{:.4}",
                    t0_ms, t1_ms, spk, escaped, s.avg_logprob, s.no_speech_prob
                )?;
            } else {
                writeln!(
                    writer,
                    "{},{},\"{}\",{:.4},{:.4}",
                    t0_ms, t1_ms, escaped, s.avg_logprob, s.no_speech_prob
                )?;
            }
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

#[derive(Debug, Clone, PartialEq)]
struct WtsWord {
    text: String,
    t0: f64,
    t1: f64,
}

/// Video karaoke bash script (.wts.sh) formatter using ffmpeg and drawtext filters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WtsFormatter {
    pub font_path: String,
    pub input_file: String,
}

impl Default for WtsFormatter {
    fn default() -> Self {
        Self {
            font_path: "/System/Library/Fonts/Supplemental/Courier New Bold.ttf".to_string(),
            input_file: "audio.wav".to_string(),
        }
    }
}

impl WtsFormatter {
    pub fn new(font_path: impl Into<String>, input_file: impl Into<String>) -> Self {
        Self {
            font_path: font_path.into(),
            input_file: input_file.into(),
        }
    }
}

impl OutputFormatter for WtsFormatter {
    fn name(&self) -> &'static str {
        "Karaoke Video Script"
    }

    fn file_extension(&self) -> &'static str {
        "wts.sh"
    }

    fn write_segments(&self, segments: &[Segment], writer: &mut dyn Write) -> std::io::Result<()> {
        let t_sec = segments
            .iter()
            .map(|s| s.t1)
            .fold(0.0f64, f64::max)
            .max(0.0)
            + 1.0;

        let input_file = if self.input_file.is_empty() {
            "audio.wav"
        } else {
            &self.input_file
        };

        writeln!(writer, "#!/bin/bash\n")?;
        write!(
            writer,
            "ffmpeg -i {} -f lavfi -i color=size=1200x120:duration={:.2}:rate=25:color=black -vf \"",
            input_file, t_sec
        )?;

        let font = &self.font_path;

        for (i, seg) in segments.iter().enumerate() {
            let t0 = seg.t0.max(0.0);
            let t1 = seg.t1.max(0.0);

            if i > 0 {
                write!(writer, ",")?;
            }

            // Background empty text marker at start of segment
            write!(
                writer,
                "drawtext=fontfile='{}':fontsize=24:fontcolor=gray:x=(w-text_w)/2:y=h/2:text='':enable='between(t,{:.2},{:.2})'",
                font, t0, t0
            )?;

            // Extract word timings from segment. If words are not available, interpolate across text.
            let words: Vec<WtsWord> = if !seg.words.is_empty() {
                seg.words
                    .iter()
                    .map(|w| WtsWord {
                        text: w.word.trim().to_string(),
                        t0: w.t0.max(0.0),
                        t1: w.t1.max(0.0),
                    })
                    .collect()
            } else {
                let parts: Vec<&str> = seg.text.split_whitespace().collect();
                if parts.is_empty() {
                    Vec::new()
                } else {
                    let span = (t1 - t0).max(0.0);
                    let dur = span / parts.len() as f64;
                    parts
                        .iter()
                        .enumerate()
                        .map(|(idx, &p)| WtsWord {
                            text: p.to_string(),
                            t0: t0 + idx as f64 * dur,
                            t1: t0 + (idx + 1) as f64 * dur,
                        })
                        .collect()
                }
            };

            let n = words.len();
            for j in 0..n {
                let mut txt_bg = String::from("> ");
                let mut txt_fg = String::from("> ");
                let mut txt_ul = String::from(r"\ \ ");

                for (k, word) in words.iter().enumerate() {
                    let w_text = &word.text;
                    if k > 0 {
                        txt_bg.push(' ');
                    }
                    txt_bg.push_str(w_text);

                    let char_count = w_text.chars().count();
                    if k == j {
                        txt_fg.push_str(w_text);
                        txt_fg.push('|');
                        for _ in 0..char_count {
                            txt_ul.push('_');
                        }
                    } else {
                        for _ in 0..char_count {
                            txt_fg.push_str(r"\ ");
                            txt_ul.push_str(r"\ ");
                        }
                    }

                    if k < n - 1 {
                        txt_fg.push_str(r"\ ");
                        txt_ul.push_str(r"\ ");
                    }
                }

                // Replace single quotes and escape double quotes for ffmpeg drawtext filter
                let escaped_bg = txt_bg.replace('\'', "\u{2019}").replace('"', "\\\"");
                let escaped_fg = txt_fg.replace('\'', "\u{2019}").replace('"', "\\\"");

                if j == 0 {
                    write!(
                        writer,
                        ",drawtext=fontfile='{}':fontsize=24:fontcolor=gray:x=(w-text_w)/2:y=h/2:text='{}':enable='between(t,{:.2},{:.2})'",
                        font, escaped_bg, t0, t1
                    )?;
                }

                let wt0 = words[j].t0;
                let wt1 = words[j].t1;

                write!(
                    writer,
                    ",drawtext=fontfile='{}':fontsize=24:fontcolor=lightgreen:x=(w-text_w)/2+8:y=h/2:text='{}':enable='between(t,{:.2},{:.2})'",
                    font, escaped_fg, wt0, wt1
                )?;
                write!(
                    writer,
                    ",drawtext=fontfile='{}':fontsize=24:fontcolor=lightgreen:x=(w-text_w)/2+8:y=h/2+16:text='{}':enable='between(t,{:.2},{:.2})'",
                    font, txt_ul, wt0, wt1
                )?;
            }
        }

        writeln!(
            writer,
            "\" -c:v libx264 -pix_fmt yuv420p -y {}.mp4\n\n",
            input_file
        )?;
        writeln!(
            writer,
            "echo \"Your video has been saved to {}.mp4\"\n",
            input_file
        )?;
        writeln!(writer, "echo \"  ffplay {}.mp4\"\n", input_file)?;

        Ok(())
    }
}

/// Configuration for output format selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputFormatOptions {
    pub output_txt: bool,
    pub output_srt: bool,
    pub output_vtt: bool,
    pub output_json: bool,
    pub output_csv: bool,
    pub output_lrc: bool,
    pub karaoke: bool,
    pub output_wts: bool,
    pub font_path: String,
    pub input_file: String,
    pub diarize: bool,
}

impl Default for OutputFormatOptions {
    fn default() -> Self {
        Self {
            output_txt: false,
            output_srt: false,
            output_vtt: false,
            output_json: false,
            output_csv: false,
            output_lrc: false,
            karaoke: false,
            output_wts: false,
            font_path: "/System/Library/Fonts/Supplemental/Courier New Bold.ttf".to_string(),
            input_file: "audio.wav".to_string(),
            diarize: false,
        }
    }
}

impl OutputFormatOptions {
    /// Collect active formatters in stable order.
    pub fn active_formatters(&self) -> Vec<Box<dyn OutputFormatter>> {
        let mut list: Vec<Box<dyn OutputFormatter>> = Vec::new();
        if self.output_txt {
            list.push(Box::new(TxtFormatter));
        }
        if self.output_srt {
            list.push(Box::new(SrtFormatter {
                diarize: self.diarize,
            }));
        }
        if self.output_vtt {
            list.push(Box::new(VttFormatter {
                diarize: self.diarize,
            }));
        }
        if self.output_json {
            list.push(Box::new(JsonFormatter {
                diarize: self.diarize,
            }));
        }
        if self.output_csv {
            list.push(Box::new(CsvFormatter {
                diarize: self.diarize,
            }));
        }
        if self.output_lrc {
            list.push(Box::new(LrcFormatter));
        }
        if self.output_wts {
            list.push(Box::new(WtsFormatter {
                font_path: self.font_path.clone(),
                input_file: if self.input_file.is_empty() {
                    "audio.wav".to_string()
                } else {
                    self.input_file.clone()
                },
            }));
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
    #[cfg(unix)]
    if formatter.file_extension().ends_with(".sh") {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(&path) {
            let mut perms = metadata.permissions();
            perms.set_mode(perms.mode() | 0o755);
            let _ = std::fs::set_permissions(&path, perms);
        }
    }

    Ok(path)
}

/// Quantize 24-bit RGB values to xterm 256-color palette code [16, 256).
pub fn rgb2xterm256(r: i32, g: i32, b: i32) -> u8 {
    let uncube = |x: i32| -> i32 {
        if x < 48 {
            0
        } else if x < 115 {
            1
        } else {
            (x - 35) / 40
        }
    };
    let cube = [0, 95, 135, 175, 215, 255];
    let av = ((r as f64 * 0.299) + (g as f64 * 0.587) + (b as f64 * 0.114) + 0.5) as i32;
    let il = if av > 238 { 23 } else { (av - 3) / 10 };
    let ql = il * 10 + 8;
    let ir = uncube(r);
    let ig = uncube(g);
    let ib = uncube(b);
    let qr = cube[ir as usize];
    let qg = cube[ig as usize];
    let qb = cube[ib as usize];

    let sqr = |x: i32| x * x;
    if sqr(qr - r) + sqr(qg - g) + sqr(qb - b) <= sqr(ql - r) + sqr(ql - g) + sqr(ql - b) {
        (ir * 36 + ig * 6 + ib + 16) as u8
    } else {
        (il + 232) as u8
    }
}

/// Compute 256-color ANSI code based on token probability $p \in [0.0, 1.0]$.
/// Maps continuously from red ($p \approx 0$) through yellow ($p \approx 0.5$) to green ($p \approx 1.0$).
pub fn color_for_prob(p: f32) -> u8 {
    let p = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
    let (r, g, b) = if p <= 0.5 {
        let t = p / 0.5;
        let r = (220.0 + t * (247.0 - 220.0)).round() as i32;
        let g = (5.0 + t * (240.0 - 5.0)).round() as i32;
        let b = (12.0 + t * (86.0 - 12.0)).round() as i32;
        (r, g, b)
    } else {
        let t = (p - 0.5) / 0.5;
        let r = (247.0 - t * (247.0 - 78.0)).round() as i32;
        let g = (240.0 - t * (240.0 - 178.0)).round() as i32;
        let b = (86.0 + t * (101.0 - 86.0)).round() as i32;
        (r, g, b)
    };
    rgb2xterm256(r, g, b)
}

/// Select ANSI text style code based on confidence probability:
/// - $p < 0.33$: inverse video / highlight (`\x1b[7m`)
/// - $0.33 \le p < 0.66$: underline (`\x1b[4m`)
/// - $p \ge 0.66$: dim (`\x1b[2m`)
pub fn confidence_style(p: f32) -> &'static str {
    if p < 0.33 {
        "\x1b[7m"
    } else if p < 0.66 {
        "\x1b[4m"
    } else {
        "\x1b[2m"
    }
}

/// Format text token with 256-color ANSI escape sequences.
pub fn format_colored_token(text: &str, p: f32) -> String {
    let color = color_for_prob(p);
    format!("\x1b[38;5;{}m{}\x1b[0m", color, text)
}

/// Format text token with confidence ANSI style (dim, underline, or highlight).
pub fn format_confidence_token(text: &str, p: f32) -> String {
    let style = confidence_style(p);
    format!("{}{}\x1b[0m", style, text)
}

/// Format segment words with confidence colors.
pub fn format_segment_colors(seg: &Segment) -> String {
    if !seg.words.is_empty() {
        seg.words
            .iter()
            .map(|w| format_colored_token(w.word.trim(), w.probability))
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        let p = (-seg.avg_logprob.abs()).exp().clamp(0.0, 1.0);
        let words: Vec<&str> = seg.text.split_whitespace().collect();
        words
            .into_iter()
            .map(|w| format_colored_token(w, p))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Format segment words with confidence text styles (highlight/underline/dim).
pub fn format_segment_confidence(seg: &Segment) -> String {
    if !seg.words.is_empty() {
        seg.words
            .iter()
            .map(|w| format_confidence_token(w.word.trim(), w.probability))
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        let p = (-seg.avg_logprob.abs()).exp().clamp(0.0, 1.0);
        let words: Vec<&str> = seg.text.split_whitespace().collect();
        words
            .into_iter()
            .map(|w| format_confidence_token(w, p))
            .collect::<Vec<_>>()
            .join(" ")
    }
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

    #[test]
    fn test_wts_formatter_generation() {
        let words = vec![
            whisper_rs::WordTiming::new("Hello", 0.0, 0.5, 0.95),
            whisper_rs::WordTiming::new("world!", 0.5, 1.2, 0.98),
        ];
        let seg = Segment::with_words(0.0, 1.2, "Hello world!".to_string(), vec![1, 2], words);
        let formatter = WtsFormatter {
            font_path: "/test/font.ttf".to_string(),
            input_file: "test_audio.wav".to_string(),
        };

        let mut buf = Vec::new();
        formatter
            .write_segments(&[seg], &mut buf)
            .expect("write wts");
        let script = String::from_utf8(buf).expect("utf8 script");

        assert!(script.starts_with("#!/bin/bash\n"));
        assert!(script.contains("ffmpeg -i test_audio.wav"));
        assert!(script.contains("fontfile='/test/font.ttf'"));
        assert!(script.contains("fontcolor=lightgreen"));
        assert!(script.contains("text='> Hello world!'"));
        assert!(script.contains("test_audio.wav.mp4"));
    }

    #[test]
    fn test_color_gradient_probabilities() {
        // p ≈ 0 -> red (160)
        assert_eq!(color_for_prob(0.0), 160);
        // p ≈ 0.5 -> yellow (227)
        assert_eq!(color_for_prob(0.5), 227);
        // p ≈ 1.0 -> green (71)
        assert_eq!(color_for_prob(1.0), 71);

        let colored_low = format_colored_token("low", 0.0);
        assert!(colored_low.contains("\x1b[38;5;160mlow\x1b[0m"));

        let colored_mid = format_colored_token("mid", 0.5);
        assert!(colored_mid.contains("\x1b[38;5;227mmid\x1b[0m"));

        let colored_high = format_colored_token("high", 1.0);
        assert!(colored_high.contains("\x1b[38;5;71mhigh\x1b[0m"));
    }

    #[test]
    fn test_confidence_styles() {
        assert_eq!(confidence_style(0.1), "\x1b[7m");
        assert_eq!(confidence_style(0.5), "\x1b[4m");
        assert_eq!(confidence_style(0.9), "\x1b[2m");

        let token_low = format_confidence_token("low", 0.1);
        assert_eq!(token_low, "\x1b[7mlow\x1b[0m");

        let token_mid = format_confidence_token("mid", 0.5);
        assert_eq!(token_mid, "\x1b[4mmid\x1b[0m");

        let token_high = format_confidence_token("high", 0.9);
        assert_eq!(token_high, "\x1b[2mhigh\x1b[0m");
    }

    #[test]
    fn test_active_formatters_wts_selection() {
        let opts = OutputFormatOptions {
            output_wts: true,
            font_path: "/fonts/courier.ttf".to_string(),
            input_file: "speech.wav".to_string(),
            ..Default::default()
        };
        let active = opts.active_formatters();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name(), "Karaoke Video Script");
        assert_eq!(active[0].file_extension(), "wts.sh");
    }

    #[test]
    fn test_diarized_srt_formatting() {
        let segments = vec![
            Segment::new(0.0, 2.5, " Hello there! ".to_string(), vec![]).with_speaker("0"),
            Segment::new(2.5, 5.0, " General Kenobi! ".to_string(), vec![]).with_speaker("1"),
            Segment::new(5.0, 7.5, " Both speaking ".to_string(), vec![]).with_speaker("?"),
        ];
        let mut buf = Vec::new();
        SrtFormatter::new(true)
            .write_segments(&segments, &mut buf)
            .unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("1\n00:00:00,000 --> 00:00:02,500\n(speaker 0) Hello there!\n\n"));
        assert!(s.contains("2\n00:00:02,500 --> 00:00:05,000\n(speaker 1) General Kenobi!\n\n"));
        assert!(s.contains("3\n00:00:05,000 --> 00:00:07,500\n(speaker ?) Both speaking\n\n"));
    }

    #[test]
    fn test_diarized_vtt_formatting() {
        let segments = vec![
            Segment::new(0.0, 3.0, " First speaker ".to_string(), vec![]).with_speaker("0"),
            Segment::new(3.0, 6.0, " Second speaker ".to_string(), vec![]).with_speaker("1"),
        ];
        let mut buf = Vec::new();
        VttFormatter::new(true)
            .write_segments(&segments, &mut buf)
            .unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("WEBVTT\n\n"));
        assert!(s.contains("00:00:00.000 --> 00:00:03.000\n<v Speaker0>First speaker\n\n"));
        assert!(s.contains("00:00:03.000 --> 00:00:06.000\n<v Speaker1>Second speaker\n\n"));
    }

    #[test]
    fn test_diarized_csv_formatting() {
        let segments = vec![
            Segment::new(0.0, 1.5, " Intro ".to_string(), vec![]).with_speaker("0"),
            Segment::new(1.5, 3.25, " Response with \"quotes\" ".to_string(), vec![])
                .with_speaker("1"),
        ];
        let mut buf = Vec::new();
        CsvFormatter::new(true)
            .write_segments(&segments, &mut buf)
            .unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(
            lines[0],
            "start,end,speaker,text,avg_logprob,no_speech_prob"
        );
        assert_eq!(lines[1], "0,1500,0,\"Intro\",-0.2000,0.0100");
        assert_eq!(
            lines[2],
            "1500,3250,1,\"Response with \"\"quotes\"\"\",-0.2000,0.0100"
        );
    }

    #[test]
    fn test_diarized_json_formatting() {
        let segments = vec![
            Segment::new(0.0, 2.0, " JSON speaker test ".to_string(), vec![]).with_speaker("0"),
        ];
        let mut buf = Vec::new();
        JsonFormatter::new(true)
            .write_segments(&segments, &mut buf)
            .unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("\"speaker\": \"0\""));
        assert!(s.contains("\"start\": 0.000"));
        assert!(s.contains("\"end\": 2.000"));
        assert!(s.contains("\"text\": \"JSON speaker test\""));
    }

    #[test]
    fn test_active_formatters_with_diarize() {
        let opts = OutputFormatOptions {
            output_srt: true,
            output_vtt: true,
            output_json: true,
            output_csv: true,
            diarize: true,
            ..Default::default()
        };
        let active = opts.active_formatters();
        assert_eq!(active.len(), 4);
    }
}
