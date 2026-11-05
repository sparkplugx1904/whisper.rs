use crate::dtw::WordTiming;

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub text: String,
    pub tokens: Vec<i32>,
    pub words: Vec<WordTiming>,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub avg_entropy: f32,
}

impl Segment {
    pub fn new(t0: f64, t1: f64, text: String, tokens: Vec<i32>) -> Self {
        Self {
            t0,
            t1,
            text,
            tokens,
            words: Vec::new(),
            avg_logprob: -0.2,
            no_speech_prob: 0.01,
            avg_entropy: 0.5,
        }
    }

    pub fn with_words(
        t0: f64,
        t1: f64,
        text: String,
        tokens: Vec<i32>,
        words: Vec<WordTiming>,
    ) -> Self {
        Self {
            t0,
            t1,
            text,
            tokens,
            words,
            avg_logprob: -0.2,
            no_speech_prob: 0.01,
            avg_entropy: 0.5,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_stats(
        t0: f64,
        t1: f64,
        text: String,
        tokens: Vec<i32>,
        words: Vec<WordTiming>,
        avg_logprob: f32,
        no_speech_prob: f32,
        avg_entropy: f32,
    ) -> Self {
        Self {
            t0,
            t1,
            text,
            tokens,
            words,
            avg_logprob,
            no_speech_prob,
            avg_entropy,
        }
    }

    /// Format as SRT subtitle entry (delegates to `format_srt`).
    pub fn to_srt(&self, index: usize) -> String {
        format_srt(self, index)
    }

    /// Format as VTT subtitle entry (delegates to `format_vtt`).
    pub fn to_vtt(&self) -> String {
        format_vtt(self)
    }

    /// Format as Karaoke WebVTT subtitle entry with word-level timing tags `<HH:MM:SS.mmm>word`.
    pub fn to_karaoke_vtt(&self) -> String {
        format_karaoke_vtt(self)
    }

    /// Format as Karaoke SubRip subtitle entry with word-level timing tags `<HH:MM:SS,mmm>word`.
    pub fn to_karaoke_srt(&self, index: usize) -> String {
        format_karaoke_srt(self, index)
    }

    /// Format as JSON segment object with `start`, `end`, and `text`.
    pub fn to_json(&self) -> String {
        format_json(self)
    }
}

/// Format a timestamp in seconds to `HH:MM:SS,mmm` (SRT style).
fn fmt_time_srt(secs: f64) -> String {
    let s_clamped = if secs.is_finite() && secs >= 0.0 {
        secs
    } else {
        0.0
    };
    let total_ms = (s_clamped * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

/// Format a timestamp in seconds to `HH:MM:SS.mmm` (VTT style).
fn fmt_time_vtt(secs: f64) -> String {
    let s_clamped = if secs.is_finite() && secs >= 0.0 {
        secs
    } else {
        0.0
    };
    let total_ms = (s_clamped * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

/// Render segment as SRT subtitle string.
pub fn format_srt(segment: &Segment, index: usize) -> String {
    format!(
        "{}\n{} --> {}\n{}\n\n",
        index,
        fmt_time_srt(segment.t0),
        fmt_time_srt(segment.t1),
        segment.text.trim()
    )
}

/// Render segment as VTT subtitle string.
pub fn format_vtt(segment: &Segment) -> String {
    format!(
        "{} --> {}\n{}\n\n",
        fmt_time_vtt(segment.t0),
        fmt_time_vtt(segment.t1),
        segment.text.trim()
    )
}

/// Render segment as Karaoke VTT string.
pub fn format_karaoke_vtt(segment: &Segment) -> String {
    let mut body = String::new();
    if segment.words.is_empty() {
        body.push_str(segment.text.trim());
    } else {
        for (i, w) in segment.words.iter().enumerate() {
            if i > 0 {
                body.push(' ');
            }
            body.push_str(&format!("<{}>{}", fmt_time_vtt(w.t0), w.word));
        }
    }

    format!(
        "{} --> {}\n{}\n\n",
        fmt_time_vtt(segment.t0),
        fmt_time_vtt(segment.t1),
        body
    )
}

/// Render segment as Karaoke SRT string.
pub fn format_karaoke_srt(segment: &Segment, index: usize) -> String {
    let mut body = String::new();
    if segment.words.is_empty() {
        body.push_str(segment.text.trim());
    } else {
        for (i, w) in segment.words.iter().enumerate() {
            if i > 0 {
                body.push(' ');
            }
            body.push_str(&format!("<{}>{}", fmt_time_srt(w.t0), w.word));
        }
    }

    format!(
        "{}\n{} --> {}\n{}\n\n",
        index,
        fmt_time_srt(segment.t0),
        fmt_time_srt(segment.t1),
        body
    )
}

/// Render segment as JSON string.
pub fn format_json(segment: &Segment) -> String {
    let t0 = if segment.t0.is_finite() && segment.t0 >= 0.0 {
        segment.t0
    } else {
        0.0
    };
    let t1 = if segment.t1.is_finite() && segment.t1 >= 0.0 {
        segment.t1
    } else {
        0.0
    };
    if segment.words.is_empty() {
        format!(
            "{{\"start\": {:.3}, \"end\": {:.3}, \"text\": \"{}\", \"avg_logprob\": {:.4}, \"no_speech_prob\": {:.4}}}",
            t0,
            t1,
            escape_json(segment.text.trim()),
            segment.avg_logprob,
            segment.no_speech_prob
        )
    } else {
        let words_json = segment
            .words
            .iter()
            .map(|w| {
                format!(
                    "{{\"word\": \"{}\", \"start\": {:.3}, \"end\": {:.3}, \"probability\": {:.4}}}",
                    escape_json(&w.word),
                    w.t0,
                    w.t1,
                    w.probability
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{\"start\": {:.3}, \"end\": {:.3}, \"text\": \"{}\", \"avg_logprob\": {:.4}, \"no_speech_prob\": {:.4}, \"words\": [{}]}}",
            t0,
            t1,
            escape_json(segment.text.trim()),
            segment.avg_logprob,
            segment.no_speech_prob,
            words_json
        )
    }
}

/// Escape string for JSON serialization according to RFC 8259
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_timestamp_format() {
        let seg = Segment::new(65.123, 3661.004, " Hello World! ".to_string(), vec![]);
        let srt = seg.to_srt(1);
        assert_eq!(srt, "1\n00:01:05,123 --> 01:01:01,004\nHello World!\n\n");
    }

    #[test]
    fn test_vtt_timestamp_format() {
        let seg = Segment::new(65.123, 3661.004, " Hello World! ".to_string(), vec![]);
        let vtt = seg.to_vtt();
        assert_eq!(vtt, "00:01:05.123 --> 01:01:01.004\nHello World!\n\n");
    }

    #[test]
    fn test_json_format_and_escaping() {
        let seg = Segment::new(
            0.0,
            2.5,
            " He said \"Hello\\World\"\nNext line\tTab ".to_string(),
            vec![],
        );
        let json = seg.to_json();
        assert_eq!(
            json,
            r#"{"start": 0.000, "end": 2.500, "text": "He said \"Hello\\World\"\nNext line\tTab", "avg_logprob": -0.2000, "no_speech_prob": 0.0100}"#
        );
    }

    #[test]
    fn test_edge_case_timestamps() {
        let seg = Segment::new(-1.0, f64::NAN, "test".to_string(), vec![]);
        let srt = seg.to_srt(2);
        assert_eq!(srt, "2\n00:00:00,000 --> 00:00:00,000\ntest\n\n");
        let vtt = seg.to_vtt();
        assert_eq!(vtt, "00:00:00.000 --> 00:00:00.000\ntest\n\n");
        let json = seg.to_json();
        assert_eq!(
            json,
            r#"{"start": 0.000, "end": 0.000, "text": "test", "avg_logprob": -0.2000, "no_speech_prob": 0.0100}"#
        );
    }
}
