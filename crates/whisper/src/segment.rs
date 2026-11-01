#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub text: String,
    pub tokens: Vec<i32>,
}

impl Segment {
    pub fn new(t0: f64, t1: f64, text: String, tokens: Vec<i32>) -> Self {
        Self {
            t0,
            t1,
            text,
            tokens,
        }
    }

    /// Format as SRT subtitle entry
    pub fn to_srt(&self, index: usize) -> String {
        let fmt_time = |secs: f64| {
            let total_ms = (secs * 1000.0).round() as u64;
            let ms = total_ms % 1000;
            let total_s = total_ms / 1000;
            let s = total_s % 60;
            let total_m = total_s / 60;
            let m = total_m % 60;
            let h = total_m / 60;
            format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
        };
        format!(
            "{}\n{} --> {}\n{}\n",
            index,
            fmt_time(self.t0),
            fmt_time(self.t1),
            self.text.trim()
        )
    }

    /// Format as VTT subtitle entry
    pub fn to_vtt(&self) -> String {
        let fmt_time = |secs: f64| {
            let total_ms = (secs * 1000.0).round() as u64;
            let ms = total_ms % 1000;
            let total_s = total_ms / 1000;
            let s = total_s % 60;
            let total_m = total_s / 60;
            let m = total_m % 60;
            let h = total_m / 60;
            format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
        };
        format!(
            "{} --> {}\n{}\n",
            fmt_time(self.t0),
            fmt_time(self.t1),
            self.text.trim()
        )
    }
}
