use crate::mel::dot_f32;
use crate::wav::WHISPER_SAMPLE_RATE;
use std::f64::consts::PI;

/// Default high-pass filter cutoff frequency in Hertz to suppress DC offset and low-frequency rumble.
pub const DEFAULT_HIGH_PASS_CUTOFF_HZ: f32 = 100.0;

/// Single-pole RC IIR high-pass filter matching whisper.cpp `examples/common.cpp:608-654`:
///
/// $$RC = \frac{1}{2\pi f_c}, \quad \alpha = \frac{\Delta t}{RC + \Delta t}, \quad y_i = \alpha(y_{i-1} + x_i - x_{i-1})$$
///
/// Filters `data` in-place to suppress DC offset and low-frequency rumble.
/// For empty data, single-element data, or invalid parameters ($f_c \le 0$ or $f_s \le 0$),
/// the function returns early leaving `data` unchanged.
pub fn high_pass_filter(data: &mut [f32], cutoff: f32, sample_rate: f32) {
    if data.len() <= 1 || cutoff <= 0.0 || sample_rate <= 0.0 {
        return;
    }
    let rc = 1.0f32 / (2.0f32 * std::f32::consts::PI * cutoff);
    let dt = 1.0f32 / sample_rate;
    let alpha = dt / (rc + dt);

    let mut y = data[0];
    let mut prev_x = data[0];
    for sample in data.iter_mut().skip(1) {
        let x = *sample;
        y = alpha * (y + x - prev_x);
        prev_x = x;
        *sample = y;
    }
}

/// Convenience function applying the single-pole RC IIR high-pass filter out-of-place.
pub fn high_pass_filtered(data: &[f32], cutoff: f32, sample_rate: f32) -> Vec<f32> {
    let mut out = data.to_vec();
    high_pass_filter(&mut out, cutoff, sample_rate);
    out
}

/// Precomputed polyphase windowed-sinc anti-aliasing filterbank for fast sample rate conversion.
///
/// Eliminates on-the-fly trigonometric evaluation by pre-tabulating 25-tap sinc filter rows
/// across discrete sub-sample phases.
pub struct PolyphaseSincFilter {
    src_rate: u32,
    dst_rate: u32,
    ratio: f64,
    fc: f64,
    num_phases: usize,
    /// Flat array of shape `[num_phases + 1, FILTER_LEN]` with pre-normalized weights.
    filter_bank: Vec<f32>,
}

impl PolyphaseSincFilter {
    pub const HALF_WINDOW: isize = 12;
    pub const FILTER_LEN: usize = (2 * Self::HALF_WINDOW + 1) as usize; // 25
    pub const DEFAULT_NUM_PHASES: usize = 512;

    /// Construct a polyphase sinc filterbank for downsampling from `src_rate` to `dst_rate`.
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        Self::with_phases(src_rate, dst_rate, Self::DEFAULT_NUM_PHASES)
    }

    /// Construct with a specified number of fractional sub-sample phases.
    pub fn with_phases(src_rate: u32, dst_rate: u32, num_phases: usize) -> Self {
        assert!(
            src_rate > dst_rate,
            "Polyphase filter is designed for downsampling"
        );
        assert!(num_phases > 0, "Number of phases must be greater than zero");

        let ratio = (dst_rate as f64) / (src_rate as f64);
        let fc = 0.5 * ratio;
        let half_window = Self::HALF_WINDOW;
        let filter_len = Self::FILTER_LEN;

        let mut filter_bank = Vec::with_capacity((num_phases + 1) * filter_len);

        for p in 0..=num_phases {
            let phase = -0.5 + (p as f64) / (num_phases as f64);
            let mut row = [0.0f64; Self::FILTER_LEN];
            let mut sum_w = 0.0f64;

            for (idx, k) in (-half_window..=half_window).enumerate() {
                let delta = phase - (k as f64);
                let x = 2.0 * fc * delta;
                let sinc = if x.abs() < 1e-7 {
                    1.0
                } else {
                    (PI * x).sin() / (PI * x)
                };
                let w = 0.5 + 0.5 * (PI * delta / (half_window as f64)).cos();
                let weight = 2.0 * fc * sinc * w;
                row[idx] = weight;
                sum_w += weight;
            }

            let inv_sum = if sum_w.abs() > 1e-9 { 1.0 / sum_w } else { 1.0 };
            for val in row {
                filter_bank.push((val * inv_sum) as f32);
            }
        }

        Self {
            src_rate,
            dst_rate,
            ratio,
            fc,
            num_phases,
            filter_bank,
        }
    }

    pub fn fc(&self) -> f64 {
        self.fc
    }

    pub fn src_rate(&self) -> u32 {
        self.src_rate
    }

    pub fn dst_rate(&self) -> u32 {
        self.dst_rate
    }

    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    pub fn num_phases(&self) -> usize {
        self.num_phases
    }

    /// Resample audio signal to the configured destination rate.
    pub fn resample(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let target_len = ((samples.len() as f64) * self.ratio).round() as usize;
        if target_len == 0 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(target_len);
        let n_samples = samples.len() as isize;
        let half_window = Self::HALF_WINDOW;
        let filter_len = Self::FILTER_LEN;
        let num_phases = self.num_phases;
        let num_phases_f = num_phases as f64;

        // When src_rate is an exact integer multiple of dst_rate (e.g. 48000 -> 16000),
        // src_pos is always an exact integer (phase == 0.0) for every output sample.
        let is_exact_integer_downsample = self.src_rate.is_multiple_of(self.dst_rate);

        for i in 0..target_len {
            let src_pos = (i as f64) / self.ratio;
            let center_idx = src_pos.round() as isize;
            let phase = src_pos - (center_idx as f64);

            let min_j = center_idx - half_window;
            let max_j = center_idx + half_window;

            let mut interp_w = [0.0f32; Self::FILTER_LEN];
            if is_exact_integer_downsample {
                let p0 = num_phases / 2;
                let offset = p0 * filter_len;
                interp_w.copy_from_slice(&self.filter_bank[offset..offset + filter_len]);
            } else {
                let p_float = (phase + 0.5) * num_phases_f;
                let p0 = (p_float.floor() as usize).min(num_phases - 1);
                let p1 = p0 + 1;
                let frac = (p_float - (p0 as f64)) as f32;
                let off0 = p0 * filter_len;
                let off1 = p1 * filter_len;
                let w0 = &self.filter_bank[off0..off0 + filter_len];
                let w1 = &self.filter_bank[off1..off1 + filter_len];
                for k in 0..filter_len {
                    interp_w[k] = w0[k] + frac * (w1[k] - w0[k]);
                }
            }

            if min_j >= 0 && max_j < n_samples {
                // Bulk path: all 25 taps fall within bounds.
                // Filter weights are pre-normalized so sum(interp_w) == 1.0.
                let in_slice = &samples[min_j as usize..(min_j as usize) + filter_len];
                let val = dot_f32(&interp_w, in_slice);
                out.push(val);
            } else {
                // Boundary path: near the edges where some taps are out of range.
                let mut sum_val = 0.0f64;
                let mut sum_w = 0.0f64;
                for (k_idx, k) in (-half_window..=half_window).enumerate() {
                    let j = center_idx + k;
                    if j >= 0 && j < n_samples {
                        let w = interp_w[k_idx] as f64;
                        sum_val += w * (samples[j as usize] as f64);
                        sum_w += w;
                    }
                }
                let sample = if sum_w.abs() > 1e-9 {
                    (sum_val / sum_w) as f32
                } else {
                    0.0
                };
                out.push(sample);
            }
        }

        out
    }
}

static RESAMPLER_48K: std::sync::LazyLock<PolyphaseSincFilter> =
    std::sync::LazyLock::new(|| PolyphaseSincFilter::new(48000, WHISPER_SAMPLE_RATE));

static RESAMPLER_44_1K: std::sync::LazyLock<PolyphaseSincFilter> =
    std::sync::LazyLock::new(|| PolyphaseSincFilter::new(44100, WHISPER_SAMPLE_RATE));

/// Resample audio signal to 16 kHz.
/// Uses precomputed polyphase windowed-sinc anti-aliasing interpolation for downsampling (e.g. 44.1 kHz, 48 kHz),
/// and linear midpoint interpolation for upsampling (e.g. 8 kHz).
pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == WHISPER_SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }

    let target_rate = WHISPER_SAMPLE_RATE as f64;
    let ratio = target_rate / (src_rate as f64);
    let target_len = ((samples.len() as f64) * ratio).round() as usize;

    if target_len == 0 {
        return Vec::new();
    }

    if ratio >= 1.0 {
        // Upsampling path: linear interpolation preserves exact midpoint ramp values
        let mut out = Vec::with_capacity(target_len);
        for i in 0..target_len {
            let src_idx = (i as f64) / ratio;
            let idx_floor = src_idx.floor() as usize;
            let frac = (src_idx - (idx_floor as f64)) as f32;

            if idx_floor + 1 < samples.len() {
                let s0 = samples[idx_floor];
                let s1 = samples[idx_floor + 1];
                out.push(s0 + frac * (s1 - s0));
            } else if idx_floor < samples.len() {
                out.push(samples[idx_floor]);
            } else {
                out.push(0.0);
            }
        }
        return out;
    }

    // Downsampling path: Polyphase Precomputed Windowed Sinc Resampler
    match src_rate {
        48000 => RESAMPLER_48K.resample(samples),
        44100 => RESAMPLER_44_1K.resample(samples),
        _ => PolyphaseSincFilter::new(src_rate, WHISPER_SAMPLE_RATE).resample(samples),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_pass_filter_edges() {
        // Empty slice
        let mut empty = vec![];
        high_pass_filter(&mut empty, 100.0, 16000.0);
        assert!(empty.is_empty());

        // Single sample
        let mut single = vec![42.0];
        high_pass_filter(&mut single, 100.0, 16000.0);
        assert_eq!(single, vec![42.0]);

        // Non-positive cutoff or sample rate does nothing
        let mut data = vec![1.0, 2.0, 3.0];
        high_pass_filter(&mut data, 0.0, 16000.0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        high_pass_filter(&mut data, -50.0, 16000.0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        high_pass_filter(&mut data, 100.0, 0.0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        high_pass_filter(&mut data, 100.0, -16000.0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_high_pass_filter_dc_suppression() {
        let sample_rate = 16000.0;
        let mut dc_signal = vec![1.5f32; 16000]; // 1 second of DC offset
        high_pass_filter(&mut dc_signal, DEFAULT_HIGH_PASS_CUTOFF_HZ, sample_rate);

        // First sample is unchanged
        assert_eq!(dc_signal[0], 1.5);

        // After 100 ms (1600 samples), DC offset should be strongly suppressed (< 0.1)
        assert!(
            dc_signal[1600].abs() < 0.1,
            "DC offset not suppressed: {}",
            dc_signal[1600]
        );

        // By the end of 1s, it should be heavily attenuated
        assert!(dc_signal.last().unwrap().abs() < 0.06);
    }

    #[test]
    fn test_high_pass_filter_frequency_response() {
        let sample_rate = 16000.0;
        let n_samples = 16000;
        let cutoff = 100.0;

        let mut low_freq = Vec::with_capacity(n_samples);
        let mut high_freq = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f32 / sample_rate;
            low_freq.push((2.0 * std::f32::consts::PI * 20.0 * t).sin()); // 20 Hz
            high_freq.push((2.0 * std::f32::consts::PI * 1000.0 * t).sin()); // 1000 Hz
        }

        let filtered_low = high_pass_filtered(&low_freq, cutoff, sample_rate);
        let filtered_high = high_pass_filtered(&high_freq, cutoff, sample_rate);

        // Measure RMS over the second half
        let half = n_samples / 2;
        let low_rms: f32 =
            (filtered_low[half..].iter().map(|&x| x * x).sum::<f32>() / half as f32).sqrt();
        let high_rms: f32 =
            (filtered_high[half..].iter().map(|&x| x * x).sum::<f32>() / half as f32).sqrt();

        // 20 Hz is attenuated compared to 1000 Hz
        assert!(
            low_rms < high_rms * 0.35,
            "Expected low frequency to be attenuated: low={}, high={}",
            low_rms,
            high_rms
        );
    }

    #[test]
    fn test_polyphase_resampler_properties() {
        let resampler = PolyphaseSincFilter::new(48000, 16000);
        assert_eq!(resampler.src_rate(), 48000);
        assert_eq!(resampler.dst_rate(), 16000);
        assert!((resampler.ratio() - 1.0 / 3.0).abs() < 1e-9);
        assert!((resampler.fc() - 1.0 / 6.0).abs() < 1e-9);
        assert_eq!(resampler.num_phases(), 512);

        // Constant amplitude preservation
        let input = vec![0.75f32; 4800];
        let out = resampler.resample(&input);
        assert_eq!(out.len(), 1600);
        for &val in &out {
            assert!((val - 0.75).abs() < 1e-5);
        }
    }
}
