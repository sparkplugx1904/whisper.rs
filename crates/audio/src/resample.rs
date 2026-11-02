use crate::wav::WHISPER_SAMPLE_RATE;
use std::f64::consts::PI;

/// Resample audio signal to 16 kHz.
/// Uses band-limited windowed-sinc anti-aliasing interpolation for downsampling (e.g. 44.1 kHz, 48 kHz),
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

    // Downsampling path: Band-limited windowed-sinc anti-aliasing filter
    // Cutoff frequency fc: 0.5 * ratio prevents aliasing above target Nyquist (8 kHz)
    let fc = 0.5 * ratio;
    let half_window = 12isize;

    let mut out = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_pos = (i as f64) / ratio;
        let center_idx = src_pos.round() as isize;

        let mut sum_val = 0.0f64;
        let mut sum_weight = 0.0f64;

        for k in -half_window..=half_window {
            let j = center_idx + k;
            if j < 0 || j >= samples.len() as isize {
                continue;
            }

            let delta = src_pos - (j as f64);
            let x = 2.0 * fc * delta;
            let sinc = if x.abs() < 1e-7 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };

            // Hann window over [-half_window, half_window]
            let w = 0.5 + 0.5 * (PI * delta / (half_window as f64)).cos();
            let weight = 2.0 * fc * sinc * w;

            sum_val += weight * (samples[j as usize] as f64);
            sum_weight += weight;
        }

        let sample = if sum_weight.abs() > 1e-9 {
            (sum_val / sum_weight) as f32
        } else {
            0.0
        };

        out.push(sample);
    }

    out
}
