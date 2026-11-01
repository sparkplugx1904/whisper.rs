use crate::wav::WHISPER_SAMPLE_RATE;

pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == WHISPER_SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }

    let target_rate = WHISPER_SAMPLE_RATE as f64;
    let ratio = target_rate / (src_rate as f64);
    let target_len = ((samples.len() as f64) * ratio).round() as usize;

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

    out
}
