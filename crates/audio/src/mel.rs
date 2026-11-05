use crate::fft::{Fft400, N_FFT};
use crate::wav::WHISPER_SAMPLE_RATE;
use rayon::prelude::*;
use std::f32::consts::PI;

pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;

pub struct MelSpectrogram {
    pub n_mel: usize,
    pub n_len: usize,
    pub n_len_org: usize,
    pub data: Vec<f32>,
}

impl MelSpectrogram {
    /// Access element at mel band `m` (0..n_mel) and frame `t` (0..n_len).
    pub fn get(&self, m: usize, t: usize) -> f32 {
        self.data[m * self.n_len + t]
    }

    /// Extract a [n_mel, 2 * n_ctx] chunk at a given frame offset for encoder consumption.
    pub fn get_chunk(&self, offset: usize, n_ctx: usize) -> Vec<f32> {
        let chunk_len = 2 * n_ctx;
        let mut dst = vec![0.0f32; self.n_mel * chunk_len];
        let i0 = offset.min(self.n_len);
        let i1 = (offset + chunk_len).min(self.n_len);
        let valid_len = i1.saturating_sub(i0);

        for j in 0..self.n_mel {
            let src_row = &self.data[j * self.n_len + i0..j * self.n_len + i0 + valid_len];
            let dst_row = &mut dst[j * chunk_len..j * chunk_len + valid_len];
            dst_row.copy_from_slice(src_row);
        }
        dst
    }
}

/// Generate Slaney triangular mel filterbank matrix of shape [n_mel, (n_fft / 2) + 1].
/// Implements standard Slaney scale with linear mapping under 1000 Hz and logarithmic mapping above,
/// with triangular area normalization.
pub fn generate_slaney_mel_filters(n_mel: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let fmin = 0.0f32;
    let fmax = (sample_rate as f32) / 2.0;
    let n_bins = (n_fft / 2) + 1;

    let hz_to_mel = |f: f32| -> f32 {
        if f < 1000.0 {
            f / (200.0 / 3.0)
        } else {
            15.0 + (f / 1000.0).ln() / (6.4f32.ln() / 27.0)
        }
    };

    let mel_to_hz = |m: f32| -> f32 {
        if m < 15.0 {
            m * (200.0 / 3.0)
        } else {
            1000.0 * ((m - 15.0) * (6.4f32.ln() / 27.0)).exp()
        }
    };

    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);

    let mut mel_f = Vec::with_capacity(n_mel + 2);
    for i in 0..(n_mel + 2) {
        let m = min_mel + (i as f32) * (max_mel - min_mel) / ((n_mel + 1) as f32);
        mel_f.push(mel_to_hz(m));
    }

    let mut fdiff = Vec::with_capacity(n_mel + 1);
    for i in 0..(n_mel + 1) {
        fdiff.push(mel_f[i + 1] - mel_f[i]);
    }

    let mut fftfreqs = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        fftfreqs.push((i as f32) * fmax / ((n_bins - 1) as f32));
    }

    let mut weights = vec![0.0f32; n_mel * n_bins];
    for i in 0..n_mel {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        let d_lower = fdiff[i];
        let d_upper = fdiff[i + 1];

        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / d_lower;
            let upper = (mel_f[i + 2] - f) / d_upper;
            let w = lower.min(upper).max(0.0) * enorm;
            weights[i * n_bins + j] = w;
        }
    }

    weights
}

pub fn hann_window(length: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; length];
    for (i, val) in out.iter_mut().enumerate() {
        *val = 0.5 * (1.0 - ((2.0 * PI * (i as f32)) / (length as f32)).cos());
    }
    out
}

pub fn compute_log_mel_spectrogram(
    samples: &[f32],
    mel_filters: &[f32],
    n_mel: usize,
) -> MelSpectrogram {
    let n_samples = samples.len();
    let frame_size = N_FFT; // 400
    let frame_step = HOP_LENGTH; // 160
    let n_fft_bins = 1 + frame_size / 2; // 201

    let generated_filters;
    let filters = if mel_filters.is_empty() {
        generated_filters =
            generate_slaney_mel_filters(n_mel, frame_size, WHISPER_SAMPLE_RATE as usize);
        &generated_filters[..]
    } else {
        assert_eq!(
            mel_filters.len(),
            n_mel * n_fft_bins,
            "Filter data size {} does not match n_mel ({}) * n_fft ({})",
            mel_filters.len(),
            n_mel,
            n_fft_bins
        );
        mel_filters
    };

    let hann = hann_window(frame_size);
    // Padding
    let stage_1_pad = (WHISPER_SAMPLE_RATE as usize) * 30; // 480000
    let stage_2_pad = frame_size / 2; // 200

    let mut samples_padded = vec![0.0f32; n_samples + stage_1_pad + stage_2_pad * 2];
    samples_padded[stage_2_pad..stage_2_pad + n_samples].copy_from_slice(samples);

    // Reflective pad at the beginning
    if n_samples > 1 {
        let n_reflect = stage_2_pad.min(n_samples - 1);
        let src = &samples[1..1 + n_reflect];
        for (i, &val) in src.iter().rev().enumerate() {
            samples_padded[stage_2_pad - n_reflect + i] = val;
        }
    }

    let n_len = (samples_padded.len() - frame_size) / frame_step;
    let n_len_org = if n_samples + stage_2_pad >= frame_size {
        1 + (n_samples + stage_2_pad - frame_size) / frame_step
    } else {
        0
    };
    let mut mel_data = vec![0.0f32; n_mel * n_len];

    let n_samples_stage2 = n_samples + stage_2_pad;
    let active_frames = ((n_samples_stage2 / frame_step) + 1).min(n_len);

    const N_FFT_BINS: usize = 1 + N_FFT / 2;

    let active_mel: Vec<Vec<f32>> = (0..active_frames)
        .into_par_iter()
        .map(|i| {
            let fft = Fft400::new();
            let mut fft_in = [0.0f32; N_FFT];
            let mut fft_out = [0.0f32; N_FFT * 2];
            let mut power = [0.0f32; N_FFT_BINS];
            let mut frame_out = vec![0.0f32; n_mel];

            let offset = i * frame_step;
            let valid_samples = if offset < n_samples_stage2 {
                N_FFT.min(n_samples_stage2 - offset)
            } else {
                0
            };

            for (j, &s) in samples_padded[offset..offset + valid_samples]
                .iter()
                .enumerate()
            {
                fft_in[j] = hann[j] * s;
            }
            fft_in[valid_samples..N_FFT].fill(0.0);
            fft.compute(&fft_in, &mut fft_out);

            for j in 0..N_FFT_BINS {
                let re = fft_out[2 * j];
                let im = fft_out[2 * j + 1];
                power[j] = re * re + im * im;
            }

            for j in 0..n_mel {
                let filter_row = &filters[j * N_FFT_BINS..(j + 1) * N_FFT_BINS];
                let mut sum = 0.0f32;
                for k in 0..N_FFT_BINS {
                    sum += power[k] * filter_row[k];
                }
                frame_out[j] = (sum.max(1e-10)).log10();
            }

            frame_out
        })
        .collect();

    for (i, frame) in active_mel.into_iter().enumerate() {
        for j in 0..n_mel {
            mel_data[j * n_len + i] = frame[j];
        }
    }

    // Frames beyond active audio
    let silent_val = 1e-10f32.log10();
    for i in active_frames..n_len {
        for j in 0..n_mel {
            mel_data[j * n_len + i] = silent_val;
        }
    }

    // Clamping and normalization
    let mut mmax = f32::NEG_INFINITY;
    for &val in &mel_data {
        if val > mmax {
            mmax = val;
        }
    }
    mmax -= 8.0;

    for val in &mut mel_data {
        if *val < mmax {
            *val = mmax;
        }
        *val = (*val + 4.0) / 4.0;
    }

    MelSpectrogram {
        n_mel,
        n_len,
        n_len_org,
        data: mel_data,
    }
}
