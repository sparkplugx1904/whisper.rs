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

/// Global precomputed Hann window table for Whisper's N_FFT = 400.
/// Precomputed once via `LazyLock` so zero heap allocations and zero trigonometric calls occur.
pub static HANN_WINDOW_400: std::sync::LazyLock<[f32; 400]> = std::sync::LazyLock::new(|| {
    let mut table = [0.0f32; 400];
    for (i, val) in table.iter_mut().enumerate() {
        *val = 0.5 * (1.0 - ((2.0 * PI * (i as f32)) / 400.0).cos());
    }
    table
});

/// Returns a reference to the static precomputed 400-element Hann window table.
#[inline(always)]
pub fn hann_window_400() -> &'static [f32; 400] {
    &HANN_WINDOW_400
}

/// Compute or retrieve a Hann window of the specified length.
/// For length 400 (standard Whisper frame size), returns the cached static table converted to Vec.
pub fn hann_window(length: usize) -> Vec<f32> {
    if length == 400 {
        HANN_WINDOW_400.to_vec()
    } else {
        let mut out = vec![0.0f32; length];
        for (i, val) in out.iter_mut().enumerate() {
            *val = 0.5 * (1.0 - ((2.0 * PI * (i as f32)) / (length as f32)).cos());
        }
        out
    }
}

/// Vectorized floating-point dot product between slices `a` and `b`.
/// Dispatches to AVX2/FMA on x86_64, NEON on aarch64, or an 8-way unrolled scalar fallback.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_f32 vector lengths must match");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2_fma(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_f32_neon(a, b) };
    }

    #[allow(unreachable_code)]
    dot_f32_scalar(a, b)
}

/// Scalar 8-way unrolled dot product fallback.
#[inline]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        acc0 += a[base] * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
        acc4 += a[base + 4] * b[base + 4];
        acc5 += a[base + 5] * b[base + 5];
        acc6 += a[base + 6] * b[base + 6];
        acc7 += a[base + 7] * b[base + 7];
    }

    let mut sum = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
    for i in (chunks * 8)..len {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let len = a.len();
    let mut p_a = a.as_ptr();
    let mut p_b = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let n_blocks = len / 32;
    for _ in 0..n_blocks {
        let va0 = _mm256_loadu_ps(p_a);
        let vb0 = _mm256_loadu_ps(p_b);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(p_a.add(8));
        let vb1 = _mm256_loadu_ps(p_b.add(8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        let va2 = _mm256_loadu_ps(p_a.add(16));
        let vb2 = _mm256_loadu_ps(p_b.add(16));
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

        let va3 = _mm256_loadu_ps(p_a.add(24));
        let vb3 = _mm256_loadu_ps(p_b.add(24));
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);

        p_a = p_a.add(32);
        p_b = p_b.add(32);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    let rem_blocks = (len % 32) / 8;
    for _ in 0..rem_blocks {
        let va = _mm256_loadu_ps(p_a);
        let vb = _mm256_loadu_ps(p_b);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        p_a = p_a.add(8);
        p_b = p_b.add(8);
    }

    // Horizontal reduction of acc0 (8 x f32)
    let hi128 = _mm256_extractf128_ps(acc0, 1);
    let lo128 = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(final_sum);

    let tail_start = (len / 8) * 8;
    for i in tail_start..len {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let len = a.len();
    let mut p_a = a.as_ptr();
    let mut p_b = b.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let n_blocks = len / 16;
    for _ in 0..n_blocks {
        let va0 = vld1q_f32(p_a);
        let vb0 = vld1q_f32(p_b);
        acc0 = vfmaq_f32(acc0, va0, vb0);

        let va1 = vld1q_f32(p_a.add(4));
        let vb1 = vld1q_f32(p_b.add(4));
        acc1 = vfmaq_f32(acc1, va1, vb1);

        let va2 = vld1q_f32(p_a.add(8));
        let vb2 = vld1q_f32(p_b.add(8));
        acc2 = vfmaq_f32(acc2, va2, vb2);

        let va3 = vld1q_f32(p_a.add(12));
        let vb3 = vld1q_f32(p_b.add(12));
        acc3 = vfmaq_f32(acc3, va3, vb3);

        p_a = p_a.add(16);
        p_b = p_b.add(16);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    let rem_blocks = (len % 16) / 4;
    for _ in 0..rem_blocks {
        let va = vld1q_f32(p_a);
        let vb = vld1q_f32(p_b);
        acc0 = vfmaq_f32(acc0, va, vb);
        p_a = p_a.add(4);
        p_b = p_b.add(4);
    }

    let mut sum = vaddvq_f32(acc0);
    let tail_start = (len / 4) * 4;
    for i in tail_start..len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    sum
}

/// Project FFT power bins into mel filter bands and compute log-energy.
/// Vectorizes across bins and mel rows with SIMD AVX2/FMA when available.
#[inline]
pub fn project_mel_frame(power: &[f32], filters: &[f32], n_mel: usize, frame_out: &mut [f32]) {
    let n_bins = power.len();
    assert_eq!(filters.len(), n_mel * n_bins);
    assert_eq!(frame_out.len(), n_mel);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                project_mel_frame_avx2_fma(power, filters, n_mel, frame_out);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            project_mel_frame_neon(power, filters, n_mel, frame_out);
        }
        return;
    }

    #[allow(unreachable_code)]
    project_mel_frame_scalar(power, filters, n_mel, frame_out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn project_mel_frame_avx2_fma(
    power: &[f32],
    filters: &[f32],
    n_mel: usize,
    frame_out: &mut [f32],
) {
    let n_bins = power.len();
    for j in 0..n_mel {
        let filter_row = filters.get_unchecked(j * n_bins..(j + 1) * n_bins);
        let sum = dot_f32_avx2_fma(power, filter_row);
        *frame_out.get_unchecked_mut(j) = (sum.max(1e-10)).log10();
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn project_mel_frame_neon(
    power: &[f32],
    filters: &[f32],
    n_mel: usize,
    frame_out: &mut [f32],
) {
    let n_bins = power.len();
    for j in 0..n_mel {
        let filter_row = filters.get_unchecked(j * n_bins..(j + 1) * n_bins);
        let sum = dot_f32_neon(power, filter_row);
        *frame_out.get_unchecked_mut(j) = (sum.max(1e-10)).log10();
    }
}

#[inline]
fn project_mel_frame_scalar(power: &[f32], filters: &[f32], n_mel: usize, frame_out: &mut [f32]) {
    let n_bins = power.len();
    for j in 0..n_mel {
        let filter_row = &filters[j * n_bins..(j + 1) * n_bins];
        let sum = dot_f32_scalar(power, filter_row);
        frame_out[j] = (sum.max(1e-10)).log10();
    }
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

    let hann = hann_window_400();
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
    // Calculate original frame length matching whisper.cpp:
    // (n_samples + stage_2_pad - frame_size) / frame_step + 1
    // For 40 <= n < 200 samples, ensure n_len_org evaluates to at least 1,
    // while for n < 40 (such as empty audio), n_len_org is 0.
    let n_len_org = if n_samples >= 40 {
        let raw = (n_samples as isize + stage_2_pad as isize - frame_size as isize)
            / frame_step as isize
            + 1;
        raw.max(1) as usize
    } else {
        0
    };
    let mut mel_data = vec![0.0f32; n_mel * n_len];

    let n_samples_stage2 = n_samples + stage_2_pad;
    let active_frames = ((n_samples_stage2 / frame_step) + 1).min(n_len);

    const N_FFT_BINS: usize = 1 + N_FFT / 2;

    let fft = Fft400::shared();

    let active_mel: Vec<Vec<f32>> = (0..active_frames)
        .into_par_iter()
        .map(|i| {
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

            project_mel_frame(&power, filters, n_mel, &mut frame_out);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_cache() {
        let cached = hann_window_400();
        assert_eq!(cached.len(), 400);

        // Boundary value: Hann window at i=0 should be 0.0
        assert_eq!(cached[0], 0.0);

        // Symmetry: h[i] == h[400 - i]
        for i in 1..200 {
            assert!(
                (cached[i] - cached[400 - i]).abs() < 1e-6,
                "Hann window asymmetry at index {}",
                i
            );
        }

        // Peak around center (index 200) should be near 1.0
        assert!((cached[200] - 1.0).abs() < 1e-5);

        // hann_window(400) matches cached table
        let dynamically_fetched = hann_window(400);
        assert_eq!(dynamically_fetched.as_slice(), cached.as_slice());

        // Dynamic sizes work
        let w100 = hann_window(100);
        assert_eq!(w100.len(), 100);
        assert_eq!(w100[0], 0.0);
    }

    #[test]
    fn test_dot_f32_accuracy_and_fallbacks() {
        let len = 201; // Whisper's exact FFT bin count
        let mut a = vec![0.0f32; len];
        let mut b = vec![0.0f32; len];

        for i in 0..len {
            a[i] = (i as f32 * 0.17).sin();
            b[i] = (i as f32 * 0.31).cos();
        }

        let scalar_result = dot_f32_scalar(&a, &b);
        let simd_result = dot_f32(&a, &b);

        assert!(
            (scalar_result - simd_result).abs() < 1e-4,
            "SIMD dot ({}) diverged from scalar ({})",
            simd_result,
            scalar_result
        );

        // Test with varying lengths
        for size in [1, 7, 8, 15, 16, 25, 31, 32, 63, 64, 201] {
            let sub_a = &a[..size];
            let sub_b = &b[..size];
            let s = dot_f32_scalar(sub_a, sub_b);
            let d = dot_f32(sub_a, sub_b);
            assert!(
                (s - d).abs() < 1e-4,
                "Mismatch at size {}: scalar={}, dot={}",
                size,
                s,
                d
            );
        }
    }

    #[test]
    fn test_project_mel_frame() {
        let n_bins = 201;
        let n_mel = 80;
        let power = vec![0.5f32; n_bins];
        let filters = vec![0.01f32; n_mel * n_bins];

        let mut frame_out = vec![0.0f32; n_mel];
        project_mel_frame(&power, &filters, n_mel, &mut frame_out);

        // Each bin: sum = 201 * 0.5 * 0.01 = 1.005
        let expected = 1.005f32.log10();
        for (idx, &val) in frame_out.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-4,
                "Band {} expected {}, got {}",
                idx,
                expected,
                val
            );
        }

        // Test clamping on silent input
        let zero_power = vec![0.0f32; n_bins];
        project_mel_frame(&zero_power, &filters, n_mel, &mut frame_out);
        let silent_expected = 1e-10f32.log10();
        for &val in &frame_out {
            assert_eq!(val, silent_expected);
        }
    }
}
