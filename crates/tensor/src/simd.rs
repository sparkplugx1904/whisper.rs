#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

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

#[inline(always)]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += a[base] * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
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

    let acc01 = _mm256_add_ps(acc0, acc1);
    let acc23 = _mm256_add_ps(acc2, acc3);
    let mut acc = _mm256_add_ps(acc01, acc23);

    // Remaining 8-float chunks
    let remaining_chunks = (len % 32) / 8;
    for _ in 0..remaining_chunks {
        let va = _mm256_loadu_ps(p_a);
        let vb = _mm256_loadu_ps(p_b);
        acc = _mm256_fmadd_ps(va, vb, acc);
        p_a = p_a.add(8);
        p_b = p_b.add(8);
    }

    // Horizontal sum of __m256 (8 floats)
    // acc = [f0, f1, f2, f3, f4, f5, f6, f7]
    let hi_128 = _mm256_extractf128_ps(acc, 1);
    let lo_128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo_128, hi_128); // [f0+f4, f1+f5, f2+f6, f3+f7]
    let shuf = _mm_movehl_ps(sum128, sum128); // [f2+f6, f3+f7, ...]
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    let mut result = _mm_cvtss_f32(sum32);

    // Tail scalar elements (0..7)
    let tail_offset = len - (len % 8);
    for i in tail_offset..len {
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

    let n_blocks = len / 8;
    for _ in 0..n_blocks {
        let va0 = vld1q_f32(p_a);
        let vb0 = vld1q_f32(p_b);
        acc0 = vmlaq_f32(acc0, va0, vb0);

        let va1 = vld1q_f32(p_a.add(4));
        let vb1 = vld1q_f32(p_b.add(4));
        acc1 = vmlaq_f32(acc1, va1, vb1);

        p_a = p_a.add(8);
        p_b = p_b.add(8);
    }

    let acc = vaddq_f32(acc0, acc1);
    let mut result = vaddvq_f32(acc);

    let tail_offset = n_blocks * 8;
    for i in tail_offset..len {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}
