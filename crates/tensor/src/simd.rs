use crate::ops::quant::Q8Activation;

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

    let mut result = hsum256_ps(acc);

    // Tail scalar elements (0..7)
    let tail_offset = len - (len % 8);
    for i in tail_offset..len {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn hsum256_ps(acc: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let hi_128 = _mm256_extractf128_ps(acc, 1);
    let lo_128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo_128, hi_128);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
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
/// # Safety
///
/// Requires target features AVX2 and FMA. Caller must ensure slices have matching lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_f32_avx2_fma(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    in_features: usize,
    out_features: usize,
) {
    let p_bias = bias.map(|b| b.as_ptr());
    let p_w = w.as_ptr();
    let p_out = out.as_mut_ptr();

    for r in 0..out_features {
        let wr_slice = std::slice::from_raw_parts(p_w.add(r * in_features), in_features);
        let mut sum = dot_f32_avx2_fma(x, wr_slice);
        if let Some(pb) = p_bias {
            sum += *pb.add(r);
        }
        *p_out.add(r) = sum;
    }
}

pub fn gemv_f32_scalar(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    in_features: usize,
    out_features: usize,
) {
    for r in 0..out_features {
        let w_row = &w[r * in_features..(r + 1) * in_features];
        let mut sum = dot_f32_scalar(x, w_row);
        if let Some(b) = bias {
            sum += b[r];
        }
        out[r] = sum;
    }
}

#[inline(always)]
pub fn gemv_f32(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    in_features: usize,
    out_features: usize,
) {
    assert_eq!(x.len(), in_features);
    assert_eq!(w.len(), out_features * in_features);
    assert_eq!(out.len(), out_features);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                gemv_f32_avx2_fma(x, w, bias, out, in_features, out_features);
            }
            return;
        }
    }

    gemv_f32_scalar(x, w, bias, out, in_features, out_features);
}

#[inline(always)]
pub fn dot_q8_0_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q8_0_f32_avx2(raw_blocks, x) };
        }
    }
    dot_q8_0_f32_scalar(raw_blocks, x)
}

#[inline(always)]
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn dot_q8_0_f32_scalar(raw_blocks: &[u8], x: &[f32]) -> f32 {
    use half::f16;

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(34).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let d = f16::from_bits(d_bits).to_f32();
        let qs = &chunk[2..34];
        let x_block = &x[i * 32..(i + 1) * 32];

        let mut block_sum = 0.0f32;
        for j in 0..32 {
            block_sum += (qs[j] as i8 as f32) * x_block[j];
        }
        sum += d * block_sum;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q8_0_f32_avx2(raw_blocks: &[u8], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    use half::f16;

    let n_blocks = x.len() / 32;
    let mut total_sum = 0.0f32;

    for i in 0..n_blocks {
        let chunk = raw_blocks.get_unchecked(i * 34..(i + 1) * 34);
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let d = f16::from_bits(d_bits).to_f32();

        let qs_ptr = chunk.as_ptr().add(2);
        let x_ptr = x.as_ptr().add(i * 32);

        let q16_lo = _mm_loadu_si128(qs_ptr as *const __m128i);
        let q16_hi = _mm_loadu_si128(qs_ptr.add(16) as *const __m128i);

        let q_lo16 = _mm256_cvtepi8_epi16(q16_lo);
        let q_hi16 = _mm256_cvtepi8_epi16(q16_hi);

        let lo8_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo16));
        let lo8_f32 = _mm256_cvtepi32_ps(lo8_i32);
        let x0 = _mm256_loadu_ps(x_ptr);
        let mut acc = _mm256_mul_ps(lo8_f32, x0);

        let hi8_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo16, 1));
        let hi8_f32 = _mm256_cvtepi32_ps(hi8_i32);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        acc = _mm256_fmadd_ps(hi8_f32, x1, acc);

        let lo8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi16));
        let lo8_hi_f32 = _mm256_cvtepi32_ps(lo8_hi_i32);
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        acc = _mm256_fmadd_ps(lo8_hi_f32, x2, acc);

        let hi8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi16, 1));
        let hi8_hi_f32 = _mm256_cvtepi32_ps(hi8_hi_i32);
        let x3 = _mm256_loadu_ps(x_ptr.add(24));
        acc = _mm256_fmadd_ps(hi8_hi_f32, x3, acc);

        let hi_128 = _mm256_extractf128_ps(acc, 1);
        let lo_128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo_128, hi_128);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, shuf2);
        let block_sum = _mm_cvtss_f32(sum32);

        total_sum += d * block_sum;
    }

    total_sum
}

#[inline(always)]
pub fn dot_q4_0_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q4_0_f32_avx2(raw_blocks, x) };
        }
    }
    dot_q4_0_f32_scalar(raw_blocks, x)
}

#[inline(always)]
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn dot_q4_0_f32_scalar(raw_blocks: &[u8], x: &[f32]) -> f32 {
    use half::f16;

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(18).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let d = f16::from_bits(d_bits).to_f32();
        let qs = &chunk[2..18];
        let x_block = &x[i * 32..(i + 1) * 32];

        let mut block_sum = 0.0f32;
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i8 - 8;
            let x1 = ((qs[j] >> 4) & 0x0F) as i8 - 8;
            block_sum += (x0 as f32) * x_block[j] + (x1 as f32) * x_block[j + 16];
        }
        sum += d * block_sum;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q4_0_f32_avx2(raw_blocks: &[u8], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    use half::f16;

    let n_blocks = x.len() / 32;
    let mut total_sum = 0.0f32;

    let mask_low = _mm_set1_epi8(0x0F);
    let eight = _mm_set1_epi8(8);

    for i in 0..n_blocks {
        let chunk = raw_blocks.get_unchecked(i * 18..(i + 1) * 18);
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let d = f16::from_bits(d_bits).to_f32();

        let qs_ptr = chunk.as_ptr().add(2);
        let x_ptr = x.as_ptr().add(i * 32);

        let q16 = _mm_loadu_si128(qs_ptr as *const __m128i);

        let q_lo = _mm_sub_epi8(_mm_and_si128(q16, mask_low), eight);
        let q_hi_shift = _mm_srli_epi16(q16, 4);
        let q_hi = _mm_sub_epi8(_mm_and_si128(q_hi_shift, mask_low), eight);

        let q_lo16 = _mm256_cvtepi8_epi16(q_lo);
        let q_hi16 = _mm256_cvtepi8_epi16(q_hi);

        let lo8_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo16));
        let lo8_f32 = _mm256_cvtepi32_ps(lo8_i32);
        let x0 = _mm256_loadu_ps(x_ptr);
        let mut acc = _mm256_mul_ps(lo8_f32, x0);

        let hi8_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo16, 1));
        let hi8_f32 = _mm256_cvtepi32_ps(hi8_i32);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        acc = _mm256_fmadd_ps(hi8_f32, x1, acc);

        let lo8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi16));
        let lo8_hi_f32 = _mm256_cvtepi32_ps(lo8_hi_i32);
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        acc = _mm256_fmadd_ps(lo8_hi_f32, x2, acc);

        let hi8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi16, 1));
        let hi8_hi_f32 = _mm256_cvtepi32_ps(hi8_hi_i32);
        let x3 = _mm256_loadu_ps(x_ptr.add(24));
        acc = _mm256_fmadd_ps(hi8_hi_f32, x3, acc);

        let hi_128 = _mm256_extractf128_ps(acc, 1);
        let lo_128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo_128, hi_128);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, shuf2);
        let block_sum = _mm_cvtss_f32(sum32);

        total_sum += d * block_sum;
    }

    total_sum
}

#[inline(always)]
pub fn dot_q4_1_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q4_1_f32_avx2(raw_blocks, x) };
        }
    }
    crate::ops::quant::vec_dot_q4_1_f32_scalar(raw_blocks, x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q4_1_f32_avx2(raw_blocks: &[u8], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    use half::f16;

    let n_blocks = x.len() / 32;
    let mut total_sum = 0.0f32;
    let mask_low = _mm_set1_epi8(0x0F);

    for i in 0..n_blocks {
        let chunk = raw_blocks.get_unchecked(i * 20..(i + 1) * 20);
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let m_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
        let d = f16::from_bits(d_bits).to_f32();
        let m = f16::from_bits(m_bits).to_f32();

        let qs_ptr = chunk.as_ptr().add(4);
        let x_ptr = x.as_ptr().add(i * 32);

        let q16 = _mm_loadu_si128(qs_ptr as *const __m128i);
        let q_lo = _mm_and_si128(q16, mask_low);
        let q_hi_shift = _mm_srli_epi16(q16, 4);
        let q_hi = _mm_and_si128(q_hi_shift, mask_low);

        let q_lo16 = _mm256_cvtepi8_epi16(q_lo);
        let q_hi16 = _mm256_cvtepi8_epi16(q_hi);

        let lo8_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo16));
        let lo8_f32 = _mm256_cvtepi32_ps(lo8_i32);
        let x0 = _mm256_loadu_ps(x_ptr);
        let mut acc = _mm256_mul_ps(lo8_f32, x0);
        let mut x_acc = x0;

        let hi8_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo16, 1));
        let hi8_f32 = _mm256_cvtepi32_ps(hi8_i32);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        acc = _mm256_fmadd_ps(hi8_f32, x1, acc);
        x_acc = _mm256_add_ps(x_acc, x1);

        let lo8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi16));
        let lo8_hi_f32 = _mm256_cvtepi32_ps(lo8_hi_i32);
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        acc = _mm256_fmadd_ps(lo8_hi_f32, x2, acc);
        x_acc = _mm256_add_ps(x_acc, x2);

        let hi8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi16, 1));
        let hi8_hi_f32 = _mm256_cvtepi32_ps(hi8_hi_i32);
        let x3 = _mm256_loadu_ps(x_ptr.add(24));
        acc = _mm256_fmadd_ps(hi8_hi_f32, x3, acc);
        x_acc = _mm256_add_ps(x_acc, x3);

        let dot_block = hsum256_ps(acc);
        let x_sum = hsum256_ps(x_acc);

        total_sum += d * dot_block + m * x_sum;
    }

    total_sum
}

#[inline(always)]
pub fn dot_q4_k_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q4_k_f32_avx2(raw_blocks, x) };
        }
    }
    crate::ops::quant::vec_dot_q4_k_f32_scalar(raw_blocks, x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q4_k_f32_avx2(raw_blocks: &[u8], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    use half::f16;

    let n_blocks = x.len() / 256;
    let mut total_sum = 0.0f32;
    let mask_low = _mm_set1_epi8(0x0F);

    for i in 0..n_blocks {
        let chunk = raw_blocks.get_unchecked(i * 144..(i + 1) * 144);
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let dmin_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
        let d = f16::from_bits(d_bits).to_f32();
        let dmin = f16::from_bits(dmin_bits).to_f32();

        let scales = &chunk[4..16];
        let qs = &chunk[16..144];
        let x_block = &x[i * 256..(i + 1) * 256];

        let mut acc = _mm256_setzero_ps();

        for sb in 0..8 {
            let sc = if sb < 4 {
                (scales[sb] & 63) as f32
            } else {
                ((scales[sb + 4] & 0x0F) | ((scales[sb - 4] >> 6) << 4)) as f32
            };
            let m = if sb < 4 {
                (scales[sb + 4] & 63) as f32
            } else {
                (((scales[sb + 4] >> 4) & 0x0F) | ((scales[sb] >> 6) << 4)) as f32
            };

            let d1 = d * sc;
            let m1 = dmin * m;

            let qs_ptr = qs.as_ptr().add(sb * 16);
            let x_ptr = x_block.as_ptr().add(sb * 32);

            let q16 = _mm_loadu_si128(qs_ptr as *const __m128i);
            let q_lo = _mm_and_si128(q16, mask_low);
            let q_hi = _mm_and_si128(_mm_srli_epi16(q16, 4), mask_low);

            let q_lo16 = _mm256_cvtepi8_epi16(q_lo);
            let q_hi16 = _mm256_cvtepi8_epi16(q_hi);

            let d1_v = _mm256_set1_ps(d1);
            let m1_v = _mm256_set1_ps(m1);

            let lo8_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo16));
            let v0 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(lo8_i32), d1_v, m1_v);
            let x0 = _mm256_loadu_ps(x_ptr);
            acc = _mm256_fmadd_ps(v0, x0, acc);

            let hi8_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo16, 1));
            let v1 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(hi8_i32), d1_v, m1_v);
            let x1 = _mm256_loadu_ps(x_ptr.add(8));
            acc = _mm256_fmadd_ps(v1, x1, acc);

            let lo8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi16));
            let v2 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(lo8_hi_i32), d1_v, m1_v);
            let x2 = _mm256_loadu_ps(x_ptr.add(16));
            acc = _mm256_fmadd_ps(v2, x2, acc);

            let hi8_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi16, 1));
            let v3 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(hi8_hi_i32), d1_v, m1_v);
            let x3 = _mm256_loadu_ps(x_ptr.add(24));
            acc = _mm256_fmadd_ps(v3, x3, acc);
        }

        let hi_128 = _mm256_extractf128_ps(acc, 1);
        let lo_128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo_128, hi_128);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, shuf2);
        total_sum += _mm_cvtss_f32(sum32);
    }

    total_sum
}

/// Dot product between a row of little-endian f16 halves and an f32 vector.
///
/// Half precision is the native on-disk format of most GGML checkpoints. The
/// kernel converts eight halves per vector register and accumulates in f32, so
/// the weight stream is half the size of a dequantized copy and no temporary
/// row buffer is needed.
#[inline(always)]
pub fn dot_f16_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len(), x.len() * 2);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            return unsafe { dot_f16_f32_avx2(row_bytes, x) };
        }
    }

    dot_f16_f32_scalar(row_bytes, x)
}

#[inline(always)]
fn f16_at(bytes: &[u8], offset: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

pub fn dot_f16_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row_bytes.len(), x.len() * 2);

    let len = x.len();
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut i = 0;

    while i + 4 <= len {
        let base = i * 2;
        acc0 += f16_at(row_bytes, base) * x[i];
        acc1 += f16_at(row_bytes, base + 2) * x[i + 1];
        acc2 += f16_at(row_bytes, base + 4) * x[i + 2];
        acc3 += f16_at(row_bytes, base + 6) * x[i + 3];
        i += 4;
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    while i < len {
        sum += f16_at(row_bytes, i * 2) * x[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_f16_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let len = x.len();
    let p_w = row_bytes.as_ptr();
    let p_x = x.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    // Four independent accumulator chains hide the FMA latency; each step
    // consumes 32 halves (64 bytes) of weights.
    let n_blocks = len / 32;
    for i in 0..n_blocks {
        let w_off = i * 64;
        let w0 = _mm_loadu_si128(p_w.add(w_off) as *const __m128i);
        let w1 = _mm_loadu_si128(p_w.add(w_off + 16) as *const __m128i);
        let w2 = _mm_loadu_si128(p_w.add(w_off + 32) as *const __m128i);
        let w3 = _mm_loadu_si128(p_w.add(w_off + 48) as *const __m128i);

        let x_off = i * 32;
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(w0), _mm256_loadu_ps(p_x.add(x_off)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(w1),
            _mm256_loadu_ps(p_x.add(x_off + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(w2),
            _mm256_loadu_ps(p_x.add(x_off + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(w3),
            _mm256_loadu_ps(p_x.add(x_off + 24)),
            acc3,
        );
    }

    let acc01 = _mm256_add_ps(acc0, acc1);
    let acc23 = _mm256_add_ps(acc2, acc3);
    let mut acc = _mm256_add_ps(acc01, acc23);

    let mut i = n_blocks * 32;
    while i + 8 <= len {
        let w = _mm_loadu_si128(p_w.add(i * 2) as *const __m128i);
        acc = _mm256_fmadd_ps(_mm256_cvtph_ps(w), _mm256_loadu_ps(p_x.add(i)), acc);
        i += 8;
    }

    let mut result = hsum256_ps(acc);
    while i < len {
        result += f16_at(row_bytes, i * 2) * *p_x.add(i);
        i += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Integer dot kernels against pre-quantized Q8 activations.
//
// The activation row is quantized once (see `Q8Activation`) and then every
// weight row is consumed by integer multiply-adds. All products are exact:
// the quantizer clamps to +-127, so a pair sum can never exceed 2*127*127 and
// `vpmaddubsw`/`vpmaddwd` cannot saturate. Only the per-block scale product is
// accumulated in f32, which keeps the result close to the f32 activation path.
// ---------------------------------------------------------------------------

/// Dot product between Q4_0 blocks and a Q8 activation row.
#[inline(always)]
pub fn dot_q4_0_q8(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    assert_eq!(raw_blocks.len(), act.len() * 18);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q4_0_q8_avx2(raw_blocks, act) };
        }
    }
    dot_q4_0_q8_scalar(raw_blocks, act)
}

#[allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]
pub fn dot_q4_0_q8_scalar(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, b) in raw_blocks.chunks_exact(18).zip(act.blocks()) {
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let qs = &chunk[2..18];

        let mut idot = 0i32;
        for j in 0..16 {
            idot += ((qs[j] & 0x0F) as i32 - 8) * b.qs[j] as i32;
            idot += ((qs[j] >> 4) as i32 - 8) * b.qs[j + 16] as i32;
        }
        sum += d * b.d * idot as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q4_0_q8_avx2(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    use core::arch::x86_64::*;

    let mask = _mm_set1_epi8(0x0F);
    let one16 = _mm256_set1_epi16(1);
    // Each block's integer partial stays in a vector lane and is scaled into
    // this f32 accumulator, so the horizontal sum happens once per row instead
    // of once per block.
    let mut acc = _mm256_setzero_ps();
    let mut corr = 0.0f32;

    for (i, b) in act.blocks().iter().enumerate() {
        let chunk = raw_blocks.get_unchecked(i * 18..(i + 1) * 18);
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();

        let q16 = _mm_loadu_si128(chunk.as_ptr().add(2) as *const __m128i);
        let q_lo = _mm_and_si128(q16, mask);
        let q_hi = _mm_and_si128(_mm_srli_epi16(q16, 4), mask);
        let qv = _mm256_set_m128i(q_hi, q_lo);

        let xv = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);
        let partial = _mm256_madd_epi16(_mm256_maddubs_epi16(qv, xv), one16);

        // The stored nibbles are unsigned 0..15, so the signed weight is q - 8.
        // The activation sum is exact in `sum_q`: sum(x) = d_act * sum_q.
        let scale = d * b.d;
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(partial), _mm256_set1_ps(scale), acc);
        corr -= 8.0 * scale * b.sum_q as f32;
    }
    hsum256_ps(acc) + corr
}

/// Dot product between Q4_1 blocks and a Q8 activation row.
#[inline(always)]
pub fn dot_q4_1_q8(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    assert_eq!(raw_blocks.len(), act.len() * 20);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q4_1_q8_avx2(raw_blocks, act) };
        }
    }
    dot_q4_1_q8_scalar(raw_blocks, act)
}

#[allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]
pub fn dot_q4_1_q8_scalar(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, b) in raw_blocks.chunks_exact(20).zip(act.blocks()) {
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();
        let qs = &chunk[4..20];

        let mut idot = 0i32;
        for j in 0..16 {
            idot += (qs[j] & 0x0F) as i32 * b.qs[j] as i32;
            idot += (qs[j] >> 4) as i32 * b.qs[j + 16] as i32;
        }
        sum += d * b.d * idot as f32 + m * b.d * b.sum_q as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q4_1_q8_avx2(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    use core::arch::x86_64::*;

    let mask = _mm_set1_epi8(0x0F);
    let one16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut corr = 0.0f32;

    for (i, b) in act.blocks().iter().enumerate() {
        let chunk = raw_blocks.get_unchecked(i * 20..(i + 1) * 20);
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();

        let q16 = _mm_loadu_si128(chunk.as_ptr().add(4) as *const __m128i);
        let q_lo = _mm_and_si128(q16, mask);
        let q_hi = _mm_and_si128(_mm_srli_epi16(q16, 4), mask);
        let qv = _mm256_set_m128i(q_hi, q_lo);

        let xv = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);
        let partial = _mm256_madd_epi16(_mm256_maddubs_epi16(qv, xv), one16);

        let scale = d * b.d;
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(partial), _mm256_set1_ps(scale), acc);
        corr += m * b.d * b.sum_q as f32;
    }
    hsum256_ps(acc) + corr
}

/// Dot product between Q5_0 blocks and a Q8 activation row.
#[inline(always)]
pub fn dot_q5_0_q8(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    assert_eq!(raw_blocks.len(), act.len() * 22);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q5_0_q8_avx2(raw_blocks, act) };
        }
    }
    dot_q5_0_q8_scalar(raw_blocks, act)
}

#[allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]
pub fn dot_q5_0_q8_scalar(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, b) in raw_blocks.chunks_exact(22).zip(act.blocks()) {
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let qh = u32::from_le_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);
        let qs = &chunk[6..22];

        let mut idot = 0i32;
        for j in 0..16 {
            let h0 = ((qh >> j) & 1) as i32;
            let h1 = ((qh >> (j + 16)) & 1) as i32;
            idot += (((qs[j] & 0x0F) as i32) | (h0 << 4)) * b.qs[j] as i32;
            idot += ((((qs[j] >> 4) & 0x0F) as i32) | (h1 << 4)) * b.qs[j + 16] as i32;
        }
        // Stored values are unsigned 0..31; the signed weight is q - 16.
        sum += d * b.d * (idot - 16 * b.sum_q) as f32;
    }
    sum
}

/// Expand the 32 high bits of a Q5 block into one byte per element.
///
/// Each element contributes `0x10` when its fifth bit is set. The broadcast
/// places the four `qh` bytes in every 32-bit lane; a per-lane byte shuffle
/// then selects byte 0 for elements 0..7 and byte 1 for elements 8..15, which
/// maps every byte of the 256-bit result onto one element.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q5_high_bits(qh: u32) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;

    let lo = _mm_set1_epi32(qh as i32);
    let hi = _mm_set1_epi32((qh >> 16) as i32);
    let src = _mm256_set_m128i(hi, lo);

    let idx = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1,
    );
    // 0x80 is `-128` as an i8 literal; it is the mask for bit 7 of a byte.
    let bit = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128,
    );

    let bytes = _mm256_shuffle_epi8(src, idx);
    let hit = _mm256_and_si256(bytes, bit);
    let unset = _mm256_cmpeq_epi8(hit, _mm256_setzero_si256());
    _mm256_andnot_si256(unset, _mm256_set1_epi8(0x10))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q5_0_q8_avx2(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    use core::arch::x86_64::*;

    let mask = _mm_set1_epi8(0x0F);
    let one16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut corr = 0.0f32;

    for (i, b) in act.blocks().iter().enumerate() {
        let chunk = raw_blocks.get_unchecked(i * 22..(i + 1) * 22);
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let qh = u32::from_le_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);

        let q16 = _mm_loadu_si128(chunk.as_ptr().add(6) as *const __m128i);
        let q_lo = _mm_and_si128(q16, mask);
        let q_hi = _mm_and_si128(_mm_srli_epi16(q16, 4), mask);
        let qv = _mm256_or_si256(_mm256_set_m128i(q_hi, q_lo), q5_high_bits(qh));

        let xv = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);
        let partial = _mm256_madd_epi16(_mm256_maddubs_epi16(qv, xv), one16);

        // Stored values are unsigned 0..31, so the signed weight is q - 16 and
        // the activation sum comes from the exact `sum_q`.
        let scale = d * b.d;
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(partial), _mm256_set1_ps(scale), acc);
        corr -= 16.0 * scale * b.sum_q as f32;
    }
    hsum256_ps(acc) + corr
}

/// Dot product between Q5_1 blocks and a Q8 activation row.
#[inline(always)]
pub fn dot_q5_1_q8(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    assert_eq!(raw_blocks.len(), act.len() * 24);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q5_1_q8_avx2(raw_blocks, act) };
        }
    }
    dot_q5_1_q8_scalar(raw_blocks, act)
}

#[allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]
pub fn dot_q5_1_q8_scalar(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, b) in raw_blocks.chunks_exact(24).zip(act.blocks()) {
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();
        let qh = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let qs = &chunk[8..24];

        let mut idot = 0i32;
        for j in 0..16 {
            let h0 = ((qh >> j) & 1) as i32;
            let h1 = ((qh >> (j + 16)) & 1) as i32;
            idot += (((qs[j] & 0x0F) as i32) | (h0 << 4)) * b.qs[j] as i32;
            idot += ((((qs[j] >> 4) & 0x0F) as i32) | (h1 << 4)) * b.qs[j + 16] as i32;
        }
        sum += d * b.d * idot as f32 + m * b.d * b.sum_q as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q5_1_q8_avx2(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    use core::arch::x86_64::*;

    let mask = _mm_set1_epi8(0x0F);
    let one16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut corr = 0.0f32;

    for (i, b) in act.blocks().iter().enumerate() {
        let chunk = raw_blocks.get_unchecked(i * 24..(i + 1) * 24);
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();
        let qh = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);

        let q16 = _mm_loadu_si128(chunk.as_ptr().add(8) as *const __m128i);
        let q_lo = _mm_and_si128(q16, mask);
        let q_hi = _mm_and_si128(_mm_srli_epi16(q16, 4), mask);
        let qv = _mm256_or_si256(_mm256_set_m128i(q_hi, q_lo), q5_high_bits(qh));

        let xv = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);
        let partial = _mm256_madd_epi16(_mm256_maddubs_epi16(qv, xv), one16);

        let scale = d * b.d;
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(partial), _mm256_set1_ps(scale), acc);
        corr += m * b.d * b.sum_q as f32;
    }
    hsum256_ps(acc) + corr
}

/// Dot product between Q8_0 blocks and a Q8 activation row.
#[inline(always)]
pub fn dot_q8_0_q8(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    assert_eq!(raw_blocks.len(), act.len() * 34);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q8_0_q8_avx2(raw_blocks, act) };
        }
    }
    dot_q8_0_q8_scalar(raw_blocks, act)
}

#[allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]
pub fn dot_q8_0_q8_scalar(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, b) in raw_blocks.chunks_exact(34).zip(act.blocks()) {
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let qs = &chunk[2..34];

        let mut idot = 0i32;
        for j in 0..32 {
            idot += (qs[j] as i8 as i32) * b.qs[j] as i32;
        }
        sum += d * b.d * idot as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_dot_q8_0_q8_avx2(raw_blocks: &[u8], act: &Q8Activation) -> f32 {
    use core::arch::x86_64::*;

    let one16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();

    for (i, b) in act.blocks().iter().enumerate() {
        let chunk = raw_blocks.get_unchecked(i * 34..(i + 1) * 34);
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();

        let qv = _mm256_loadu_si256(chunk.as_ptr().add(2) as *const __m256i);
        let xv = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);

        // vpmaddubsw needs an unsigned first operand: take |q| and fold the
        // sign into the activation bytes, which leaves the product unchanged.
        let signed_x = _mm256_sign_epi8(xv, qv);
        let abs_q = _mm256_abs_epi8(qv);
        let partial = _mm256_madd_epi16(_mm256_maddubs_epi16(abs_q, signed_x), one16);

        let scale = d * b.d;
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(partial), _mm256_set1_ps(scale), acc);
    }
    hsum256_ps(acc)
}
