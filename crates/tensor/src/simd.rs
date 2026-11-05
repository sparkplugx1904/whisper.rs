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
