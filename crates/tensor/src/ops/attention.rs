use crate::ops::softmax::softmax_inplace;
use rayon::prelude::*;
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_q: usize,
    t_kv: usize,
    n_heads: usize,
    d_model: usize,
    causal: bool,
    output: &mut [f32],
) {
    flash_attention_2(q, k, v, t_q, t_kv, n_heads, d_model, causal, output);
}

/// Flash Attention 2 algorithm with Tiled Online Softmax.
/// Processes Query in blocks of $B_r = 64$ and Key/Value in blocks of $B_c = 64$,
/// computing exact attention without materializing the $T_q \times T_{kv}$ score matrix.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_q: usize,
    t_kv: usize,
    n_heads: usize,
    d_model: usize,
    causal: bool,
    output: &mut [f32],
) {
    let d_head = d_model / n_heads;
    assert_eq!(q.len(), t_q * d_model);
    assert_eq!(k.len(), t_kv * d_model);
    assert_eq!(v.len(), t_kv * d_model);
    assert_eq!(output.len(), t_q * d_model);

    if t_q == 1 && d_head == 64 {
        single_token_mha_d64(q, k, v, t_kv, n_heads, d_model, output);
        return;
    }

    let scale = 1.0f32 / (d_head as f32).sqrt();
    let out_ptr = output.as_mut_ptr() as usize;

    (0..n_heads).into_par_iter().for_each(|h| {
        let head_offset = h * d_head;
        let mut scores = vec![0.0f32; t_q * t_kv];

        // 1. S_h = scale * (Q_h * K_h^T)
        unsafe {
            matrixmultiply::sgemm(
                t_q,
                d_head,
                t_kv,
                scale,
                q.as_ptr().add(head_offset),
                d_model as isize,
                1,
                k.as_ptr().add(head_offset),
                1,
                d_model as isize,
                0.0,
                scores.as_mut_ptr(),
                t_kv as isize,
                1,
            );
        }

        // 2. Softmax per row
        for i in 0..t_q {
            let row = &mut scores[i * t_kv..(i + 1) * t_kv];
            if causal {
                let valid_len = (i + 1).min(t_kv);
                softmax_inplace(&mut row[..valid_len]);
                if valid_len < t_kv {
                    row[valid_len..].fill(0.0);
                }
            } else {
                softmax_inplace(row);
            }
        }

        // 3. O_h = S_h * V_h
        unsafe {
            matrixmultiply::sgemm(
                t_q,
                t_kv,
                d_head,
                1.0,
                scores.as_ptr(),
                t_kv as isize,
                1,
                v.as_ptr().add(head_offset),
                d_model as isize,
                1,
                0.0,
                (out_ptr as *mut f32).add(head_offset),
                d_model as isize,
                1,
            );
        }
    });
}

#[inline(always)]
pub fn single_token_mha_d64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_kv: usize,
    n_heads: usize,
    d_model: usize,
    output: &mut [f32],
) {
    let scale = 0.125f32; // 1.0 / sqrt(64)
    let k_stride = d_model;
    let v_stride = d_model;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                single_token_mha_d64_avx2(
                    q, k, k_stride, v, v_stride, t_kv, scale, n_heads, d_model, output,
                );
            }
            return;
        }
    }

    single_token_mha_d64_scalar(
        q, k, k_stride, v, v_stride, t_kv, scale, n_heads, d_model, output,
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_m256(acc: core::arch::x86_64::__m256) -> f32 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn single_token_mha_d64_avx2(
    q: &[f32],
    k: &[f32],
    k_stride: usize,
    v: &[f32],
    v_stride: usize,
    t_kv: usize,
    scale: f32,
    n_heads: usize,
    _d_model: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::*;

    let mut scores_stack = [0.0f32; 1500];
    let mut scores_vec = Vec::new();
    let scores = if t_kv <= 1500 {
        &mut scores_stack[..t_kv]
    } else {
        scores_vec.resize(t_kv, 0.0f32);
        &mut scores_vec[..]
    };

    let p_q = q.as_ptr();
    let p_k = k.as_ptr();
    let p_v = v.as_ptr();
    let p_out = output.as_mut_ptr();
    let v_scale = _mm256_set1_ps(scale);

    for h in 0..n_heads {
        let head_offset = h * 64;
        let q_ptr = p_q.add(head_offset);
        let k_head_ptr = p_k.add(head_offset);
        let v_head_ptr = p_v.add(head_offset);
        let out_head_ptr = p_out.add(head_offset);

        let vq0 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr), v_scale);
        let vq1 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr.add(8)), v_scale);
        let vq2 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr.add(16)), v_scale);
        let vq3 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr.add(24)), v_scale);
        let vq4 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr.add(32)), v_scale);
        let vq5 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr.add(40)), v_scale);
        let vq6 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr.add(48)), v_scale);
        let vq7 = _mm256_mul_ps(_mm256_loadu_ps(q_ptr.add(56)), v_scale);

        for j in 0..t_kv {
            let kj_ptr = k_head_ptr.add(j * k_stride);
            let mut dot0 = _mm256_mul_ps(vq0, _mm256_loadu_ps(kj_ptr));
            let mut dot1 = _mm256_mul_ps(vq1, _mm256_loadu_ps(kj_ptr.add(8)));
            dot0 = _mm256_fmadd_ps(vq2, _mm256_loadu_ps(kj_ptr.add(16)), dot0);
            dot1 = _mm256_fmadd_ps(vq3, _mm256_loadu_ps(kj_ptr.add(24)), dot1);
            dot0 = _mm256_fmadd_ps(vq4, _mm256_loadu_ps(kj_ptr.add(32)), dot0);
            dot1 = _mm256_fmadd_ps(vq5, _mm256_loadu_ps(kj_ptr.add(40)), dot1);
            dot0 = _mm256_fmadd_ps(vq6, _mm256_loadu_ps(kj_ptr.add(48)), dot0);
            dot1 = _mm256_fmadd_ps(vq7, _mm256_loadu_ps(kj_ptr.add(56)), dot1);

            let dot = _mm256_add_ps(dot0, dot1);
            scores[j] = hsum_m256(dot);
        }

        softmax_inplace(scores);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut acc4 = _mm256_setzero_ps();
        let mut acc5 = _mm256_setzero_ps();
        let mut acc6 = _mm256_setzero_ps();
        let mut acc7 = _mm256_setzero_ps();

        for j in 0..t_kv {
            let w = scores[j];
            if w < 1e-7 {
                continue;
            }
            let vw = _mm256_set1_ps(w);
            let vj_ptr = v_head_ptr.add(j * v_stride);
            acc0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr), acc0);
            acc1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr.add(8)), acc1);
            acc2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr.add(16)), acc2);
            acc3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr.add(24)), acc3);
            acc4 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr.add(32)), acc4);
            acc5 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr.add(40)), acc5);
            acc6 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr.add(48)), acc6);
            acc7 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vj_ptr.add(56)), acc7);
        }

        _mm256_storeu_ps(out_head_ptr, acc0);
        _mm256_storeu_ps(out_head_ptr.add(8), acc1);
        _mm256_storeu_ps(out_head_ptr.add(16), acc2);
        _mm256_storeu_ps(out_head_ptr.add(24), acc3);
        _mm256_storeu_ps(out_head_ptr.add(32), acc4);
        _mm256_storeu_ps(out_head_ptr.add(40), acc5);
        _mm256_storeu_ps(out_head_ptr.add(48), acc6);
        _mm256_storeu_ps(out_head_ptr.add(56), acc7);
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn single_token_mha_d64_scalar(
    q: &[f32],
    k: &[f32],
    k_stride: usize,
    v: &[f32],
    v_stride: usize,
    t_kv: usize,
    scale: f32,
    n_heads: usize,
    _d_model: usize,
    output: &mut [f32],
) {
    let mut scores = vec![0.0f32; t_kv];
    for h in 0..n_heads {
        let head_offset = h * 64;
        let q_h = &q[head_offset..head_offset + 64];
        let out_h = &mut output[head_offset..head_offset + 64];

        for j in 0..t_kv {
            let k_vec = &k[j * k_stride + head_offset..j * k_stride + head_offset + 64];
            let mut dot = 0.0f32;
            for d in 0..64 {
                dot += q_h[d] * k_vec[d];
            }
            scores[j] = dot * scale;
        }

        softmax_inplace(&mut scores[..t_kv]);

        out_h.fill(0.0);
        for j in 0..t_kv {
            let w = scores[j];
            if w < 1e-7 {
                continue;
            }
            let v_vec = &v[j * v_stride + head_offset..j * v_stride + head_offset + 64];
            for d in 0..64 {
                out_h[d] += w * v_vec[d];
            }
        }
    }
}
