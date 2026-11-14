use crate::ops::softmax::softmax_inplace;
use rayon::prelude::*;

/// Query rows processed per tile by [`flash_attention_2`].
///
/// With a 64-wide head a query tile is `128 * 64 * 4 = 32 KiB` of query panel
/// and the same of output accumulator. 128 rows give one task per head per
/// 128 decoder positions: for 6 heads and a 1500-frame encoder that is 72
/// independent tasks, wide enough to fill a large core count while keeping the
/// per-task scratch below 100 KiB. Smaller tiles re-read the K/V stream more
/// often (traffic scales with `t_kv / Q_TILE`); larger tiles gain little and
/// inflate the scratch that every rayon worker keeps live.
pub const Q_TILE: usize = 128;

/// Keys processed per tile by [`flash_attention_2`].
///
/// 128 is a multiple of the 8-lane f32 micro-kernel used by `matrixmultiply`,
/// so only the last tile of a sequence has a ragged edge. The score tile is
/// `Q_TILE * KV_TILE * 4 = 64 KiB`, which does not fit a 32 KiB L1D but does
/// fit a 512 KiB L2 together with the 32 KiB query and accumulator panels;
/// it is written once and read twice per key tile, so L2 residency is what
/// matters. The K and V panels (`KV_TILE * d_head * 4 = 32 KiB` each) are
/// streamed sequentially and prefetched by hardware. Doubling the key tile
/// halves the number of passes over the query/accumulator panels and the
/// `sgemm` dispatch count, but pushes the score tile out of L2 entirely.
pub const KV_TILE: usize = 128;

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

/// Flash Attention 2 with tiled online softmax.
///
/// The kernel never materializes the $T_q \times T_{kv}$ score matrix. Work is
/// split into `(head, query tile)` tasks; each task walks the key tiles of its
/// head in order and maintains, per query row, the running row maximum `m`,
/// the running normalizer `l` and the unnormalized output accumulator `acc`
/// (the online-softmax recurrence of Dao et al.). A key tile contributes
///
/// ```text
/// m_new = max(m, max_j s_j)
/// l     = l * exp(m - m_new) + sum_j exp(s_j - m_new)
/// acc   = acc * exp(m - m_new) + sum_j exp(s_j - m_new) * v_j
/// ```
///
/// and the final output is `acc / l`. Because the statistics are folded in as
/// the key tiles stream past, only a `Q_TILE x KV_TILE` score tile, a
/// `Q_TILE x d_head` accumulator and two `Q_TILE` row vectors are live: no
/// `T_q * T_kv` buffer is allocated and the score matrix is never written to
/// or read from memory beyond the L1-resident tile.
///
/// `causal` masks keys with `j > i` (clamped to `t_kv`) and skips key tiles
/// that lie entirely beyond the last query of the tile. Both matrix products
/// are f32 `matrixmultiply` calls with runtime AVX2/FMA dispatch; the scalar
/// fallback of that crate keeps the same f32 numerics on other targets.
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
    assert!(n_heads > 0, "n_heads must be non-zero");
    assert!(
        d_model.is_multiple_of(n_heads),
        "d_model ({d_model}) is not divisible by n_heads ({n_heads})"
    );
    let d_head = d_model / n_heads;
    assert_eq!(q.len(), t_q * d_model);
    assert_eq!(k.len(), t_kv * d_model);
    assert_eq!(v.len(), t_kv * d_model);
    assert_eq!(output.len(), t_q * d_model);

    if t_q == 0 {
        return;
    }
    if t_kv == 0 {
        output.fill(0.0);
        return;
    }

    // A single decoder query has no query-tile dimension worth tiling and the
    // dedicated kernel keeps the row statistics in registers. Causal
    // single-query attention still needs the general path: it may only look at
    // the first key, which the decode kernel does not mask.
    if t_q == 1 && d_head == 64 && !causal {
        single_token_mha_d64(q, k, v, t_kv, n_heads, d_model, output);
        return;
    }

    let scale = 1.0f32 / (d_head as f32).sqrt();
    let n_q_tiles = t_q.div_ceil(Q_TILE);
    let out_ptr = output.as_mut_ptr() as usize;

    (0..n_heads * n_q_tiles).into_par_iter().for_each(|task| {
        let h = task / n_q_tiles;
        let i0 = (task % n_q_tiles) * Q_TILE;
        let tr = (t_q - i0).min(Q_TILE);
        let head_offset = h * d_head;

        // Scratch is owned by one `(head, query tile)` task and reused for
        // every key tile, so the tiling loops below never allocate. Only `tr`
        // rows are ever touched, which keeps short decoder calls cheap.
        let mut scores = vec![0.0f32; tr * KV_TILE];
        let mut acc = vec![0.0f32; tr * d_head];
        let mut row_max = [f32::NEG_INFINITY; Q_TILE];
        let mut row_sum = [0.0f32; Q_TILE];

        let p_q = unsafe { q.as_ptr().add(i0 * d_model + head_offset) };
        let p_out = out_ptr as *mut f32;

        let mut j0 = 0usize;
        while j0 < t_kv {
            // Under a causal mask no query of this tile can reach a key past
            // `i0 + tr - 1`, so stop instead of masking whole tiles.
            if causal && j0 > i0 + tr - 1 {
                break;
            }
            let tc = (t_kv - j0).min(KV_TILE);

            // S = scale * Q_tile * K_tile^T. Rows use a fixed KV_TILE stride
            // so a ragged final tile only shortens the row, it does not
            // change the layout the two passes below index.
            unsafe {
                matrixmultiply::sgemm(
                    tr,
                    d_head,
                    tc,
                    scale,
                    p_q,
                    d_model as isize,
                    1,
                    k.as_ptr().add(j0 * d_model + head_offset),
                    1,
                    d_model as isize,
                    0.0,
                    scores.as_mut_ptr(),
                    KV_TILE as isize,
                    1,
                );
            }

            for r in 0..tr {
                let row = &mut scores[r * KV_TILE..r * KV_TILE + tc];
                let end = if causal {
                    (i0 + r + 1).saturating_sub(j0).min(tc)
                } else {
                    tc
                };
                if end == 0 {
                    // The whole tile is masked for this row; clear the raw
                    // scores so the PV product contributes nothing.
                    row.fill(0.0);
                    continue;
                }

                let mut m_tile = f32::NEG_INFINITY;
                for &s in &row[..end] {
                    m_tile = m_tile.max(s);
                }
                let m_prev = row_max[r];
                let m_new = m_prev.max(m_tile);
                let alpha = if m_prev == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m_prev - m_new).exp()
                };

                let mut l_tile = 0.0f32;
                for s in &mut row[..end] {
                    let p = (*s - m_new).exp();
                    *s = p;
                    l_tile += p;
                }
                // Masked columns and the ragged tail must multiply to zero.
                row[end..].fill(0.0);

                row_max[r] = m_new;
                row_sum[r] = row_sum[r] * alpha + l_tile;
                if alpha != 1.0 {
                    let acc_row = &mut acc[r * d_head..(r + 1) * d_head];
                    for a in acc_row.iter_mut() {
                        *a *= alpha;
                    }
                }
            }

            // acc = acc + P_tile * V_tile. Rows that were fully masked hold a
            // zero probability row and leave `acc` untouched.
            unsafe {
                matrixmultiply::sgemm(
                    tr,
                    tc,
                    d_head,
                    1.0,
                    scores.as_ptr(),
                    KV_TILE as isize,
                    1,
                    v.as_ptr().add(j0 * d_model + head_offset),
                    d_model as isize,
                    1,
                    1.0,
                    acc.as_mut_ptr(),
                    d_head as isize,
                    1,
                );
            }

            j0 += KV_TILE;
        }

        for r in 0..tr {
            let inv_sum = if row_sum[r] > 0.0 {
                1.0 / row_sum[r]
            } else {
                0.0
            };
            let dst = unsafe { p_out.add((i0 + r) * d_model + head_offset) };
            let acc_row = &acc[r * d_head..(r + 1) * d_head];
            for (d, &a) in acc_row.iter().enumerate() {
                unsafe {
                    *dst.add(d) = a * inv_sum;
                }
            }
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
