use crate::ops::softmax::softmax_inplace;
use crate::simd::dot_f32;
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
    let d_head = d_model / n_heads;
    assert_eq!(q.len(), t_q * d_model);
    assert_eq!(k.len(), t_kv * d_model);
    assert_eq!(v.len(), t_kv * d_model);
    assert_eq!(output.len(), t_q * d_model);

    let scale = 1.0f32 / (d_head as f32).sqrt();

    // Use Rayon parallel iterator across heads for multi-token / encoder passes
    if t_q * t_kv >= 4096 {
        let out_ptr = output.as_mut_ptr() as usize;

        (0..n_heads).into_par_iter().for_each(|h| {
            let head_offset = h * d_head;
            let mut scores = vec![0.0f32; t_kv];

            for i in 0..t_q {
                let q_vec = &q[i * d_model + head_offset..i * d_model + head_offset + d_head];
                let max_j = if causal { i + 1 } else { t_kv };

                // 1. Compute attention scores for row i
                for (j, score_entry) in scores[..max_j].iter_mut().enumerate() {
                    let k_vec = &k[j * d_model + head_offset..j * d_model + head_offset + d_head];
                    *score_entry = dot_f32(q_vec, k_vec) * scale;
                }

                // 2. Softmax
                softmax_inplace(&mut scores[0..max_j]);

                // 3. Compute weighted sum of values into disjoint head slice
                unsafe {
                    let head_out_ptr = (out_ptr as *mut f32).add(i * d_model + head_offset);
                    let out_vec = std::slice::from_raw_parts_mut(head_out_ptr, d_head);
                    out_vec.fill(0.0);

                    for (j, &weight) in scores[..max_j].iter().enumerate() {
                        let v_vec =
                            &v[j * d_model + head_offset..j * d_model + head_offset + d_head];
                        for d in 0..d_head {
                            out_vec[d] += weight * v_vec[d];
                        }
                    }
                }
            }
        });
    } else {
        // Fast sequential path for single-token decode
        let mut scores = vec![0.0f32; t_kv];

        for h in 0..n_heads {
            let head_offset = h * d_head;

            for i in 0..t_q {
                let q_vec = &q[i * d_model + head_offset..i * d_model + head_offset + d_head];
                let max_j = if causal { i + 1 } else { t_kv };

                for (j, score_entry) in scores[..max_j].iter_mut().enumerate() {
                    let k_vec = &k[j * d_model + head_offset..j * d_model + head_offset + d_head];
                    *score_entry = dot_f32(q_vec, k_vec) * scale;
                }

                softmax_inplace(&mut scores[0..max_j]);

                let out_vec =
                    &mut output[i * d_model + head_offset..i * d_model + head_offset + d_head];
                out_vec.fill(0.0);

                for (j, &weight) in scores[..max_j].iter().enumerate() {
                    let v_vec = &v[j * d_model + head_offset..j * d_model + head_offset + d_head];
                    for d in 0..d_head {
                        out_vec[d] += weight * v_vec[d];
                    }
                }
            }
        }
    }
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

    let scale = 1.0f32 / (d_head as f32).sqrt();
    let br = 64usize;
    let bc = 64usize;

    let out_ptr = output.as_mut_ptr() as usize;

    (0..n_heads).into_par_iter().for_each(|h| {
        let head_offset = h * d_head;

        for i_start in (0..t_q).step_by(br) {
            let i_end = (i_start + br).min(t_q);
            let b_i = i_end - i_start;

            let mut m_vec = vec![f32::NEG_INFINITY; b_i];
            let mut l_vec = vec![0.0f32; b_i];
            let mut o_acc = vec![0.0f32; b_i * d_head];

            for j_start in (0..t_kv).step_by(bc) {
                if causal && j_start > i_end {
                    break;
                }
                let j_end = (j_start + bc).min(t_kv);
                let b_j = j_end - j_start;

                for r in 0..b_i {
                    let i = i_start + r;
                    let q_vec = &q[i * d_model + head_offset..i * d_model + head_offset + d_head];

                    let mut tile_scores = [f32::NEG_INFINITY; 64];
                    let mut local_max = f32::NEG_INFINITY;

                    for (c, score_entry) in tile_scores[..b_j].iter_mut().enumerate() {
                        let j = j_start + c;
                        if causal && j > i {
                            continue;
                        }
                        let k_vec =
                            &k[j * d_model + head_offset..j * d_model + head_offset + d_head];
                        let score = dot_f32(q_vec, k_vec) * scale;
                        *score_entry = score;
                        if score > local_max {
                            local_max = score;
                        }
                    }

                    if local_max == f32::NEG_INFINITY {
                        continue;
                    }

                    let m_prev = m_vec[r];
                    let m_new = m_prev.max(local_max);
                    let alpha = if m_prev == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (m_prev - m_new).exp()
                    };

                    l_vec[r] *= alpha;
                    let o_row = &mut o_acc[r * d_head..(r + 1) * d_head];
                    for val in o_row.iter_mut() {
                        *val *= alpha;
                    }

                    for (c, &s) in tile_scores[..b_j].iter().enumerate() {
                        if s == f32::NEG_INFINITY {
                            continue;
                        }
                        let p = (s - m_new).exp();
                        l_vec[r] += p;

                        let j = j_start + c;
                        let v_vec =
                            &v[j * d_model + head_offset..j * d_model + head_offset + d_head];
                        for d_idx in 0..d_head {
                            o_row[d_idx] += p * v_vec[d_idx];
                        }
                    }

                    m_vec[r] = m_new;
                }
            }

            for r in 0..b_i {
                let i = i_start + r;
                let l = l_vec[r].max(1e-10);
                let inv_l = 1.0 / l;
                let o_row = &o_acc[r * d_head..(r + 1) * d_head];

                unsafe {
                    let out_slice_ptr = (out_ptr as *mut f32).add(i * d_model + head_offset);
                    for (d_idx, &val) in o_row.iter().take(d_head).enumerate() {
                        *out_slice_ptr.add(d_idx) = val * inv_l;
                    }
                }
            }
        }
    });
}
