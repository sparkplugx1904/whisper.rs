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
