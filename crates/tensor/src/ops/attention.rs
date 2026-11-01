use crate::ops::softmax::softmax_inplace;

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

    // Reusable scores buffer for one row of attention
    let mut scores = vec![0.0f32; t_kv];

    for h in 0..n_heads {
        let head_offset = h * d_head;

        for i in 0..t_q {
            let q_vec = &q[i * d_model + head_offset..i * d_model + head_offset + d_head];

            let max_j = if causal { i + 1 } else { t_kv };

            // 1. Compute attention scores for row i
            for j in 0..max_j {
                let k_vec = &k[j * d_model + head_offset..j * d_model + head_offset + d_head];
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[j] = dot * scale;
            }

            // 2. Softmax
            softmax_inplace(&mut scores[0..max_j]);

            // 3. Compute weighted sum of values
            let out_vec =
                &mut output[i * d_model + head_offset..i * d_model + head_offset + d_head];

            // Initialize or clear for this head
            out_vec.fill(0.0);

            for j in 0..max_j {
                let weight = scores[j];
                let v_vec = &v[j * d_model + head_offset..j * d_model + head_offset + d_head];
                for d in 0..d_head {
                    out_vec[d] += weight * v_vec[d];
                }
            }
        }
    }
}
