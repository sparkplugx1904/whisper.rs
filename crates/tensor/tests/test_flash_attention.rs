use whisper_tensor::ops::{flash_attention_2, multi_head_attention};

#[test]
fn test_flash_attention_2_matches_standard() {
    let t_q = 64;
    let t_kv = 128;
    let n_heads = 4;
    let d_model = 64; // d_head = 16

    let mut q = Vec::with_capacity(t_q * d_model);
    for i in 0..t_q * d_model {
        q.push(((i % 17) as f32 - 8.0) * 0.1);
    }

    let mut k = Vec::with_capacity(t_kv * d_model);
    for i in 0..t_kv * d_model {
        k.push(((i % 19) as f32 - 9.0) * 0.1);
    }

    let mut v = Vec::with_capacity(t_kv * d_model);
    for i in 0..t_kv * d_model {
        v.push(((i % 23) as f32 - 11.0) * 0.1);
    }

    // 1. Non-causal comparison
    let mut out_std = vec![0.0f32; t_q * d_model];
    reference_mha(&q, &k, &v, t_q, t_kv, n_heads, d_model, false, &mut out_std);

    let mut out_mha = vec![0.0f32; t_q * d_model];
    multi_head_attention(&q, &k, &v, t_q, t_kv, n_heads, d_model, false, &mut out_mha);
    let mut out_flash = vec![0.0f32; t_q * d_model];
    flash_attention_2(
        &q,
        &k,
        &v,
        t_q,
        t_kv,
        n_heads,
        d_model,
        false,
        &mut out_flash,
    );

    for (i, (&std_val, &flash_val)) in out_std.iter().zip(out_flash.iter()).enumerate() {
        let diff = (std_val - flash_val).abs();
        assert!(
            diff <= 1e-4,
            "Non-causal mismatch at {}: std {} vs flash {}, diff {}",
            i,
            std_val,
            flash_val,
            diff
        );
    }

    // 2. Causal comparison
    let mut out_std_causal = vec![0.0f32; t_q * d_model];
    reference_mha(&q, &k, &v, t_q, t_kv, n_heads, d_model, true, &mut out_std_causal);

    let mut out_flash_causal = vec![0.0f32; t_q * d_model];
    flash_attention_2(
        &q,
        &k,
        &v,
        t_q,
        t_kv,
        n_heads,
        d_model,
        true,
        &mut out_flash_causal,
    );

    for (i, (&std_val, &flash_val)) in out_std_causal
        .iter()
        .zip(out_flash_causal.iter())
        .enumerate()
    {
        let diff = (std_val - flash_val).abs();
        assert!(
            diff <= 1e-4,
            "Causal mismatch at {}: std {} vs flash {}, diff {}",
            i,
            std_val,
            flash_val,
            diff
        );
    }
}
#[allow(clippy::too_many_arguments)]
fn reference_mha(
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
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let mut scores = vec![0.0f32; t_kv];

    for h in 0..n_heads {
        let head_offset = h * d_head;
        for i in 0..t_q {
            let q_vec = &q[i * d_model + head_offset..i * d_model + head_offset + d_head];
            let max_j = if causal { i + 1 } else { t_kv };

            for (j, score_entry) in scores[..max_j].iter_mut().enumerate() {
                let k_vec = &k[j * d_model + head_offset..j * d_model + head_offset + d_head];
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    dot += q_vec[d] * k_vec[d];
                }
                *score_entry = dot * scale;
            }

            whisper_tensor::ops::softmax_inplace(&mut scores[..max_j]);

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
