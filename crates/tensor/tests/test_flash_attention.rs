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
    multi_head_attention(&q, &k, &v, t_q, t_kv, n_heads, d_model, false, &mut out_std);

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
    multi_head_attention(
        &q,
        &k,
        &v,
        t_q,
        t_kv,
        n_heads,
        d_model,
        true,
        &mut out_std_causal,
    );

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
