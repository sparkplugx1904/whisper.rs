pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    num_tokens: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), num_tokens * in_features);
    assert_eq!(weight.len(), out_features * in_features);
    assert_eq!(output.len(), num_tokens * out_features);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features);
    }

    for t in 0..num_tokens {
        let in_row = &input[t * in_features..(t + 1) * in_features];
        let out_row = &mut output[t * out_features..(t + 1) * out_features];

        for r in 0..out_features {
            let w_row = &weight[r * in_features..(r + 1) * in_features];
            let mut dot = 0.0f32;
            for c in 0..in_features {
                dot += in_row[c] * w_row[c];
            }
            if let Some(b) = bias {
                dot += b[r];
            }
            out_row[r] = dot;
        }
    }
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (val_a, &val_b) in a.iter_mut().zip(b) {
        *val_a += val_b;
    }
}
