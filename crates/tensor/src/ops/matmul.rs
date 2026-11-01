use crate::simd::dot_f32;

pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    num_tokens: usize,
    output: &mut [f32],
) {
    linear_parallel(
        input,
        weight,
        bias,
        in_features,
        out_features,
        num_tokens,
        output,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn linear_parallel(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    num_tokens: usize,
    output: &mut [f32],
    n_threads: usize,
) {
    assert_eq!(input.len(), num_tokens * in_features);
    assert_eq!(weight.len(), out_features * in_features);
    assert_eq!(output.len(), num_tokens * out_features);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features);
    }

    let n_threads = n_threads.clamp(1, 16);

    // If single thread or very small problem, compute sequentially
    if n_threads == 1 || (num_tokens == 1 && out_features < 256) {
        linear_sequential(
            input,
            weight,
            bias,
            in_features,
            out_features,
            0..num_tokens,
            0..out_features,
            output,
        );
        return;
    }

    if num_tokens >= n_threads {
        // Parallelize across tokens
        let chunk_size = num_tokens.div_ceil(n_threads);
        std::thread::scope(|s| {
            for (t_start, out_chunk) in output
                .chunks_mut(chunk_size * out_features)
                .enumerate()
                .map(|(i, c)| (i * chunk_size, c))
            {
                let t_count = out_chunk.len() / out_features;
                let in_chunk = &input[t_start * in_features..(t_start + t_count) * in_features];

                s.spawn(move || {
                    for t in 0..t_count {
                        let in_row = &in_chunk[t * in_features..(t + 1) * in_features];
                        let out_row = &mut out_chunk[t * out_features..(t + 1) * out_features];

                        for r in 0..out_features {
                            let w_row = &weight[r * in_features..(r + 1) * in_features];
                            let mut dot = dot_f32(in_row, w_row);
                            if let Some(b) = bias {
                                dot += b[r];
                            }
                            out_row[r] = dot;
                        }
                    }
                });
            }
        });
    } else {
        // Parallelize across out_features (e.g. single-token decoding with large vocab)
        let chunk_size = out_features.div_ceil(n_threads);
        std::thread::scope(|s| {
            for r_start in (0..out_features).step_by(chunk_size) {
                let r_end = (r_start + chunk_size).min(out_features);
                let w_sub = &weight[r_start * in_features..r_end * in_features];
                let b_sub = bias.map(|b| &b[r_start..r_end]);

                // We write into output at [t * out_features + r_start .. t * out_features + r_end]
                // Since threads write disjoint slices of each row, we use raw pointer or unsafe slice
                let out_ptr = output.as_mut_ptr() as usize;

                s.spawn(move || {
                    let out_slice_ptr = out_ptr as *mut f32;
                    let r_count = r_end - r_start;

                    for t in 0..num_tokens {
                        let in_row = &input[t * in_features..(t + 1) * in_features];
                        for r_rel in 0..r_count {
                            let r_abs = r_start + r_rel;
                            let w_row = &w_sub[r_rel * in_features..(r_rel + 1) * in_features];
                            let mut dot = dot_f32(in_row, w_row);
                            if let Some(b) = b_sub {
                                dot += b[r_rel];
                            }
                            unsafe {
                                *out_slice_ptr.add(t * out_features + r_abs) = dot;
                            }
                        }
                    }
                });
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_sequential(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    t_range: std::ops::Range<usize>,
    r_range: std::ops::Range<usize>,
    output: &mut [f32],
) {
    for t in t_range {
        let in_row = &input[t * in_features..(t + 1) * in_features];
        let out_row = &mut output[t * out_features..(t + 1) * out_features];

        for r in r_range.clone() {
            let w_row = &weight[r * in_features..(r + 1) * in_features];
            let mut dot = dot_f32(in_row, w_row);
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
