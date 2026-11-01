use crate::simd::dot_f32;
use rayon::prelude::*;

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

    // When num_tokens is small (e.g. single-token decoding), run sequentially to eliminate thread dispatch overhead
    if n_threads == 1 || num_tokens < 16 {
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

    // Parallelize across tokens using Rayon's thread pool
    let chunk_size = num_tokens.div_ceil(n_threads);
    output
        .par_chunks_mut(chunk_size * out_features)
        .enumerate()
        .for_each(|(i, out_chunk)| {
            let t_start = i * chunk_size;
            let t_count = out_chunk.len() / out_features;
            let in_chunk = &input[t_start * in_features..(t_start + t_count) * in_features];

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
