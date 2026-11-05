use crate::simd::{dot_f32, gemv_f32};
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

    // When num_tokens is small (e.g. single-token decoding), use optimized GEMV microkernels
    if num_tokens == 1 {
        let in_row = &input[..in_features];
        if n_threads > 1 && out_features >= 1024 {
            let chunk_size = out_features.div_ceil(n_threads).div_ceil(4) * 4;
            output
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let r_start = chunk_idx * chunk_size;
                    let r_count = out_chunk.len();
                    let w_chunk = &weight[r_start * in_features..(r_start + r_count) * in_features];
                    let b_chunk = bias.map(|b| &b[r_start..r_start + r_count]);
                    gemv_f32(in_row, w_chunk, b_chunk, out_chunk, in_features, r_count);
                });
        } else {
            gemv_f32(in_row, weight, bias, output, in_features, out_features);
        }
        return;
    }

    if num_tokens < 16 {
        for t in 0..num_tokens {
            let in_row = &input[t * in_features..(t + 1) * in_features];
            let out_token = &mut output[t * out_features..(t + 1) * out_features];
            if n_threads > 1 && out_features >= 1024 {
                let chunk_size = out_features.div_ceil(n_threads).div_ceil(4) * 4;
                out_token
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let r_start = chunk_idx * chunk_size;
                        let r_count = out_chunk.len();
                        let w_chunk = &weight[r_start * in_features..(r_start + r_count) * in_features];
                        let b_chunk = bias.map(|b| &b[r_start..r_start + r_count]);
                        gemv_f32(in_row, w_chunk, b_chunk, out_chunk, in_features, r_count);
                    });
            } else {
                gemv_f32(in_row, weight, bias, out_token, in_features, out_features);
            }
        }
        return;
    }

    // Parallelize across token chunks using Rayon and BLIS-optimized GEMM microkernels
    let chunk_size = num_tokens.div_ceil(n_threads);
    output
        .par_chunks_mut(chunk_size * out_features)
        .enumerate()
        .for_each(|(i, out_chunk)| {
            let t_start = i * chunk_size;
            let t_count = out_chunk.len() / out_features;
            let in_chunk = &input[t_start * in_features..(t_start + t_count) * in_features];

            unsafe {
                matrixmultiply::sgemm(
                    t_count,
                    in_features,
                    out_features,
                    1.0,
                    in_chunk.as_ptr(),
                    in_features as isize,
                    1,
                    weight.as_ptr(),
                    1,
                    in_features as isize,
                    0.0,
                    out_chunk.as_mut_ptr(),
                    out_features as isize,
                    1,
                );
            }

            if let Some(b) = bias {
                for t in 0..t_count {
                    let out_row = &mut out_chunk[t * out_features..(t + 1) * out_features];
                    for (r, &b_val) in b.iter().enumerate() {
                        out_row[r] += b_val;
                    }
                }
            }
        });
}
#[allow(clippy::too_many_arguments, dead_code)]
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
    if a.len() >= 8192 {
        a.par_chunks_mut(4096)
            .zip(b.par_chunks(4096))
            .for_each(|(a_chunk, b_chunk)| {
                for (val_a, &val_b) in a_chunk.iter_mut().zip(b_chunk) {
                    *val_a += val_b;
                }
            });
    } else {
        for (val_a, &val_b) in a.iter_mut().zip(b) {
            *val_a += val_b;
        }
    }
}
