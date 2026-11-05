use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn layer_norm_row_avx2(
    in_slice: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    d_model: usize,
    out_slice: &mut [f32],
) {
    use crate::simd::hsum256_ps;
    use core::arch::x86_64::*;

    let n_blocks = d_model / 8;
    let in_ptr = in_slice.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let beta_ptr = beta.as_ptr();
    let out_ptr = out_slice.as_mut_ptr();

    // 1. Mean
    let mut acc_sum = _mm256_setzero_ps();
    for i in 0..n_blocks {
        acc_sum = _mm256_add_ps(acc_sum, _mm256_loadu_ps(in_ptr.add(i * 8)));
    }
    let mut sum = hsum256_ps(acc_sum);
    for i in (n_blocks * 8)..d_model {
        sum += *in_ptr.add(i);
    }
    let mean = sum / (d_model as f32);
    let v_mean = _mm256_set1_ps(mean);

    // 2. Variance
    let mut acc_var = _mm256_setzero_ps();
    for i in 0..n_blocks {
        let diff = _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i * 8)), v_mean);
        acc_var = _mm256_fmadd_ps(diff, diff, acc_var);
    }
    let mut sum_sq = hsum256_ps(acc_var);
    for i in (n_blocks * 8)..d_model {
        let diff = *in_ptr.add(i) - mean;
        sum_sq += diff * diff;
    }
    let variance = sum_sq / (d_model as f32);
    let inv_std = 1.0f32 / (variance + eps).sqrt();
    let v_inv_std = _mm256_set1_ps(inv_std);

    // 3. Normalize
    for i in 0..n_blocks {
        let x = _mm256_loadu_ps(in_ptr.add(i * 8));
        let g = _mm256_loadu_ps(gamma_ptr.add(i * 8));
        let b = _mm256_loadu_ps(beta_ptr.add(i * 8));

        let diff = _mm256_sub_ps(x, v_mean);
        let norm = _mm256_mul_ps(diff, v_inv_std);
        let res = _mm256_fmadd_ps(norm, g, b);

        _mm256_storeu_ps(out_ptr.add(i * 8), res);
    }
    for i in (n_blocks * 8)..d_model {
        *out_ptr.add(i) = (*in_ptr.add(i) - mean) * inv_std * *gamma_ptr.add(i) + *beta_ptr.add(i);
    }
}

#[inline(always)]
fn layer_norm_row_scalar(
    in_slice: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    d_model: usize,
    out_slice: &mut [f32],
) {
    let mut sum = 0.0f32;
    for &x in in_slice {
        sum += x;
    }
    let mean = sum / (d_model as f32);

    let mut sum_sq = 0.0f32;
    for &x in in_slice {
        let diff = x - mean;
        sum_sq += diff * diff;
    }
    let variance = sum_sq / (d_model as f32);
    let inv_std = 1.0 / (variance + eps).sqrt();

    for i in 0..d_model {
        out_slice[i] = (in_slice[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

#[inline(always)]
fn layer_norm_row(
    in_slice: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    d_model: usize,
    out_slice: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                layer_norm_row_avx2(in_slice, gamma, beta, eps, d_model, out_slice);
            }
            return;
        }
    }

    layer_norm_row_scalar(in_slice, gamma, beta, eps, d_model, out_slice);
}

pub fn layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    num_tokens: usize,
    d_model: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), num_tokens * d_model);
    assert_eq!(output.len(), num_tokens * d_model);
    assert_eq!(gamma.len(), d_model);
    assert_eq!(beta.len(), d_model);

    if num_tokens > 1 {
        output
            .par_chunks_mut(d_model)
            .enumerate()
            .for_each(|(t, out_slice)| {
                let in_slice = &input[t * d_model..(t + 1) * d_model];
                layer_norm_row(in_slice, gamma, beta, eps, d_model, out_slice);
            });
        return;
    }

    layer_norm_row(input, gamma, beta, eps, d_model, output);
}
