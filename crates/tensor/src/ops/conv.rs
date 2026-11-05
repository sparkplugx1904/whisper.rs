use crate::ops::gelu::gelu_scalar;
use rayon::prelude::*;

/// 1D convolution over sequence length for input with shape [in_channels, L_in].
#[allow(clippy::too_many_arguments)]
pub fn conv1d_c_first(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    out_channels: usize,
    l_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    apply_gelu: bool,
    output: &mut [f32],
) {
    let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
    assert_eq!(input.len(), in_channels * l_in);
    assert_eq!(weight.len(), out_channels * in_channels * kernel_size);
    assert_eq!(bias.len(), out_channels);
    assert_eq!(output.len(), out_channels * l_out);
    let k_total = in_channels * kernel_size;
    let mut x_col = vec![0.0f32; k_total * l_out];

    // Construct im2col matrix of shape [k_total, l_out]
    for cin in 0..in_channels {
        let in_cin_offset = cin * l_in;
        for k in 0..kernel_size {
            let p = cin * kernel_size + k;
            let row = &mut x_col[p * l_out..(p + 1) * l_out];
            for (tout, val) in row.iter_mut().enumerate() {
                let tin = (tout * stride) as isize - (padding as isize) + k as isize;
                if tin >= 0 && (tin as usize) < l_in {
                    *val = input[in_cin_offset + (tin as usize)];
                }
            }
        }
    }

    // Output [out_channels, l_out] = Weight [out_channels, k_total] * X_col [k_total, l_out]
    unsafe {
        matrixmultiply::sgemm(
            out_channels,
            k_total,
            l_out,
            1.0,
            weight.as_ptr(),
            k_total as isize,
            1,
            x_col.as_ptr(),
            l_out as isize,
            1,
            0.0,
            output.as_mut_ptr(),
            l_out as isize,
            1,
        );
    }

    // Bias addition and optional GELU activation in parallel
    output
        .par_chunks_mut(l_out)
        .enumerate()
        .for_each(|(cout, row)| {
            let b = bias[cout];
            if apply_gelu {
                for val in row.iter_mut() {
                    *val = gelu_scalar(*val + b);
                }
            } else {
                for val in row.iter_mut() {
                    *val += b;
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_t_first(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    out_channels: usize,
    l_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    apply_gelu: bool,
    output: &mut [f32],
) {
    let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
    assert_eq!(input.len(), in_channels * l_in);
    assert_eq!(weight.len(), out_channels * in_channels * kernel_size);
    assert_eq!(bias.len(), out_channels);
    assert_eq!(output.len(), l_out * out_channels);
    let k_total = in_channels * kernel_size;
    let mut x_col = vec![0.0f32; l_out * k_total];

    // Construct im2col matrix of shape [l_out, k_total] in parallel
    x_col
        .par_chunks_mut(k_total)
        .enumerate()
        .for_each(|(tout, row)| {
            let tin_center = (tout * stride) as isize - (padding as isize);
            for cin in 0..in_channels {
                let in_cin_offset = cin * l_in;
                for k in 0..kernel_size {
                    let tin = tin_center + k as isize;
                    if tin >= 0 && (tin as usize) < l_in {
                        row[cin * kernel_size + k] = input[in_cin_offset + (tin as usize)];
                    }
                }
            }
        });

    // Output [l_out, out_channels] = X_col [l_out, k_total] * Weight^T [k_total, out_channels]
    unsafe {
        matrixmultiply::sgemm(
            l_out,
            k_total,
            out_channels,
            1.0,
            x_col.as_ptr(),
            k_total as isize,
            1,
            weight.as_ptr(),
            1,
            k_total as isize,
            0.0,
            output.as_mut_ptr(),
            out_channels as isize,
            1,
        );
    }

    // Bias addition and optional GELU activation in parallel
    output.par_chunks_mut(out_channels).for_each(|row| {
        if apply_gelu {
            for (cout, val) in row.iter_mut().enumerate() {
                *val = gelu_scalar(*val + bias[cout]);
            }
        } else {
            for (cout, val) in row.iter_mut().enumerate() {
                *val += bias[cout];
            }
        }
    });
}
