use crate::ops::gelu::gelu_scalar;

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

    for cout in 0..out_channels {
        let b = bias[cout];
        let w_cout_offset = cout * (in_channels * kernel_size);

        for tout in 0..l_out {
            let tin_center = (tout * stride) as isize - (padding as isize);
            let mut sum = 0.0f32;

            for cin in 0..in_channels {
                let w_cin_offset = w_cout_offset + cin * kernel_size;
                let in_cin_offset = cin * l_in;

                for k in 0..kernel_size {
                    let tin = tin_center + k as isize;
                    if tin >= 0 && (tin as usize) < l_in {
                        let x = input[in_cin_offset + (tin as usize)];
                        let w = weight[w_cin_offset + k];
                        sum += x * w;
                    }
                }
            }

            let val = if apply_gelu {
                gelu_scalar(sum + b)
            } else {
                sum + b
            };

            output[cout * l_out + tout] = val;
        }
    }
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

    for tout in 0..l_out {
        let tin_center = (tout * stride) as isize - (padding as isize);

        for cout in 0..out_channels {
            let b = bias[cout];
            let w_cout_offset = cout * (in_channels * kernel_size);
            let mut sum = 0.0f32;

            for cin in 0..in_channels {
                let w_cin_offset = w_cout_offset + cin * kernel_size;
                let in_cin_offset = cin * l_in;

                for k in 0..kernel_size {
                    let tin = tin_center + k as isize;
                    if tin >= 0 && (tin as usize) < l_in {
                        let x = input[in_cin_offset + (tin as usize)];
                        let w = weight[w_cin_offset + k];
                        sum += x * w;
                    }
                }
            }

            let val = if apply_gelu {
                gelu_scalar(sum + b)
            } else {
                sum + b
            };

            output[tout * out_channels + cout] = val;
        }
    }
}
