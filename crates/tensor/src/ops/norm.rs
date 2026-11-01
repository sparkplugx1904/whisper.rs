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

    for t in 0..num_tokens {
        let in_slice = &input[t * d_model..(t + 1) * d_model];
        let out_slice = &mut output[t * d_model..(t + 1) * d_model];

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
}
