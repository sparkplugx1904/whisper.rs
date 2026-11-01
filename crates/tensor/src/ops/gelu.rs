pub const GELU_COEF_A: f32 = 0.044715;
pub const SQRT_2_OVER_PI: f32 = 0.797_884_6;

#[inline(always)]
pub fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

pub fn gelu_inplace(data: &mut [f32]) {
    for val in data.iter_mut() {
        *val = gelu_scalar(*val);
    }
}
