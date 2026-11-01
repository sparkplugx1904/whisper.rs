pub fn softmax_inplace(slice: &mut [f32]) {
    if slice.is_empty() {
        return;
    }

    let mut max_val = f32::NEG_INFINITY;
    for &x in slice.iter() {
        if x > max_val {
            max_val = x;
        }
    }

    let mut sum = 0.0f32;
    for x in slice.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    let inv_sum = 1.0 / sum;
    for x in slice.iter_mut() {
        *x *= inv_sum;
    }
}
