use std::f32::consts::PI;

pub const N_FFT: usize = 400;

pub struct Fft400 {
    sin_vals: [f32; N_FFT],
    cos_vals: [f32; N_FFT],
}

impl Default for Fft400 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fft400 {
    pub fn new() -> Self {
        let mut sin_vals = [0.0f32; N_FFT];
        let mut cos_vals = [0.0f32; N_FFT];

        for i in 0..N_FFT {
            let theta = (2.0 * (PI as f64) * (i as f64)) / (N_FFT as f64);
            sin_vals[i] = theta.sin() as f32;
            cos_vals[i] = theta.cos() as f32;
        }

        Self { sin_vals, cos_vals }
    }

    fn dft(&self, input: &[f32], n: usize, out: &mut [f32]) {
        let pi_2_n = 2.0 * PI / (n as f32);
        for k in 0..n {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (m, &val) in input.iter().enumerate().take(n) {
                let angle = (m * k) as f32 * pi_2_n;
                re += val * angle.cos();
                im -= val * angle.sin();
            }
            out[k * 2] = re;
            out[k * 2 + 1] = im;
        }
    }

    fn fft_recursive(&self, input: &[f32], n: usize, out: &mut [f32]) {
        if n == 1 {
            out[0] = input[0];
            out[1] = 0.0;
            return;
        }

        let half_n = n / 2;
        if !n.is_multiple_of(2) {
            self.dft(input, n, out);
            return;
        }

        let mut even = vec![0.0f32; half_n];
        let mut odd = vec![0.0f32; half_n];
        for i in 0..half_n {
            even[i] = input[2 * i];
            odd[i] = input[2 * i + 1];
        }

        let mut even_fft = vec![0.0f32; 2 * half_n];
        let mut odd_fft = vec![0.0f32; 2 * half_n];

        self.fft_recursive(&even, half_n, &mut even_fft);
        self.fft_recursive(&odd, half_n, &mut odd_fft);

        let sin_cos_step = N_FFT / n;
        for k in 0..half_n {
            let idx = k * sin_cos_step;
            let re = self.cos_vals[idx];
            let im = -self.sin_vals[idx];

            let re_odd = odd_fft[2 * k];
            let im_odd = odd_fft[2 * k + 1];

            out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
            out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

            out[2 * (k + half_n)] = even_fft[2 * k] - re * re_odd + im * im_odd;
            out[2 * (k + half_n) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
        }
    }

    pub fn compute(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), N_FFT);
        assert_eq!(output.len(), N_FFT * 2);
        self.fft_recursive(input, N_FFT, output);
    }
}
