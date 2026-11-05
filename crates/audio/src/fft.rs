use rustfft::{num_complex::Complex32, FftPlanner};
use std::sync::Arc;

pub const N_FFT: usize = 400;

#[derive(Clone)]
pub struct Fft400 {
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl Default for Fft400 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fft400 {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self { fft }
    }

    /// Compute FFT on real input [N_FFT], writing complex output to [2 * N_FFT] as [re, im, re, im, ...].
    pub fn compute(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), N_FFT);
        assert_eq!(output.len(), N_FFT * 2);

        let mut buffer = [Complex32::new(0.0, 0.0); N_FFT];
        for i in 0..N_FFT {
            buffer[i] = Complex32::new(input[i], 0.0);
        }

        self.fft.process(&mut buffer);

        for (i, c) in buffer.iter().enumerate() {
            output[2 * i] = c.re;
            output[2 * i + 1] = c.im;
        }
    }
}
