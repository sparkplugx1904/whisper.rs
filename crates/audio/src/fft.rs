use rustfft::{num_complex::Complex32, FftPlanner};
use std::sync::{Arc, LazyLock};

static SHARED_FFT: LazyLock<Fft400> = LazyLock::new(Fft400::new);

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

    /// Returns a reference to a globally cached `Fft400` instance, avoiding repeated planning.
    pub fn shared() -> &'static Self {
        &SHARED_FFT
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft400_shared() {
        let fft1 = Fft400::shared();
        let fft2 = Fft400::shared();
        assert!(Arc::ptr_eq(&fft1.fft, &fft2.fft));

        let mut input = [0.0f32; N_FFT];
        input[0] = 1.0;
        let mut out = [0.0f32; N_FFT * 2];
        fft1.compute(&input, &mut out);

        for i in 0..N_FFT {
            assert!((out[2 * i] - 1.0).abs() < 1e-5);
            assert!(out[2 * i + 1].abs() < 1e-5);
        }
    }
}
