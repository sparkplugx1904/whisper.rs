//! Native f16 weight storage: kernel correctness and memory-bandwidth microbenchmarks.

use std::hint::black_box;
use std::time::Instant;

use half::f16;
use whisper_tensor::simd::{dot_f16_f32, dot_f16_f32_scalar};
use whisper_tensor::weight::LinearWeight;
use whisper_tensor::DType;

fn packed_halves(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

fn deterministic(i: usize) -> f32 {
    // Cheap deterministic pattern in [-1, 1); no floating point transcendentals
    // so setup time stays negligible next to the measured kernels.
    let v = (i.wrapping_mul(2_654_435_761) >> 8) & 0xFFFF;
    v as f32 / 32768.0 - 1.0
}

#[test]
fn f16_dot_matches_scalar_and_explicit_reference() {
    // Sweeps every tail length around the 8-lane convert step and the 32-element
    // unrolled body, so the AVX2 path and the scalar fallback agree exactly on
    // rows that are not multiples of the SIMD width.
    for len in 1..=70usize {
        let values: Vec<f32> = (0..len).map(|i| deterministic(i + 17)).collect();
        let bytes = packed_halves(&values);
        let x: Vec<f32> = (0..len).map(|i| deterministic(i * 3 + 1)).collect();

        let expected: f32 = values
            .iter()
            .zip(&x)
            .map(|(v, a)| f16::from_f32(*v).to_f32() * a)
            .sum();
        let vector = dot_f16_f32(&bytes, &x);
        let scalar = dot_f16_f32_scalar(&bytes, &x);

        assert!(
            (vector - expected).abs() <= 1e-5,
            "len {len}: vector {vector} vs reference {expected}"
        );
        assert!(
            (scalar - expected).abs() <= 1e-5,
            "len {len}: scalar {scalar} vs reference {expected}"
        );
        assert!(
            (vector - scalar).abs() <= 1e-5,
            "len {len}: vector {vector} vs scalar {scalar}"
        );
    }
}

#[test]
fn f16_linear_handles_ragged_shapes() {
    // Rows and columns deliberately avoid multiples of 4 and 8, and the row
    // count does not divide evenly across the default thread count.
    for &(rows, cols) in &[(3usize, 9usize), (11, 21), (5, 67), (13, 1024)] {
        let values: Vec<f32> = (0..rows * cols).map(|i| deterministic(i + 5)).collect();
        let bytes = packed_halves(&values);
        let weight = LinearWeight::from_f16(rows, cols, bytes);

        let x: Vec<f32> = (0..cols).map(|i| deterministic(i * 7 + 2)).collect();
        let mut out = vec![0.0f32; rows];
        weight.gemv(&x, None, &mut out);

        for r in 0..rows {
            let expected: f32 = (0..cols)
                .map(|k| f16::from_f32(values[r * cols + k]).to_f32() * x[k])
                .sum();
            assert!(
                (out[r] - expected).abs() <= 1e-4,
                "{rows}x{cols} row {r}: {} vs {}",
                out[r],
                expected
            );
        }
    }
}

#[test]
fn f16_residency_is_half_of_f32() {
    let (rows, cols) = (37usize, 129usize);
    let values: Vec<f32> = (0..rows * cols).map(deterministic).collect();
    let dense = LinearWeight::from_f32(rows, cols, values.clone());
    let half = LinearWeight::from_f16(rows, cols, packed_halves(&values));

    assert_eq!(half.dtype(), DType::F16);
    assert_eq!(half.rows(), rows);
    assert_eq!(half.cols(), cols);
    assert_eq!(half.resident_bytes() * 2, dense.resident_bytes());
}

fn cool_down() {
    // Timed variants run strictly in sequence; let the core's turbo state and
    // the memory controller settle between contestants.
    std::thread::sleep(std::time::Duration::from_secs(2));
}

#[test]
#[ignore = "performance microbenchmark; run with --release -- --ignored --nocapture"]
fn f16_vs_f32_gemv_bandwidth() {
    println!("\nWeight-stream benchmark: f32 vs native f16, effective weight bandwidth");
    println!(
        "(GB/s = weight bytes touched per call / elapsed; M>1 f32 dispatch re-reads the matrix)"
    );

    for &d in &[1280usize, 5120usize] {
        let values: Vec<f32> = (0..d * d).map(deterministic).collect();
        let dense = LinearWeight::from_f32(d, d, values.clone());
        let halves = LinearWeight::from_f16(d, d, packed_halves(&values));

        // Large matrices move ~26 MB per call (f16) or ~52 MB (f32), so a few
        // iterations already saturate the memory controllers; small matrices
        // need more repetitions to average out timer noise.
        let iters = if d <= 1280 { 40 } else { 6 };
        let f32_bytes = (d * d * std::mem::size_of::<f32>()) as f64;
        let f16_bytes = (d * d * 2) as f64;

        for num_tokens in [1usize, 8] {
            let x: Vec<f32> = (0..num_tokens * d).map(|i| deterministic(i + 9)).collect();
            let mut out = vec![0.0f32; num_tokens * d];

            // Warm both kernels and the allocator before timing.
            dense.linear(&x, None, num_tokens, &mut out);
            halves.linear(&x, None, num_tokens, &mut out);

            cool_down();
            let f32_elapsed = {
                let start = Instant::now();
                for _ in 0..iters {
                    dense.linear(black_box(&x), None, num_tokens, black_box(&mut out));
                }
                start.elapsed()
            };
            cool_down();
            let f16_elapsed = {
                let start = Instant::now();
                for _ in 0..iters {
                    halves.linear(black_box(&x), None, num_tokens, black_box(&mut out));
                }
                start.elapsed()
            };

            let f32_per_call = f32_elapsed.as_secs_f64() * 1e3 / iters as f64;
            let f16_per_call = f16_elapsed.as_secs_f64() * 1e3 / iters as f64;
            let f32_gbs = f32_bytes * iters as f64 / f32_elapsed.as_secs_f64() / 1e9;
            let f16_gbs = f16_bytes * iters as f64 / f16_elapsed.as_secs_f64() / 1e9;

            println!(
                "  d_model={d:5} M={num_tokens}: f32 {f32_per_call:8.3} ms  {f32_gbs:6.2} GB/s | \
                 f16 {f16_per_call:8.3} ms  {f16_gbs:6.2} GB/s | speedup {:.2}x",
                f32_per_call / f16_per_call
            );

            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
