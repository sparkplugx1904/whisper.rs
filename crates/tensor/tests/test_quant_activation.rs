//! Parity and throughput checks for the Q8-activation integer kernels.
//!
//! The reference is the pre-existing f32-activation path: the same weight
//! bytes fed through `vec_dot_q*_f32`, which multiplies dequantized weights by
//! the exact float activation. Comparing against it isolates the error that
//! activation quantization adds.

use std::time::Instant;

use whisper_tensor::dtype::DType;
use whisper_tensor::ops::quant::{has_q8_activation_kernel, Q8Activation};
use whisper_tensor::ops::{
    quantized_linear, quantized_linear_vector, quantized_matmul, vec_dot_q4_0_f32,
    vec_dot_q4_1_f32, vec_dot_q5_0_f32, vec_dot_q5_1_f32, vec_dot_q8_0_f32,
};
use whisper_tensor::simd::{
    dot_q4_0_q8, dot_q4_0_q8_scalar, dot_q4_1_q8, dot_q4_1_q8_scalar, dot_q5_0_q8,
    dot_q5_0_q8_scalar, dot_q5_1_q8, dot_q5_1_q8_scalar, dot_q8_0_q8, dot_q8_0_q8_scalar,
};

const DTYPES: [DType; 5] = [
    DType::Q4_0,
    DType::Q4_1,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q8_0,
];

/// splitmix64: tiny, deterministic, no external dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)` with 24 bits of mantissa.
    fn uniform(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 23) as f32) - 1.0
    }

    fn fill(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.uniform()).collect()
    }
}

fn f32_reference(dtype: DType, row: &[u8], x: &[f32]) -> f32 {
    match dtype {
        DType::Q4_0 => vec_dot_q4_0_f32(row, x),
        DType::Q4_1 => vec_dot_q4_1_f32(row, x),
        DType::Q5_0 => vec_dot_q5_0_f32(row, x),
        DType::Q5_1 => vec_dot_q5_1_f32(row, x),
        DType::Q8_0 => vec_dot_q8_0_f32(row, x),
        other => panic!("no f32-activation reference for {other:?}"),
    }
}

#[derive(Debug)]
struct Metrics {
    max_abs: f32,
    /// Relative L2 error over the whole output vector. A per-element ratio is
    /// meaningless here because individual dot products can cancel to zero.
    rel_l2: f32,
    cosine: f32,
}

fn metrics(got: &[f32], want: &[f32]) -> Metrics {
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0.0f32;
    let mut diff_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut got_sq = 0.0f64;
    let mut want_sq = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        let g64 = g as f64;
        let w64 = w as f64;
        let diff = (g - w).abs();
        max_abs = max_abs.max(diff);
        diff_sq += ((g - w) as f64) * ((g - w) as f64);
        dot += g64 * w64;
        got_sq += g64 * g64;
        want_sq += w64 * w64;
    }
    let rel_l2 = if want_sq > 0.0 {
        (diff_sq / want_sq).sqrt() as f32
    } else {
        0.0
    };
    let cosine = if got_sq > 0.0 && want_sq > 0.0 {
        (dot / (got_sq.sqrt() * want_sq.sqrt())) as f32
    } else {
        1.0
    };
    Metrics {
        max_abs,
        rel_l2,
        cosine,
    }
}

/// Aggregated error statistics over every shape tested for one dtype.
#[derive(Default)]
struct Accum {
    diff_sq: f64,
    want_sq: f64,
    got_sq: f64,
    dot: f64,
    max_abs: f32,
}

impl Accum {
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, got: &[f32], want: &[f32], m: &Metrics) {
        for (&g, &w) in got.iter().zip(want) {
            self.diff_sq += ((g - w) as f64) * ((g - w) as f64);
            self.want_sq += (w as f64) * (w as f64);
            self.got_sq += (g as f64) * (g as f64);
            self.dot += (g as f64) * (w as f64);
        }
        self.max_abs = self.max_abs.max(m.max_abs);
    }

    fn rel_l2(&self) -> f32 {
        (self.diff_sq / self.want_sq).sqrt() as f32
    }

    /// Cosine similarity over every output element seen so far.
    fn cosine(&self) -> f32 {
        (self.dot / (self.got_sq.sqrt() * self.want_sq.sqrt())) as f32
    }
}

fn quantize_weights(values: &[f32], dtype: DType) -> Vec<u8> {
    dtype
        .quantize_from_f32(values)
        .expect("plain block formats must quantize")
}

/// Quantized-activation GEMV must track the f32-activation kernels for every
/// plain block format, including row counts that do not divide the worker
/// chunks and column counts that are one block wide.
#[test]
fn quantized_activation_matches_f32_activation() {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut overall = Accum::new();

    for &dtype in &DTYPES {
        let mut acc = Accum::new();
        for &cols in &[32usize, 64, 96, 160, 256, 1280] {
            for &rows in &[1usize, 3, 7, 33] {
                let weights = rng.fill(rows * cols);
                let raw = quantize_weights(&weights, dtype);
                let bytes_per_row = dtype.byte_size_for(cols);
                let x = rng.fill(cols);
                let bias = rng.fill(rows);

                let mut got = vec![0.0f32; rows];
                quantized_linear_vector(&raw, dtype, rows, cols, &x, Some(&bias), &mut got, 3);

                let want: Vec<f32> = (0..rows)
                    .map(|r| {
                        let row = &raw[r * bytes_per_row..(r + 1) * bytes_per_row];
                        f32_reference(dtype, row, &x) + bias[r]
                    })
                    .collect();

                let m = metrics(&got, &want);
                println!(
                    "{dtype:?} rows={rows} cols={cols}: max_abs={:.3e} rel_l2={:.3e} cos={:.8}",
                    m.max_abs, m.rel_l2, m.cosine
                );
                assert!(
                    m.max_abs < 0.5,
                    "{dtype:?} rows={rows} cols={cols}: absolute error {} too large",
                    m.max_abs
                );
                acc.add(&got, &want, &m);
                overall.add(&got, &want, &m);
            }
        }

        println!(
            "{dtype:?} aggregate: max_abs={:.3e} rel_l2={:.3e} cos={:.8}",
            acc.max_abs,
            acc.rel_l2(),
            acc.cosine()
        );
        assert!(
            acc.rel_l2() < 1e-2,
            "{dtype:?}: aggregate relative L2 error {} too large",
            acc.rel_l2()
        );
        assert!(
            acc.cosine() > 0.9999,
            "{dtype:?}: aggregate cosine {} too low",
            acc.cosine()
        );
    }

    println!(
        "worst over all shapes: max_abs={:.3e} rel_l2={:.3e} cos={:.8}",
        overall.max_abs,
        overall.rel_l2(),
        overall.cosine()
    );
}

/// Multi-token `quantized_linear` reuses the quantized activation per token and
/// must agree with the f32-activation path for small beams and large batches
/// alike (the batch that used to take the dequantize-then-GEMM shortcut).
#[test]
fn quantized_activation_batched_matches_f32_activation() {
    let mut rng = Rng(0x5EED_1234_ABCD_0002);

    for &dtype in &DTYPES {
        let mut acc = Accum::new();
        for &num_tokens in &[2usize, 5, 8, 40] {
            for &cols in &[64usize, 160, 1280] {
                let rows = 11;
                let weights = rng.fill(rows * cols);
                let raw = quantize_weights(&weights, dtype);
                let bytes_per_row = dtype.byte_size_for(cols);
                let x = rng.fill(num_tokens * cols);
                let bias = rng.fill(rows);

                let mut got = vec![0.0f32; num_tokens * rows];
                quantized_linear(
                    &raw,
                    dtype,
                    rows,
                    cols,
                    &x,
                    Some(&bias),
                    num_tokens,
                    &mut got,
                );

                let want: Vec<f32> = (0..num_tokens * rows)
                    .map(|i| {
                        let t = i / rows;
                        let r = i % rows;
                        let row = &raw[r * bytes_per_row..(r + 1) * bytes_per_row];
                        f32_reference(dtype, row, &x[t * cols..(t + 1) * cols]) + bias[r]
                    })
                    .collect();

                let m = metrics(&got, &want);
                println!(
                    "{dtype:?} tokens={num_tokens} cols={cols}: max_abs={:.3e} rel_l2={:.3e} cos={:.8}",
                    m.max_abs, m.rel_l2, m.cosine
                );
                assert!(
                    m.max_abs < 0.5,
                    "{dtype:?} tokens={num_tokens} cols={cols}: absolute error {} too large",
                    m.max_abs
                );
                acc.add(&got, &want, &m);
            }
        }

        println!(
            "{dtype:?} batched aggregate: max_abs={:.3e} rel_l2={:.3e} cos={:.8}",
            acc.max_abs,
            acc.rel_l2(),
            acc.cosine()
        );
        assert!(
            acc.rel_l2() < 1e-2,
            "{dtype:?}: batched aggregate relative L2 error {} too large",
            acc.rel_l2()
        );
        assert!(
            acc.cosine() > 0.9999,
            "{dtype:?}: batched aggregate cosine {} too low",
            acc.cosine()
        );
    }
}

/// The scalar kernels are the fallback for non-AVX2 targets, so they must
/// reproduce the dispatched kernels bit for bit.
#[test]
fn scalar_and_vector_kernels_agree() {
    let mut rng = Rng(0x5EED_1234_ABCD_0003);
    let cols = 256;
    let rows = 5;
    let x = rng.fill(cols);
    let act = Q8Activation::quantize(&x).expect("256 is a block multiple");

    for &dtype in &DTYPES {
        let raw = quantize_weights(&rng.fill(rows * cols), dtype);
        let bpr = dtype.byte_size_for(cols);
        for r in 0..rows {
            let row = &raw[r * bpr..(r + 1) * bpr];
            let (fast, scalar) = match dtype {
                DType::Q4_0 => (dot_q4_0_q8(row, &act), dot_q4_0_q8_scalar(row, &act)),
                DType::Q4_1 => (dot_q4_1_q8(row, &act), dot_q4_1_q8_scalar(row, &act)),
                DType::Q5_0 => (dot_q5_0_q8(row, &act), dot_q5_0_q8_scalar(row, &act)),
                DType::Q5_1 => (dot_q5_1_q8(row, &act), dot_q5_1_q8_scalar(row, &act)),
                DType::Q8_0 => (dot_q8_0_q8(row, &act), dot_q8_0_q8_scalar(row, &act)),
                other => panic!("unexpected dtype {other:?}"),
            };
            // Both paths sum the same exact integer products; only the order
            // of the per-block f32 additions can differ.
            let tol = 1e-5 * (1.0 + scalar.abs());
            assert!(
                (fast - scalar).abs() <= tol,
                "{dtype:?} row {r}: vector {fast} vs scalar {scalar}"
            );
        }
    }
}

/// Odd activations must degrade to the f32 path instead of producing garbage.
#[test]
fn activation_quantization_rejects_unsupported_inputs() {
    assert!(Q8Activation::quantize(&[]).is_none());
    assert!(Q8Activation::quantize(&[0.5; 33]).is_none());

    let mut nan = vec![0.25f32; 64];
    nan[17] = f32::NAN;
    assert!(Q8Activation::quantize(&nan).is_none());

    let mut inf = vec![0.25f32; 64];
    inf[63] = f32::INFINITY;
    assert!(Q8Activation::quantize(&inf).is_none());

    // Block-multiple length but non-finite: the GEMV keeps the f32 path, so
    // the result stays NaN instead of panicking or quantizing to nonsense.
    let cols = 64;
    let rows = 3;
    let raw = quantize_weights(&vec![0.5f32; rows * cols], DType::Q4_0);
    let mut out = vec![0.0f32; rows];
    quantized_linear_vector(&raw, DType::Q4_0, rows, cols, &nan, None, &mut out, 2);
    assert!(out.iter().all(|v| v.is_nan()));

    // Only the plain 32-element block formats have integer activation kernels.
    assert!(has_q8_activation_kernel(DType::Q4_0));
    assert!(has_q8_activation_kernel(DType::Q8_0));
    assert!(!has_q8_activation_kernel(DType::Q4_K));
    assert!(!has_q8_activation_kernel(DType::Q6_K));
    assert!(!has_q8_activation_kernel(DType::F16));
}

/// The Q8 activation round trip must stay within one quantization step of the
/// original row, which is the bound the dot kernels rely on.
#[test]
fn activation_round_trip_error_is_bounded() {
    let mut rng = Rng(0x5EED_1234_ABCD_0004);
    let cols = 512;
    let x = rng.fill(cols);
    let act = Q8Activation::quantize(&x).expect("512 is a block multiple");
    assert_eq!(act.len(), cols / 32);

    let mut max_err = 0.0f32;
    for (block, src) in act.blocks().iter().zip(x.as_chunks::<32>().0) {
        let amax = src.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let step = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let mut sum = 0i32;
        let mut block_err = 0.0f32;
        for (&q, &v) in block.qs.iter().zip(src) {
            let dequant = q as f32 * block.d;
            block_err = block_err.max((dequant - v).abs());
            sum += q as i32;
        }
        assert_eq!(sum, block.sum_q);
        assert!((block.d - step).abs() <= 1e-6 * (1.0 + amax));
        assert!(
            block_err <= step * 0.5 + 1e-6 * (1.0 + amax),
            "block error {block_err} exceeds half step {step}"
        );
        max_err = max_err.max(block_err);
    }
    println!("max activation round-trip error: {max_err:.3e}");
}

/// Deterministic throughput probe: one weight matrix per (cols, tokens) pair
/// and a fixed thread count, averaged over enough repetitions to reach ~50M
/// MACs. It prints rather than asserts on timing so it stays stable on shared
/// CI runners.
#[test]
fn microbench_quantized_activation() {
    let mut rng = Rng(0x5EED_1234_ABCD_0005);
    let rows = 256usize;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);

    for &dtype in &[DType::Q4_0, DType::Q8_0] {
        for &cols in &[1280usize, 5120] {
            let weights = rng.fill(rows * cols);
            let raw = quantize_weights(&weights, dtype);
            let bytes_per_row = dtype.byte_size_for(cols);

            for &tokens in &[1usize, 8] {
                let x = rng.fill(tokens * cols);
                let mut out = vec![0.0f32; tokens * rows];
                let reps = (50_000_000 / (rows * cols * tokens)).max(1);

                // Warm the caches and the thread pool.
                if tokens == 1 {
                    quantized_linear_vector(&raw, dtype, rows, cols, &x, None, &mut out, threads);
                } else {
                    quantized_linear(&raw, dtype, rows, cols, &x, None, tokens, &mut out);
                }

                let start = Instant::now();
                for _ in 0..reps {
                    if tokens == 1 {
                        quantized_linear_vector(
                            &raw, dtype, rows, cols, &x, None, &mut out, threads,
                        );
                    } else {
                        quantized_linear(&raw, dtype, rows, cols, &x, None, tokens, &mut out);
                    }
                }
                let q8_secs = start.elapsed().as_secs_f64() / reps as f64;

                let start = Instant::now();
                for _ in 0..reps {
                    quantized_matmul(&raw, dtype, rows, cols, &x, tokens, &mut out, threads);
                }
                let f32_secs = start.elapsed().as_secs_f64() / reps as f64;

                let moved = (rows * bytes_per_row) as f64;
                let gmacs = (rows * cols * tokens) as f64 / q8_secs / 1e9;
                println!(
                    "{dtype:?} rows={rows} cols={cols} M={tokens} threads={threads}: q8_act {:.3} ms ({:.2} GB/s, {:.2} GMAC/s), f32_act {:.3} ms, speedup {:.2}x",
                    q8_secs * 1e3,
                    moved / q8_secs / 1e9,
                    gmacs,
                    f32_secs * 1e3,
                    f32_secs / q8_secs
                );
            }
        }
    }
}
