//! Deterministic microbenchmark for the tiled online-softmax attention kernel.
//!
//! Run with:
//!
//! ```text
//! cargo test -p whisper-tensor --release --test bench_flash_attention_tiled -- --ignored --nocapture
//! ```
//!
//! The comparison baseline is the previous implementation (full score matrix
//! per head, softmax, then score-matrix times V), reproduced below so the
//! reported numbers are measured on the same machine and input.

use std::time::Instant;

use rayon::prelude::*;
use whisper_tensor::ops::attention::{KV_TILE, Q_TILE};
use whisper_tensor::ops::{flash_attention_2, softmax_inplace};

const N_HEADS: usize = 6;
const D_HEAD: usize = 64;
const REPS: usize = 9;
/// `(t_q, t_kv)` pairs: one row of a Whisper encoder (1500 audio frames), a
/// short decoder cross-attention call, and a shape large enough that the
/// materialized score matrix no longer fits in the last-level cache.
const SHAPES: [(usize, usize); 3] = [(1500, 1500), (64, 1500), (4096, 4096)];

struct Rng(u32);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn fill(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }
}

/// The implementation replaced by the tiled kernel: one `t_q x t_kv` score
/// buffer per head, materialized and traversed three times.
#[allow(clippy::too_many_arguments)]
fn materialized_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_q: usize,
    t_kv: usize,
    n_heads: usize,
    d_model: usize,
    output: &mut [f32],
) {
    let d_head = d_model / n_heads;
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let out_ptr = output.as_mut_ptr() as usize;

    (0..n_heads).into_par_iter().for_each(|h| {
        let head_offset = h * d_head;
        let mut scores = vec![0.0f32; t_q * t_kv];

        unsafe {
            matrixmultiply::sgemm(
                t_q,
                d_head,
                t_kv,
                scale,
                q.as_ptr().add(head_offset),
                d_model as isize,
                1,
                k.as_ptr().add(head_offset),
                1,
                d_model as isize,
                0.0,
                scores.as_mut_ptr(),
                t_kv as isize,
                1,
            );
        }

        for i in 0..t_q {
            softmax_inplace(&mut scores[i * t_kv..(i + 1) * t_kv]);
        }

        unsafe {
            matrixmultiply::sgemm(
                t_q,
                t_kv,
                d_head,
                1.0,
                scores.as_ptr(),
                t_kv as isize,
                1,
                v.as_ptr().add(head_offset),
                d_model as isize,
                1,
                0.0,
                (out_ptr as *mut f32).add(head_offset),
                d_model as isize,
                1,
            );
        }
    });
}

/// Times both kernels in alternating order so CPU frequency drift, thermal
/// state and background load bias both measurements equally.
fn time_both<A: FnMut(), B: FnMut()>(mut a: A, mut b: B) -> (f64, f64) {
    let mut best_a = f64::MAX;
    let mut best_b = f64::MAX;
    for rep in 0..REPS {
        if rep % 2 == 0 {
            let start = Instant::now();
            a();
            best_a = best_a.min(start.elapsed().as_secs_f64() * 1e3);

            let start = Instant::now();
            b();
            best_b = best_b.min(start.elapsed().as_secs_f64() * 1e3);
        } else {
            let start = Instant::now();
            b();
            best_b = best_b.min(start.elapsed().as_secs_f64() * 1e3);

            let start = Instant::now();
            a();
            best_a = best_a.min(start.elapsed().as_secs_f64() * 1e3);
        }
    }
    (best_a, best_b)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Runs both kernels on one shape, prints the timings and the buffer sizes,
/// and returns `(tiled_ms, materialized_ms, max_abs_diff)`.
fn run_shape(t_q: usize, t_kv: usize, threads: usize) -> (f64, f64, f32) {
    let d_model = N_HEADS * D_HEAD;
    let mut rng = Rng(0x5eed_1234);
    let q = rng.fill(t_q * d_model);
    let k = rng.fill(t_kv * d_model);
    let v = rng.fill(t_kv * d_model);

    let mut tiled_out = vec![0.0f32; t_q * d_model];
    let mut materialized_out = vec![0.0f32; t_q * d_model];

    // Warm both kernels once; the measurement then takes the best of REPS.
    flash_attention_2(
        &q,
        &k,
        &v,
        t_q,
        t_kv,
        N_HEADS,
        d_model,
        false,
        &mut tiled_out,
    );
    materialized_attention(
        &q,
        &k,
        &v,
        t_q,
        t_kv,
        N_HEADS,
        d_model,
        &mut materialized_out,
    );

    let max_diff = tiled_out
        .iter()
        .zip(materialized_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let (tiled_ms, materialized_ms) = time_both(
        || {
            flash_attention_2(
                &q,
                &k,
                &v,
                t_q,
                t_kv,
                N_HEADS,
                d_model,
                false,
                &mut tiled_out,
            );
        },
        || {
            materialized_attention(
                &q,
                &k,
                &v,
                t_q,
                t_kv,
                N_HEADS,
                d_model,
                &mut materialized_out,
            );
        },
    );

    let score_bytes_per_head = t_q * t_kv * 4;
    let score_bytes_total = score_bytes_per_head * N_HEADS;
    let scratch_per_task = (Q_TILE * KV_TILE + Q_TILE * D_HEAD + 2 * Q_TILE) * 4;
    let scratch_live = scratch_per_task * threads.min(N_HEADS * t_q.div_ceil(Q_TILE));
    let kv_stream_bytes = N_HEADS * t_q.div_ceil(Q_TILE) * t_kv * D_HEAD * 4 * 2;
    let flops = 4.0 * (t_q as f64) * (t_kv as f64) * (D_HEAD as f64) * (N_HEADS as f64);

    println!(
        "\nshape: t_q={t_q} t_kv={t_kv} d_head={D_HEAD} heads={N_HEADS} threads={threads} reps={REPS}"
    );
    println!(
        "  materialized: {materialized_ms:9.2} ms | score buffer b = t_q*t_kv*heads*4 = \
         {score_bytes_total} B ({:.1} MiB), per head {score_bytes_per_head} B ({:.1} MiB)",
        mib(score_bytes_total),
        mib(score_bytes_per_head),
    );
    println!(
        "  tiled:        {tiled_ms:9.2} ms | scratch <= {scratch_per_task} B per (head, q-tile) \
         task, {scratch_live} B live across {threads} tasks ({:.3} MiB)",
        mib(scratch_live),
    );
    println!(
        "  tiled streams ~{kv_stream_bytes} B ({:.1} MiB) of K/V re-reads at {:.1} GFLOP/s",
        mib(kv_stream_bytes),
        flops / (tiled_ms * 1e-3) / 1e9,
    );
    println!(
        "  speedup: {:.2}x (materialized {:.1} GFLOP/s) | max |tiled - materialized| = {max_diff:e}",
        materialized_ms / tiled_ms,
        flops / (materialized_ms * 1e-3) / 1e9,
    );

    (tiled_ms, materialized_ms, max_diff)
}

#[test]
#[ignore = "microbenchmark: run with --release --ignored --nocapture"]
fn bench_tiled_vs_materialized() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("== flash attention microbenchmark ==");
    let mut worst_diff = 0.0f32;
    for (t_q, t_kv) in SHAPES {
        let (_, _, diff) = run_shape(t_q, t_kv, threads);
        worst_diff = worst_diff.max(diff);
    }

    assert!(
        worst_diff <= 1e-4,
        "tiled kernel disagrees with materialized reference: {worst_diff:e}"
    );
}
