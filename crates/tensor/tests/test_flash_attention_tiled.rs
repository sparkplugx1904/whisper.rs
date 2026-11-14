//! Numerical tests for the tiled online-softmax attention kernel.
//!
//! The reference below is the previous implementation: materialize the full
//! `t_q x t_kv` score matrix, softmax each row, then multiply by `V`. The
//! tiled kernel folds the same softmax statistics over key tiles, so both
//! paths compute the same function with different summation orders. All
//! assertions use a maximum absolute error of `1e-5`, which is roughly two
//! orders of magnitude above the f32 rounding spread observed on these inputs.

use whisper_tensor::ops::attention::{KV_TILE, Q_TILE};
use whisper_tensor::ops::{flash_attention_2, multi_head_attention, softmax_inplace};

/// Deterministic xorshift32 generator, so failures are reproducible without a
/// dependency on `rand`.
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        // Map the full u32 range onto [-1, 1).
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn fill(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }
}

/// Previous materialized implementation, kept verbatim in behaviour so the
/// tiled kernel is compared against what it replaces rather than against a
/// re-derivation of the same tile math.
#[allow(clippy::too_many_arguments)]
fn reference_materialized(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_q: usize,
    t_kv: usize,
    n_heads: usize,
    d_model: usize,
    causal: bool,
    output: &mut [f32],
) {
    let d_head = d_model / n_heads;
    let scale = 1.0f32 / (d_head as f32).sqrt();
    for h in 0..n_heads {
        let head_offset = h * d_head;
        let mut scores = vec![0.0f32; t_q * t_kv];

        for i in 0..t_q {
            for j in 0..t_kv {
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    dot += q[i * d_model + head_offset + d] * k[j * d_model + head_offset + d];
                }
                scores[i * t_kv + j] = dot * scale;
            }
        }

        for i in 0..t_q {
            let row = &mut scores[i * t_kv..(i + 1) * t_kv];
            if causal {
                let valid_len = (i + 1).min(t_kv);
                softmax_inplace(&mut row[..valid_len]);
                row[valid_len..].fill(0.0);
            } else {
                softmax_inplace(row);
            }
        }

        for i in 0..t_q {
            for d in 0..d_head {
                let mut sum = 0.0f32;
                for j in 0..t_kv {
                    sum += scores[i * t_kv + j] * v[j * d_model + head_offset + d];
                }
                output[i * d_model + head_offset + d] = sum;
            }
        }
    }
}

/// Runs both kernels on deterministic inputs and returns the maximum absolute
/// deviation between them.
fn max_abs_error(t_q: usize, t_kv: usize, n_heads: usize, d_head: usize, causal: bool) -> f32 {
    let d_model = n_heads * d_head;
    let mut rng = Rng::new(0x1234_5678 ^ ((t_q as u32) << 16) ^ (t_kv as u32));
    let q = rng.fill(t_q * d_model);
    let k = rng.fill(t_kv * d_model);
    let v = rng.fill(t_kv * d_model);

    let mut reference = vec![0.0f32; t_q * d_model];
    reference_materialized(
        &q,
        &k,
        &v,
        t_q,
        t_kv,
        n_heads,
        d_model,
        causal,
        &mut reference,
    );

    let mut tiled = vec![0.0f32; t_q * d_model];
    flash_attention_2(&q, &k, &v, t_q, t_kv, n_heads, d_model, causal, &mut tiled);

    reference
        .iter()
        .zip(tiled.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn assert_matches(t_q: usize, t_kv: usize, n_heads: usize, d_head: usize, causal: bool) {
    let err = max_abs_error(t_q, t_kv, n_heads, d_head, causal);
    assert!(
        err <= 1e-5,
        "t_q={t_q} t_kv={t_kv} n_heads={n_heads} d_head={d_head} causal={causal}: \
         max abs error {err} exceeds 1e-5"
    );
}

#[test]
fn tiled_matches_reference_d_head_64_non_causal() {
    // 137 keys is one full key tile plus a 9-key tail; 100 queries is a
    // partial query tile.
    assert_matches(100, 137, 3, 64, false);
}

#[test]
fn tiled_matches_reference_d_head_64_causal() {
    assert_matches(100, 137, 3, 64, true);
}

#[test]
fn tiled_matches_reference_d_head_80_non_causal() {
    assert_matches(37, 91, 2, 80, false);
}

#[test]
fn tiled_matches_reference_d_head_80_causal() {
    assert_matches(37, 91, 2, 80, true);
}

#[test]
fn tiled_matches_reference_on_tile_boundaries() {
    // Shapes just past exact multiples of the tile sizes exercise a full tile
    // followed by a ragged tail on both the query and key axes, and the
    // multi-tile shapes force several online-softmax updates per row.
    for causal in [false, true] {
        assert_matches(Q_TILE + 1, KV_TILE + 1, 2, 64, causal);
        assert_matches(2 * Q_TILE + 3, 2 * KV_TILE + 5, 4, 64, causal);
    }
}

#[test]
fn tiled_matches_reference_with_fewer_keys_than_queries_causal() {
    // Causal masking clamps the visible window at `t_kv`, so every query past
    // the key range attends all keys. The second shape spans several query
    // tiles, so the key-tile early exit is exercised with a short key range.
    assert_matches(90, 33, 3, 64, true);
    assert_matches(Q_TILE + 40, 37, 3, 64, true);
}

#[test]
fn single_query_decode_matches_reference_d_head_64() {
    // t_q == 1 with d_head == 64 takes the dedicated decode kernel.
    assert_matches(1, 257, 4, 64, false);
}

#[test]
fn single_query_decode_matches_reference_d_head_80() {
    // d_head != 64 keeps the tiled path even for one query.
    assert_matches(1, 193, 2, 80, false);
}

#[test]
fn single_query_causal_only_sees_first_key() {
    // A causal single query may attend exactly one key, so the output must be
    // that key's value vector.
    let d_model = 64;
    let t_kv = 17;
    let mut rng = Rng::new(0xfeed_face);
    let q = rng.fill(d_model);
    let k = rng.fill(t_kv * d_model);
    let v = rng.fill(t_kv * d_model);

    let mut out = vec![0.0f32; d_model];
    flash_attention_2(&q, &k, &v, 1, t_kv, 1, d_model, true, &mut out);
    assert!(
        out.iter()
            .zip(v[..d_model].iter())
            .all(|(a, b)| (a - b).abs() <= 1e-6),
        "causal single-query attention must return v[0]"
    );
}

#[test]
fn multi_head_attention_matches_reference() {
    let t_q = 48;
    let t_kv = 70;
    let n_heads = 5;
    let d_head = 64;
    let d_model = n_heads * d_head;
    let mut rng = Rng::new(0x0bad_c0de);
    let q = rng.fill(t_q * d_model);
    let k = rng.fill(t_kv * d_model);
    let v = rng.fill(t_kv * d_model);

    let mut reference = vec![0.0f32; t_q * d_model];
    reference_materialized(
        &q,
        &k,
        &v,
        t_q,
        t_kv,
        n_heads,
        d_model,
        false,
        &mut reference,
    );

    let mut out = vec![0.0f32; t_q * d_model];
    multi_head_attention(&q, &k, &v, t_q, t_kv, n_heads, d_model, false, &mut out);

    let err = reference
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(err <= 1e-5, "multi_head_attention max abs error {err}");
}

#[test]
fn tiled_kernel_ignores_stale_output() {
    // Every output element must be written by the kernel; pre-filling with a
    // sentinel catches rows that a partially masked tile could skip.
    let t_q = 40;
    let t_kv = 100;
    let n_heads = 2;
    let d_head = 64;
    let d_model = n_heads * d_head;
    let mut rng = Rng::new(0xdead_beef);
    let q = rng.fill(t_q * d_model);
    let k = rng.fill(t_kv * d_model);
    let v = rng.fill(t_kv * d_model);

    let mut out = vec![f32::NAN; t_q * d_model];
    flash_attention_2(&q, &k, &v, t_q, t_kv, n_heads, d_model, true, &mut out);
    assert!(
        out.iter().all(|x| x.is_finite()),
        "kernel left unwritten rows"
    );
}
