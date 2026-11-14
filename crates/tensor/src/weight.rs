//! Weight storage that keeps GGML tensors compressed at rest.
//!
//! Expanding Q4/Q5/Q8 weights into `Vec<f32>` at load time inflates the model
//! footprint by 4x-8x and turns each decode step into a full-bandwidth float
//! stream. The same argument applies to f16 checkpoints, where the load-time
//! conversion alone doubles resident memory. `LinearWeight` remembers the
//! storage format and dispatches to the matching kernel in place, so resident
//! RAM tracks the on-disk size and the memory system only moves the bytes that
//! are actually consumed by the dot products.

use crate::dtype::DType;
use half::f16;

/// A linear-layer weight matrix in dense f32, dense f16, or GGML block form.
///
/// Rows are contiguous and row-major: row `r` occupies
/// `data[r * row_stride .. (r + 1) * row_stride]`.
#[derive(Debug, Clone)]
pub enum LinearWeight {
    /// Dense little-endian f32 weights, `rows * cols` values.
    F32 {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    },
    /// Dense little-endian f16 weights, `rows * cols` halves.
    ///
    /// The halves stay packed so the matrix occupies half the RAM of an f32
    /// copy and the decode kernels read half the bytes from memory.
    F16 {
        rows: usize,
        cols: usize,
        data: Vec<u8>,
    },
    /// GGML block-quantized weights kept exactly as loaded from the model file.
    Quantized {
        rows: usize,
        cols: usize,
        dtype: DType,
        data: Vec<u8>,
    },
}

impl LinearWeight {
    /// Build a dense weight matrix from already-dequantized values.
    pub fn from_f32(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "F32 weight shape mismatch");
        Self::F32 { rows, cols, data }
    }

    /// Wrap raw little-endian f16 halves without expanding them to f32.
    ///
    /// `data` must hold `rows * cols` halves in row-major order.
    pub fn from_f16(rows: usize, cols: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), rows * cols * 2, "F16 weight shape mismatch");
        Self::F16 { rows, cols, data }
    }

    /// Wrap raw GGML block bytes without expanding them.
    ///
    /// Keeping the bytes compressed is what avoids the load-time 4x-8x RAM
    /// blow-up and the associated bandwidth cost during decoding.
    pub fn from_quantized(rows: usize, cols: usize, dtype: DType, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            rows * dtype.byte_size_for(cols),
            "quantized weight shape mismatch"
        );
        Self::Quantized {
            rows,
            cols,
            dtype,
            data,
        }
    }

    /// Number of output features (rows of the matrix).
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::F16 { rows, .. } | Self::Quantized { rows, .. } => *rows,
        }
    }

    /// Number of input features (columns of the matrix).
    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::F16 { cols, .. } | Self::Quantized { cols, .. } => *cols,
        }
    }

    /// Storage dtype: [`DType::F32`] for dense floats, [`DType::F16`] for packed
    /// halves, and the original GGML block type for quantized weights.
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32 { .. } => DType::F32,
            Self::F16 { .. } => DType::F16,
            Self::Quantized { dtype, .. } => *dtype,
        }
    }

    /// Whether the backing bytes are still in a GGML block format.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Heap bytes currently held by the weight.
    ///
    /// Comparing this against `rows * cols * 4` shows how much memory the
    /// compressed representation saves, which is the point of the type.
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::F32 { data, .. } => data.len() * std::mem::size_of::<f32>(),
            Self::F16 { data, .. } | Self::Quantized { data, .. } => data.len(),
        }
    }

    /// Multi-token projection: `output[t * rows + r] = W[r] · x[t] + bias[r]`.
    ///
    /// The F16 and quantized branches never materialize a float copy of the
    /// matrix, so prompt evaluation stays bandwidth bound on the stored bytes.
    pub fn linear(&self, x: &[f32], bias: Option<&[f32]>, num_tokens: usize, output: &mut [f32]) {
        match self {
            Self::F32 { rows, cols, data } => {
                crate::ops::matmul::linear(x, data, bias, *cols, *rows, num_tokens, output)
            }
            Self::F16 { rows, cols, data } => {
                crate::ops::quant::f16_linear(data, *rows, *cols, x, bias, num_tokens, output)
            }
            Self::Quantized {
                rows,
                cols,
                dtype,
                data,
            } => crate::ops::quant::quantized_linear(
                data, *dtype, *rows, *cols, x, bias, num_tokens, output,
            ),
        }
    }

    /// Single-token projection `y = W · x + bias`.
    ///
    /// Decoding issues one token per step, so dense f32 weights use the float
    /// GEMV microkernel, f16 weights convert halves inside the vector kernel,
    /// and quantized weights stream their blocks through the parallel GEMV.
    pub fn gemv(&self, x: &[f32], bias: Option<&[f32]>, output: &mut [f32]) {
        match self {
            Self::F32 { rows, cols, data } => {
                crate::simd::gemv_f32(x, data, bias, output, *cols, *rows)
            }
            Self::F16 { rows, cols, data } => {
                crate::ops::quant::f16_gemv_parallel(data, *rows, *cols, x, bias, output)
            }
            Self::Quantized {
                rows,
                cols,
                dtype,
                data,
            } => crate::ops::quant::quantized_gemv_parallel(
                data, *dtype, *rows, *cols, x, bias, output,
            ),
        }
    }

    /// Dequantize a single row into `out`.
    ///
    /// Token embedding lookup only needs one row; doing it per row avoids
    /// expanding the whole vocabulary matrix just to read one vector.
    pub fn row_into(&self, row: usize, out: &mut [f32]) {
        let cols = self.cols();
        assert!(row < self.rows(), "row {row} out of range");
        assert_eq!(out.len(), cols, "row output length mismatch");
        match self {
            Self::F32 { data, .. } => {
                out.copy_from_slice(&data[row * cols..(row + 1) * cols]);
            }
            Self::F16 { data, .. } => {
                let row_bytes = &data[row * cols * 2..(row + 1) * cols * 2];
                let (pairs, _) = row_bytes.as_chunks::<2>();
                for (dst, src) in out.iter_mut().zip(pairs) {
                    *dst = f16::from_bits(u16::from_le_bytes(*src)).to_f32();
                }
            }
            Self::Quantized { dtype, data, .. } => {
                let stride = dtype.byte_size_for(cols);
                let row_bytes = &data[row * stride..(row + 1) * stride];
                let dequantized = dtype
                    .dequantize_to_f32(row_bytes, cols)
                    .expect("validated weight row must dequantize");
                out.copy_from_slice(&dequantized[..cols]);
            }
        }
    }

    /// Stack the rows of several weights into one matrix.
    ///
    /// Fusing `q`, `k` and `v` into a single projection lets one GEMM read the
    /// activations once instead of three times. Block-quantized rows are copied
    /// byte-for-byte, so their per-row scales remain valid in the fused layout.
    pub fn concat_rows(parts: &[&LinearWeight]) -> Self {
        assert!(
            !parts.is_empty(),
            "concat_rows requires at least one weight"
        );
        let cols = parts[0].cols();
        let rows: usize = parts.iter().map(|w| w.rows()).sum();
        for part in parts {
            assert_eq!(part.cols(), cols, "concat_rows column mismatch");
        }

        match parts[0] {
            Self::F32 { .. } => {
                let mut data = Vec::with_capacity(rows * cols);
                for part in parts {
                    match part {
                        Self::F32 { data: d, .. } => data.extend_from_slice(d),
                        Self::F16 { .. } | Self::Quantized { .. } => {
                            panic!("cannot concat F32 with other weight formats")
                        }
                    }
                }
                Self::from_f32(rows, cols, data)
            }
            Self::F16 { .. } => {
                let mut data = Vec::with_capacity(rows * cols * 2);
                for part in parts {
                    match part {
                        Self::F16 { data: d, .. } => data.extend_from_slice(d),
                        Self::F32 { .. } | Self::Quantized { .. } => {
                            panic!("cannot concat F16 with other weight formats")
                        }
                    }
                }
                Self::from_f16(rows, cols, data)
            }
            Self::Quantized { dtype, .. } => {
                let dtype = *dtype;
                let mut data = Vec::with_capacity(rows * dtype.byte_size_for(cols));
                for part in parts {
                    match part {
                        Self::Quantized {
                            dtype: part_dtype,
                            data: d,
                            ..
                        } => {
                            assert_eq!(*part_dtype, dtype, "concat_rows dtype mismatch");
                            data.extend_from_slice(d);
                        }
                        Self::F32 { .. } | Self::F16 { .. } => {
                            panic!("cannot concat quantized with other weight formats")
                        }
                    }
                }
                Self::from_quantized(rows, cols, dtype, data)
            }
        }
    }

    /// Full float32 copy, used by tests and by callers that genuinely need the
    /// dense matrix. Allocates the expanded size, so it must not sit on the
    /// decode hot path.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::F32 { data, .. } => data.clone(),
            Self::F16 { rows, cols, data } => DType::F16
                .dequantize_to_f32(data, *rows * *cols)
                .expect("validated F16 weight must dequantize"),
            Self::Quantized {
                rows,
                cols,
                dtype,
                data,
            } => {
                let mut out = Vec::with_capacity(rows * cols);
                let stride = dtype.byte_size_for(*cols);
                for r in 0..*rows {
                    let row_bytes = &data[r * stride..(r + 1) * stride];
                    let row = dtype
                        .dequantize_to_f32(row_bytes, *cols)
                        .expect("validated weight row must dequantize");
                    out.extend_from_slice(&row[..*cols]);
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manual_dot(row: &[f32], x: &[f32]) -> f32 {
        row.iter().zip(x).map(|(w, v)| w * v).sum()
    }

    #[test]
    fn f32_linear_matches_manual_dot() {
        let (rows, cols) = (3usize, 4usize);
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25).collect();
        let weight = LinearWeight::from_f32(rows, cols, data.clone());
        let x: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0];
        let bias = vec![0.1, 0.2, 0.3];
        let mut out = vec![0.0f32; rows];

        weight.linear(&x, Some(&bias), 1, &mut out);

        for r in 0..rows {
            let expected = manual_dot(&data[r * cols..(r + 1) * cols], &x) + bias[r];
            assert!((out[r] - expected).abs() < 1e-5);
        }
    }

    fn q8_weight(rows: usize, cols: usize) -> LinearWeight {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7) % 23) as f32 * 0.05 - 0.4)
            .collect();
        let bytes = DType::Q8_0.quantize_from_f32(&values).unwrap();
        LinearWeight::from_quantized(rows, cols, DType::Q8_0, bytes)
    }

    #[test]
    fn quantized_linear_matches_dequantized_reference() {
        let (rows, cols) = (5usize, 32usize);
        let weight = q8_weight(rows, cols);
        let reference = weight.to_f32();
        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.03 - 0.2).collect();
        let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0f32; rows];

        weight.linear(&x, Some(&bias), 1, &mut out);

        // Activations are quantized to Q8 before the integer dot kernel, so
        // each activation carries at most half a quantization step of error.
        // The dot error is bounded by that half step times the row's weight
        // magnitude; this is the accuracy contract of the Q8 activation path.
        let amax = x.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let half_step = amax / 254.0;
        for r in 0..rows {
            let row = &reference[r * cols..(r + 1) * cols];
            let expected = manual_dot(row, &x) + bias[r];
            let weight_abs: f32 = row.iter().map(|v| v.abs()).sum();
            let tol = half_step * weight_abs + 1e-4;
            assert!(
                (out[r] - expected).abs() < tol,
                "row {r}: {} vs {} (bound {})",
                out[r],
                expected,
                tol
            );
        }
    }

    #[test]
    fn row_into_matches_dense_rows() {
        let (rows, cols) = (4usize, 32usize);
        let weight = q8_weight(rows, cols);
        let reference = weight.to_f32();
        let mut row = vec![0.0f32; cols];

        for r in 0..rows {
            weight.row_into(r, &mut row);
            let want = &reference[r * cols..(r + 1) * cols];
            for (got, expected) in row.iter().zip(want) {
                assert!((got - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn concat_rows_equals_separate_projections() {
        let cols = 4usize;
        let parts_data: Vec<Vec<f32>> = (0..3)
            .map(|p| (0..2 * cols).map(|i| (p * 100 + i) as f32 * 0.01).collect())
            .collect();
        let parts: Vec<LinearWeight> = parts_data
            .iter()
            .map(|d| LinearWeight::from_f32(2, cols, d.clone()))
            .collect();
        let refs: Vec<&LinearWeight> = parts.iter().collect();
        let fused = LinearWeight::concat_rows(&refs);

        assert_eq!(fused.rows(), 6);
        let x = vec![0.5f32, -1.0, 2.0, 0.25];
        let mut fused_out = vec![0.0f32; 6];
        fused.linear(&x, None, 1, &mut fused_out);

        for (p, part) in parts.iter().enumerate() {
            let mut part_out = vec![0.0f32; 2];
            part.linear(&x, None, 1, &mut part_out);
            for r in 0..2 {
                assert!((fused_out[p * 2 + r] - part_out[r]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn quantized_residency_is_smaller() {
        let (rows, cols) = (64usize, 64usize);
        let weight = q8_weight(rows, cols);
        let dense_bytes = rows * cols * std::mem::size_of::<f32>();
        assert!(weight.resident_bytes() < dense_bytes);
        assert_eq!(weight.to_f32().len(), rows * cols);
    }

    fn f16_weight(rows: usize, cols: usize) -> (LinearWeight, Vec<f32>) {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 13) % 31) as f32 * 0.05 - 0.7)
            .collect();
        let bytes = DType::F16.quantize_from_f32(&values).unwrap();
        let reference = DType::F16.dequantize_to_f32(&bytes, rows * cols).unwrap();
        (LinearWeight::from_f16(rows, cols, bytes), reference)
    }

    #[test]
    fn f16_linear_matches_f32_reference() {
        // Odd row/column counts, non-multiples of the 8-wide f16 SIMD step and
        // of the 4-wide scalar unroll, plus one shape wider than a full block.
        for &(rows, cols) in &[(7usize, 33usize), (5, 13), (17, 128), (1, 1), (3, 31)] {
            let (weight, reference) = f16_weight(rows, cols);
            assert_eq!(weight.rows(), rows);
            assert_eq!(weight.cols(), cols);
            assert_eq!(weight.dtype(), DType::F16);
            assert!(!weight.is_quantized());
            assert_eq!(weight.resident_bytes(), rows * cols * 2);
            assert_eq!(weight.to_f32(), reference);

            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.11).sin() + 0.4).collect();
            let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.02 - 0.1).collect();
            let mut out = vec![0.0f32; rows];
            weight.linear(&x, Some(&bias), 1, &mut out);

            let mut max_err = 0.0f32;
            for r in 0..rows {
                let expected = manual_dot(&reference[r * cols..(r + 1) * cols], &x) + bias[r];
                max_err = max_err.max((out[r] - expected).abs());
            }
            assert!(max_err <= 1e-4, "{rows}x{cols}: max abs error {max_err}");

            let mut gemv_out = vec![0.0f32; rows];
            weight.gemv(&x, Some(&bias), &mut gemv_out);
            for r in 0..rows {
                assert!((gemv_out[r] - out[r]).abs() <= 1e-5, "gemv row {r}");
            }
        }
    }

    #[test]
    fn f16_row_into_matches_packed_halves() {
        let (rows, cols) = (9usize, 13usize);
        let (weight, reference) = f16_weight(rows, cols);
        let mut row = vec![0.0f32; cols];
        for r in 0..rows {
            weight.row_into(r, &mut row);
            assert_eq!(row, reference[r * cols..(r + 1) * cols]);
        }
    }

    #[test]
    fn f16_batched_matches_reference() {
        let (rows, cols, num_tokens) = (6usize, 37usize, 8usize);
        let (weight, reference) = f16_weight(rows, cols);
        let x: Vec<f32> = (0..num_tokens * cols)
            .map(|i| ((i * 5) % 17) as f32 * 0.03 - 0.2)
            .collect();
        let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.01).collect();
        let mut out = vec![0.0f32; num_tokens * rows];

        weight.linear(&x, Some(&bias), num_tokens, &mut out);

        let mut max_err = 0.0f32;
        for t in 0..num_tokens {
            for r in 0..rows {
                let expected = manual_dot(
                    &reference[r * cols..(r + 1) * cols],
                    &x[t * cols..(t + 1) * cols],
                ) + bias[r];
                max_err = max_err.max((out[t * rows + r] - expected).abs());
            }
        }
        assert!(max_err <= 1e-4, "batched max abs error {max_err}");
    }

    #[test]
    fn f16_concat_rows_preserves_projection() {
        let cols = 5usize;
        let parts: Vec<LinearWeight> = (0..3)
            .map(|p| {
                // Distinct per-part values so a bad row offset shows up.
                let values: Vec<f32> = (0..2 * cols)
                    .map(|i| ((i * 7 + p * 11) % 23) as f32 * 0.05 - 0.5)
                    .collect();
                let bytes = DType::F16.quantize_from_f32(&values).unwrap();
                LinearWeight::from_f16(2, cols, bytes)
            })
            .collect();
        let refs: Vec<&LinearWeight> = parts.iter().collect();
        let fused = LinearWeight::concat_rows(&refs);
        assert_eq!(fused.rows(), 6);
        assert_eq!(fused.dtype(), DType::F16);

        let x = vec![0.5f32, -1.0, 2.0, 0.25, 0.75];
        let mut fused_out = vec![0.0f32; 6];
        fused.linear(&x, None, 1, &mut fused_out);

        for (p, part) in parts.iter().enumerate() {
            let mut part_out = vec![0.0f32; 2];
            part.linear(&x, None, 1, &mut part_out);
            for r in 0..2 {
                assert!((fused_out[p * 2 + r] - part_out[r]).abs() <= 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "F16 weight shape mismatch")]
    fn f16_shape_validation_rejects_short_data() {
        LinearWeight::from_f16(2, 3, vec![0u8; 11]);
    }
}
