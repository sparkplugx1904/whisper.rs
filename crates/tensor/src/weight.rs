//! Weight storage that keeps quantized GGML tensors compressed at rest.
//!
//! Expanding Q4/Q5/Q8 weights into `Vec<f32>` at load time inflates the model
//! footprint by 4x-8x and turns each decode step into a full-bandwidth float
//! stream. `LinearWeight` remembers whether the tensor is still quantized and
//! dispatches to the quantized kernels in place, so resident RAM tracks the
//! on-disk size and the memory system only moves the bytes that are actually
//! consumed by the dot products.

use crate::dtype::DType;

/// A linear-layer weight matrix in either dense float or GGML block form.
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
            Self::F32 { rows, .. } | Self::Quantized { rows, .. } => *rows,
        }
    }

    /// Number of input features (columns of the matrix).
    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::Quantized { cols, .. } => *cols,
        }
    }

    /// Storage dtype; the dense variant always reports [`DType::F32`].
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32 { .. } => DType::F32,
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
            Self::Quantized { data, .. } => data.len(),
        }
    }

    /// Multi-token projection: `output[t * rows + r] = W[r] · x[t] + bias[r]`.
    ///
    /// The quantized branch never materializes a float copy of the matrix, so
    /// prompt evaluation stays bandwidth bound on the compressed bytes.
    pub fn linear(&self, x: &[f32], bias: Option<&[f32]>, num_tokens: usize, output: &mut [f32]) {
        match self {
            Self::F32 { rows, cols, data } => {
                crate::ops::matmul::linear(x, data, bias, *cols, *rows, num_tokens, output)
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
    /// Decoding issues one token per step, so this path keeps the existing
    /// float GEMV microkernel for dense weights and streams the compressed
    /// blocks through the parallel quantized GEMV for quantized ones.
    pub fn gemv(&self, x: &[f32], bias: Option<&[f32]>, output: &mut [f32]) {
        match self {
            Self::F32 { rows, cols, data } => {
                crate::simd::gemv_f32(x, data, bias, output, *cols, *rows)
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
                        Self::Quantized { .. } => {
                            panic!("cannot concat F32 and quantized weights")
                        }
                    }
                }
                Self::from_f32(rows, cols, data)
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
                        Self::F32 { .. } => {
                            panic!("cannot concat F32 and quantized weights")
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

        for r in 0..rows {
            let expected = manual_dot(&reference[r * cols..(r + 1) * cols], &x) + bias[r];
            assert!(
                (out[r] - expected).abs() < 1e-3,
                "row {r}: {} vs {}",
                out[r],
                expected
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
}
