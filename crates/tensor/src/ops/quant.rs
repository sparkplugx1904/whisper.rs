#![allow(clippy::chunks_exact_to_as_chunks)]
use crate::dtype::DType;
use half::f16;

/// Direct dot product between Q8_0 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 34 bytes each (scale f16 + 32 int8).
pub fn vec_dot_q8_0_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 34);
    crate::simd::dot_q8_0_f32(raw_blocks, x)
}

/// Direct dot product between Q4_0 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 18 bytes each (scale f16 + 16 bytes nibbles).
pub fn vec_dot_q4_0_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 18);
    crate::simd::dot_q4_0_f32(raw_blocks, x)
}

/// Direct dot product between Q4_1 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 20 bytes each (scale f16 + min f16 + 16 bytes nibbles).
pub fn vec_dot_q4_1_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 20);
    crate::simd::dot_q4_1_f32(raw_blocks, x)
}

pub fn vec_dot_q4_1_f32_scalar(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(20).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let m_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
        let d = f16::from_bits(d_bits).to_f32();
        let m = f16::from_bits(m_bits).to_f32();
        let qs = &chunk[4..20];
        let x_block = &x[i * 32..(i + 1) * 32];

        let mut block_sum = 0.0f32;
        let mut x_sum = 0.0f32;
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as f32;
            let x1 = ((qs[j] >> 4) & 0x0F) as f32;
            block_sum += x0 * x_block[j] + x1 * x_block[j + 16];
            x_sum += x_block[j] + x_block[j + 16];
        }
        sum += d * block_sum + m * x_sum;
    }
    sum
}

/// Direct dot product between Q5_0 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 22 bytes each (scale f16 + 4 bytes qh + 16 bytes qs).
pub fn vec_dot_q5_0_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 22);

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(22).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let d = f16::from_bits(d_bits).to_f32();
        let qh = u32::from_le_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);
        let qs = &chunk[6..22];
        let x_block = &x[i * 32..(i + 1) * 32];

        let mut block_sum = 0.0f32;
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = (((qs[j] & 0x0F) | xh_0) as i32) - 16;
            let x1 = ((((qs[j] >> 4) & 0x0F) | xh_1) as i32) - 16;
            block_sum += (x0 as f32) * x_block[j] + (x1 as f32) * x_block[j + 16];
        }
        sum += d * block_sum;
    }
    sum
}

/// Direct dot product between Q5_1 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 24 bytes each (scale f16 + min f16 + 4 bytes qh + 16 bytes qs).
pub fn vec_dot_q5_1_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 24);

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(24).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let m_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
        let d = f16::from_bits(d_bits).to_f32();
        let m = f16::from_bits(m_bits).to_f32();
        let qh = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let qs = &chunk[8..24];
        let x_block = &x[i * 32..(i + 1) * 32];

        let mut block_sum = 0.0f32;
        let mut x_sum = 0.0f32;
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0F) | xh_0) as f32;
            let x1 = (((qs[j] >> 4) & 0x0F) | xh_1) as f32;
            block_sum += x0 * x_block[j] + x1 * x_block[j + 16];
            x_sum += x_block[j] + x_block[j + 16];
        }
        sum += d * block_sum + m * x_sum;
    }
    sum
}

/// Direct matrix-vector multiplication with quantized weights: y = W * x + bias
/// Rows of W are quantized directly without dequantizing the full matrix.
#[allow(clippy::too_many_arguments)]
pub fn quantized_linear_vector(
    weight_bytes: &[u8],
    dtype: DType,
    rows: usize,
    cols: usize,
    x: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    n_threads: usize,
) {
    assert_eq!(x.len(), cols);
    assert_eq!(output.len(), rows);
    if let Some(b) = bias {
        assert_eq!(b.len(), rows);
    }

    let bytes_per_row = dtype.byte_size_for(cols);
    assert_eq!(weight_bytes.len(), rows * bytes_per_row);

    use rayon::prelude::*;

    let n_threads = n_threads.clamp(1, 16);
    let chunk_size = rows.div_ceil(n_threads);

    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let r_start = chunk_idx * chunk_size;

            for (r_rel, out_val) in out_chunk.iter_mut().enumerate() {
                let r_abs = r_start + r_rel;
                let row_bytes = &weight_bytes[r_abs * bytes_per_row..(r_abs + 1) * bytes_per_row];

                let dot = match dtype {
                    DType::Q8_0 => vec_dot_q8_0_f32(row_bytes, x),
                    DType::Q4_0 => vec_dot_q4_0_f32(row_bytes, x),
                    DType::Q4_1 => vec_dot_q4_1_f32(row_bytes, x),
                    DType::Q5_0 => vec_dot_q5_0_f32(row_bytes, x),
                    DType::Q5_1 => vec_dot_q5_1_f32(row_bytes, x),
                    DType::Q4_K => vec_dot_q4_k_f32(row_bytes, x),
                    DType::Q5_K => vec_dot_q5_k_f32(row_bytes, x),
                    DType::Q6_K => vec_dot_q6_k_f32(row_bytes, x),
                    _ => panic!("Unsupported quantized linear type {:?}", dtype),
                };

                let val = if let Some(b) = bias {
                    dot + b[r_abs]
                } else {
                    dot
                };

                *out_val = val;
            }
        });
}

/// Direct dot product between Q4_K super-blocks (256 elements, 144 bytes each) and f32 slice.
pub fn vec_dot_q4_k_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 256;
    assert_eq!(raw_blocks.len(), n_blocks * 144);
    crate::simd::dot_q4_k_f32(raw_blocks, x)
}

pub fn vec_dot_q4_k_f32_scalar(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(144).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let dmin_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
        let d = f16::from_bits(d_bits).to_f32();
        let dmin = f16::from_bits(dmin_bits).to_f32();

        let scales = &chunk[4..16];
        let qs = &chunk[16..144];
        let x_block = &x[i * 256..(i + 1) * 256];

        for sb in 0..8 {
            let sc = if sb < 4 {
                (scales[sb] & 63) as f32
            } else {
                ((scales[sb + 4] & 0x0F) | ((scales[sb - 4] >> 6) << 4)) as f32
            };
            let m = if sb < 4 {
                (scales[sb + 4] & 63) as f32
            } else {
                (((scales[sb + 4] >> 4) & 0x0F) | ((scales[sb] >> 6) << 4)) as f32
            };

            let d1 = d * sc;
            let m1 = dmin * m;

            let qs_sub = &qs[sb * 16..(sb + 1) * 16];
            let x_sub = &x_block[sb * 32..(sb + 1) * 32];

            let mut sb_sum = 0.0f32;
            for j in 0..16 {
                let q0 = (qs_sub[j] & 0x0F) as f32;
                let q1 = ((qs_sub[j] >> 4) & 0x0F) as f32;
                sb_sum += (d1 * q0 - m1) * x_sub[j] + (d1 * q1 - m1) * x_sub[j + 16];
            }
            sum += sb_sum;
        }
    }
    sum
}

/// Direct dot product between Q5_K super-blocks (256 elements, 176 bytes each) and f32 slice.
pub fn vec_dot_q5_k_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 256;
    assert_eq!(raw_blocks.len(), n_blocks * 176);

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(176).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let dmin_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
        let d = f16::from_bits(d_bits).to_f32();
        let dmin = f16::from_bits(dmin_bits).to_f32();

        let scales = &chunk[4..16];
        let qh = &chunk[16..48];
        let qs = &chunk[48..176];
        let x_block = &x[i * 256..(i + 1) * 256];

        for sb in 0..8 {
            let sc = if sb < 4 {
                (scales[sb] & 63) as f32
            } else {
                ((scales[sb + 4] & 0x0F) | ((scales[sb - 4] >> 6) << 4)) as f32
            };
            let m = if sb < 4 {
                (scales[sb + 4] & 63) as f32
            } else {
                (((scales[sb + 4] >> 4) & 0x0F) | ((scales[sb] >> 6) << 4)) as f32
            };

            let d1 = d * sc;
            let m1 = dmin * m;

            let qs_sub = &qs[sb * 16..(sb + 1) * 16];
            let x_sub = &x_block[sb * 32..(sb + 1) * 32];
            let qh_sub = &qh[sb * 4..(sb + 1) * 4];

            let mut sb_sum = 0.0f32;
            for j in 0..16 {
                let h0 = ((qh_sub[j / 4] >> ((j % 4) * 2)) & 1) as f32;
                let h1 = ((qh_sub[j / 4] >> ((j % 4) * 2 + 1)) & 1) as f32;

                let q0 = ((qs_sub[j] & 0x0F) as f32) + h0 * 16.0;
                let q1 = (((qs_sub[j] >> 4) & 0x0F) as f32) + h1 * 16.0;

                sb_sum += (d1 * q0 - m1) * x_sub[j] + (d1 * q1 - m1) * x_sub[j + 16];
            }
            sum += sb_sum;
        }
    }
    sum
}

/// Direct dot product between Q6_K super-blocks (256 elements, 210 bytes each) and f32 slice.
pub fn vec_dot_q6_k_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 256;
    assert_eq!(raw_blocks.len(), n_blocks * 210);

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(210).enumerate() {
        let ql = &chunk[0..128];
        let qh = &chunk[128..192];
        let scales = &chunk[192..208];
        let d_bits = u16::from_le_bytes([chunk[208], chunk[209]]);
        let d = f16::from_bits(d_bits).to_f32();
        let x_block = &x[i * 256..(i + 1) * 256];

        for sb in 0..16 {
            let sc = (scales[sb] as i8 as f32) * d;
            let ql_sub = &ql[sb * 8..(sb + 1) * 8];
            let qh_sub = &qh[sb * 4..(sb + 1) * 4];
            let x_sub = &x_block[sb * 16..(sb + 1) * 16];

            let mut sb_sum = 0.0f32;
            for j in 0..8 {
                let h0 = ((qh_sub[j / 2] >> ((j % 2) * 4)) & 3) as i8;
                let h1 = ((qh_sub[j / 2] >> ((j % 2) * 4 + 2)) & 3) as i8;

                let q0 = ((ql_sub[j] & 0x0F) as i8 | (h0 << 4)) - 32;
                let q1 = (((ql_sub[j] >> 4) & 0x0F) as i8 | (h1 << 4)) - 32;

                sb_sum += sc * (q0 as f32) * x_sub[j] + sc * (q1 as f32) * x_sub[j + 8];
            }
            sum += sb_sum;
        }
    }
    sum
}
