#![allow(clippy::chunks_exact_to_as_chunks)]
use crate::dtype::DType;
use half::f16;

/// Direct dot product between Q8_0 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 34 bytes each (scale f16 + 32 int8).
pub fn vec_dot_q8_0_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 34);

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(34).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let d = f16::from_bits(d_bits).to_f32();
        let qs = &chunk[2..34];
        let x_block = &x[i * 32..(i + 1) * 32];

        let mut block_sum = 0.0f32;
        for j in 0..32 {
            block_sum += (qs[j] as i8 as f32) * x_block[j];
        }
        sum += d * block_sum;
    }
    sum
}

/// Direct dot product between Q4_0 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 18 bytes each (scale f16 + 16 bytes nibbles).
pub fn vec_dot_q4_0_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 18);

    let mut sum = 0.0f32;
    for (i, chunk) in raw_blocks.chunks_exact(18).enumerate() {
        let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let d = f16::from_bits(d_bits).to_f32();
        let qs = &chunk[2..18];
        let x_block = &x[i * 32..(i + 1) * 32];

        let mut block_sum = 0.0f32;
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i8 - 8;
            let x1 = ((qs[j] >> 4) & 0x0F) as i8 - 8;
            block_sum += (x0 as f32) * x_block[j] + (x1 as f32) * x_block[j + 16];
        }
        sum += d * block_sum;
    }
    sum
}

/// Direct dot product between Q4_1 quantized blocks and float32 slice.
/// `raw_blocks` contains (cols / 32) blocks of 20 bytes each (scale f16 + min f16 + 16 bytes nibbles).
pub fn vec_dot_q4_1_f32(raw_blocks: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / 32;
    assert_eq!(raw_blocks.len(), n_blocks * 20);

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

    let n_threads = n_threads.clamp(1, 16);
    let chunk_size = rows.div_ceil(n_threads);

    std::thread::scope(|s| {
        for (r_start, _) in (0..rows).step_by(chunk_size).map(|r| (r, ())) {
            let r_end = (r_start + chunk_size).min(rows);
            let w_sub = &weight_bytes[r_start * bytes_per_row..r_end * bytes_per_row];
            let b_sub = bias.map(|b| &b[r_start..r_end]);
            let out_ptr = output.as_mut_ptr() as usize;

            s.spawn(move || {
                let out_slice_ptr = out_ptr as *mut f32;
                let r_count = r_end - r_start;

                for r_rel in 0..r_count {
                    let r_abs = r_start + r_rel;
                    let row_bytes = &w_sub[r_rel * bytes_per_row..(r_rel + 1) * bytes_per_row];

                    let dot = match dtype {
                        DType::Q8_0 => vec_dot_q8_0_f32(row_bytes, x),
                        DType::Q4_0 => vec_dot_q4_0_f32(row_bytes, x),
                        DType::Q4_1 => vec_dot_q4_1_f32(row_bytes, x),
                        DType::Q5_0 => vec_dot_q5_0_f32(row_bytes, x),
                        DType::Q5_1 => vec_dot_q5_1_f32(row_bytes, x),
                        _ => panic!("Unsupported quantized linear type {:?}", dtype),
                    };

                    let val = if let Some(b) = b_sub {
                        dot + b[r_rel]
                    } else {
                        dot
                    };

                    unsafe {
                        *out_slice_ptr.add(r_abs) = val;
                    }
                }
            });
        }
    });
}
