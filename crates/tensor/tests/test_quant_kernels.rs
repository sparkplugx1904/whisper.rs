use half::f16;
use whisper_tensor::dtype::DType;
use whisper_tensor::ops::{
    quantized_linear_vector, vec_dot_q4_0_f32, vec_dot_q4_1_f32, vec_dot_q5_1_f32, vec_dot_q8_0_f32,
};
use whisper_tensor::simd::dot_f32;

#[test]
fn test_kernel_dot_q8_0_vs_dequant() {
    let mut raw_block = vec![0u8; 34];
    // scale d = 0.5 (f16: 0x3800)
    let d_bits = f16::from_f32(0.5).to_bits();
    raw_block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    for i in 0..32 {
        raw_block[2 + i] = ((i as i8) - 16) as u8;
    }

    let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

    // 1. Direct quantized dot product
    let dot_direct = vec_dot_q8_0_f32(&raw_block, &x);

    // 2. Dequantize then float dot product
    let dequant = DType::Q8_0.dequantize_to_f32(&raw_block, 32).unwrap();
    let dot_dequant = dot_f32(&dequant, &x);

    let diff = (dot_direct - dot_dequant).abs();
    assert!(
        diff < 1e-4,
        "Q8_0 direct dot {} vs dequant dot {}: diff {}",
        dot_direct,
        dot_dequant,
        diff
    );
}

#[test]
fn test_kernel_dot_q4_0_vs_dequant() {
    let mut raw_block = vec![0u8; 18];
    let d_bits = f16::from_f32(0.25).to_bits();
    raw_block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    for i in 0..16 {
        raw_block[2 + i] = ((i as u8) & 0x0F) | (((15 - i as u8) & 0x0F) << 4);
    }

    let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();

    let dot_direct = vec_dot_q4_0_f32(&raw_block, &x);
    let dequant = DType::Q4_0.dequantize_to_f32(&raw_block, 32).unwrap();
    let dot_dequant = dot_f32(&dequant, &x);

    let diff = (dot_direct - dot_dequant).abs();
    assert!(
        diff < 1e-4,
        "Q4_0 direct dot {} vs dequant dot {}: diff {}",
        dot_direct,
        dot_dequant,
        diff
    );
}

#[test]
fn test_kernel_dot_q4_1_vs_dequant() {
    let mut raw_block = vec![0u8; 20];
    let d_bits = f16::from_f32(0.25).to_bits();
    let m_bits = f16::from_f32(-0.5).to_bits();
    raw_block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    raw_block[2..4].copy_from_slice(&m_bits.to_le_bytes());
    for i in 0..16 {
        raw_block[4 + i] = ((i as u8) & 0x0F) | (((15 - i as u8) & 0x0F) << 4);
    }

    let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();

    let dot_direct = vec_dot_q4_1_f32(&raw_block, &x);
    let dequant = DType::Q4_1.dequantize_to_f32(&raw_block, 32).unwrap();
    let dot_dequant = dot_f32(&dequant, &x);

    let diff = (dot_direct - dot_dequant).abs();
    assert!(
        diff < 1e-4,
        "Q4_1 direct dot {} vs dequant dot {}: diff {}",
        dot_direct,
        dot_dequant,
        diff
    );
}

#[test]
fn test_kernel_dot_q5_1_vs_dequant() {
    let mut raw_block = vec![0u8; 24];
    let d_bits = f16::from_f32(0.125).to_bits();
    let m_bits = f16::from_f32(-1.0).to_bits();
    raw_block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    raw_block[2..4].copy_from_slice(&m_bits.to_le_bytes());
    raw_block[4..8].copy_from_slice(&0xAA55AA55u32.to_le_bytes());
    for i in 0..16 {
        raw_block[8 + i] = ((i as u8 * 3) & 0x0F) | (((i as u8 * 7) & 0x0F) << 4);
    }

    let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.02).collect();

    let dot_direct = vec_dot_q5_1_f32(&raw_block, &x);
    let dequant = DType::Q5_1.dequantize_to_f32(&raw_block, 32).unwrap();
    let dot_dequant = dot_f32(&dequant, &x);

    let diff = (dot_direct - dot_dequant).abs();
    assert!(
        diff < 1e-4,
        "Q5_1 direct dot {} vs dequant dot {}: diff {}",
        dot_direct,
        dot_dequant,
        diff
    );
}

#[test]
fn test_quantized_linear_vector() {
    let rows = 4;
    let cols = 32;
    let bytes_per_row = 34; // Q8_0
    let mut weight_bytes = vec![0u8; rows * bytes_per_row];

    for r in 0..rows {
        let d_bits = f16::from_f32((r + 1) as f32 * 0.1).to_bits();
        let row_slice = &mut weight_bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
        row_slice[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for i in 0..32 {
            row_slice[2 + i] = ((r * 7 + i) % 30) as u8;
        }
    }

    let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.05).collect();
    let bias: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; rows];

    quantized_linear_vector(
        &weight_bytes,
        DType::Q8_0,
        rows,
        cols,
        &x,
        Some(&bias),
        &mut out,
        2,
    );

    // Verify each row
    for r in 0..rows {
        let row_slice = &weight_bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
        let dequant = DType::Q8_0.dequantize_to_f32(row_slice, cols).unwrap();
        let expected = dot_f32(&dequant, &x) + bias[r];
        let diff = (out[r] - expected).abs();
        assert!(
            diff < 1e-4,
            "Row {}: got {}, expected {}",
            r,
            out[r],
            expected
        );
    }
}

#[test]
fn test_quantized_linear_batched_matches_gemv() {
    let rows = 6;
    let cols = 64;
    let num_tokens = 3;
    let bytes_per_row = 68; // Q8_0: 2 blocks of 34 bytes
    let mut weight_bytes = vec![0u8; rows * bytes_per_row];

    for r in 0..rows {
        let d_bits = f16::from_f32((r + 1) as f32 * 0.05).to_bits();
        let row_slice = &mut weight_bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
        for blk in 0..2 {
            let off = blk * 34;
            row_slice[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            for i in 0..32 {
                row_slice[off + 2 + i] = ((r * 11 + blk * 5 + i) % 40) as u8;
            }
        }
    }

    let bias: Vec<f32> = (0..rows).map(|r| r as f32 * 0.5).collect();
    let input: Vec<f32> = (0..num_tokens * cols)
        .map(|i| ((i % 17) as f32) * 0.03 - 0.2)
        .collect();

    let mut batched = vec![0.0f32; num_tokens * rows];
    whisper_tensor::ops::quantized_linear(
        &weight_bytes,
        DType::Q8_0,
        rows,
        cols,
        &input,
        Some(&bias),
        num_tokens,
        &mut batched,
    );

    for t in 0..num_tokens {
        let x_t = &input[t * cols..(t + 1) * cols];
        let mut gemv = vec![0.0f32; rows];
        quantized_linear_vector(
            &weight_bytes,
            DType::Q8_0,
            rows,
            cols,
            x_t,
            Some(&bias),
            &mut gemv,
            2,
        );
        for r in 0..rows {
            let got = batched[t * rows + r];
            let diff = (got - gemv[r]).abs();
            assert!(
                diff < 1e-4,
                "token {} row {}: batched {} vs gemv {}",
                t,
                r,
                got,
                gemv[r]
            );
        }
    }
}
