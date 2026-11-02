#![allow(clippy::needless_range_loop)]
use half::f16;
use whisper_tensor::ops::quant::{vec_dot_q4_k_f32, vec_dot_q5_k_f32, vec_dot_q6_k_f32};
use whisper_tensor::DType;

#[test]
fn test_k_quants_block_and_type_sizes() {
    assert_eq!(DType::Q4_K.block_size(), 256);
    assert_eq!(DType::Q4_K.type_size(), 144);

    assert_eq!(DType::Q5_K.block_size(), 256);
    assert_eq!(DType::Q5_K.type_size(), 176);

    assert_eq!(DType::Q6_K.block_size(), 256);
    assert_eq!(DType::Q6_K.type_size(), 210);

    assert_eq!(DType::from_u32(12).unwrap(), DType::Q4_K);
    assert_eq!(DType::from_u32(13).unwrap(), DType::Q5_K);
    assert_eq!(DType::from_u32(14).unwrap(), DType::Q6_K);
}

#[test]
fn test_q4_k_dot_product_and_dequant() {
    // Construct a synthetic Q4_K block of 256 elements
    let mut raw = vec![0u8; 144];
    // scale d = 1.0 (FP16: 0x3C00)
    let d_bits = f16::from_f32(1.0).to_bits();
    raw[0..2].copy_from_slice(&d_bits.to_le_bytes());
    // dmin = 0.0
    let dmin_bits = f16::from_f32(0.0).to_bits();
    raw[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
    // subblock scales: set scale 1 for all subblocks
    for i in 0..4 {
        raw[4 + i] = 1; // scale 1
        raw[8 + i] = 0; // min 0
    }

    // Set non-zero nibbles
    for i in 16..144 {
        raw[i] = 0x21; // low nibble = 1, high nibble = 2
    }

    let x = vec![1.0f32; 256];
    let dot = vec_dot_q4_k_f32(&raw, &x);
    assert!(dot > 0.0, "Dot product must be strictly positive");

    let dequant = DType::Q4_K.dequantize_to_f32(&raw, 256).unwrap();
    assert_eq!(dequant.len(), 256);
    for &val in &dequant {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q5_k_dot_product_and_dequant() {
    let mut raw = vec![0u8; 176];
    let d_bits = f16::from_f32(1.0).to_bits();
    raw[0..2].copy_from_slice(&d_bits.to_le_bytes());
    for i in 0..4 {
        raw[4 + i] = 1;
    }
    for i in 48..176 {
        raw[i] = 0x32;
    }

    let x = vec![0.5f32; 256];
    let dot = vec_dot_q5_k_f32(&raw, &x);
    assert!(dot > 0.0);

    let dequant = DType::Q5_K.dequantize_to_f32(&raw, 256).unwrap();
    assert_eq!(dequant.len(), 256);
}

#[test]
fn test_q6_k_dot_product_and_dequant() {
    let mut raw = vec![0u8; 210];
    let d_bits = f16::from_f32(1.0).to_bits();
    raw[208..210].copy_from_slice(&d_bits.to_le_bytes());
    // scales: set scale 1 for all subblocks
    for i in 192..208 {
        raw[i] = 1;
    }
    // Set non-zero nibbles
    for i in 0..128 {
        raw[i] = 0x22;
    }

    let x = vec![1.0f32; 256];
    let dot = vec_dot_q6_k_f32(&raw, &x);
    assert!(dot.is_finite());

    let dequant = DType::Q6_K.dequantize_to_f32(&raw, 256).unwrap();
    assert_eq!(dequant.len(), 256);
}
