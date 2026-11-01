use crate::error::TensorError;
use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
}

impl DType {
    pub fn from_u32(val: u32) -> Result<Self, TensorError> {
        match val {
            0 => Ok(DType::F32),
            1 => Ok(DType::F16),
            2 => Ok(DType::Q4_0),
            3 => Ok(DType::Q4_1),
            6 => Ok(DType::Q5_0),
            7 => Ok(DType::Q5_1),
            8 => Ok(DType::Q8_0),
            9 => Ok(DType::Q8_1),
            _ => Err(TensorError::InvalidOperation {
                message: format!("Unsupported GGML tensor type: {}", val),
            }),
        }
    }

    /// Number of elements per block.
    pub fn block_size(self) -> usize {
        match self {
            DType::F32 | DType::F16 => 1,
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q8_1 | DType::Q5_0 | DType::Q5_1 => 32,
        }
    }

    /// Size in bytes of one block.
    pub fn type_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q4_0 => 2 + 16,         // d(f16) + 16 bytes of 4-bit pairs
            DType::Q4_1 => 2 + 2 + 16,     // d(f16) + m(f16) + 16 bytes
            DType::Q8_0 => 2 + 32,         // d(f16) + 32 bytes of int8
            DType::Q8_1 => 2 + 2 + 32,     // d(f16) + s(f16) + 32 bytes of int8
            DType::Q5_0 => 2 + 4 + 16,     // d(f16) + 4 bytes qh + 16 bytes qs
            DType::Q5_1 => 2 + 2 + 4 + 16, // d(f16) + m(f16) + 4 bytes qh + 16 bytes qs
        }
    }

    /// Calculate total bytes required for a given number of elements.
    pub fn byte_size_for(self, num_elements: usize) -> usize {
        let blk = self.block_size();
        let n_blocks = num_elements.div_ceil(blk);
        n_blocks * self.type_size()
    }

    /// Dequantize raw bytes to a Vec<f32>.
    pub fn dequantize_to_f32(
        self,
        bytes: &[u8],
        num_elements: usize,
    ) -> Result<Vec<f32>, TensorError> {
        let expected_bytes = self.byte_size_for(num_elements);
        if bytes.len() < expected_bytes {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Buffer length {} is shorter than expected {} for {:?} ({} elements)",
                    bytes.len(),
                    expected_bytes,
                    self,
                    num_elements
                ),
            });
        }

        match self {
            DType::F32 => {
                let mut out = Vec::with_capacity(num_elements);
                let (chunks, _) = bytes[..num_elements * 4].as_chunks::<4>();
                for chunk in chunks {
                    let val = f32::from_le_bytes(*chunk);
                    out.push(val);
                }
                Ok(out)
            }
            DType::F16 => {
                let mut out = Vec::with_capacity(num_elements);
                let (chunks, _) = bytes[..num_elements * 2].as_chunks::<2>();
                for chunk in chunks {
                    let bits = u16::from_le_bytes(*chunk);
                    let val = f16::from_bits(bits).to_f32();
                    out.push(val);
                }
                Ok(out)
            }
            DType::Q8_0 => {
                let mut out = Vec::with_capacity(num_elements);
                let block_bytes = self.type_size();
                let mut remaining = num_elements;
                for chunk in bytes.chunks_exact(block_bytes) {
                    let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let d = f16::from_bits(d_bits).to_f32();
                    let qs = &chunk[2..34];
                    let count = remaining.min(32);
                    for &q in &qs[..count] {
                        out.push((q as i8 as f32) * d);
                    }
                    remaining -= count;
                    if remaining == 0 {
                        break;
                    }
                }
                Ok(out)
            }
            DType::Q4_0 => {
                let mut out = Vec::with_capacity(num_elements);
                let block_bytes = self.type_size();
                for chunk in bytes.chunks_exact(block_bytes) {
                    let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let d = f16::from_bits(d_bits).to_f32();
                    let qs = &chunk[2..18];
                    let mut block = [0.0f32; 32];
                    for j in 0..16 {
                        let x0 = (qs[j] & 0x0F) as i8 - 8;
                        let x1 = ((qs[j] >> 4) & 0x0F) as i8 - 8;
                        block[j] = (x0 as f32) * d;
                        block[j + 16] = (x1 as f32) * d;
                    }
                    let take = (num_elements - out.len()).min(32);
                    out.extend_from_slice(&block[..take]);
                    if out.len() == num_elements {
                        break;
                    }
                }
                Ok(out)
            }
            DType::Q4_1 => {
                let mut out = Vec::with_capacity(num_elements);
                let block_bytes = self.type_size();
                for chunk in bytes.chunks_exact(block_bytes) {
                    let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let m_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
                    let d = f16::from_bits(d_bits).to_f32();
                    let m = f16::from_bits(m_bits).to_f32();
                    let qs = &chunk[4..20];
                    let mut block = [0.0f32; 32];
                    for j in 0..16 {
                        let x0 = (qs[j] & 0x0F) as f32;
                        let x1 = ((qs[j] >> 4) & 0x0F) as f32;
                        block[j] = x0 * d + m;
                        block[j + 16] = x1 * d + m;
                    }
                    let take = (num_elements - out.len()).min(32);
                    out.extend_from_slice(&block[..take]);
                    if out.len() == num_elements {
                        break;
                    }
                }
                Ok(out)
            }
            DType::Q5_0 => {
                let mut out = Vec::with_capacity(num_elements);
                let block_bytes = self.type_size();
                for chunk in bytes.chunks_exact(block_bytes) {
                    let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let d = f16::from_bits(d_bits).to_f32();
                    let qh = u32::from_le_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);
                    let qs = &chunk[6..22];
                    let mut block = [0.0f32; 32];
                    for j in 0..16 {
                        let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                        let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
                        let x0 = (((qs[j] & 0x0F) | xh_0) as i32) - 16;
                        let x1 = ((((qs[j] >> 4) & 0x0F) | xh_1) as i32) - 16;
                        block[j] = (x0 as f32) * d;
                        block[j + 16] = (x1 as f32) * d;
                    }
                    let take = (num_elements - out.len()).min(32);
                    out.extend_from_slice(&block[..take]);
                    if out.len() == num_elements {
                        break;
                    }
                }
                Ok(out)
            }
            DType::Q5_1 => {
                let mut out = Vec::with_capacity(num_elements);
                let block_bytes = self.type_size();
                for chunk in bytes.chunks_exact(block_bytes) {
                    let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let m_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
                    let d = f16::from_bits(d_bits).to_f32();
                    let m = f16::from_bits(m_bits).to_f32();
                    let qh = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                    let qs = &chunk[8..24];
                    let mut block = [0.0f32; 32];
                    for j in 0..16 {
                        let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                        let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
                        let x0 = ((qs[j] & 0x0F) | xh_0) as f32;
                        let x1 = (((qs[j] >> 4) & 0x0F) | xh_1) as f32;
                        block[j] = x0 * d + m;
                        block[j + 16] = x1 * d + m;
                    }
                    let take = (num_elements - out.len()).min(32);
                    out.extend_from_slice(&block[..take]);
                    if out.len() == num_elements {
                        break;
                    }
                }
                Ok(out)
            }
            DType::Q8_1 => {
                let mut out = Vec::with_capacity(num_elements);
                let block_bytes = self.type_size();
                for chunk in bytes.chunks_exact(block_bytes) {
                    let d_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let d = f16::from_bits(d_bits).to_f32();
                    let qs = &chunk[4..36];
                    let count = (num_elements - out.len()).min(32);
                    for &q in &qs[..count] {
                        out.push((q as i8 as f32) * d);
                    }
                    if out.len() == num_elements {
                        break;
                    }
                }
                Ok(out)
            }
        }
    }
}
