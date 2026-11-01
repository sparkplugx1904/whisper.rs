use crate::ModelError;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

#[derive(Debug, Clone, PartialEq)]
pub struct MelFilters {
    pub n_mel: usize,
    pub n_fft: usize,
    pub data: Vec<f32>,
}

impl MelFilters {
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, ModelError> {
        let n_mel = reader.read_i32::<LittleEndian>()? as usize;
        let n_fft = reader.read_i32::<LittleEndian>()? as usize;

        let num_elements = n_mel * n_fft;
        let mut data = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(reader.read_f32::<LittleEndian>()?);
        }

        Ok(Self { n_mel, n_fft, data })
    }
}
