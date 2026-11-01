use crate::filters::MelFilters;
use crate::header::Hyperparameters;
use crate::tensor_info::TensorInfo;
use crate::vocab::Vocab;
use crate::ModelError;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use whisper_tensor::{DType, Tensor};

pub enum ModelStorage {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl ModelStorage {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            ModelStorage::Mmap(m) => m,
            ModelStorage::Bytes(b) => b,
        }
    }
}

pub struct WhisperModel {
    pub hparams: Hyperparameters,
    pub filters: MelFilters,
    pub vocab: Vocab,
    pub tensors: HashMap<String, TensorInfo>,
    pub total_weight_bytes: usize,
    storage: Arc<ModelStorage>,
}

impl WhisperModel {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let file = File::open(path)?;
        // Use unsafe for mmap since it can map arbitrary files into memory
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::load_from_storage(ModelStorage::Mmap(mmap))
    }

    pub fn load_from_bytes(bytes: Vec<u8>) -> Result<Self, ModelError> {
        Self::load_from_storage(ModelStorage::Bytes(bytes))
    }

    fn load_from_storage(storage: ModelStorage) -> Result<Self, ModelError> {
        let slice = storage.as_slice();
        let mut cursor = Cursor::new(slice);

        let hparams = Hyperparameters::read(&mut cursor)?;
        let filters = MelFilters::read(&mut cursor)?;
        let vocab = Vocab::read(&mut cursor, &hparams)?;

        let mut tensors = HashMap::new();
        let mut total_weight_bytes = 0;

        loop {
            let n_dims_res = cursor.read_i32::<LittleEndian>();
            let n_dims = match n_dims_res {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(ModelError::Io(e)),
            };

            if !(0..=4).contains(&n_dims) {
                return Err(ModelError::FormatError(format!(
                    "Invalid n_dims {} (expected 0..=4)",
                    n_dims
                )));
            }

            let name_len = cursor.read_i32::<LittleEndian>()? as usize;
            let ttype_raw = cursor.read_u32::<LittleEndian>()?;
            let dtype = DType::from_u32(ttype_raw).map_err(|e| {
                ModelError::FormatError(format!("Invalid dtype {}: {}", ttype_raw, e))
            })?;

            let mut ne = Vec::with_capacity(n_dims as usize);
            let mut numel = 1usize;
            for _ in 0..n_dims {
                let dim = cursor.read_i32::<LittleEndian>()? as usize;
                ne.push(dim);
                numel *= dim;
            }

            let mut name_buf = vec![0u8; name_len];
            cursor.read_exact(&mut name_buf)?;
            let name = String::from_utf8_lossy(&name_buf).into_owned();

            let byte_size = dtype.byte_size_for(numel);
            let offset = cursor.position() as usize;

            // Seek past tensor data
            cursor.seek(SeekFrom::Current(byte_size as i64))?;

            total_weight_bytes += byte_size;

            let tensor_info = TensorInfo {
                name: name.clone(),
                n_dims: n_dims as usize,
                ne,
                dtype,
                offset,
                byte_size,
            };

            tensors.insert(name, tensor_info);
        }

        Ok(Self {
            hparams,
            filters,
            vocab,
            tensors,
            total_weight_bytes,
            storage: Arc::new(storage),
        })
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn get_tensor_raw_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let slice = self.storage.as_slice();
        slice.get(info.offset..info.offset + info.byte_size)
    }

    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let info = self.tensors.get(name).ok_or_else(|| {
            ModelError::FormatError(format!("Tensor '{}' not found in model", name))
        })?;
        let bytes = self.get_tensor_raw_bytes(name).ok_or_else(|| {
            ModelError::FormatError(format!("Failed to slice tensor bytes for '{}'", name))
        })?;

        info.dtype
            .dequantize_to_f32(bytes, info.numel())
            .map_err(|e| ModelError::FormatError(format!("Dequant error for '{}': {}", name, e)))
    }

    pub fn get_tensor(&self, name: &str) -> Result<Tensor, ModelError> {
        let info = self.tensors.get(name).ok_or_else(|| {
            ModelError::FormatError(format!("Tensor '{}' not found in model", name))
        })?;
        let data = self.get_tensor_f32(name)?;
        Tensor::from_data(&info.ne, data)
            .map_err(|e| ModelError::FormatError(format!("Tensor shape error: {}", e)))
    }
}
