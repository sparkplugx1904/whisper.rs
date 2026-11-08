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

/// Reader wrapper that records every byte consumed by the model parser.
struct TeeReader<'a, R> {
    inner: &'a mut R,
    buffer: &'a mut Vec<u8>,
}

impl<R: Read> Read for TeeReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.buffer.extend_from_slice(&buf[..n]);
        Ok(n)
    }
}

impl<R: Read> TeeReader<'_, R> {
    /// Read tensor payload directly into the backing buffer, avoiding a second
    /// allocation the size of the tensor.
    fn read_data(&mut self, len: usize) -> std::io::Result<()> {
        let start = self.buffer.len();
        self.buffer.resize(start + len, 0);
        self.inner.read_exact(&mut self.buffer[start..])
    }
}

/// Zero-copy, non-owning view of a raw tensor directly from the model's backing storage.
///
/// Holds a direct borrowed slice into the memory-mapped file or in-memory byte buffer,
/// avoiding any memory copies or upfront float32 dequantization.
///
/// # Architecture & Performance Rationale
/// In standard transformer inference pipelines, weights are frequently converted into 32-bit floats
/// upfront during model loading (`load_from_file`). For large models (e.g. Whisper large with
/// 1.5 billion parameters), FP32 weights consume over 6 GB of RAM and saturate the CPU memory bus
/// during single-token autoregressive decoding ($M=1$ GEMV operations).
///
/// By providing `RawTensorView`, the model exposes the underlying quantized byte buffers (Q8_0,
/// Q4_0, Q4_K, etc.) with zero copies. These views can be passed directly to SIMD quantized GEMV
/// kernels (`quantized_gemv_parallel`), keeping weights compressed in memory and loading 4x to 7x
/// more weights per 64-byte cache line during inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawTensorView<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub ne: [usize; 4],
    pub data: &'a [u8],
}

impl<'a> RawTensorView<'a> {
    /// Total number of logical elements across all 4 dimensions.
    #[inline]
    pub fn numel(&self) -> usize {
        self.ne.iter().product()
    }

    /// Number of rows (outer dimension, `ne[1]`) for 2D matrix representations.
    #[inline]
    pub fn rows(&self) -> usize {
        self.ne[1]
    }

    /// Number of columns (inner dimension, `ne[0]`) for 2D matrix representations.
    #[inline]
    pub fn cols(&self) -> usize {
        self.ne[0]
    }

    /// Dequantizes the raw byte buffer into an owned `Vec<f32>` on demand.
    /// Useful for fallback paths or operators that only operate on FP32.
    pub fn dequantize(&self) -> Result<Vec<f32>, whisper_tensor::TensorError> {
        self.dtype.dequantize_to_f32(self.data, self.numel())
    }

    /// Converts this zero-copy view into an owned `QuantizedTensor` by copying
    /// only the compact quantized bytes (not dequantizing to float32).
    pub fn to_quantized_tensor(&self) -> QuantizedTensor {
        let mut shape = Vec::new();
        for &d in &self.ne {
            if d > 1 || shape.is_empty() {
                shape.push(d);
            }
        }
        QuantizedTensor {
            name: self.name.to_string(),
            dtype: self.dtype,
            shape,
            ne: self.ne,
            data: self.data.to_vec(),
        }
    }

    /// Performs direct parallel matrix-vector multiplication $y = A_{\text{raw}} \cdot x + b$
    /// directly on the zero-copy slice without allocating any heap memory.
    pub fn gemv(&self, x: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
        let cols = self.ne[0];
        let rows = self.ne[1];
        whisper_tensor::ops::quantized_gemv_parallel(self.data, self.dtype, rows, cols, x, bias, y);
    }
}

/// An owned representation of a tensor whose weights remain in compact quantized format
/// (e.g. Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K) without upfront dequantization to `Vec<f32>`.
///
/// # Architecture & Performance Rationale
/// 1. **Dramatically Lower RAM Footprint**:
///    Q8_0 requires ~1.06 bytes per weight (vs 4.0 bytes for FP32, a 3.76x reduction).
///    Q4_0 / Q4_K require ~0.56 bytes per weight (a 7.1x reduction).
///    For Whisper Large (1.5 billion parameters), keeping weights quantized in memory
///    drops the weight storage from ~6.0 GB to ~850 MB.
/// 2. **Eliminates Startup Dequantization Stall**:
///    Loading a model no longer requires iterating through millions of weights to convert
///    them to float32 upfront before inference can begin.
/// 3. **Relieves Memory Bus Saturation & CPU Cache Pressure**:
///    In autoregressive decoding, each token generation step requires streaming the entire weight
///    matrix through the CPU once ($M=1$ GEMV). Because $M=1$ has an arithmetic intensity of only
///    ~0.5 FLOP/byte, execution is strictly memory-bandwidth bound. Storing weights in quantized
///    format cuts memory traffic by 70%-85%, dramatically accelerating decoding throughput.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub ne: [usize; 4],
    pub data: Vec<u8>,
}

impl QuantizedTensor {
    /// Creates a new `QuantizedTensor` from owned byte data and shape.
    pub fn new(name: impl Into<String>, dtype: DType, shape: Vec<usize>, data: Vec<u8>) -> Self {
        let mut ne = [1usize; 4];
        for (i, &d) in shape.iter().enumerate().take(4) {
            ne[i] = d;
        }
        Self {
            name: name.into(),
            dtype,
            shape,
            ne,
            data,
        }
    }

    /// Name identifier of the tensor within the model.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Quantization format (dtype) of the underlying weights.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Slice of the tensor's dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 4-element GGML dimension array [ne0, ne1, ne2, ne3].
    #[inline]
    pub fn ne(&self) -> &[usize; 4] {
        &self.ne
    }

    /// Raw quantized byte buffer.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Total number of logical elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of rows (outer dimension, `ne[1]`) for 2D matrix representations.
    #[inline]
    pub fn rows(&self) -> usize {
        self.ne[1]
    }

    /// Number of columns (inner dimension, `ne[0]`) for 2D matrix representations.
    #[inline]
    pub fn cols(&self) -> usize {
        self.ne[0]
    }

    /// Borrows a zero-copy view of this quantized tensor.
    pub fn as_view(&self) -> RawTensorView<'_> {
        RawTensorView {
            name: &self.name,
            dtype: self.dtype,
            ne: self.ne,
            data: &self.data,
        }
    }

    /// Dequantizes the weights on-demand into an owned `Vec<f32>`.
    pub fn dequantize(&self) -> Result<Vec<f32>, whisper_tensor::TensorError> {
        self.dtype.dequantize_to_f32(&self.data, self.numel())
    }

    /// Converts this quantized tensor into an owned `whisper_tensor::Tensor` (FP32).
    pub fn to_f32_tensor(&self) -> Result<Tensor, ModelError> {
        let f32_data = self.dequantize().map_err(|e| {
            ModelError::FormatError(format!("Failed to dequantize '{}': {}", self.name, e))
        })?;
        Tensor::from_data(&self.shape, f32_data)
            .map_err(|e| ModelError::FormatError(format!("Tensor shape error: {}", e)))
    }

    /// Performs direct parallel matrix-vector multiplication $y = A_{\text{quantized}} \cdot x + b$
    /// without converting matrix $A$ to $f32$ in RAM.
    pub fn gemv(&self, x: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
        let cols = self.ne[0];
        let rows = self.ne[1];
        whisper_tensor::ops::quantized_gemv_parallel(
            &self.data, self.dtype, rows, cols, x, bias, y,
        );
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

    /// Load a model from a streaming reader.
    ///
    /// `eof` is consulted before each tensor header because the C model-loader
    /// contract lets `read` return the requested size even at end of file; the
    /// callback is the authoritative end marker. Only the bytes that belong to
    /// the model are retained, so a loader that over-reports its final read
    /// does not leave trailing garbage in the backing storage.
    pub fn load_from_reader<R: Read>(
        reader: &mut R,
        mut eof: impl FnMut() -> bool,
    ) -> Result<Self, ModelError> {
        let mut buffer = Vec::new();
        let mut tensors = HashMap::new();
        let mut total_weight_bytes = 0;

        let (hparams, filters, vocab) = {
            let mut tee = TeeReader {
                inner: reader,
                buffer: &mut buffer,
            };

            let hparams = Hyperparameters::read(&mut tee)?;
            let filters = MelFilters::read(&mut tee)?;
            let vocab = Vocab::read(&mut tee, &hparams)?;

            loop {
                if eof() {
                    break;
                }

                let n_dims = match tee.read_i32::<LittleEndian>() {
                    Ok(n_dims) => n_dims,
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(error) => return Err(ModelError::Io(error)),
                };

                // The reference loader checks eof after reading the header
                // triple; a final over-reported read must not be parsed.
                if eof() {
                    break;
                }

                if !(0..=4).contains(&n_dims) {
                    return Err(ModelError::FormatError(format!(
                        "Invalid n_dims {} (expected 0..=4)",
                        n_dims
                    )));
                }

                let name_len = tee.read_i32::<LittleEndian>()? as usize;
                let ttype_raw = tee.read_u32::<LittleEndian>()?;
                let dtype = DType::from_u32(ttype_raw).map_err(|e| {
                    ModelError::FormatError(format!("Invalid dtype {}: {}", ttype_raw, e))
                })?;

                let mut ne = Vec::with_capacity(n_dims as usize);
                let mut numel = 1usize;
                for _ in 0..n_dims {
                    let dim = tee.read_i32::<LittleEndian>()? as usize;
                    ne.push(dim);
                    numel *= dim;
                }

                let mut name_buf = vec![0u8; name_len];
                tee.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).into_owned();

                let byte_size = dtype.byte_size_for(numel);
                let offset = tee.buffer.len();
                tee.read_data(byte_size)?;

                total_weight_bytes += byte_size;
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        name,
                        n_dims: n_dims as usize,
                        ne,
                        dtype,
                        offset,
                        byte_size,
                    },
                );
            }

            (hparams, filters, vocab)
        };

        Ok(Self {
            hparams,
            filters,
            vocab,
            tensors,
            total_weight_bytes,
            storage: Arc::new(ModelStorage::Bytes(buffer)),
        })
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
    /// Returns a zero-copy, non-owning view of a raw tensor directly from the model's backing storage.
    ///
    /// This method provides direct access to the quantized block data (such as Q8_0 or Q4_K bytes)
    /// without copying or allocating memory on the heap. If the model was loaded via memory mapping
    /// (`ModelStorage::Mmap`), the returned view directly references the OS page cache.
    ///
    /// # Performance Advantages
    /// - **Zero Allocation**: Avoids creating temporary vectors or dequantized float buffers.
    /// - **Minimal Memory Bandwidth**: Enables passing the slice directly into SIMD microkernels
    ///   like `quantized_gemv_parallel`, keeping RAM bandwidth consumption at a minimum.
    pub fn get_tensor_raw<'a>(&'a self, name: &str) -> Result<RawTensorView<'a>, ModelError> {
        let info = self.tensors.get(name).ok_or_else(|| {
            ModelError::FormatError(format!("Tensor '{}' not found in model", name))
        })?;
        let data = self.get_tensor_raw_bytes(name).ok_or_else(|| {
            ModelError::FormatError(format!("Failed to slice tensor bytes for '{}'", name))
        })?;

        let mut ne = [1usize; 4];
        for (i, &d) in info.ne.iter().enumerate().take(4) {
            ne[i] = d;
        }

        Ok(RawTensorView {
            name: &info.name,
            dtype: info.dtype,
            ne,
            data,
        })
    }

    /// Returns an owned `QuantizedTensor` representing the tensor's weights in their compact
    /// quantized format (e.g. Q8_0, Q4_0, Q4_K) without upfront dequantization to `Vec<f32>`.
    ///
    /// Preserving quantized weights in memory saves 4x to 8x RAM and enables direct quantized
    /// GEMV operations during decoding.
    pub fn get_tensor_quantized(&self, name: &str) -> Result<QuantizedTensor, ModelError> {
        let raw = self.get_tensor_raw(name)?;
        let info = self.tensors.get(name).unwrap();
        Ok(QuantizedTensor {
            name: info.name.clone(),
            dtype: raw.dtype,
            shape: info.ne.clone(),
            ne: raw.ne,
            data: raw.data.to_vec(),
        })
    }

    /// Returns a 2D weight tensor as a [`whisper_tensor::LinearWeight`], preserving its
    /// on-disk storage format.
    ///
    /// Quantized tensors are deliberately not expanded to `f32` here. Single-token
    /// autoregressive decoding is memory-bandwidth bound: each step streams the whole
    /// weight matrix through the CPU for an $M=1$ GEMV, so keeping the compact block
    /// layout cuts memory traffic by 4x to 8x. Only F32/F16 tensors, which are already
    /// float data, are materialized into a `Vec<f32>`.
    pub fn get_linear_weight(
        &self,
        name: &str,
    ) -> Result<whisper_tensor::LinearWeight, ModelError> {
        let raw = self.get_tensor_raw(name)?;
        let numel = raw.numel();
        let rows = raw.ne[1];
        let cols = raw.ne[0];

        if cols * rows != numel {
            return Err(ModelError::FormatError(format!(
                "Tensor '{}' is not a 2D matrix: ne = {:?}",
                name, raw.ne
            )));
        }

        let expected_bytes = raw.dtype.byte_size_for(numel);
        if raw.data.len() != expected_bytes {
            return Err(ModelError::FormatError(format!(
                "Tensor '{}' has {} bytes, expected {} for {:?} ({} elements)",
                name,
                raw.data.len(),
                expected_bytes,
                raw.dtype,
                numel
            )));
        }

        match raw.dtype {
            DType::F32 | DType::F16 => {
                let data = raw.dequantize().map_err(|e| {
                    ModelError::FormatError(format!("Failed to dequantize '{}': {}", name, e))
                })?;
                Ok(whisper_tensor::LinearWeight::from_f32(rows, cols, data))
            }
            _ => Ok(whisper_tensor::LinearWeight::from_quantized(
                rows,
                cols,
                raw.dtype,
                raw.data.to_vec(),
            )),
        }
    }

    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let raw = self.get_tensor_raw(name)?;
        raw.dequantize()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Little-endian f16 encoding of a scale factor, produced through the
    /// tensor dtype helper so the test file needs no extra dependency.
    fn f16_le(value: f32) -> [u8; 2] {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(&DType::F16.quantize_from_f32(&[value]).unwrap());
        bytes
    }

    fn create_synthetic_model() -> WhisperModel {
        let mut storage_bytes = Vec::new();

        // 1. Synthetic Q8_0 tensor (2 rows, 32 cols -> 2 blocks of 34 bytes = 68 bytes)
        let q8_offset = storage_bytes.len();
        let mut q8_bytes = vec![0u8; 68];
        // Row 0: scale 0.5 (f16: 0x3800), weights 0..32
        q8_bytes[0..2].copy_from_slice(&f16_le(0.5));
        for i in 0..32 {
            q8_bytes[2 + i] = (i as i8 - 16) as u8;
        }
        // Row 1: scale 0.25, weights 32..64
        q8_bytes[34..36].copy_from_slice(&f16_le(0.25));
        for i in 0..32 {
            q8_bytes[36 + i] = (i as i8) as u8;
        }
        storage_bytes.extend_from_slice(&q8_bytes);

        // 2. Synthetic Q4_0 tensor (1 row, 32 cols -> 1 block of 18 bytes)
        let q4_offset = storage_bytes.len();
        let mut q4_bytes = vec![0u8; 18];
        q4_bytes[0..2].copy_from_slice(&f16_le(1.0));
        for i in 0..16 {
            q4_bytes[2 + i] = (i as u8) | (((15 - i) as u8) << 4);
        }
        storage_bytes.extend_from_slice(&q4_bytes);

        // 3. Synthetic F32 tensor (2 rows, 4 cols -> 32 bytes)
        let f32_offset = storage_bytes.len();
        let f32_values: [f32; 8] = [1.0, -2.0, 3.5, 0.25, -0.5, 4.0, 2.0, -1.0];
        for v in f32_values {
            storage_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let mut tensors = HashMap::new();
        tensors.insert(
            "test.weight.q8".to_string(),
            TensorInfo {
                name: "test.weight.q8".to_string(),
                n_dims: 2,
                ne: vec![32, 2],
                dtype: DType::Q8_0,
                offset: q8_offset,
                byte_size: 68,
            },
        );
        tensors.insert(
            "test.weight.q4".to_string(),
            TensorInfo {
                name: "test.weight.q4".to_string(),
                n_dims: 2,
                ne: vec![32, 1],
                dtype: DType::Q4_0,
                offset: q4_offset,
                byte_size: 18,
            },
        );
        tensors.insert(
            "test.weight.f32".to_string(),
            TensorInfo {
                name: "test.weight.f32".to_string(),
                n_dims: 2,
                ne: vec![4, 2],
                dtype: DType::F32,
                offset: f32_offset,
                byte_size: 32,
            },
        );

        WhisperModel {
            hparams: Hyperparameters {
                n_vocab: 51864,
                n_audio_ctx: 1500,
                n_audio_state: 384,
                n_audio_head: 6,
                n_audio_layer: 4,
                n_text_ctx: 448,
                n_text_state: 384,
                n_text_head: 6,
                n_text_layer: 4,
                n_mels: 80,
                ftype: 1,
                qntvr: 0,
                model_type: crate::header::ModelType::Tiny,
            },
            filters: MelFilters {
                n_mel: 80,
                n_fft: 201,
                data: vec![0.0; 80 * 201],
            },
            vocab: Vocab {
                token_to_id: HashMap::new(),
                id_to_token: Vec::new(),
                token_bytes: Vec::new(),
                n_vocab: 51864,
                token_eot: 50256,
                token_sot: 50257,
                token_translate: 50358,
                token_transcribe: 50359,
                token_solm: 50360,
                token_prev: 50361,
                token_nosp: 50362,
                token_not: 50363,
                token_beg: 50364,
            },
            tensors,
            total_weight_bytes: storage_bytes.len(),
            storage: Arc::new(ModelStorage::Bytes(storage_bytes)),
        }
    }

    #[test]
    fn test_raw_tensor_view_zero_copy() {
        let model = create_synthetic_model();

        let raw = model
            .get_tensor_raw("test.weight.q8")
            .expect("Q8 tensor must exist");
        assert_eq!(raw.name, "test.weight.q8");
        assert_eq!(raw.dtype, DType::Q8_0);
        assert_eq!(raw.ne, [32, 2, 1, 1]);
        assert_eq!(raw.cols(), 32);
        assert_eq!(raw.rows(), 2);
        assert_eq!(raw.numel(), 64);
        assert_eq!(raw.data.len(), 68);

        // Verify dequantization on demand
        let dequant = raw.dequantize().expect("dequant must succeed");
        assert_eq!(dequant.len(), 64);
        // First value in row 0: (-16) * 0.5 = -8.0
        assert!((dequant[0] - (-8.0)).abs() < 1e-4);

        // Direct gemv on RawTensorView
        let x = vec![1.0f32; 32];
        let mut y = vec![0.0f32; 2];
        raw.gemv(&x, None, &mut y);
        assert!(y[0].is_finite());
        assert!(y[1].is_finite());
    }

    #[test]
    fn test_quantized_tensor_owned() {
        let model = create_synthetic_model();

        let q_tensor = model
            .get_tensor_quantized("test.weight.q4")
            .expect("Q4 tensor must exist");
        assert_eq!(q_tensor.name(), "test.weight.q4");
        assert_eq!(q_tensor.dtype(), DType::Q4_0);
        assert_eq!(q_tensor.ne(), &[32, 1, 1, 1]);
        assert_eq!(q_tensor.shape(), &[32, 1]);
        assert_eq!(q_tensor.numel(), 32);
        assert_eq!(q_tensor.data().len(), 18);

        // Test conversion to view
        let view = q_tensor.as_view();
        assert_eq!(view.name, "test.weight.q4");
        assert_eq!(view.data.len(), 18);

        // Test conversion to FP32 Tensor
        let f32_tensor = q_tensor.to_f32_tensor().expect("to_f32_tensor succeeds");
        assert_eq!(f32_tensor.numel(), 32);

        // Test GEMV on QuantizedTensor. Compare against the dequantized matrix
        // instead of a sign, because the synthetic block is negative on
        // average: low nibbles sum to -8 and high nibbles to -8 with d = 1.0.
        let x = vec![0.5f32; 32];
        let mut y = vec![0.0f32; 1];
        q_tensor.gemv(&x, Some(&[2.0]), &mut y);

        let reference: f32 = q_tensor
            .dequantize()
            .expect("dequantize succeeds")
            .iter()
            .zip(x.iter())
            .map(|(w, xv)| w * xv)
            .sum::<f32>()
            + 2.0;
        assert!(
            (y[0] - reference).abs() < 1e-4,
            "gemv {} must match dequantized reference {}",
            y[0],
            reference
        );
    }

    #[test]
    fn test_get_linear_weight_preserves_quantization() {
        let model = create_synthetic_model();

        let q8 = model
            .get_linear_weight("test.weight.q8")
            .expect("Q8 linear weight must exist");
        assert!(q8.is_quantized());
        assert_eq!(q8.dtype(), DType::Q8_0);
        assert_eq!(q8.rows(), 2);
        assert_eq!(q8.cols(), 32);
        // Resident footprint must stay at the raw block size, not 4 * numel.
        assert_eq!(q8.resident_bytes(), 68);

        let expected = model.get_tensor_f32("test.weight.q8").expect("dequantizes");
        let actual = q8.to_f32();
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-4, "mismatch: {} vs {}", a, e);
        }

        let f32w = model
            .get_linear_weight("test.weight.f32")
            .expect("F32 linear weight must exist");
        assert!(!f32w.is_quantized());
        assert_eq!(f32w.dtype(), DType::F32);
        assert_eq!(f32w.rows(), 2);
        assert_eq!(f32w.cols(), 4);
        assert_eq!(f32w.resident_bytes(), 32);
        assert_eq!(
            f32w.to_f32(),
            vec![1.0, -2.0, 3.5, 0.25, -0.5, 4.0, 2.0, -1.0]
        );
    }

    #[test]
    fn test_tensor_not_found() {
        let model = create_synthetic_model();
        assert!(model.get_tensor_raw("non_existent").is_err());
        assert!(model.get_tensor_quantized("non_existent").is_err());
    }
}
