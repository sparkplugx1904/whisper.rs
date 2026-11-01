use whisper_tensor::DType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: usize,
    pub ne: Vec<usize>,
    pub dtype: DType,
    pub offset: usize,
    pub byte_size: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.ne.iter().product()
    }
}
