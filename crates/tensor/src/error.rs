use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Dimension out of bounds: dim {dim} for rank {rank}")]
    DimOutOfBounds { dim: usize, rank: usize },
    #[error("Invalid slice range: {message}")]
    InvalidSlice { message: String },
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },
}
