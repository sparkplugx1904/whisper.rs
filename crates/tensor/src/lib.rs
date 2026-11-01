//! Tensor data structures and operations for whisper.rs.

pub mod dtype;
pub mod error;
pub mod tensor;

pub use dtype::DType;
pub use error::TensorError;
pub use tensor::Tensor;
