use crate::error::TensorError;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for (i, &d) in shape.iter().enumerate().rev() {
            strides[i] = stride;
            stride *= d;
        }
        strides
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let strides = Self::compute_strides(shape);
        Self {
            shape: shape.to_vec(),
            strides,
            data: vec![0.0; numel],
        }
    }

    pub fn from_data(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![data.len()],
            });
        }
        let strides = Self::compute_strides(shape);
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            data,
        })
    }

    pub fn from_slice(shape: &[usize], slice: &[f32]) -> Result<Self, TensorError> {
        Self::from_data(shape, slice.to_vec())
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn dim(&self, d: usize) -> usize {
        self.shape[d]
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn is_contiguous(&self) -> bool {
        let expected = Self::compute_strides(&self.shape);
        self.strides == expected
    }

    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        let mut new_data = vec![0.0; self.numel()];
        let mut indices = vec![0; self.ndim()];
        for val in &mut new_data {
            let offset = indices
                .iter()
                .zip(&self.strides)
                .map(|(&idx, &s)| idx * s)
                .sum::<usize>();
            *val = self.data[offset];

            // increment indices
            for dim in (0..self.ndim()).rev() {
                indices[dim] += 1;
                if indices[dim] < self.shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }
        let strides = Self::compute_strides(&self.shape);
        Self {
            shape: self.shape.clone(),
            strides,
            data: new_data,
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: new_shape.to_vec(),
                actual: self.shape.clone(),
            });
        }
        let contiguous = self.to_contiguous();
        let strides = Self::compute_strides(new_shape);
        Ok(Self {
            shape: new_shape.to_vec(),
            strides,
            data: contiguous.data,
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(TensorError::DimOutOfBounds {
                dim: dim0.max(dim1),
                rank: self.ndim(),
            });
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            data: self.data.clone(),
        })
    }
}
