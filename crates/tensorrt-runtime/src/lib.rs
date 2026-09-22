//! TensorRT 10/11 inference with owned resources and synchronous host tensor I/O.
//! Enable `native` to compile the C++ bridge against your installed SDK.
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// TensorRT I/O types with whole-byte storage. Packed INT4/FP4 are unsupported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    I32 = 3,
    Bool = 4,
    U8 = 5,
    Fp8 = 6,
    Bf16 = 7,
    I64 = 8,
}
impl DataType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I64 => 8,
            _ => 1,
        }
    }
}
impl TryFrom<i32> for DataType {
    type Error = Error;
    fn try_from(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::I8),
            3 => Ok(Self::I32),
            4 => Ok(Self::Bool),
            5 => Ok(Self::U8),
            6 => Ok(Self::Fp8),
            7 => Ok(Self::Bf16),
            8 => Ok(Self::I64),
            _ => Err(Error(format!("Unsupported TensorRT dtype {value}"))),
        }
    }
}

/// Checked storage size for a concrete shape, including scalar and empty tensors.
pub fn tensor_bytes(shape: &[i64], dtype: DataType) -> Result<usize> {
    if shape.len() > 8 || shape.iter().any(|&d| d < 0) {
        return Err(Error(
            "Expected at most eight nonnegative dimensions".into(),
        ));
    }
    shape.iter().try_fold(dtype.size_bytes(), |n, &d| {
        usize::try_from(d)
            .ok()
            .and_then(|d| n.checked_mul(d))
            .ok_or_else(|| Error("Tensor size overflow".into()))
    })
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DataType,
    pub is_input: bool,
    /// Engine metadata may contain -1; inference outputs have concrete dimensions.
    pub shape: Vec<i64>,
}

/// A contiguous, native-endian tensor borrowed for one synchronous inference call.
#[derive(Debug)]
pub struct Input<'a> {
    pub name: &'a str,
    pub dtype: DataType,
    pub shape: &'a [i64],
    pub data: &'a [u8],
}
impl<'a> Input<'a> {
    pub fn f32(name: &'a str, shape: &'a [i64], data: &'a [f32]) -> Self {
        // SAFETY: f32 has no padding; the byte slice borrows the same live allocation.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data))
        };
        Self {
            name,
            dtype: DataType::F32,
            shape,
            data: bytes,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.name.as_bytes().contains(&0) {
            return Err(Error("Tensor name contains NUL".into()));
        }
        if tensor_bytes(self.shape, self.dtype)? != self.data.len() {
            return Err(Error(format!(
                "Input {} byte count does not match shape/dtype",
                self.name
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Output {
    pub info: TensorInfo,
    pub data: Vec<u8>,
}
impl Output {
    /// Copies INT64 indices, including an empty data-dependent output.
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        if self.info.dtype != DataType::I64 || self.data.len() % 8 != 0 {
            return Err(Error("Output is not a valid INT64 buffer".into()));
        }
        Ok(self
            .data
            .chunks_exact(8)
            .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
    /// Copies FP32 values without requiring the byte buffer to be aligned.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        if self.info.dtype != DataType::F32 || self.data.len() % 4 != 0 {
            return Err(Error("Output is not a valid FP32 buffer".into()));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
}

#[cfg(feature = "native")]
mod native;
#[cfg(feature = "native")]
pub use native::{header_version, Session};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_and_invalid_shapes() {
        assert_eq!(tensor_bytes(&[], DataType::F32).unwrap(), 4);
        assert_eq!(tensor_bytes(&[2, 0, 3], DataType::F16).unwrap(), 0);
        assert_eq!(tensor_bytes(&[2, 3], DataType::I64).unwrap(), 48);
        assert!(tensor_bytes(&[-1, 3], DataType::F32).is_err());
        assert!(tensor_bytes(&[1; 9], DataType::U8).is_err());
        assert!(tensor_bytes(&[i64::MAX, 2], DataType::F32).is_err());
    }
    #[test]
    fn input_validation_and_float_roundtrip() {
        let data = [1.25, -4.0];
        let input = Input::f32("x", &[2], &data);
        input.validate().unwrap();
        let output = Output {
            info: TensorInfo {
                name: "y".into(),
                dtype: DataType::F32,
                is_input: false,
                shape: vec![2],
            },
            data: input.data.to_vec(),
        };
        assert_eq!(output.to_f32().unwrap(), data);
        assert!(Input::f32("x", &[3], &data).validate().is_err());
        assert!(Input::f32("x\0", &[2], &data).validate().is_err());
        assert!(DataType::try_from(9).is_err());
    }
}
