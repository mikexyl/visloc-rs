use super::*;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

#[repr(C)]
struct Info {
    name: *const c_char,
    dtype: i32,
    input: i32,
    rank: i32,
    dims: [i64; 8],
}
impl Default for Info {
    fn default() -> Self {
        Self {
            name: std::ptr::null(),
            dtype: 0,
            input: 0,
            rank: 0,
            dims: [0; 8],
        }
    }
}
#[repr(C)]
struct NativeInput {
    name: *const c_char,
    data: *const u8,
    bytes: usize,
    dtype: i32,
    rank: i32,
    dims: [i64; 8],
}
extern "C" {
    fn vt_error() -> *const c_char;
    fn vt_version() -> i32;
    fn vt_open(plan: *const u8, bytes: usize, device: i32) -> *mut c_void;
    fn vt_close(session: *mut c_void);
    fn vt_count(session: *mut c_void) -> i32;
    fn vt_info(session: *mut c_void, index: i32, info: *mut Info) -> i32;
    fn vt_run(session: *mut c_void, inputs: *const NativeInput, count: usize, profile: i32) -> i32;
    fn vt_output(
        session: *mut c_void,
        index: i32,
        info: *mut Info,
        data: *mut *const u8,
        bytes: *mut usize,
    ) -> i32;
}
fn last_error() -> Error {
    // SAFETY: bridge returns a thread-local NUL-terminated diagnostic.
    Error(
        unsafe { CStr::from_ptr(vt_error()) }
            .to_string_lossy()
            .into_owned(),
    )
}
fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(last_error())
    }
}
fn convert(info: &Info) -> Result<TensorInfo> {
    if info.name.is_null() || !(0..=8).contains(&info.rank) {
        return Err(Error("Invalid native tensor metadata".into()));
    }
    Ok(TensorInfo {
        // SAFETY: metadata names belong to the still-live native engine.
        name: unsafe { CStr::from_ptr(info.name) }
            .to_string_lossy()
            .into_owned(),
        dtype: DataType::try_from(info.dtype)?,
        is_input: info.input != 0,
        shape: info.dims[..info.rank as usize].to_vec(),
    })
}
/// Build-time TensorRT version encoded as major * 10000 + minor * 100 + patch.
pub fn header_version() -> i32 {
    unsafe { vt_version() }
}

/// One engine, execution context, CUDA stream, and reusable device buffers.
///
/// Sessions are deliberately neither Send nor Sync. `run` requires exclusive
/// access and completes all GPU work before returning owned CPU outputs.
/// Only deserialize trusted plans built for a compatible SDK and GPU.
pub struct Session {
    raw: NonNull<c_void>,
    tensors: Vec<TensorInfo>,
    _thread_affinity: PhantomData<Rc<()>>,
}
impl Session {
    pub fn from_file(path: impl AsRef<Path>, device: i32) -> Result<Self> {
        let plan = std::fs::read(path).map_err(|e| Error(e.to_string()))?;
        Self::from_bytes(&plan, device)
    }
    pub fn from_bytes(plan: &[u8], device: i32) -> Result<Self> {
        if plan.is_empty() {
            return Err(Error("Empty engine plan".into()));
        }
        if device < 0 {
            return Err(Error("Device ordinal must be nonnegative".into()));
        }
        // SAFETY: bytes stay live for deserialization; ownership of the result transfers here.
        let raw = NonNull::new(unsafe { vt_open(plan.as_ptr(), plan.len(), device) })
            .ok_or_else(last_error)?;
        let mut session = Self {
            raw,
            tensors: Vec::new(),
            _thread_affinity: PhantomData,
        };
        for i in 0..unsafe { vt_count(raw.as_ptr()) } {
            let mut info = Info::default();
            check(unsafe { vt_info(raw.as_ptr(), i, &mut info) })?;
            session.tensors.push(convert(&info)?);
        }
        Ok(session)
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Run all named inputs with the selected optimization profile (usually 0).
    /// Supports static/dynamic execution shapes and LINEAR device I/O.
    /// Shape-tensor I/O, packed types, and data-dependent output sizes return errors.
    pub fn run(&mut self, inputs: &[Input<'_>], profile: i32) -> Result<Vec<Output>> {
        let mut names = Vec::with_capacity(inputs.len());
        for input in inputs {
            input.validate()?;
            names.push(CString::new(input.name).map_err(|e| Error(e.to_string()))?);
        }
        let native: Vec<_> = inputs
            .iter()
            .zip(&names)
            .map(|(input, name)| {
                let mut dims = [0; 8];
                dims[..input.shape.len()].copy_from_slice(input.shape);
                NativeInput {
                    name: name.as_ptr(),
                    data: input.data.as_ptr(),
                    bytes: input.data.len(),
                    dtype: input.dtype as i32,
                    rank: input.shape.len() as i32,
                    dims,
                }
            })
            .collect();
        // SAFETY: all names/data remain borrowed until vt_run drains its stream,
        // including failure paths. Exclusive access prevents context races.
        check(unsafe { vt_run(self.raw.as_ptr(), native.as_ptr(), native.len(), profile) })?;
        let mut outputs = Vec::new();
        for (i, tensor) in self.tensors.iter().enumerate() {
            if tensor.is_input {
                continue;
            }
            let mut info = Info::default();
            let mut data = std::ptr::null();
            let mut bytes = 0;
            check(unsafe {
                vt_output(
                    self.raw.as_ptr(),
                    i as i32,
                    &mut info,
                    &mut data,
                    &mut bytes,
                )
            })?;
            // SAFETY: native output is initialized, synchronized, and live until
            // the next run or drop. Copy it before releasing this mutable borrow.
            let data = if bytes == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(data, bytes) }.to_vec()
            };
            outputs.push(Output {
                info: convert(&info)?,
                data,
            });
        }
        Ok(outputs)
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: exactly one owner, no work can escape a run call.
        unsafe { vt_close(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_abi_and_rejections() {
        assert!((100000..120000).contains(&header_version()));
        assert!(Session::from_bytes(&[], 0).is_err());
        assert!(Session::from_bytes(&[0], -1).is_err());
        // Calls the actual C ABI without requiring a GPU; empty plans fail before CUDA.
        assert!(unsafe { vt_open(std::ptr::null(), 0, 0) }.is_null());
        assert!(last_error().0.contains("Empty engine"));
    }
}
