//! Safe Rust over the onnxruntime C-shim (`shim/onnx_shim.{h,cpp}`). Runs ONNX graphs via the
//! system libonnxruntime — used for the GigaAM-v3 mel preprocessor + RNNT decoder/joint, since the
//! `ort` crate won't build against onnxruntime 1.26 here. See docs/14.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::rc::Rc;

#[repr(C)]
struct CEnv {
    _p: [u8; 0],
}
#[repr(C)]
struct CSession {
    _p: [u8; 0],
}
#[repr(C)]
struct CRun {
    _p: [u8; 0],
}

extern "C" {
    fn sort_env_create() -> *mut CEnv;
    fn sort_env_free(e: *mut CEnv);
    fn sort_session_create(e: *mut CEnv, path: *const c_char) -> *mut CSession;
    fn sort_session_free(s: *mut CSession);
    fn sort_run(
        s: *mut CSession,
        n_in: c_int,
        in_names: *const *const c_char,
        in_data: *const *const c_void,
        in_dims: *const *const i64,
        in_ndims: *const c_int,
        in_dtypes: *const c_int,
        n_out: c_int,
        out_names: *const *const c_char,
    ) -> *mut CRun;
    fn sort_run_ndims(r: *mut CRun, i: c_int) -> c_int;
    fn sort_run_dims(r: *mut CRun, i: c_int, dims: *mut i64);
    fn sort_run_data(r: *mut CRun, i: c_int) -> *const c_void;
    fn sort_run_free(r: *mut CRun);
    fn sort_last_error() -> *const c_char;
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(sort_last_error()).to_string_lossy().into_owned() }
}

pub type Result<T> = std::result::Result<T, String>;

/// An onnxruntime environment (create one, share across sessions).
pub struct Env {
    ptr: *mut CEnv,
}
impl Env {
    pub fn new() -> Result<Rc<Env>> {
        let ptr = unsafe { sort_env_create() };
        if ptr.is_null() {
            Err(format!("env_create: {}", last_error()))
        } else {
            Ok(Rc::new(Env { ptr }))
        }
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        unsafe { sort_env_free(self.ptr) }
    }
}

/// A loaded ONNX model session.
pub struct Session {
    ptr: *mut CSession,
    _env: Rc<Env>,
}

/// A typed input/output tensor (borrowed data + shape).
pub enum Tensor<'a> {
    F32(&'a [f32], Vec<i64>),
    I64(&'a [i64], Vec<i64>),
}
impl Tensor<'_> {
    fn dtype(&self) -> c_int {
        match self {
            Tensor::F32(..) => 0,
            Tensor::I64(..) => 1,
        }
    }
    fn dims(&self) -> &[i64] {
        match self {
            Tensor::F32(_, d) | Tensor::I64(_, d) => d,
        }
    }
    fn data_ptr(&self) -> *const c_void {
        match self {
            Tensor::F32(d, _) => d.as_ptr() as *const c_void,
            Tensor::I64(d, _) => d.as_ptr() as *const c_void,
        }
    }
}

impl Session {
    pub fn load(env: &Rc<Env>, model_path: &str) -> Result<Session> {
        let c = CString::new(model_path).map_err(|e| e.to_string())?;
        let ptr = unsafe { sort_session_create(env.ptr, c.as_ptr()) };
        if ptr.is_null() {
            Err(format!("load {model_path}: {}", last_error()))
        } else {
            Ok(Session {
                ptr,
                _env: env.clone(),
            })
        }
    }

    /// Run with named inputs; returns the named outputs.
    pub fn run(&self, inputs: &[(&str, Tensor)], out_names: &[&str]) -> Result<Outputs> {
        let in_cnames: Vec<CString> = inputs
            .iter()
            .map(|(n, _)| CString::new(*n).unwrap())
            .collect();
        let in_name_ptrs: Vec<*const c_char> = in_cnames.iter().map(|c| c.as_ptr()).collect();
        let in_data: Vec<*const c_void> = inputs.iter().map(|(_, t)| t.data_ptr()).collect();
        let in_dims_vecs: Vec<&[i64]> = inputs.iter().map(|(_, t)| t.dims()).collect();
        let in_dims_ptrs: Vec<*const i64> = in_dims_vecs.iter().map(|d| d.as_ptr()).collect();
        let in_ndims: Vec<c_int> = in_dims_vecs.iter().map(|d| d.len() as c_int).collect();
        let in_dtypes: Vec<c_int> = inputs.iter().map(|(_, t)| t.dtype()).collect();

        let out_cnames: Vec<CString> = out_names.iter().map(|n| CString::new(*n).unwrap()).collect();
        let out_name_ptrs: Vec<*const c_char> = out_cnames.iter().map(|c| c.as_ptr()).collect();

        let r = unsafe {
            sort_run(
                self.ptr,
                inputs.len() as c_int,
                in_name_ptrs.as_ptr(),
                in_data.as_ptr(),
                in_dims_ptrs.as_ptr(),
                in_ndims.as_ptr(),
                in_dtypes.as_ptr(),
                out_names.len() as c_int,
                out_name_ptrs.as_ptr(),
            )
        };
        if r.is_null() {
            Err(format!("run: {}", last_error()))
        } else {
            Ok(Outputs {
                ptr: r,
                n: out_names.len(),
            })
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        unsafe { sort_session_free(self.ptr) }
    }
}

/// Owns the output tensors until dropped.
pub struct Outputs {
    ptr: *mut CRun,
    n: usize,
}
impl Outputs {
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn shape(&self, i: usize) -> Vec<i64> {
        let nd = unsafe { sort_run_ndims(self.ptr, i as c_int) };
        if nd < 0 {
            return vec![];
        }
        let mut dims = vec![0i64; nd as usize];
        unsafe { sort_run_dims(self.ptr, i as c_int, dims.as_mut_ptr()) };
        dims
    }
    fn numel(&self, i: usize) -> usize {
        self.shape(i).iter().product::<i64>() as usize
    }
    /// f32 output as a slice (valid until this `Outputs` is dropped).
    pub fn f32(&self, i: usize) -> &[f32] {
        let n = self.numel(i);
        let p = unsafe { sort_run_data(self.ptr, i as c_int) } as *const f32;
        unsafe { std::slice::from_raw_parts(p, n) }
    }
    /// i64 output as a slice.
    pub fn i64(&self, i: usize) -> &[i64] {
        let n = self.numel(i);
        let p = unsafe { sort_run_data(self.ptr, i as c_int) } as *const i64;
        unsafe { std::slice::from_raw_parts(p, n) }
    }
}
impl Drop for Outputs {
    fn drop(&mut self) {
        unsafe { sort_run_free(self.ptr) }
    }
}
