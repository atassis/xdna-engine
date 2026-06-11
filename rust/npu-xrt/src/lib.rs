//! Safe Rust over the C++ XRT shim (`shim/xrt_shim.{h,cpp}`). Mirrors the dispatch sequence our
//! pyxrt runners use: open device -> load xclbin (register + hw_context + kernel) -> alloc BOs ->
//! write/sync -> run -> sync back. See internal notes.
//!
//! The NPU is single-tenant: stop `flm-asr.service`/`voxd.service` before constructing a Device.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::rc::Rc;

#[repr(C)]
struct CDevice {
    _private: [u8; 0],
}
#[repr(C)]
struct CKernel {
    _private: [u8; 0],
}
#[repr(C)]
struct CBo {
    _private: [u8; 0],
}
#[repr(C)]
struct CRun {
    _private: [u8; 0],
}

extern "C" {
    fn shim_device_open(index: c_uint) -> *mut CDevice;
    fn shim_device_close(d: *mut CDevice);
    fn shim_kernel_load(d: *mut CDevice, path: *const c_char, name: *const c_char) -> *mut CKernel;
    fn shim_kernel_close(k: *mut CKernel);
    fn shim_kernel_group_id(k: *mut CKernel, arg: c_int) -> c_int;
    fn shim_bo_alloc(
        d: *mut CDevice,
        k: *mut CKernel,
        n: usize,
        flag: c_int,
        gid: c_int,
    ) -> *mut CBo;
    fn shim_bo_free(b: *mut CBo);
    fn shim_bo_write(b: *mut CBo, src: *const c_void, n: usize, off: usize) -> c_int;
    fn shim_bo_read(b: *mut CBo, dst: *mut c_void, n: usize, off: usize) -> c_int;
    fn shim_bo_sync_to_device(b: *mut CBo) -> c_int;
    fn shim_bo_sync_from_device(b: *mut CBo) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn shim_run_matmul8(
        k: *mut CKernel,
        opcode: c_uint,
        instr: *mut CBo,
        count: usize,
        a: *mut CBo,
        b: *mut CBo,
        c: *mut CBo,
        tmp: *mut CBo,
        trace: *mut CBo,
    ) -> c_int;
    fn shim_run_dwconv6(
        k: *mut CKernel,
        opcode: c_uint,
        instr: *mut CBo,
        count: usize,
        x: *mut CBo,
        w: *mut CBo,
        y: *mut CBo,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn shim_run_matmul8_start(
        k: *mut CKernel,
        opcode: c_uint,
        instr: *mut CBo,
        count: usize,
        a: *mut CBo,
        b: *mut CBo,
        c: *mut CBo,
        tmp: *mut CBo,
        trace: *mut CBo,
    ) -> *mut CRun;
    fn shim_run_wait(r: *mut CRun) -> c_int;
    fn shim_run_free(r: *mut CRun);
    fn shim_last_error() -> *const c_char;
}

/// BO allocation flags (matching the C shim / our pyxrt usage).
pub const FLAG_NORMAL: i32 = 0;
pub const FLAG_CACHEABLE: i32 = 1; // instruction buffer
pub const FLAG_HOST_ONLY: i32 = 2; // data buffers

fn last_error() -> String {
    unsafe { CStr::from_ptr(shim_last_error()).to_string_lossy().into_owned() }
}

pub type Result<T> = std::result::Result<T, String>;

/// A single shared XDNA2 device. NPU is single-tenant; create exactly one.
///
/// Kernels (each owning a hw_context) are cached by xclbin path: the NPU's 8 columns are a
/// limited hw_context budget, so loading the same xclbin twice returns the SAME shared kernel
/// (many engines with different weight BOs share one context). Mirrors `npu_asr/device.py`.
pub struct Device {
    ptr: *mut CDevice,
    kernels: RefCell<HashMap<String, Rc<Kernel>>>,
}

/// An xclbin loaded into a hw_context with its kernel resolved.
pub struct Kernel {
    ptr: *mut CKernel,
}

/// A device buffer object.
pub struct Bo {
    ptr: *mut CBo,
    nbytes: usize,
}

/// An in-flight (async) NPU dispatch. Created by [`Kernel::run_matmul8_start`], which submits the
/// command and returns immediately (the NPU runs while the host does other work). Call [`Run::wait`]
/// to block for completion. Dropping without waiting still frees the handle (XRT joins on destroy).
pub struct Run {
    ptr: *mut CRun,
}

impl Run {
    /// Block until this dispatch completes. Consumes the handle (a run is waited at most once).
    pub fn wait(self) -> Result<()> {
        // SAFETY: ptr is a live handle from run_matmul8_start; Drop frees it after this.
        let r = unsafe { shim_run_wait(self.ptr) };
        if r != 0 {
            Err(format!("run_wait: {}", last_error()))
        } else {
            Ok(())
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        unsafe { shim_run_free(self.ptr) }
    }
}

impl Device {
    pub fn open(index: u32) -> Result<Device> {
        let ptr = unsafe { shim_device_open(index) };
        if ptr.is_null() {
            Err(format!("device_open({index}): {}", last_error()))
        } else {
            Ok(Device {
                ptr,
                kernels: RefCell::new(HashMap::new()),
            })
        }
    }

    /// Load an xclbin and resolve its kernel. `name=None` uses the first kernel in the xclbin.
    /// Cached by (path, name): repeated loads of the same xclbin return the SAME shared kernel
    /// (one hw_context), so many engines coexist within the 8-column budget.
    pub fn load_kernel(&self, xclbin_path: &str, name: Option<&str>) -> Result<Rc<Kernel>> {
        let key = format!("{xclbin_path}\u{0}{}", name.unwrap_or(""));
        if let Some(k) = self.kernels.borrow().get(&key) {
            return Ok(k.clone());
        }
        let cpath = CString::new(xclbin_path).map_err(|e| e.to_string())?;
        let cname = match name {
            Some(s) => Some(CString::new(s).map_err(|e| e.to_string())?),
            None => None,
        };
        let name_ptr = cname.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let ptr = unsafe { shim_kernel_load(self.ptr, cpath.as_ptr(), name_ptr) };
        if ptr.is_null() {
            return Err(format!("load_kernel({xclbin_path}): {}", last_error()));
        }
        let k = Rc::new(Kernel { ptr });
        self.kernels.borrow_mut().insert(key, k.clone());
        Ok(k)
    }

    pub fn alloc_bo(&self, kernel: &Kernel, nbytes: usize, flag: i32, group_id: i32) -> Result<Bo> {
        let ptr = unsafe { shim_bo_alloc(self.ptr, kernel.ptr, nbytes, flag, group_id) };
        if ptr.is_null() {
            Err(format!("alloc_bo({nbytes}, flag={flag}, gid={group_id}): {}", last_error()))
        } else {
            Ok(Bo { ptr, nbytes })
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { shim_device_close(self.ptr) }
    }
}

impl Kernel {
    /// Memory bank for kernel argument `arg` (used to allocate the matching BO).
    pub fn group_id(&self, arg: i32) -> Result<i32> {
        let g = unsafe { shim_kernel_group_id(self.ptr, arg) };
        if g < 0 {
            Err(format!("group_id({arg}): {}", last_error()))
        } else {
            Ok(g)
        }
    }

    /// Dispatch the whole_array/matmul host ABI: (opcode, instr, count, A, B, C, tmp, trace).
    #[allow(clippy::too_many_arguments)]
    pub fn run_matmul8(
        &self,
        opcode: u32,
        instr: &Bo,
        count: usize,
        a: &Bo,
        b: &Bo,
        c: &Bo,
        tmp: &Bo,
        trace: &Bo,
    ) -> Result<()> {
        let r = unsafe {
            shim_run_matmul8(
                self.ptr, opcode, instr.ptr, count, a.ptr, b.ptr, c.ptr, tmp.ptr, trace.ptr,
            )
        };
        if r != 0 {
            Err(format!("run_matmul8: {}", last_error()))
        } else {
            Ok(())
        }
    }

    /// Async variant of [`run_matmul8`]: submit the dispatch and return immediately with a [`Run`]
    /// handle (the NPU executes while the host continues). Call [`Run::wait`] before consuming `c`.
    /// All BOs (and the kernel) must outlive the returned `Run`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_matmul8_start(
        &self,
        opcode: u32,
        instr: &Bo,
        count: usize,
        a: &Bo,
        b: &Bo,
        c: &Bo,
        tmp: &Bo,
        trace: &Bo,
    ) -> Result<Run> {
        let ptr = unsafe {
            shim_run_matmul8_start(
                self.ptr, opcode, instr.ptr, count, a.ptr, b.ptr, c.ptr, tmp.ptr, trace.ptr,
            )
        };
        if ptr.is_null() {
            Err(format!("run_matmul8_start: {}", last_error()))
        } else {
            Ok(Run { ptr })
        }
    }

    /// Dispatch the depthwise-conv1d host ABI: (opcode, instr, count, X, W, Y).
    pub fn run_dwconv6(
        &self,
        opcode: u32,
        instr: &Bo,
        count: usize,
        x: &Bo,
        w: &Bo,
        y: &Bo,
    ) -> Result<()> {
        let r = unsafe {
            shim_run_dwconv6(self.ptr, opcode, instr.ptr, count, x.ptr, w.ptr, y.ptr)
        };
        if r != 0 {
            Err(format!("run_dwconv6: {}", last_error()))
        } else {
            Ok(())
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { shim_kernel_close(self.ptr) }
    }
}

impl Bo {
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        let r = unsafe { shim_bo_write(self.ptr, bytes.as_ptr() as *const c_void, bytes.len(), 0) };
        if r != 0 {
            Err(format!("bo_write: {}", last_error()))
        } else {
            Ok(())
        }
    }

    pub fn read_bytes(&self, out: &mut [u8]) -> Result<()> {
        let r = unsafe { shim_bo_read(self.ptr, out.as_mut_ptr() as *mut c_void, out.len(), 0) };
        if r != 0 {
            Err(format!("bo_read: {}", last_error()))
        } else {
            Ok(())
        }
    }

    pub fn sync_to_device(&self) -> Result<()> {
        let r = unsafe { shim_bo_sync_to_device(self.ptr) };
        if r != 0 {
            Err(format!("sync_to_device: {}", last_error()))
        } else {
            Ok(())
        }
    }

    pub fn sync_from_device(&self) -> Result<()> {
        let r = unsafe { shim_bo_sync_from_device(self.ptr) };
        if r != 0 {
            Err(format!("sync_from_device: {}", last_error()))
        } else {
            Ok(())
        }
    }
}

impl Drop for Bo {
    fn drop(&mut self) {
        unsafe { shim_bo_free(self.ptr) }
    }
}

// --- bf16 helpers (IEEE bfloat16, round-to-nearest-even — matches ml_dtypes/numpy) ---

/// Round an f32 to bf16, returned as raw u16 bits.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    // NaN: keep it a (quiet) NaN.
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Expand bf16 bits back to f32.
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
