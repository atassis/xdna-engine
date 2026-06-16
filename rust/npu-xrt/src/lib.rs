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
#[repr(C)]
struct CElfKernel {
    _private: [u8; 0],
}
#[repr(C)]
struct CElfCtx {
    _private: [u8; 0],
}
#[repr(C)]
struct CElfKernel2 {
    _private: [u8; 0],
}
#[repr(C)]
struct CElfResident {
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
    fn shim_elf_kernel_load(
        d: *mut CDevice,
        elf_bytes: *const c_void,
        nbytes: usize,
        name: *const c_char,
    ) -> *mut CElfKernel;
    fn shim_elf_kernel_close(k: *mut CElfKernel);
    fn shim_run_elf(k: *mut CElfKernel, bos: *const *mut CBo, n_bos: usize) -> c_int;
    fn shim_run_elf_start(k: *mut CElfKernel, bos: *const *mut CBo, n_bos: usize) -> *mut CRun;
    fn shim_elf_ctx_open(d: *mut CDevice, base_elf: *const c_void, nbytes: usize) -> *mut CElfCtx;
    fn shim_elf_ctx_close(c: *mut CElfCtx);
    fn shim_elf_kernel_rebind(
        c: *mut CElfCtx,
        elf_bytes: *const c_void,
        nbytes: usize,
        name: *const c_char,
    ) -> *mut CElfKernel2;
    fn shim_elf_kernel2_close(k: *mut CElfKernel2);
    fn shim_run_elf2(k: *mut CElfKernel2, bos: *const *mut CBo, n_bos: usize) -> c_int;
    fn shim_elf_resident_open(
        d: *mut CDevice,
        elf_bytes: *const c_void,
        nbytes: usize,
        name: *const c_char,
    ) -> *mut CElfResident;
    fn shim_elf_resident_close(r: *mut CElfResident);
    fn shim_elf_resident_scratchpad_size(r: *mut CElfResident) -> usize;
    fn shim_elf_resident_bind(r: *mut CElfResident, bos: *const *mut CBo, n_bos: usize) -> c_int;
    fn shim_elf_resident_write(
        r: *mut CElfResident,
        offset: usize,
        data: *const c_void,
        len: usize,
    ) -> c_int;
    fn shim_elf_resident_dispatch(r: *mut CElfResident) -> c_int;
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

/// A fused full-ELF kernel: the IRON `FusedMLIROperator` dispatch path. Built from raw ELF bytes
/// (`xrt::elf` → `hw_context(device, elf)` → `ext::kernel(ctx, "main:sequence")`); no xclbin, no
/// insts BO. Dispatched with N positional BO args via [`ElfKernel::run_elf`] (the fused-arena ABI is
/// 3 BOs: input/output/scratch). The ELF bytes are copied in, so the source buffer may be patched.
pub struct ElfKernel {
    ptr: *mut CElfKernel,
}

/// A resident hw_context built ONCE from a base fused ELF. Per-token patched ELFs are bound onto it
/// via [`ElfCtx::rebind`] (rebuilding only a lightweight `xrt::module` + `ext::kernel`), hoisting the
/// ~20 ms/token NPU partition-config out of the per-token loop. The context must outlive every
/// [`ElfKernel2`] rebound onto it. See `shim_elf_ctx_open`.
pub struct ElfCtx {
    ptr: *mut CElfCtx,
}

/// A per-token kernel bound onto a persistent [`ElfCtx`] (borrows its hw_context; owns its own
/// patched ELF + module + kernel). Dispatched like [`ElfKernel`] but via `shim_run_elf2`.
pub struct ElfKernel2 {
    ptr: *mut CElfKernel2,
}

/// A resident runner for a CONSTANT full ELF built with `aiex.scratchpad_parameter` (Option C). The
/// ELF + hw_context + kernel + run are created ONCE; the arena BOs are bound once; per dispatch the
/// host writes the per-token parameter word(s) into the ctrl scratchpad and dispatches — no ELF
/// re-registration. Construct via [`Device::open_elf_resident`].
pub struct ElfResident {
    ptr: *mut CElfResident,
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

    /// Allocate a BO not tied to a kernel's arg bank — for the fused-ELF arenas, which IRON
    /// allocates as `xrt::bo(device, nbytes, host_only, group_id=0)` (the shim ignores the kernel).
    pub fn alloc_bo_raw(&self, nbytes: usize, flag: i32, group_id: i32) -> Result<Bo> {
        let ptr = unsafe { shim_bo_alloc(self.ptr, std::ptr::null_mut(), nbytes, flag, group_id) };
        if ptr.is_null() {
            Err(format!("alloc_bo_raw({nbytes}, flag={flag}, gid={group_id}): {}", last_error()))
        } else {
            Ok(Bo { ptr, nbytes })
        }
    }

    /// Load a fused full-ELF kernel from raw ELF bytes. `name=None` uses `"main:sequence"` (IRON's
    /// default `device:sequence`). Unlike [`load_kernel`], this is NOT cached — each fused ELF (and
    /// each per-token patched variant) is its own hw_context. The bytes are copied into XRT.
    pub fn load_elf_kernel(&self, elf_bytes: &[u8], name: Option<&str>) -> Result<ElfKernel> {
        let cname = match name {
            Some(s) => Some(CString::new(s).map_err(|e| e.to_string())?),
            None => None,
        };
        let name_ptr = cname.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let ptr = unsafe {
            shim_elf_kernel_load(
                self.ptr,
                elf_bytes.as_ptr() as *const c_void,
                elf_bytes.len(),
                name_ptr,
            )
        };
        if ptr.is_null() {
            Err(format!("load_elf_kernel({} bytes): {}", elf_bytes.len(), last_error()))
        } else {
            Ok(ElfKernel { ptr })
        }
    }

    /// Build a persistent [`ElfCtx`] (hw_context) ONCE from a base fused ELF. Per-token patched ELFs
    /// are then bound via [`ElfCtx::rebind`] without re-running the partition config. `base_elf` is
    /// any same-shape ELF (e.g. the unpatched base); the bytes are copied into XRT.
    pub fn open_elf_ctx(&self, base_elf: &[u8]) -> Result<ElfCtx> {
        let ptr =
            unsafe { shim_elf_ctx_open(self.ptr, base_elf.as_ptr() as *const c_void, base_elf.len()) };
        if ptr.is_null() {
            Err(format!("open_elf_ctx({} bytes): {}", base_elf.len(), last_error()))
        } else {
            Ok(ElfCtx { ptr })
        }
    }

    /// Open a resident runner for a CONSTANT scratchpad-parameter ELF (Option C). Registers the ELF
    /// ONCE. Returns `Err` if the ELF has no ctrl scratchpad (i.e. not a scratchpad-parameter build —
    /// caller should fall back to the patch path). `name=None` uses `"main:sequence"`.
    pub fn open_elf_resident(&self, elf_bytes: &[u8], name: Option<&str>) -> Result<ElfResident> {
        let cname = match name {
            Some(s) => Some(CString::new(s).map_err(|e| e.to_string())?),
            None => None,
        };
        let name_ptr = cname.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let ptr = unsafe {
            shim_elf_resident_open(self.ptr, elf_bytes.as_ptr() as *const c_void, elf_bytes.len(), name_ptr)
        };
        if ptr.is_null() {
            Err(format!("open_elf_resident({} bytes): {}", elf_bytes.len(), last_error()))
        } else {
            Ok(ElfResident { ptr })
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

impl ElfKernel {
    /// Dispatch the fused ELF with `bos` mapped positionally to args 0..N (`run.set_arg(i, bo)`),
    /// then start + wait. The fused-arena ABI passes 3 BOs: `[input, output, scratch]`.
    pub fn run_elf(&self, bos: &[&Bo]) -> Result<()> {
        let ptrs: Vec<*mut CBo> = bos.iter().map(|b| b.ptr).collect();
        let r = unsafe { shim_run_elf(self.ptr, ptrs.as_ptr(), ptrs.len()) };
        if r != 0 {
            Err(format!("run_elf({} bos): {}", ptrs.len(), last_error()))
        } else {
            Ok(())
        }
    }

    /// Async variant of [`run_elf`] (the PIPE lever): submit the dispatch over `bos` and return
    /// immediately with a [`Run`] handle. The NPU executes while the host registers the next token's
    /// position-only patched ELF; call [`Run::wait`] before reading the output arena. This `ElfKernel`
    /// and all BOs must outlive the returned `Run`.
    pub fn start_elf(&self, bos: &[&Bo]) -> Result<Run> {
        let ptrs: Vec<*mut CBo> = bos.iter().map(|b| b.ptr).collect();
        let ptr = unsafe { shim_run_elf_start(self.ptr, ptrs.as_ptr(), ptrs.len()) };
        if ptr.is_null() {
            Err(format!("start_elf({} bos): {}", ptrs.len(), last_error()))
        } else {
            Ok(Run { ptr })
        }
    }
}

impl Drop for ElfKernel {
    fn drop(&mut self) {
        unsafe { shim_elf_kernel_close(self.ptr) }
    }
}

impl ElfCtx {
    /// Bind a patched ELF onto this resident hw_context, rebuilding only the module + kernel (no
    /// partition re-config). `name=None` uses `"main:sequence"`. The returned [`ElfKernel2`] borrows
    /// this context — keep the `ElfCtx` alive at least as long as any kernel rebound from it.
    pub fn rebind(&self, elf_bytes: &[u8], name: Option<&str>) -> Result<ElfKernel2> {
        let cname = match name {
            Some(s) => Some(CString::new(s).map_err(|e| e.to_string())?),
            None => None,
        };
        let name_ptr = cname.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let ptr = unsafe {
            shim_elf_kernel_rebind(
                self.ptr,
                elf_bytes.as_ptr() as *const c_void,
                elf_bytes.len(),
                name_ptr,
            )
        };
        if ptr.is_null() {
            Err(format!("rebind({} bytes): {}", elf_bytes.len(), last_error()))
        } else {
            Ok(ElfKernel2 { ptr })
        }
    }
}

impl Drop for ElfCtx {
    fn drop(&mut self) {
        unsafe { shim_elf_ctx_close(self.ptr) }
    }
}

impl ElfKernel2 {
    /// Dispatch over `bos` mapped positionally to args 0..N, start + wait. Same ABI as
    /// [`ElfKernel::run_elf`] but on a kernel bound to a persistent [`ElfCtx`].
    pub fn run_elf(&self, bos: &[&Bo]) -> Result<()> {
        let ptrs: Vec<*mut CBo> = bos.iter().map(|b| b.ptr).collect();
        let r = unsafe { shim_run_elf2(self.ptr, ptrs.as_ptr(), ptrs.len()) };
        if r != 0 {
            Err(format!("run_elf2({} bos): {}", ptrs.len(), last_error()))
        } else {
            Ok(())
        }
    }
}

impl Drop for ElfKernel2 {
    fn drop(&mut self) {
        unsafe { shim_elf_kernel2_close(self.ptr) }
    }
}

impl ElfResident {
    /// Size of the ctrl scratchpad (bytes). >0 means the ELF carries scratchpad parameters.
    pub fn scratchpad_size(&self) -> usize {
        unsafe { shim_elf_resident_scratchpad_size(self.ptr) }
    }

    /// Bind the arena BOs to run args 0..N once (reused every dispatch).
    pub fn bind(&self, bos: &[&Bo]) -> Result<()> {
        let ptrs: Vec<*mut CBo> = bos.iter().map(|b| b.ptr).collect();
        let r = unsafe { shim_elf_resident_bind(self.ptr, ptrs.as_ptr(), ptrs.len()) };
        if r != 0 {
            Err(format!("resident bind({} bos): {}", ptrs.len(), last_error()))
        } else {
            Ok(())
        }
    }

    /// Write `bytes` at byte `offset` into the host-mapped ctrl scratchpad (no device sync yet).
    pub fn write_scratchpad(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        let r = unsafe {
            shim_elf_resident_write(self.ptr, offset, bytes.as_ptr() as *const c_void, bytes.len())
        };
        if r != 0 {
            Err(format!("resident write_scratchpad@{offset}: {}", last_error()))
        } else {
            Ok(())
        }
    }

    /// Sync the scratchpad to device, start the bound run, wait for completion.
    pub fn dispatch(&self) -> Result<()> {
        let r = unsafe { shim_elf_resident_dispatch(self.ptr) };
        if r != 0 {
            Err(format!("resident dispatch: {}", last_error()))
        } else {
            Ok(())
        }
    }
}

impl Drop for ElfResident {
    fn drop(&mut self) {
        unsafe { shim_elf_resident_close(self.ptr) }
    }
}

/// Which of the three fused-ELF arenas a buffer lives in (IRON's input/output/scratch split:
/// `input` = args synced every dispatch, `output` = results synced back, `scratch` = resident
/// weights + KV caches + launch→launch intermediates, never re-synced after the first load).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arena {
    Input,
    Output,
    Scratch,
}

/// The three resident arena BOs for a fused full ELF (IRON `FusedFullELFCallable`). Buffers are
/// addressed by byte offset within their arena (the layout comes from the generator's `meta.json`).
/// Dispatch passes exactly `[input, output, scratch]` positionally to the ELF.
pub struct FusedArena {
    input: Bo,
    output: Bo,
    scratch: Bo,
}

impl FusedArena {
    /// Allocate the three arenas (host_only, group_id 0 — IRON's XRTTensor convention). Sizes come
    /// from the fused operator's `buffer_sizes`. A zero-size arena is rounded up to 2 bytes (XRT
    /// rejects 0-byte BOs; IRON does the same `max(size, itemsize)`).
    pub fn new(
        dev: &Device,
        input_size: usize,
        output_size: usize,
        scratch_size: usize,
    ) -> Result<FusedArena> {
        Ok(FusedArena {
            input: dev.alloc_bo_raw(input_size.max(2), FLAG_HOST_ONLY, 0)?,
            output: dev.alloc_bo_raw(output_size.max(2), FLAG_HOST_ONLY, 0)?,
            scratch: dev.alloc_bo_raw(scratch_size.max(2), FLAG_HOST_ONLY, 0)?,
        })
    }

    fn bo(&self, a: Arena) -> &Bo {
        match a {
            Arena::Input => &self.input,
            Arena::Output => &self.output,
            Arena::Scratch => &self.scratch,
        }
    }

    /// Write `bytes` into `arena` at `offset` (host-side; call [`FusedArena::sync_to_device`] before dispatch).
    pub fn write_at(&self, arena: Arena, offset: usize, bytes: &[u8]) -> Result<()> {
        let b = self.bo(arena);
        let r =
            unsafe { shim_bo_write(b.ptr, bytes.as_ptr() as *const c_void, bytes.len(), offset) };
        if r != 0 {
            Err(format!("arena write {arena:?}@{offset}: {}", last_error()))
        } else {
            Ok(())
        }
    }

    /// Read `out.len()` bytes from `arena` at `offset` (call [`FusedArena::sync_from_device`] after dispatch).
    pub fn read_at(&self, arena: Arena, offset: usize, out: &mut [u8]) -> Result<()> {
        let b = self.bo(arena);
        let r = unsafe { shim_bo_read(b.ptr, out.as_mut_ptr() as *mut c_void, out.len(), offset) };
        if r != 0 {
            Err(format!("arena read {arena:?}@{offset}: {}", last_error()))
        } else {
            Ok(())
        }
    }

    /// Sync input + scratch (the host-written arenas) to the device. Scratch holds resident weights,
    /// so after the first sync, per-token dispatches need only re-sync input (use [`FusedArena::sync_input`]).
    pub fn sync_to_device(&self) -> Result<()> {
        self.input.sync_to_device()?;
        self.scratch.sync_to_device()
    }

    /// Sync only the input arena to device (per-token fast path; scratch already resident).
    pub fn sync_input(&self) -> Result<()> {
        self.input.sync_to_device()
    }

    /// Sync the output arena back from the device.
    pub fn sync_from_device(&self) -> Result<()> {
        self.output.sync_from_device()
    }

    /// Dispatch the fused ELF over `[input, output, scratch]`.
    pub fn dispatch(&self, kern: &ElfKernel) -> Result<()> {
        kern.run_elf(&[&self.input, &self.output, &self.scratch])
    }

    /// Async dispatch over `[input, output, scratch]` (the PIPE lever): returns a [`Run`] handle
    /// immediately so the host can register the next token's ELF while the NPU runs. `kern` and this
    /// arena must outlive the returned `Run`; call [`Run::wait`] before [`FusedArena::sync_from_device`].
    pub fn dispatch_start(&self, kern: &ElfKernel) -> Result<Run> {
        kern.start_elf(&[&self.input, &self.output, &self.scratch])
    }

    /// Dispatch over `[input, output, scratch]` with a persistent-context kernel ([`ElfKernel2`]).
    pub fn dispatch2(&self, kern: &ElfKernel2) -> Result<()> {
        kern.run_elf(&[&self.input, &self.output, &self.scratch])
    }

    /// Bind the three arenas to a resident runner once ([`ElfResident::bind`]). After this, a token
    /// loop is: write x into input + [`FusedArena::sync_input`], write scratchpad param(s) via the
    /// resident, then [`ElfResident::dispatch`] (it syncs the scratchpad + runs the bound args).
    pub fn bind_resident(&self, r: &ElfResident) -> Result<()> {
        r.bind(&[&self.input, &self.output, &self.scratch])
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

/// Pack a contiguous `f32` slice into bf16 bits (`u16`), round-to-nearest-even.
/// **Byte-identical** to applying [`f32_to_bf16_bits`] element-wise, for every input
/// (finite, denormal, inf, NaN) — the SIMD path replicates the exact integer bias formula,
/// so it does not depend on any hardware bf16-convert rounding. Uses AVX-512F (present on
/// Zen5/Krackan, enabled by `target-cpu=native`) when detected at runtime; scalar otherwise.
/// `dst` is filled for `min(src.len(), dst.len())` elements.
#[inline]
pub fn pack_f32_to_bf16(src: &[f32], dst: &mut [u16]) {
    let n = src.len().min(dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: gated on runtime AVX-512F detection; ptrs valid for `n` elems.
            unsafe {
                pack_f32_to_bf16_avx512(src.as_ptr(), dst.as_mut_ptr(), n);
            }
            return;
        }
    }
    for i in 0..n {
        dst[i] = f32_to_bf16_bits(src[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_f32_to_bf16_avx512(src: *const f32, dst: *mut u16, n: usize) {
    use std::arch::x86_64::*;
    let c_7fff = _mm512_set1_epi32(0x0000_7fff);
    let c_one = _mm512_set1_epi32(1);
    let c_absmask = _mm512_set1_epi32(0x7fff_ffff);
    let c_inf = _mm512_set1_epi32(0x7f80_0000);
    let c_qnan = _mm512_set1_epi32(0x0000_0040);
    let mut i = 0usize;
    while i + 16 <= n {
        let bits = _mm512_loadu_si512(src.add(i) as *const __m512i);
        // finite RNE: (bits + (0x7fff + ((bits>>16)&1))) >> 16
        let lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), c_one);
        let bias = _mm512_add_epi32(c_7fff, lsb);
        let rounded = _mm512_srli_epi32(_mm512_add_epi32(bits, bias), 16);
        // NaN: (bits>>16) | 0x40   when (bits & 0x7fffffff) > 0x7f800000
        let nan_res = _mm512_or_si512(_mm512_srli_epi32(bits, 16), c_qnan);
        let absv = _mm512_and_si512(bits, c_absmask);
        let nan_mask = _mm512_cmpgt_epi32_mask(absv, c_inf);
        let res32 = _mm512_mask_blend_epi32(nan_mask, rounded, nan_res);
        let res16 = _mm512_cvtepi32_epi16(res32); // 16x i32 low-16 -> 16x i16
        _mm256_storeu_si256(dst.add(i) as *mut __m256i, res16);
        i += 16;
    }
    while i < n {
        *dst.add(i) = f32_to_bf16_bits(*src.add(i));
        i += 1;
    }
}

/// Expand a contiguous bf16-bits (`u16`) slice into `f32` (`<<16`), the inverse of the pack.
/// Byte-identical to [`bf16_bits_to_f32`] element-wise. Uses AVX-512F when detected; scalar otherwise.
#[inline]
pub fn unpack_bf16_to_f32(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: gated on runtime AVX-512F detection; ptrs valid for `n` elems.
            unsafe {
                unpack_bf16_to_f32_avx512(src.as_ptr(), dst.as_mut_ptr(), n);
            }
            return;
        }
    }
    for i in 0..n {
        dst[i] = bf16_bits_to_f32(src[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_bf16_to_f32_avx512(src: *const u16, dst: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    while i + 16 <= n {
        let u16x16 = _mm256_loadu_si256(src.add(i) as *const __m256i); // 16x u16
        let u32x16 = _mm512_cvtepu16_epi32(u16x16); // zero-extend to 16x u32
        let f = _mm512_slli_epi32(u32x16, 16); // bf16 bits << 16 = f32 bits
        _mm512_storeu_ps(dst.add(i), _mm512_castsi512_ps(f));
        i += 16;
    }
    while i < n {
        *dst.add(i) = bf16_bits_to_f32(*src.add(i));
        i += 1;
    }
}

// --- Fused-ELF per-token patching (IRON fusion.py get_patch_locs / patch_elf) -------------------
//
// A whole-decode fused ELF bakes magic sentinels where the per-token KV-cache write offset and the
// softmax mask length must change each token. IRON finds these once (scan the ELF u32 stream for the
// magic) and rewrites them per token, then re-registers the ELF. We mirror that exactly. The ELF is
// treated as a little-endian u32 array (no alignment assumptions — byte-wise LE read/write).

/// IRON's `strided_copy_cache_magic` — base sentinel for KV-cache write-offset patch sites.
pub const KV_CACHE_MAGIC: u32 = 0xDEAD_BEE0;
/// IRON's `softmax_magic` — sentinel for softmax mask-length (context length) patch sites.
pub const SOFTMAX_MAGIC: u32 = 0xBA5E_BA11;

#[inline]
fn elf_u32_at(elf: &[u8], idx: usize) -> u32 {
    let b = idx * 4;
    u32::from_le_bytes([elf[b], elf[b + 1], elf[b + 2], elf[b + 3]])
}

#[inline]
fn elf_set_u32(elf: &mut [u8], idx: usize, val: u32, mask: u32) {
    let b = idx * 4;
    let old = u32::from_le_bytes([elf[b], elf[b + 1], elf[b + 2], elf[b + 3]]);
    let new = (old & !mask) | (val & mask);
    elf[b..b + 4].copy_from_slice(&new.to_le_bytes());
}

/// All u32 indices in `elf` whose word equals `magic` (IRON `get_patch_locs`).
pub fn find_patch_locs(elf: &[u8], magic: u32) -> Vec<usize> {
    let n = elf.len() / 4;
    (0..n).filter(|&i| elf_u32_at(elf, i) == magic).collect()
}

/// Precomputed per-token patch plan for a whole-decode fused ELF. Built ONCE against the freshly
/// generated ELF bytes (scanning for the KV/softmax magics relative to each cache buffer's byte
/// offset), then [`FusedElfPatcher::patch`] produces the per-token ELF to (re)load.
pub struct FusedElfPatcher {
    /// (elf_u32_index, base_byte_offset) — base is the cache buffer's arena offset (0 for the
    /// magic-only sites). Per token: write `base + num_preceding * head_dim * 2` (bf16 bytes).
    kv_locs: Vec<(usize, u32)>,
    /// elf_u32 indices that hold the softmax mask length; per token: write `context_len`.
    softmax_locs: Vec<usize>,
    head_dim: u32,
}

impl FusedElfPatcher {
    /// Scan `elf` for the patch sites. `cache_offsets` = byte offset (within its arena) of every
    /// per-layer K/V cache buffer that receives a per-token strided-copy write. Mirrors
    /// `llama_npu.py:600-640`: sites holding `off + MAGIC*2` get base `off`; sites holding `MAGIC*2`
    /// get base 0; softmax sites hold `SOFTMAX_MAGIC`.
    pub fn build(elf: &[u8], cache_offsets: &[u32], head_dim: u32) -> FusedElfPatcher {
        let mut kv_locs = Vec::new();
        for &off in cache_offsets {
            let target = off.wrapping_add(KV_CACHE_MAGIC.wrapping_mul(2));
            for loc in find_patch_locs(elf, target) {
                kv_locs.push((loc, off));
            }
        }
        for loc in find_patch_locs(elf, KV_CACHE_MAGIC.wrapping_mul(2)) {
            kv_locs.push((loc, 0));
        }
        let softmax_locs = find_patch_locs(elf, SOFTMAX_MAGIC);
        FusedElfPatcher { kv_locs, softmax_locs, head_dim }
    }

    /// Number of KV patch sites found (sanity-check against `4*n_layers+2` for llama-style graphs).
    pub fn kv_site_count(&self) -> usize {
        self.kv_locs.len()
    }
    /// Number of softmax (mask-length) patch sites found.
    pub fn softmax_site_count(&self) -> usize {
        self.softmax_locs.len()
    }

    /// Produce the per-token ELF: KV sites ← `base + num_preceding*head_dim*2`; softmax sites ←
    /// `num_preceding+1` (context length). Returns a fresh buffer to hand to `load_elf_kernel`.
    pub fn patch(&self, elf: &[u8], num_preceding: u32) -> Vec<u8> {
        let mut out = elf.to_vec();
        let offset_val = num_preceding.wrapping_mul(self.head_dim).wrapping_mul(2);
        let context_len = num_preceding + 1;
        for &(loc, base) in &self.kv_locs {
            elf_set_u32(&mut out, loc, base.wrapping_add(offset_val), 0xFFFF_FFFF);
        }
        for &loc in &self.softmax_locs {
            elf_set_u32(&mut out, loc, context_len, 0xFFFF_FFFF);
        }
        out
    }
}

#[cfg(test)]
mod elf_patch_tests {
    use super::*;

    fn put(elf: &mut Vec<u8>, idx: usize, val: u32) {
        if elf.len() < (idx + 1) * 4 {
            elf.resize((idx + 1) * 4, 0);
        }
        elf[idx * 4..idx * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    #[test]
    fn find_locs_matches_exact_words() {
        let mut elf = vec![0u8; 0];
        put(&mut elf, 0, 0x1111_1111);
        put(&mut elf, 1, SOFTMAX_MAGIC);
        put(&mut elf, 2, 0x2222_2222);
        put(&mut elf, 3, SOFTMAX_MAGIC);
        assert_eq!(find_patch_locs(&elf, SOFTMAX_MAGIC), vec![1, 3]);
        assert!(find_patch_locs(&elf, 0xDEAD_0000).is_empty());
    }

    #[test]
    fn kv_and_softmax_patch_per_token() {
        // Two KV caches at byte offsets 0x100 and 0x200; one magic-only (base 0) site; two softmax.
        let head_dim = 64u32;
        let off_a = 0x100u32;
        let off_b = 0x200u32;
        let m2 = KV_CACHE_MAGIC.wrapping_mul(2);
        let mut elf = vec![0u8; 0];
        put(&mut elf, 0, off_a.wrapping_add(m2)); // KV site for cache A
        put(&mut elf, 1, 0xAAAA_AAAA); // untouched
        put(&mut elf, 2, off_b.wrapping_add(m2)); // KV site for cache B
        put(&mut elf, 3, m2); // magic-only site (base 0)
        put(&mut elf, 4, SOFTMAX_MAGIC); // softmax site
        put(&mut elf, 5, SOFTMAX_MAGIC); // softmax site

        let p = FusedElfPatcher::build(&elf, &[off_a, off_b], head_dim);
        assert_eq!(p.kv_site_count(), 3); // A, B, magic-only
        assert_eq!(p.softmax_site_count(), 2);

        // token index 3: offset_val = 3*64*2 = 384 bytes; context_len = 4.
        let out = p.patch(&elf, 3);
        let g = |i| elf_u32_at(&out, i);
        assert_eq!(g(0), off_a + 384);
        assert_eq!(g(1), 0xAAAA_AAAA); // untouched
        assert_eq!(g(2), off_b + 384);
        assert_eq!(g(3), 0 + 384); // base 0 + offset
        assert_eq!(g(4), 4); // context_len
        assert_eq!(g(5), 4);
    }
}

#[cfg(test)]
mod bf16_pack_tests {
    use super::*;

    #[test]
    fn unpack_bf16_to_f32_byte_identical_to_scalar() {
        let src: Vec<u16> = (0u32..=0xffff).map(|b| b as u16).collect(); // every bf16 bit pattern
        let mut got = vec![0f32; src.len()];
        unpack_bf16_to_f32(&src, &mut got);
        for (i, &b) in src.iter().enumerate() {
            let want = bf16_bits_to_f32(b);
            // bit-compare (covers NaN payloads too, since both are pure <<16)
            assert_eq!(got[i].to_bits(), want.to_bits(), "mismatch at bf16 bits {b:#06x}");
        }
    }

    #[test]
    fn pack_unpack_roundtrip_idempotent_on_bf16_values() {
        // bf16-valued f32 (low 16 bits zero) must survive f32->bf16->f32 unchanged.
        let mut src = vec![0f32; 4096 + 3];
        let mut s: u32 = 0xC0FFEE11;
        for v in src.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = f32::from_bits(s & 0xffff_0000); // bf16-valued: low 16 bits cleared
        }
        let mut bits = vec![0u16; src.len()];
        pack_f32_to_bf16(&src, &mut bits);
        let mut back = vec![0f32; src.len()];
        unpack_bf16_to_f32(&bits, &mut back);
        for i in 0..src.len() {
            if src[i].is_nan() {
                continue;
            }
            assert_eq!(src[i].to_bits(), back[i].to_bits(), "roundtrip drift at {i}");
        }
    }

    #[test]
    fn pack_f32_to_bf16_byte_identical_to_scalar() {
        let mut src: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-30, -1e-30, 1e30, -1e30,
            f32::MIN_POSITIVE, f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY,
            f32::NAN, -f32::NAN, 3.14159, 2.71828, 65504.0, 1.0 / 3.0,
        ];
        // deterministic pseudo-random sweep over the full bit range (incl. denormals/NaN/inf)
        let mut s: u32 = 0x1234_5678;
        for _ in 0..200_000 {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            src.push(f32::from_bits(s));
        }
        // non-multiple-of-16 length exercises the scalar tail
        src.extend_from_slice(&[1.5, -2.5, 0.123]);

        let mut got = vec![0u16; src.len()];
        pack_f32_to_bf16(&src, &mut got);
        for (i, &x) in src.iter().enumerate() {
            assert_eq!(
                got[i],
                f32_to_bf16_bits(x),
                "mismatch at {i} for {x:?} (bits {:#010x})",
                x.to_bits()
            );
        }
    }
}
