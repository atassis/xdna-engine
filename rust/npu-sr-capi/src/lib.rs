//! C ABI over npu-sr. Handle-based, return-code errors + thread-local last-error. No panic crosses the
//! FFI boundary. Mirrors npu-capi conventions. This is the boundary the ffmpeg vf_xdna_sr filter links.
use npu_sr::SrEngine;
use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

thread_local! { static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn set_error(m: impl Into<String>) {
    let c = CString::new(m.into()).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Opaque SR engine handle.
pub struct XdnaSr(SrEngine);

/// 1 if an NPU device is present, else 0.
#[no_mangle]
pub extern "C" fn xdna_sr_available() -> c_int {
    catch_unwind(|| if npu_sr::npu_available() { 1 } else { 0 }).unwrap_or(0)
}

/// Load a schedule (path to `<net>.json`). `use_npu`!=0 uses the NPU frontier. NULL on error
/// (see `xdna_sr_last_error`).
#[no_mangle]
pub extern "C" fn xdna_sr_create(schedule_path: *const c_char, use_npu: c_int) -> *mut XdnaSr {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if schedule_path.is_null() {
            set_error("schedule_path is null");
            return ptr::null_mut();
        }
        let p = match unsafe { CStr::from_ptr(schedule_path) }.to_str() {
            Ok(p) => p,
            Err(_) => {
                set_error("schedule_path is not valid UTF-8");
                return ptr::null_mut();
            }
        };
        match SrEngine::load(p, use_npu != 0) {
            Ok(e) => Box::into_raw(Box::new(XdnaSr(e))),
            Err(e) => {
                set_error(e.to_string());
                ptr::null_mut()
            }
        }
    }));
    r.unwrap_or_else(|_| {
        set_error("panic in xdna_sr_create");
        ptr::null_mut()
    })
}

/// The integer scale factor of the loaded net (e.g. 3), or -1 on error.
#[no_mangle]
pub extern "C" fn xdna_sr_scale(h: *const XdnaSr) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_ref() }) else {
            set_error("handle is null");
            return -1;
        };
        h.0.scale() as c_int
    }))
    .unwrap_or(-1)
}

/// Upscale one interleaved RGB8 frame. `out_rgb` must hold at least (w*scale)*(h*scale)*3 bytes.
/// Returns 0 on success (writing out_w/out_h if non-null), <0 on error.
#[no_mangle]
pub extern "C" fn xdna_sr_process_rgb8(
    h: *mut XdnaSr,
    in_rgb: *const u8,
    w: usize,
    height: usize,
    out_rgb: *mut u8,
    out_cap: usize,
    out_w: *mut usize,
    out_h: *mut usize,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_mut() }) else {
            set_error("handle is null");
            return -1;
        };
        if in_rgb.is_null() || out_rgb.is_null() {
            set_error("null buffer");
            return -1;
        }
        let src = unsafe { std::slice::from_raw_parts(in_rgb, w * height * 3) };
        match h.0.upscale_rgb8(src, w, height) {
            Ok((buf, ow, oh)) => {
                if buf.len() > out_cap {
                    set_error(format!("out buffer too small: need {}, have {}", buf.len(), out_cap));
                    return -1;
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(buf.as_ptr(), out_rgb, buf.len());
                    if !out_w.is_null() {
                        *out_w = ow;
                    }
                    if !out_h.is_null() {
                        *out_h = oh;
                    }
                }
                0
            }
            Err(e) => {
                set_error(e.to_string());
                -1
            }
        }
    }));
    r.unwrap_or_else(|_| {
        set_error("panic in xdna_sr_process_rgb8");
        -1
    })
}

/// Free an engine handle.
#[no_mangle]
pub extern "C" fn xdna_sr_free(h: *mut XdnaSr) {
    if h.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(h)); }));
}

/// Thread-local last error message (empty string if none). Pointer valid until the next call on this thread.
#[no_mangle]
pub extern "C" fn xdna_sr_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}
