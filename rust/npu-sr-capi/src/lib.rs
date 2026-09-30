//! C ABI over npu-sr. Handle-based, return-code errors + thread-local last-error. No panic crosses the
//! FFI boundary. Mirrors npu-capi conventions. This is the boundary the ffmpeg vf_xdna_sr filter links.
use npu_sr::SrEngine;
use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

/// Padded-frame geometry a `layout: "frame"` fsr1 backend expects, for a zero-copy dma-buf caller
/// that builds its own buffers. See [`xdna_sr_frame_layout`].
#[repr(C)]
pub struct XdnaSrFrameLayout {
    pub in_w: usize,
    pub in_h: usize,
    pub in_pad_w: usize,
    pub in_pad_h: usize,
    pub in_pad_x: usize,
    pub in_pad_y: usize,
    pub out_pad_w: usize,
    pub out_pad_h: usize,
    pub scale: usize,
}

/// Padded-frame geometry of a one-design Y engine at any scale, for a zero-copy dma-buf caller. The
/// input plane is in_pad_w x in_pad_h bytes with the frame at (in_pad_x, in_pad_y), the rest
/// edge-replicated by the producer; the output plane is out_pad_w x out_pad_h bytes with the valid
/// out_w x out_h at its top left. 1 byte per pixel (the NV12 Y plane). Scale is scale_num/scale_den.
#[repr(C)]
pub struct XdnaSrLayout {
    pub in_w: usize,
    pub in_h: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub in_pad_w: usize,
    pub in_pad_h: usize,
    pub in_pad_x: usize,
    pub in_pad_y: usize,
    pub out_pad_w: usize,
    pub out_pad_h: usize,
    pub scale_num: usize,
    pub scale_den: usize,
}

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
pub unsafe extern "C" fn xdna_sr_create(schedule_path: *const c_char, use_npu: c_int) -> *mut XdnaSr {
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

/// 1 if the one-design export at `export_path` (as for `xdna_sr_create_scaled`) has a scale for
/// `in` -> `out`, writing it to *num / *den if non-null; else 0. Reads the export's json only, no
/// device.
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_scale_supported(export_path: *const c_char, in_w: usize, in_h: usize,
                                                 out_w: usize, out_h: usize, num: *mut usize,
                                                 den: *mut usize) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        if export_path.is_null() {
            return 0;
        }
        let Ok(p) = (unsafe { CStr::from_ptr(export_path) }).to_str() else { return 0 };
        match SrEngine::scale_supported(std::path::Path::new(p), in_w, in_h, out_w, out_h) {
            Some((a, b)) => {
                unsafe {
                    if !num.is_null() {
                        *num = a;
                    }
                    if !den.is_null() {
                        *den = b;
                    }
                }
                1
            }
            None => 0,
        }
    }))
    .unwrap_or(0)
}

/// A one-design upscaler on the NV12 Y plane for `in` -> `out`: `export_path` is FSR1's export
/// (fsr1_rt.json) or another design's tiled-table export (frame_rt.json), or the directory holding
/// it; the scale comes from the sizes (see `xdna_sr_scale_supported`). `sharpness`: FSR1's RCAS in
/// stops, 0 the sharpest (ignored by other designs). Query the buffers with `xdna_sr_layout`,
/// dispatch with `xdna_sr_process_dmabuf`, change shape with `xdna_sr_configure`. NULL on error
/// (an unsupported scale included; see `xdna_sr_last_error`).
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_create_scaled(export_path: *const c_char, in_w: usize, in_h: usize,
                                               out_w: usize, out_h: usize, sharpness: f32) -> *mut XdnaSr {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if export_path.is_null() {
            set_error("export_path is null");
            return ptr::null_mut();
        }
        let Ok(p) = (unsafe { CStr::from_ptr(export_path) }).to_str() else {
            set_error("export_path is not valid UTF-8");
            return ptr::null_mut();
        };
        match SrEngine::load_scaled(p, in_w, in_h, out_w, out_h, sharpness as f64) {
            Ok(e) => Box::into_raw(Box::new(XdnaSr(e))),
            Err(e) => {
                set_error(e.to_string());
                ptr::null_mut()
            }
        }
    }));
    r.unwrap_or_else(|_| {
        set_error("panic in xdna_sr_create_scaled");
        ptr::null_mut()
    })
}

/// Switch an `xdna_sr_create_scaled` engine to another frame shape (any supported scale and size)
/// and sharpness without reloading the design. The layout changes: query it again. 0, or <0 on
/// error (the engine keeps its previous shape).
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_configure(h: *mut XdnaSr, in_w: usize, in_h: usize, out_w: usize,
                                           out_h: usize, sharpness: f32) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_mut() }) else {
            set_error("handle is null");
            return -1;
        };
        match h.0.configure(in_w, in_h, out_w, out_h, sharpness as f64) {
            Ok(()) => 0,
            Err(e) => {
                set_error(e.to_string());
                -1
            }
        }
    }));
    r.unwrap_or_else(|_| {
        set_error("panic in xdna_sr_configure");
        -1
    })
}

/// Fills `*out` for an `xdna_sr_create_scaled` engine (any scale) or a frame-layout schedule
/// (integer scale); 0, or <0 for other backends.
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_layout(h: *const XdnaSr, out: *mut XdnaSrLayout) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_ref() }) else {
            set_error("handle is null");
            return -1;
        };
        if out.is_null() {
            set_error("out is null");
            return -1;
        }
        let l = match h.0.rt_layout() {
            Ok(l) => XdnaSrLayout {
                in_w: l.in_w, in_h: l.in_h, out_w: l.out_w, out_h: l.out_h, in_pad_w: l.in_pad_w,
                in_pad_h: l.in_pad_h, in_pad_x: l.in_pad_x, in_pad_y: l.in_pad_y, out_pad_w: l.out_pad_w,
                out_pad_h: l.out_pad_h, scale_num: l.scale_num, scale_den: l.scale_den,
            },
            Err(_) => match h.0.frame_layout() {
                Ok(l) => XdnaSrLayout {
                    in_w: l.in_w, in_h: l.in_h, out_w: l.scale * l.in_w, out_h: l.scale * l.in_h,
                    in_pad_w: l.in_pad_w, in_pad_h: l.in_pad_h, in_pad_x: l.in_pad_x, in_pad_y: l.in_pad_y,
                    out_pad_w: l.out_pad_w, out_pad_h: l.out_pad_h, scale_num: l.scale, scale_den: 1,
                },
                Err(e) => {
                    set_error(e.to_string());
                    return -1;
                }
            },
        };
        unsafe { *out = l };
        0
    }));
    r.unwrap_or_else(|_| {
        set_error("panic in xdna_sr_layout");
        -1
    })
}

/// The integer scale factor of the loaded net (e.g. 3), 0 for a one-design engine at a scale with
/// no integer form (3/2, 5/3: see `xdna_sr_layout`), or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_scale(h: *const XdnaSr) -> c_int {
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
pub unsafe extern "C" fn xdna_sr_process_rgb8(
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

/// Upscale one BGRA8 frame (DRM ARGB8888 byte order: B, G, R, A in memory). Rows are `in_stride`
/// / `out_stride` bytes apart; `out_rgba` holds at least (h*scale-1)*out_stride + w*scale*4 bytes
/// (`out_cap`). Output alpha is 0xff. Returns 0 on success (writing out_w/out_h if non-null).
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_process_bgra8(
    h: *mut XdnaSr,
    in_bgra: *const u8,
    w: usize,
    height: usize,
    in_stride: usize,
    out_bgra: *mut u8,
    out_stride: usize,
    out_cap: usize,
    out_w: *mut usize,
    out_h: *mut usize,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_mut() }) else {
            set_error("handle is null");
            return -1;
        };
        if in_bgra.is_null() || out_bgra.is_null() || w == 0 || height == 0 || in_stride < w * 4 {
            set_error("null buffer, empty frame, or in_stride < w*4");
            return -1;
        }
        let src = unsafe { std::slice::from_raw_parts(in_bgra, (height - 1) * in_stride + w * 4) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out_bgra, out_cap) };
        match h.0.upscale_bgra8(src, w, height, in_stride, dst, out_stride) {
            Ok((ow, oh)) => {
                unsafe {
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
        set_error("panic in xdna_sr_process_bgra8");
        -1
    })
}

/// Fills `*out` with the loaded backend's padded-frame geometry and returns 0 if it takes padded
/// BGRA frames (fsr1 frame layout); <0 (e.g. a conv-net or tiled-fsr1 schedule) otherwise.
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_frame_layout(h: *const XdnaSr, out: *mut XdnaSrFrameLayout) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_ref() }) else {
            set_error("handle is null");
            return -1;
        };
        if out.is_null() {
            set_error("out is null");
            return -1;
        }
        match h.0.frame_layout() {
            Ok(l) => {
                unsafe {
                    *out = XdnaSrFrameLayout {
                        in_w: l.in_w,
                        in_h: l.in_h,
                        in_pad_w: l.in_pad_w,
                        in_pad_h: l.in_pad_h,
                        in_pad_x: l.in_pad_x,
                        in_pad_y: l.in_pad_y,
                        out_pad_w: l.out_pad_w,
                        out_pad_h: l.out_pad_h,
                        scale: l.scale,
                    };
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
        set_error("panic in xdna_sr_frame_layout");
        -1
    })
}

/// Zero-copy dispatch: `in_fd`/`out_fd` are dma-buf fds of BGRA8 buffers laid out per
/// Bytes per pixel of the frame layout's planes: 4 for BGRA, 1 for the NV12 Y plane (the caller
/// fills the chroma plane of its output buffer itself). <0 without a frame-layout backend.
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_frame_bytes_per_px(h: *const XdnaSr) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_ref() }) else {
            set_error("handle is null");
            return -1;
        };
        match h.0.frame_bytes_per_px() {
            Ok(b) => b as c_int,
            Err(e) => {
                set_error(e.to_string());
                -1
            }
        }
    }));
    r.unwrap_or_else(|_| {
        set_error("panic in xdna_sr_frame_bytes_per_px");
        -1
    })
}

/// `xdna_sr_frame_layout` (in: producer-filled INCLUDING the edge-replicated padding; out: the full
/// padded output). Imports and caches each fd (by the dma-buf's inode). Blocking: returns after the
/// NPU finished. `*out_fence_fd` is set to -1 (reserved for an async version). <0 on error.
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_process_dmabuf(
    h: *mut XdnaSr,
    in_fd: c_int,
    out_fd: c_int,
    out_fence_fd: *mut c_int,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(h) = (unsafe { h.as_mut() }) else {
            set_error("handle is null");
            return -1;
        };
        if !out_fence_fd.is_null() {
            unsafe {
                *out_fence_fd = -1;
            }
        }
        match h.0.process_dmabuf(in_fd, out_fd) {
            Ok(()) => 0,
            Err(e) => {
                set_error(e.to_string());
                -1
            }
        }
    }));
    r.unwrap_or_else(|_| {
        set_error("panic in xdna_sr_process_dmabuf");
        -1
    })
}

/// Free an engine handle.
#[no_mangle]
pub unsafe extern "C" fn xdna_sr_free(h: *mut XdnaSr) {
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
