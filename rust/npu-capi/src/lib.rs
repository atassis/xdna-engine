//! C ABI over npu-engine. Handle-based, return-code errors + thread-local last-error. Every entry
//! point catches unwinds (no panic may cross the FFI boundary).

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use npu_engine::{Engine, Model, ModelKind};

thread_local! { static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }

fn set_error(msg: impl Into<String>) {
    let c = CString::new(msg.into()).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Opaque model handle.
pub struct NpuModel(Model);

/// 1 if an NPU device is present, else 0.
#[no_mangle]
pub extern "C" fn npu_available() -> c_int {
    catch_unwind(|| if Engine::available() { 1 } else { 0 }).unwrap_or(0)
}

/// Load a model from a scenario TOML path. Returns NULL on error (see `npu_last_error`).
#[no_mangle]
pub extern "C" fn npu_model_load(scenario_path: *const c_char) -> *mut NpuModel {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if scenario_path.is_null() { set_error("scenario_path is null"); return ptr::null_mut(); }
        let path = match unsafe { CStr::from_ptr(scenario_path) }.to_str() {
            Ok(p) => p,
            Err(_) => { set_error("scenario_path is not valid UTF-8"); return ptr::null_mut(); }
        };
        match Model::load(path) {
            Ok(m) => Box::into_raw(Box::new(NpuModel(m))),
            Err(e) => { set_error(e.to_string()); ptr::null_mut() }
        }
    }));
    r.unwrap_or_else(|_| { set_error("panic in npu_model_load"); ptr::null_mut() })
}

/// 0 = asr, 1 = embed, -1 = error.
#[no_mangle]
pub extern "C" fn npu_model_kind(m: *const NpuModel) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(m) = (unsafe { m.as_ref() }) else { set_error("model is null"); return -1; };
        match m.0.kind() { ModelKind::Asr => 0, ModelKind::Embed => 1 }
    })).unwrap_or(-1)
}

/// ASR: PCM i16 mono -> malloc'd UTF-8 C string (free with `npu_string_free`). NULL on error.
#[no_mangle]
pub extern "C" fn npu_transcribe(m: *mut NpuModel, pcm: *const i16, n: usize, sample_rate: u32)
    -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(m) = (unsafe { m.as_ref() }) else { set_error("model is null"); return ptr::null_mut(); };
        if pcm.is_null() && n != 0 { set_error("pcm is null"); return ptr::null_mut(); }
        let samples = if n == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(pcm, n) } };
        match m.0.transcribe(samples, sample_rate) {
            Ok(text) => CString::new(text).map(|c| c.into_raw()).unwrap_or_else(|_| {
                set_error("transcript contained a NUL byte"); ptr::null_mut() }),
            Err(e) => { set_error(e.to_string()); ptr::null_mut() }
        }
    }));
    r.unwrap_or_else(|_| { set_error("panic in npu_transcribe"); ptr::null_mut() })
}

/// Embedding. Call with out_cap==0 (or out==NULL) to get the dimension; call again with a buffer of
/// at least that many floats to fill it. Returns the dimension, or -1 on error.
#[no_mangle]
pub extern "C" fn npu_embed(m: *mut NpuModel, text: *const c_char, out: *mut f32, out_cap: usize)
    -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(m) = (unsafe { m.as_ref() }) else { set_error("model is null"); return -1; };
        if out.is_null() || out_cap == 0 {
            return match m.0.embed_dim() {
                Some(d) => d as c_int,
                None => { set_error("model is not an embedder"); -1 }
            };
        }
        if text.is_null() { set_error("text is null"); return -1; }
        let text = match unsafe { CStr::from_ptr(text) }.to_str() {
            Ok(t) => t, Err(_) => { set_error("text is not valid UTF-8"); return -1; }
        };
        match m.0.embed(text) {
            Ok(v) => {
                let n = v.len().min(out_cap);
                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out, n); }
                v.len() as c_int
            }
            Err(e) => { set_error(e.to_string()); -1 }
        }
    }));
    r.unwrap_or_else(|_| { set_error("panic in npu_embed"); -1 })
}

/// Free a string returned by `npu_transcribe`.
#[no_mangle]
pub extern "C" fn npu_string_free(s: *mut c_char) {
    if s.is_null() { return; }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { drop(CString::from_raw(s)); }));
}

/// Free a model handle.
#[no_mangle]
pub extern "C" fn npu_model_free(m: *mut NpuModel) {
    if m.is_null() { return; }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(m)); }));
}

/// Thread-local last error message (empty string if none). Pointer valid until the next call on
/// this thread.
#[no_mangle]
pub extern "C" fn npu_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

// --- Runtime control plane (multi-model, config-driven) over npu-runtime ---

use npu_runtime::actor::Handle as RtHandle;
use npu_runtime::config::Config;
use npu_runtime::loader::EngineLoader;

/// Opaque control-plane handle: a running device actor + its config path.
pub struct NpuRuntime { handle: RtHandle, join: Option<std::thread::JoinHandle<()>>, cfg_path: std::path::PathBuf }

/// Start the control plane from a config TOML path (reconciles its models). NULL on error.
#[no_mangle]
pub extern "C" fn npu_runtime_start(config_path: *const c_char) -> *mut NpuRuntime {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if config_path.is_null() { set_error("config_path is null"); return ptr::null_mut(); }
        let p = match unsafe { CStr::from_ptr(config_path) }.to_str() {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => { set_error("config_path is not valid UTF-8"); return ptr::null_mut(); }
        };
        let cfg = match Config::load(&p) { Ok(c) => c, Err(e) => { set_error(e); return ptr::null_mut(); } };
        let root = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let (handle, join) = npu_runtime::start(cfg, Box::new(EngineLoader { root }));
        Box::into_raw(Box::new(NpuRuntime { handle, join: Some(join), cfg_path: p }))
    }));
    r.unwrap_or_else(|_| { set_error("panic in npu_runtime_start"); ptr::null_mut() })
}

/// ASR through the control plane (model name or NULL for the configured default). Caller frees.
#[no_mangle]
pub extern "C" fn npu_runtime_transcribe(rt: *mut NpuRuntime, model: *const c_char,
    pcm: *const i16, n: usize, sample_rate: u32) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(rt) = (unsafe { rt.as_ref() }) else { set_error("runtime is null"); return ptr::null_mut(); };
        let model = opt_str(model);
        let samples = if n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(pcm, n) }.to_vec() };
        match rt.handle.transcribe(model.as_deref(), samples, sample_rate) {
            Ok(s) => CString::new(s.value).map(|c| c.into_raw()).unwrap_or_else(|_| { set_error("NUL in transcript"); ptr::null_mut() }),
            Err(e) => { set_error(e.to_string()); ptr::null_mut() }
        }
    }));
    r.unwrap_or_else(|_| { set_error("panic in npu_runtime_transcribe"); ptr::null_mut() })
}

/// Embedding through the control plane. Pass a buffer; returns the embedding length (>= written), or -1.
#[no_mangle]
pub extern "C" fn npu_runtime_embed(rt: *mut NpuRuntime, model: *const c_char, text: *const c_char,
    out: *mut f32, out_cap: usize) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(rt) = (unsafe { rt.as_ref() }) else { set_error("runtime is null"); return -1; };
        if text.is_null() { set_error("text is null"); return -1; }
        let text = match unsafe { CStr::from_ptr(text) }.to_str() { Ok(t) => t, Err(_) => { set_error("text not UTF-8"); return -1; } };
        match rt.handle.embed(opt_str(model).as_deref(), text) {
            Ok(s) => {
                if !out.is_null() && out_cap > 0 {
                    let k = s.value.len().min(out_cap);
                    unsafe { std::ptr::copy_nonoverlapping(s.value.as_ptr(), out, k); }
                }
                s.value.len() as c_int
            }
            Err(e) => { set_error(e.to_string()); -1 }
        }
    }));
    r.unwrap_or_else(|_| { set_error("panic in npu_runtime_embed"); -1 })
}

/// Re-read the config file and reconcile. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn npu_runtime_reload(rt: *mut NpuRuntime) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(rt) = (unsafe { rt.as_ref() }) else { set_error("runtime is null"); return -1; };
        let cfg = match Config::load(&rt.cfg_path) { Ok(c) => c, Err(e) => { set_error(e); return -1; } };
        match rt.handle.reconcile(cfg) { Ok(_) => 0, Err(e) => { set_error(e.to_string()); -1 } }
    })).unwrap_or(-1)
}

/// Model statuses as a JSON list (malloc'd; free with npu_string_free). NULL on error.
#[no_mangle]
pub extern "C" fn npu_runtime_models_json(rt: *mut NpuRuntime) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(rt) = (unsafe { rt.as_ref() }) else { set_error("runtime is null"); return ptr::null_mut(); };
        let json = npu_runtime::http::models_json(&rt.handle.status());
        CString::new(json).map(|c| c.into_raw()).unwrap_or(ptr::null_mut())
    }));
    r.unwrap_or_else(|_| { set_error("panic in npu_runtime_models_json"); ptr::null_mut() })
}

/// Stop the control plane and free the handle.
#[no_mangle]
pub extern "C" fn npu_runtime_stop(rt: *mut NpuRuntime) {
    if rt.is_null() { return; }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let mut rt = unsafe { Box::from_raw(rt) };
        rt.handle.shutdown();
        if let Some(j) = rt.join.take() { let _ = j.join(); }
    }));
}

fn opt_str(p: *const c_char) -> Option<String> {
    if p.is_null() { None } else { unsafe { CStr::from_ptr(p) }.to_str().ok().map(String::from) }
}
