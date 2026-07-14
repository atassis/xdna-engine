// C ABI shim over the C++ xrt:: classes. Mirrors the exact dispatch sequence our pyxrt runners
// use (register_xclbin -> hw_context -> kernel -> bo -> variadic kernel() run). docs/12.
#include "xrt_shim.h"
#include <string>
#include <utility>
#include <exception>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

struct ShimDevice { xrt::device dev; };
struct ShimKernel { xrt::hw_context ctx; xrt::kernel kern; };
struct ShimBo     { xrt::bo bo; };
struct ShimRun    { xrt::run run; };
// Full-ELF kernel: own the elf + hw_context so they outlive the ext::kernel that references them.
struct ShimElfKernel { xrt::elf elf; xrt::hw_context ctx; xrt::ext::kernel kern; };
// Persistent-context path: ctx owns the partition (built once); ShimElfKernel2 borrows it.
struct ShimElfCtx     { xrt::elf base_elf; xrt::hw_context ctx; };
struct ShimElfKernel2 { xrt::elf elf; xrt::module mod; xrt::ext::kernel kern; };

// Per-sub-step timing of the ELF load path (attribution of the per-token re-registration cost),
// gated by env XRT_SHIM_ELF_TIMING so it is a true no-op in production.
static bool elf_timing() {
  static const bool on = std::getenv("XRT_SHIM_ELF_TIMING") != nullptr;
  return on;
}
using shim_clock = std::chrono::steady_clock;
static double ms_since(shim_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(shim_clock::now() - t0).count();
}

static thread_local std::string g_err;
static void set_err(const char* s) { g_err = s ? s : "null"; }

const char* shim_last_error(void) { return g_err.c_str(); }

// Variadic so commas inside the body aren't parsed as extra macro arguments.
#define GUARD_PTR(...) \
  try { __VA_ARGS__ } \
  catch (const std::exception& e) { set_err(e.what()); return nullptr; } \
  catch (...) { set_err("unknown C++ exception"); return nullptr; }

#define GUARD_INT(...) \
  try { __VA_ARGS__ } \
  catch (const std::exception& e) { set_err(e.what()); return -1; } \
  catch (...) { set_err("unknown C++ exception"); return -1; }

ShimDevice* shim_device_open(unsigned int index) {
  GUARD_PTR( return new ShimDevice{ xrt::device(index) }; )
}
void shim_device_close(ShimDevice* d) { delete d; }

ShimKernel* shim_kernel_load(ShimDevice* d, const char* xclbin_path, const char* kernel_name) {
  GUARD_PTR(
    xrt::xclbin xb(std::string(xclbin_path ? xclbin_path : ""));
    auto uuid = d->dev.register_xclbin(xb);
    xrt::hw_context ctx(d->dev, uuid);
    std::string name = (kernel_name && kernel_name[0])
                         ? std::string(kernel_name)
                         : xb.get_kernels().front().get_name();
    xrt::kernel k(ctx, name);
    return new ShimKernel{ std::move(ctx), std::move(k) };
  )
}
void shim_kernel_close(ShimKernel* k) { delete k; }

int shim_kernel_group_id(ShimKernel* k, int arg_index) {
  GUARD_INT( return static_cast<int>(k->kern.group_id(arg_index)); )
}

static xrt::bo::flags to_flags(int f) {
  switch (f) {
    case 1:  return xrt::bo::flags::cacheable;
    case 2:  return xrt::bo::flags::host_only;
    default: return xrt::bo::flags::normal;
  }
}

ShimBo* shim_bo_alloc(ShimDevice* d, ShimKernel* /*k*/, size_t nbytes, int flag, int group_id) {
  GUARD_PTR( return new ShimBo{ xrt::bo(d->dev, nbytes, to_flags(flag), group_id) }; )
}
// Sub-buffer view: a device-side [offset, offset+size) window of `parent`, sharing its memory
// (XRT-native xrt::bo(parent, size, offset)). Lets a kernel read/write a slice of a larger BO with
// no host round-trip -- e.g. a chunk of a chunk-major fc2 A buffer, or a KV-cache slice.
ShimBo* shim_bo_subbuffer(ShimBo* parent, size_t size, size_t offset) {
  GUARD_PTR( return new ShimBo{ xrt::bo(parent->bo, size, offset) }; )
}
void shim_bo_free(ShimBo* b) { delete b; }

int shim_bo_write(ShimBo* b, const void* src, size_t nbytes, size_t offset) {
  GUARD_INT( b->bo.write(src, nbytes, offset); return 0; )
}
int shim_bo_read(ShimBo* b, void* dst, size_t nbytes, size_t offset) {
  GUARD_INT( b->bo.read(dst, nbytes, offset); return 0; )
}
int shim_bo_sync_to_device(ShimBo* b) {
  GUARD_INT( b->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); return 0; )
}
int shim_bo_sync_from_device(ShimBo* b) {
  GUARD_INT( b->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); return 0; )
}

int shim_run_matmul8(ShimKernel* k, unsigned int opcode, ShimBo* instr, size_t instr_count,
                     ShimBo* a, ShimBo* b, ShimBo* c, ShimBo* tmp, ShimBo* trace) {
  GUARD_INT(
    auto run = k->kern(opcode, instr->bo, instr_count,
                       a->bo, b->bo, c->bo, tmp->bo, trace->bo);
    ert_cmd_state st = run.wait();
    if (st != ERT_CMD_STATE_COMPLETED) { set_err("kernel run did not complete"); return -1; }
    return 0;
  )
}

int shim_run_dwconv6(ShimKernel* k, unsigned int opcode, ShimBo* instr, size_t instr_count,
                     ShimBo* x, ShimBo* w, ShimBo* y) {
  GUARD_INT(
    auto run = k->kern(opcode, instr->bo, instr_count, x->bo, w->bo, y->bo);
    ert_cmd_state st = run.wait();
    if (st != ERT_CMD_STATE_COMPLETED) { set_err("kernel run did not complete"); return -1; }
    return 0;
  )
}

// MHA host ABI: kernel(opcode, instr, instr_count, Q, K, V, O) — 4 data BOs (the IRON MHA op's
// rt.sequence(Q, K, V, O)). Same structure as shim_run_matmul8, one fewer data arg.
int shim_run_mha7(ShimKernel* k, unsigned int opcode, ShimBo* instr, size_t instr_count,
                  ShimBo* q, ShimBo* kk, ShimBo* v, ShimBo* o) {
  GUARD_INT(
    auto run = k->kern(opcode, instr->bo, instr_count, q->bo, kk->bo, v->bo, o->bo);
    ert_cmd_state st = run.wait();
    if (st != ERT_CMD_STATE_COMPLETED) { set_err("kernel run did not complete"); return -1; }
    return 0;
  )
}

// xrt::kernel::operator() constructs a run AND starts it (enqueues the command); the wait is
// separate. So building the run here = the async "start"; the host returns immediately while the
// NPU executes. Same arg layout as shim_run_matmul8.
ShimRun* shim_run_matmul8_start(ShimKernel* k, unsigned int opcode, ShimBo* instr, size_t instr_count,
                                ShimBo* a, ShimBo* b, ShimBo* c, ShimBo* tmp, ShimBo* trace) {
  GUARD_PTR(
    auto run = k->kern(opcode, instr->bo, instr_count,
                       a->bo, b->bo, c->bo, tmp->bo, trace->bo);
    return new ShimRun{ std::move(run) };
  )
}

int shim_run_wait(ShimRun* r) {
  GUARD_INT(
    ert_cmd_state st = r->run.wait();
    if (st != ERT_CMD_STATE_COMPLETED) { set_err("kernel run did not complete"); return -1; }
    return 0;
  )
}

void shim_run_free(ShimRun* r) { delete r; }

// --- Fused full-ELF dispatch (IRON FusedMLIROperator path) ---------------------------------------

ShimElfKernel* shim_elf_kernel_load(ShimDevice* d, const void* elf_bytes, size_t nbytes,
                                    const char* kernel_name) {
  GUARD_PTR(
    const bool tm = elf_timing();
    auto t0 = shim_clock::now();
    // xrt::elf copies the bytes (data,size ctor), so the caller's buffer can be reused/patched.
    xrt::elf elf(elf_bytes, nbytes);
    double t_elf = tm ? ms_since(t0) : 0.0; auto t1 = shim_clock::now();
    xrt::hw_context ctx(d->dev, elf);
    double t_ctx = tm ? ms_since(t1) : 0.0; auto t2 = shim_clock::now();
    std::string name = (kernel_name && kernel_name[0]) ? std::string(kernel_name)
                                                       : std::string("main:sequence");
    xrt::ext::kernel k(ctx, name);
    double t_kern = tm ? ms_since(t2) : 0.0;
    if (tm) {
      std::fprintf(stderr, "[XRT_SHIM_ELF] load: elf=%.3f hw_context=%.3f ext_kernel=%.3f ms\n",
                   t_elf, t_ctx, t_kern);
    }
    return new ShimElfKernel{ std::move(elf), std::move(ctx), std::move(k) };
  )
}

void shim_elf_kernel_close(ShimElfKernel* k) { delete k; }

int shim_run_elf(ShimElfKernel* k, ShimBo* const* bos, size_t n_bos) {
  GUARD_INT(
    xrt::run run(k->kern);
    for (size_t i = 0; i < n_bos; ++i) {
      run.set_arg(static_cast<int>(i), bos[i]->bo);
    }
    run.start();
    ert_cmd_state st = run.wait();
    if (st != ERT_CMD_STATE_COMPLETED) { set_err("elf run did not complete"); return -1; }
    return 0;
  )
}

// Async "start" of the full-ELF dispatch: set args + run.start() enqueues the command and returns
// immediately; the host registers the next token's position-only ELF while the NPU runs, then
// shim_run_wait blocks for completion. Same ShimRun handle as the matmul async path.
ShimRun* shim_run_elf_start(ShimElfKernel* k, ShimBo* const* bos, size_t n_bos) {
  GUARD_PTR(
    xrt::run run(k->kern);
    for (size_t i = 0; i < n_bos; ++i) {
      run.set_arg(static_cast<int>(i), bos[i]->bo);
    }
    run.start();
    return new ShimRun{ std::move(run) };
  )
}

// --- Persistent-hw_context path ---------------------------------------------------------------

ShimElfCtx* shim_elf_ctx_open(ShimDevice* d, const void* base_elf, size_t nbytes) {
  GUARD_PTR(
    xrt::elf elf(base_elf, nbytes);
    xrt::hw_context ctx(d->dev, elf);   // partition config ONCE — the recurring cost we hoist out
    return new ShimElfCtx{ std::move(elf), std::move(ctx) };
  )
}

void shim_elf_ctx_close(ShimElfCtx* c) { delete c; }

ShimElfKernel2* shim_elf_kernel_rebind(ShimElfCtx* c, const void* elf_bytes, size_t nbytes,
                                       const char* kernel_name) {
  GUARD_PTR(
    const bool tm = elf_timing();
    auto t0 = shim_clock::now();
    xrt::elf elf(elf_bytes, nbytes);
    double t_elf = tm ? ms_since(t0) : 0.0; auto t1 = shim_clock::now();
    xrt::module mod(elf);
    double t_mod = tm ? ms_since(t1) : 0.0; auto t2 = shim_clock::now();
    std::string name = (kernel_name && kernel_name[0]) ? std::string(kernel_name)
                                                       : std::string("main:sequence");
    xrt::ext::kernel k(c->ctx, mod, name);   // bind patched module onto the resident context
    double t_kern = tm ? ms_since(t2) : 0.0;
    if (tm) {
      std::fprintf(stderr, "[XRT_SHIM_ELF] rebind: elf=%.3f module=%.3f ext_kernel=%.3f ms\n",
                   t_elf, t_mod, t_kern);
    }
    return new ShimElfKernel2{ std::move(elf), std::move(mod), std::move(k) };
  )
}

void shim_elf_kernel2_close(ShimElfKernel2* k) { delete k; }

int shim_run_elf2(ShimElfKernel2* k, ShimBo* const* bos, size_t n_bos) {
  GUARD_INT(
    xrt::run run(k->kern);
    for (size_t i = 0; i < n_bos; ++i) {
      run.set_arg(static_cast<int>(i), bos[i]->bo);
    }
    run.start();
    ert_cmd_state st = run.wait();
    if (st != ERT_CMD_STATE_COMPLETED) { set_err("elf2 run did not complete"); return -1; }
    return 0;
  )
}

// --- Resident full-ELF runner with ctrl-scratchpad parameters --------------------------------------

struct ShimElfResident {
  xrt::elf elf;
  xrt::hw_context ctx;
  xrt::ext::kernel kern;
  xrt::run run;
  xrt::bo scratchpad;     // ctrl scratchpad BO (from the run), empty if the ELF has none
  uint8_t* scratch_map;   // host mapping of the scratchpad (nullptr if none)
  size_t scratch_size;
};

ShimElfResident* shim_elf_resident_open(ShimDevice* d, const void* elf_bytes, size_t nbytes,
                                        const char* kernel_name) {
  GUARD_PTR(
    xrt::elf elf(elf_bytes, nbytes);
    xrt::hw_context ctx(d->dev, elf);
    std::string name = (kernel_name && kernel_name[0]) ? std::string(kernel_name)
                                                       : std::string("main:sequence");
    xrt::ext::kernel k(ctx, name);
    xrt::run run(k);
    // get_ctrl_scratchpad_bo throws if the ELF has no scratchpad section — that means this ELF is not
    // a scratchpad-parameter build, so the caller must use the patch path. Surface as NULL.
    xrt::bo sp = run.get_ctrl_scratchpad_bo();
    uint8_t* mp = sp.map<uint8_t*>();
    size_t sz = sp.size();
    return new ShimElfResident{ std::move(elf), std::move(ctx), std::move(k), std::move(run),
                                std::move(sp), mp, sz };
  )
}

void shim_elf_resident_close(ShimElfResident* r) { delete r; }

size_t shim_elf_resident_scratchpad_size(ShimElfResident* r) {
  return r ? r->scratch_size : 0;
}

int shim_elf_resident_bind(ShimElfResident* r, ShimBo* const* bos, size_t n_bos) {
  GUARD_INT(
    for (size_t i = 0; i < n_bos; ++i) {
      r->run.set_arg(static_cast<int>(i), bos[i]->bo);
    }
    return 0;
  )
}

int shim_elf_resident_write(ShimElfResident* r, size_t offset, const void* data, size_t len) {
  GUARD_INT(
    if (!r->scratch_map) { set_err("resident has no ctrl scratchpad"); return -1; }
    if (offset + len > r->scratch_size) { set_err("scratchpad write out of range"); return -1; }
    std::memcpy(r->scratch_map + offset, data, len);
    return 0;
  )
}

int shim_elf_resident_dispatch(ShimElfResident* r) {
  GUARD_INT(
    r->scratchpad.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    r->run.start();
    ert_cmd_state st = r->run.wait();
    if (st != ERT_CMD_STATE_COMPLETED) { set_err("resident run did not complete"); return -1; }
    return 0;
  )
}
