// C ABI shim over the C++ xrt:: classes. Mirrors the exact dispatch sequence our pyxrt runners
// use (register_xclbin -> hw_context -> kernel -> bo -> variadic kernel() run). docs/12.
#include "xrt_shim.h"
#include <string>
#include <utility>
#include <exception>

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"

struct ShimDevice { xrt::device dev; };
struct ShimKernel { xrt::hw_context ctx; xrt::kernel kern; };
struct ShimBo     { xrt::bo bo; };
struct ShimRun    { xrt::run run; };
// Full-ELF kernel: own the elf + hw_context so they outlive the ext::kernel that references them.
struct ShimElfKernel { xrt::elf elf; xrt::hw_context ctx; xrt::ext::kernel kern; };

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
    // xrt::elf copies the bytes (data,size ctor), so the caller's buffer can be reused/patched.
    xrt::elf elf(elf_bytes, nbytes);
    xrt::hw_context ctx(d->dev, elf);
    std::string name = (kernel_name && kernel_name[0]) ? std::string(kernel_name)
                                                       : std::string("main:sequence");
    xrt::ext::kernel k(ctx, name);
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
