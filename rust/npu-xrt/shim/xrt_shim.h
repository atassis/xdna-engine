/* Minimal C ABI over the C++ xrt:: classes (the same surface pyxrt wraps), so Rust can drive
 * the XDNA2 NPU via FFI without exposing C++ name-mangling/ABI. See internal notes.
 * Every fallible call returns NULL / nonzero and stashes a message in shim_last_error(). */
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ShimDevice    ShimDevice;
typedef struct ShimKernel    ShimKernel;
typedef struct ShimBo        ShimBo;
typedef struct ShimRun       ShimRun;
typedef struct ShimElfKernel ShimElfKernel;

ShimDevice* shim_device_open(unsigned int index);
void        shim_device_close(ShimDevice*);

/* Load xclbin -> register with device -> hw_context -> kernel. NULL/empty kernel_name uses the
 * first kernel in the xclbin (what our pyxrt runners do). Returns NULL on failure. */
ShimKernel* shim_kernel_load(ShimDevice*, const char* xclbin_path, const char* kernel_name);
void        shim_kernel_close(ShimKernel*);
int         shim_kernel_group_id(ShimKernel*, int arg_index); /* -1 on error */

/* flag: 0=normal, 1=cacheable (instr), 2=host_only (data) */
ShimBo*     shim_bo_alloc(ShimDevice*, ShimKernel*, size_t nbytes, int flag, int group_id);
void        shim_bo_free(ShimBo*);
int         shim_bo_write(ShimBo*, const void* src, size_t nbytes, size_t offset); /* 0 ok */
int         shim_bo_read (ShimBo*, void* dst, size_t nbytes, size_t offset);
int         shim_bo_sync_to_device(ShimBo*);
int         shim_bo_sync_from_device(ShimBo*);

/* whole_array / matmul host ABI: kernel(opcode, instr, instr_count, A, B, C, tmp, trace).
 * Creates a run, dispatches, waits for completion. 0 on success. */
int shim_run_matmul8(ShimKernel*, unsigned int opcode, ShimBo* instr, size_t instr_count,
                     ShimBo* a, ShimBo* b, ShimBo* c, ShimBo* tmp, ShimBo* trace);

/* depthwise-conv1d host ABI: kernel(opcode, instr, instr_count, X, W, Y) — no tmp/trace. */
int shim_run_dwconv6(ShimKernel*, unsigned int opcode, ShimBo* instr, size_t instr_count,
                     ShimBo* x, ShimBo* w, ShimBo* y);

/* ASYNC split of shim_run_matmul8: _start submits the run (xrt::kernel::operator() enqueues +
 * starts execution) and returns a run handle WITHOUT waiting, so the host can do other work (prep
 * the next dispatch, post-process the previous) while the NPU computes. _wait blocks for completion.
 * Returns NULL/-1 on failure (message in shim_last_error). Free the handle with shim_run_free. */
ShimRun* shim_run_matmul8_start(ShimKernel*, unsigned int opcode, ShimBo* instr, size_t instr_count,
                                ShimBo* a, ShimBo* b, ShimBo* c, ShimBo* tmp, ShimBo* trace);
int      shim_run_wait(ShimRun*); /* 0 = completed, -1 = error/not-completed */
void     shim_run_free(ShimRun*);

/* --- Fused full-ELF dispatch (the IRON FusedMLIROperator path) ----------------------------------
 * A full ELF carries its own instructions+config (no xclbin, no insts BO). Mirrors IRON's
 * fusion.py FullELFCallable: xrt::elf(bytes) -> hw_context(device, elf) -> ext::kernel(ctx, name).
 * kernel_name NULL/empty defaults to "main:sequence" (IRON's device:sequence default). The ELF bytes
 * are copied in, so the caller may patch+reload its own buffer freely. Returns NULL on failure. */
ShimElfKernel* shim_elf_kernel_load(ShimDevice*, const void* elf_bytes, size_t nbytes,
                                    const char* kernel_name);
void           shim_elf_kernel_close(ShimElfKernel*);

/* Dispatch a full-ELF kernel with N buffer-object args (run.set_arg(i, bo) for i in 0..n_bos),
 * start + wait. The fused-arena ABI passes exactly 3 BOs (input, output, scratch), but this is
 * generic-N to match IRON's variadic set_arg loop. 0 on success. */
int shim_run_elf(ShimElfKernel*, ShimBo* const* bos, size_t n_bos);

/* ASYNC split of shim_run_elf (the PIPE lever): _start submits the ELF run (set_arg + run.start())
 * and returns a run handle WITHOUT waiting, so the host can register the NEXT token's position-only
 * patched ELF while this dispatch executes on the NPU. Wait via shim_run_wait, free via shim_run_free
 * (the same ShimRun handle the matmul async path uses). The ShimElfKernel and all BOs must outlive the
 * returned ShimRun. Returns NULL on failure (message in shim_last_error). */
ShimRun* shim_run_elf_start(ShimElfKernel*, ShimBo* const* bos, size_t n_bos);

/* --- Persistent-hw_context full-ELF path (kills per-token re-registration) -----------------------
 * The standard shim_elf_kernel_load above rebuilds the WHOLE hw_context per patched ELF — on aie2p
 * that re-runs the NPU partition config (~20 ms/token, measured). This split builds the hw_context
 * ONCE from a base ELF (shim_elf_ctx_open), then per token rebuilds only a lightweight xrt::module
 * (patched bytes) + xrt::ext::kernel bound to the SAME context (shim_elf_kernel_rebind). The ctrlcode
 * rides the ERT run packet, so a same-shape patched module reuses the resident partition. If the
 * 20 ms is partition config (not ctrlcode upload) this removes most of it. The context must outlive
 * every kernel rebound onto it. ShimElfKernel2 borrows the ctx; close it before the ctx. */
typedef struct ShimElfCtx     ShimElfCtx;
typedef struct ShimElfKernel2 ShimElfKernel2;
ShimElfCtx*     shim_elf_ctx_open(ShimDevice*, const void* base_elf, size_t nbytes);
void            shim_elf_ctx_close(ShimElfCtx*);
ShimElfKernel2* shim_elf_kernel_rebind(ShimElfCtx*, const void* elf_bytes, size_t nbytes,
                                       const char* kernel_name);
void            shim_elf_kernel2_close(ShimElfKernel2*);
int             shim_run_elf2(ShimElfKernel2*, ShimBo* const* bos, size_t n_bos);

/* --- Resident full-ELF runner with ctrl-scratchpad parameters (Option C: kills re-registration) ---
 * Registers a CONSTANT full ELF ONCE (elf -> hw_context -> ext::kernel -> run), binds the arena BOs
 * once, and fetches the ELF's ctrl scratchpad BO (xrt::run::get_ctrl_scratchpad_bo, valid only if the
 * ELF was built with aiex.scratchpad_parameter). Per dispatch the host writes the per-token parameter
 * word(s) (e.g. KV-write offset, softmax mask length) into the mapped scratchpad and dispatches — no
 * ELF patch, no re-registration. `shim_elf_resident_open` returns NULL if the ELF has no scratchpad
 * (so the caller can fall back). scratchpad_size()==0 likewise signals no scratchpad. */
typedef struct ShimElfResident ShimElfResident;
ShimElfResident* shim_elf_resident_open(ShimDevice*, const void* elf_bytes, size_t nbytes,
                                        const char* kernel_name);
void             shim_elf_resident_close(ShimElfResident*);
size_t           shim_elf_resident_scratchpad_size(ShimElfResident*);
/* Bind the N arena BOs to run args 0..N once (same handles reused every dispatch). 0 on success. */
int              shim_elf_resident_bind(ShimElfResident*, ShimBo* const* bos, size_t n_bos);
/* Write `len` bytes at byte `offset` into the host-mapped ctrl scratchpad (no device sync yet). */
int              shim_elf_resident_write(ShimElfResident*, size_t offset, const void* data, size_t len);
/* Sync the scratchpad to device, start the bound run, wait for completion. 0 on success. */
int              shim_elf_resident_dispatch(ShimElfResident*);

const char* shim_last_error(void);

#ifdef __cplusplus
}
#endif
