# SPDX-License-Identifier: Apache-2.0
"""Engine-owned fused-ELF dispatch helpers -- relocated off ``iron.common.fusion``.

Upstream IRON renamed the build half (``FusedMLIROperator`` -> ``iron.common.sequence.
OperatorSequence``, amd/IRON #117) and natively replaced ELF-patching with runtime
ctrl-scratchpad parameters (#131), deleting ``iron/common/fusion.py``. We re-grafted that file
onto the fork every refresh; this module ends the carry. The build half now comes from upstream;
the ELF-read/patch host driver below stays here because it is the engine's own generator and
verify infrastructure -- the SHIPPED decode runtime is Rust (``npu-xrt::ElfResident`` +
``write_scratchpad``), which does not use it.

``OperatorSequence`` is ctor-compatible with the old class (name, runlist, input_args,
output_args, buffer_sizes) plus ``dispatch="auto"``, which selects the single-fused-ELF path on
NPU2. The RUNTIME half is not a drop-in: upstream's ``SequenceFullELFCallable`` takes an op and
loads the ELF from a path, so it has no ``reload_elf`` -- per-step ELF patching is exactly what
#131 replaced. Harnesses that patch (``verify_fused_decode.py``) therefore construct
``FusedFullELFCallable`` from here rather than calling ``op.get_callable()``.

The two callable classes below are copied VERBATIM from ``iron/common/fusion.py`` at
integration-stack 798da79, so they keep the fixes made there after the carry was written:
bd121e3 (sub-view parent link + forced output sync), e48befe (build sub-views with the runtime's
``subview()``), 4dff5d0 (space arena buffers by the coherence granule) and e49245f (flush the
output arena before dispatch).
"""

import newstack_compat  # noqa: F401 -- MUST precede iron imports (new-mlir-aie port shim)

import ctypes

import numpy as np
import ml_dtypes
import pyxrt

from iron.common import compilation as comp
# XRTSubBuffer is the pre-subview() fallback below and upstream DELETED it (IRON 1838f82, "drop
# XRTSubBuffer fork, use upstream Tensor.subview"). Import it optionally: the primary path is
# main_buffer.subview(), which iron.common.sequence itself uses, so a current IRON needs no
# fallback and a hard import would break every generator that reaches this module.
try:
    from iron.common.utils import XRTSubBuffer
except ImportError:  # current IRON: subview() is the only path
    XRTSubBuffer = None
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

# The build half, upstream's own (amd/IRON #117). No fallback to the pre-rename
# FusedMLIROperator: the carried fusion.py is deleted, so a missing sequence.py means the build
# is pointed at a pre-#117 IRON line (xdna2-asr) and must fail loud rather than route elsewhere.
from iron.common.sequence import OperatorSequence  # noqa: F401

__all__ = [
    "OperatorSequence",
    "load_elf",
    "patch_elf",
    "FullELFCallable",
    "FusedFullELFCallable",
]


def load_elf(op):
    assert isinstance(op.artifacts[0], comp.FullElfArtifact)
    elf_data = None
    with open(op.artifacts[0].filename, "rb") as f:
        elf_data = np.frombuffer(f.read(), dtype=np.uint32)
    return elf_data


def patch_elf(elf_data, patches):
    for i, patch in patches.items():
        val, mask = patch
        val = np.uint64(val)
        mask = np.uint64(mask)  # avoid numpy overflow errors
        elf_data[i] = np.uint32((elf_data[i] & ~mask) | (val & mask))
    return elf_data


class FullELFCallable:
    def __init__(
        self,
        elf_data,
        device_name="main",
        sequence_name="sequence",
    ):
        self.device_name = device_name
        self.sequence_name = sequence_name
        self.reload_elf(elf_data)

    def __call__(self, *args):
        run = pyxrt.run(self.xrt_kernel)
        for i, arg in enumerate(args):
            assert isinstance(arg, pyxrt.bo), f"Argument {i} is not a pyxrt.bo"
            run.set_arg(i, arg)
        run.start()
        ret_code = run.wait()
        if ret_code != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"Kernel execution failed with return code {ret_code}")

    def reload_elf(self, elf_data):
        # Create a PyCapsule from the numpy array pointer for pybind11
        elf_data_u8 = elf_data.view(dtype=np.uint8)
        ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
        ctypes.pythonapi.PyCapsule_New.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        capsule = ctypes.pythonapi.PyCapsule_New(elf_data_u8.ctypes.data, None, None)
        xrt_elf = pyxrt.elf(capsule, elf_data.nbytes)
        xrt_context = pyxrt.hw_context(aie_utils.DefaultNPURuntime._device, xrt_elf)
        self.xrt_kernel = pyxrt.ext.kernel(
            xrt_context, f"{self.device_name}:{self.sequence_name}"
        )


class FusedFullELFCallable(FullELFCallable):
    def __init__(self, op, elf_data=None):
        if elf_data is None:
            elf_data = load_elf(op)
        super().__init__(elf_data)

        self.op = op
        input_buffer_size, output_buffer_size, scratch_buffer_size = op.buffer_sizes
        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize

        self.input_buffer = XRTTensor(
            (max(input_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )

        self.output_buffer = XRTTensor(
            (max(output_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )

        self.scratch_buffer = XRTTensor(
            (max(scratch_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )

        self._buffer_cache = {}

    def get_buffer(self, buffer_name):
        # Return cached buffer if already allocated
        if buffer_name in self._buffer_cache:
            return self._buffer_cache[buffer_name]

        buf_type, offset, length = self.op.get_layout_for_buffer(buffer_name)

        # Select the appropriate main buffer
        if buf_type == "input":
            main_buffer = self.input_buffer
        elif buf_type == "output":
            main_buffer = self.output_buffer
        elif buf_type == "scratch":
            main_buffer = self.scratch_buffer
        else:
            raise ValueError(
                f"Unknown buffer type '{buf_type}' for buffer '{buffer_name}'"
            )

        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
        shape = (length // itemsize,)
        try:
            # Preferred: the runtime's own sub-region view. It tracks residency in the
            # coherence map shared with the parent, which is what our `parent=` link was
            # emulating (issue Xilinx/mlir-aie#3420), and it refuses views that share a
            # 64-byte granule instead of letting them silently clobber each other's syncs.
            sub_buffer = main_buffer.subview(offset, shape, ml_dtypes.bfloat16)
        except (AttributeError, NotImplementedError):
            # Backends predating hostruntime subview(); XRTSubBuffer needs the parent link
            # explicitly or the parent stays "npu" from allocation, the sync no-ops, and
            # kernels read init-zeros.
            if XRTSubBuffer is None:
                raise RuntimeError(
                    f"{buffer_name}: this IRON has neither Tensor.subview() nor the deleted "
                    "XRTSubBuffer fallback -- the buffer cannot be sub-viewed at all"
                )
            sub_buffer = XRTSubBuffer(
                parent_bo=main_buffer.buffer_object(),
                offset_bytes=offset,
                size_bytes=length,
                shape=shape,
                dtype=ml_dtypes.bfloat16,
                parent=main_buffer,
            )

        self._buffer_cache[buffer_name] = sub_buffer
        return sub_buffer

    def _sync_inputs(self):
        # Sub-views handed out by get_buffer() mark this parent host-dirty on .data access
        # (XRTSubBuffer.data), so `to("npu")` here actually fires the host->device sync for
        # the freshly written inputs. Mirrors OperatorSequence._sync_inputs.
        #
        # The residency assert is deliberate belt-and-braces: .data is the only write hook,
        # so a caller that grabs `buf = get_buffer(x).data` ONCE and then writes into that
        # ndarray in a loop never re-marks the parent, and the guarded `.to()` would no-op
        # on every later dispatch. Host is always the authority for inputs here, and this
        # buffer is small (scratch -- weights/KV -- is synced separately and is untouched),
        # so forcing the sync costs nothing and removes a silent-staleness class.
        self.input_buffer.device = "cpu"
        self.input_buffer.to("npu")
        # The output arena too, and NOT for its contents: a caller that pre-fills or clears
        # it through `get_buffer(...).data` (an unreconciled write -- see NpuTensor.data)
        # leaves DIRTY host cache lines over the region the DMA is about to write. The
        # device-to-host sync afterwards does not discard them, so those lines shadow the
        # device's output and the caller reads back its own pre-fill. MEASURED: with a
        # sentinel pre-fill, 1024/1024 elements read the sentinel and none read the device's
        # bytes; with this flush, 1024/1024 exact. Flushing writes the dirty lines out, so
        # none outlive the dispatch. Costs one sync of a small arena.
        #
        # Scratch is deliberately not flushed here: it carries the weights and KV, is synced
        # by whoever loads it, and is large enough that an unconditional flush per dispatch
        # would cost real time. A caller that pre-writes scratch through `.data` has the same
        # hazard and must flush it itself.
        self.output_buffer.device = "cpu"
        self.output_buffer.to("npu")

    def _sync_outputs(self):
        # _run just rewrote the output arena on the device, so the device holds the
        # authoritative copy. Force the device->host sync: assert device residency first
        # so `to("cpu")` fires even if a prior read of get_buffer(...).data marked the
        # buffer "cpu" (otherwise a looped dispatch would read stale output).
        # Mirrors OperatorSequence._sync_outputs.
        self.output_buffer.device = "npu"
        self.output_buffer.to("cpu")

    def __call__(self):
        self._sync_inputs()
        super().__call__(
            self.input_buffer.buffer_object(),
            self.output_buffer.buffer_object(),
            self.scratch_buffer.buffer_object(),
        )
        self._sync_outputs()
