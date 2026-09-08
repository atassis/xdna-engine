# SPDX-License-Identifier: Apache-2.0
"""Redispatch check: the same callable twice, byte-identical output. Python rail.

Dispatch the SAME callable twice with no input write between the two calls and require the output
buffer to read back byte-identical. No oracle, no tolerance: it separates output staleness, input
staleness and a wrong computation in one shot, and it is the instrument that root-caused
the bug where the FIRST dispatch after a host input write computed on the PREVIOUS input. The
Rust-side twin lives in `rust/npu-probes/src/bin/verify_whisper_decode.rs` (`--redispatch`).

The bug this catches is specific to the FIRST dispatch after a host write: `_sync_inputs()`
(`iron/common/sequence.py`) trusts the coherence map, and a write through the raw `.data` handle
does not update it (see `redispatch_check` module in bisect/verify scripts, and A5's conversion of
this harness's own `np.copyto(t.data, ...)` sites to `mutate()`/`overwrite()`). A second dispatch
with untouched inputs is a control: if the FIRST call already raced, this check runs past it and
would not see it. Call it after at least one prior dispatch through the same buffers, or run it
twice back to back and require both pairs to agree.
"""
import numpy as np


def assert_redispatch_identical(callable_, output_buf, label="", vocab=None):
    """Dispatch `callable_` twice with no write to any input buffer between the calls, and
    require `output_buf.data` to read back byte-identical both times.

    `output_buf` is whatever `callable_.get_buffer(name)` returned for the buffer to compare
    (e.g. logits). `vocab`, if given, truncates the comparison to `output_buf.data[:vocab]` --
    the buffer is padded, and the padding is not part of the model's output.

    Raises AssertionError naming the mismatch count on failure. Returns nothing on success (the
    caller decides whether to print).
    """
    callable_()
    d1 = np.array(output_buf.data[:vocab] if vocab else output_buf.data, copy=True)
    callable_()
    d2 = np.array(output_buf.data[:vocab] if vocab else output_buf.data, copy=True)
    if not np.array_equal(d1, d2):
        mism = int(np.count_nonzero(d1 != d2))
        raise AssertionError(
            f"REDISPATCH FAIL{f' [{label}]' if label else ''}: {mism}/{d1.size} output elements "
            "differ between two dispatches with no input write between them -- input or output "
            "staleness (see first-dispatch-after-a-host-input-write-computes-on-the-previous-input)"
        )
    print(f"REDISPATCH PASS{f' [{label}]' if label else ''}: {d1.size} elements byte-identical "
          "across 2 dispatches with no input write between them")
