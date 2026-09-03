# SPDX-License-Identifier: Apache-2.0
"""Import gate for the decode generators on the CURRENT IRON pin.

gen_decode.py and gen_projout.py imported `iron.common.fusion` for ten days after the
fork deleted it, so neither fused_decode12 nor projout_elf could be built and the
breakage was invisible until someone tried to rebuild. This test is that trip-wire.

Run inside the IRON env:
  PYTHONPATH=route_b_kernels/decode_fused:$IRON .venv-iron/bin/python -m pytest \
      route_b_kernels/decode_fused/test_generator_imports.py -v
"""
import glob
import importlib
import os

import pytest

# DISCOVERED, not listed. The thing this gate exists to catch is a generator drifting off the
# current IRON API, and a hardcoded list drifts exactly the same way: gen_llm_decode was added
# after the first port and sat on the deleted driver until 2026-09-03 because no list named it.
# Every gen_*.py plus the verify entry points, found at collection time.
_HERE = os.path.dirname(os.path.abspath(__file__))
IN_SCOPE = sorted(
    [os.path.basename(p)[:-3] for p in glob.glob(os.path.join(_HERE, "gen_*.py"))]
    + ["verify_fused_decode", "verify_fused_decode_sp"]
)


@pytest.mark.parametrize("mod", IN_SCOPE)
def test_generator_imports(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", IN_SCOPE)
def test_generator_does_not_import_deleted_fusion_module(mod):
    src = open(os.path.join(_HERE, f"{mod}.py")).read()
    assert "iron.common.fusion" not in src, (
        f"{mod}.py imports iron.common.fusion, which IRON integration-stack deleted; "
        "use newstack_compat.read_full_elf / iron.common.sequence.OperatorSequence"
    )


def test_shim_exports_the_replacement_helpers():
    """The ELF helpers the deleted `iron.common.fusion` used to provide must exist SOMEWHERE the
    generators can import, or the port is only half done and every one of them fails at build
    rather than at import -- which is later and much more expensive.

    They live in `elf_dispatch_compat`, engine-owned and copied verbatim off the deleted module
    (see its header). `newstack_compat` is the API-drift shim and does not carry them.
    """
    import elf_dispatch_compat

    assert callable(elf_dispatch_compat.load_elf)
    assert callable(elf_dispatch_compat.patch_elf)
