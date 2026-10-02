#!/usr/bin/env python3
"""Map a whole_array artifact stem to the Makefile invocation that builds it."""

import re
import sys


STEM_RE = re.compile(r"^(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)c(?:_(.+))?$")
MODAL_MODES = {"silu": None, "id": "no_silu=1", "gelu": "gelu=1"}


def resolve(stem):
    match = STEM_RE.match(stem)
    if not match:
        raise ValueError(f"{stem!r} does not match <M>x<K>x<N>_<m>x<k>x<n>_<C>c[_<variant>]")
    m, k, n, tm, tk, tn, cols, variant = match.groups()
    dims = {"M": m, "K": k, "N": n}
    tile = {"m": tm, "k": tk, "n": tn}

    if variant is None:
        if k == "1024":
            makefile = "Makefile.resident"
            fast = tile == {"m": "64", "k": "32", "n": "128"}
            variables = [f"{key}={value}" for key, value in {**dims, **tile}.items()]
            variables += [f"n_aie_cols={cols}", "dtype_in=bf16", "dtype_out=f32", "use_iron=1"]
            if fast:
                variables = ["WA_C_DEPTH=1", *variables, "emulate_bfloat16_mmul_with_bfp16=1", "bfp16_iree=1"]
            return makefile, variables, f"build/final_{stem}.xclbin"
        variables = [f"{key}={value}" for key, value in dims.items()]
        if tile != {"m": "32", "k": "32", "n": "32"}:
            variables += [f"{key}={value}" for key, value in tile.items()]
        variables += ["dtype_in=bf16", "dtype_out=f32", f"n_aie_cols={cols}", "use_iron=1"]
        return None, variables, f"build/final_{stem}.xclbin"

    rest = variant
    if rest.startswith("modalint8dq"):
        makefile, mode_flag, fast = "Makefile.modal.int8", None, False
        rest = rest[len("modalint8dq") :]
    elif rest.startswith("modal"):
        rest = rest[len("modal") :]
        for name, flag in MODAL_MODES.items():
            if rest.startswith(name):
                mode_flag, rest = flag, rest[len(name) :]
                break
        else:
            raise ValueError(f"{stem!r}: unknown modal mode in variant {variant!r}")
        makefile, fast = "Makefile.modal", True
    elif rest == "bias":
        makefile, mode_flag, fast, rest = "Makefile.silu", "no_silu=1", False, ""
    elif rest == "silu":
        makefile, mode_flag, fast, rest = "Makefile.silu", None, False, ""
    else:
        raise ValueError(f"{stem!r}: unrecognized variant {variant!r}")

    k_loop_rtp = rest.startswith("krtp")
    if k_loop_rtp:
        rest = rest[len("krtp") :]
    native = rest.endswith("nat")
    if native:
        rest = rest[: -len("nat")]
    tail = re.match(r"^(bf16out)?(panel(\d+))?$", rest)
    if not tail or tail.group(0) != rest:
        raise ValueError(f"{stem!r}: unrecognized trailing tokens {rest!r} in variant {variant!r}")
    bf16out, panel_width = bool(tail.group(1)), tail.group(3)
    if native:
        fast = False

    variables = [f"{key}={value}" for key, value in {**dims, **tile}.items()]
    variables.append(f"n_aie_cols={cols}")
    if fast:
        if not bf16out:
            variables = ["WA_C_DEPTH=1", *variables]
        variables += ["emulate_bfloat16_mmul_with_bfp16=1", "bfp16_iree=1"]
    if mode_flag:
        variables.append(mode_flag)
    if k_loop_rtp:
        variables.append("k_loop_rtp=1")
    if bf16out:
        variables.append("dtype_out=bf16")
    if panel_width:
        variables.append(f"c_panel_width={panel_width}")
    return makefile, variables, f"build/final_{stem}.xclbin"


def main(argv):
    if len(argv) != 2:
        print("usage: whole_array_recipe.py <stem>", file=sys.stderr)
        return 2
    try:
        makefile, variables, target = resolve(argv[1])
    except ValueError as error:
        print(f"[whole_array_recipe] refusing: {error}", file=sys.stderr)
        return 1
    print(makefile or "")
    print(" ".join(variables))
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
