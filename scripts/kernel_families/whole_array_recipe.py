#!/usr/bin/env python3
"""Map a whole_array artifact stem to the Makefile invocation that builds it."""

import hashlib
import json
from pathlib import Path
import re
import sys


STEM_RE = re.compile(r"^(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)c(?:_(.+))?$")
MODAL_MODES = {"silu": None, "id": "no_silu=1", "gelu": "gelu=1"}


def resolve(stem):
    match = STEM_RE.match(stem)
    if not match:
        raise ValueError(f"{stem!r} does not match <M>x<K>x<N>_<m>x<k>x<n>_<C>c[_<variant>]")
    m, k, n, tm, tk, tn, cols, variant = match.groups()
    M, K, N, mt, kt, nt, C = map(int, (m, k, n, tm, tk, tn, cols))
    if min(M, K, N, mt, kt, nt, C) <= 0 or M % (mt * 4) or K % kt or N % (nt * C):
        raise ValueError(f"invalid M/K/N tile grid in {stem!r}: require M%(4*m), K%k, N%(n*C) == 0")
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
    tail = re.match(r"^(bf16out)?(panel(\d+))?(apanel(\d+))?$", rest)
    if not tail or tail.group(0) != rest:
        raise ValueError(f"{stem!r}: unrecognized trailing tokens {rest!r} in variant {variant!r}")
    bf16out, panel_width, a_panel_width = bool(tail.group(1)), tail.group(3), tail.group(5)
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
    if a_panel_width:
        variables.append(f"a_panel_width={a_panel_width}")
    return makefile, variables, f"build/final_{stem}.xclbin"


def record_native_contract(stem, build_dir):
    """Bind the emitted runtime operand extents to the built native artifact bytes."""
    resolve(stem)
    match = STEM_RE.fullmatch(stem)
    M, K, N, mt, kt, nt, cols = map(int, match.groups()[:7])
    if match.group(8) not in ("modalidnat", "modalsilunat") or (mt, kt, nt, cols) != (32, 32, 32, 8):
        raise ValueError("native ctxA contract requires the 32x32x32/8c f32-out modal recipe")
    build_dir = Path(build_dir)
    mlir = (build_dir / f"aie_{stem}.mlir").read_text()
    sequences = re.findall(r"aie\.runtime_sequence\s*(?:@\w+)?\s*\((.*?)\)\s*\{", mlir, re.S)
    expected = [(M * K, "bf16"), (K * N, "bf16"), (M * N, "f32")]
    actual = [[(int(size), dtype) for size, dtype in re.findall(r"memref<(\d+)x(bf16|f32)>", seq)] for seq in sequences]
    if actual != [expected]:
        raise ValueError(f"emitted runtime operand shapes {actual} != requested {expected}")
    def digest(name):
        data = (build_dir / name).read_bytes()
        if not data:
            raise ValueError(f"empty artifact {name}")
        return hashlib.sha256(data).hexdigest()
    contract = dict(rows=M, k=K, n=N, tile=[mt, kt, nt], precision="native",
                    xclbin_sha256=digest(f"final_{stem}.xclbin"),
                    insts_sha256=digest(f"insts_{stem}.txt"))
    (build_dir / f"contract_{stem}.json").write_text(json.dumps(contract, indent=2) + "\n")


def main(argv):
    if len(argv) == 4 and argv[1] == "--record-native-contract":
        try:
            record_native_contract(argv[2], argv[3])
        except (ValueError, OSError) as error:
            print(f"[whole_array_recipe] refusing: {error}", file=sys.stderr)
            return 1
        return 0
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
