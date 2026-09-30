"""Build a generated resident-forward design to a full ELF:
`rf_build.py <name> <generator module> <generator args...>`; the kernel object is compiled with the
generator's KERN_DEFS.

Kernels compile through IRON's compile_cxx_core_function (intrinsics PCH, ccache launcher), with
kcc.sh's flags. RF_DUMP=1 passes --dump-intermediates; RF_KCC=1 compiles with kcc.sh instead."""
import importlib, logging, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_paths
sys.path.insert(0, rf_paths.iron_dir())
os.environ.setdefault("AIE_KERNEL_COMPILER_LAUNCHER", "ccache")
os.environ.setdefault("CCACHE_SLOPPINESS", "time_macros")
from aie.utils.compile.utils import compile_mlir_module, compile_cxx_core_function
from kobj_cache import kobj_key, cached_compile, place

KCC = os.environ.get("RF_KCC_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kcc.sh"))
SRC = rf_paths.iron_kernel("chain_mm_bfp16.cc")
LLD = os.environ.get("RF_LLD", os.path.join(
    os.environ.get("PEANO_INSTALL_DIR", str(rf_paths.REPO / ".venv-iron/lib/python3.14/site-packages/llvm-aie")),
    "bin", "ld.lld"))


class _CacheLog(logging.Handler):
    hit = False

    def emit(self, rec):
        msg = rec.getMessage()
        if msg.startswith("aiecc cache"):
            _CacheLog.hit |= msg.startswith("aiecc cache hit")
            print(msg, flush=True)


def kcc(src, out, defs):
    if os.environ.get("RF_KCC"):
        subprocess.run([KCC, src, out, *defs], check=True)
    else:
        compile_cxx_core_function(src, "aie2p", out, compile_args=list(defs))


def main():
    name, gen = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    lg = logging.getLogger("aie.utils.compile.utils")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(_CacheLog())
    G = importlib.import_module(gen)
    work = Path(os.environ.get("RF_BUILD", str(rf_paths.BUILD_ROOT))) / name
    work.mkdir(parents=True, exist_ok=True)
    text = G.build_text(args)          # first: build flags can select kernel objects
    (work / "gen_args.txt").write_text(" ".join([gen] + args) + "\n")
    (work / "gen_env.txt").write_text("".join(f"{k}={v}\n" for k, v in sorted(os.environ.items()) if k.startswith("RF_")))
    for fname, body in getattr(G, "extra_files", lambda: {})().items():
        (work / fname).write_text(body)
    run_dir = os.environ.get("RF_BUILD", str(rf_paths.BUILD_ROOT))
    for obj, src, defs in G.kernels():
        dest = work / obj
        if defs is None:          # a prebuilt object, used as is
            shutil.copy(src, dest)
        elif isinstance(src, list):     # several sources, partially linked into one object
            def build_link(tmp, src=src):
                parts = []
                for i, (s_, d_) in enumerate(src):
                    parts.append(f"{tmp}.part{i}.o")
                    kcc(s_, parts[-1], d_)
                subprocess.run([LLD, "-r", "-o", tmp, *parts], check=True)
                for p in parts:
                    os.unlink(p)
            place(cached_compile(run_dir, kobj_key(src), build_link), dest)
        else:
            place(cached_compile(run_dir, kobj_key([(src, defs)]),
                                  lambda tmp, src=src, defs=defs: kcc(src, tmp, defs)), dest)
    extra = getattr(G, "aiecc_options", lambda: [])()
    (work / "design.mlir").write_text(text)
    compile_mlir_module(text, full_elf_path=str(work / "design.elf"), work_dir=str(work),
                        options=[f"-j{os.environ.get('AIECC_JOBS', '8')}"] + extra
                        + (["--dump-intermediates"] if os.environ.get("RF_DUMP") else [])
                        + ([f"-O{os.environ['RF_AIECC_O']}"] if os.environ.get("RF_AIECC_O") else []))
    if not _CacheLog.hit:
        print("aiecc cache miss", flush=True)
    print("BUILT", work / "design.elf")


main()
