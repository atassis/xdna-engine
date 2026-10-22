#!/usr/bin/env python3
"""Dispatch the configured model-artifact recipes used by install.sh.

The plan is deliberately limited to engine.toml's selected models. It records
actual build/export commands and their declared inputs, but performs no device,
service, package, or toolchain action on its own.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from check_iron_artifact_provenance import validate as validate_iron_provenance
from lib.data_root import XDNA_DATA
from pyannote_local_inputs import PipelineComponent, pipeline_components, validate_local_pipeline_inputs
from model_artifact_stages import Stage, StageRunner, View, digest as stage_digest, rebase_stage, source_identity


LOCK_ROOT = XDNA_DATA / "locks/model-artifacts"


@contextmanager
def resource_locks(resources):
    """Stable coordination inodes; shared ancestors allow disjoint targets to proceed."""
    modes = {}
    for raw in resources:
        raw = Path(raw)
        target = raw.parent.resolve() / raw.name
        if LOCK_ROOT.resolve().is_relative_to(target):
            raise ValueError(f"recipe resource contains the coordination locks: {target}")
        modes[target] = fcntl.LOCK_EX
        for parent in target.parents:
            modes.setdefault(parent, fcntl.LOCK_SH)
    LOCK_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    with ExitStack() as stack:
        for target, mode in sorted(modes.items(), key=lambda item: str(item[0])):
            name = hashlib.sha256(os.fsencode(target)).hexdigest() + ".lock"
            handle = stack.enter_context((LOCK_ROOT / name).open("a+b"))
            fcntl.flock(handle, mode)
        yield


def require_no_recovery(rollback: Path, key: str) -> None:
    if (rollback / key).exists():
        raise RuntimeError(f"publication recovery requires inspection before retry: {rollback / key}")


@dataclass(frozen=True)
class Command:
    step: str
    argv: tuple[str, ...]
    env: dict[str, str]
    stage: Stage | None = None


@dataclass(frozen=True)
class Recipe:
    model: str
    scenario: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    commands: tuple[Command, ...]


def path(root: Path, value: str) -> str:
    return str(root / value)


def command(step: str, argv: list[str], **env: str) -> Command:
    return Command(step, tuple(argv), dict(sorted(env.items())))


def configured_models(config: Path) -> list[dict]:
    data = tomllib.loads(config.read_text())
    models = data.get("model", [])
    if not isinstance(models, list) or not models:
        raise ValueError(f"no [[model]] entries in {config}")
    selected = []
    for entry in models:
        name, scenario = entry.get("name"), entry.get("scenario")
        if not isinstance(name, str) or not isinstance(scenario, str):
            raise ValueError(f"each [[model]] in {config} needs string name and scenario")
        selected.append({"name": name, "scenario": scenario})
    return selected


PYANNOTE_SDKS = {
    "pyannote-3.1": ("legacy", ("pyannote-audio", "torch", "torchaudio", "huggingface-hub", "onnx", "PyYAML")),
    "pyannote-community-1": ("community", ("pyannote-audio", "torch", "torchaudio", "torchcodec", "onnx", "numpy", "PyYAML")),
}


def pyannote_sdk_for_model(model: str) -> str:
    return PYANNOTE_SDKS[model][0]


def configured_pyannote_sdks(config: Path) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        pyannote_sdk_for_model(entry["name"])
        for entry in configured_models(config)
        if entry["name"] in PYANNOTE_SDKS
    ))


def recipe_for(model: str, scenario: str, repo: Path, artifacts: Path, inputs: Path, build: Path) -> Recipe:
    return declare_stages(_recipe_for(model, scenario, repo, artifacts, inputs, build), repo, build)


def _recipe_for(model: str, scenario: str, repo: Path, artifacts: Path, inputs: Path, build: Path) -> Recipe:
    # CLS changes pooling/scheduling, not BGE's exported model data or its receipt.
    if model == "bge-base-cls" and scenario == "scenarios/bge-base-cls.toml":
        return _recipe_for("bge-base", "scenarios/bge-base.toml", repo, artifacts, inputs, build)
    py = sys.executable
    export_py = os.environ.get("MODEL_EXPORT_PY", sys.executable)
    iron_py = os.environ.get("MODEL_IRON_PY", str(repo / ".venv-iron/bin/python"))
    pyannote_py = os.environ.get("MODEL_PYANNOTE_PY", str(repo / ".venv-pyannote/bin/python"))
    pyannote_community_py = os.environ.get(
        "MODEL_PYANNOTE_COMMUNITY_PY", str(repo / ".venv-pyannote-community/bin/python"))
    npu = os.environ.get("NPU_BIN", "npu")
    scripts = repo / "scripts"
    weight_env = {"IRON_DIR": str(repo / "third_party/iron"), "PYTHONPATH": str(repo / "third_party/iron")}
    out = lambda rel: path(artifacts, rel)
    src = lambda rel: path(inputs, rel)

    if model == "parakeet" and scenario == "scenarios/asr.toml":
        return Recipe(
            model, scenario, (src("parakeet"),), (out("parakeet"),),
            (command("export", [export_py, str(scripts / "extract_parakeet_encoder.py")],
                     PARAKEET_SOURCE=src("parakeet"), PARAKEET_OUT=out("parakeet")),),
        )
    if model == "bge-base" and scenario == "scenarios/bge-base.toml":
        return Recipe(
            model, scenario, (src("bge-base"),), (out("bge-base"),),
            (command("export", [export_py, str(scripts / "export_bge.py")],
                     BGE_SOURCE=src("bge-base"), BGE_OUT=out("bge-base"), HF_HUB_OFFLINE="1"),),
        )
    if model in {"pyannote-community-1", "pyannote-3.1"}:
        expected = {
            "pyannote-community-1": "scenarios/diarize-pyannote-community-1.toml",
            "pyannote-3.1": "scenarios/diarize-pyannote-3.1.toml",
        }[model]
        if scenario != expected:
            raise ValueError(f"{model} is bound to {expected}, not {scenario}")
        slug = "speaker-diarization-community-1" if model == "pyannote-community-1" else "speaker-diarization-3.1"
        sdk_python = pyannote_community_py if model == "pyannote-community-1" else pyannote_py
        pipeline = src(f"pyannote/{slug}")
        return Recipe(
            model, scenario, (pipeline,), (out(f"pyannote/{slug}"),),
            (
                command("validate", [sdk_python, str(scripts / "verify_pyannote_local_load.py"), pipeline],
                        HF_HUB_OFFLINE="1"),
                command("export", [sdk_python, str(scripts / "export_pyannote.py")],
                        PYANNOTE_PIPELINE=pipeline, PYANNOTE_OUT=out(f"pyannote/{slug}"), HF_HUB_OFFLINE="1"),
            ),
        )
    if model == "whisper-turbo" and scenario == "scenarios/asr-whisper-turbo.toml":
        whisper = out("whisper-turbo")
        return Recipe(
            model, scenario, (src("whisper-turbo"),), (whisper,),
            (
                command("encoder", [export_py, str(scripts / "extract_whisper_encoder.py")],
                        WHISPER_MODEL=src("whisper-turbo"), WHISPER_OUT=whisper, HF_HUB_OFFLINE="1"),
                command("decode", ["bash", str(scripts / "build_whisper_rail.sh"), "whisper-turbo", f"{whisper}/decode_rail"],
                        WEIGHTS=f"{whisper}/weights", VENV_IRON=str(Path(iron_py).parent.parent),
                        WHISPER_CHECKPOINT_DIR=src("whisper-turbo")),
                command("serving-companions", [export_py, str(scripts / "whisper_serving_companions.py"),
                        "export", "--checkpoint", src("whisper-turbo"), "--out", whisper],
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
            ),
        )
    if model == "gemma3-270m" and scenario == "scenarios/generate-gemma3-270m.toml":
        weights, decode, tokenizer = out("gemma3-270m/weights_qat_int4g32"), out("gemma3-270m/decode_qat_int4_pe8e957d"), out("gemma3-270m/tokenizer_qat")
        return Recipe(
            model, scenario, (src("gemma3-270m/checkpoint"),), (decode, weights, tokenizer),
            (
                command("weights", [iron_py, str(scripts / "dump_llm_weights.py"), "--spec", model, "--checkpoint-dir", src("gemma3-270m/checkpoint"), "--out", weights, "--quant", "int4", "--quant-group", "32", "--quant-full-range", "--quant-leaves", "gate_proj,up_proj,down_proj,o_proj,embed_tokens"],
                        **weight_env),
                command("decode", ["bash", str(scripts / "build_llm_decode.sh"), model, "", decode], WEIGHTS=weights),
                command("tokenizer", [py, str(scripts / "model_artifact_dispatch.py"), "--copy-tokenizer", src("gemma3-270m/checkpoint"), tokenizer]),
            ),
        )
    if model == "qwen3-0.6b" and scenario == "scenarios/generate-qwen3-0.6b.toml":
        weights, decode, prefill, tokenizer = out("qwen3-0.6b/weights"), out("qwen3-0.6b/decode_p8e3958"), out("qwen3-0.6b/prefill_p8e3958"), out("qwen3-0.6b/tokenizer")
        return Recipe(
            model, scenario, (src("qwen3-0.6b/checkpoint"),), (decode, prefill, weights, tokenizer),
            (
                command("weights", [iron_py, str(scripts / "dump_llm_weights.py"), "--spec", model, "--checkpoint-dir", src("qwen3-0.6b/checkpoint"), "--out", weights], **weight_env),
                command("decode", ["bash", str(scripts / "build_llm_decode.sh"), model, "", decode], WEIGHTS=weights, WINDOW_RUNGS="256,512,1024,2048", KV_BLOCK_T="128", GEN_EXTRA="--max-seq 4096"),
                command("prefill", ["bash", str(scripts / "build_prefill.sh"), "28", "256", "4096", prefill], WEIGHTS=weights, SPEC=model, DECODE_META=f"{decode}/meta.json"),
                command("tokenizer", [py, str(scripts / "model_artifact_dispatch.py"), "--copy-tokenizer", src("qwen3-0.6b/checkpoint"), tokenizer]),
            ),
        )
    if model == "gemma4-12b" and scenario == "scenarios/generate-gemma4-12b-resident-256k.toml":
        resident, tokenizer, checkpoint = out("gemma4-12b/resident_rf48C_p7148a7"), out("gemma4-12b/tokenizer"), out("gemma4-12b-qat/checkpoint")
        data_outputs = (
            out("gemma4-12b/rf_stack"),
            out("gemma4-12b/store"),
            out("gemma4-12b/weights_int4g32sbf16_planar_qat_rg"),
            out("gemma4-12b/hf_config"),
            out("gemma4-12b/towers_qat"),
        )
        rf_build = path(build, "gemma4-12b/rf48C")
        return Recipe(
            model, scenario, (src("gemma4-12b/checkpoint"),), (resident, tokenizer, checkpoint, *data_outputs),
            (
                command("data", ["bash", str(repo / "designs/resident_forward/recipes/gemma4_data.sh"), "--out", str(artifacts), "--checkpoint-dir", src("gemma4-12b/checkpoint")], **weight_env),
                command("build", ["bash", str(repo / "designs/resident_forward/recipes/rf48C.sh")],
                        RF_BUILD=rf_build, RF_STORE=out("gemma4-12b/store")),
                command("package", [py, str(scripts / "model_artifact_dispatch.py"), "--package-gemma4", "--repo", str(repo), "--artifacts-root", str(artifacts), "--build-dir", rf_build, "--out", resident, "--tokenizer-source", src("gemma4-12b/checkpoint"), "--tokenizer-out", tokenizer]),
            ),
        )
    if model == "qwen3.5-4b" and scenario == "scenarios/generate-qwen3.5-4b.toml":
        weights = path(build, "qwen3.5-4b/weight-dump")
        decode, prefill, tokenizer = out("qwen3.5-4b/decode_s4096_gdrl_p8e3958"), out("qwen3.5-4b/prefill_m256f_s4096_p8e3958"), out("qwen3.5-4b/tokenizer")
        return Recipe(
            model, scenario, (src("qwen3.5-4b/checkpoint"),), (decode, prefill, tokenizer),
            (
                command("weights", [iron_py, str(scripts / "dump_llm_weights.py"), "--spec", model, "--checkpoint-dir", src("qwen3.5-4b/checkpoint"), "--out", weights], **weight_env),
                command("decode", ["bash", str(scripts / "build_llm_decode.sh"), model, "", decode], WEIGHTS=weights, GEN_EXTRA="--max-seq 4096", ACT_POLY="1", DECODE_GDR_LIMBS="1", PRECISION='{"mlp": "int4/g32/clip", "attn_o": "int4/g32/clip", "qkv": "int4/g32/clip"}'),
                command("prefill", ["bash", str(scripts / "build_prefill.sh"), "32", "256", "4096", prefill], WEIGHTS=weights, SPEC=model, NO_GOLDEN="1", PREFILL_SEGMENTS="16", PREFILL_BFP16="0", PREFILL_ACC="1", ACT_POLY="1", PREFILL_ACT_FAST="1", DECODE_META=f"{decode}/meta.json"),
                command("tokenizer", [py, str(scripts / "model_artifact_dispatch.py"), "--copy-tokenizer", src("qwen3.5-4b/checkpoint"), tokenizer]),
            ),
        )
    if model == "espcn" and scenario == "scenarios/upscale-espcn.toml":
        source, output = src("espcn/espcn_x3_dyn.onnx"), out("espcn")
        return Recipe(
            model, scenario, (source,), (output,),
            (
                command("export", [export_py, str(scripts / "export_espcn.py")], ESPCN_MODEL=source, ESPCN_OUT=output),
                command("bake", [npu, "checkpoint", "bake", "--source", f"path:{source}", "--arch", "espcn",
                                 "--checkpoint", f"{output}/espcn.safetensors"]),
            ),
        )
    raise ValueError(f"no model artifact recipe for configured model {model!r} scenario {scenario!r}")


def declare_stages(recipe: Recipe, repo: Path, build: Path) -> Recipe:
    checkpoint = recipe.inputs[0]
    export_py = os.environ.get("MODEL_EXPORT_PY", sys.executable)
    iron_py = os.environ.get("MODEL_IRON_PY", str(repo / ".venv-iron/bin/python"))
    compiler_runtime = {"IRON": str(repo / "third_party/iron"), "IRON_DIR": str(repo / "third_party/iron"),
                        "VENV_IRON": str(Path(iron_py).parent.parent),
                        "PYTHONPATH": str(repo / "third_party/iron"), "TOOLCHAIN_RESOLVE_ONLY": "1"}
    compiler_env = ("IRON*", "VENV_IRON", "PEANO_INSTALL_DIR", "AIE*", "MODEL_AIECC",
                    "RF_*", "PREFILL_*", "DECODE_*", "GEN_EXTRA", "ACT_POLY", "PRECISION",
                    "WINDOW_RUNGS", "KV_BLOCK_T", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    compiler_sources = ("toolchain.lock", "scripts/amd_paths.sh", "scripts/toolchain_up.sh",
                        "scripts/toolchain_gc.sh", "scripts/fast_build_env.sh", "scripts/cache_env.sh",
                        "scripts/require_disk_backed.sh", "scripts/lib", "aie_kernels", "third_party/iron",
                        "third_party/iron/iron/common/test_utils.py")
    weight_sources = ("scripts/dump_llm_weights.py", "designs/decode_fused/llm_decode_spec.py",
                      "third_party/iron/iron/common/quant.py")
    weight_input = View(checkpoint, ("*.safetensors", "*.bin", "config.json", "*.index.json"))
    tokenizer_input = View(checkpoint, ("*.json", "*.jinja", "*.model"))
    rf = "designs/resident_forward/"
    rf_shared = tuple(rf + name + ".py" for name in (
        "bfp16_model", "chain_ref", "head_ref", "kvring", "m1_design", "m2_design", "mlp_ref",
        "o_ref", "rattn2_design", "rattnh", "rf_model", "rf_paths", "rforward", "rglobal", "rhead",
        "rlayer_design", "rmlp_design", "rmlp_ref", "rseg_design", "rsplitl"))
    commands = recipe.commands
    if recipe.model == "gemma4-12b":
        data = commands[0]
        commands = tuple(replace(data, step="data-" + name.replace("_", "-"),
                                 argv=(*data.argv, "--stages", name))
                         for name in ("hf_config", "weights", "towers", "store", "rf_stack")) + commands[1:]
    result = []
    for item in commands:
        step, model = item.step, recipe.model
        if model == "gemma4-12b":
            item = replace(item, env=item.env | {"RF_MODEL": model})
        if model in {"parakeet", "bge-base"}:
            packages = ("numpy", "onnx", "onnxruntime") if model == "parakeet" else ("numpy", "torch", "transformers", "onnx", "tokenizers", "safetensors")
            stage = Stage(inputs=(View(checkpoint),), outputs=(View(recipe.outputs[0]),),
                          sources=(str(Path(item.argv[1]).relative_to(repo)),),
                          sdk_python=export_py, sdk_packages=packages)
        elif model in PYANNOTE_SDKS:
            stage = Stage(inputs=(View(checkpoint),),
                          outputs=() if step == "validate" else (View(recipe.outputs[0]),),
                          dependencies=() if step == "validate" else ("validate",),
                          sources=(str(Path(item.argv[1]).relative_to(repo)), "scripts/pyannote_local_inputs.py"),
                          sdk_python=item.argv[0], sdk_packages=PYANNOTE_SDKS[model][1] +
                          (("numpy",) if model == "pyannote-3.1" and step == "export" else ()) +
                          (("scipy",) if model == "pyannote-community-1" and step == "export" else ()))
        elif model == "whisper-turbo":
            root = recipe.outputs[0]
            if step == "encoder":
                stage = Stage(inputs=(View(checkpoint),),
                              outputs=(View(root, ("config.json", "conv", "refs", "L[0-9]*"), True),),
                              sources=("scripts/extract_whisper_encoder.py", "tests/data/wer_clips/en_01.wav"),
                              sdk_python=export_py, sdk_packages=("numpy", "torch", "transformers", "soundfile", "safetensors"))
            elif step == "decode":
                stage = Stage(inputs=(weight_input,), outputs=(View(root + "/decode_rail"), View(root + "/weights")),
                              sources=(*compiler_sources, *weight_sources, "scripts/build_whisper_rail.sh",
                                       "scripts/build_llm_decode.sh", "designs/decode_fused"),
                              environment=(*compiler_env, "MAX_SEQ"), sdk_python=iron_py, sdk_packages=("numpy", "safetensors"))
                work = str(build / model / "decode-work")
                item = replace(item, env=item.env | compiler_runtime | {"KEEP_WORK": work})
                stage = replace(stage, scratch=(work,), toolchain=True)
            else:
                stage = Stage(inputs=(View(checkpoint),), dependencies=("encoder",),
                              outputs=(View(root, ("preprocessor.onnx*", "onnx", "tokenizer.json", "serving-companions.json"), True),),
                              sources=("scripts/whisper_serving_companions.py", recipe.scenario),
                              sdk_python=export_py, sdk_packages=("numpy", "torch", "transformers", "onnx", "onnxruntime", "optimum", "optimum-onnx", "tokenizers", "safetensors"))
        elif model in {"gemma3-270m", "qwen3-0.6b", "qwen3.5-4b"}:
            if step == "weights":
                target = item.argv[item.argv.index("--out") + 1]
                stage = Stage(inputs=(weight_input,), outputs=(View(target),), sources=weight_sources,
                              sdk_python=iron_py, sdk_packages=("numpy", "safetensors"))
            elif step == "tokenizer":
                stage = Stage(inputs=(tokenizer_input,), outputs=(View(item.argv[-1]),),
                              symbols=("copy_tokenizer", "main"))
            else:
                target = item.argv[-1]
                stage = Stage(dependencies=("weights",) if step == "decode" else ("weights", "decode"),
                              outputs=(View(target),),
                              sources=(*compiler_sources, "scripts/build_llm_decode.sh" if step == "decode" else "scripts/build_prefill.sh",
                                       "designs/decode_fused", "scripts/gate_numeric.py", "scripts/rope_int_phase.py"),
                              environment=compiler_env, sdk_python=iron_py, sdk_packages=("numpy",))
                work = str(build / model / (step + "-work"))
                item = replace(item, env=item.env | compiler_runtime | {"VENV_IRON": str(Path(iron_py).parent.parent), "KEEP_WORK": work})
                stage = replace(stage, scratch=(work,), toolchain=True)
        elif model == "gemma4-12b":
            if step.startswith("data-"):
                root_for = {"hf-config": recipe.outputs[6], "weights": recipe.outputs[5],
                            "towers": recipe.outputs[7], "store": recipe.outputs[4], "rf-stack": recipe.outputs[3]}
                part = step.removeprefix("data-")
                dependencies = {"hf-config": (), "weights": (), "towers": (),
                                "store": ("data-weights", "data-towers"),
                                "rf-stack": ("data-weights", "data-store")}[part]
                producer_sources = {"hf-config": (), "weights": weight_sources,
                                    "towers": ("scripts/dump_gemma4_towers.py",),
                                    "store": (*rf_shared, rf + "qkv_ref.py", rf + "weight_store.py"),
                                    "rf-stack": (*rf_shared, rf + "attn_layout.py", rf + "grun.py", rf + "qkv_ref.py", rf + "stack_prep.py", rf + "weight_store.py")}[part]
                # The served hf_config also owns tokenizer sidecars; geometry reads only config.json.
                if part == "hf-config":
                    inputs = (View(checkpoint, ("*.json",)),)
                elif part in {"weights", "towers"}:
                    inputs = (weight_input,)
                else:
                    inputs = (View(checkpoint, ("config.json",)),)
                stage = Stage(inputs=inputs, dependencies=dependencies, outputs=(View(root_for[part]),),
                              sources=(rf + "recipes/gemma4_data.sh", *producer_sources),
                              environment=("IRON_DIR", "VENV_IRON", "RF_*"), sdk_python=iron_py,
                              sdk_packages=("numpy", "safetensors") if part in {"weights", "towers"} else ("numpy",))
                item = replace(item, env=item.env | {"VENV_IRON": str(Path(iron_py).parent.parent),
                                                     "RF_HF_CONFIG": checkpoint + "/config.json"})
            elif step == "build":
                packed = item.env["RF_BUILD"] + "/rf48C"
                stage = Stage(inputs=(View(checkpoint, ("config.json",)),), dependencies=("data-store",),
                              outputs=(View(packed, ("design.elf.zst", "full_elf_config.json", "fwd_layout.json", "pack.json",
                                                    "gen_args.txt", "gen_env.txt", "params.txt"), True),),
                              sources=(*compiler_sources, *rf_shared, "designs/decode_fused/elf_zst.py",
                                       *(rf + name for name in ("recipes/rf48C.sh", "fwd_ladder.sh", "env.sh", "rf_build.py", "rf_input.py", "fwd_pack.py", "kobj_cache.py"))),
                              environment=compiler_env, sdk_python=iron_py, sdk_packages=("numpy",),
                              retain=True, scratch=(item.env["RF_BUILD"],), toolchain=True)
                item = replace(item, env=item.env | compiler_runtime | {"VENV_IRON": str(Path(iron_py).parent.parent),
                                                                        "RF_HF_CONFIG": checkpoint + "/config.json"})
            else:
                stage = Stage(inputs=(tokenizer_input,), dependencies=("data-rf-stack", "data-store", "build"),
                              outputs=tuple(View(raw) for raw in recipe.outputs[:3]),
                              sources=(*rf_shared, "third_party/iron/iron/common/quant.py", rf + "artifact_meta.py", rf + "package_artifact.py"),
                              symbols=("package_gemma4", "copy_tokenizer", "main"), environment=("RF_*",),
                              sdk_python=sys.executable, sdk_packages=("numpy", "safetensors"))
                item = replace(item, env=item.env | {"RF_HF_CONFIG": checkpoint + "/config.json",
                                                     "RF_STORE": recipe.outputs[4]})
        elif model == "espcn":
            stage = Stage(inputs=(View(checkpoint),),
                          outputs=(View(recipe.outputs[0], ("conv*_*.npy", "gate_*.npy"), True),) if step == "export" else (View(recipe.outputs[0] + "/espcn.safetensors"),),
                          sources=("scripts/export_espcn.py",) if step == "export" else (),
                          sdk_python=export_py if step == "export" else '',
                          sdk_packages=("numpy", "onnx", "onnxruntime") if step == "export" else ())
        else:
            raise ValueError(f"no stage contract for {model}: {step}")
        result.append(replace(item, stage=stage))
    return replace(recipe, commands=tuple(result))


def plan(config: Path, repo: Path, artifacts: Path, inputs: Path, build: Path) -> list[Recipe]:
    recipes = [recipe_for(row["name"], row["scenario"], repo, artifacts, inputs, build)
               for row in configured_models(config)]
    return list({recipe.model: recipe for recipe in recipes}.values())


def source_digest(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            raise FileNotFoundError(f"required model input is missing: {root}")
        entries = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for entry in entries:
            digest.update(str(entry.relative_to(root.parent)).encode())
            with entry.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path, *, source: bool = False) -> str:
    if not root.exists():
        return "absent"
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    for item in files:
        if source and (any(part in {".git", "__pycache__", ".pytest_cache"}
                           for part in item.relative_to(root.parent).parts)
                       or item.suffix in {".pyc", ".pyo"}):
            continue
        digest.update(str(item.relative_to(root.parent)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def required_members(recipe: Recipe, output: bool) -> tuple[tuple[str, ...], ...]:
    if output:
        required = {
            "parakeet": (("preprocessor.onnx",), ("decoder_joint.onnx",), ("vocab.txt",), ("encoder/manifest.json",)),
            "bge-base": (("model.onnx",), ("tokenizer.json",)),
            "pyannote-community-1": (("segmentation.onnx",), ("embedding.onnx",), ("diarize.json",)),
            "pyannote-3.1": (("segmentation.onnx",), ("embedding.onnx",), ("diarize.json",)),
            "whisper-turbo": (("decode_rail/meta.json",), ("config.json",), ("preprocessor.onnx",),
                              ("onnx/decoder_model.onnx",), ("onnx/decoder_with_past_model.onnx",),
                              ("tokenizer.json",), ("serving-companions.json",)),
            "gemma3-270m": (("meta.json",),),
            "qwen3-0.6b": (("meta.json",),),
            "qwen3.5-4b": (("meta.json",),),
            "gemma4-12b": (("meta.json",),),
            "espcn": (("espcn.safetensors",),),
        }
    else:
        required = {
            "parakeet": (("encoder-model.onnx",), ("encoder-model.onnx.data",),
                         ("decoder_joint.onnx", "decoder_joint-model.onnx"), ("vocab.txt",), ("preprocessor.onnx",)),
            "bge-base": (("config.json",), ("tokenizer.json",), ("model.safetensors", "pytorch_model.bin")),
            "whisper-turbo": (("config.json",), ("model.safetensors", "pytorch_model.bin")),
            "gemma3-270m": (("config.json",), ("tokenizer.json",), ("*.safetensors",)),
            "qwen3-0.6b": (("config.json",), ("tokenizer.json",), ("*.safetensors",)),
            "qwen3.5-4b": (("config.json",), ("tokenizer.json",), ("*.safetensors",)),
            "gemma4-12b": (("config.json",), ("tokenizer.json",), ("*.safetensors",)),
            "espcn": (("espcn_x3_dyn.onnx",),),
        }
    return required.get(recipe.model, ())


def require_members(recipe: Recipe, root: Path, output: bool) -> None:
    kind = "artifact output" if output else "model input"
    if not output and recipe.model.startswith("pyannote"):
        validate_local_pipeline_inputs(root)
        return
    for choices in required_members(recipe, output):
        if not any(candidate for choice in choices for candidate in root.glob(choice)):
            raise FileNotFoundError(f"{recipe.model}: required {kind} missing under {root}: {' or '.join(choices)}")


def validate_recipe_inputs(recipes: list[Recipe]) -> None:
    for recipe in recipes:
        for raw in recipe.inputs:
            root = Path(raw)
            if not root.exists():
                raise FileNotFoundError(f"{recipe.model}: required model input is missing: {root}")
            require_members(recipe, root if root.is_dir() else root.parent, False)


def validate_recipe_outputs(recipe: Recipe, *, references: Recipe | None = None) -> None:
    for raw in recipe.outputs:
        if not Path(raw).exists():
            raise FileNotFoundError(f"{recipe.model}: recipe output missing: {raw}")
    require_members(recipe, Path(recipe.outputs[0]), True)
    if recipe.model == "qwen3.5-4b":
        validate_qwen35_closure(recipe, references or recipe)
    if recipe.model == "whisper-turbo":
        from whisper_serving_companions import validate
        validate(Path(recipe.outputs[0]))


def validate_qwen35_closure(recipe: Recipe, references: Recipe, *, step: str | None = None) -> None:
    decode, prefill, tokenizer = map(Path, recipe.outputs)
    reference_decode = Path(references.outputs[0])

    def require_file(target: Path) -> None:
        if not target.is_file():
            raise FileNotFoundError(f"{recipe.model}: required artifact dependency missing: {target}")
        if not any(target.resolve().is_relative_to(Path(raw).resolve()) for raw in recipe.outputs):
            raise ValueError(f"{recipe.model}: artifact dependency escapes published outputs: {target}")

    if step == "tokenizer":
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            require_file(tokenizer / name)
        return
    decode_meta = json.loads((decode / "meta.json").read_text())
    roots = (decode,) if step == "decode" else (prefill,) if step == "prefill" else (decode, prefill)
    for root in roots:
        require_file(root / "meta.json")
        meta = json.loads((root / "meta.json").read_text())
        elf = root / meta["elf"]
        require_file(elf if elf.is_file() else Path(str(elf) + ".zst"))
        if root == prefill:
            if (meta.get("weights_from") != str(reference_decode / "buffers")
                    or (meta.get("decode_artifact") or {}).get("meta") != str(reference_decode / "meta.json")):
                raise ValueError(f"{recipe.model}: prefill must reference its published decode output")
        for name in meta["weights"]:
            # Shared prefill weights are uploaded by decode; prefill-only weights are local.
            owner = decode if root == prefill and name in decode_meta["layout"] else root
            if owner == decode and name not in decode_meta["weights"]:
                raise ValueError(f"{recipe.model}: shared weight is not uploaded by decode: {name}")
            blob = owner / "buffers" / f"{name}.bin"
            require_file(blob)
            if blob.stat().st_size != meta["layout"][name]["len"]:
                raise ValueError(f"{recipe.model}: blob length disagrees with layout: {blob}")
        if root == decode:
            host_blobs = [meta.get("embed_blob") or "W_head"]
            protocol = meta.get("host_protocol", {})
            host_blobs += [protocol[key] for key in ("pos_blob", "logit_bias_blob") if protocol.get(key)]
            for name in host_blobs:
                require_file(decode / "buffers" / f"{name}.bin")
    if step is None:
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            require_file(tokenizer / name)


def validate_stage_outputs(recipe: Recipe, step: str) -> None:
    if recipe.model == "qwen3.5-4b" and step in {"decode", "prefill", "tokenizer"}:
        validate_qwen35_closure(recipe, recipe, step=step)
        if step in {"decode", "prefill"}:
            dump = next(item.argv[item.argv.index("--out") + 1] for item in recipe.commands if item.step == "weights")
            metadata = Path(recipe.outputs[0 if step == "decode" else 1]) / "meta.json"
            if dump.encode() in metadata.read_bytes():
                raise RuntimeError(f"{recipe.model}: staged path leaked into published metadata: {metadata}")
    elif step == "export" and recipe.model in {"parakeet", "bge-base", *PYANNOTE_SDKS}:
        validate_recipe_outputs(recipe)
    elif recipe.model == "whisper-turbo" and step == "serving-companions":
        from whisper_serving_companions import validate
        validate(Path(recipe.outputs[0]))
    elif recipe.model == "gemma4-12b" and step == "package":
        validate_recipe_outputs(recipe)


def authority_digest(recipe: Recipe, repo: Path) -> str:
    if recipe.commands and all(item.stage is not None for item in recipe.commands):
        return stage_digest({item.step: source_identity(item, repo, symbols=globals()) for item in recipe.commands})
    digest = hashlib.sha256()
    root = repo.resolve()
    candidates = (
        root / "toolchain.lock",
        root / recipe.scenario,
        root / "scripts",
        root / "designs/decode_fused",
        root / "designs/resident_forward",
        root / "aie_kernels",
        root / "third_party/iron",
    )
    dispatcher = Path(__file__).resolve()
    if dispatcher.is_relative_to(root):
        candidates += (dispatcher,)
    command_files = []
    for item in recipe.commands:
        for raw in item.argv[1:]:
            candidate = Path(raw)
            if candidate.is_file() and candidate.resolve().is_relative_to(root):
                command_files.append(candidate.resolve())
    candidates += tuple(command_files)
    for candidate in candidates:
        digest.update(str(candidate.relative_to(root)).encode())
        digest.update(tree_digest(candidate, source=True).encode())
    for name in ("MODEL_IRON_PY", "MODEL_EXPORT_PY", "MODEL_PYANNOTE_PY", "MODEL_PYANNOTE_COMMUNITY_PY", "MODEL_AIECC"):
        value = os.environ.get(name)
        if value:
            candidate = Path(value)
            digest.update(name.encode())
            digest.update(tree_digest(candidate).encode())
    if recipe.model in PYANNOTE_SDKS:
        digest.update(pyannote_sdk_identity(recipe).encode())
    return digest.hexdigest()


def pyannote_sdk_identity(recipe: Recipe) -> str:
    executable = recipe.commands[0].argv[0]
    packages = PYANNOTE_SDKS[recipe.model][1]
    probe = (
        "import hashlib,json;from importlib.metadata import distribution;"
        f"names={packages!r};"
        "print(json.dumps({name:{'version':(d:=distribution(name)).version,"
        "'record':hashlib.sha256((d.read_text('RECORD') or d.read_text('METADATA') or '').encode()).hexdigest()}"
        " for name in names},sort_keys=True))"
    )
    try:
        return subprocess.check_output([executable, "-c", probe], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"{recipe.model}: SDK identity probe failed for {executable}; run its selected setup script") from error


def recipe_key(recipe: Recipe, repo: Path, config: Path, *, authority: str | None = None,
               inputs: str | None = None) -> str:
    # Selection is already represented by this recipe; deployment-only defaults are not build inputs.
    del config
    h = hashlib.sha256()
    h.update((repo / recipe.scenario).read_bytes())
    h.update(json.dumps(asdict(recipe), sort_keys=True).encode())
    h.update((authority or authority_digest(recipe, repo)).encode())
    h.update((inputs or source_digest(recipe.inputs)).encode())
    return h.hexdigest()


def output_digest(outputs: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for raw in outputs:
        root = Path(raw)
        if not root.exists():
            raise FileNotFoundError(f"recipe output missing: {root}")
        if root.is_file():
            files = [root]
        else:
            files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".model-recipe.json")
        for entry in files:
            h.update(str(entry.relative_to(root.parent)).encode())
            with entry.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
    return h.hexdigest()


def manifest_path(recipe: Recipe) -> Path:
    return Path(recipe.outputs[0]) / ".model-recipe.json"


def cache_hit(recipe: Recipe, key: str, *, references: Recipe | None = None) -> bool:
    manifest = manifest_path(recipe)
    if not manifest.is_file():
        return False
    try:
        validate_recipe_outputs(recipe, references=references)
        record = json.loads(manifest.read_text())
        return record["recipe_key"] == key and record["output_digest"] == output_digest(recipe.outputs)
    except (OSError, KeyError, json.JSONDecodeError, FileNotFoundError):
        return False
    except (ValueError, TypeError):
        if recipe.model == "qwen3.5-4b":
            return False
        raise


def ensure_outputs_ready(recipe: Recipe, key: str) -> None:
    """Live outputs are never mutated until a verified staging build is ready."""
    del recipe, key


def copy_tokenizer(source: Path, target: Path) -> None:
    files = [p for p in source.iterdir() if p.is_file() and (p.suffix in {".json", ".jinja"} or p.name.endswith(".model"))]
    if not files:
        raise FileNotFoundError(f"no tokenizer/config JSON files in {source}")
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        shutil.copy2(item, target / item.name)
    template, config = target / "chat_template.jinja", target / "tokenizer_config.json"
    if template.is_file() and config.is_file():
        data = json.loads(config.read_text())
        if not data.get("chat_template"):
            data["chat_template"] = template.read_text()
            config.write_text(json.dumps(data, indent=2) + "\n")


def link_or_copy(source: str, destination: str) -> str:
    if Path(source).name.endswith(".onnx.data"):
        return shutil.copy2(source, destination)
    try:
        os.link(Path(source).resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, copy_function=link_or_copy, symlinks=False)


def package_gemma4(repo: Path, artifacts: Path, build: Path, out: Path, tokenizer_source: Path, tokenizer_out: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    checkpoint = artifacts / "gemma4-12b-qat/checkpoint"
    copy_tokenizer(tokenizer_source, checkpoint)
    store = artifacts / "gemma4-12b/store"
    manifest = json.loads((store / "manifest.json").read_text())
    towers = {}
    for name, entry in manifest["towers"].items():
        dtype = np.dtype(entry["dtype"])
        if entry["layout"] != "raw_native" or entry["length"] % dtype.itemsize:
            raise ValueError(f"{name}: invalid tower storage layout")
        towers[name] = np.fromfile(store / "blobs" / f"{entry['blob']}.bin", dtype=dtype,
                                   count=entry["length"] // dtype.itemsize,
                                   offset=entry["offset"]).reshape(entry["shape"])
    save_file(towers, checkpoint / "model.safetensors")
    with tempfile.TemporaryDirectory(prefix=".gemma4-package-", dir=build.parent) as temporary:
        metadata = Path(temporary) / "meta.json"
        subprocess.run([
            sys.executable, str(repo / "designs/resident_forward/artifact_meta.py"),
            "--build", str(build / "rf48C"), "--store", str(store),
            "--config", str(checkpoint / "config.json"), "--out", str(metadata),
        ], check=True, cwd=repo)
        cmd = [
            sys.executable,
            str(repo / "designs/resident_forward/package_artifact.py"),
            "--out", str(out),
            "--build", str(build / "rf48C"),
            "--meta", str(metadata),
            "--weights", str(artifacts / "gemma4-12b/rf_stack"),
            "--store", str(artifacts / "gemma4-12b/store"),
        ]
        subprocess.run(cmd, check=True, cwd=repo)
    copy_tokenizer(tokenizer_source, tokenizer_out)


def run_commands(recipe: Recipe, repo: Path) -> None:
    for item in recipe.commands:
        subprocess.run(item.argv, check=True, cwd=repo, env=os.environ | item.env)


def validate_command_executables(recipes: list[Recipe]) -> None:
    for recipe in recipes:
        for item in recipe.commands:
            executable = Path(item.argv[0])
            if executable.is_absolute() and not executable.is_file():
                raise FileNotFoundError(
                    f"missing executable for {recipe.model} {item.step}: {executable}")


def input_source(recipe: Recipe) -> tuple[str, str] | None:
    sources = {
        "parakeet": ("istupakov/parakeet-tdt-0.6b-v3-onnx", "scripts/fetch_models.sh"),
        "bge-base": ("BAAI/bge-base-en-v1.5", "scripts/fetch_models.sh"),
        "pyannote-community-1": ("pyannote/speaker-diarization-community-1", "docs/model-support.md"),
        "pyannote-3.1": ("pyannote/speaker-diarization-3.1", "scripts/export_pyannote.py"),
        "whisper-turbo": ("openai/whisper-large-v3-turbo", "scripts/dump_llm_weights.py"),
        "gemma3-270m": ("unsloth/gemma-3-270m-it", "scripts/dump_llm_weights.py"),
        "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "scripts/dump_llm_weights.py"),
        "gemma4-12b": ("google/gemma-4-12B-it-qat-q4_0-unquantized", "designs/resident_forward/recipes/gemma4_data.sh"),
        "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "scripts/dump_llm_weights.py"),
    }
    return sources.get(recipe.model)


def source_manifest_path(recipe: Recipe) -> Path:
    input_path = Path(recipe.inputs[0])
    return (input_path.parent if input_path.is_file() else input_path) / ".model-input-source.json"


def local_input_override(recipe: Recipe) -> Path | None:
    name = "MODEL_INPUT_SOURCE_" + "".join(char if char.isalnum() else "_" for char in recipe.model.upper())
    value = os.environ.get(name)
    return Path(value).resolve() if value else None


def cached_snapshot(repo_id: str) -> Path | None:
    hub = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")))
    snapshots = sorted((hub / f"models--{repo_id.replace('/', '--')}" / "snapshots").glob("*"))
    snapshots = [item for item in snapshots if item.is_dir()]
    if len(snapshots) == 1:
        return snapshots[0]
    if len(snapshots) > 1:
        raise RuntimeError(f"multiple cached snapshots for {repo_id}; materialize the intended revision explicitly")
    return None


def update_pyannote_input(recipe: Recipe) -> None:
    target = Path(recipe.inputs[0])
    manifest = source_manifest_path(recipe)
    try:
        record = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{recipe.model}: cannot update local pyannote input without a valid source manifest") from error
    components = pipeline_components(target)
    sources: list[tuple[PipelineComponent, Path, tuple[Path, ...]]] = []
    needs_update = False
    for component in components:
        missing = tuple(path for path in component.files if not path.is_file())
        if component.source is None:
            if missing:
                raise FileNotFoundError(f"{recipe.model}: local pyannote input lacks {' or '.join(map(str, missing))}")
            continue
        needs_update = needs_update or bool(missing)
        snapshot = cached_snapshot(component.source)
        if snapshot is None:
            raise FileNotFoundError(
                f"{recipe.model}: authorized local HF cache required for {component.source}; no credential is read by the installer")
        source_files = tuple(snapshot / path.name for path in component.files)
        for source in source_files:
            if not source.is_file():
                raise FileNotFoundError(f"{recipe.model}: cached dependency lacks {source}")
        sources.append((component, snapshot, source_files))
    if not needs_update:
        return
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.pyannote-", dir=target.parent) as directory:
        staged = Path(directory) / target.name
        copy_tree(target, staged)
        dependencies = []
        for component, snapshot, source_files in sources:
            files = []
            for source, destination in zip(source_files, component.files, strict=True):
                staged_destination = staged / destination.relative_to(target)
                staged_destination.parent.mkdir(parents=True, exist_ok=True)
                if staged_destination.is_file():
                    if tree_digest(staged_destination) != tree_digest(source):
                        raise RuntimeError(f"{recipe.model}: local dependency differs from cached {component.source}: {destination}")
                else:
                    shutil.copy2(source, staged_destination)
                files.append({"source": source.name, "target": str(destination.relative_to(target)),
                              "sha256": tree_digest(source)})
            dependencies.append({"component": component.name, "source": component.source,
                                 "source_path": str(snapshot), "source_digest": tree_digest(snapshot),
                                 "files": files})
        validate_local_pipeline_inputs(staged)
        staged_manifest = staged / manifest.name
        staged_manifest.unlink(missing_ok=True)
        record["dependencies"] = dependencies
        record["materialized_digest"] = tree_digest(staged)
        staged_manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        backup = target.parent / f".{target.name}.pyannote-backup"
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(target, backup)
        try:
            os.replace(staged, target)
        except Exception:
            if not target.exists() and backup.exists():
                os.replace(backup, target)
            raise
        shutil.rmtree(backup)


def materialize_input(recipe: Recipe, source: Path, repo_id: str, authority: str) -> None:
    target = Path(recipe.inputs[0])
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied = target.parent
    else:
        if target.exists():
            raise RuntimeError(f"{recipe.model}: refusing to overwrite unmanaged model input {target}")
        copy_tree(source, target)
        copied = target
    source_manifest_path(recipe).write_text(json.dumps({
        "model": recipe.model,
        "source": repo_id,
        "authority": authority,
        "source_path": str(source),
        "source_digest": tree_digest(source),
        "materialized_digest": tree_digest(copied),
    }, indent=2, sort_keys=True) + "\n")


def add_parakeet_preprocessor(recipe: Recipe) -> None:
    target = Path(recipe.inputs[0]) / "preprocessor.onnx"
    if target.is_file():
        return
    configured = os.environ.get("PARAKEET_PREPROCESSOR")
    if configured:
        source = Path(configured)
    else:
        export_py = os.environ.get("MODEL_EXPORT_PY", sys.executable)
        package = subprocess.check_output(
            [export_py, "-c", "import onnx_asr, pathlib; print(pathlib.Path(onnx_asr.__file__).parent / 'preprocessors/data/nemo128.onnx')"],
            text=True).strip()
        source = Path(package)
    if not source.is_file():
        raise FileNotFoundError(f"parakeet: nemo128 preprocessor source is missing: {source}")
    shutil.copy2(source, target)


def provision_inputs(recipes: list[Recipe], allow_download: bool) -> None:
    for recipe in recipes:
        target = Path(recipe.inputs[0])
        try:
            require_members(recipe, target if target.is_dir() else target.parent, False)
            if source_manifest_path(recipe).is_file():
                continue
            raise RuntimeError(f"{recipe.model}: unmanaged local input {target}; use an HF cache source or provide a source manifest")
        except FileNotFoundError:
            if (recipe.model.startswith("pyannote") and target.is_dir()
                    and (target / "config.yaml").is_file() and source_manifest_path(recipe).is_file()):
                update_pyannote_input(recipe)
                continue
        local = local_input_override(recipe)
        if local is not None:
            require_members(recipe, local if local.is_dir() else local.parent, False)
            materialize_input(recipe, local, f"local:{local}", "explicit MODEL_INPUT_SOURCE_* override")
            continue
        spec = input_source(recipe)
        if spec is None:
            raise FileNotFoundError(f"{recipe.model}: no existing source authority for {target}; provide the declared ONNX input")
        repo_id, authority = spec
        source = cached_snapshot(repo_id)
        gated = recipe.model.startswith("pyannote") or recipe.model.startswith("gemma")
        if source is None and gated:
            raise FileNotFoundError(f"{recipe.model}: authorized local HF cache required for {repo_id}; no credential is read by the installer")
        if source is None and not allow_download:
            raise FileNotFoundError(f"{recipe.model}: cache miss for {repo_id}; re-run install.sh with MODEL_INPUT_DOWNLOAD=1")
        if source is None:
            executable = os.environ.get("HF_CLI", "huggingface-cli")
            if shutil.which(executable) is None and not Path(executable).is_file():
                raise FileNotFoundError(f"{recipe.model}: missing Hugging Face downloader {executable}")
            subprocess.run([executable, "download", repo_id], check=True)
            source = cached_snapshot(repo_id)
            if source is None:
                raise RuntimeError(f"{recipe.model}: downloader completed without a cache snapshot for {repo_id}")
        materialize_input(recipe, source, repo_id, authority)
        if recipe.model == "parakeet":
            add_parakeet_preprocessor(recipe)
            manifest = source_manifest_path(recipe)
            record = json.loads(manifest.read_text())
            record["materialized_digest"] = tree_digest(Path(recipe.inputs[0]))
            manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def rebase_recipe(recipe: Recipe, live_artifacts: Path, staged_artifacts: Path,
                  live_build: Path, staged_build: Path) -> Recipe:
    def replace(value: str) -> str:
        for live, staged in ((live_artifacts, staged_artifacts), (live_build, staged_build)):
            if value == str(live) or value.startswith(str(live) + os.sep):
                return str(staged) + value[len(str(live)):]
        return value
    return Recipe(recipe.model, recipe.scenario, recipe.inputs,
                  tuple(replace(value) for value in recipe.outputs),
                  tuple(Command(item.step, tuple(replace(value) for value in item.argv),
                                {key: replace(value) for key, value in item.env.items()},
                                rebase_stage(item.stage, replace))
                        for item in recipe.commands))


def write_receipt(recipe: Recipe, key: str, *, commands: tuple[Command, ...] | None = None,
                  stages: dict | None = None) -> None:
    record = {"model": recipe.model, "scenario": recipe.scenario, "recipe_key": key,
              "output_digest": output_digest(recipe.outputs),
              "commands": [asdict(c) for c in (recipe.commands if commands is None else commands)]}
    if stages is not None:
        record["stages"] = stages
    target = manifest_path(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def relocate_recipe_metadata(recipe: Recipe, staged: Recipe) -> None:
    pairs = sorted(zip(staged.outputs, recipe.outputs, strict=True),
                   key=lambda pair: len(pair[0]), reverse=True)

    def relocate(value):
        if isinstance(value, str):
            for source, destination in pairs:
                if value == source or value.startswith(source + os.sep):
                    return destination + value[len(source):]
        elif isinstance(value, list):
            return [relocate(item) for item in value]
        elif isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        return value

    for raw in staged.outputs:
        for metadata in Path(raw).rglob("meta.json"):
            original = json.loads(metadata.read_text())
            relocated = relocate(original)
            if relocated != original:
                metadata.write_text(json.dumps(relocated, indent=2) + "\n")


def no_staging_references(recipe: Recipe, staging_root: Path) -> None:
    needle = str(staging_root).encode()
    for raw in recipe.outputs:
        root = Path(raw)
        for item in root.rglob("*"):
            if item.is_file() and item.suffix in {".json", ".toml", ".txt"} and needle in item.read_bytes():
                raise RuntimeError(f"{recipe.model}: staged path leaked into published metadata: {item}")


def publish_recipe(recipe: Recipe, staged: Recipe, key: str, rollback: Path) -> None:
    with resource_locks((*recipe.outputs, *staged.outputs, rollback / key)):
        _publish_recipe(recipe, staged, key, rollback)


def _publish_recipe(recipe: Recipe, staged: Recipe, key: str, rollback: Path, *, stages: dict | None = None) -> None:
    require_no_recovery(rollback, key)
    reference_args = {"references": recipe} if recipe.model == "qwen3.5-4b" else {}
    validate_recipe_outputs(staged, **reference_args)
    write_receipt(staged, key, commands=recipe.commands, stages=stages)
    if not cache_hit(staged, key, **reference_args):
        raise RuntimeError(f"recipe manifest did not verify in staging: {recipe.model}")
    moved: list[tuple[Path, Path]] = []
    try:
        for index, (live_raw, staged_raw) in enumerate(zip(recipe.outputs, staged.outputs, strict=True)):
            live, fresh = Path(live_raw), Path(staged_raw)
            backup = rollback / key / str(index) / live.name
            if live.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(live, backup)
                moved.append((live, backup))
            live.parent.mkdir(parents=True, exist_ok=True)
            os.replace(fresh, live)
    except Exception:
        for live, backup in reversed(moved):
            if not live.exists() and backup.exists():
                os.replace(backup, live)
        raise
    if not cache_hit(recipe, key):
        raise RuntimeError(f"published receipt did not verify: {recipe.model}")
    backup_root = rollback / key
    if backup_root.exists():
        shutil.rmtree(backup_root)


def verified_existing_gemma4(recipe: Recipe, key: str, repo: Path) -> bool:
    if recipe.model != "gemma4-12b":
        return False
    try:
        validate_recipe_outputs(recipe)
        expected = next((line.split("=", 1)[1].split("#", 1)[0].strip()
                         for line in (repo / "toolchain.lock").read_text().splitlines()
                         if line.startswith("IRON_SOURCE_COMMIT=")), None)
        if not expected or validate_iron_provenance(Path(recipe.outputs[0]) / "meta.json", expected):
            return False
        for raw in recipe.outputs[3:]:
            if not (Path(raw) / ".recipe-manifest.json").is_file():
                return False
    except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError):
        return False
    write_receipt(recipe, key)
    return cache_hit(recipe, key)


def stage_witness_path(stage_root: Path) -> Path:
    return stage_root.parent / f".{stage_root.name}.published.json"


def write_stage_witness(recipe: Recipe, staged: Recipe, key: str, stage_root: Path) -> None:
    # Prepared intent is cleanup authority only after the matching final payload verifies.
    info = stage_root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("transaction stage is not a directory")
    expected = {"recipe_key": key, "output_digest": output_digest(staged.outputs),
                "outputs": list(recipe.outputs), "stage": str(stage_root.parent.resolve() / stage_root.name),
                "stage_identity": [info.st_dev, info.st_ino]}
    with stage_witness_path(stage_root).open("x") as handle:
        json.dump(expected, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def expire_published_stage(recipe: Recipe, key: str, stage_root: Path, *, record: bool = False) -> None:
    witness = stage_witness_path(stage_root)
    try:
        if not record and not witness.exists() and not witness.is_symlink():
            if stage_root.exists() or stage_root.is_symlink():
                raise ValueError("transaction stage has no publication witness")
            return
        receipt = json.loads(manifest_path(recipe).read_text())
        expected = {"recipe_key": key, "output_digest": receipt["output_digest"],
                    "outputs": list(recipe.outputs),
                    "stage": str(stage_root.parent.resolve() / stage_root.name)}
        if record:
            write_stage_witness(recipe, recipe, key, stage_root)
        if not stat.S_ISREG(witness.lstat().st_mode):
            raise ValueError("publication witness is not a regular file")
        saved = json.loads(witness.read_text())
        if not isinstance(saved, dict) or any(saved.get(name) != value for name, value in expected.items()):
            raise ValueError("publication witness does not match verified outputs")
        if stage_root.exists() or stage_root.is_symlink():
            info = stage_root.lstat()
            if not stat.S_ISDIR(info.st_mode) or saved.get("stage_identity") != [info.st_dev, info.st_ino]:
                raise ValueError("transaction stage identity changed")
            shutil.rmtree(stage_root)
        witness.unlink()
    except (OSError, ValueError, KeyError) as error:
        print(f"WARNING {recipe.model}: cleanup pending at {stage_root}: {error}", file=sys.stderr)


def stage_locations(recipe: Recipe, artifacts: Path, build: Path) -> tuple[Path, Path]:
    layout = stage_digest({"model": recipe.model, "outputs": recipe.outputs,
                           "artifacts": str(artifacts), "build": str(build)})[:16]
    return (build / "model-artifact-staging" / f"{recipe.model}-active-{layout}",
            build / "model-artifact-state" / f"{recipe.model}-{layout}")


def build_recipe(recipe: Recipe, repo: Path, config: Path, artifacts: Path, build: Path) -> str:
    authority_before = authority_digest(recipe, repo)
    inputs_before = source_digest(recipe.inputs)
    key = recipe_key(recipe, repo, config, authority=authority_before, inputs=inputs_before)
    runner = None
    resources = list(recipe.outputs)
    if recipe.commands and all(item.stage is not None for item in recipe.commands):
        stage_root, state = stage_locations(recipe, artifacts, build)
        staged = rebase_recipe(recipe, artifacts, stage_root / "artifacts", build, stage_root / "build")
        runner = StageRunner(recipe, staged, repo, stage_root, state, artifacts, build, run_commands,
                             symbols=globals(), validate=validate_stage_outputs)
        resources.append(state)
    else:
        stage_root = build / "model-artifact-staging" / f"{recipe.model}-{key[:16]}"
    rollback = artifacts / ".model-artifact-rollback"
    with resource_locks((*resources, stage_root, stage_witness_path(stage_root), rollback / key)):
        return _build_recipe(recipe, repo, artifacts, build, key, stage_root,
                             rollback, authority_before, inputs_before, runner=runner)


def _build_recipe(recipe: Recipe, repo: Path, artifacts: Path, build: Path, key: str,
                  stage_root: Path, rollback: Path, authority_before: str, inputs_before: str,
                  *, runner: StageRunner | None = None) -> str:
    require_no_recovery(rollback, key)
    if cache_hit(recipe, key):
        if runner is not None:
            runner.resume_settlement(key)
        expire_published_stage(recipe, key, stage_root)
        return "HIT"
    witness = stage_witness_path(stage_root)
    if witness.exists() or witness.is_symlink():
        raise RuntimeError(f"publication recovery requires inspection before retry: {witness}")
    if runner is not None and manifest_path(recipe).exists() and (
            not (runner.state / "owner.json").exists() or
            not json.loads(manifest_path(recipe).read_text()).get("stages")):
        raise RuntimeError(f"{recipe.model}: historical receipt has no qualified stage witnesses; "
                           "preserve it and qualify frontier adoption, or use prepared installation; "
                           "a new cache protocol does not authorize rebuilding or restamping accepted artifacts")
    if runner is None and verified_existing_gemma4(recipe, key, repo):
        return "VERIFIED"
    ensure_outputs_ready(recipe, key)
    if runner is None and (stage_root.exists() or stage_root.is_symlink()):
        raise RuntimeError(f"unowned recipe staging requires inspection before retry: {stage_root}")
    staged_artifacts, staged_build = stage_root / "artifacts", stage_root / "build"
    staged = rebase_recipe(recipe, artifacts, staged_artifacts, build, staged_build)
    if runner is not None:
        runner.run()
    else:
        run_commands(staged, repo)
    if recipe.model == "qwen3.5-4b":
        validate_recipe_outputs(staged)
    relocate_recipe_metadata(recipe, staged)
    validate_recipe_outputs(staged, references=recipe if recipe.model == "qwen3.5-4b" else None)
    no_staging_references(staged, stage_root)
    if authority_before != authority_digest(recipe, repo) or inputs_before != source_digest(recipe.inputs):
        raise RuntimeError(f"{recipe.model}: source authority changed while the staged build ran")
    if runner is not None:
        runner.prepare_settlement(key)
    write_stage_witness(recipe, staged, key, stage_root)
    _publish_recipe(recipe, staged, key, rollback,
                    stages=runner.publication_record() if runner is not None else None)
    if not cache_hit(recipe, key):
        raise RuntimeError(f"recipe manifest did not verify after publish: {recipe.model}")
    if runner is not None:
        try:
            runner.settle()
        except (OSError, ValueError, RuntimeError) as error:
            print(f"WARNING {recipe.model}: verified publication; frontier settlement pending at {runner.state}: {error}", file=sys.stderr)
            return "BUILT"
    expire_published_stage(recipe, key, stage_root)
    return "BUILT"


def public_plan(recipes: list[Recipe]) -> list[dict]:
    return [{"model": recipe.model, "scenario": recipe.scenario, "inputs": list(recipe.inputs),
             "outputs": list(recipe.outputs), "commands": [asdict(item) for item in recipe.commands]}
            for recipe in recipes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--model-input-root", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--pyannote-sdks", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--copy-tokenizer", nargs=2, metavar=("SOURCE", "OUT"))
    parser.add_argument("--package-gemma4", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tokenizer-source", type=Path)
    parser.add_argument("--tokenizer-out", type=Path)
    args = parser.parse_args()

    if args.copy_tokenizer:
        copy_tokenizer(Path(args.copy_tokenizer[0]), Path(args.copy_tokenizer[1]))
        return 0
    if args.package_gemma4:
        required = (args.repo, args.artifacts_root, args.build_dir, args.out, args.tokenizer_source, args.tokenizer_out)
        if any(value is None for value in required):
            parser.error("--package-gemma4 needs --repo --artifacts-root --build-dir --out --tokenizer-source --tokenizer-out")
        package_gemma4(args.repo.resolve(), args.artifacts_root.resolve(), args.build_dir.resolve(), args.out.resolve(), args.tokenizer_source.resolve(), args.tokenizer_out.resolve())
        return 0

    if sum((args.plan, args.build, args.pyannote_sdks)) != 1:
        parser.error("choose exactly one of --plan, --build, or --pyannote-sdks")
    if args.pyannote_sdks:
        if args.config is None:
            parser.error("--pyannote-sdks needs --config")
        print("\n".join(configured_pyannote_sdks(args.config.resolve())))
        return 0
    for name in ("config", "repo", "artifacts_root", "model_input_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    repo, artifacts, inputs = args.repo.resolve(), args.artifacts_root.resolve(), args.model_input_root.resolve()
    build = (args.build_dir or artifacts.parent / "build/model-artifacts").resolve()
    recipes = plan(args.config.resolve(), repo, artifacts, inputs, build)
    if args.plan:
        rendered = public_plan(recipes)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({"configured_count": len(rendered), "recipes": rendered}, indent=2) + "\n")
        print(json.dumps(rendered, indent=2))
        return 0

    validate_command_executables(recipes)
    provision_inputs(recipes, os.environ.get("MODEL_INPUT_DOWNLOAD") == "1")
    validate_recipe_inputs(recipes)
    results = []
    for recipe in recipes:
        status = build_recipe(recipe, repo, args.config.resolve(), artifacts, build)
        results.append({"model": recipe.model, "scenario": recipe.scenario, "status": status})
        print(f"{status} {recipe.model} {recipe.scenario}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"configured_count": len(results), "recipes": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
