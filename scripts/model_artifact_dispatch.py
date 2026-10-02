#!/usr/bin/env python3
"""Dispatch the configured model-artifact recipes used by install.sh.

The plan is deliberately limited to engine.toml's selected models. It records
actual build/export commands and their declared inputs, but performs no device,
service, package, or toolchain action on its own.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Command:
    step: str
    argv: tuple[str, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class Recipe:
    model: str
    scenario: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    commands: tuple[Command, ...]


def path(root: Path, value: str) -> str:
    return str((root / value).resolve())


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


def recipe_for(model: str, scenario: str, repo: Path, artifacts: Path, inputs: Path, build: Path) -> Recipe:
    py = sys.executable
    export_py = os.environ.get("MODEL_EXPORT_PY", sys.executable)
    iron_py = os.environ.get("MODEL_IRON_PY", str(repo / ".venv-iron/bin/python"))
    pyannote_py = os.environ.get("MODEL_PYANNOTE_PY", str(repo / ".venv-pyannote/bin/python"))
    npu = os.environ.get("NPU_BIN", "npu")
    scripts = repo / "scripts"
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
        return Recipe(
            model, scenario, (src(f"pyannote/{slug}"),), (out(f"pyannote/{slug}"),),
            (command("export", [pyannote_py, str(scripts / "export_pyannote.py")],
                     PYANNOTE_PIPELINE=src(f"pyannote/{slug}"),
                     PYANNOTE_OUT=out(f"pyannote/{slug}"), HF_HUB_OFFLINE="1"),),
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
            ),
        )
    if model == "gemma3-270m" and scenario == "scenarios/generate-gemma3-270m.toml":
        weights, decode, tokenizer = out("gemma3-270m/weights_qat_int4g32"), out("gemma3-270m/decode_qat_int4_pe8e957d"), out("gemma3-270m/tokenizer_qat")
        return Recipe(
            model, scenario, (src("gemma3-270m/checkpoint"),), (decode, weights, tokenizer),
            (
                command("weights", [iron_py, str(scripts / "dump_llm_weights.py"), "--spec", model, "--checkpoint-dir", src("gemma3-270m/checkpoint"), "--out", weights, "--quant", "int4", "--quant-group", "32", "--quant-full-range", "--quant-leaves", "gate_proj,up_proj,down_proj,o_proj,embed_tokens"],
                        IRON=str(repo / "third_party/iron")),
                command("decode", ["bash", str(scripts / "build_llm_decode.sh"), model, "", decode], WEIGHTS=weights),
                command("tokenizer", [py, str(scripts / "model_artifact_dispatch.py"), "--copy-tokenizer", src("gemma3-270m/checkpoint"), tokenizer]),
            ),
        )
    if model == "qwen3-0.6b" and scenario == "scenarios/generate-qwen3-0.6b.toml":
        weights, decode, prefill, tokenizer = out("qwen3-0.6b/weights"), out("qwen3-0.6b/decode_p8e3958"), out("qwen3-0.6b/prefill_p8e3958"), out("qwen3-0.6b/tokenizer")
        return Recipe(
            model, scenario, (src("qwen3-0.6b/checkpoint"),), (decode, prefill, weights, tokenizer),
            (
                command("weights", [iron_py, str(scripts / "dump_llm_weights.py"), "--spec", model, "--checkpoint-dir", src("qwen3-0.6b/checkpoint"), "--out", weights]),
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
                command("data", ["bash", str(repo / "designs/resident_forward/recipes/gemma4_data.sh"), "--out", str(artifacts), "--checkpoint-dir", src("gemma4-12b/checkpoint")]),
                command("build", ["bash", str(repo / "designs/resident_forward/recipes/rf48C.sh")], RF_BUILD=rf_build),
                command("package", [py, str(scripts / "model_artifact_dispatch.py"), "--package-gemma4", "--repo", str(repo), "--artifacts-root", str(artifacts), "--build-dir", rf_build, "--out", resident, "--tokenizer-source", src("gemma4-12b/checkpoint"), "--tokenizer-out", tokenizer]),
            ),
        )
    if model == "qwen3.5-4b" and scenario == "scenarios/generate-qwen3.5-4b.toml":
        weights, decode, prefill, tokenizer = out("qwen3.5-4b/weights"), out("qwen3.5-4b/decode_s4096_gdrl_p8e3958"), out("qwen3.5-4b/prefill_m256f_s4096_p8e3958"), out("qwen3.5-4b/tokenizer")
        return Recipe(
            model, scenario, (src("qwen3.5-4b/checkpoint"),), (decode, prefill, weights, tokenizer),
            (
                command("weights", [iron_py, str(scripts / "dump_llm_weights.py"), "--spec", model, "--checkpoint-dir", src("qwen3.5-4b/checkpoint"), "--out", weights]),
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


def plan(config: Path, repo: Path, artifacts: Path, inputs: Path, build: Path) -> list[Recipe]:
    recipes = [recipe_for(row["name"], row["scenario"], repo, artifacts, inputs, build)
               for row in configured_models(config)]
    return recipes


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


def tree_digest(root: Path) -> str:
    if not root.exists():
        return "absent"
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    for item in files:
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
            "whisper-turbo": (("decode_rail/meta.json",),),
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
            "pyannote-community-1": (("config.yaml",), ("segmentation",)),
            "pyannote-3.1": (("config.yaml",),),
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


def validate_recipe_outputs(recipe: Recipe) -> None:
    for raw in recipe.outputs:
        if not Path(raw).exists():
            raise FileNotFoundError(f"{recipe.model}: recipe output missing: {raw}")
    require_members(recipe, Path(recipe.outputs[0]), True)


def authority_digest(recipe: Recipe, repo: Path) -> str:
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
        digest.update(tree_digest(candidate).encode())
    for name in ("MODEL_IRON_PY", "MODEL_EXPORT_PY", "MODEL_PYANNOTE_PY", "MODEL_AIECC"):
        value = os.environ.get(name)
        if value:
            candidate = Path(value)
            digest.update(name.encode())
            digest.update(tree_digest(candidate).encode())
    return digest.hexdigest()


def recipe_key(recipe: Recipe, repo: Path, config: Path, *, authority: str | None = None,
               inputs: str | None = None) -> str:
    h = hashlib.sha256()
    h.update(config.read_bytes())
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


def cache_hit(recipe: Recipe, key: str) -> bool:
    manifest = manifest_path(recipe)
    if not manifest.is_file():
        return False
    try:
        validate_recipe_outputs(recipe)
        record = json.loads(manifest.read_text())
        return record["recipe_key"] == key and record["output_digest"] == output_digest(recipe.outputs)
    except (OSError, KeyError, json.JSONDecodeError, FileNotFoundError):
        return False


def ensure_outputs_ready(recipe: Recipe, key: str) -> None:
    """Live outputs are never mutated until a verified staging build is ready."""
    del recipe, key


def copy_tokenizer(source: Path, target: Path) -> None:
    files = [p for p in source.iterdir() if p.is_file() and (p.suffix == ".json" or p.name.endswith(".model"))]
    if not files:
        raise FileNotFoundError(f"no tokenizer/config JSON files in {source}")
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        shutil.copy2(item, target / item.name)


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, copy_function=link_or_copy, symlinks=False)


def package_gemma4(repo: Path, artifacts: Path, build: Path, out: Path, tokenizer_source: Path, tokenizer_out: Path) -> None:
    checkpoint = artifacts / "gemma4-12b-qat/checkpoint"
    copy_tree(tokenizer_source, checkpoint)
    cmd = [
        sys.executable,
        str(repo / "designs/resident_forward/package_artifact.py"),
        "--out", str(out),
        "--build", str(build / "rf48C"),
        "--meta", str(build / "rf48C" / "meta.json"),
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
            pass
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
        return value.replace(str(live_artifacts), str(staged_artifacts)).replace(str(live_build), str(staged_build))
    return Recipe(recipe.model, recipe.scenario, recipe.inputs,
                  tuple(replace(value) for value in recipe.outputs),
                  tuple(Command(item.step, tuple(replace(value) for value in item.argv),
                                {key: replace(value) for key, value in item.env.items()})
                        for item in recipe.commands))


def write_receipt(recipe: Recipe, key: str) -> None:
    record = {"model": recipe.model, "scenario": recipe.scenario, "recipe_key": key,
              "output_digest": output_digest(recipe.outputs), "commands": [asdict(c) for c in recipe.commands]}
    target = manifest_path(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def no_staging_references(recipe: Recipe, staging_root: Path) -> None:
    needle = str(staging_root).encode()
    for raw in recipe.outputs:
        root = Path(raw)
        for item in root.rglob("*"):
            if item.is_file() and item.suffix in {".json", ".toml", ".txt"} and needle in item.read_bytes():
                raise RuntimeError(f"{recipe.model}: staged path leaked into published metadata: {item}")


def publish_recipe(recipe: Recipe, staged: Recipe, key: str, rollback: Path) -> None:
    validate_recipe_outputs(staged)
    write_receipt(staged, key)
    if not cache_hit(staged, key):
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


def verified_existing_gemma4(recipe: Recipe, key: str) -> bool:
    if recipe.model != "gemma4-12b":
        return False
    try:
        validate_recipe_outputs(recipe)
        meta = json.loads((Path(recipe.outputs[0]) / "meta.json").read_text())
        iron = meta.get("iron")
        if not isinstance(iron, dict) or not all(iron.get(name) is not None for name in ("commit", "dirty", "identity")):
            return False
        for raw in recipe.outputs[3:]:
            if not (Path(raw) / ".recipe-manifest.json").is_file():
                return False
    except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError):
        return False
    write_receipt(recipe, key)
    return cache_hit(recipe, key)


def build_recipe(recipe: Recipe, repo: Path, config: Path, artifacts: Path, build: Path) -> str:
    authority_before = authority_digest(recipe, repo)
    inputs_before = source_digest(recipe.inputs)
    key = recipe_key(recipe, repo, config, authority=authority_before, inputs=inputs_before)
    if cache_hit(recipe, key):
        return "HIT"
    if verified_existing_gemma4(recipe, key):
        return "VERIFIED"
    ensure_outputs_ready(recipe, key)
    stage_root = build / "model-artifact-staging" / f"{recipe.model}-{key[:16]}"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    staged_artifacts, staged_build = stage_root / "artifacts", stage_root / "build"
    staged = rebase_recipe(recipe, artifacts, staged_artifacts, build, staged_build)
    run_commands(staged, repo)
    validate_recipe_outputs(staged)
    no_staging_references(staged, stage_root)
    if authority_before != authority_digest(recipe, repo) or inputs_before != source_digest(recipe.inputs):
        raise RuntimeError(f"{recipe.model}: source authority changed while the staged build ran")
    publish_recipe(recipe, staged, key, artifacts / ".model-artifact-rollback")
    if not cache_hit(recipe, key):
        raise RuntimeError(f"recipe manifest did not verify after publish: {recipe.model}")
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

    if args.plan == args.build:
        parser.error("choose exactly one of --plan or --build")
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
