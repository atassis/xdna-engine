from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineComponent:
    name: str
    source: str | None
    files: tuple[Path, ...]


def _pipeline_params(root: Path) -> dict:
    config = root / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"local pyannote input lacks {config}")
    pipeline_indent = None
    params_indent = None
    params_item_indent = None
    params = {}
    for raw in config.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip())
        key, separator, value = line.lstrip().partition(":")
        if not separator:
            continue
        if pipeline_indent is None:
            if indent == 0 and key == "pipeline":
                pipeline_indent = indent
            continue
        if params_indent is None:
            if indent <= pipeline_indent:
                break
            if key == "params" and not value.strip():
                params_indent = indent
            continue
        if indent <= pipeline_indent:
            break
        if params_item_indent is None and value.strip():
            params_item_indent = indent
        if indent == params_item_indent and value.strip():
            params[key] = value.strip().strip("\"'")
    if params_indent is None:
        raise ValueError(f"local pyannote pipeline config lacks pipeline.params: {config}")
    return params


def _component(root: Path, name: str, reference: object, filenames: tuple[str, ...]) -> PipelineComponent:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"local pyannote pipeline config lacks {name}")
    if reference.startswith("$model/"):
        location = root / reference.removeprefix("$model/")
        source = None
    else:
        if "/" not in reference:
            raise ValueError(f"local pyannote {name} authority is not a Hugging Face repo id: {reference!r}")
        location = root / name
        source = reference
    return PipelineComponent(name, source, tuple(location / filename for filename in filenames))


def pipeline_components(root: Path) -> tuple[PipelineComponent, ...]:
    params = _pipeline_params(root)
    components = (
        _component(root, "segmentation", params.get("segmentation"), ("pytorch_model.bin",)),
        _component(root, "embedding", params.get("embedding"), ("pytorch_model.bin",)),
    )
    if "plda" in params:
        components += (_component(root, "plda", params["plda"],
                                  ("xvec_transform.npz", "plda.npz")),)
    return components


def local_checkpoint_path(root: Path, component: str) -> Path:
    for item in pipeline_components(root):
        if item.name == component:
            if len(item.files) != 1:
                raise ValueError(f"local pyannote {component} does not have one checkpoint")
            path = item.files[0]
            if not path.is_file():
                raise FileNotFoundError(f"local pyannote input lacks {path}")
            return path
    raise ValueError(f"local pyannote pipeline has no {component} component")


def validate_local_pipeline_inputs(root: Path) -> tuple[PipelineComponent, ...]:
    components = pipeline_components(root)
    for component in components:
        for path in component.files:
            if not path.is_file():
                raise FileNotFoundError(f"local pyannote input lacks {path}")
    return components
