#!/usr/bin/env python3
"""Load each configured local pyannote checkpoint through the installed SDK."""
import argparse
import json
from pathlib import Path

import pyannote.audio
from pyannote.audio import Model

from pyannote_local_inputs import local_checkpoint_path, validate_local_pipeline_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path)
    args = parser.parse_args()
    pipeline = args.pipeline.resolve()
    components = validate_local_pipeline_inputs(pipeline)
    for component in components:
        if len(component.files) != 1:
            continue
        checkpoint = local_checkpoint_path(pipeline, component.name)
        model = Model.from_pretrained(checkpoint, map_location="cpu")
        if model is None:
            raise RuntimeError(f"{component.name}: SDK returned no model for {checkpoint}")
        print(json.dumps({
            "component": component.name,
            "checkpoint": str(checkpoint),
            "model": f"{type(model).__module__}.{type(model).__qualname__}",
            "pyannote_audio": pyannote.audio.__version__,
            "specifications": repr(model.specifications),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
