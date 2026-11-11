"""Metadata for the scratchpad sidecar belonging to one compiled sequence."""
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from designs.resident_forward.scratchpad_params import read_params


def sequence_scratchpad_metadata(sequence, output_dir):
    from iron.common.compilation.base import _aiecc_work_dir

    source = _aiecc_work_dir(sequence.artifacts[0].mlir_input.filename) / "params.txt"
    contract = read_params(source)
    parameters = {}
    for name, (slot, dtype, kind, minimum, maximum) in contract.parameters.items():
        if slot is None:
            continue
        if dtype != "i32" or kind not in {"addr", "core"}:
            raise ValueError(f"LLM scratchpad parameter {name} requires i32 addr/core encoding")
        parameters[name] = {"byte_offset": slot * 4, "dtype": dtype, "kind": kind,
                            "minimum": minimum, "maximum": maximum}
    # Keep the complete command/range contract, including parameters with no allocated slot.
    shutil.copy2(source, Path(output_dir) / "params.txt")
    return parameters
