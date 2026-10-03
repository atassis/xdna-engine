"""Resident loader metadata derived from a compiled forward layout."""
import json
import os
from pathlib import Path


def metadata(build: Path, store: Path, config: Path) -> dict:
    build, store, config = Path(build), Path(store), Path(config)
    for line in (build / "gen_env.txt").read_text().splitlines():
        key, _, value = line.partition("=")
        if key in {"RF_FAST", "RF_ATTN_H"}:
            os.environ[key] = value
    import rlayer_design as layer
    import rforward as forward
    import rglobal
    import rsplitl
    import rattn2_design as attention

    args = (build / "gen_args.txt").read_text().split()
    if args[0] != "rlayer_design":
        raise ValueError("unsupported forward generator")
    forward.STORE = str(store / "manifest.json")
    layer.configure([arg for arg in args[1:] if not arg.startswith("emit=")])
    layout = json.loads((build / "fwd_layout.json").read_text())
    expected = json.loads(json.dumps(forward.layout(layer)))
    if layout != expected:
        raise ValueError("compiled layout does not match its recorded generator inputs")
    cfg = json.loads(config.read_text())
    cfg = cfg.get("text_config", cfg)
    lo, hi = layout["range"]
    if lo != 0 or hi + 1 != cfg["num_hidden_layers"] or not layout["head"]:
        raise ValueError("layout must include the complete model and head")
    if layer.D != cfg["hidden_size"]:
        raise ValueError("layout hidden size differs from checkpoint")
    rows = (build / "params.txt").read_text().splitlines()
    slots = {name: int(slot) for name, slot, dtype, kind in (row.split() for row in rows[1:])}
    if int(rows[0]) != len(slots) or set(slots) != set(layout["params"]):
        raise ValueError("compiled scratchpad parameters differ from layout")
    if sorted(slots.values()) != list(range(len(slots))):
        raise ValueError("compiled scratchpad slots must be unique and contiguous")
    plan = forward.Plan(layer, lo, hi, True)
    row_block = plan.XROWS // (layer.PCAP_T * layer.D * 2)
    layer.set_geo(False)
    kvrow_s = layer.KVROW
    return {
        "kind": "resident_forward_ladder", "elf": "design.elf", "boot": "boot",
        "weight_dir": "weights", "embedding_store": "store",
        "d_model": cfg["hidden_size"], "nlayer": hi + 1,
        "full_attention_layers": layout["global_layers"],
        "row_block": row_block, "pcap_t": layer.PCAP_T,
        "pmax": row_block * max(rung["nt"] for rung in layout["rungs"]),
        "sliding_window": cfg["sliding_window"],
        **{key: layout[key] for key in ("s_cap", "s_ring", "s_rows", "g_cap", "param_unit_bytes", "s_ring_layout", "rungs")},
        "kvrow_s": kvrow_s, "kvrow_g": rglobal.KVROW,
        "xrows": plan.XROWS, "rope_s_off": plan.RS, "rope_g_off": plan.RG,
        "widths_s_off": plan.WS, "widths_g_off": plan.WG, "widths_bytes": attention.WB,
        "xbuf": layout["XF"], "hidden_slot_bytes": plan.OB, "logits_off": plan.LOGITS,
        "obuf_f1": layout["OF"], "obuf_f2": layout["OF"],
        "layer_weight_off": layout["wbase"], "layer_kv_off": layout["kvbase"],
        "head_off": layout["hbase"], "scratch_bytes": layout["SF"], "cache_bytes": layout["KF"],
        "vocab": cfg["vocab_size"], "logit_softcap": cfg.get("final_logit_softcapping"),
        "scratchpad_params": slots,
        "split_widths_x_off": [plan.RG + rsplitl.widths_x(layer, col) - plan.XROWS for col in range(layer.NC)],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    (args.build / "meta.json").write_text(json.dumps(metadata(args.build, args.store, args.config), indent=2) + "\n")
