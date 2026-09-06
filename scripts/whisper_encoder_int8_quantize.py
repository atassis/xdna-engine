#!/usr/bin/env python3
"""Dynamic int8 PTQ of Whisper ONNX graphs (CPU, no NPU).

Per-channel (per-output-column) weight scale, computed statically from the weight tensor
(max|W|/127, the SAME rule ctx2.rs's int8 encoder path uses); activations dynamically
quantized per ORT's DynamicQuantizeLinear insertion, matching ctx2.rs's per-row dynamic
activation quant conceptually (both derive the scale from the actual input at run time,
not a calibration set). CPU execution, not the (unbuilt) NPU int8 kernel at this shape.

Usage: python scripts/whisper_encoder_int8_quantize.py <model-dir-name> [enc|dec|both]
Reads:  artifacts/<model>/onnx/{encoder_model,decoder_model,decoder_with_past_model}.onnx
Writes: artifacts/<model>/onnx_quant/{encoder_model_int8,decoder_model_int8,decoder_with_past_model_int8}.onnx
"""
import sys
import time
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL = sys.argv[1] if len(sys.argv) > 1 else "whisper-turbo"
WHICH = sys.argv[2] if len(sys.argv) > 2 else "both"
SRC_DIR = Path("artifacts") / MODEL / "onnx"
DST_DIR = Path("artifacts") / MODEL / "onnx_quant"
DST_DIR.mkdir(parents=True, exist_ok=True)

STEMS = {"enc": ["encoder_model"], "dec": ["decoder_model", "decoder_with_past_model"]}
stems = STEMS["enc"] + STEMS["dec"] if WHICH == "both" else STEMS[WHICH]

for stem in stems:
    src = SRC_DIR / f"{stem}.onnx"
    dst = DST_DIR / f"{stem}_int8.onnx"
    t0 = time.time()
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
        use_external_data_format=True,
    )
    print(f"quantized {src} -> {dst} in {time.time()-t0:.1f}s")
