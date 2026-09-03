# RETIRED 2026-09-01

This package was the GigaAM-v3 reference encoder that `rust/npu-asr` was ported from. It is frozen at
2026-06-11; the Rust crate moved to 2026-08-24 and **84% of it (4154 lines) has no counterpart here**.

It is kept for provenance, not as a spec. Do NOT restore structural parity with it: `block.py` is an
op-at-a-time facade (18 `ops.*` calls, one host round-trip each) and `rust/npu-asr/src/block.rs` is
built on resident contexts ("one resident xclbin, zero switches"). Per-dispatch cost is 90.2% of a
decode step, so mirroring this structure would reintroduce exactly what the decode work deletes.

The numerical oracle is ONNX, not this package. `scripts/asr_service.py` is retired too --
`install.sh:97-99` records that Rust `npu serve` replaced it.
