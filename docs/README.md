# Engineering deep-dives

Longer-form notes on how the engine is built and why. Start with the data-movement
thesis - it is the frame everything else follows from.

- [data-movement-thesis.md](data-movement-thesis.md) - why this pipeline is
  data-movement-bound, not compute-bound, and what that implies for optimization.
- [where-time-goes.md](where-time-goes.md) - a precise accounting of NPU encoder
  time and energy: most of it is avoidable overhead, not compute.
- [aie2p-architecture-and-roofline.md](aie2p-architecture-and-roofline.md) - the XDNA2
  hardware's fixed set of capabilities across five layers (compute/movement/memory/
  orchestration/format) and a roofline model for picking the right one per regime.
- [execution-graph.md](execution-graph.md) - which hardware primitive to use at each
  node of the encoder/decoder/lm-head/vision graph, keyed on the compute regime.
- [case-study-on-npu-logits.md](case-study-on-npu-logits.md) - moving the Whisper
  decoder's lm-head and argmax onto the NPU: the dead-ends and the fix.
- [benchmark-methodology.md](benchmark-methodology.md) - how the NPU-vs-CPU numbers are
  measured (RAPL energy, quiesce + idle-subtract) so they are reproducible.
- [measurement.md](measurement.md) - what every served request records about itself, the
  attribution rules behind those numbers, and the JSONL run-log format.
- [s2-ar-graph-map.md](s2-ar-graph-map.md) - inventory of the S2 autoregressive
  forward pass: every operation with tensor names, shapes, brick assignments, and L1
  sizing notes.
- [s2-bd-gather-feasibility.md](s2-bd-gather-feasibility.md) - whether AIE2P's DMA/BD
  engine supports runtime-offset embedding gathers, verified against toolchain sources
  and hardware.
- [s2-weight-blob-format.md](s2-weight-blob-format.md) - the on-disk format for
  S2-Pro's dequantized weight blob and JSON manifest: layout specification and
  verification procedures.
