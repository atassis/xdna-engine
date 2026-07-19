# Cascade accessor -- on-device test (aie2p)

Validates `aie_api/cascade.hpp` (`aie::cascade_out` / `aie::cascade_in_i32`, the
`atassis/aie_api feat/adf-free-cascade` PR) on real aie2p silicon.

## What it does
3 compute cores in a cascade chain, connected by `aie.cascade_flow`, each using
the aie_api accessor instead of raw `put_mcd`/`get_scd`:
- `kernel1` seeds lane 0 = 14, `aie::cascade_out`.
- `kernel2` `aie::cascade_in_i32`, lane0 += 100 (-> 114), `aie::cascade_out`.
- `kernel3` `aie::cascade_in_i32`, writes `buf[5] = lane0 + 100` (-> 214).
- Host checks `buf[5] == 214` -> the value crossed both cascade hops through the
  accessor. (Structure mirrors mlir-aie `test/npu-xrt/cascade_flows`, retargeted
  to aie2p and routed through the typed accessor.)

## Status (prepared, NOT yet run on device)
- The 3 kernels COMPILE CLEAN for aie2p (Peano + fork aie_api) and emit the real
  cascade moves (`vmov mcd,x0` / `vmov x0,scd`) -- verified on CPU.
- `cascade.mlir` (device `npu2_2col`), `test.cpp` (host), `build.sh`, `run.sh`
  are ready. The xclbin build + on-silicon run are the remaining step.

## To run (in a coordinated device window)
1. Pause the decode session (shared NPU + shared toolchain): `systemctl --user stop npu-serve.service`.
2. `./build.sh`  -> cascade.xclbin, insts.bin, test.exe.
3. `./run.sh`    -> expect `PASS!`.
4. Restart decode: `systemctl --user start npu-serve.service`.

## Risks to check at build/run time (this was prepared blind)
- **Cascade direction on aie2p**: LARGELY DE-RISKED by source review. The lower-
  cascade-flows pass requires dst to be East (col+1) or South (row-1) of src;
  both flows here qualify -- `(0,3)->(1,3)` is East, `(1,3)->(1,2)` is South. And
  NPU2 (aie2p) inherits AIE2TargetModel with a 512-bit accumulator/cascade
  (= 16x int32, the accessor's width). So the topology is valid by the pass's own
  rule and it is the proven-on-aie2 layout. If place-tiles still objects, adjust
  the tile coords in `cascade.mlir`.
- **aiecc peano flag**: the `--peano` invocation in build.sh is best-effort;
  cross-check against `cascade_flows/run.lit` + live `aiecc --help` if rejected.
- **Read-race**: if `buf[5]` reads back as 0/stale, suspect the known unfenced
  xdna-driver CLFLUSH host-only-BO read race, not a cascade failure -- read the
  output buffer twice (a cold-zero first read is the tell) or use a fenced
  driver runtime to disambiguate.
