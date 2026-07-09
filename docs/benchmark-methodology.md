# Evaluation and Benchmark Methodology

This is the reusable NPU-vs-CPU benchmark suite that ships with an open, Linux-native
"ONNX backbone -> AMD XDNA2 NPU" runtime library. It defines the metric set, the measurement
method for each metric (especially real joules on this box), the experiment matrix, and the
reporting format. It is grounded in established accelerator-benchmark methodology (the MLPerf
family, EEMBC) and in first-hand probing of the power/telemetry surface on a Ryzen AI 9 465 /
Krackan / XDNA2 box.

The load-bearing idea: a single "speedup" number is a category error for an accelerator whose
selling point is doing the same work for less power while freeing the CPU. Report energy and
offload alongside latency, never latency alone, with a measurement method rigorous enough that the
energy numbers are real joules and not RAPL-contaminated noise.

---

## 0. The thesis: latency alone is a misleading score for an NPU

The worked example that motivates the whole suite. For the GigaAM encoder, NPU vs
onnxruntime-CPU:

| Metric | NPU | CPU (onnxruntime) | Reading |
|---|---|---|---|
| Wall-clock / inference | ~662 ms | ~563 ms | CPU slightly wins - a near-tie, the NPU even loses |
| CPU-core-seconds / inference (proxy) | ~2.36 | ~10.38 | NPU uses ~4.4x less CPU work |

If the suite reported only latency, it would conclude "the NPU is not worth it here." That is
the wrong conclusion. The NPU's value on this workload is energy and CPU-offload, not speed: it
frees ~4.4x of CPU core-time (those cores can do other work, or clock down / sleep), and it
should draw less total energy. But the NPU's own power draw is an unmeasured term in a core-seconds
proxy (the proxy counts CPU core-seconds, not NPU joules), so an energy verdict built on the proxy
alone is directionally argued, not closed. RAPL package energy (Section 3) is exactly the missing
term, because the on-die NPU is inside the package domain (see Section 3.3).

Conclusion the suite must structurally enforce: report energy plus offload alongside latency,
never latency alone.

---

## 1. The metric axes (and why each matters for an NPU)

| Axis | Definition | Why it matters for an NPU specifically |
|---|---|---|
| Warm latency p50 | Median single-inference wall time, steady-state, after warmup | The headline number, but only one of many |
| Warm latency p95 / p99 (tail) | 95th/99th-percentile latency | NPU dispatch goes through a driver + firmware queue; the context-switch wall (~2.67 ms/switch) and OS scheduling create a fat tail. A p50 that looks fine can hide p99 spikes that break a real-time/streaming SLA. Tail is where NPUs surprise you. |
| Cold / first-inference | Latency of the very first inference, including xclbin load + hw_context create + weight-BO upload | On XDNA2 this is large and distinct: loading the xclbin into the array + creating a hw_context is a one-time cost paid before any compute. For a service that loads-on-demand this is the user-visible latency. Report separately, never fold into the warm number. |
| Throughput | Inferences/sec at the best batching/concurrency the device allows | XDNA2 is single-tenant and this regime is latency-bound, not throughput-saturating (dispatch-bound, <1% array util). Throughput is roughly 1/latency here; report it but know it is not the device's strong axis. |
| Energy / inference (joules) | Total package energy consumed per inference, idle-baseline-subtracted | The real apples-to-apples accelerator metric. Directly answers "did offloading to the NPU save energy vs the CPU?" |
| Perf-per-watt | Inferences/sec / average watts (or its inverse, J/inf) | The standard edge-ML figure of merit. Lets you compare NPU vs CPU vs (future) iGPU on equal footing. |
| CPU-offload | CPU-core-seconds consumed per inference (and peak cores busy) | The axis where the NPU already provably wins ~4.4x. Measures how much CPU an inference cost, i.e. how many cores are freed to do other work. Cheap to measure (Section 4), high signal. |
| Thermal-sustained throughput | Steady-state throughput over a long (minutes) run vs the first-N-run burst | Laptop XDNA2 shares a power/thermal budget with CPU + iGPU. A 30-run burst can look great, then throttle. Run long enough to hit steady thermal state and report sustained vs peak. |
| Determinism / jitter | Stddev, IQR, min/max of warm latency; ideally per-inference, not just summary | Real-time ASR/streaming needs bounded jitter, not just a good mean. Driver-queue + context-switch variance shows up here. |
| Accuracy at bf16/int8 vs f32 | Task metric (e.g. WER for ASR) at the deployed dtype vs an f32 reference | An NPU "win" that silently degrades accuracy is not a win. The matmul path here is bf16 + f32-accumulate; verify the speed/energy gain does not cost task quality. Pair every perf row with an accuracy row. |
| Model-switch cost | Latency to swap the resident model/xclbin set (unload + load + first-inference) | XDNA2 is single-tenant with a bounded hw-context budget and a per-distinct-shape switch wall. A multi-model or hot-swap service pays this; it must be a first-class measured number, not an afterthought. |

Misleading-latency summary: latency hides (a) energy - the NPU may be slower and lower-energy;
(b) offload - the NPU frees CPU even at equal wall-time; (c) the tail - p50 lies about p99;
(d) cold cost - warm latency excludes the load you actually pay; (e) thermals - burst lies about
sustained.

---

## 2. Established methodologies to borrow from

| Source | What to borrow | How it maps here |
|---|---|---|
| MLPerf Inference (datacenter/edge) | Warmup then steady-state; report percentiles not just mean; well-defined scenarios (single-stream = p50/p99 latency; offline = throughput). MLPerf's latency-bounded runs report a percentile constraint (e.g. "99% of queries under X ms"), not an average - exactly the tail discipline an NPU needs. | Adopt single-stream (latency) + offline (throughput) as the two core scenarios. Adopt "X warmup, N timed, report p50/p95/p99". |
| MLPerf Power / power-measurement ruleset | The rules that make a power number trustworthy: measure at steady state, over a fixed measurement window aligned to the timed run, report average watts AND total energy, and subtract/disclose the idle baseline. MLPerf uses external meters; the analog here is RAPL with the same disciplines (window alignment, baseline subtraction). | The energy method (Section 3) mirrors these rules with RAPL instead of an external meter. Disclose that RAPL is package-domain estimation, not wall-socket truth. |
| MLPerf Tiny / Mobile / EEMBC MLMark / EEMBC ULPMark | Perf-per-watt and energy-per-inference as the headline for edge/embedded, not raw latency. MLPerf Tiny explicitly co-reports accuracy + latency + energy. EEMBC pioneered energy-per-workload on a fixed harness. | Validates the thesis: the suite's headline is J/inference and perf/W, with latency as a supporting column. Co-report accuracy on the same row (MLPerf Tiny pattern). |
| General accelerator-bench hygiene | Pin/disclose clocks (governor, AC vs battery), fix input set, run multiple repetitions on separate process invocations (cold) + within-process (warm), report environment (driver/FW/XRT versions). | Pin the `performance` governor where possible, always run on AC (battery clocking differs and `power_now`=0 on AC anyway - Section 3.4), record FW 1.1.2.64 / XRT 2.21.75 / amdxdna driver in every result. |

Established practice is unanimous on the load-bearing point: for edge/NPU inference the figure of
merit is energy-per-inference and perf-per-watt with accuracy co-reported; latency is a
constraint, not the score.

---

## 3. Measuring real joules on this box (the hard part)

This is the section that turns the suite from "latency comparison" into "accelerator benchmark."
All findings below are first-hand probed on the target box (Ryzen AI 9 465 / Krackan).

### 3.1 RAPL via /sys/class/powercap/ - the primary energy method (verified working)

The box exposes (despite the `intel-rapl` name on an AMD chip - the kernel mislabels the AMD RAPL
MSR interface under the `intel-rapl` powercap class; the counters are real):

```
/sys/class/powercap/intel-rapl:0/energy_uj        # name="package-0"  package energy, microjoules
/sys/class/powercap/intel-rapl:0:0/energy_uj      # name="core"       core(s) subdomain energy
/sys/class/powercap/intel-rapl:0/max_energy_range_uj = 65532610987    # wrap point
```

Method (energy-counter delta around a timed window):

```
E0 = read(package energy_uj);  t0 = clock_monotonic()
... run the timed N-inference window ...
E1 = read(package energy_uj);  t1 = clock_monotonic()
dE_uj   = (E1 - E0) mod max_energy_range_uj      # handle wrap (Section 3.5)
avg_W   = dE_uj / 1e6 / (t1 - t0)
J_per_inf = (dE_uj / 1e6) / N                    # then subtract idle baseline (Section 3.6)
```

Verified live on this box: a controlled idle-vs-busy probe read the package counter at 6.56 W
idle and 26.46 W under a 4-core busy loop over a ~1.5 s window - the counter is monotonically
advancing and clearly load-sensitive. RAPL is the suite's primary energy instrument here, not a
theoretical option.

### 3.2 The root-only permission issue - not present on this box (no chmod needed)

The well-known RAPL hardening (a Spectre-class side-channel mitigation, CVE-2020-8694, that makes
`energy_uj` root-only `0400`) is not applied here: the files are world-readable `0444 root:root`,
and a non-root user (uid 1000) read them successfully. So the suite does not need the usual
workaround on this machine. Still, ship the workaround for portability, and prefer the non-sudo
path:

- Documented fix where files are `0400`: `sudo chmod -R a+r /sys/class/powercap/intel-rapl*/`
  (transient; reverts on reboot - add a udev/tmpfiles rule for permanence).
- The suite should detect unreadable RAPL and emit the exact chmod command for the user to run,
  then degrade gracefully (Section 4 fallbacks). Do not run `sudo` from the harness itself.

### 3.3 Does package RAPL include the on-die XDNA2 NPU? (likely yes - unverified directly)

The XDNA2 NPU is an on-die IP block sharing the package power rail with the Zen5 cores and the
Radeon 880M iGPU. The `package-0` RAPL domain is the whole-package energy accumulator; on AMD APUs
it is the package rail, which physically powers the NPU. So package RAPL energy should capture NPU
activity, which is precisely why it is the right instrument to close the missing energy term: run
the same workload NPU-path vs CPU-path, diff `package energy_uj`, and the difference is the
energy-offload story including the NPU's own draw.

Two caveats, both flagged unverified:

- There is no separate RAPL subdomain for the NPU (only `package-0` and `core` exist - no `dram`,
  no `uncore`/`pp1`, no NPU domain). You get package total and a core subset; the NPU's
  contribution is `package - (everything else)`, not directly isolable from RAPL alone.
- Whether AMD's package-RAPL model actually integrates NPU power draw, vs only modelling CPU + SoC,
  is not separately documented and has not been isolated here. A clean way to test it (on-box):
  hold CPU/iGPU load constant, toggle a heavy NPU-only workload, and see if `package-0` energy
  rises beyond core + idle. Until run, treat "package RAPL includes the NPU" as likely but
  unconfirmed. This is the single most important open verification for the energy methodology.

### 3.4 Battery sysfs (true discharge watts) - unavailable on AC (the usual state)

`/sys/class/power_supply/BAT0/` exists and exposes `power_now` (uW, native - this battery reports
energy-based `energy_now`/`power_now`, so no `current_now x voltage_now` multiply is needed; in
fact `current_now` does not exist on this battery, only `power_now`/`energy_now`/`voltage_now`).
But `AC0/online = 1` and `power_now = 0`: plugged in, the battery is not discharging, so it reports
zero draw. Battery-discharge wattage is therefore a fallback only usable when unplugged - and you
generally do not want to bench on battery because:

- On battery the SoC clocks differently (power/thermal policy changes) - results are not comparable
  to AC.
- Battery `power_now` is whole-system (display, wifi, SSD, ...), much noisier than RAPL package.

Rule: bench on AC, use RAPL. Keep battery `power_now` only as a sanity cross-check on an unplugged
run (where it gives true wall draw including everything RAPL misses, e.g. DRAM/display), with a
heavy idle-baseline subtraction. For this battery, watts = `power_now / 1e6` directly (no I x V).

### 3.5 Counter wrap

`max_energy_range_uj = 65532610987` (~65.5 GJ-worth, i.e. ~65532 J before wrap). At ~26 W the
package counter wraps in ~42 minutes; at higher draw, sooner. For short timed windows wrap is rare
but a long thermal-sustained run will hit it. Always compute `dE = (E1 - E0 + max) mod max` so a
single wrap is handled. For very long runs, sample the counter periodically (e.g. every few
seconds) and sum deltas, rather than one start/end pair.

### 3.6 Idle-baseline subtraction (mandatory)

Idle package draw here is ~6.5 W. Per-inference energy must subtract the idle energy over the same
window: `J_active = J_measured - idle_W x window_s`. Measure idle in the same session, same
governor, same AC state, immediately before/after the run (idle drifts with temperature). Report
both the raw and baseline-subtracted numbers - the discipline is to disclose the baseline, not hide
it.

### 3.7 XDNA2 / amdxdna NPU-specific power telemetry - absent on this box

Probed exhaustively; there is no usable NPU power telemetry on this stack today:

- `xrt-smi examine` reports only `aie-partitions`, `host`, `platform`. The platform report does
  have a `Power Mode: Default` field and an `Estimated Power : N/A` field - i.e. the
  firmware/driver has a power-estimation hook, but it returns N/A on this box (not wired up for
  FW 1.1.2.64).
- No `hwmon` node under `/sys/class/accel/accel0/device/` (only PCI runtime-PM stats:
  `runtime_active_time`, `runtime_status` - useful for utilisation/duty-cycle, not power).
- `amd-smi` is not installed (no pacman/ROCm binary). Even where present, `amd-smi` NPU power on
  XDNA2-under-Linux is unproven; do not bank on it.

So per-NPU joules are not directly readable - package RAPL (Section 3.3) is the only path to NPU
energy, by difference. The `runtime_active_time` counter under accel0 is worth capturing as an NPU
duty-cycle / busy-time signal to correlate against package-energy deltas.

### 3.8 Pitfalls checklist

- Counter wrap - handle mod arithmetic (3.5).
- Idle-baseline subtraction - mandatory, measured in-session (3.6).
- Contention - bench single-tenant (XDNA2 is single-tenant anyway); pin/quiet background load; RAPL
  captures all package activity, so a background process pollutes the NPU-vs-CPU diff. Quiesce the
  machine, close the resident serving process when benching the alternative path.
- AC vs battery clocking - always AC; battery changes clocks and zeroes `power_now` (3.4).
- NPU-in-package unverified - Section 3.3; flag every energy verdict accordingly until isolated.
- `intel-rapl` label on AMD - cosmetic, but document it so a reader does not think it is wrong HW.
- Frequency/governor drift and thermal - pin governor, warm the machine to steady thermal state
  before the energy window, or you measure the cooldown not the workload.

---

## 4. The reusable harness: measurement method per metric

The suite is one library-shipped CLI/module (`npu-bench`) that wraps any registered workload and
any registered backend behind a uniform timer + sampler. Per-metric method:

| Metric | How the harness measures it |
|---|---|
| Warm p50/p95/p99, jitter | Warmup K runs (default K=5), then N timed runs (default N=100) within one process (warm = caches/xclbin/context resident). Record every per-inference `clock_monotonic` delta; compute percentiles + stddev/IQR/min/max from the full vector, not a running mean. |
| Cold / first-inference | Measure the first inference of a fresh process (new device open, fresh xclbin load + hw_context create + weight upload). Repeat over M fresh process invocations (default M=10) for a distribution, since cold cost itself has variance. |
| Throughput | Offline scenario: submit N inferences back-to-back, throughput = N / total_wall. (Single-tenant means no real concurrency knob; report 1/p50 as the latency-bound ceiling too.) |
| Energy / inference, perf/W | Read `package energy_uj` (+ `core`) and `clock_monotonic` immediately before/after the timed warm window; `J/inf = (dE - idle_W . window) / N`; `perf/W = (N/window)/avg_W`. Long runs: periodic sampling + summed deltas (3.5). Co-record battery `power_now` if unplugged. |
| CPU-offload (core-seconds) | Wrap the timed window with `getrusage(RUSAGE_SELF/CHILDREN)` (utime + stime) or read `/proc/self/stat` + thread CPU time; `core_sec/inf = delta_cpu_time / N`. This reproduces the offload proxy (NPU ~2.36 vs CPU ~10.38). Also sample `/proc/stat` system-wide to catch driver/firmware-thread CPU. Capture `runtime_active_time` (accel0) as NPU duty-cycle. |
| Thermal-sustained | Run the warm loop for a fixed long duration (default 5 min). Bucket results into time windows; report first-30s throughput vs last-30s throughput and the trend. Sample `k10temp`/`acpitz` hwmon temps and package watts across the run. |
| Accuracy @ dtype | Run the workload's task metric (WER for ASR) at the deployed dtype, compare to an f32/reference run. One accuracy row per perf row. |
| Model-switch cost | Time `unload(modelA) + load(modelB) + first_inference(modelB)`; report as its own latency distribution. Drives the multi-model/hot-swap cost story. |

Graceful degradation (must-have): the harness detects (a) RAPL unreadable -> print the exact
non-sudo chmod command, fall back to CPU-core-seconds + a "no-energy" flag in the report; (b) on
battery -> warn and either refuse or tag results "battery, non-comparable"; (c) no NPU telemetry ->
expected, proceed with RAPL-by-difference. Never silently drop a missing metric - emit it as `N/A`
with the reason, so a reader knows energy was unavailable, not zero (mirrors xrt-smi's own honest
`Estimated Power: N/A`).

---

## 5. The experiment matrix (workload x backend x input-size)

| Dimension | Levels (initial) |
|---|---|
| Backend | `npu` (XDNA2 via the XRT path) . `cpu-ort` (onnxruntime CPUExecutionProvider, the CPU baseline) . (future) `cpu-native` (a rayon host path) . (future) `igpu` (Radeon 880M) |
| Workload | GigaAM-v3 encoder (validated end-to-end) . a single whole-array bf16 GEMM microbench (isolates dispatch/switch cost) . (future) Parakeet encoder . a "model-switch" pair (two distinct xclbin shape sets) |
| Input size | The frozen static window (1x64x1600 -> 1x768x400, 16 s) as primary; plus a fixed 11.92 s fixture (so CPU numbers stay comparable). Vary batch where the backend allows. |
| dtype | NPU: bf16 (+ f32 accumulate) - the validated path. CPU-ort: int8 and f32. Pair each with its accuracy row. |

Two tiers:

- Microbench (single GEMM / single dispatch): isolates the context-switch wall and per-dispatch
  floor from end-to-end noise - the cleanest place to see NPU energy-per-op and dispatch jitter.
- End-to-end (full encoder): the number that matters to a user; the place the energy-vs-latency
  thesis (Section 0) actually pays off.

Hold constant across every cell: AC power, governor, FW/driver/XRT versions, quiesced background,
same input set, same warmup/timed counts. Vary exactly one axis per comparison.

---

## 6. How results should be reported

Each run emits a machine-readable JSON record (for regression tracking across commits - the suite
ships with the library, so it must gate "did this change regress energy/latency?") plus a
human-readable table. Minimum fields per (workload x backend x input x dtype) cell:

```
{ workload, backend, input_size, dtype,
  latency_ms: {p50, p95, p99, mean, std, min, max},
  cold_ms:    {p50, p95, n_proc},
  throughput_infps,
  energy:     {j_per_inf, j_per_inf_baseline_subtracted, avg_W, idle_W, perf_per_W,
               source:"rapl-package-0", npu_in_package:"assumed-unverified"},
  cpu_offload:{core_sec_per_inf, peak_cores, npu_active_ms},
  thermal:    {sustained_infps_first30s, sustained_infps_last30s, max_pkg_W, max_temp_C},
  accuracy:   {metric:"WER", value, ref_dtype:"f32", delta},
  model_switch_ms: {p50} | null,
  env: {fw:"1.1.2.64", xrt:"2.21.75", driver:"amdxdna", governor, ac:true},
  caveats: ["energy=package-RAPL incl-NPU-assumed", ...] }
```

Headline view (the one a human reads first) - not a single speedup number, but the four-axis card
per backend, because that is the only honest summary for an accelerator:

```
GigaAM-v3 encoder, 16s window, AC, perf governor
  backend   p50      p99     J/inf*   core-sec/inf   WER     verdict
  npu       662 ms   ~ms     ?? J     2.36           x.x%    offload-win (4.4x less CPU); energy TBD on RAPL
  cpu-ort   563 ms   ~ms     ?? J    10.38           x.x%    latency-win, CPU-heavy
  * J/inf = package RAPL, idle-subtracted; NPU-in-package assumed (Section 3.3)
```

Reporting rules: (1) always show energy + offload columns next to latency - the suite's reason for
existing; if energy is unavailable, show `N/A (reason)`, never blank. (2) p99 next to p50 - never
p50 alone. (3) accuracy on the same row as the perf it was measured at. (4) every energy figure
carries its caveat tag (`package-RAPL`, `NPU-in-package-assumed`) so no reader over-trusts it.
(5) JSON is the source of truth for CI regression gating; the table is a render of it.

---

## 7. Flagged unverified / open items

- [Blocking for energy claims] Does `package-0` RAPL actually include XDNA2 NPU draw? Argued likely
  (on-die, shared package rail) but not isolated. Verification experiment specified in 3.3 - run it
  before publishing any NPU-energy verdict. Until then every J/inf is tagged "NPU-in-package
  assumed."
- NPU has no direct power telemetry on this stack (xrt-smi `Estimated Power: N/A`, no accel0 hwmon,
  no amd-smi). Energy is RAPL-by-difference only. If a future FW/driver wires up `Estimated Power`,
  add it as a cross-check.
- The offload core-seconds (2.36 vs 10.38) and wall-times (662 vs 563 ms) should be re-derived with
  the harness's `cpu_offload` path to confirm before quoting as suite output.
- RAPL is package-domain estimation, not wall-socket truth - it omits DRAM (no `dram` subdomain
  here), display, SSD, etc. Battery `power_now` (unplugged only) is the closest to true wall draw
  and is the recommended occasional cross-check.
- The `intel-rapl` label on AMD silicon is a kernel cosmetic, not a wrong-hardware signal -
  documented to preempt confusion.
- The Krackan thermal/power budget differs from Strix; sustained-throughput throttling behavior is
  workload- and chassis-specific and must be measured, not assumed.

---

### Sources / grounding

- First-hand box probes: RAPL domains and live idle-vs-busy delta, BAT0/AC sysfs, accel0 sysfs,
  `xrt-smi examine` reports.
- MLPerf Inference and MLPerf Power rules (warmup/steady-state/percentile/power-measurement
  discipline): https://mlcommons.org/benchmarks/inference-edge/ and the MLPerf power-measurement
  methodology.
- MLPerf Tiny (co-reported latency + energy + accuracy for edge):
  https://mlcommons.org/benchmarks/inference-tiny/
- EEMBC MLMark / ULPMark (energy-per-workload harness lineage): https://www.eembc.org/mlmark/
- Linux RAPL powercap interface and the CVE-2020-8694 root-only hardening: kernel
  `Documentation/power/powercap/powercap.rst`.
