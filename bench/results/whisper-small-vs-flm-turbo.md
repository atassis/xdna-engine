# Whisper-small (ours, NPU) vs FastFlowLM whisper-v3:turbo (NPU)

First real head-to-head over **17 FLEURS clips** (4 EN + 13 RU). Scenario: `scenarios/asr-whisper-small.toml`. Run: 2026-06-14T22:58:35+0300.

All backends run **sequentially** on the single-tenant NPU. Latency = wall time of the
transcribe call (after 2 warmups). Energy = RAPL package J/clip. RAM = peak RSS of the
serving process. CPU-idle% = mean CPU idle fraction during the call (higher = more work
offloaded off the CPU, i.e. on the NPU).

| backend | model | EN WER | RU WER | median latency | J/clip | peak RAM | CPU-idle% |
|---|---|---|---|---|---|---|---|
| ours | whisper-small | 0.099 | 0.124 | 2.35s | 59.4 | 2.55 GB | 78.3% |
| flm | whisper-v3:turbo | 0.180 | 0.204 | 2.65s | 42.2 | 5.31 GB | 91.8% |

Reference (CPU whisper-small oracle): EN WER 0.174 / RU WER 0.119.
