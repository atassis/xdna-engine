# Bench results

`bench.py` runs RU+EN ASR models on voxd's fixture and prints warm inference latency + transcripts.
Measured 2026-06-10 on the target machine, **CPU INT8** (onnxruntime CPUExecutionProvider), Python 3.12.

Input: `../../voxd/tests/fixtures/sample-ru-en.wav` — 11.92s, 16kHz mono, RU+EN code-switch.
Reference (what was said): *"observation, дезоксирибонуклеиновая кислота, привет, пока, я хочу понять как
быстро я могу говорить, ла-ла-ла-ла, звуки рандомные"*.

| model | latency | xRT | transcript (abridged) |
|---|---|---|---|
| FLM Whisper turbo (baseline, on NPU) | ~3.4s | 3.5x | "observation, дозок **серебро-нуклеиновая** кисл кислота…" — EN in Latin, RU mangled |
| **gigaam-v3-rnnt** | **0.89s** | 13.4x | "обзервейшн **дозоксирибонуклеиновая кислота**…" — best RU, EN→Cyrillic |
| nemo-fastconformer-ru-rnnt | 0.70s | 17x | "Обзорвэйшн, дозо серебряну клиновая…" — fast, RU weak |
| parakeet-tdt-0.6b-v3 | 1.55s | 7.7x | "Обзорвейшн дозок сериба нуклеиновая…" — multilingual, RU weak |
| t-one | 2.29s | 5.2x | "обзывайшон долосребенуклиновая…" — worst |
| canary-1b-v2 | 2.94s | 4.1x | "observation to the serial nucleic acid. Hello…" — translated RU→EN (mis-config) |

## How to re-run
```bash
# from a python 3.12 venv with onnx-asr installed:
uv venv --python 3.12 .venv
uv pip install --python .venv "onnx-asr[cpu,hub]" soundfile librosa
.venv/bin/python bench.py                 # all models
.venv/bin/python bench.py gigaam parakeet  # subset by id substring
```
Edit `WAV` in `bench.py` if the fixture path differs. The original throwaway env was `~/npuvox-asr-bench`.

## Conclusions
- Model swap **kills the latency floor** even on CPU (0.7–0.9s vs 3.4s). On NPU/dGPU it'd be faster still.
- **GigaAM-v3** = Russian-quality champion. Parakeet-v3 = the multilingual RU+EN option.
- All fast RU models render English in **Cyrillic**; only Whisper keeps Latin. Acceptable for this user.
- None of this runs on the NPU under Linux yet — that's the whole point of this repo (see `../docs/`).
