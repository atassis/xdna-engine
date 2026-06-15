"""ASR backends: FLM + our engine, same OpenAI-style /v1/audio/transcriptions API."""
import requests
import time


def transcribe(url, wav_path, model):
    t0 = time.perf_counter()
    with open(wav_path, "rb") as f:
        r = requests.post(url, files={"file": f}, data={"model": model}, timeout=600)
    r.raise_for_status()
    return r.json()["text"], time.perf_counter() - t0


class Backend:
    def __init__(self, name, url, model, start_cmd=None, stop_cmd=None):
        self.name, self.url, self.model = name, url, model
        self.start_cmd, self.stop_cmd = start_cmd, stop_cmd

    def transcribe(self, wav):
        return transcribe(self.url, wav, self.model)


# FLM: served via flm-asr.service or `flm serve`; single-tenant so it must be
# stopped to free the NPU for ours.
FLM = Backend(
    "flm",
    "http://127.0.0.1:11434/v1/audio/transcriptions",
    "whisper-v3:turbo",
    start_cmd=["systemctl", "--user", "start", "flm-asr.service"],
    stop_cmd=["systemctl", "--user", "stop", "flm-asr.service"],
)


def ours(port, model_name="whisper-small"):
    return Backend("ours", f"http://127.0.0.1:{port}/v1/audio/transcriptions", model_name)
