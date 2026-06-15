import requests

from bench.backends import FLM, Backend, ours, transcribe


class _FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"text": "hi"}


def test_transcribe_returns_text_and_latency(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp())
    # bench/__init__.py is a real, openable file in the worktree.
    text, dt = transcribe("u", "bench/__init__.py", "m")
    assert text == "hi"
    assert isinstance(dt, float)
    assert dt >= 0


def test_backend_transcribe_method(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp())
    b = Backend("x", "u", "m")
    text, dt = b.transcribe("bench/__init__.py")
    assert text == "hi"
    assert isinstance(dt, float)


def test_flm_backend_metadata():
    assert FLM.name == "flm"
    assert FLM.url == "http://127.0.0.1:11434/v1/audio/transcriptions"
    assert FLM.model == "whisper-v3:turbo"
    assert FLM.start_cmd == ["systemctl", "--user", "start", "flm-asr.service"]
    assert FLM.stop_cmd == ["systemctl", "--user", "stop", "flm-asr.service"]


def test_ours_backend_metadata():
    b = ours(9000)
    assert b.name == "ours"
    assert b.url == "http://127.0.0.1:9000/v1/audio/transcriptions"
    assert b.model == "whisper-small"
    assert ours(9000, "whisper-base").model == "whisper-base"
