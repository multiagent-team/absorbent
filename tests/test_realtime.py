"""Тесты локального backend'а Voxtral Realtime: сервер vLLM, WS-клиент, нарезка.

Обмен по протоколу проверяется на настоящем локальном WebSocket-сервере
(поднимается на 127.0.0.1:0 в отдельном потоке и повторяет поведение vLLM);
всё остальное — подделки subprocess/urlopen. GPU не нужен.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import subprocess
import threading
from pathlib import Path

import pytest
from websockets.asyncio.server import serve

import absorbent.realtime as realtime
from absorbent.config import Provider
from absorbent.realtime import RealtimeError

REAL_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def _provider(**kwargs) -> Provider:
    base = {"name": "vllm-voxtral", "base_url": "http://127.0.0.1:12435", "api_key_env": ""}
    return Provider(**{**base, **kwargs})


def _models_payload(*ids: str) -> dict:
    return {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}


class _FakeResponse:
    """Ответ urlopen: контекстный менеджер с read()."""

    def __init__(self, body: str) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def read(self) -> bytes:
        return self.body.encode("utf-8")


class _FakeProcess:
    """Подделка процесса vLLM: pid, poll() и wait()."""

    def __init__(self, returncode: int | None = None, wait_error: Exception | None = None) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.wait_error = wait_error
        self.waits: list[float | None] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.wait_error is not None:
            raise self.wait_error
        return 0


class TestConstantsAndPaths:
    def test_max_model_len_scales_with_chunk(self) -> None:
        # 300 с * 12.5 токена/с + 2048 = 5798.
        assert realtime.max_model_len(300.0) == 5798
        # Мелкий чанк упирается в минимальное окно.
        assert realtime.max_model_len(10.0) == 4096

    def test_cache_paths(self, tmp_path: Path) -> None:
        assert realtime.cache_root(tmp_path) == tmp_path / ".absorbent"
        assert realtime.flat_cache_dir(tmp_path, "lecture") == tmp_path / ".absorbent" / "lecture"

    def test_vllm_log_path_uses_xdg_cache(self, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert realtime.vllm_log_path() == tmp_path / "absorbent" / "vllm.log"

    def test_vllm_log_path_defaults_to_home_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert realtime.vllm_log_path() == Path.home() / ".cache" / "absorbent" / "vllm.log"


class TestProbeServer:
    def test_returns_payload_and_strips_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []

        def _fake_urlopen(url, timeout=None):
            seen.append(url)
            return _FakeResponse(json.dumps(_models_payload(REAL_MODEL)))

        monkeypatch.setattr(realtime.urllib.request, "urlopen", _fake_urlopen)
        payload = realtime.probe_server("http://127.0.0.1:12435/")
        assert payload is not None and len(payload["data"]) == 1
        assert seen == ["http://127.0.0.1:12435/v1/models"]

    def test_connection_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(url, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(realtime.urllib.request, "urlopen", _boom)
        assert realtime.probe_server("http://127.0.0.1:12435") is None

    def test_non_json_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(realtime.urllib.request, "urlopen",
                            lambda url, timeout=None: _FakeResponse("<html>nope</html>"))
        assert realtime.probe_server("http://127.0.0.1:12435") is None

    def test_json_list_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(realtime.urllib.request, "urlopen",
                            lambda url, timeout=None: _FakeResponse("[1, 2]"))
        assert realtime.probe_server("http://127.0.0.1:12435") is None


class TestPortOpen:
    def test_open_and_closed_port(self) -> None:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            assert realtime.port_open(f"http://127.0.0.1:{port}") is True
        finally:
            listener.close()
        assert realtime.port_open(f"http://127.0.0.1:{port}") is False

    def test_url_without_port(self) -> None:
        assert realtime.port_open("http://127.0.0.1") is False


class TestServesModel:
    def test_matches_by_substring(self) -> None:
        assert realtime._serves_model(_models_payload("/models/" + REAL_MODEL), REAL_MODEL) is True

    def test_absent_model(self) -> None:
        assert realtime._serves_model(_models_payload("other-model"), REAL_MODEL) is False

    def test_empty_data(self) -> None:
        assert realtime._serves_model({"object": "list"}, REAL_MODEL) is False


class TestLogTail:
    def test_missing_file_is_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(realtime, "vllm_log_path", lambda: tmp_path / "nope.log")
        assert realtime._log_tail() == ""

    def test_keeps_last_lines(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        log = tmp_path / "vllm.log"
        log.write_text("\n".join(f"line {i}" for i in range(1, 51)), encoding="utf-8")
        monkeypatch.setattr(realtime, "vllm_log_path", lambda: log)
        tail = realtime._log_tail()
        assert tail.splitlines()[0] == "line 11"
        assert tail.splitlines()[-1] == "line 50"


class TestStartServer:
    def _capture(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                 vllm_bin: str, which: str | None) -> dict:
        """Подменить Popen/which/лог и вернуть запись вызова."""
        seen: dict = {}

        def _fake_popen(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs
            seen["proc"] = _FakeProcess()
            return seen["proc"]

        monkeypatch.setattr(realtime.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(realtime.shutil, "which", lambda name: which)
        monkeypatch.setattr(realtime, "vllm_log_path", lambda: tmp_path / "vllm.log")
        seen["vllm_bin"] = vllm_bin
        seen["tmp_path"] = tmp_path
        return seen

    def test_binary_from_path_builds_command(self, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        seen = self._capture(monkeypatch, tmp_path, "vllm", "/usr/bin/vllm")
        proc = realtime.start_server("http://127.0.0.1:12435", REAL_MODEL, "vllm", "1", 300.0)
        assert proc is seen["proc"]
        assert seen["cmd"] == ["/usr/bin/vllm", "serve", REAL_MODEL, "--host", "127.0.0.1",
                               "--port", "12435", "--enforce-eager",
                               "--max-model-len", "5798", "--gpu-memory-utilization", "0.85"]
        assert seen["kwargs"]["start_new_session"] is True
        assert seen["kwargs"]["stderr"] is subprocess.STDOUT
        assert seen["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        assert (tmp_path / "vllm.log").exists()

    def test_binary_as_file_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        binary = tmp_path / "vllm"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        seen = self._capture(monkeypatch, tmp_path, str(binary), None)
        realtime.start_server("http://127.0.0.1:12435", REAL_MODEL, str(binary), "", 60.0)
        assert seen["cmd"][0] == str(binary)
        assert seen["cmd"][seen["cmd"].index("--max-model-len") + 1] == "4096"
        # gpu пуст -> CUDA_VISIBLE_DEVICES не подменяется.
        assert "CUDA_VISIBLE_DEVICES" not in seen["kwargs"]["env"]

    def test_binary_from_path_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._capture(monkeypatch, tmp_path, "", "/usr/bin/vllm")
        realtime.start_server("http://127.0.0.1:12435", REAL_MODEL, "", "", 300.0)
        assert seen["cmd"][0] == "/usr/bin/vllm"

    def test_missing_binary_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._capture(monkeypatch, tmp_path, "/nope/vllm", None)
        with pytest.raises(RealtimeError, match="vllm\\[audio\\]"):
            realtime.start_server("http://127.0.0.1:12435", REAL_MODEL, "/nope/vllm", "1", 300.0)

    def test_url_without_port_raises(self, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        self._capture(monkeypatch, tmp_path, "vllm", "/usr/bin/vllm")
        with pytest.raises(RealtimeError, match="не указан порт"):
            realtime.start_server("http://127.0.0.1", REAL_MODEL, "vllm", "1", 300.0)


class TestWaitReady:
    def test_ready_immediately(self, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: _models_payload(REAL_MODEL))
        realtime.wait_ready(_FakeProcess(), "http://127.0.0.1:12435", REAL_MODEL, timeout=0.0)
        assert "сервер vLLM готов" in capsys.readouterr().out

    def test_waits_between_probes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payloads = [None, _models_payload(REAL_MODEL)]
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: payloads.pop(0))
        slept: list[float] = []
        monkeypatch.setattr(realtime.time, "sleep", lambda seconds: slept.append(seconds))
        realtime.wait_ready(_FakeProcess(), "http://127.0.0.1:12435", REAL_MODEL, timeout=30.0)
        assert slept == [2.0]

    def test_process_death_reports_log_tail(self, monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: None)
        log = tmp_path / "vllm.log"
        log.write_text("CUDA out of memory", encoding="utf-8")
        monkeypatch.setattr(realtime, "vllm_log_path", lambda: log)
        with pytest.raises(RealtimeError) as err:
            realtime.wait_ready(_FakeProcess(returncode=1), "http://127.0.0.1:12435", REAL_MODEL,
                                timeout=30.0)
        message = str(err.value)
        assert "завершился с кодом 1" in message
        assert "CUDA out of memory" in message

    def test_timeout_reports_missing_log(self, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path) -> None:
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: {"data": []})
        monkeypatch.setattr(realtime, "vllm_log_path", lambda: tmp_path / "absent.log")
        with pytest.raises(RealtimeError, match="не дождались готовности"):
            realtime.wait_ready(_FakeProcess(), "http://127.0.0.1:12435", REAL_MODEL, timeout=0.0)


class TestStopServer:
    def _signals(self, monkeypatch: pytest.MonkeyPatch, errors: list[Exception | None]) -> list:
        calls: list[tuple[int, int]] = []

        def _fake_killpg(pgid, signum):
            calls.append((pgid, signum))
            error = errors.pop(0) if errors else None
            if error is not None:
                raise error

        monkeypatch.setattr(realtime.os, "killpg", _fake_killpg)
        monkeypatch.setattr(realtime.os, "getpgid", lambda pid: pid)
        return calls

    def test_terminates_and_waits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._signals(monkeypatch, [None])
        proc = _FakeProcess()
        realtime.stop_server(proc)
        assert calls == [(4242, realtime.signal.SIGTERM)]
        assert proc.waits == [15]

    def test_gone_process_group_returns_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._signals(monkeypatch, [ProcessLookupError()])
        proc = _FakeProcess()
        realtime.stop_server(proc)
        assert calls == [(4242, realtime.signal.SIGTERM)]
        assert proc.waits == []

    def test_kills_after_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._signals(monkeypatch, [None, None])
        proc = _FakeProcess(wait_error=subprocess.TimeoutExpired("vllm", 15))
        realtime.stop_server(proc)
        assert calls == [(4242, realtime.signal.SIGTERM), (4242, realtime.signal.SIGKILL)]

    def test_swallows_permission_error_on_kill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._signals(monkeypatch, [None, PermissionError()])
        proc = _FakeProcess(wait_error=subprocess.TimeoutExpired("vllm", 15))
        realtime.stop_server(proc)  # Не должно поднять исключение.
        assert [signum for _, signum in calls] == [realtime.signal.SIGTERM, realtime.signal.SIGKILL]


class TestEnsureServer:
    def test_already_running(self, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: _models_payload(REAL_MODEL))

        def _boom(*args, **kwargs):  # pragma: no cover - не должен вызываться.
            raise AssertionError("сервер не должен запускаться повторно")

        monkeypatch.setattr(realtime, "start_server", _boom)
        holder: list = []
        realtime.ensure_server(_provider(), REAL_MODEL, 300.0, proc_holder=holder)
        assert holder == []
        assert "уже запущен" in capsys.readouterr().out

    def test_foreign_service_on_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: _models_payload("other"))
        with pytest.raises(RealtimeError, match="другой сервис"):
            realtime.ensure_server(_provider(), REAL_MODEL, 300.0, proc_holder=[])

    def test_port_occupied_by_other_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: None)
        monkeypatch.setattr(realtime, "port_open", lambda base_url: True)
        with pytest.raises(RealtimeError, match="занят не-vLLM сервисом"):
            realtime.ensure_server(_provider(), REAL_MODEL, 300.0, proc_holder=[])

    def test_autostart_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: None)
        monkeypatch.setattr(realtime, "port_open", lambda base_url: False)
        with pytest.raises(RealtimeError, match="недоступен"):
            realtime.ensure_server(_provider(autostart=False), REAL_MODEL, 300.0, proc_holder=[])

    def test_starts_and_hands_process_to_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _FakeProcess()
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: None)
        monkeypatch.setattr(realtime, "port_open", lambda base_url: False)
        recorded: dict = {}

        def _fake_start(base_url, model, vllm_bin, gpu, chunk_seconds):
            recorded["args"] = (base_url, model, vllm_bin, gpu, chunk_seconds)
            return proc

        monkeypatch.setattr(realtime, "start_server", _fake_start)
        monkeypatch.setattr(realtime, "wait_ready", lambda p, base_url, model: None)
        holder: list = []
        realtime.ensure_server(_provider(vllm_bin="/usr/bin/vllm", gpu="1"), REAL_MODEL, 120.0,
                               proc_holder=holder)
        assert holder == [proc]
        assert recorded["args"] == ("http://127.0.0.1:12435", REAL_MODEL, "/usr/bin/vllm", "1",
                                    120.0)

    def test_failed_start_stops_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _FakeProcess()
        stopped: list = []
        monkeypatch.setattr(realtime, "probe_server", lambda base_url: None)
        monkeypatch.setattr(realtime, "port_open", lambda base_url: False)
        monkeypatch.setattr(realtime, "start_server", lambda *args: proc)
        monkeypatch.setattr(realtime, "stop_server", lambda p: stopped.append(p))

        def _fail(p, base_url, model):
            raise RealtimeError("сервер упал")

        monkeypatch.setattr(realtime, "wait_ready", _fail)
        holder: list = []
        with pytest.raises(RealtimeError, match="сервер упал"):
            realtime.ensure_server(_provider(), REAL_MODEL, 300.0, proc_holder=holder)
        assert stopped == [proc]
        assert holder == []


class TestExtractPcm:
    def test_cache_hit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pcm = tmp_path / "audio.s16le"
        pcm.write_bytes(b"\0" * 64000)

        def _boom(cmd, **kwargs):  # pragma: no cover - вызов не должен случиться.
            raise AssertionError("ffmpeg не нужен при готовом файле")

        monkeypatch.setattr(realtime.subprocess, "run", _boom)
        assert realtime.extract_pcm(tmp_path / "in.mp4", pcm) == 2.0

    def test_runs_ffmpeg_and_returns_duration(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"\0" * 32000)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(realtime.subprocess, "run", _fake_run)
        pcm = tmp_path / "nested" / "audio.s16le"
        assert realtime.extract_pcm(tmp_path / "in.mp4", pcm) == 1.0
        assert seen["cmd"][:2] == ["ffmpeg", "-nostdin"]
        assert "-vn" in seen["cmd"] and "s16le" in seen["cmd"]
        assert seen["cmd"][seen["cmd"].index("-ar") + 1] == "16000"

    def test_force_reruns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pcm = tmp_path / "audio.s16le"
        pcm.write_bytes(b"\0" * 32000)
        calls: list = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(realtime.subprocess, "run", _fake_run)
        assert realtime.extract_pcm(tmp_path / "in.mp4", pcm, force=True) == 1.0
        assert len(calls) == 1

    def test_ffmpeg_failure_raises(self, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            realtime.subprocess, "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="",
                                                              stderr="no audio stream"),
        )
        with pytest.raises(RealtimeError, match="no audio stream"):
            realtime.extract_pcm(tmp_path / "in.mp4", tmp_path / "audio.s16le")


class TestFindSilences:
    def _stderr(self, *starts: float) -> str:
        return "\n".join(f"[silencedetect @ 0x1] silence_start: {value}" for value in starts)

    def test_parses_stderr_and_caches(self, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_run(cmd, **kwargs):
            assert "silencedetect=noise=-35dB:d=0.5" in cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=self._stderr(1.5, 864.25))

        monkeypatch.setattr(realtime.subprocess, "run", _fake_run)
        cache = tmp_path / "silences.json"
        assert realtime.find_silences(tmp_path / "audio.s16le", cache) == [1.5, 864.25]
        assert json.loads(cache.read_text(encoding="utf-8")) == [1.5, 864.25]

    def test_cache_hit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = tmp_path / "silences.json"
        cache.write_text("[3.5, 10]", encoding="utf-8")

        def _boom(cmd, **kwargs):  # pragma: no cover - вызов не должен случиться.
            raise AssertionError("ffmpeg не нужен при готовом кэше")

        monkeypatch.setattr(realtime.subprocess, "run", _boom)
        assert realtime.find_silences(tmp_path / "audio.s16le", cache) == [3.5, 10.0]

    def test_force_reruns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = tmp_path / "silences.json"
        cache.write_text("[1.0]", encoding="utf-8")
        calls: list = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=self._stderr(7.5))

        monkeypatch.setattr(realtime.subprocess, "run", _fake_run)
        assert realtime.find_silences(tmp_path / "audio.s16le", cache, force=True) == [7.5]
        assert len(calls) == 1

    def test_ffmpeg_failure_warns_and_returns_empty(self, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(
            realtime.subprocess, "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="broken"),
        )
        cache = tmp_path / "silences.json"
        assert realtime.find_silences(tmp_path / "audio.s16le", cache) == []
        assert "не удалось определить паузы" in capsys.readouterr().out
        assert json.loads(cache.read_text(encoding="utf-8")) == []


class TestChunkBoundaries:
    def test_no_silences_cuts_by_time(self) -> None:
        assert realtime.chunk_boundaries(600.0, [], 300.0) == [300.0]

    def test_silence_near_target_wins(self) -> None:
        assert realtime.chunk_boundaries(600.0, [290.0], 300.0) == [290.0, 590.0]

    def test_silence_far_from_target_ignored(self) -> None:
        assert realtime.chunk_boundaries(600.0, [10.0], 300.0) == [300.0]

    def test_short_file_has_no_cuts(self) -> None:
        assert realtime.chunk_boundaries(120.0, [60.0], 300.0) == []

    def test_cut_clamped_to_previous_plus_one(self) -> None:
        # Пауза 0.5 ближе к цели 0.5, но рез не может отстоять от нуля
        # меньше чем на секунду — он сдвигается к 1.0.
        assert realtime.chunk_boundaries(1.2, [0.5, 0.6], 0.5) == [1.0]

    def test_cut_at_duration_stops(self) -> None:
        # Сдвинутый рез выходит за конец файла: резов нет вовсе.
        assert realtime.chunk_boundaries(1.0, [0.5, 0.6], 0.5) == []


class TestSegmentsFromChunks:
    def test_boundaries_of_first_and_last_chunk(self) -> None:
        segments = realtime.segments_from_chunks(600.0, [290.0, 590.0], ["а", "б", "в"])
        assert segments == [
            {"start": 0.0, "end": 290.0, "text": "а", "speaker": None},
            {"start": 290.0, "end": 590.0, "text": "б", "speaker": None},
            {"start": 590.0, "end": 600.0, "text": "в", "speaker": None},
        ]

    def test_single_chunk_covers_file(self) -> None:
        assert realtime.segments_from_chunks(42.0, [], [" весь текст "]) == [
            {"start": 0.0, "end": 42.0, "text": "весь текст", "speaker": None},
        ]

    def test_empty_texts_skipped(self) -> None:
        # Пустой фрагмент (ошибка распознавания) не даёт сегмента.
        assert realtime.segments_from_chunks(600.0, [300.0], ["а", ""]) == [
            {"start": 0.0, "end": 300.0, "text": "а", "speaker": None},
        ]


class TestRealtimeWsUrl:
    def test_http_and_https(self) -> None:
        assert realtime.realtime_ws_url("http://127.0.0.1:12435") == \
            "ws://127.0.0.1:12435/v1/realtime"
        assert realtime.realtime_ws_url("https://vllm.test/v1/") == "wss://vllm.test/v1/v1/realtime"

    def test_unknown_scheme(self) -> None:
        with pytest.raises(RealtimeError, match="неизвестная схема"):
            realtime.realtime_ws_url("ftp://vllm.test")


class FakeVllmServer:
    """Мини-сервер /v1/realtime: повторяет протокол vLLM в отдельном потоке."""

    def __init__(self, *, deltas: tuple[str, ...] = ("гото", "вый текст"),
                 send_error: str | None = None, close_without_done: bool = False,
                 silent: bool = False, noise_events: tuple[dict, ...] = ()) -> None:
        self.deltas = deltas
        self.send_error = send_error
        self.close_without_done = close_without_done
        self.silent = silent
        self.noise_events = noise_events
        self.port: int | None = None
        self.model: str | None = None
        self.events: list[dict] = []
        self.audio_bytes = 0
        self.final_commit = False
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    async def _handler(self, ws) -> None:
        await ws.send(json.dumps({"type": "session.created", "id": "sess-1"}))
        async for raw in ws:
            event = json.loads(raw)
            self.events.append(event)
            if event["type"] == "session.update":
                self.model = event.get("model")
            elif event["type"] == "input_audio_buffer.append":
                self.audio_bytes += len(base64.b64decode(event["audio"]))
            elif event["type"] == "input_audio_buffer.commit" and event.get("final"):
                self.final_commit = True
                if self.send_error is not None:
                    await ws.send(json.dumps({"type": "error", "error": self.send_error,
                                              "code": "processing_error"}))
                    return
                if self.silent:
                    while not self._stop.is_set():
                        await asyncio.sleep(0.05)
                    return
                for event in self.noise_events:
                    await ws.send(json.dumps(event))
                for delta in self.deltas:
                    await ws.send(json.dumps({"type": "transcription.delta", "delta": delta}))
                if not self.close_without_done:
                    await ws.send(json.dumps({"type": "transcription.done",
                                              "text": "".join(self.deltas), "usage": {}}))
                return

    async def _main(self) -> None:
        async with serve(self._handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.05)

    def __enter__(self) -> "FakeVllmServer":
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=10), "сервер не поднялся"
        return self

    def __exit__(self, *exc_info) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return False

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"


class TestTranscribeChunk:
    MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"

    def test_full_exchange_splits_audio_into_appends(self) -> None:
        pcm = b"\x01\x02" * (realtime.WS_APPEND_BYTES + 100)
        with FakeVllmServer() as server:
            text = realtime.transcribe_chunk(pcm, server.url, self.MODEL, timeout=30.0)
        assert text == "готовый текст"
        assert server.model == self.MODEL
        assert server.audio_bytes == len(pcm)
        appends = [event for event in server.events
                   if event["type"] == "input_audio_buffer.append"]
        assert len(appends) == 3  # два полных блока по 65536 байт и остаток.
        assert server.events[0]["type"] == "session.update"
        assert server.events[1]["type"] == "input_audio_buffer.commit"
        assert server.final_commit is True

    def test_error_event_raises(self) -> None:
        with FakeVllmServer(send_error="Maximum file size exceeded") as server:
            with pytest.raises(RealtimeError, match="Maximum file size exceeded"):
                realtime.transcribe_chunk(b"\x00\x01" * 100, server.url, self.MODEL, timeout=30.0)

    def test_closed_without_done_returns_deltas(self) -> None:
        with FakeVllmServer(close_without_done=True) as server:
            text = realtime.transcribe_chunk(b"\x00\x01" * 100, server.url, self.MODEL,
                                             timeout=30.0)
        assert text == "готовый текст"

    def test_empty_deltas_give_empty_text(self) -> None:
        with FakeVllmServer(deltas=()) as server:
            text = realtime.transcribe_chunk(b"\x00\x01" * 100, server.url, self.MODEL,
                                             timeout=30.0)
        assert text == ""

    def test_unknown_events_are_ignored(self) -> None:
        """Незнакомые события сервера (задел протокола) не ломают разбор."""
        noise = ({"type": "session.updated", "session": {}}, {"type": "conversation.created"})
        with FakeVllmServer(noise_events=noise) as server:
            text = realtime.transcribe_chunk(b"\x00\x01" * 100, server.url, self.MODEL,
                                             timeout=30.0)
        assert text == "готовый текст"

    def test_timeout_raises_realtime_error(self) -> None:
        with FakeVllmServer(silent=True) as server:
            with pytest.raises(RealtimeError, match="таймаут"):
                realtime.transcribe_chunk(b"\x00\x01" * 100, server.url, self.MODEL, timeout=0.3)

    def test_connection_refused_raises_realtime_error(self) -> None:
        """Закрытый порт: соединение не встаёт, ошибка — RealtimeError (не OSError)."""
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with pytest.raises(RealtimeError, match="соединение с vLLM"):
            realtime.transcribe_chunk(b"\x00\x01" * 100, f"ws://127.0.0.1:{port}/v1/realtime",
                                      self.MODEL, timeout=5.0)
