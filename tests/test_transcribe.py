"""Тесты распознавания речи: кэш, whisper-бэкенд, voxtral-бэкенд, ошибки."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from absorbent.config import ConfigError, ModelSpec, Provider, load_config
from absorbent.transcribe import STTError, run_stt

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _whisper_spec() -> ModelSpec:
    return ModelSpec(name="whisper-turbo", backend="whisper", whisper_model="turbo", default=True)


def _voxtral_spec() -> ModelSpec:
    return ModelSpec(name="voxtral-transcribe-2", backend="voxtral", provider="mistral",
                     model="voxtral-mini-latest")


def _write_cached(out_json: Path, language: str | None = None) -> None:
    """Заранее записать transcript.json (+ language.txt)."""
    out_json.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "привет", "speaker": None}]),
                        encoding="utf-8")
    if language is not None:
        out_json.with_name("language.txt").write_text(language, encoding="utf-8")


class TestCache:
    def test_returns_cached_language(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out, language="ru")

        def _boom(*a, **k):  # pragma: no cover - вызов не должен случиться.
            raise AssertionError("транскрипция не должна запускаться при кэше")

        monkeypatch.setattr("absorbent.transcribe._whisper_transcribe", _boom)
        monkeypatch.setattr("absorbent.transcribe._voxtral_transcribe", _boom)
        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "input.mp4",
                           "auto", out, tmp_path)
        assert detected == "ru"

    def test_cached_without_language_returns_none(self, tmp_path: Path) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out)
        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "input.mp4",
                           "auto", out, tmp_path)
        assert detected is None

    def test_force_reruns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out, language="ru")
        calls = {"n": 0}

        def _fake_whisper(spec, input_path, language):
            calls["n"] += 1
            return [], None

        monkeypatch.setattr("absorbent.transcribe._whisper_transcribe", _fake_whisper)
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "input.mp4", "auto", out,
                tmp_path, force=True)
        assert calls["n"] == 1


class TestWhisperBackend:
    def test_transcribes_to_json(self, tmp_path: Path, fake_whisper: dict) -> None:
        fake_whisper["segments"] = [
            {"start": 0.0, "end": 2.5, "text": "  первый сегмент  "},
            {"start": 2.5, "end": 5.0, "text": "второй"},
        ]
        fake_whisper["language"] = "ru"
        input_path = tmp_path / "lecture.mp4"
        input_path.write_bytes(b"media")
        out = tmp_path / "transcript.json"

        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), input_path, "auto", out, tmp_path)
        assert detected == "ru"
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [
            {"start": 0.0, "end": 2.5, "text": "первый сегмент", "speaker": None},
            {"start": 2.5, "end": 5.0, "text": "второй", "speaker": None},
        ]
        assert out.with_name("language.txt").read_text(encoding="utf-8") == "ru"
        # Параметры вызова whisper: auto -> language None, beam_size=5.
        call = fake_whisper["transcribe_calls"][0]
        assert call["language"] is None
        assert call["beam_size"] == 5
        assert call["verbose"] is False
        assert fake_whisper["load_calls"][0][0] == "turbo"

    def test_device_cpu_when_no_cuda(self, tmp_path: Path, fake_whisper: dict) -> None:
        fake_whisper["cuda"] = False
        fake_whisper["language"] = None
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "ru",
                tmp_path / "t.json", tmp_path)
        assert fake_whisper["load_calls"][0][1] == "cpu"

    def test_explicit_language_passed(self, tmp_path: Path, fake_whisper: dict) -> None:
        fake_whisper["language"] = "ru"
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "en",
                tmp_path / "t.json", tmp_path)
        assert fake_whisper["transcribe_calls"][0]["language"] == "en"

    def test_empty_segments_warns_and_writes_empty(self, tmp_path: Path, fake_whisper: dict,
                                                   capsys: pytest.CaptureFixture) -> None:
        fake_whisper["segments"] = []
        fake_whisper["language"] = None
        out = tmp_path / "transcript.json"
        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4",
                           "auto", out, tmp_path)
        assert detected is None
        assert json.loads(out.read_text(encoding="utf-8")) == []
        assert "пустой транскрипт" in capsys.readouterr().out

    def test_language_file_removed_when_no_detection(self, tmp_path: Path, fake_whisper: dict) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out, language="ru")
        fake_whisper["language"] = None
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "auto", out,
                tmp_path, force=True)
        assert not out.with_name("language.txt").exists()


class TestVoxtralBackend:
    def _prepare(self, tmp_path: Path, fake_openai, monkeypatch: pytest.MonkeyPatch,
                 payload: dict) -> tuple[Path, Provider]:
        """Заглушить ffmpeg-хелперы и вернуть (out_json, provider mistral)."""
        import absorbent.transcribe as transcribe_module

        monkeypatch.setattr(transcribe_module, "_extract_audio",
                            lambda input_path, audio_path: audio_path.write_bytes(b"wav-data"))
        fake_openai.audio.transcriptions.payload = payload
        cfg = load_config(str(REPO_CONFIG))
        return tmp_path / "transcript.json", cfg.providers["mistral"]

    def test_voxtral_segments_mapped(self, tmp_path: Path, fake_openai,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "key123")
        payload = {
            "text": "весь текст",
            "language": "ru",
            "segments": [
                {"text": " раз ", "start": 0.0, "end": 1.2, "speaker_id": "speaker_1",
                 "type": "transcription_segment"},
                {"text": "два", "start": 1.2, "end": 2.0, "speaker_id": None,
                 "type": "transcription_segment"},
                {"text": "", "start": 2.0, "end": 3.0, "speaker_id": None},  # пустой сегмент.
            ],
            "usage": {},
        }
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch, payload)
        detected = run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "auto", out, tmp_path)
        assert detected == "ru"
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [
            {"start": 0.0, "end": 1.2, "text": "раз", "speaker": "speaker_1"},
            {"start": 1.2, "end": 2.0, "text": "два", "speaker": None},
        ]
        assert out.with_name("language.txt").read_text(encoding="utf-8") == "ru"
        # Параметры запроса: verbose_json, сегменты + диаризация, x-api-key.
        call = fake_openai.audio.transcriptions.calls[0]
        assert call["model"] == "voxtral-mini-latest"
        assert call["response_format"] == "verbose_json"
        assert call["extra_body"]["diarize"] is True
        assert call["extra_body"]["timestamp_granularities"] == "segment"
        assert call["extra_body"]["language"] is None
        assert call["extra_headers"] == {"X-Api-Key": "key123"}

    def test_voxtral_explicit_language(self, tmp_path: Path, fake_openai,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "key123")
        payload = {"text": "t", "language": "en", "segments": [], "usage": {}}
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch, payload)
        run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "ru", out, tmp_path)
        assert fake_openai.audio.transcriptions.calls[0]["extra_body"]["language"] == "ru"

    def test_voxtral_fallback_single_segment(self, tmp_path: Path, fake_openai,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "key123")
        import absorbent.transcribe as transcribe_module

        payload = {"text": "только текст без сегментов", "language": None, "segments": [], "usage": {}}
        monkeypatch.setattr(transcribe_module, "_audio_duration_seconds", lambda path: 42.5)
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch, payload)
        detected = run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "auto", out, tmp_path)
        assert detected is None
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [{"start": 0.0, "end": 42.5, "text": "только текст без сегментов",
                         "speaker": None}]

    def test_voxtral_missing_key_raises(self, tmp_path: Path, fake_openai,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch,
                                      {"text": "", "language": None, "segments": []})
        with pytest.raises(ConfigError, match="MISTRAL_API_KEY"):
            run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "auto", out, tmp_path)

    def test_unknown_backend(self, tmp_path: Path) -> None:
        class Bogus:
            backend = "bogus"
            name = "x"
            api_model = "x"

        with pytest.raises(STTError, match="bogus"):
            run_stt(Bogus(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "auto",
                    tmp_path / "t.json", tmp_path)


class TestAudioHelpers:
    def test_extract_audio_ffmpeg_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        def _fail(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no audio track")

        monkeypatch.setattr(subprocess, "run", _fail)
        with pytest.raises(STTError, match="no audio track"):
            transcribe_module._extract_audio(tmp_path / "in.mp4", tmp_path / "audio.wav")

    def test_audio_duration_seconds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="12.5\n", stderr=""),
        )
        assert transcribe_module._audio_duration_seconds(tmp_path / "audio.wav") == 12.5

    def test_audio_duration_seconds_failure_returns_zero(self, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err"),
        )
        assert transcribe_module._audio_duration_seconds(tmp_path / "audio.wav") == 0.0


    def test_extract_audio_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        def _ok(cmd, **kwargs):
            (tmp_path / "audio.wav").write_bytes(b"wav")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _ok)
        transcribe_module._extract_audio(tmp_path / "in.mp4", tmp_path / "audio.wav")
        assert (tmp_path / "audio.wav").read_bytes() == b"wav"


REALTIME_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def _realtime_spec(chunk_seconds: float = 300.0) -> ModelSpec:
    return ModelSpec(name="voxtral-realtime", backend="voxtral-realtime", provider="vllm-voxtral",
                     model=REALTIME_MODEL, chunk_seconds=chunk_seconds)


@pytest.fixture
def fake_realtime(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Подменяет этапы absorbent.realtime: нарезка и WS-обмен под контролем теста.

    state["script"] — очередь ответов transcribe_chunk: строка или исключение
    (исключение поднимается вместо ответа), по одному на вызов.
    """
    import absorbent.realtime as realtime_module

    state: dict = {"script": [], "chunks": [], "calls": [], "duration": 600.0, "silences": []}

    def _extract_pcm(input_path, pcm_path, force=False):
        state["calls"].append(("extract_pcm", input_path, pcm_path, force))
        pcm_path.parent.mkdir(parents=True, exist_ok=True)
        size = int(state["duration"] * realtime_module.PCM_BYTES_PER_SECOND)
        pcm_path.write_bytes(b"\0" * size)
        return state["duration"]

    def _find_silences(pcm_path, cache_path, force=False):
        state["calls"].append(("find_silences", pcm_path, cache_path, force))
        return state["silences"]

    def _transcribe_chunk(pcm, ws_url, model, timeout):
        state["chunks"].append({"size": len(pcm), "url": ws_url, "model": model,
                                "timeout": timeout})
        item = state["script"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(realtime_module, "extract_pcm", _extract_pcm)
    monkeypatch.setattr(realtime_module, "find_silences", _find_silences)
    monkeypatch.setattr(realtime_module, "transcribe_chunk", _transcribe_chunk)
    return state


class TestVoxtralRealtimeBackend:
    def _run(self, tmp_path: Path, fake_realtime: dict, language: str = "auto",
             chunk_seconds: float = 300.0, force: bool = False) -> Path:
        out = tmp_path / "lecture.json"
        run_stt(_realtime_spec(chunk_seconds), Provider(name="vllm-voxtral",
                                                        base_url="http://127.0.0.1:12435"),
                tmp_path / "lecture.wav", language, out, tmp_path, force=force)
        return out

    def test_fragments_become_segments(self, tmp_path: Path, fake_realtime: dict,
                                       capsys: pytest.CaptureFixture) -> None:
        """Пауза на 290 с режет часовую запись на три фрагмента."""
        fake_realtime["silences"] = [290.0]
        fake_realtime["script"] = ["первый фрагмент", "второй", ""]
        out = self._run(tmp_path, fake_realtime)
        assert json.loads(out.read_text(encoding="utf-8")) == [
            {"start": 0.0, "end": 290.0, "text": "первый фрагмент", "speaker": None},
            {"start": 290.0, "end": 590.0, "text": "второй", "speaker": None},
        ]
        # Пустой текст (третий фрагмент) сегмента не даёт, язык не детектирован.
        assert not (tmp_path / "language.txt").exists()
        # Размеры блоков равны длительности фрагментов в байтах PCM.
        assert [chunk["size"] for chunk in fake_realtime["chunks"]] == [290 * 32000,
                                                                       300 * 32000,
                                                                       10 * 32000]
        # Адрес и модель уходят в протокол, таймаут растёт от chunk_seconds.
        assert fake_realtime["chunks"][0]["url"] == "ws://127.0.0.1:12435/v1/realtime"
        assert fake_realtime["chunks"][0]["model"] == REALTIME_MODEL
        assert fake_realtime["chunks"][0]["timeout"] == 300.0 * 4 + 300
        printed = capsys.readouterr().out
        assert "длительность 600.0 с, фрагментов 3" in printed
        assert "[1/3] 0.0–290.0 с" in printed
        assert "[3/3] 590.0–600.0 с" in printed

    def test_work_dir_keeps_service_files(self, tmp_path: Path, fake_realtime: dict) -> None:
        fake_realtime["script"] = ["первый", "второй"]
        self._run(tmp_path, fake_realtime)
        calls = dict((name, (a, b, c)) for name, a, b, c in fake_realtime["calls"])
        assert calls["extract_pcm"] == (tmp_path / "lecture.wav",
                                        tmp_path / "audio.s16le", False)
        assert calls["find_silences"] == (tmp_path / "audio.s16le",
                                          tmp_path / "silences.json", False)

    def test_force_is_forwarded(self, tmp_path: Path, fake_realtime: dict) -> None:
        fake_realtime["script"] = ["первый", "второй"]
        self._run(tmp_path, fake_realtime, force=True)
        assert [call[3] for call in fake_realtime["calls"]] == [True, True]

    def test_retry_after_transient_error(self, tmp_path: Path, fake_realtime: dict,
                                        capsys: pytest.CaptureFixture) -> None:
        import absorbent.realtime as realtime_module

        fake_realtime["script"] = [realtime_module.RealtimeError("обрыв"), "текст", "второй"]
        out = self._run(tmp_path, fake_realtime)
        assert json.loads(out.read_text(encoding="utf-8"))[0]["text"] == "текст"
        assert "повтор фрагмента 0.0–300.0" in capsys.readouterr().out

    def test_failed_fragment_is_skipped(self, tmp_path: Path, fake_realtime: dict,
                                        capsys: pytest.CaptureFixture) -> None:
        import absorbent.realtime as realtime_module

        # Два фрагмента (рез на 300 с), второй не распознан обеими попытками.
        fake_realtime["script"] = ["первый", realtime_module.RealtimeError("нет связи"),
                                   realtime_module.RealtimeError("нет связи")]
        out = self._run(tmp_path, fake_realtime, chunk_seconds=300.0, force=True)
        assert json.loads(out.read_text(encoding="utf-8")) == [
            {"start": 0.0, "end": 300.0, "text": "первый", "speaker": None},
        ]
        printed = capsys.readouterr().out
        assert "ошибка фрагмента 300.0–600.0: нет связи" in printed
        assert "внимание: 1 фрагмент(ов) не распознаны" in printed

    def test_first_fragment_failure_propagates(self, tmp_path: Path, fake_realtime: dict) -> None:
        import absorbent.realtime as realtime_module

        fake_realtime["script"] = [realtime_module.RealtimeError("сервер мёртв"),
                                   realtime_module.RealtimeError("сервер мёртв")]
        with pytest.raises(realtime_module.RealtimeError, match="сервер мёртв"):
            self._run(tmp_path, fake_realtime, chunk_seconds=600.0)

    def test_language_hint_warning_once(self, tmp_path: Path, fake_realtime: dict,
                                        capsys: pytest.CaptureFixture) -> None:
        fake_realtime["silences"] = [290.0]
        fake_realtime["script"] = ["а", "б", "в"]
        self._run(tmp_path, fake_realtime, language="ru")
        printed = capsys.readouterr().out
        assert printed.count("не принимает подсказку языка") == 1
        # Детектированного языка нет: протокол его не отдаёт.
        assert not (tmp_path / "language.txt").exists()

    def test_auto_language_prints_nothing(self, tmp_path: Path, fake_realtime: dict,
                                          capsys: pytest.CaptureFixture) -> None:
        fake_realtime["silences"] = [290.0]
        fake_realtime["script"] = ["а", "б", "в"]
        self._run(tmp_path, fake_realtime, language="auto")
        assert "подсказку языка" not in capsys.readouterr().out

    def test_unknown_ws_scheme_raises(self, tmp_path: Path, fake_realtime: dict) -> None:
        fake_realtime["script"] = ["текст"]
        out = tmp_path / "lecture.json"
        with pytest.raises(Exception, match="неизвестная схема"):
            run_stt(_realtime_spec(), Provider(name="vllm-voxtral", base_url="localhost:12435"),
                    tmp_path / "lecture.wav", "auto", out, tmp_path)
