"""Тесты CLI: разбор аргументов, коды выхода, обработка директорий, ошибки."""

from __future__ import annotations

from pathlib import Path

import pytest

from absorbent.cli import MEDIA_EXTENSIONS, build_parser, collect_inputs, main
from absorbent.config import ConfigError


class TestCollectInputs:
    def test_single_file(self, tmp_path: Path) -> None:
        video = tmp_path / "a.mp4"
        video.write_bytes(b"x")
        assert collect_inputs(str(video)) == [video]

    def test_directory_filters_and_sorts(self, tmp_path: Path) -> None:
        (tmp_path / "b.mkv").write_bytes(b"x")
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "notes.txt").write_bytes(b"x")  # не медиа.
        result = collect_inputs(str(tmp_path))
        assert [p.name for p in result] == ["a.mp4", "b.mkv"]

    def test_directory_supports_audio_extensions(self, tmp_path: Path) -> None:
        for ext in (".mp3", ".wav", ".opus"):
            (tmp_path / f"x{ext}").write_bytes(b"x")
        result = collect_inputs(str(tmp_path))
        assert len(result) == 3

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="не существует"):
            collect_inputs(str(tmp_path / "nope.mp4"))

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="не найдено медиафайлов"):
            collect_inputs(str(tmp_path))

    def test_extensions_set_matches_docs(self) -> None:
        assert ".mp4" in MEDIA_EXTENSIONS and ".mkv" in MEDIA_EXTENSIONS
        assert ".txt" not in MEDIA_EXTENSIONS


class TestParser:
    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["in.mp4"])
        assert args.language == "auto"
        assert args.output_language == "ru"
        assert args.fps == 1.0
        assert args.hash_threshold == 10
        assert args.output_dir is None
        assert args.force is False
        assert args.skip_stt is False and args.skip_synthesize is False
        assert args.skip_render is False
        assert args.stt is None and args.vision_model is None and args.llm_model is None
        assert args.config is None
        assert args.flat is False

    def test_explicit_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "in.mkv", "--config", "c.yaml", "--stt", "x", "--vision-model", "v",
            "--llm-model", "l", "--language", "ru", "--output-language", "en",
            "--fps", "2", "--hash-threshold", "5",
            "--output-dir", "out", "--force", "--skip-stt", "--skip-frames",
            "--skip-describe", "--skip-synthesize", "--skip-render", "--flat",
        ])
        assert args.language == "ru" and args.output_language == "en"
        assert args.fps == 2.0 and args.hash_threshold == 5
        assert args.skip_render is True
        assert args.stt == "x" and args.output_dir == "out" and args.force is True
        assert args.skip_describe is True and args.skip_synthesize is True
        assert args.flat is True

    def test_bad_language_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["in.mp4", "--language", "fr"])

    def test_bad_output_language_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["in.mp4", "--output-language", "fr"])

    def test_missing_input_exits_2(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2


class TestMain:
    def test_processes_single_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        video = tmp_path / "lec.mp4"
        video.write_bytes(b"x")
        config = tmp_path / "config.yaml"
        config.write_text(
            "providers:\n  p: {base_url: u, api_key_env: ''}\n"
            "stt:\n  s: {backend: whisper, whisper_model: tiny, default: true}\n"
            "vision:\n  v: {provider: p, default: true}\n"
            "llm:\n  l: {provider: p, default: true}\n",
            encoding="utf-8",
        )
        processed: list[Path] = []

        def _fake_process(input_path, cfg, args, server_holder=None):
            processed.append(input_path)
            return input_path.with_name("notes.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        code = main([str(video), "--config", str(config)])
        assert code == 0
        assert processed == [video]

    def test_flat_flag_reaches_process_file(self, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        """--flat доезжает до process_file в аргументах."""
        video = tmp_path / "lec.mp4"
        video.write_bytes(b"x")
        config = tmp_path / "config.yaml"
        config.write_text(
            "providers:\n  p: {base_url: u, api_key_env: ''}\n"
            "stt:\n  s: {backend: whisper, whisper_model: tiny, default: true}\n"
            "vision:\n  v: {provider: p, default: true}\n"
            "llm:\n  l: {provider: p, default: true}\n",
            encoding="utf-8",
        )
        seen: dict = {}

        def _fake_process(input_path, cfg, args, server_holder=None):
            seen["flat"] = args.flat
            return input_path.with_name("lec.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        assert main([str(video), "--config", str(config), "--flat"]) == 0
        assert seen["flat"] is True

    def test_processes_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "b.mkv").write_bytes(b"x")
        processed: list[str] = []

        def _fake_process(input_path, cfg, args, server_holder=None):
            processed.append(input_path.name)
            return input_path.with_name("notes.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        code = main([str(tmp_path), "--skip-stt"])
        assert code == 0
        assert processed == ["a.mp4", "b.mkv"]

    def test_stage_error_continues_and_exits_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "b.mp4").write_bytes(b"x")
        processed: list[str] = []

        def _fake_process(input_path, cfg, args, server_holder=None):
            processed.append(input_path.name)
            if input_path.name == "a.mp4":
                raise RuntimeError("boom")
            return input_path.with_name("notes.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        code = main([str(tmp_path)])
        assert code == 1
        assert processed == ["a.mp4", "b.mp4"]  # второй файл обработан несмотря на ошибку.

    def test_unknown_model_lists_valid(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            "providers:\n  p: {base_url: u, api_key_env: ''}\n"
            "stt:\n  aa: {backend: whisper, whisper_model: tiny, default: true}\n"
            "vision:\n  bb: {provider: p, default: true}\n"
            "llm:\n  cc: {provider: p, default: true}\n",
            encoding="utf-8",
        )
        code = main([str(tmp_path / "x.mp4"), "--config", str(config), "--stt", "nope"])
        assert code == 2
        err = capsys.readouterr().err
        assert "неизвестная модель 'nope'" in err
        assert "aa" in err

    def test_missing_config_file(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["/tmp/whatever.mp4", "--config", "/nonexistent/config.yaml"])
        assert code == 2
        assert "не найден" in capsys.readouterr().err

    def test_empty_input_directory_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        code = main([str(tmp_path)])
        assert code == 2
        assert "медиафайлов" in capsys.readouterr().err


class TestBatchServerLifetime:
    """Сервер vLLM на весь запуск: один старт на пакет файлов, один останов."""

    def _realtime_config(self, tmp_path: Path) -> Path:
        config = tmp_path / "config.yaml"
        config.write_text(
            "providers:\n"
            "  prov-a: {base_url: https://api.example.test/v1, api_key_env: TEST_KEY_A}\n"
            "  vllm-voxtral: {base_url: http://127.0.0.1:12435, api_key_env: '', gpu: '1'}\n"
            "stt:\n"
            "  rt: {backend: voxtral-realtime, provider: vllm-voxtral, model: realtime-model,\n"
            "       chunk_seconds: 120, default: true}\n"
            "vision:\n  vis-a: {provider: prov-a, default: true}\n"
            "llm:\n  llm-a: {provider: prov-a, default: true}\n",
            encoding="utf-8",
        )
        return config

    def test_server_started_once_for_batch(self, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.pipeline as pipeline_module
        import absorbent.realtime as realtime_module

        media = tmp_path / "media"
        media.mkdir()
        for name in ("a.mp3", "b.mp3"):
            (media / name).write_bytes(b"audio")
        proc = object()
        events: list[str] = []

        def _ensure(provider, model, chunk_seconds, *, proc_holder):
            events.append("ensure")
            proc_holder.append(proc)

        monkeypatch.setattr(realtime_module, "ensure_server", _ensure)
        monkeypatch.setattr(realtime_module, "stop_server",
                            lambda stopped: events.append(f"stop:{stopped is proc}"))
        monkeypatch.setattr(pipeline_module, "run_stt",
                            lambda *args, **kwargs: events.append("stt") or None)

        code = main([str(media), "--config", str(self._realtime_config(tmp_path)),
                     "--flat", "--skip-frames", "--skip-describe", "--skip-synthesize",
                     "--skip-render"])
        assert code == 0
        # Один старт сервера на два файла и одна остановка в конце.
        assert events == ["ensure", "stt", "stt", "stop:True"]

    def test_no_server_without_realtime_backend(self, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.realtime as realtime_module

        (tmp_path / "a.mp4").write_bytes(b"x")

        def _boom(*args, **kwargs):  # pragma: no cover - не должен вызываться.
            raise AssertionError("для whisper сервер vLLM не нужен")

        monkeypatch.setattr(realtime_module, "ensure_server", _boom)
        monkeypatch.setattr(realtime_module, "stop_server", _boom)
        assert main([str(tmp_path / "a.mp4"), "--skip-stt", "--skip-frames",
                     "--skip-describe", "--skip-synthesize", "--skip-render"]) == 0
