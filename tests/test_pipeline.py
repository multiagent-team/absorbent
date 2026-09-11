"""Тесты оркестрации конвейера: порядок этапов, кэш, skip-флаги, force."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from absorbent.config import load_config
from absorbent.pipeline import _load_json, process_file


@pytest.fixture
def cfg(minimal_config: Path):
    return load_config(str(minimal_config))


@pytest.fixture
def fake_stages(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Подменяет функции этапов записывающими заглушками."""
    import absorbent.pipeline as pipeline_module

    state: dict = {"calls": []}

    def _extract(input_path, frames_dir, fps, force=False):
        state["calls"].append("extract")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (frames_dir / f"frame_{i+1:06d}.jpg").write_bytes(b"jpeg")
        return sorted(frames_dir.glob("frame_*.jpg"))

    def _dedup(frames, images_dir, fps, threshold):
        state["calls"].append(f"dedup:{threshold}")
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "frame_00_00_00.jpg").write_bytes(b"jpeg")
        return [{"image": "frame_00_00_00.jpg", "seconds": 0.0, "timestamp": "00:00:00"}]

    def _run_stt(spec, provider, input_path, language, out_json, work_dir, force=False):
        state["calls"].append(f"stt:{language}")
        out_json.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "привет",
                                         "speaker": None}]), encoding="utf-8")
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "language.txt").write_text("ru", encoding="utf-8")
        return "ru"

    def _describe(vision_spec, provider, keyframes, transcript, images_dir, output_language,
                  out_json, force=False):
        state["calls"].append(f"describe:{output_language}:{len(keyframes)}")
        data = {kf["image"]: {"timestamp": kf["timestamp"], "description": f"кадр {kf['image']}"}
                for kf in keyframes}
        out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                    images_prefix="images/"):
        state["calls"].append(f"synthesize:{output_language}:{len(descriptions)}")
        state["images_prefix"] = images_prefix
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("# Конспект\n", encoding="utf-8")
        return out_md

    def _render(notes_md, out_html=None, out_pdf=None):
        state["calls"].append("render")
        state["render_paths"] = (out_html, out_pdf)
        return notes_md.with_name("notes.html"), notes_md.with_name("notes.pdf")

    def _has_video(input_path):
        # Наличие видеопотока не является этапом конвейера: в state["calls"]
        # его нет, порядок этапов остаётся прежним.
        return state.get("video", True)

    monkeypatch.setattr(pipeline_module, "has_video_stream", _has_video)
    monkeypatch.setattr(pipeline_module, "extract_frames", _extract)
    monkeypatch.setattr(pipeline_module, "dedup_frames", _dedup)
    monkeypatch.setattr(pipeline_module, "run_stt", _run_stt)
    monkeypatch.setattr(pipeline_module, "describe_keyframes", _describe)
    monkeypatch.setattr(pipeline_module, "run_synthesize", _synthesize)
    monkeypatch.setattr(pipeline_module, "render_all", _render)
    state["module"] = pipeline_module
    return state


def _input(tmp_path: Path, name: str = "Лекция.mkv") -> Path:
    path = tmp_path / name
    path.write_bytes(b"media")
    return path


class TestProcessFile:
    def test_full_pipeline_layout_and_stages(self, tmp_path: Path, cfg, fake_stages: dict,
                                             args_namespace) -> None:
        input_path = _input(tmp_path)
        notes = process_file(input_path, cfg, args_namespace)
        # Раскладка по умолчанию: <stem>/ рядом с входом.
        out_dir = tmp_path / "Лекция"
        assert notes == out_dir / "notes.md"
        assert notes.exists() and notes.read_text(encoding="utf-8") == "# Конспект\n"
        assert (out_dir / "images" / "frame_00_00_00.jpg").exists()
        assert (out_dir / ".absorbent" / "frames").is_dir()
        for name in ("keyframes.json", "transcript.json", "descriptions.json"):
            assert (out_dir / ".absorbent" / name).exists()
        # Порядок этапов: язык речи (auto) и язык результата (ru) независимы.
        assert fake_stages["calls"] == [
            "extract", "dedup:10", "stt:auto", "describe:ru:1", "synthesize:ru:1", "render",
        ]

    def test_custom_output_dir_and_languages(self, tmp_path: Path, cfg, fake_stages: dict,
                                             args_namespace) -> None:
        args_namespace.output_dir = str(tmp_path / "custom")
        args_namespace.language = "en"  # подсказка whisper.
        args_namespace.output_language = "en"  # язык конспекта.
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert (tmp_path / "custom" / "notes.md").exists()
        assert "stt:en" in fake_stages["calls"]
        assert "describe:en:1" in fake_stages["calls"]
        assert "synthesize:en:1" in fake_stages["calls"]
        assert (tmp_path / "custom" / ".absorbent" / "language.txt").read_text(encoding="utf-8") == "ru"

    def test_second_run_reuses_cache(self, tmp_path: Path, cfg, fake_stages: dict,
                                     args_namespace) -> None:
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        fake_stages["calls"].clear()
        process_file(input_path, cfg, args_namespace)
        # Без force дедупликация пропускается (есть keyframes.json), а
        # extract_frames/run_stt вызываются, но возвращают кэш внутри.
        assert fake_stages["calls"] == ["extract", "stt:auto", "describe:ru:1", "synthesize:ru:1",
                                        "render"]

    def test_force_reruns_all(self, tmp_path: Path, cfg, fake_stages: dict, args_namespace) -> None:
        args_namespace.force = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert fake_stages["calls"] == ["extract", "dedup:10", "stt:auto", "describe:ru:1",
                                        "synthesize:ru:1", "render"]

    def test_skip_all_flags(self, tmp_path: Path, cfg, fake_stages: dict, args_namespace) -> None:
        args_namespace.skip_stt = True
        args_namespace.skip_frames = True
        args_namespace.skip_describe = True
        args_namespace.skip_synthesize = True
        input_path = _input(tmp_path)
        notes = process_file(input_path, cfg, args_namespace)
        assert fake_stages["calls"] == []
        # Каталоги созданы, notes.md отсутствует.
        assert notes == tmp_path / "Лекция" / "notes.md"
        assert not notes.exists()

    def test_skip_stt_still_synthesizes(self, tmp_path: Path, cfg, fake_stages: dict,
                                        args_namespace) -> None:
        # Первый прогон кэширует transcript.json, затем --skip-stt: синтез
        # работает по кэшу транскрипта, язык результата не зависит от STT.
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        fake_stages["calls"].clear()
        args_namespace.skip_stt = True
        process_file(input_path, cfg, args_namespace)
        assert not any(c.startswith("stt:") for c in fake_stages["calls"])
        assert "describe:ru:1" in fake_stages["calls"]
        assert "synthesize:ru:1" in fake_stages["calls"]

    def test_skip_frames_with_cached_keyframes(self, tmp_path: Path, cfg, fake_stages: dict,
                                               args_namespace) -> None:
        # Второй прогон без извлечения, но с кэшем keyframes.json: describe всё
        # равно получает кадры из кэша.
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        fake_stages["calls"].clear()
        args_namespace.skip_frames = True
        process_file(input_path, cfg, args_namespace)
        # Кадры не извлекаются; keyframes.json при этом не читается (skip),
        # поэтому describe не запускается, а готовые описания берутся из кэша.
        assert fake_stages["calls"] == ["stt:auto", "synthesize:ru:1", "render"]

    def test_hash_threshold_passed(self, tmp_path: Path, cfg, fake_stages: dict,
                                   args_namespace) -> None:
        args_namespace.hash_threshold = 20
        args_namespace.force = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert "dedup:20" in fake_stages["calls"]

    def test_skip_render_skips_postprocess(self, tmp_path: Path, cfg, fake_stages: dict,
                                           args_namespace) -> None:
        args_namespace.skip_render = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        # Синтез отработал, но постпроцессинг не вызван.
        assert "render" not in fake_stages["calls"]


class TestHelpers:
    def test_load_json_missing_returns_empty_list(self, tmp_path: Path) -> None:
        assert _load_json(tmp_path / "nope.json") == []

    def test_load_json_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text('[{"a": 1}]', encoding="utf-8")
        assert _load_json(path) == [{"a": 1}]

    def test_describe_from_list_cache_resets(self, tmp_path: Path, cfg, fake_stages: dict,
                                             args_namespace) -> None:
        """Сломанный (списочный) descriptions.json при пустых кадрах -> {}."""
        import absorbent.pipeline as pipeline_module

        # Перехватываем синтез, чтобы проверить, какие описания туда доехали.
        seen: dict = {}

        def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                        images_prefix="images/"):
            seen["descriptions"] = descriptions
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("x", encoding="utf-8")
            return out_md

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pipeline_module, "run_stt", lambda *a, **k: None)  # noqa: E731
        monkeypatch.setattr(pipeline_module, "run_synthesize", _synthesize)
        out_dir = tmp_path / "Лекция"
        cache = out_dir / ".absorbent"
        cache.mkdir(parents=True)
        # Кэш кадров пуст, а descriptions.json — список (не словарь) -> сброс в {}.
        (cache / "descriptions.json").write_text("[]", encoding="utf-8")
        # skip_frames -> keyframes остаются пустыми; skip_describe=False,
        # поэтому отрабатывает ветка «кадров нет, читаем кэш описаний».
        args_namespace.skip_frames = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert seen["descriptions"] == {}
        monkeypatch.undo()

    def test_skip_describe_loads_existing_descriptions(self, tmp_path: Path, cfg,
                                                       fake_stages: dict, args_namespace) -> None:
        """--skip-describe: готовые описания читаются из кэша и уходят в синтез."""
        import absorbent.pipeline as pipeline_module

        seen: dict = {}

        def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                        images_prefix="images/"):
            seen["descriptions"] = descriptions
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("x", encoding="utf-8")
            return out_md

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pipeline_module, "run_stt", lambda *a, **k: None)  # noqa: E731
        monkeypatch.setattr(pipeline_module, "run_synthesize", _synthesize)
        out_dir = tmp_path / "Лекция"
        cache = out_dir / ".absorbent"
        cache.mkdir(parents=True)
        (cache / "descriptions.json").write_text(
            json.dumps({"f.jpg": {"timestamp": "00:00:00", "description": "d"}}), encoding="utf-8")
        args_namespace.skip_describe = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert seen["descriptions"] == {"f.jpg": {"timestamp": "00:00:00", "description": "d"}}
        monkeypatch.undo()


class TestFlatMode:
    """Плоская раскладка (--flat): результат рядом с медиафайлом, кэш в .absorbent/."""

    def test_flat_layout_and_prefix(self, tmp_path: Path, cfg, fake_stages: dict,
                                    args_namespace, capsys: pytest.CaptureFixture) -> None:
        args_namespace.flat = True
        input_path = _input(tmp_path, "lecture.mp4")
        notes = process_file(input_path, cfg, args_namespace)

        assert notes == tmp_path / "lecture.md"
        assert notes.read_text(encoding="utf-8") == "# Конспект\n"
        assert (tmp_path / "lecture.json").exists()
        assert (tmp_path / "lecture.images" / "frame_00_00_00.jpg").exists()
        cache = tmp_path / ".absorbent" / "lecture"
        assert (cache / "frames").is_dir()
        assert (cache / "language.txt").read_text(encoding="utf-8") == "ru"
        # Каталога <имя>/ с notes.md в плоском режиме нет.
        assert not (tmp_path / "lecture").exists()
        # Префикс картинок для синтеза — <имя>.images/, а не images/.
        assert fake_stages["images_prefix"] == "lecture.images/"
        # Постпроцессинг получает явные пути и печатает итог без хвоста (+ …).
        assert fake_stages["render_paths"] == (tmp_path / "lecture.html", tmp_path / "lecture.pdf")
        out = capsys.readouterr().out
        assert f"готово: {notes}" in out
        assert "(+ notes.html" not in out

    def test_flat_with_output_dir(self, tmp_path: Path, cfg, fake_stages: dict,
                                  args_namespace) -> None:
        """--flat --output-dir: результат в DIR, кэш всё равно у каталога входа."""
        args_namespace.flat = True
        args_namespace.output_dir = str(tmp_path / "res")
        input_path = _input(tmp_path, "lecture.mp4")
        notes = process_file(input_path, cfg, args_namespace)
        assert notes == tmp_path / "res" / "lecture.md"
        assert (tmp_path / "res" / "lecture.json").exists()
        assert (tmp_path / "res" / "lecture.images").is_dir()
        assert (tmp_path / ".absorbent" / "lecture" / "language.txt").exists()
        assert fake_stages["images_prefix"] == "lecture.images/"

    def test_default_layout_passes_no_render_paths(self, tmp_path: Path, cfg, fake_stages: dict,
                                                   args_namespace, capsys: pytest.CaptureFixture) -> None:
        process_file(_input(tmp_path), cfg, args_namespace)
        assert fake_stages["images_prefix"] == "images/"
        assert fake_stages["render_paths"] == (None, None)
        assert "(+ notes.html, notes.pdf)" in capsys.readouterr().out

    def test_audio_input_skips_frame_stages(self, tmp_path: Path, cfg, fake_stages: dict,
                                           args_namespace, capsys: pytest.CaptureFixture) -> None:
        """Аудиофайл без видеопотока: кадры не извлекаются, конспект без картинок."""
        fake_stages["video"] = False
        args_namespace.flat = True
        notes = process_file(_input(tmp_path, "lecture.mp3"), cfg, args_namespace)
        assert "extract" not in fake_stages["calls"]
        assert not any(c.startswith("dedup") for c in fake_stages["calls"])
        assert not any(c.startswith("describe") for c in fake_stages["calls"])
        assert fake_stages["calls"] == ["stt:auto", "synthesize:ru:0", "render"]
        assert "кадров нет (аудио без видеопотока)" in capsys.readouterr().out
        assert notes.exists()


class TestServerLifecycle:
    """Локальный сервер vLLM: поднимается до STT и гасится в finally."""

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

    def test_started_before_stt_and_stopped_after_render(self, tmp_path: Path, fake_stages: dict,
                                                        args_namespace,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.realtime as realtime_module

        cfg = load_config(str(self._realtime_config(tmp_path)))
        proc = object()
        recorded: list = []

        def _ensure(provider, model, chunk_seconds, *, proc_holder):
            recorded.append(("ensure", provider.name, model, chunk_seconds))
            fake_stages["calls"].append("ensure_server")
            proc_holder.append(proc)

        def _stop(stopped):
            recorded.append(("stop", stopped))
            fake_stages["calls"].append("stop_server")

        monkeypatch.setattr(realtime_module, "ensure_server", _ensure)
        monkeypatch.setattr(realtime_module, "stop_server", _stop)
        process_file(_input(tmp_path, "lecture.mp3"), cfg, args_namespace)
        assert recorded == [("ensure", "vllm-voxtral", "realtime-model", 120.0), ("stop", proc)]
        assert fake_stages["calls"] == ["extract", "dedup:10", "ensure_server", "stt:auto",
                                        "describe:ru:1", "synthesize:ru:1", "render",
                                        "stop_server"]

    def test_stopped_even_if_stage_fails(self, tmp_path: Path, fake_stages: dict, args_namespace,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.pipeline as pipeline_module
        import absorbent.realtime as realtime_module

        cfg = load_config(str(self._realtime_config(tmp_path)))
        proc = object()
        stopped: list = []
        monkeypatch.setattr(realtime_module, "ensure_server",
                            lambda provider, model, chunk_seconds, *, proc_holder:
                            proc_holder.append(proc))
        monkeypatch.setattr(realtime_module, "stop_server", lambda p: stopped.append(p))

        def _boom(*args, **kwargs):
            raise RuntimeError("синтез упал")

        monkeypatch.setattr(pipeline_module, "run_synthesize", _boom)
        with pytest.raises(RuntimeError, match="синтез упал"):
            process_file(_input(tmp_path, "lecture.mp3"), cfg, args_namespace)
        assert stopped == [proc]

    def test_whisper_backend_never_starts_server(self, tmp_path: Path, cfg, fake_stages: dict,
                                                 args_namespace,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.realtime as realtime_module

        def _boom(*args, **kwargs):  # pragma: no cover - не должен вызываться.
            raise AssertionError("для whisper сервер vLLM не нужен")

        monkeypatch.setattr(realtime_module, "ensure_server", _boom)
        monkeypatch.setattr(realtime_module, "stop_server", _boom)
        process_file(_input(tmp_path), cfg, args_namespace)

    def test_borrowed_holder_leaves_server_to_caller(self, tmp_path: Path, fake_stages: dict,
                                                    args_namespace,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """Пакетный прогон: сервером распоряжается вызывающий, pipeline его не трогает."""
        import absorbent.realtime as realtime_module

        cfg = load_config(str(self._realtime_config(tmp_path)))
        events: list = []

        def _boom(*args, **kwargs):  # pragma: no cover - не должен вызываться.
            raise AssertionError("сервером владеет вызывающий")

        monkeypatch.setattr(realtime_module, "ensure_server", _boom)
        monkeypatch.setattr(realtime_module, "stop_server", _boom)
        holder: list = [object()]
        for name in ("a.mp3", "b.mp3"):
            process_file(_input(tmp_path, name), cfg, args_namespace, server_holder=holder)
        assert events == []
        assert len(holder) == 1
