"""Тесты конфигурации: парсинг YAML, поиск файла, резолв моделей, ключи API."""

from __future__ import annotations

from pathlib import Path

import pytest

from absorbent.config import CONFIG_CANDIDATES, Config, ConfigError, ModelSpec, Provider, api_key_from, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _repo_cfg() -> Config:
    return load_config(str(REPO_CONFIG))


class TestLoadConfig:
    def test_explicit_path(self, sample_config: Path) -> None:
        cfg = load_config(str(sample_config))
        assert isinstance(cfg, Config)
        # Провайдеры из штатного файла.
        assert set(cfg.providers) == {"deepseek", "llama-swap", "mistral", "vllm-voxtral"}
        assert cfg.providers["deepseek"].base_url == "https://api.deepseek.com"
        assert cfg.providers["deepseek"].api_key_env == "DEEPSEEK_API_KEY"
        assert cfg.providers["llama-swap"].api_key_env == ""
        # Провайдер локального vLLM: путь к бинарю (с раскрытой тильдой), GPU,
        # автозапуск; у остальных провайдеров эти поля дефолтные.
        vllm = cfg.providers["vllm-voxtral"]
        assert vllm.base_url == "http://127.0.0.1:12435"
        assert vllm.api_key_env == ""
        assert vllm.vllm_bin == str(Path.home() / ".local" / "bin" / "vllm")
        assert vllm.gpu == "1"
        assert vllm.autostart is True
        assert cfg.providers["mistral"].vllm_bin == ""
        assert cfg.providers["mistral"].gpu == ""
        assert cfg.providers["mistral"].autostart is True
        # Категории моделей.
        assert set(cfg.stt) == {"whisper-turbo", "voxtral-transcribe-2", "voxtral-realtime"}
        assert cfg.stt["whisper-turbo"].backend == "whisper"
        assert cfg.stt["whisper-turbo"].whisper_model == "turbo"
        assert cfg.stt["whisper-turbo"].default is True
        assert cfg.stt["voxtral-transcribe-2"].backend == "voxtral"
        assert cfg.stt["voxtral-transcribe-2"].provider == "mistral"
        assert cfg.stt["voxtral-transcribe-2"].model == "voxtral-mini-latest"
        assert cfg.stt["voxtral-transcribe-2"].api_model == "voxtral-mini-latest"
        assert cfg.stt["voxtral-realtime"].backend == "voxtral-realtime"
        assert cfg.stt["voxtral-realtime"].provider == "vllm-voxtral"
        assert cfg.stt["voxtral-realtime"].model == "mistralai/Voxtral-Mini-4B-Realtime-2602"
        assert cfg.stt["voxtral-realtime"].chunk_seconds == 300.0
        assert cfg.vision["deepseek-v4-flash-vision-exp"].default is True
        assert cfg.llm["deepseek-v4-flash"].default is True
        assert not cfg.llm["deepseek-v4-pro"].default

    def test_missing_explicit_path(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="не найден"):
            load_config(str(tmp_path / "nope.yaml"))

    def test_search_first_candidate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        first = tmp_path / "c1.yaml"
        first.write_text("providers: {}\n", encoding="utf-8")
        monkeypatch.setattr("absorbent.config.CONFIG_CANDIDATES", (str(first), str(tmp_path / "c2.yaml")))
        cfg = load_config(None)
        assert cfg.providers == {}

    def test_search_falls_through_to_second(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        first = tmp_path / "c1.yaml"
        second = tmp_path / "c2.yaml"
        second.write_text("providers: {x: {base_url: u}}\n", encoding="utf-8")
        monkeypatch.setattr("absorbent.config.CONFIG_CANDIDATES", (str(first), str(second)))
        cfg = load_config(None)
        assert "x" in cfg.providers

    def test_no_candidate_lists_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        paths = (str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"))
        monkeypatch.setattr("absorbent.config.CONFIG_CANDIDATES", paths)
        with pytest.raises(ConfigError) as err:
            load_config(None)
        assert str(tmp_path / "a.yaml") in str(err.value)
        assert str(tmp_path / "b.yaml") in str(err.value)

    def test_non_dict_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML-словарём"):
            load_config(str(path))

    def test_provider_must_be_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("providers:\n  bad: 42\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="словарём"):
            load_config(str(path))

    def test_model_must_be_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  bad: 42\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="словарём"):
            load_config(str(path))

    def test_default_candidates_are_sane(self) -> None:
        # Штатный порядок поиска: явный файл в cwd, затем ~/.config/absorbent.
        assert CONFIG_CANDIDATES[0] == "./config.yaml"
        assert CONFIG_CANDIDATES[1] == "~/.config/absorbent/config.yaml"


class TestChunkSeconds:
    def test_explicit_value(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "stt:\n  rt: {backend: voxtral-realtime, chunk_seconds: 45.5, default: true}\n",
            encoding="utf-8",
        )
        cfg = load_config(str(path))
        assert cfg.stt["rt"].chunk_seconds == 45.5

    def test_default_is_300(self) -> None:
        cfg = _repo_cfg()
        assert cfg.stt["whisper-turbo"].chunk_seconds == 300.0

    def test_non_positive_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("stt:\n  rt: {backend: voxtral-realtime, chunk_seconds: 0}\n",
                        encoding="utf-8")
        with pytest.raises(ConfigError, match="chunk_seconds должен быть положительным"):
            load_config(str(path))

    def test_negative_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("stt:\n  rt: {chunk_seconds: -1}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="'rt' в категории 'stt'"):
            load_config(str(path))


class TestLocalServerProvider:
    def test_defaults_without_local_fields(self, minimal_config: Path) -> None:
        cfg = load_config(str(minimal_config))
        provider = cfg.providers["prov-a"]
        assert provider.vllm_bin == ""
        assert provider.gpu == ""
        assert provider.autostart is True

    def test_explicit_local_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "providers:\n"
            "  local:\n"
            "    base_url: http://127.0.0.1:12435\n"
            "    api_key_env: ''\n"
            "    vllm_bin: ~/bin/vllm\n"
            "    gpu: '0'\n"
            "    autostart: false\n",
            encoding="utf-8",
        )
        provider = load_config(str(path)).providers["local"]
        assert provider.vllm_bin == str(Path.home() / "bin" / "vllm")
        assert provider.gpu == "0"
        assert provider.autostart is False

    def test_empty_local_fields(self, tmp_path: Path) -> None:
        """Пустые значения (null или '') не превращаются в строку 'None'."""
        path = tmp_path / "config.yaml"
        path.write_text(
            "providers:\n  local: {base_url: u, vllm_bin: null, gpu: ''}\n",
            encoding="utf-8",
        )
        provider = load_config(str(path)).providers["local"]
        assert provider.vllm_bin == ""
        assert provider.gpu == ""


class TestResolve:
    def test_defaults(self) -> None:
        cfg = _repo_cfg()
        stt_provider, stt_spec = cfg.resolve("stt")
        assert stt_spec.name == "whisper-turbo"
        assert stt_provider.name == "whisper"  # whisper-бэкенд не имеет провайдера.
        assert stt_provider.base_url == ""
        _, vision_spec = cfg.resolve("vision")
        assert vision_spec.name == "deepseek-v4-flash-vision-exp"
        _, llm_spec = cfg.resolve("llm")
        assert llm_spec.name == "deepseek-v4-flash"

    def test_named(self) -> None:
        cfg = _repo_cfg()
        provider, spec = cfg.resolve("vision", "gemma4-26b-vision")
        assert spec.name == "gemma4-26b-vision"
        assert provider.name == "llama-swap"
        assert provider.base_url == "http://127.0.0.1:12434/v1"

    def test_unknown_name_lists_valid(self) -> None:
        cfg = _repo_cfg()
        with pytest.raises(ConfigError) as err:
            cfg.resolve("stt", "no-such")
        assert "неизвестная модель 'no-such'" in str(err.value)
        assert "whisper-turbo" in str(err.value)
        assert "voxtral" in str(err.value)

    def test_no_default_in_category(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "providers:\n  p: {base_url: u, api_key_env: K}\nvision:\n  m1: {provider: p}\n",
            encoding="utf-8",
        )
        cfg = load_config(str(path))
        with pytest.raises(ConfigError, match="default: true"):
            cfg.resolve("vision")

    def test_unknown_provider(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  m1: {provider: ghost, default: true}\n", encoding="utf-8")
        cfg = load_config(str(path))
        with pytest.raises(ConfigError, match="ghost"):
            cfg.resolve("llm")

    def test_empty_category(self) -> None:
        cfg = Config()
        with pytest.raises(ConfigError, match="не задана модель"):
            cfg.resolve("stt")

    def test_api_model_falls_back_to_name(self) -> None:
        spec = ModelSpec(name="only-name", provider="p")
        assert spec.api_model == "only-name"
        assert spec.default is False
        assert spec.whisper_model == ""


class TestApiKey:
    def test_empty_env_name_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_VAR", raising=False)
        assert api_key_from(Provider(name="p", base_url="u", api_key_env="")) is None

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ABSORBENT_MISSING_KEY", raising=False)
        with pytest.raises(ConfigError, match="missing env var ABSORBENT_MISSING_KEY"):
            api_key_from(Provider(name="p", base_url="u", api_key_env="ABSORBENT_MISSING_KEY"))

    def test_present_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ABSORBENT_TEST_KEY", "sekret")
        assert api_key_from(Provider(name="p", base_url="u", api_key_env="ABSORBENT_TEST_KEY")) == "sekret"
