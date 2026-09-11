"""Загрузка и валидация конфигурации Absorbent из YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Порядок поиска конфигурации, если не задан явный путь через --config.
CONFIG_CANDIDATES: tuple[str, ...] = (
    "./config.yaml",
    "~/.config/absorbent/config.yaml",
)


class ConfigError(Exception):
    """Ошибка конфигурации: отсутствующий файл, ключ или переменная окружения."""


@dataclass
class Provider:
    """Провайдер OpenAI-совместимого API.

    Поля локального сервера (vllm_bin, gpu, autostart) используются только
    провайдерами, которые Absorbent поднимает сам (vLLM для Voxtral
    Realtime); у остальных они остаются дефолтными.
    """

    name: str
    base_url: str
    api_key_env: str = ""
    vllm_bin: str = ""
    gpu: str = ""
    autostart: bool = True


@dataclass
class ModelSpec:
    """Спецификация модели в одной из категорий (stt, vision, llm)."""

    name: str
    provider: str = ""
    backend: str = ""
    whisper_model: str = ""
    model: str = ""
    default: bool = False
    chunk_seconds: float = 300.0

    @property
    def api_model(self) -> str:
        """Имя модели для API (для whisper-бэкенда не используется)."""
        return self.model or self.name


@dataclass
class Config:
    """Полностью разобранная конфигурация."""

    providers: dict[str, Provider] = field(default_factory=dict)
    stt: dict[str, ModelSpec] = field(default_factory=dict)
    vision: dict[str, ModelSpec] = field(default_factory=dict)
    llm: dict[str, ModelSpec] = field(default_factory=dict)

    def resolve(self, category: str, name: str | None = None) -> tuple[Provider, ModelSpec]:
        """Вернуть (provider, spec) для имени или дефолта категории."""
        table = getattr(self, category)
        if name is None:
            for spec in table.values():
                if spec.default:
                    return self._provider_for(spec), spec
            raise ConfigError(f"в категории '{category}' не задана модель с default: true")
        if name not in table:
            valid = ", ".join(table) or "(нет моделей)"
            raise ConfigError(f"неизвестная модель '{name}' в категории '{category}'; доступны: {valid}")
        spec = table[name]
        return self._provider_for(spec), spec

    def _provider_for(self, spec: ModelSpec) -> Provider:
        if spec.backend == "whisper":
            return Provider(name="whisper", base_url="")
        if spec.provider not in self.providers:
            raise ConfigError(f"модель '{spec.name}' ссылается на неизвестного провайдера '{spec.provider}'")
        return self.providers[spec.provider]


def api_key_from(provider: Provider) -> str | None:
    """Ключ API провайдера из переменной окружения; None при пустой api_key_env."""
    if not provider.api_key_env:
        return None
    value = os.environ.get(provider.api_key_env)
    if not value:
        raise ConfigError(f"missing env var {provider.api_key_env}")
    return value


def _parse_model_table(raw: dict, category: str) -> dict[str, ModelSpec]:
    table: dict[str, ModelSpec] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ConfigError(f"модель '{name}' в категории '{category}' должна быть словарём")
        chunk_seconds = float(body.get("chunk_seconds", 300.0))
        if chunk_seconds <= 0:
            raise ConfigError(
                f"у модели '{name}' в категории '{category}' chunk_seconds должен быть положительным"
            )
        table[name] = ModelSpec(
            name=name,
            provider=str(body.get("provider", "")),
            backend=str(body.get("backend", "")),
            whisper_model=str(body.get("whisper_model", "")),
            model=str(body.get("model", "")),
            default=bool(body.get("default", False)),
            chunk_seconds=chunk_seconds,
        )
    return table


def load_config(path: str | None) -> Config:
    """Загрузить конфигурацию из path или первого существующего кандидата."""
    chosen: Path | None = None
    if path is not None:
        chosen = Path(path)
        if not chosen.is_file():
            raise ConfigError(f"файл конфигурации не найден: {path}")
    else:
        for candidate in CONFIG_CANDIDATES:
            expanded = Path(candidate).expanduser()
            if expanded.is_file():
                chosen = expanded
                break
    if chosen is None:
        raise ConfigError(
            "конфигурация не найдена; искал: " + ", ".join(str(Path(c).expanduser()) for c in CONFIG_CANDIDATES)
        )
    with open(chosen, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"конфигурация '{chosen}' должна быть YAML-словарём")

    providers: dict[str, Provider] = {}
    for name, body in (raw.get("providers") or {}).items():
        if not isinstance(body, dict):
            raise ConfigError(f"провайдер '{name}' должен быть словарём")
        providers[name] = Provider(
            name=name,
            base_url=str(body.get("base_url", "")),
            api_key_env=str(body.get("api_key_env", "")),
            # Тильду раскрываем здесь: значение уходит в argv для Popen.
            vllm_bin=str(body.get("vllm_bin", "") or "").replace("~", str(Path.home())),
            gpu=str(body.get("gpu", "") or ""),
            autostart=bool(body.get("autostart", True)),
        )

    return Config(
        providers=providers,
        stt=_parse_model_table(raw.get("stt") or {}, "stt"),
        vision=_parse_model_table(raw.get("vision") or {}, "vision"),
        llm=_parse_model_table(raw.get("llm") or {}, "llm"),
    )
