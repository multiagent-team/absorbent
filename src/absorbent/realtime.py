"""Локальный backend Voxtral Realtime: сервер vLLM, WS-клиент и нарезка аудио.

Веса `mistralai/Voxtral-Mini-4B-Realtime-2602` поддерживает только vLLM
(llama.cpp и llama-swap эту архитектуру не умеют), поэтому модуль сам
поднимает `vllm serve`, ждёт готовности, гоняет протокол `/v1/realtime` по
WebSocket и останавливает сервер, когда конвейер закончил работу.

Ни таймкодов, ни спикеров протокол не отдаёт, а `session.update` принимает
только имя модели. Сегменты контракта transcript.json получаются нарезкой
аудио на фрагменты по паузам (`silencedetect`); поле `speaker` всегда None.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from absorbent.config import Provider

# Один текстовый токен Voxtral Realtime закрывает ~80 мс аудио.
AUDIO_TOKENS_PER_SECOND = 12.5
# Формат обмена: s16le, 16 кГц, моно — столько байт в секунду.
PCM_BYTES_PER_SECOND = 32000
# Размер одного append: ~2 с аудио, чтобы не дробить кадры без нужды.
WS_APPEND_BYTES = 65536
# Порог и минимальная длительность паузы для silencedetect.
SILENCE_NOISE = "-35dB"
SILENCE_MIN_DURATION = 0.5
# Первый старт vLLM скачивает ~9 ГБ весов.
SERVER_READY_TIMEOUT = 900.0
# Таймаут фрагмента: chunk_seconds * это плюс запас на прогрев и хвост.
CHUNK_TIMEOUT_FACTOR = 4.0

# Начало паузы в stderr silencedetect: "silence_start: 12.34".
_SILENCE_RE = re.compile(r"silence_start: ([0-9.]+)")


class RealtimeError(Exception):
    """Ошибка локального сервера vLLM или протокола /v1/realtime."""


def cache_root(input_dir: Path) -> Path:
    """Каталог кэша Absorbent внутри каталога входных файлов."""
    return input_dir / ".absorbent"


def flat_cache_dir(input_dir: Path, stem: str) -> Path:
    """Кэш плоского режима для одного файла: <каталог>/.absorbent/<имя>."""
    return cache_root(input_dir) / stem


def vllm_log_path() -> Path:
    """Путь лога автозапущенного сервера vLLM."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "absorbent" / "vllm.log"


def max_model_len(chunk_seconds: float) -> int:
    """Окно модели под фрагмент: аудиотокены фрагмента плюс запас на промпт."""
    return max(4096, int(chunk_seconds * AUDIO_TOKENS_PER_SECOND) + 2048)


def probe_server(base_url: str) -> dict | None:
    """Ответ /v1/models как словарь; None, если сервер молчит или ответ не JSON."""
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:  # HTTP-ошибка, обрыв, не-JSON — все означают «сервера нет».
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def port_open(base_url: str) -> bool:
    """Принимает ли соединения TCP-порт из base_url (занят ли он хоть кем-нибудь)."""
    parts = urlsplit(base_url)
    if parts.port is None:
        return False
    try:
        with socket.create_connection((parts.hostname or "127.0.0.1", parts.port), timeout=1.0):
            return True
    except OSError:
        return False


def _serves_model(payload: dict, model: str) -> bool:
    """Есть ли модель среди id ответа /v1/models (vLLM отдаёт путь или имя модели)."""
    entries = payload.get("data") or []
    return any(model in str(entry.get("id", "")) for entry in entries)


def _log_tail(lines: int = 40) -> str:
    """Хвост лога сервера vLLM для сообщений об ошибке."""
    path = vllm_log_path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def start_server(base_url: str, model: str, vllm_bin: str, gpu: str,
                 chunk_seconds: float) -> subprocess.Popen:
    """Запустить `vllm serve` отдельной сессией процессов; вернуть процесс."""
    parts = urlsplit(base_url)
    host = parts.hostname or "127.0.0.1"
    if parts.port is None:
        raise RealtimeError(f"в адресе vLLM '{base_url}' не указан порт")
    if vllm_bin:
        binary = shutil.which(vllm_bin) or (vllm_bin if Path(vllm_bin).is_file() else "")
    else:
        binary = shutil.which("vllm") or ""
    if not binary:
        raise RealtimeError(
            f"не найден бинарь vLLM '{vllm_bin or 'vllm'}'; "
            'установите: uv tool install "vllm[audio]"'
        )
    log_path = vllm_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [binary, "serve", model, "--host", host, "--port", str(parts.port),
           "--enforce-eager", "--max-model-len", str(max_model_len(chunk_seconds)),
           "--gpu-memory-utilization", "0.85"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu} if gpu else dict(os.environ)
    print(f"запуск сервера vLLM: {' '.join(cmd)}")
    print(f"лог сервера: {log_path}")
    with open(log_path, "w", encoding="utf-8") as log:
        # Файл лога наследуется процессом: своей копии дескриптора хватает,
        # поэтому родительский дескриптор закрывается сразу после запуска.
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)


def wait_ready(proc: subprocess.Popen, base_url: str, model: str,
               timeout: float = SERVER_READY_TIMEOUT) -> None:
    """Дождаться готовности сервера: модель видна в /v1/models.

    Смерть процесса или исчерпание timeout — RealtimeError с хвостом лога.
    """
    deadline = time.monotonic() + timeout
    while True:
        payload = probe_server(base_url)
        if payload is not None and _serves_model(payload, model):
            print(f"сервер vLLM готов: {base_url}")
            return
        if proc.poll() is not None:
            raise RealtimeError(
                f"сервер vLLM завершился с кодом {proc.returncode}; "
                f"хвост {vllm_log_path()}:\n{_log_tail()}"
            )
        if time.monotonic() >= deadline:
            raise RealtimeError(
                f"не дождались готовности vLLM ({base_url}) за {timeout:.0f} с; "
                f"хвост {vllm_log_path()}:\n{_log_tail()}"
            )
        time.sleep(2.0)


def stop_server(proc: subprocess.Popen) -> None:
    """Остановить сервер: SIGTERM всей группе процессов, при упорстве — SIGKILL."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def ensure_server(provider: Provider, model: str, chunk_seconds: float, *,
                  proc_holder: list) -> None:
    """Гарантировать живой сервер vLLM с нужной моделью.

    Сервер отвечает и отдаёт модель — ничего не делаем. Порт занят чужим
    сервисом — RealtimeError. Порт закрыт и autostart выключен — RealtimeError.
    Иначе запускаем `vllm serve`, ждём готовности и кладём процесс в
    proc_holder: вызывающий останавливает его в finally.
    """
    payload = probe_server(provider.base_url)
    if payload is not None and _serves_model(payload, model):
        print(f"сервер vLLM уже запущен: {provider.base_url}")
        return
    if payload is not None:
        raise RealtimeError(
            f"на порту {provider.base_url} работает другой сервис: "
            f"в /v1/models нет модели '{model}'"
        )
    if port_open(provider.base_url):
        raise RealtimeError(f"порт {provider.base_url} занят не-vLLM сервисом")
    if not provider.autostart:
        raise RealtimeError(f"сервер vLLM недоступен: {provider.base_url}")
    proc = start_server(provider.base_url, model, provider.vllm_bin, provider.gpu,
                        chunk_seconds)
    try:
        wait_ready(proc, provider.base_url, model)
    except RealtimeError:
        stop_server(proc)  # Не оставляем полуживой сервер после неудачного старта.
        raise
    proc_holder.append(proc)


def extract_pcm(input_path: Path, pcm_path: Path, force: bool = False) -> float:
    """Извлечь аудио как s16le 16 кГц моно; вернуть длительность в секундах.

    Повторный вызов без force переиспользует готовый файл: длительность равна
    размеру, делённому на байты в секунду.
    """
    if pcm_path.exists() and not force:
        return pcm_path.stat().st_size / PCM_BYTES_PER_SECOND
    pcm_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(input_path), "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", str(pcm_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RealtimeError(f"ffmpeg не извлёк аудио: {proc.stderr.strip()}")
    return pcm_path.stat().st_size / PCM_BYTES_PER_SECOND


def find_silences(pcm_path: Path, cache_path: Path, force: bool = False) -> list[float]:
    """Начала пауз в PCM-дорожке через silencedetect; кэш — silences.json.

    Ошибка ffmpeg не прерывает прогон: печатается предупреждение, а пустой
    список заставляет нарезать аудио ровно по времени.
    """
    if cache_path.exists() and not force:
        return [float(value) for value in json.loads(cache_path.read_text(encoding="utf-8"))]
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-f", "s16le",
         "-ar", "16000", "-ac", "1", "-i", str(pcm_path),
         "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_MIN_DURATION}",
         "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"внимание: не удалось определить паузы ({proc.stderr.strip()}); "
              "нарезка по времени")
        silences: list[float] = []
    else:
        silences = [float(match) for match in _SILENCE_RE.findall(proc.stderr)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(silences), encoding="utf-8")
    return silences


def chunk_boundaries(duration: float, silences: list[float], target: float) -> list[float]:
    """Моменты резов аудио: каждый — пауза, ближайшая к очередной цели target.

    Если пауз рядом нет, режем по времени (target). Резы строго возрастают и
    не выходят за duration.
    """
    cuts: list[float] = []
    target_pos = target
    while target_pos < duration:
        window = [s for s in silences if abs(s - target_pos) <= target / 2]
        cut = min(window, key=lambda s: abs(s - target_pos)) if window else target_pos
        cut = max(cut, (cuts[-1] if cuts else 0.0) + 1.0)
        if cut >= duration:
            break
        cuts.append(cut)
        target_pos = cut + target
    return cuts


def segments_from_chunks(duration: float, cuts: list[float],
                         texts: list[str]) -> list[dict]:
    """Сегменты контракта transcript.json из текстов фрагментов и резов.

    Чанк i идёт от cuts[i-1] (или нуля) до cuts[i] (или конца файла); пустой
    текст означает, что фрагмент распознать не удалось, и сегмент пропускается.
    """
    segments: list[dict] = []
    for index, text in enumerate(texts):
        start = cuts[index - 1] if index else 0.0
        end = cuts[index] if index < len(cuts) else duration
        text = text.strip()
        if not text:
            continue
        segments.append({"start": float(start), "end": float(end), "text": text,
                         "speaker": None})
    return segments


def realtime_ws_url(base_url: str) -> str:
    """Адрес /v1/realtime по http(s)-адресу провайдера."""
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + "/v1/realtime"
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + "/v1/realtime"
    raise RealtimeError(f"неизвестная схема адреса vLLM: '{base_url}'")


async def _transcribe_chunk(pcm: bytes, ws_url: str, model: str) -> str:
    """Один фрагмент по протоколу /v1/realtime: append-ы, commit(final), текст."""
    async with connect(ws_url, max_size=None) as ws:
        await ws.recv()  # session.created сервер шлёт сразу после accept.
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        for offset in range(0, len(pcm), WS_APPEND_BYTES):
            block = pcm[offset:offset + WS_APPEND_BYTES]
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(block).decode("ascii")}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        parts: list[str] = []
        async for raw in ws:
            event = json.loads(raw)
            if event.get("type") == "transcription.delta":
                parts.append(event.get("delta", ""))
            elif event.get("type") == "transcription.done":
                return (event.get("text") or "".join(parts)).strip()
            elif event.get("type") == "error":
                raise RealtimeError(f"vLLM: {event.get('error')} (code={event.get('code')})")
        # Соединение закрылось без done: отдаём накопленные дельты.
        return "".join(parts).strip()


def transcribe_chunk(pcm: bytes, ws_url: str, model: str, timeout: float) -> str:
    """Синхронная обёртка: распознать фрагмент с таймаутом на весь обмен.

    Таймаут и обрыв соединения поднимаются как RealtimeError: вызывающий
    обрабатывает ошибки фрагмента единообразно.
    """
    try:
        return asyncio.run(asyncio.wait_for(_transcribe_chunk(pcm, ws_url, model), timeout))
    except TimeoutError as exc:
        raise RealtimeError(f"таймаут распознавания фрагмента ({timeout:.0f} с)") from exc
    except (OSError, WebSocketException) as exc:
        raise RealtimeError(f"соединение с vLLM не удалось: {exc}") from exc
