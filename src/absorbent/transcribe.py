"""Распознавание речи: whisper (локально), voxtral (Mistral API), voxtral-realtime."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from absorbent import realtime
from absorbent.config import ModelSpec, Provider, api_key_from

# Канонический промежуточный формат — transcript.json:
# [{"start": float, "end": float, "text": str, "speaker": str | None}, ...]


class STTError(Exception):
    """Ошибка этапа распознавания речи."""


def _write_transcript(out_json: Path, segments: list[dict]) -> None:
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(segments, handle, ensure_ascii=False, indent=2)


def _whisper_transcribe(spec: ModelSpec, input_path: Path, language: str) -> tuple[list[dict], str | None]:
    """Локальная транскрипция через openai-whisper; возвращает (сегменты, язык)."""
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(spec.whisper_model, device=device)
    result = model.transcribe(
        str(input_path),
        language=(None if language == "auto" else language),
        beam_size=5,
        verbose=False,
    )
    segments = [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip(), "speaker": None}
        for seg in result["segments"]
    ]
    detected = result.get("language")
    return segments, detected


def _audio_duration_seconds(audio_path: Path) -> float:
    """Длительность аудиофайла через ffprobe (для одиночного сегмента-фолбэка)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
         str(audio_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0.0
    return float(proc.stdout.strip())


def _extract_audio(input_path: Path, audio_path: Path) -> None:
    """Извлечь моно-дорожку 16 кГц для отправки в Voxtral."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(input_path), "-vn", "-ar", "16000", "-ac", "1",
         str(audio_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise STTError(f"ffmpeg не извлёк аудио: {proc.stderr.strip()}")


def _voxtral_transcribe(spec: ModelSpec, provider: Provider, input_path: Path, language: str,
                        audio_path: Path) -> tuple[list[dict], str | None]:
    """Облачная транскрипция через OpenAI-совместимый /v1/audio/transcriptions Mistral.

    Диаризация и тайминги запрашиваются явно (diarize, сегменты); имена полей
    ответа соответствуют живой документации Mistral: segments[].{text,start,end,
    speaker_id}. Ошибки API всплывают как STTError.
    """
    from openai import OpenAI

    api_key = api_key_from(provider)
    # Заголовок x-api-key: именно так устроен задокументированный curl-вызов
    # Mistral; Bearer от OpenAI SDK дублируется на тот случай, если
    # OpenAI-совместимая прослойка ожидает именно его.
    extra_headers = {"X-Api-Key": api_key} if api_key else {}
    client = OpenAI(base_url=provider.base_url, api_key=api_key)
    _extract_audio(input_path, audio_path)

    with open(audio_path, "rb") as audio_file:
        # Mistral-специфичные поля уходят через extra_body, а не именованные
        # аргументы: diarize отсутствует в сигнатуре OpenAI SDK, а типизация
        # timestamp_granularities меняется между версиями openai.
        raw = client.audio.transcriptions.with_raw_response.create(
            model=spec.api_model,
            file=(audio_path.name, audio_file, "audio/wav"),
            response_format="verbose_json",
            extra_headers=extra_headers,
            extra_body={
                "diarize": True,
                "timestamp_granularities": "segment",
                "language": (None if language == "auto" else language),
            },
        )
    data = raw.json()

    segments = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
            "speaker": seg.get("speaker_id") or seg.get("speaker"),
        })
    if not segments and data.get("text"):
        # Фолбэк без сегментов: один сегмент на всю длительность дорожки.
        segments.append({
            "start": 0.0,
            "end": _audio_duration_seconds(audio_path),
            "text": data["text"].strip(),
            "speaker": None,
        })
    detected = data.get("language") or None
    return segments, detected


def _transcribe_fragment_with_retry(pcm: bytes, ws_url: str, model: str, timeout: float,
                                    start: float, end: float) -> str:
    """Распознать фрагмент; при ошибке — одна повторная попытка."""
    try:
        return realtime.transcribe_chunk(pcm, ws_url, model, timeout)
    except realtime.RealtimeError as exc:
        print(f"повтор фрагмента {start:.1f}–{end:.1f}: {exc}")
        return realtime.transcribe_chunk(pcm, ws_url, model, timeout)


def _voxtral_realtime_transcribe(spec: ModelSpec, provider: Provider, input_path: Path,
                                 language: str, work_dir: Path,
                                 force: bool) -> tuple[list[dict], str | None]:
    """Локальная транскрипция Voxtral Realtime через vLLM: фрагменты по паузам.

    Протокол /v1/realtime не отдаёт ни таймкодов, ни спикеров, поэтому текст
    раскладывается по фрагментам, нарезанным по паузам (границы фрагментов и
    становятся start/end сегментов). Сервер к этому моменту уже поднят
    вызывающим (pipeline.process_file), поэтому здесь только обмен.

    Ошибка фрагмента не роняет прогон: фрагмент пропускается, а в конце
    печатается число неудач. Ошибка первого фрагмента пробрасывается —
    это признак мёртвого сервера.
    """
    pcm_path = work_dir / "audio.s16le"
    duration = realtime.extract_pcm(input_path, pcm_path, force)
    silences = realtime.find_silences(pcm_path, work_dir / "silences.json", force)
    cuts = realtime.chunk_boundaries(duration, silences, spec.chunk_seconds)
    chunk_count = len(cuts) + 1
    print(f"длительность {duration:.1f} с, фрагментов {chunk_count}")
    if language in ("ru", "en"):
        print("backend voxtral-realtime не принимает подсказку языка, модель определяет его сама")

    ws_url = realtime.realtime_ws_url(provider.base_url)
    timeout = spec.chunk_seconds * realtime.CHUNK_TIMEOUT_FACTOR + 300
    pcm = pcm_path.read_bytes()
    texts: list[str] = []
    failed = 0
    for index in range(chunk_count):
        start = cuts[index - 1] if index else 0.0
        end = cuts[index] if index < len(cuts) else duration
        print(f"[{index + 1}/{chunk_count}] {start:.1f}–{end:.1f} с")
        block = pcm[int(start * realtime.PCM_BYTES_PER_SECOND):
                    int(end * realtime.PCM_BYTES_PER_SECOND)]
        try:
            texts.append(_transcribe_fragment_with_retry(block, ws_url, spec.api_model,
                                                         timeout, start, end))
        except realtime.RealtimeError as exc:
            if index == 0:
                raise  # Первый фрагмент: сервер, скорее всего, умер — не глотаем.
            print(f"ошибка фрагмента {start:.1f}–{end:.1f}: {exc}")
            failed += 1
            texts.append("")
    if failed:
        print(f"внимание: {failed} фрагмент(ов) не распознаны")
    return realtime.segments_from_chunks(duration, cuts, texts), None


def run_stt(spec: ModelSpec, provider: Provider, input_path: Path, language: str, out_json: Path,
            work_dir: Path, force: bool = False) -> str | None:
    """Запустить распознавание речи; вернуть обнаруженный язык ("ru"/"en"/None).

    Пишет transcript.json в out_json, а служебные файлы этапа — в work_dir
    (каталог кэша): language.txt и извлечённое для облачного Voxtral аудио
    audio.wav. В плоском режиме результат (transcript.json) лежит рядом с
    медиафайлом, а кэш — в <каталог>/.absorbent/<имя>/, поэтому пути
    результата и кэша разведены. При наличии кэша и без force возвращает
    язык из кэша, не транскрибируя. Пустой результат (нет аудиодорожки)
    пишется как [] с предупреждением.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    lang_file = work_dir / "language.txt"
    if out_json.exists() and not force:
        if lang_file.exists():
            return lang_file.read_text(encoding="utf-8").strip() or None
        return None

    if spec.backend == "whisper":
        segments, detected = _whisper_transcribe(spec, input_path, language)
    elif spec.backend == "voxtral":
        audio_path = work_dir / "audio.wav"
        segments, detected = _voxtral_transcribe(spec, provider, input_path, language, audio_path)
    elif spec.backend == "voxtral-realtime":
        segments, detected = _voxtral_realtime_transcribe(spec, provider, input_path, language,
                                                          work_dir, force)
    else:
        raise STTError(f"неизвестный backend распознавания: '{spec.backend}'")

    if not segments:
        print("внимание: пустой транскрипт (нет аудиодорожки или речи)")
    _write_transcript(out_json, segments)
    if detected:
        lang_file.write_text(detected, encoding="utf-8")
    elif lang_file.exists():
        lang_file.unlink()
    return detected
