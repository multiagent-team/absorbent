"""Точка входа командной строки: разбор аргументов и запуск конвейера."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from absorbent.config import ConfigError, load_config

# Расширения, обрабатываемые при передаче директории на вход.
MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".webm", ".avi", ".mov", ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
)


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="absorbent",
        description="Превращает запись лекции (видео/аудио) в структурированный конспект Markdown.",
    )
    parser.add_argument("input", help="путь к медиафайлу или директории с медиафайлами")
    parser.add_argument("--config", metavar="PATH", help="путь к YAML-конфигурации")
    parser.add_argument("--stt", metavar="NAME", help="модель распознавания речи (по умолчанию — из конфига)")
    parser.add_argument("--vision-model", metavar="NAME", help="модель описания кадров (по умолчанию — из конфига)")
    parser.add_argument("--llm-model", metavar="NAME", help="модель синтеза конспекта (по умолчанию — из конфига)")
    parser.add_argument(
        "--language",
        choices=("ru", "en", "auto"),
        default="auto",
        help="подсказка языка для распознавания речи: ru, en или auto "
             "(по умолчанию auto; ru/en форсируют весь файл как один язык)",
    )
    parser.add_argument(
        "--output-language",
        choices=("ru", "en"),
        default="ru",
        help="язык итогового конспекта и промптов описания: ru или en (по умолчанию ru)",
    )
    parser.add_argument("--fps", type=float, default=1.0, metavar="N", help="частота извлечения кадров (по умолчанию 1)")
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=10,
        metavar="N",
        help="порог pHash-расстояния для дедупликации кадров, 0-64 (по умолчанию 10)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="каталог результата (по умолчанию: <имя_файла>/ рядом с входным файлом)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="результат и сырой транскрипт рядом с входным файлом "
             "(<имя>.md, <имя>.json), кэш — в <каталог>/.absorbent/<имя>/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="игнорировать кэш промежуточных результатов и перезапустить все этапы",
    )
    parser.add_argument("--skip-stt", action="store_true", help="пропустить распознавание речи")
    parser.add_argument("--skip-frames", action="store_true", help="пропустить извлечение и дедупликацию кадров")
    parser.add_argument("--skip-describe", action="store_true", help="пропустить описание кадров")
    parser.add_argument("--skip-synthesize", action="store_true", help="пропустить синтез Markdown")
    parser.add_argument("--skip-render", action="store_true",
                        help="пропустить постпроцессинг: выравнивание таблиц, HTML, PDF")
    return parser


def collect_inputs(raw: str) -> list[Path]:
    """Развернуть аргумент input в список медиафайлов."""
    path = Path(raw)
    if not path.exists():
        raise ConfigError(f"входной путь не существует: {raw}")
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
    if not files:
        raise ConfigError(f"в директории {raw} не найдено медиафайлов с расширениями "
                          + ", ".join(sorted(MEDIA_EXTENSIONS)))
    return files


def main(argv: list[str] | None = None) -> int:
    """Точка входа; возвращает код завершения."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        # Ранняя проверка имён моделей: ошибка печатает допустимые имена категории.
        cfg.resolve("stt", args.stt)
        cfg.resolve("vision", args.vision_model)
        cfg.resolve("llm", args.llm_model)
        inputs = collect_inputs(args.input)
    except ConfigError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2

    # Импорт внутри функции: позволяет --help работать без полного конвейера.
    from absorbent import realtime
    from absorbent.pipeline import process_file

    # Локальный сервер vLLM поднимается на весь запуск, а не на каждый файл:
    # пакет из десятков лекций иначе платил бы за старт сервера каждый раз.
    server_holder: list = []
    exit_code = 0
    try:
        if not args.skip_stt:
            stt_provider, stt_spec = cfg.resolve("stt", args.stt)
            if stt_spec.backend == "voxtral-realtime":
                realtime.ensure_server(stt_provider, stt_spec.api_model, stt_spec.chunk_seconds,
                                       proc_holder=server_holder)
        for input_path in inputs:
            try:
                process_file(input_path, cfg, args, server_holder=server_holder)
            except Exception as exc:  # ошибка этапа не должна ронять обработку остальных файлов.
                print(f"ошибка обработки {input_path}: {exc}", file=sys.stderr)
                exit_code = 1
    finally:
        if server_holder:
            print("остановка сервера vLLM...")
            realtime.stop_server(server_holder[0])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
