# Образ Absorbent: Python 3.14 (Debian trixie), ffmpeg, pandoc+lualatex, LibreOffice.
#
# CUDA-рантайм в базе не нужен: колёса torch тянут библиотеки NVIDIA с собой,
# для GPU хватает хостового драйвера и `docker run --gpus all`.
# Debian trixie (а не bookworm) потому, что в bookworm pandoc 2.17, где нет
# `--embed-resources`, который использует render.render_html().

# ---------- сборка окружения ----------
FROM python:3.14-slim-trixie AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ARG GIT_REVISION
ENV ABSORBENT_BUILD_COMMIT=${GIT_REVISION}
COPY pyproject.toml uv.lock README.md LICENSE hatch_build.py ./
# Сначала только зависимости: слой кэшируется, пока не менялись uv.lock и pyproject.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------- рабочий образ ----------
FROM python:3.14-slim-trixie AS runtime

# Версия и ревизия подставляются на сборке релиза ключом `--build-arg`
# (см. AGENTS.md: метки OCI должны нести номер версии и git-хеш).
ARG VERSION=dev
ARG GIT_REVISION=unknown
LABEL org.opencontainers.image.title="absorbent" \
      org.opencontainers.image.description="Лекционное видео -> конспект Markdown" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_REVISION}" \
      org.opencontainers.image.source="https://github.com/multiagent-team/absorbent"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

# ffmpeg/ffprobe — кадры и аудио; poppler-utils — pdftoppm; pandoc + texlive —
# HTML/PDF; libreoffice-impress — не-PDF колоды; fonts-dejavu — кириллица в PDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      poppler-utils \
      pandoc \
      curl \
      fonts-dejavu \
      texlive-luatex \
      texlive-latex-recommended \
      texlive-latex-extra \
      texlive-fonts-recommended \
      texlive-lang-cyrillic \
      texlive-pictures \
      libreoffice-impress \
 && rm -rf /var/lib/apt/lists/*

# Чекпойнт Whisper turbo: URL и хеш взяты из whisper/__init__.py (_MODELS),
# хеш совпадает с именем каталога в URL и проверяется sha256sum.
ENV WHISPER_CKPT=/root/.cache/whisper/large-v3-turbo.pt
RUN mkdir -p /root/.cache/whisper \
 && curl -fL --retry 3 -o "$WHISPER_CKPT" \
      https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt \
 && echo "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a  $WHISPER_CKPT" | sha256sum -c -

# В рабочий слой идёт установленное окружение с кодом и сведениями о сборке.
COPY --from=builder /app/.venv /app/.venv
# Штатная конфигурация образа. Пользовательскую кладут томом поверх:
# -v ~/.config/absorbent:/root/.config/absorbent:ro
COPY config.yaml /root/.config/absorbent/config.yaml

WORKDIR /data
ENTRYPOINT ["absorbent"]
