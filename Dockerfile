# Two stages: build the virtualenv with uv, then copy it into a small image that runs as a normal user.
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra postgres --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra postgres --no-editable

FROM python:3.11-slim
RUN useradd --create-home --uid 10001 cardpilot
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY data/cards ./data/cards
COPY web ./web
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CARDPILOT_CARDS_DIR=/app/data/cards \
    CARDPILOT_WEB_DIR=/app/web \
    CARDPILOT_INDEX_DIR=/app/data/index
RUN mkdir -p /app/data/index && chown cardpilot /app/data/index
USER cardpilot
# The SQLite index is built into the image, so the first request is fast.
RUN cardpilot-ingest
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "cardpilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
