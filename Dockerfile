FROM python:3.12-slim

# ─────────────────────────────
# Integración de uv optimizada
# ─────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/


ENV PATH="/src/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONPATH=/src

WORKDIR /src

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# ──────────────────────
# Copiar config del proyecto
# ──────────────────────
COPY pyproject.toml uv.lock ./

# Instalar dependencias SIN proyecto (solo lock)
RUN --mount=type=cache,target=/root/.cache/uv \
uv sync --frozen --no-install-project

# ─────────────────────────────
# Copiar código
# ─────────────────────────────
COPY . .

# Instalar proyecto en entorno virtual
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# ──────────────────────
# Comando principal
# ──────────────────────
CMD ["uv", "run", "-m", "main"]