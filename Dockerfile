# ── Stage 1: build dependencies ──────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# Install project dependencies into a virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ── Stage 2: runtime ────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code and data (API + dashboard only — no dbt/airflow/scripts)
COPY src/ ./src/
COPY data/ ./data/

# Ensure data directory and config dirs are writable for runtime outputs
RUN chown -R appuser:appuser /app/data && \
    mkdir -p /app/.streamlit /app/.config/matplotlib && \
    chown -R appuser:appuser /app/.streamlit /app/.config

# Switch to non-root user
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT','8000') + '/health')" || exit 1

# Render injects PORT; default to 8000 for local dev
CMD ["sh", "-c", "uvicorn credit_domino.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
