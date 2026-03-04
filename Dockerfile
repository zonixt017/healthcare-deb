# ─────────────────────────────────────────────────────────────────────────────
# Healthcare Assistant Chatbot — Docker Image
# Optimized for HuggingFace Spaces and cloud deployment
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="zonixt017"
LABEL description="Healthcare Assistant Chatbot with RAG"
LABEL version="2.1"

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Create runtime directories ────────────────────────────────────────────────
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers vectorstore models \
    && sed -i 's/\r$//' /app/start.sh \
    && chmod +x /app/start.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

# ── Expose common Streamlit ports ─────────────────────────────────────────────
EXPOSE 7860 8501

# ── Health check (uses PORT if set) ───────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail "http://localhost:${PORT:-7860}/_stcore/health" || exit 1

# ── Run Streamlit ─────────────────────────────────────────────────────────────
CMD ["./start.sh"]
