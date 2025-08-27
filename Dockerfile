# syntax=docker/dockerfile:1

# ===== Builder (install dependencies) =====
FROM python:3.11-slim AS builder
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

# ===== Runtime =====
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Create non-root user
RUN useradd -m uvicorn
COPY --from=builder /install /usr/local
COPY . .
# **Train the demo model inside the image** so the API is ready
RUN python scripts/train.py
RUN chown -R uvicorn:uvicorn /app
USER uvicorn

EXPOSE 8000
# Respect Cloud Run's $PORT (falls back to 8000 locally)
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
