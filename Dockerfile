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
RUN chown -R uvicorn:uvicorn /app
USER uvicorn

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
