# MLOps Sandbox Demo

A **compact, audit-ready MLOps showcase** that demonstrates:
- Containerised **FastAPI** inference service for a scikit-learn model
- **CI/CD** with linting, tests, type checks, and container build/scan
- **On‑prem (docker-compose)** and **cloud** (example: Cloud Run) deployment paths
- **Observability & health**: `/health` and `/metrics` (Prometheus)
- **Reproducibility & lineage**: fixed seeds, pinned dependencies, saved trained artifact
- **Robustness check**: simple perturbation test + optional **IBM ART** adversarial evaluation
- **Audit-ready docs**: model card, evaluation protocol, risk register

---

## Quickstart (local, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py             # trains a small classifier and saves artifacts/model.joblib
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs
```

## Docker (on‑prem style)

```bash
# Build & run the API
docker build -t edps-mlops-demo:latest .
docker run -p 8000:8000 edps-mlops-demo:latest
# Open http://127.0.0.1:8000/health and /metrics
```

Or with **docker-compose**:

```bash
docker compose up --build
```

## Cloud (example: Google Cloud Run)

```bash
# Assuming you have gcloud configured and a project selected
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/edps-mlops-demo
gcloud run deploy edps-mlops-demo --image gcr.io/$(gcloud config get-value project)/edps-mlops-demo --platform managed --allow-unauthenticated
```

(Alternatively, use AWS ECS/Fargate or Azure Container Apps — the container is portable.)

---

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `GET /metrics` → Prometheus metrics (latency, requests, etc.)
- `POST /predict` → Predict on one or more samples
- `POST /explain` → Return simple feature importances (for the demo model)

Try it in the interactive docs at `/docs` (Swagger UI).

---
### Input format for `/predict`
Each sample can be either:
- a **flattened list of 64 floats** (8×8 grayscale image flattened), **or**
- a **nested 8×8 list** of floats. (The API will auto-flatten it.)

Swagger UI shows both examples. Minimal JSON examples:

**Flattened 64:**
```json
{ "samples": [[0.0, 0.1, 0.2, /* ... 61 more ... */ 0.0]] }
```

**Nested 8×8:**
```json
{ "samples": [[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.1,0.2,0.1,0.0,0.0,0.0,0.0],
               [0.0,0.2,0.3,0.2,0.0,0.0,0.0,0.0],
               [0.0,0.1,0.2,0.1,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]] }
```

## CI/CD (GitHub Actions)

This repository includes `.github/workflows/ci.yml` that runs on every push/PR:
- Linting (`ruff`), typing (`mypy`), tests (`pytest`)
- Build the Docker image
- Security scan (Trivy) — optional if you add `TRIVY_IGNORE_UNFIXED` etc.
- (Optional) Push to GHCR if you add the `CR_PAT` secret

---


### Optional: install robustness extras

If you want to run the IBM ART adversarial demo, install the optional requirements **after** the main install:

```bash
pip install -r optional/requirements-robustness.txt
```

> If installation of ART fails on your platform/Python version, you can skip this step — the robustness script will automatically **skip** the ART attack and still run the noise-based checks.

## Robustness

Run a quick robustness check after training:

```bash
python robustness/robustness_eval.py
```

- Applies noise-based stress tests to probe sensitivity.
- If **IBM ART** is installed (`pip install adversarial-robustness-toolbox`), runs a small FGSM-style attack on a simple model for illustration.

---

## Audit-ready documentation

See `docs/`:
- `model_card.md` — what the model does, data, metrics, limitations
- `evaluation_protocol.md` — how we test/monitor (functional, stress, drift, misuse)
- `risk_register.md` — identified risks & mitigations (security, robustness, reliability)