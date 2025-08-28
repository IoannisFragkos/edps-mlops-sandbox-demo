from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
from app.schemas import PredictRequest, PredictResponse, ExplainResponse
from app.model import predict, explain_global_importance
from time import time
import numpy as np
import json
import pathlib

app = FastAPI(title="EDPS MLOps Sandbox Demo")

REQUESTS = Counter("requests_total", "Total requests", ["endpoint"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])


@app.get("/health")
def health():
    REQUESTS.labels(endpoint="/health").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

# Load examples for Swagger from artifacts (created by scripts/train.py)
def _load_examples():
    artifacts = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "example_payloads.json"
    if artifacts.exists():
        data = json.loads(artifacts.read_text())
        # FastAPI expects {"name": {"summary": "...", "value": {...}}}
        return {k: {"summary": k.replace("_", " "), "value": v} for k, v in data.items()}
    # Fallbacks if file missing
    return {
        "flattened_64": {"summary": "flattened 64 (fallback)", "value": {"samples": [[0.0]*64]}},
        "nested_8x8":   {"summary": "nested 8×8 (fallback)",   "value": {"samples": [[[0.0]*8 for _ in range(8)]]}},
    }

_DOC_EXAMPLES = _load_examples()

# Choose a sensible default example (prefer the nested digit-0 if present)
if "digit0_nested" in _DOC_EXAMPLES:
    _DEFAULT_EXAMPLE = _DOC_EXAMPLES["digit0_nested"]["value"]
else:
    # first available example's value
    _DEFAULT_EXAMPLE = next(iter(_DOC_EXAMPLES.values()))["value"]

@app.post("/predict", response_model=PredictResponse, 
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    # this creates the Examples dropdown in Swagger
                    "examples": _DOC_EXAMPLES
                }
            }
        }
    },)
def predict_endpoint(req: PredictRequest):
    start = time()
    REQUESTS.labels(endpoint="/predict").inc()
    try:
        X = np.array(req.samples, dtype=float)
        preds = predict(X).tolist()
        return PredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        LATENCY.labels(endpoint="/predict").observe(time() - start)

@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint():
    REQUESTS.labels(endpoint="/explain").inc()
    importances = explain_global_importance().tolist()
    return ExplainResponse(importances=importances)
