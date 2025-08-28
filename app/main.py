from fastapi import FastAPI, HTTPException, Body
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
from app.schemas import PredictRequest, PredictResponse, ExplainResponse
from app.model import predict, explain_global_importance
import numpy as np
from time import time

app = FastAPI(title="EDPS MLOps Sandbox Demo")

REQUESTS = Counter("requests_total", "Total requests", ["endpoint"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

_EX_8x8_DIGIT0 = {"samples": [[[0,1,1,1,1,1,1,0],
                               [0,1,0,0,0,0,1,0],
                               [0,1,0,0,0,0,1,0],
                               [0,1,0,0,0,0,1,0],
                               [0,1,0,0,0,0,1,0],
                               [0,1,0,0,0,0,1,0],
                               [0,1,1,1,1,1,1,0],
                               [0,0,0,0,0,0,0,0]]]}
_EX_FLAT_DIGIT0 = {"samples": [[
  0,1,1,1,1,1,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0,
  0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,1,1,1,1,1,0, 0,0,0,0,0,0,0,0
]]}


@app.get("/health")
def health():
    REQUESTS.labels(endpoint="/health").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    req: PredictRequest = Body(
        ...,
        example=_EX_8x8_DIGIT0,
        examples={
            "flattened_64": {"summary": "Flattened 64-length vector", "value": _EX_FLAT_DIGIT0},
            "nested_8x8":  {"summary": "Nested 8×8 array (auto-flattened)", "value": _EX_8x8_DIGIT0},
        },
    )
):
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
