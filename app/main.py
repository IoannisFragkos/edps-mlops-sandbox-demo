from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
from app.schemas import PredictRequest, PredictResponse, ExplainResponse
from app.model import predict, explain_global_importance
import numpy as np
from time import time

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

@app.post("/predict", response_model=PredictResponse)
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
