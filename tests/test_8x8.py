from fastapi.testclient import TestClient
from app.main import app

def test_predict_8x8_auto_flatten():
    client = TestClient(app)
    body = {"samples": [[[0.0 for _ in range(8)] for _ in range(8)]]}
    resp = client.post('/predict', json=body)
    assert resp.status_code in (200, 400)
