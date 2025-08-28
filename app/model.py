import joblib
import pathlib
from typing import Tuple
import numpy as np

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
META_PATH = ARTIFACTS_DIR / "metadata.json"

_model = None
_meta = None

def load_model() -> Tuple[object, dict]:
    global _model, _meta
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run scripts/train.py first.")
        _model = joblib.load(MODEL_PATH)
    if _meta is None and META_PATH.exists():
        import json
        _meta = json.loads(META_PATH.read_text())
    return _model, (_meta or {})

def predict(X: np.ndarray) -> np.ndarray:
    model, _ = load_model()
    if X.ndim != 2 or X.shape[1] != 64:
        raise ValueError(f"Expected samples with 64 features (8x8 flattened). Got shape {X.shape}.")
    preds = model.predict(X)
    return preds

def explain_global_importance() -> np.ndarray:
    model, _ = load_model()
    # Simple mean absolute coefficient magnitudes for linear models; fall back to zeros otherwise
    if hasattr(model, "coef_"):
        import numpy as np
        return np.mean(np.abs(model.coef_), axis=0)
    return np.array([])
