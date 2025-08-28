"""Train a tiny classifier on sklearn's digits dataset and save artifacts/model.joblib."""
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import json
import pathlib
import random

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def main():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # normalise pixels 0..1
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto", n_jobs=None)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    joblib.dump(clf, ARTIFACTS_DIR / "model.joblib")
    meta = {"seed": SEED, "test_accuracy": acc}
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    # --- Save dataset-backed example payloads for Swagger examples
    # Pick one representative sample each for digits 0, 3, 8
    labels = [0, 3, 8]
    examples = {}
    for lbl in labels:
        idx = int(np.where(y == lbl)[0][0])      # first occurrence
        sample_flat = X[idx].tolist()            # 64-length, values in [0,1]
        sample_nested = [sample_flat[i*8:(i+1)*8] for i in range(8)]
        # Round for readability
        sample_flat_r = [round(float(v), 3) for v in sample_flat]
        sample_nested_r = [[round(float(v), 3) for v in row] for row in sample_nested]
        examples[f"digit{lbl}_flat"]   = {"samples": [sample_flat_r]}
        examples[f"digit{lbl}_nested"] = {"samples": [sample_nested_r]}

    (ARTIFACTS_DIR / "example_payloads.json").write_text(json.dumps(examples, indent=2))
    
    print(f"Saved model to {ARTIFACTS_DIR / 'model.joblib'} (test acc ~ {acc:.3f})")
    print(f"Saved seed and test accuracy to {ARTIFACTS_DIR / 'metadata.json'}")
    print(f"Saved example labels 0, 3 and 8 to {ARTIFACTS_DIR / 'example_payloads.json'}")

if __name__ == "__main__":
    main()
