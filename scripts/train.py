"""Train a tiny classifier on sklearn's digits dataset and save artifacts/model.joblib."""
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib, json, pathlib, random

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
    print(f"Saved model to {ARTIFACTS_DIR / 'model.joblib'} (test acc ~ {acc:.3f})")

if __name__ == "__main__":
    main()
