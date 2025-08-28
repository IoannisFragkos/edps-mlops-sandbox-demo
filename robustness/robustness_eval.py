"""Lightweight robustness checks.
- Adds Gaussian noise to inputs to measure sensitivity.
- If IBM ART is installed, runs a tiny FGSM-style attack for illustration.
"""
import numpy as np
import joblib
import pathlib

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
META_PATH = ARTIFACTS_DIR / "metadata.json"

def noise_test():
    from sklearn.datasets import load_digits
    from sklearn.metrics import accuracy_score
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    # Simple holdout
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X_te = X[1400:]
    y_te = y[1400:]

    clf = joblib.load(MODEL_PATH)
    base_acc = accuracy_score(y_te, clf.predict(X_te))

    # Additive Gaussian noise
    X_noisy = X_te + np.random.normal(0, 0.1, X_te.shape)
    X_noisy = np.clip(X_noisy, 0.0, 1.0)
    noisy_acc = accuracy_score(y_te, clf.predict(X_noisy))
    print(f"Base acc: {base_acc:.3f}, Noisy acc: {noisy_acc:.3f}")

def art_demo():
    try:
        from art.attacks.evasion import FastGradientMethod
        from art.estimators.classification import SklearnClassifier
        from sklearn.datasets import load_digits
        from sklearn.metrics import accuracy_score
    except Exception:
        print("IBM ART not installed or import failed; skipping ART demo.")
        return

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X_te = X[1400:]
    y_te = y[1400:]

    clf = joblib.load(MODEL_PATH)
    art_clf = SklearnClassifier(model=clf, clip_values=(0.0, 1.0))

    base_acc = accuracy_score(y_te, clf.predict(X_te))
    attack = FastGradientMethod(estimator=art_clf, eps=0.2)
    X_adv = attack.generate(x=X_te)
    adv_acc = accuracy_score(y_te, clf.predict(X_adv))
    print(f"Base acc: {base_acc:.3f}, Adversarial acc: {adv_acc:.3f}")

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("Model not found; run `python scripts/train.py` first.")
    else:
        noise_test()
        art_demo()
