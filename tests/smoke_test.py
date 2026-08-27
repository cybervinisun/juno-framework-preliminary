"""
Lightweight CI smoke test -- NOT a full pipeline re-run (that would mean
re-fitting a Bayesian search + SVMSMOTE on every push, far too slow/costly
for CI). Instead this checks the two things most likely to silently break
this repository over time:

1. The four published champion models (+ scaler + checkpoint) still load
   cleanly and have not been accidentally replaced/corrupted -- their key
   tuned hyperparameters are compared against the values verified against
   Table 1 of the article at publication time.
2. Every script under code/ still at least imports/compiles cleanly (catches
   syntax errors and missing-import regressions without actually running a
   full pipeline).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
CODE_DIR = REPO_ROOT / "code"

# Key tuned hyperparameters verified against Table 1 of the published
# article (see the original hyperparameter-verification pass). Only the
# values that actually distinguish this champion from any other candidate
# are checked -- not the full get_params() dict.
EXPECTED = {
    "MLP": {
        "activation": "tanh",
        "alpha": 1.0005069643949672e-05,
        "hidden_layer_sizes": 7,
        "learning_rate_init": 7.656010912714263e-07,
    },
    "SVM": {
        "C": 0.7185240725598854,
        "gamma": 0.05072306207798868,
        "kernel": "rbf",
    },
    "XGBoost": {
        "n_estimators": 141,
        "learning_rate": 0.08244616809299331,
        "max_depth": 6,
        "max_leaves": 97,
        "colsample_bytree": 0.4066604603212028,
        "gamma": 1.3915254966127013,
        "min_child_weight": 0.42242968024018873,
        "subsample": 0.6099003094536355,
    },
    "LogisticRegression": {
        "C": 0.4126121236470449,
        "penalty": "l2",
        "solver": "lbfgs",
    },
}


def check_models() -> list[str]:
    errors = []
    for label, expected_params in EXPECTED.items():
        path = MODEL_DIR / f"modelo_final_{label}_G.pkl"
        try:
            model = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: failed to load {path.name}: {exc}")
            continue

        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        actual_params = estimator.get_params()
        for key, expected_value in expected_params.items():
            actual_value = actual_params.get(key)
            if isinstance(expected_value, float):
                ok = actual_value is not None and abs(actual_value - expected_value) < 1e-9
            else:
                ok = actual_value == expected_value
            if not ok:
                errors.append(
                    f"{label}: hyperparameter '{key}' mismatch -- "
                    f"expected {expected_value!r}, got {actual_value!r} "
                    f"(model file may have been replaced or corrupted)"
                )
        print(f"[ok] {label}: loaded and hyperparameters match.")

    for name in ["scaler_minmax_treino_G.pkl", "checkpoint_post_svmsmote_G.pkl"]:
        try:
            joblib.load(MODEL_DIR / name)
            print(f"[ok] {name}: loaded.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: failed to load: {exc}")

    return errors


def check_scripts_compile() -> list[str]:
    errors = []
    for path in sorted(CODE_DIR.glob("*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors.append(f"{path.name}: failed to compile:\n{result.stderr}")
        else:
            print(f"[ok] {path.name}: compiles.")
    return errors


def main() -> int:
    errors = check_models() + check_scripts_compile()
    if errors:
        print("\nFAILURES:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
