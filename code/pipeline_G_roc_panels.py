"""Held-out ROC curves for the three retained champions, in two panels.

Panel (a): the uncalibrated scores that underlie the final held-out evaluation
           (``tab4_heldout_test_metrics_G.csv``; Table 5 of the article).
Panel (b): the same champions after Platt calibration fitted on the 224
           original, non-synthetic training ligands (Section 3.6 of the article).

Platt scaling is monotonic, so the two panels coincide exactly for the SVM and
XGBoost champions; only the MLP differs, and only because the saturated sigmoid
collapses distinct scores into ties.

Outputs
-------
results/fig7_roc_curve_points_G.csv   fpr/tpr for both panels
figures_G/figG4_roc_curves_wide.png

Note on loading the XGBoost champion
------------------------------------
``models/final_model_XGBoost_G.pkl`` stores the Booster as the byte buffer
produced by ``XGBoosterSerializeToBuffer``, which is not portable across
XGBoost builds: plain ``joblib.load`` raises "input stream corrupted" on a
different build even at the same version.  ``load_booster`` below works around
that by extracting the ``Model`` sub-document from the buffer and loading it
through the model reader.  ``models/final_model_XGBoost_G.ubj`` is the same
champion in the portable UBJSON format and needs no workaround -- prefer it.
"""
import ctypes
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import xgboost.core as xgb_core
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost.core import _LIB, _check_call

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures_G"

MODEL_COLORS = {"MLP": "#d62728", "SVM": "#2ca02c", "XGBoost": "#1f77b4"}
LABEL_MAP = {"Inactive": 0, "Active": 1}


def load_booster_from_pickle(path):
    """Recover the champion Booster from a SerializeToBuffer pickle."""
    original = xgb_core.Booster.__setstate__
    captured = {}

    def _capture(self, state):
        captured["raw"] = state.get("handle")
        self.__dict__.update({k: v for k, v in state.items() if k != "handle"})

    xgb_core.Booster.__setstate__ = _capture
    try:
        pipeline = joblib.load(path)
    finally:
        xgb_core.Booster.__setstate__ = original

    raw = bytes(captured["raw"])
    start = raw.find(b"Model") + len(b"Model")          # value of the "Model" key
    sub = bytearray(raw[start:len(raw) - 1])            # drop the outer closing brace
    booster = xgb.Booster()
    ptr = (ctypes.c_char * len(sub)).from_buffer(sub)
    _check_call(_LIB.XGBoosterLoadModelFromBuffer(booster.handle, ptr,
                                                  ctypes.c_uint64(len(sub))))
    pipeline[-1]._Booster = booster
    return pipeline


def load_champion(name):
    ubj = MODEL_DIR / "final_model_XGBoost_G.ubj"
    if name == "XGBoost" and ubj.exists():
        pipeline = load_booster_from_pickle(MODEL_DIR / "final_model_XGBoost_G.pkl")
        booster = xgb.Booster()
        blob = bytearray(ubj.read_bytes())
        ptr = (ctypes.c_char * len(blob)).from_buffer(blob)
        _check_call(_LIB.XGBoosterLoadModelFromBuffer(booster.handle, ptr,
                                                      ctypes.c_uint64(len(blob))))
        pipeline[-1]._Booster = booster
        return pipeline
    if name == "XGBoost":
        return load_booster_from_pickle(MODEL_DIR / "final_model_XGBoost_G.pkl")
    return joblib.load(MODEL_DIR / f"final_model_{name}_G.pkl")


def uncalibrated_score(pipeline, X):
    """The score the champion's own decision rule thresholds."""
    if hasattr(pipeline[-1], "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(pipeline[:-1].transform(X)
                                      if len(pipeline) > 1 else X)


def platt_calibrated_score(pipeline, X, X_cal, y_cal):
    """Platt calibration fitted on the 224 original, non-synthetic ligands."""
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(pipeline),
                                            method="sigmoid")
    except ImportError:                                  # scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(estimator=pipeline, method="sigmoid",
                                            cv="prefit")
    return calibrated.fit(X_cal, y_cal).predict_proba(X)[:, 1]


def main():
    checkpoint = joblib.load(MODEL_DIR / "checkpoint_post_svmsmote_G.pkl")
    X_train = checkpoint["X_train_final"]
    X_test = checkpoint["X_test"]
    y_train = np.array([LABEL_MAP[c] for c in checkpoint["y_train_final"]])
    y_test = np.array([LABEL_MAP[c] for c in checkpoint["y_test"].iloc[:, 0]])
    is_original = (checkpoint["tracking_table"]["is_synthetic"] == False).to_numpy()
    X_cal = X_train.loc[is_original].reset_index(drop=True)
    y_cal = y_train[is_original]

    scores, rows = {}, []
    for name in MODEL_COLORS:
        pipeline = load_champion(name)
        scores[name] = {
            "uncalibrated": uncalibrated_score(pipeline, X_test),
            "platt_calibrated": platt_calibrated_score(pipeline, X_test, X_cal, y_cal),
        }
        for panel, score in scores[name].items():
            fpr, tpr, _ = roc_curve(y_test, score)
            rows += [{"Model": name, "panel": panel, "fpr": a, "tpr": b}
                     for a, b in zip(fpr, tpr)]
            print(f"{name:<8} {panel:<18} AUC = {roc_auc_score(y_test, score):.6f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "fig7_roc_curve_points_G.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.1), dpi=150)
    panels = [("(a) Final held-out evaluation\n(uncalibrated scores)", "uncalibrated"),
              ("(b) After Platt calibration\n(Platt-calibrated scores)", "platt_calibrated")]
    for ax, (title, key) in zip(axes, panels):
        for name, colour in MODEL_COLORS.items():
            score = scores[name][key]
            fpr, tpr, _ = roc_curve(y_test, score)
            ax.plot(fpr, tpr, color=colour, linewidth=2.6,
                    label=f"{name} (AUC = {roc_auc_score(y_test, score):.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.4)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel("False positive rate", labelpad=10)
        ax.tick_params(labelsize=17)
        ax.legend(loc="lower right", frameon=False, fontsize=17)
    axes[0].set_ylabel("True positive rate", labelpad=10)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figG4_roc_curves_wide.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
