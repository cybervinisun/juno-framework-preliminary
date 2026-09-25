"""
Version-G pipeline, part E: calibration scenarios for Model A (trained WITH
the SVMSMOTE synthetics, X_resampled) versus Model B (trained WITHOUT them,
X_orig only), each compared uncalibrated, calibrated on X_orig, and
calibrated on X_resampled -- the experimental design reported in Table 6 of
Article 1, applied to all three retained algorithms.

The calibration sets here are the whole partitions (224 or 334 instances),
which avoids the variance problem of calibrating on small held-out folds, at
the cost of reintroducing some overlap between fitting and calibration data
in some scenarios -- exactly the trade-off this study is designed to expose
rather than hide.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from xgboost import XGBClassifier

from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedGroupKFold
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

# Search spaces and protocol IDENTICAL to those in pipeline_G.py.
# Budget: that of the RETAINED champion -- n_iter=5 for all three algorithms.
# XGBoost uses 5, not the 15 that pipeline_G.py still carries, because the
# n_iter=15 champion was superseded (see rebuild_xgb_niter5_cascade.py).
PAIR_GRID = {
    "MLP": {
        "NN__hidden_layer_sizes": Integer(5, 15),
        "NN__alpha": Real(1e-5, 1.0005965763586375e-05, "log-uniform"),
        "NN__activation": Categorical(["tanh"]),
        "NN__learning_rate_init": Real(1e-7, 1e-6, "log-uniform"),
    },
    "XGBoost": {
        "xgb__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "xgb__n_estimators": Integer(50, 400),
        "xgb__max_depth": Integer(5, 8),
        "xgb__max_leaves": Integer(50, 100),
        "xgb__min_child_weight": Real(1e-1, 10.0, prior="log-uniform"),
        "xgb__subsample": Real(0.6, 0.9, prior="uniform"),
        "xgb__colsample_bytree": Real(0.4, 0.8, prior="uniform"),
        "xgb__gamma": Real(0.0, 5.0, prior="uniform"),
        "xgb__reg_alpha": Real(1e-8, 1.0, prior="log-uniform"),
        "xgb__reg_lambda": Real(1e-3, 5.0, prior="log-uniform"),
    },
    "SVM": {
        "svm__C": Real(0.5, 1, prior="log-uniform"),
        "svm__gamma": Real(0.01, 1, prior="log-uniform"),
        "svm__kernel": Categorical(["rbf"]),
    },
}
N_ITER_B = {"MLP": 5, "XGBoost": 5, "SVM": 5}
KAPPA_SCORER = make_scorer(cohen_kappa_score)

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "version_G_outputs"
FIG_DIR = OUT_DIR / "figures_G"
FIG_DIR.mkdir(exist_ok=True)

ckpt = joblib.load(OUT_DIR / "checkpoint_post_svmsmote_G.pkl")
X_train_final = ckpt["X_train_final"]
y_train_final = ckpt["y_train_final"]
X_test = ckpt["X_test"]
y_test = ckpt["y_test"]
tracking_table = ckpt["tracking_table"]

X_resampled = X_train_final.copy()
LABEL_MAP = {"Inactive": 0, "Active": 1}
y_train_bin = np.array([LABEL_MAP[c] for c in y_train_final], dtype=np.int64)
y_test_bin = np.array([LABEL_MAP[c] for c in y_test["Activity"]], dtype=np.int64)

is_orig = (tracking_table["is_synthetic"] == False).to_numpy()
X_orig = X_resampled.loc[is_orig].reset_index(drop=True)
y_orig_bin = y_train_bin[is_orig]
print(f"X_orig (without synthetics): {X_orig.shape}  |  X_resampled (with synthetics): {X_resampled.shape}")

groups_original = tracking_table.loc[is_orig, "original_row_id"].astype(int).to_numpy()

hp = pd.read_csv(OUT_DIR / "table_champion_hyperparameters_G.csv").set_index("Model")


def fresh_pipeline(label: str):
    row = hp.loc[label]
    if label == "MLP":
        return Pipeline(steps=[("NN", MLPClassifier(
            solver="lbfgs", max_iter=20000, random_state=23,
            activation=row["NN__activation"], alpha=float(row["NN__alpha"]),
            hidden_layer_sizes=int(row["NN__hidden_layer_sizes"]),
            learning_rate_init=float(row["NN__learning_rate_init"]),
        ))])
    if label == "XGBoost":
        return Pipeline(steps=[("xgb", XGBClassifier(
            random_state=0, booster="gbtree", objective="binary:logistic",
            colsample_bytree=float(row["xgb__colsample_bytree"]), gamma=float(row["xgb__gamma"]),
            learning_rate=float(row["xgb__learning_rate"]), max_depth=int(row["xgb__max_depth"]),
            max_leaves=int(row["xgb__max_leaves"]), min_child_weight=float(row["xgb__min_child_weight"]),
            n_estimators=int(row["xgb__n_estimators"]), reg_alpha=float(row["xgb__reg_alpha"]),
            reg_lambda=float(row["xgb__reg_lambda"]), subsample=float(row["xgb__subsample"]),
        ))])
    if label == "SVM":
        return Pipeline(steps=[("svm", svm.SVC(
            gamma=float(row["svm__gamma"]), C=float(row["svm__C"]),
            kernel=row["svm__kernel"], max_iter=-1, probability=False,
        ))])
    raise ValueError(label)


def fit_platt_calibration(fitted_model, X_c, y_c):
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(estimator=FrozenEstimator(fitted_model), method="sigmoid").fit(X_c, y_c)
    except ImportError:
        return CalibratedClassifierCV(estimator=fitted_model, method="sigmoid", cv="prefit").fit(X_c, y_c)


def native_score(fitted_model, X):
    """Native, uncalibrated score. MLP/XGBoost: predict_proba (their own
    logistic link). SVM (probability=False): min-max normalised
    decision_function -- that is a score, NOT a probability, and is used
    only to complete the uncalibrated scenario, clearly flagged as such."""
    step = fitted_model.steps[-1][1]
    if hasattr(step, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    raw = fitted_model.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min())


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask_bin = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask_bin.sum() == 0:
            continue
        ece += (mask_bin.sum() / n) * abs(y_prob[mask_bin].mean() - y_true[mask_bin].mean())
    return float(ece)


all_rows = []
reliability_by_model = {}

for label in ["MLP", "SVM", "XGBoost"]:
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    # Model A: already trained (version-G champion) on X_resampled.
    model_a = joblib.load(OUT_DIR / f"final_model_{label}_G.pkl")

    # Model B: RE-OPTIMISED in its own regime -- same Bayesian search, same
    # space and same budget as the retained champion, but over X_orig (without
    # the SVMSMOTE synthetics). Without synthetics each ligand is its own group,
    # so StratifiedGroupKFold reduces to plain stratified CV.
    search_b = BayesSearchCV(
        estimator=fresh_pipeline(label), search_spaces=PAIR_GRID[label],
        n_iter=N_ITER_B[label], n_jobs=-1,
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=21),
        scoring=KAPPA_SCORER, error_score="raise", random_state=21, refit=True,
    ).fit(X_orig, y_orig_bin, groups=groups_original)
    model_b = search_b.best_estimator_
    print(f"  Model B re-optimised on X_orig: CV kappa {search_b.best_score_:.4f} | "
          f"{search_b.best_params_}")

    scenarios = {}
    is_svm = label == "SVM"

    # A1 / B1: uncalibrated (native score)
    scenarios["A1"] = {"model": "A", "label": "A1 (uncalibrated)",
                        "scores": native_score(model_a, X_test), "is_probability": not is_svm}
    scenarios["B1"] = {"model": "B", "label": "B1 (uncalibrated)",
                        "scores": native_score(model_b, X_test), "is_probability": not is_svm}

    # A2 / A3: model A calibrated on X_orig / X_resampled
    scenarios["A2"] = {"model": "A", "label": "A2 (calibrated on original)",
                        "scores": fit_platt_calibration(model_a, X_orig, y_orig_bin).predict_proba(X_test)[:, 1], "is_probability": True}
    scenarios["A3"] = {"model": "A", "label": "A3 (calibrated on balanced)",
                        "scores": fit_platt_calibration(model_a, X_resampled, y_train_bin).predict_proba(X_test)[:, 1], "is_probability": True}

    # B2 / B3: model B calibrated on X_resampled / X_orig
    scenarios["B2"] = {"model": "B", "label": "B2 (calibrated on balanced)",
                        "scores": fit_platt_calibration(model_b, X_resampled, y_train_bin).predict_proba(X_test)[:, 1], "is_probability": True}
    scenarios["B3"] = {"model": "B", "label": "B3 (calibrated on original)",
                        "scores": fit_platt_calibration(model_b, X_orig, y_orig_bin).predict_proba(X_test)[:, 1], "is_probability": True}

    for sid, item in scenarios.items():
        scores = np.asarray(item["scores"], dtype=float)
        row = {
            "Algorithm": label, "Scenario": sid, "Description": item["label"],
            "AUC": roc_auc_score(y_test_bin, scores),
            "Brier": brier_score_loss(y_test_bin, scores) if item["is_probability"] else np.nan,
            "ECE": expected_calibration_error(y_test_bin, scores) if item["is_probability"] else np.nan,
            "LogLoss": log_loss(y_test_bin, np.clip(scores, 1e-7, 1 - 1e-7)) if item["is_probability"] else np.nan,
            "is_probability": item["is_probability"],
        }
        all_rows.append(row)
        flag = "" if item["is_probability"] else "  [decision score, not a probability]"
        print(f"  {sid} {item['label']:<32} AUC={row['AUC']:.4f} Brier={row['Brier']:.4f} "
              f"ECE={row['ECE']:.4f} LogLoss={row['LogLoss']:.4f}{flag}")

    reliability_by_model[label] = scenarios

results_df = pd.DataFrame(all_rows)
results_df.to_csv(OUT_DIR / "table_calibration_scenarios_G.csv", index=False)
print(f"\n[table saved] {OUT_DIR / 'table_calibration_scenarios_G.csv'}")

# ====================================================================
# Figures: reliability diagrams, Model A (with synthetics) vs Model B
# (without synthetics), one figure per algorithm with two panels each.
# ====================================================================
COLORS = {"A1": "#d62728", "A2": "#1f77b4", "A3": "#ff7f0e",
          "B1": "#2ca02c", "B2": "#9467bd", "B3": "#8c564b"}

for label, scenarios in reliability_by_model.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    for ax, model_id, title in zip(axes, ["A", "B"],
                                    [f"{label}: trained WITH synthetics (Model A)",
                                     f"{label}: trained WITHOUT synthetics (Model B)"]):
        for sid, item in scenarios.items():
            if item["model"] != model_id or not item["is_probability"]:
                continue
            scores = np.asarray(item["scores"], dtype=float)
            frac, pred = calibration_curve(y_test_bin, scores, n_bins=5, strategy="quantile")
            ax.plot(pred, frac, "o-", label=item["label"], color=COLORS[sid], linewidth=2)
        ax.plot([0, 1], [0, 1], ":", color="grey", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency of Active")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle(f"Calibration scenarios on the held-out test set (n = 96): {label}", fontsize=12)
    fig.tight_layout()
    fname = f"figG20_calibration_scenarios_{label.lower()}.png"
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / fname}")

print()
print("=" * 70)
print("COMPLETE: Model A/B calibration scenarios (3 algorithms)")
print("=" * 70)
