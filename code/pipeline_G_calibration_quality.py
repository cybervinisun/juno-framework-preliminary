"""
Version-G pipeline, part E: properly cross-validated calibration (without
reusing training data as calibration data) plus calibration-quality metrics
(Brier score, ECE, reliability diagram).

Explicit methodological correction relative to the earlier round: there, the
champions were calibrated via `CalibratedClassifierCV(FrozenEstimator(model),
method='sigmoid').fit(X_resampled, y_train_bin)`. scikit-learn's own docstring
warns that, with FrozenEstimator, "all provided data is used for calibration"
and that "the user has to take care manually that data for model fitting and
calibration are disjoint". Since X_resampled is EXACTLY the set used to train
the champion, that calibration was not leak-free: the sigmoid was fitted
against predictions of a model that had already seen -- and, in the XGBoost
case, perfectly memorised -- those same data.

Here the calibration uses cross-validation properly: for each of the 5
parent-child-grouped folds, a CLONE of the pipeline (carrying the winning
hyperparameters of Table 2) is refitted on the 4 training folds and the
sigmoid is fitted on the held-out calibration fold, never the one used to fit
that copy of the model. The 5 model+calibrator pairs are ensembled at
prediction time (scikit-learn's default ensemble=True behaviour).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.base import clone

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "version_G_outputs"
REPO_ROOT_FOR_CKPT = BASE_DIR.parent
FIG_DIR = OUT_DIR / "figures_G"
FIG_DIR.mkdir(exist_ok=True)

# Produced by pipeline_G.py into OUT_DIR; fall back to the copy deposited
# under models/ so this script also runs standalone from a fresh clone.
_ckpt_path = OUT_DIR / "checkpoint_post_svmsmote_G.pkl"
if not _ckpt_path.exists():
    _ckpt_path = REPO_ROOT_FOR_CKPT / "models" / "checkpoint_post_svmsmote_G.pkl"
ckpt = joblib.load(_ckpt_path)
X_train_final = ckpt["X_train_final"]
y_train_final = ckpt["y_train_final"]
X_test = ckpt["X_test"]
y_test = ckpt["y_test"]
tracking_table = ckpt["tracking_table"]

X_resampled = X_train_final.copy()
y_resampled = y_train_final.copy()

groups = tracking_table["mother_original_row_id"].copy()
mask = tracking_table["is_synthetic"] == False
groups.loc[mask] = tracking_table.loc[mask, "original_row_id"]
groups = groups.astype(int)

LABEL_MAP = {"Inactive": 0, "Active": 1}
y_test_bin = np.array([LABEL_MAP[c] for c in y_test["Activity"]], dtype=np.int64)
y_train_bin = np.array([LABEL_MAP[c] for c in y_resampled], dtype=np.int64)

hp = pd.read_csv(OUT_DIR / "table_champion_hyperparameters_G.csv").set_index("Model")


def fresh_pipeline(label: str) -> Pipeline:
    """Builds an UNFITTED pipeline carrying the winning hyperparameters
    (Table 2) -- required by the properly cross-validated calibration, which
    has to refit copies of the model on every fold."""
    row = hp.loc[label]
    if label == "MLP":
        return Pipeline(steps=[("NN", MLPClassifier(
            solver="lbfgs", max_iter=20000, random_state=23,
            activation=row["NN__activation"],
            alpha=float(row["NN__alpha"]),
            hidden_layer_sizes=int(row["NN__hidden_layer_sizes"]),
            learning_rate_init=float(row["NN__learning_rate_init"]),
        ))])
    if label == "XGBoost":
        return Pipeline(steps=[("xgb", XGBClassifier(
            random_state=0, booster="gbtree", objective="binary:logistic",
            colsample_bytree=float(row["xgb__colsample_bytree"]),
            gamma=float(row["xgb__gamma"]),
            learning_rate=float(row["xgb__learning_rate"]),
            max_depth=int(row["xgb__max_depth"]),
            max_leaves=int(row["xgb__max_leaves"]),
            min_child_weight=float(row["xgb__min_child_weight"]),
            n_estimators=int(row["xgb__n_estimators"]),
            reg_alpha=float(row["xgb__reg_alpha"]),
            reg_lambda=float(row["xgb__reg_lambda"]),
            subsample=float(row["xgb__subsample"]),
        ))])
    if label == "SVM":
        return Pipeline(steps=[("svm", svm.SVC(
            gamma=float(row["svm__gamma"]), C=float(row["svm__C"]),
            kernel=row["svm__kernel"], max_iter=-1, probability=False,
        ))])
    raise ValueError(label)


def _raw_score(fitted_pipe, X):
    """Continuous pre-calibration score: decision_function when available
    (SVM), otherwise predict_proba (MLP/XGBoost)."""
    step = fitted_pipe.steps[-1][1]
    if hasattr(step, "decision_function"):
        return fitted_pipe.decision_function(X)
    return fitted_pipe.predict_proba(X)[:, 1]


def leakfree_grouped_calibration(label: str, X_resampled, y_train_bin, groups, cv):
    """Properly cross-validated, parent-child-grouped Platt (sigmoid)
    calibration: on each fold a CLONE of the pipeline (already carrying the
    winning hyperparameters) is refitted on the 4 training folds, and the
    sigmoid (a 1-D logistic regression) is fitted ONLY on the held-out fold,
    never seen by that copy of the model. The 5 (model, calibrator) pairs are
    ensembled (mean of the probabilities) at prediction time -- the same logic
    as `CalibratedClassifierCV(cv=5, ensemble=True)`, implemented by hand to
    avoid `groups` metadata-routing problems in this scikit-learn version."""
    pairs = []
    for train_idx, calib_idx in cv.split(X_resampled, y_train_bin, groups):
        fold_model = clone(fresh_pipeline(label))
        fold_model.fit(X_resampled.iloc[train_idx], y_train_bin[train_idx])

        score_calib = _raw_score(fold_model, X_resampled.iloc[calib_idx]).reshape(-1, 1)
        sigmoid = LogisticRegression().fit(score_calib, y_train_bin[calib_idx])

        pairs.append((fold_model, sigmoid))
    return pairs


def leakfree_predict_proba(pairs, X):
    probs = []
    for fold_model, sigmoid in pairs:
        score = _raw_score(fold_model, X).reshape(-1, 1)
        probs.append(sigmoid.predict_proba(score)[:, 1])
    return np.mean(probs, axis=0)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Identical to the reference notebook's function, reused for fidelity --
    ECE with uniform bins over [0,1]."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask_bin = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask_bin.sum() == 0:
            continue
        mean_confidence = float(y_prob[mask_bin].mean())
        observed_frequency = float(y_true[mask_bin].mean())
        weight = float(mask_bin.sum() / n)
        ece += weight * abs(mean_confidence - observed_frequency)
        rows.append({"bin": f"[{lo:.1f},{hi:.1f}]", "n": int(mask_bin.sum()),
                      "mean_confidence": mean_confidence, "observed_frequency": observed_frequency})
    return float(ece), pd.DataFrame(rows)


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=21)

print("=" * 70)
print("Leak-free calibration (CalibratedClassifierCV, cv=5, ensemble=True)")
print("vs. the previous calibration (FrozenEstimator, same fit+calibration data)")
print("=" * 70)

rows = []
reliability_data = {}
old_new_compare = []

for label in ["MLP", "SVM", "XGBoost"]:
    print(f"\n--- {label} ---")
    calibration_pairs = leakfree_grouped_calibration(label, X_resampled, y_train_bin, groups, cv)

    y_score_leakfree = leakfree_predict_proba(calibration_pairs, X_test)
    auc_leakfree = roc_auc_score(y_test_bin, y_score_leakfree)
    brier_leakfree = brier_score_loss(y_test_bin, y_score_leakfree)
    ece_leakfree, ece_bins_leakfree = expected_calibration_error(y_test_bin, y_score_leakfree)

    previous_calibrator = joblib.load(OUT_DIR / f"final_model_{label}_calibrated_G.pkl")
    y_score_previous = previous_calibrator.predict_proba(X_test)[:, 1]
    auc_previous = roc_auc_score(y_test_bin, y_score_previous)
    brier_previous = brier_score_loss(y_test_bin, y_score_previous)
    ece_previous, _ = expected_calibration_error(y_test_bin, y_score_previous)

    print(f"  Leak-free (cv=5 ensemble):    AUC={auc_leakfree:.4f}  Brier={brier_leakfree:.4f}  ECE={ece_leakfree:.4f}")
    print(f"  Previous (FrozenEstimator):   AUC={auc_previous:.4f}  Brier={brier_previous:.4f}  ECE={ece_previous:.4f}")

    rows.append({
        "Model": label,
        "AUC_leakfree": auc_leakfree, "Brier_leakfree": brier_leakfree, "ECE_leakfree": ece_leakfree,
        "AUC_frozen_prefit_style": auc_previous, "Brier_frozen_prefit_style": brier_previous,
        "ECE_frozen_prefit_style": ece_previous,
    })
    reliability_data[label] = (y_score_leakfree, y_score_previous)

    joblib.dump(calibration_pairs, OUT_DIR / f"final_model_{label}_calibrated_leakfree_G.pkl")

calib_quality_df = pd.DataFrame(rows)
calib_quality_df.to_csv(OUT_DIR / "table_calibration_quality_G.csv", index=False)
print(f"\n[table saved] {OUT_DIR / 'table_calibration_quality_G.csv'}")

# Update Table 4 test-set AUC to the leak-free values (hard labels/other
# metrics unaffected -- unchanged from the champion's own .predict()).
test_df = pd.read_csv(OUT_DIR / "table_heldout_test_metrics_G.csv")
test_df = test_df.set_index("Model")
for label in ["MLP", "SVM", "XGBoost"]:
    test_df.loc[label, "AUC"] = calib_quality_df.set_index("Model").loc[label, "AUC_leakfree"]
test_df = test_df.reset_index()
test_df.to_csv(OUT_DIR / "table_heldout_test_metrics_G.csv", index=False)
print(f"[table updated] the AUC in {OUT_DIR / 'table_heldout_test_metrics_G.csv'} now uses the leak-free calibration")

# ====================================================================
# Fig. reliability diagram (calibration curve): leak-free vs. previous
# ====================================================================
MODEL_COLORS = {"MLP": "#d62728", "SVM": "#2ca02c", "XGBoost": "#1f77b4"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
for label, color in MODEL_COLORS.items():
    y_score_leakfree, y_score_previous = reliability_data[label]
    frac_pos_novo, mean_pred_novo = calibration_curve(y_test_bin, y_score_leakfree, n_bins=5, strategy="quantile")
    frac_pos_old, mean_pred_old = calibration_curve(y_test_bin, y_score_previous, n_bins=5, strategy="quantile")
    axes[0].plot(mean_pred_novo, frac_pos_novo, "o-", color=color, label=label)
    axes[1].plot(mean_pred_old, frac_pos_old, "o--", color=color, label=label)

for ax, title in zip(axes, ["Leak-free calibration (CalibratedClassifierCV, cv=5 ensemble)",
                             "Previous calibration (FrozenEstimator, non-disjoint fit/calibration data)"]):
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Mean predicted probability (quintile bins)")
    ax.set_ylabel("Observed frequency of Active")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

fig.suptitle("Reliability diagrams on the held-out test set (n = 96)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG14_reliability_diagrams.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'figG14_reliability_diagrams.png'}")

print()
print("=" * 70)
print("COMPLETE: leak-free calibration + quality metrics (Brier/ECE)")
print("=" * 70)
