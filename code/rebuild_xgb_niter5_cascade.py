"""
Recomputes EVERYTHING that depends on the XGBoost champion when switching
from n_iter=15 to n_iter=5 (the marginal gain of the larger budget is small
next to the risk of the Bayesian search overfitting the CV partition).

Produces: the full test metrics (Table 5), feature importance (Table 7), and
the NEW test-set misclassifications feeding the Tanimoto/MCS reassessment.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss, cohen_kappa_score, confusion_matrix, log_loss, roc_auc_score,
)

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "version_G_outputs"

ckpt = joblib.load(OUT_DIR / "checkpoint_post_svmsmote_G.pkl")
X_train_final = ckpt["X_train_final"]
y_train_final = ckpt["y_train_final"]
X_test = ckpt["X_test"]
y_test = ckpt["y_test"]
tracking_table = ckpt["tracking_table"]

X_resampled = X_train_final.copy()
LABEL_MAP = {"Inactive": 0, "Active": 1}
y_test_bin = np.array([LABEL_MAP[c] for c in y_test["Activity"]], dtype=np.int64)
y_train_bin = np.array([LABEL_MAP[c] for c in y_train_final], dtype=np.int64)

is_orig = (tracking_table["is_synthetic"] == False).to_numpy()
X_orig = X_resampled.loc[is_orig].reset_index(drop=True)
y_orig_bin = y_train_bin[is_orig]

xgb_champion = joblib.load(OUT_DIR / "final_model_XGBoost_n_iter5_G.pkl")

print("=" * 70)
print("1. Full test metrics (XGBoost, n_iter=5)")
print("=" * 70)

y_test_pred = xgb_champion.predict(X_test)
y_train_pred = xgb_champion.predict(X_resampled)
tn, fp, fn, tp = confusion_matrix(y_test_bin, y_test_pred).ravel()
acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp) if (tp + fp) else 0
sens = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
kappa_test = cohen_kappa_score(y_test_bin, y_test_pred)
kappa_train = cohen_kappa_score(y_train_bin, y_train_pred)

try:
    from sklearn.frozen import FrozenEstimator
    calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(xgb_champion), method="sigmoid").fit(X_orig, y_orig_bin)
except ImportError:
    calibrator = CalibratedClassifierCV(estimator=xgb_champion, method="sigmoid", cv="prefit").fit(X_orig, y_orig_bin)

y_test_score = calibrator.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test_bin, y_test_score)

print(f"Train kappa: {kappa_train:.4f}")
print(f"Test: Acc={acc:.4f} Error={1-acc:.4f} Prec={prec:.4f} Sens={sens:.4f} F1={f1:.4f} "
      f"Kappa={kappa_test:.4f} AUC={auc_test:.4f} TP={tp} TN={tn} FP={fp} FN={fn}")

pd.DataFrame([{
    "Model": "XGBoost", "n_iter": 5, "Accuracy": acc, "Error": 1 - acc, "Precision": prec,
    "Sensitivity": sens, "F1": f1, "Kappa_train": kappa_train, "Kappa_test": kappa_test,
    "AUC": auc_test, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
}]).to_csv(OUT_DIR / "table_xgb_niter5_final_metrics_G.csv", index=False)

print("\n" + "=" * 70)
print("2. Feature importance (gain/cover), XGBoost n_iter=5 champion")
print("=" * 70)

xgb_model = xgb_champion.named_steps["xgb"]
booster = xgb_model.get_booster()
importance_types = ["gain", "total_gain", "cover", "total_cover"]
importance_data = {imp: booster.get_score(importance_type=imp) for imp in importance_types}
importance_df = pd.DataFrame.from_dict(importance_data)
importance_df.index.name = "Feature"
importance_df = importance_df.reset_index()
importance_df[importance_types] = importance_df[importance_types].fillna(0)
importance_df = importance_df.sort_values(by="gain", ascending=False).reset_index(drop=True)

all_features = pd.DataFrame({"Feature": X_resampled.columns})
importance_full = all_features.merge(importance_df, on="Feature", how="left").fillna(0)
importance_full = importance_full.sort_values(by="gain", ascending=False).reset_index(drop=True)

top20_df = importance_df.sort_values("total_gain", ascending=False).head(20)
top20_df_by_gain = importance_df.head(20)
top20_path = OUT_DIR / "table_feature_importance_top20_niter5_G.csv"
full_path = OUT_DIR / "table_feature_importance_all57_niter5_G.csv"
top20_df_by_gain.to_csv(top20_path, index=False)
importance_full.to_csv(full_path, index=False)
print(top20_df_by_gain.head(10).to_string(index=False))
print(f"\n[tables saved] {top20_path}, {full_path}")

print("\n" + "=" * 70)
print("3. Identify the NEW test-set errors (XGBoost n_iter=5) for the Tanimoto analysis")
print("=" * 70)
y_pred_series = pd.Series(y_test_pred, index=X_test.index)
fp_idx = y_pred_series[(y_pred_series == 1) & (y_test_bin == 0)].index.tolist()
fn_idx = y_pred_series[(y_pred_series == 0) & (y_test_bin == 1)].index.tolist()
print(f"XGBoost (n_iter=5): {len(fp_idx)} false positives {fp_idx}, {len(fn_idx)} false negatives {fn_idx}")
pd.DataFrame({"index": fp_idx + fn_idx, "type": ["FP"] * len(fp_idx) + ["FN"] * len(fn_idx)}).to_csv(
    OUT_DIR / "table_xgb_niter5_test_errors_G.csv", index=False)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)
