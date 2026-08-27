"""
Recalcula TUDO que depende do campeao XGBoost, trocando de n_iter=15
para n_iter=5 (decisao do usuario: ganho marginal pequeno frente ao
risco de overfitting do processo de busca bayesiana).

Gera: metricas completas de teste (Tabela 4), importancia de features
(Tabela 5/Fig22), cenarios de calibracao (Modelo A para XGBoost), e
identifica os NOVOS erros de classificacao no teste para a reavaliacao
Tanimoto/MCS.
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
OUT_DIR = BASE_DIR / "versao_G_outputs"

ckpt = joblib.load(OUT_DIR / "checkpoint_post_svmsmote_G.pkl")
X_train_final = ckpt["X_train_final"]
y_train_final = ckpt["y_train_final"]
X_test = ckpt["X_test"]
y_test = ckpt["y_test"]
tracking_table = ckpt["tracking_table"]

X_resampled = X_train_final.copy()
mapeamento = {"Inativo": 0, "Ativo": 1}
y_test_bin = np.array([mapeamento[c] for c in y_test["Atividade"]], dtype=np.int64)
y_train_bin = np.array([mapeamento[c] for c in y_train_final], dtype=np.int64)

is_orig = (tracking_table["is_synthetic"] == False).to_numpy()
X_orig = X_resampled.loc[is_orig].reset_index(drop=True)
y_orig_bin = y_train_bin[is_orig]

xgb_champion = joblib.load(OUT_DIR / "modelo_final_XGBoost_n_iter5_G.pkl")

print("=" * 70)
print("1. Metricas de teste completas (XGBoost, n_iter=5)")
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
    calibrador = CalibratedClassifierCV(estimator=FrozenEstimator(xgb_champion), method="sigmoid").fit(X_orig, y_orig_bin)
except ImportError:
    calibrador = CalibratedClassifierCV(estimator=xgb_champion, method="sigmoid", cv="prefit").fit(X_orig, y_orig_bin)

y_test_score = calibrador.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test_bin, y_test_score)

print(f"Train kappa: {kappa_train:.4f}")
print(f"Test: Acc={acc:.4f} Error={1-acc:.4f} Prec={prec:.4f} Sens={sens:.4f} F1={f1:.4f} "
      f"Kappa={kappa_test:.4f} AUC={auc_test:.4f} TP={tp} TN={tn} FP={fp} FN={fn}")

pd.DataFrame([{
    "Model": "XGBoost", "n_iter": 5, "Accuracy": acc, "Error": 1 - acc, "Precision": prec,
    "Sensitivity": sens, "F1": f1, "Kappa_train": kappa_train, "Kappa_test": kappa_test,
    "AUC": auc_test, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
}]).to_csv(OUT_DIR / "tabela_xgb_niter5_metricas_finais_G.csv", index=False)

print("\n" + "=" * 70)
print("2. Importancia de features (gain/cover), campeao XGBoost n_iter=5")
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

todas_features = pd.DataFrame({"Feature": X_resampled.columns})
importance_full = todas_features.merge(importance_df, on="Feature", how="left").fillna(0)
importance_full = importance_full.sort_values(by="gain", ascending=False).reset_index(drop=True)

top20_df = importance_df.sort_values("total_gain", ascending=False).head(20)
top20_df_by_gain = importance_df.head(20)
salvar1 = OUT_DIR / "tabela5_feature_importance_niter5_G.csv"
salvar2 = OUT_DIR / "tabela_feature_importance_completa_niter5_G.csv"
top20_df_by_gain.to_csv(salvar1, index=False)
importance_full.to_csv(salvar2, index=False)
print(top20_df_by_gain.head(10).to_string(index=False))
print(f"\n[tabelas salvas] {salvar1}, {salvar2}")

print("\n" + "=" * 70)
print("3. Identificar NOVOS erros de teste (XGBoost n_iter=5) para Tanimoto")
print("=" * 70)
y_pred_series = pd.Series(y_test_pred, index=X_test.index)
fp_idx = y_pred_series[(y_pred_series == 1) & (y_test_bin == 0)].index.tolist()
fn_idx = y_pred_series[(y_pred_series == 0) & (y_test_bin == 1)].index.tolist()
print(f"XGBoost (n_iter=5): {len(fp_idx)} falsos positivos {fp_idx}, {len(fn_idx)} falsos negativos {fn_idx}")
pd.DataFrame({"index": fp_idx + fn_idx, "type": ["FP"] * len(fp_idx) + ["FN"] * len(fn_idx)}).to_csv(
    OUT_DIR / "tabela_xgb_niter5_erros_teste_G.csv", index=False)

print()
print("=" * 70)
print("CONCLUIDO")
print("=" * 70)
