"""
Pipeline G - Parte E: calibracao propriamente cross-validada (sem
reaproveitar dados de treino como dados de calibracao) + metricas de
qualidade de calibracao (Brier score, ECE, diagrama de confiabilidade).

Correcao metodologica explicita em relacao a rodada anterior: la, os
campeoes eram calibrados via `CalibratedClassifierCV(FrozenEstimator(
modelo), method='sigmoid').fit(X_resampled, y_train_bin)` -- a propria
docstring do scikit-learn adverte que, com FrozenEstimator, "all
provided data is used for calibration" e que "the user has to take
care manually that data for model fitting and calibration are
disjoint". Como X_resampled e EXATAMENTE o conjunto usado para treinar
o campeao, essa calibracao nao era leak-free (o sigmoide era ajustado
contra previsoes de um modelo que ja tinha visto -- e, no caso do
XGBoost, memorizado perfeitamente -- esses mesmos dados).

Aqui, a calibracao passa a usar o proprio mecanismo de CV do
CalibratedClassifierCV (sem FrozenEstimator): para cada uma das 5
dobras agrupadas mae-filha, um CLONE do pipeline (com os hiperparametros
ja vencedores da busca bayesiana, Tabela 2) e re-ajustado nas 4 dobras
de treino e o sigmoide e ajustado na dobra de calibracao retida --
nunca a mesma usada para ajustar aquela copia do modelo. As 5
copias+calibradores sao ensembladas na predicao final (comportamento
padrao ensemble=True do scikit-learn).
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
OUT_DIR = BASE_DIR / "versao_G_outputs"
FIG_DIR = OUT_DIR / "figuras_ingles_G"
FIG_DIR.mkdir(exist_ok=True)

ckpt = joblib.load(OUT_DIR / "checkpoint_post_svmsmote_G.pkl")
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

mapeamento = {"Inativo": 0, "Ativo": 1}
y_test_bin = np.array([mapeamento[c] for c in y_test["Atividade"]], dtype=np.int64)
y_train_bin = np.array([mapeamento[c] for c in y_resampled], dtype=np.int64)

hp = pd.read_csv(OUT_DIR / "tabela_hiperparametros_campeoes_G.csv").set_index("Model")


def fresh_pipeline(label: str) -> Pipeline:
    """Constroi um pipeline NAO ajustado com os hiperparametros vencedores
    (Tabela 2) -- necessario para a calibracao propriamente cross-validada,
    que precisa reajustar copias do modelo a cada dobra."""
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
    """Score continuo pre-calibracao: decision_function se existir (SVM),
    senao predict_proba (MLP/XGBoost)."""
    step = fitted_pipe.steps[-1][1]
    if hasattr(step, "decision_function"):
        return fitted_pipe.decision_function(X)
    return fitted_pipe.predict_proba(X)[:, 1]


def leakfree_grouped_calibration(label: str, X_resampled, y_train_bin, groups, cv):
    """Calibracao Platt (sigmoide) propriamente cross-validada e agrupada
    mae-filha: a cada dobra, um CLONE do pipeline (hiperparametros ja
    vencedores) e reajustado nas 4 dobras de treino; o sigmoide (regressao
    logistica 1D) e ajustado SOMENTE na dobra retida, nunca vista por
    aquela copia do modelo. As 5 copias (modelo, calibrador) sao
    ensembladas (media das probabilidades) na predicao final -- mesma
    logica que `CalibratedClassifierCV(cv=5, ensemble=True)`, implementada
    manualmente para evitar problemas de metadata-routing de `groups`
    nesta versao do scikit-learn."""
    pares = []
    for train_idx, calib_idx in cv.split(X_resampled, y_train_bin, groups):
        modelo_fold = clone(fresh_pipeline(label))
        modelo_fold.fit(X_resampled.iloc[train_idx], y_train_bin[train_idx])

        score_calib = _raw_score(modelo_fold, X_resampled.iloc[calib_idx]).reshape(-1, 1)
        sigmoide = LogisticRegression().fit(score_calib, y_train_bin[calib_idx])

        pares.append((modelo_fold, sigmoide))
    return pares


def leakfree_predict_proba(pares, X):
    probs = []
    for modelo_fold, sigmoide in pares:
        score = _raw_score(modelo_fold, X).reshape(-1, 1)
        probs.append(sigmoide.predict_proba(score)[:, 1])
    return np.mean(probs, axis=0)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Identica a funcao do notebook RECONSTRUIDO (celula 116), reusada
    para fidelidade -- ECE com bins uniformes em [0,1]."""
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
print("Calibracao leak-free (CalibratedClassifierCV, cv=5, ensemble=True)")
print("vs. calibracao anterior (FrozenEstimator, mesmos dados de fit+calib)")
print("=" * 70)

rows = []
reliability_data = {}
old_new_compare = []

for label in ["MLP", "SVM", "XGBoost"]:
    print(f"\n--- {label} ---")
    pares_calibracao = leakfree_grouped_calibration(label, X_resampled, y_train_bin, groups, cv)

    y_score_novo = leakfree_predict_proba(pares_calibracao, X_test)
    auc_novo = roc_auc_score(y_test_bin, y_score_novo)
    brier_novo = brier_score_loss(y_test_bin, y_score_novo)
    ece_novo, ece_bins_novo = expected_calibration_error(y_test_bin, y_score_novo)

    calibrador_antigo = joblib.load(OUT_DIR / f"modelo_final_{label}_calibrado_G.pkl")
    y_score_antigo = calibrador_antigo.predict_proba(X_test)[:, 1]
    auc_antigo = roc_auc_score(y_test_bin, y_score_antigo)
    brier_antigo = brier_score_loss(y_test_bin, y_score_antigo)
    ece_antigo, _ = expected_calibration_error(y_test_bin, y_score_antigo)

    print(f"  Leak-free (cv=5 ensemble):    AUC={auc_novo:.4f}  Brier={brier_novo:.4f}  ECE={ece_novo:.4f}")
    print(f"  Anterior (FrozenEstimator):   AUC={auc_antigo:.4f}  Brier={brier_antigo:.4f}  ECE={ece_antigo:.4f}")

    rows.append({
        "Model": label,
        "AUC_leakfree": auc_novo, "Brier_leakfree": brier_novo, "ECE_leakfree": ece_novo,
        "AUC_frozen_prefit_style": auc_antigo, "Brier_frozen_prefit_style": brier_antigo,
        "ECE_frozen_prefit_style": ece_antigo,
    })
    reliability_data[label] = (y_score_novo, y_score_antigo)

    joblib.dump(pares_calibracao, OUT_DIR / f"modelo_final_{label}_calibrado_leakfree_G.pkl")

calib_quality_df = pd.DataFrame(rows)
calib_quality_df.to_csv(OUT_DIR / "tabela_qualidade_calibracao_G.csv", index=False)
print(f"\n[tabela salva] {OUT_DIR / 'tabela_qualidade_calibracao_G.csv'}")

# Update Table 4 test-set AUC to the leak-free values (hard labels/other
# metrics unaffected -- unchanged from the champion's own .predict()).
test_df = pd.read_csv(OUT_DIR / "tabela_metricas_teste_externo_G.csv")
test_df = test_df.set_index("Model")
for label in ["MLP", "SVM", "XGBoost"]:
    test_df.loc[label, "AUC"] = calib_quality_df.set_index("Model").loc[label, "AUC_leakfree"]
test_df = test_df.reset_index()
test_df.to_csv(OUT_DIR / "tabela_metricas_teste_externo_G.csv", index=False)
print(f"[tabela atualizada] AUC de {OUT_DIR / 'tabela_metricas_teste_externo_G.csv'} agora usa a calibracao leak-free")

# ====================================================================
# Fig. reliability diagram (calibration curve): leak-free vs. previous
# ====================================================================
MODEL_COLORS = {"MLP": "#d62728", "SVM": "#2ca02c", "XGBoost": "#1f77b4"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
for label, color in MODEL_COLORS.items():
    y_score_novo, y_score_antigo = reliability_data[label]
    frac_pos_novo, mean_pred_novo = calibration_curve(y_test_bin, y_score_novo, n_bins=5, strategy="quantile")
    frac_pos_old, mean_pred_old = calibration_curve(y_test_bin, y_score_antigo, n_bins=5, strategy="quantile")
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
print(f"Salvo: {FIG_DIR / 'figG14_reliability_diagrams.png'}")

print()
print("=" * 70)
print("CONCLUIDO: calibracao leak-free + metricas de qualidade (Brier/ECE)")
print("=" * 70)
