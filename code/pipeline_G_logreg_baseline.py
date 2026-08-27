"""
Pipeline G - Parte G: baseline de Regressao Logistica, avaliada pelo
MESMO protocolo (BayesSearchCV, scoring=kappa, StratifiedGroupKFold
mae-filha) usado para MLP/SVM/XGBoost -- para quantificar, em vez de
apenas afirmar qualitativamente, o quanto os modelos mais sofisticados
melhoram sobre um classificador linear simples (ja apontado como
insatisfatorio no estudo de comparacao de amostradores, Secao 2.6).
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
from sklearn.metrics import (
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

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

kappa_scorer = make_scorer(cohen_kappa_score)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=21)


class RepeatedStratifiedGroupKFold:
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            cvf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=rng)
            for train_idx, test_idx in cvf.split(X, y, groups):
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


cv_repetida = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=20, random_state=21)

print("=" * 70)
print("Regressao Logistica: busca bayesiana (mesmo protocolo, kappa, CV agrupada)")
print("=" * 70)

pipe_lgr = Pipeline(steps=[("LGR", LogisticRegression(max_iter=5000, random_state=23))])
pair_grid_lgr = {
    "LGR__C": Real(1e-3, 1e2, prior="log-uniform"),
    "LGR__penalty": Categorical(["l2"]),
    "LGR__solver": Categorical(["lbfgs"]),
}

bscv_lgr = BayesSearchCV(
    estimator=pipe_lgr, search_spaces=pair_grid_lgr, n_iter=5,
    n_jobs=-1, cv=cv, scoring=kappa_scorer, error_score="raise", random_state=21,
    return_train_score=True, refit=True, verbose=0,
).fit(X_resampled, y_train_bin, groups=groups)

champion_lgr = bscv_lgr.best_estimator_
print(f"best_params_: {dict(bscv_lgr.best_params_)}")
print(f"best_score_ (CV, kappa): {bscv_lgr.best_score_:.4f}")

dispersion_rows = []
for idx, (mean_score, std_score) in enumerate(zip(bscv_lgr.cv_results_["mean_test_score"], bscv_lgr.cv_results_["std_test_score"])):
    dispersion_rows.append({
        "Model": "LogisticRegression", "candidate_idx": idx, "mean_cv_kappa": mean_score,
        "std_cv_kappa": std_score, "is_best": mean_score == bscv_lgr.best_score_,
    })
dispersion_df = pd.DataFrame(dispersion_rows)
dispersion_df.to_csv(OUT_DIR / "tabela_logreg_dispersao_candidatos_G.csv", index=False)

rep_scores = cross_validate(champion_lgr, X_resampled, y_train_bin, cv=cv_repetida, groups=groups,
                             scoring=kappa_scorer, n_jobs=-1)["test_score"]
print(f"Repeated CV (100 folds): mean={rep_scores.mean():.4f} SD={rep_scores.std():.4f}")

# Held-out test performance (hard labels from champion; probability via
# native predict_proba -- logistic regression's own sigmoid link, no
# extra calibration needed for this simple baseline)
y_train_pred = champion_lgr.predict(X_resampled)
y_test_pred = champion_lgr.predict(X_test)
y_test_score = champion_lgr.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test_bin, y_test_pred).ravel()
acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp) if (tp + fp) else 0
sens = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
kappa_test = cohen_kappa_score(y_test_bin, y_test_pred)
auc_test = roc_auc_score(y_test_bin, y_test_score)
brier_test = brier_score_loss(y_test_bin, y_test_score)

kappa_train = cohen_kappa_score(y_train_bin, y_train_pred)

print(f"\nHeld-out test: Acc={acc:.4f} Prec={prec:.4f} Sens={sens:.4f} F1={f1:.4f} "
      f"Kappa={kappa_test:.4f} AUC={auc_test:.4f} Brier={brier_test:.4f} "
      f"TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"Train Kappa={kappa_train:.4f}")

summary_row = {
    "Model": "LogisticRegression (baseline)",
    "search_best_kappa": bscv_lgr.best_score_,
    "repeated_cv_mean_kappa": rep_scores.mean(),
    "repeated_cv_sd_kappa": rep_scores.std(),
    "train_kappa": kappa_train,
    "test_accuracy": acc, "test_precision": prec, "test_sensitivity": sens, "test_f1": f1,
    "test_kappa": kappa_test, "test_auc": auc_test, "test_brier": brier_test,
    "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    "best_params": str(dict(bscv_lgr.best_params_)),
}
pd.DataFrame([summary_row]).to_csv(OUT_DIR / "tabela_logreg_resumo_G.csv", index=False)
joblib.dump(champion_lgr, OUT_DIR / "modelo_final_LogisticRegression_G.pkl")
print(f"\n[tabela salva] {OUT_DIR / 'tabela_logreg_resumo_G.csv'}")
print(f"Salvo: {OUT_DIR / 'modelo_final_LogisticRegression_G.pkl'}")

# ====================================================================
# Updated Table-3-companion figures including Logistic Regression
# ====================================================================
# Reload the existing 3-model dispersion + repeated-CV data to build a
# unified 4-model comparison (MLP, SVM, XGBoost, LogisticRegression).
xgb_sensitivity_disp = pd.read_csv(OUT_DIR / "tabela_xgb_niter_dispersao_candidatos_G.csv")
xgb_15 = xgb_sensitivity_disp[xgb_sensitivity_disp["n_iter_budget"] == 15].copy()
xgb_15["Model"] = "XGBoost"

# MLP/SVM dispersion: extract from the original combined dispersion table
orig_disp = pd.read_csv(OUT_DIR / "tabela3_dispersao_candidatos_bayesianos_G.csv")
mlp_svm_disp = orig_disp[orig_disp["Model"].isin(["MLP", "SVM"])].copy()

combined_disp = pd.concat([
    mlp_svm_disp[["Model", "mean_cv_kappa"]],
    xgb_15[["Model", "mean_cv_kappa"]],
    dispersion_df[["Model", "mean_cv_kappa"]],
], ignore_index=True)

order = ["LogisticRegression", "MLP", "SVM", "XGBoost"]
colors = {"LogisticRegression": "#7f7f7f", "MLP": "#d62728", "SVM": "#2ca02c", "XGBoost": "#1f77b4"}

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
data = [combined_disp.loc[combined_disp["Model"] == m, "mean_cv_kappa"].to_numpy() for m in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True)
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor(colors[m])
    patch.set_alpha(0.6)
for m, x in zip(order, range(1, len(order) + 1)):
    vals = combined_disp.loc[combined_disp["Model"] == m, "mean_cv_kappa"]
    ax.scatter(np.full(len(vals), x) + np.random.default_rng(0).uniform(-0.05, 0.05, len(vals)),
               vals, color="black", alpha=0.6, s=18, zorder=3)
ax.set_ylabel("Internal 5-fold CV Cohen's $\\kappa$ (mother-child grouped)")
ax.set_title("Bayesian-search candidate dispersion including a\nlogistic-regression baseline")
fig.tight_layout()
fig.savefig(FIG_DIR / "figG19_logreg_vs_models_dispersion.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG19_logreg_vs_models_dispersion.png'}")

print()
print("=" * 70)
print("CONCLUIDO: baseline de Regressao Logistica")
print("=" * 70)
