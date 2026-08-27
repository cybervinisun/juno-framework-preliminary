"""
Pipeline G - Parte B: sensibilidade do orcamento de busca bayesiana para
XGBoost (n_iter = 5, 10, 15) + robustez estatistica (bootstrap) da CV
interna, para substanciar/"convergir" com a Tabela 3 do Artigo 1.

Reusa o checkpoint pos-SVMSMOTE gerado por pipeline_G.py (mesmo conjunto
de treino balanceado, sem re-derivar/re-amostrar nada) para isolar
exclusivamente o efeito do orcamento de busca (n_iter) sobre o campeao
selecionado e sua robustez.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

warnings.filterwarnings("ignore")

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

pipe_xgb = Pipeline(steps=[("xgb", XGBClassifier(random_state=0, booster="gbtree", objective="binary:logistic"))])
pair_grid_2 = {
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
}

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


def bootstrap_ci_of_mean(values, n_boot=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    return float(boot_means.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


print("=" * 70)
print("XGBoost: sensibilidade do orcamento de busca bayesiana (n_iter)")
print("=" * 70)

n_iter_values = [5, 10, 15]
sensitivity_rows = []
dispersion_rows_all = []
repeated_cv_by_niter = {}
champions_by_niter = {}

for n_iter in n_iter_values:
    print(f"\n--- XGBoost, n_iter={n_iter} ---")
    bscv = BayesSearchCV(
        estimator=pipe_xgb, search_spaces=pair_grid_2, n_iter=n_iter,
        n_jobs=-1, cv=cv, scoring=kappa_scorer, error_score="raise", random_state=21,
        return_train_score=True, refit=True, verbose=0,
    ).fit(X_resampled, y_train_bin, groups=groups)

    champion = bscv.best_estimator_
    champions_by_niter[n_iter] = champion

    for idx, (mean_score, std_score) in enumerate(zip(bscv.cv_results_["mean_test_score"], bscv.cv_results_["std_test_score"])):
        dispersion_rows_all.append({
            "n_iter_budget": n_iter, "candidate_idx": idx, "mean_cv_kappa": mean_score,
            "std_cv_kappa": std_score, "is_best": mean_score == bscv.best_score_,
        })

    rep_scores = cross_validate(champion, X_resampled, y_train_bin, cv=cv_repetida, groups=groups,
                                 scoring=kappa_scorer, n_jobs=-1)["test_score"]
    repeated_cv_by_niter[n_iter] = rep_scores

    boot_mean, boot_lo, boot_hi = bootstrap_ci_of_mean(rep_scores)

    y_test_pred = champion.predict(X_test)
    test_kappa = cohen_kappa_score(y_test_bin, y_test_pred)
    try:
        calibrator = CalibratedClassifierCV(estimator=champion, method="sigmoid", cv="prefit").fit(X_resampled, y_train_bin)
    except TypeError:
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(champion), method="sigmoid").fit(X_resampled, y_train_bin)
    y_test_score_cal = calibrator.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test_bin, y_test_score_cal)

    sensitivity_rows.append({
        "n_iter_budget": n_iter,
        "best_params": str(dict(bscv.best_params_)),
        "search_best_kappa": bscv.best_score_,
        "repeated_cv_mean_kappa": rep_scores.mean(),
        "repeated_cv_sd_kappa": rep_scores.std(),
        "bootstrap_mean_kappa": boot_mean,
        "bootstrap_ci95_low": boot_lo,
        "bootstrap_ci95_high": boot_hi,
        "held_out_test_kappa": test_kappa,
        "held_out_test_auc_calibrated": test_auc,
    })
    print(f"  search_best_kappa={bscv.best_score_:.4f}  repeated_cv={rep_scores.mean():.4f}+/-{rep_scores.std():.4f}  "
          f"bootstrap95%CI=[{boot_lo:.4f},{boot_hi:.4f}]  test_kappa={test_kappa:.4f}  test_AUC={test_auc:.4f}")

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df.to_csv(OUT_DIR / "tabela_xgb_niter_sensibilidade_G.csv", index=False)
print(f"\n[tabela salva] {OUT_DIR / 'tabela_xgb_niter_sensibilidade_G.csv'}")

dispersion_all_df = pd.DataFrame(dispersion_rows_all)
dispersion_all_df.to_csv(OUT_DIR / "tabela_xgb_niter_dispersao_candidatos_G.csv", index=False)

# ====================================================================
# Also bring in MLP/SVM's existing single-budget champions (n_iter=5),
# reusing the exact champion pkl files, for a unified bootstrap-CI
# table/figure that stands alongside Table 3 in the article.
# ====================================================================
print()
print("=" * 70)
print("Bootstrap da robustez da CV interna: MLP e SVM (orcamento fixo, n_iter=5)")
print("=" * 70)

all_bootstrap_rows = []
for n_iter, rep_scores in repeated_cv_by_niter.items():
    boot_mean, boot_lo, boot_hi = bootstrap_ci_of_mean(rep_scores)
    all_bootstrap_rows.append({
        "Model": f"XGBoost (n_iter={n_iter})", "repeated_cv_mean": rep_scores.mean(),
        "repeated_cv_sd": rep_scores.std(), "bootstrap_ci95_low": boot_lo, "bootstrap_ci95_high": boot_hi,
    })

for label in ["MLP", "SVM"]:
    champion = joblib.load(OUT_DIR / f"modelo_final_{label}_G.pkl")
    rep_scores = cross_validate(champion, X_resampled, y_train_bin, cv=cv_repetida, groups=groups,
                                 scoring=kappa_scorer, n_jobs=-1)["test_score"]
    repeated_cv_by_niter[label] = rep_scores
    boot_mean, boot_lo, boot_hi = bootstrap_ci_of_mean(rep_scores)
    all_bootstrap_rows.append({
        "Model": label, "repeated_cv_mean": rep_scores.mean(), "repeated_cv_sd": rep_scores.std(),
        "bootstrap_ci95_low": boot_lo, "bootstrap_ci95_high": boot_hi,
    })
    print(f"  {label:<20} repeated_cv={rep_scores.mean():.4f}+/-{rep_scores.std():.4f}  "
          f"bootstrap95%CI=[{boot_lo:.4f},{boot_hi:.4f}]")

bootstrap_summary_df = pd.DataFrame(all_bootstrap_rows)
bootstrap_summary_df.to_csv(OUT_DIR / "tabela_bootstrap_robustez_CV_interna_G.csv", index=False)
print(f"\n[tabela salva] {OUT_DIR / 'tabela_bootstrap_robustez_CV_interna_G.csv'}")

# ====================================================================
# Figures (English)
# ====================================================================
MODEL_COLORS = {"XGBoost (n_iter=5)": "#9ecae1", "XGBoost (n_iter=10)": "#4292c6",
                "XGBoost (n_iter=15)": "#08519c", "MLP": "#d62728", "SVM": "#2ca02c"}

# Fig. G7: convergence plot -- selected search kappa & repeated-CV mean
# (with bootstrap 95% CI) vs. XGBoost search budget (n_iter)
fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
niters = sensitivity_df["n_iter_budget"].to_numpy()
search_k = sensitivity_df["search_best_kappa"].to_numpy()
rep_mean = sensitivity_df["repeated_cv_mean_kappa"].to_numpy()
ci_lo = sensitivity_df["bootstrap_ci95_low"].to_numpy()
ci_hi = sensitivity_df["bootstrap_ci95_high"].to_numpy()

ax.plot(niters, search_k, "o--", color="gold", markeredgecolor="black", markersize=10, label="Bayesian-search best score (single 5-fold CV)")
ax.plot(niters, rep_mean, "s-", color="#08519c", markersize=8, label="Repeated grouped CV mean (20$\\times$5 folds)")
ax.fill_between(niters, ci_lo, ci_hi, color="#08519c", alpha=0.2, label="Bootstrap 95% CI of repeated-CV mean")
ax.set_xticks(niters)
ax.set_xlabel("Bayesian-search budget (n_iter) for XGBoost")
ax.set_ylabel("Cohen's $\\kappa$ (mother-child grouped internal CV)")
ax.set_title("XGBoost: convergence of internal cross-validated $\\kappa$\nacross Bayesian-search budgets")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG7_xgb_niter_convergence.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG7_xgb_niter_convergence.png'}")

# Fig. G8: repeated-CV distributions (boxplot) for all 5 scenarios
# (XGBoost x 3 budgets + MLP + SVM champions), with bootstrap CI markers
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
order = ["MLP", "SVM", "XGBoost (n_iter=5)", "XGBoost (n_iter=10)", "XGBoost (n_iter=15)"]
data = [repeated_cv_by_niter.get(5) if k == "XGBoost (n_iter=5)" else
        repeated_cv_by_niter.get(10) if k == "XGBoost (n_iter=10)" else
        repeated_cv_by_niter.get(15) if k == "XGBoost (n_iter=15)" else
        repeated_cv_by_niter.get(k) for k in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True)
for patch, k in zip(bp["boxes"], order):
    patch.set_facecolor(MODEL_COLORS[k])
    patch.set_alpha(0.6)
for i, k in enumerate(order):
    row = bootstrap_summary_df[bootstrap_summary_df["Model"] == k]
    if len(row):
        lo = row["bootstrap_ci95_low"].values[0]
        hi = row["bootstrap_ci95_high"].values[0]
        ax.plot([i + 1, i + 1], [lo, hi], color="black", linewidth=2.5, zorder=5)
        ax.plot(i + 1, (lo + hi) / 2, marker="_", color="black", markersize=14, zorder=6)
ax.set_ylabel("Cohen's $\\kappa$ (repeated grouped internal CV, 100 folds)")
ax.set_title("Repeated internal cross-validation robustness across models\nand XGBoost search budgets (black bars: bootstrap 95% CI of the mean)")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG8_bootstrap_cv_robustness.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG8_bootstrap_cv_robustness.png'}")

print()
print("=" * 70)
print("CONCLUIDO: sensibilidade XGBoost + robustez bootstrap da CV interna")
print("=" * 70)
