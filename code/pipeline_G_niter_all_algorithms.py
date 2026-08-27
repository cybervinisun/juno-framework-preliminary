"""
Pipeline G - Parte H: sensibilidade do orcamento de busca bayesiana
(n_iter = 1, 3, 5) para os QUATRO algoritmos (MLP, SVM, XGBoost,
Regressao Logistica), no MESMO protocolo (kappa, StratifiedGroupKFold
mae-filha), a partir do checkpoint pos-SVMSMOTE ja existente.

Decisao adotada por instrucao explicita do usuario: para o XGBoost, o
campeao final passa a ser o de n_iter=5 (nao mais n_iter=15), pois o
ganho marginal em kappa interno e pequeno frente ao risco de
overfitting do proprio processo de busca bayesiana a uma particao de
CV especifica. MLP e SVM ja usavam n_iter=5; Regressao Logistica
tambem passa a ser reportada com n_iter=5 como orcamento oficial.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.utils import check_random_state
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "versao_G_outputs"

ckpt = joblib.load(OUT_DIR / "checkpoint_post_svmsmote_G.pkl")
X_train_final = ckpt["X_train_final"]
y_train_final = ckpt["y_train_final"]
X_test = ckpt["X_test"]
y_test = ckpt["y_test"]
tracking_table = ckpt["tracking_table"]

X_resampled = X_train_final.copy()
groups = tracking_table["mother_original_row_id"].copy()
mask = tracking_table["is_synthetic"] == False
groups.loc[mask] = tracking_table.loc[mask, "original_row_id"]
groups = groups.astype(int)

mapeamento = {"Inativo": 0, "Ativo": 1}
y_test_bin = np.array([mapeamento[c] for c in y_test["Atividade"]], dtype=np.int64)
y_train_bin = np.array([mapeamento[c] for c in y_train_final], dtype=np.int64)

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


def bootstrap_ci_of_mean(values, n_boot=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    return float(boot_means.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


PIPELINES = {
    "MLP": (
        Pipeline(steps=[("NN", MLPClassifier(solver="lbfgs", max_iter=20000, random_state=23))]),
        {
            "NN__hidden_layer_sizes": Integer(5, 15),
            "NN__alpha": Real(1e-5, 1.0005965763586375e-05, "log-uniform"),
            "NN__activation": Categorical(["tanh"]),
            "NN__learning_rate_init": Real(1e-7, 1e-6, "log-uniform"),
        },
    ),
    "SVM": (
        Pipeline(steps=[("svm", svm.SVC(gamma="scale", max_iter=-1, probability=False))]),
        {"svm__C": Real(0.5, 1, prior="log-uniform"), "svm__gamma": Real(0.01, 1, prior="log-uniform"),
         "svm__kernel": Categorical(["rbf"])},
    ),
    "XGBoost": (
        Pipeline(steps=[("xgb", XGBClassifier(random_state=0, booster="gbtree", objective="binary:logistic"))]),
        {
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
    ),
    "LogisticRegression": (
        Pipeline(steps=[("LGR", LogisticRegression(max_iter=5000, random_state=23))]),
        {"LGR__C": Real(1e-3, 1e2, prior="log-uniform"), "LGR__penalty": Categorical(["l2"]),
         "LGR__solver": Categorical(["lbfgs"])},
    ),
}

N_ITER_VALUES = [1, 3, 5]

all_results = []
dispersion_rows = []
champions = {}

for algo, (pipe, space) in PIPELINES.items():
    print(f"\n{'='*70}\n{algo}\n{'='*70}")
    for n_iter in N_ITER_VALUES:
        bscv = BayesSearchCV(
            estimator=pipe, search_spaces=space, n_iter=n_iter,
            n_jobs=-1, cv=cv, scoring=kappa_scorer, error_score="raise", random_state=21,
            return_train_score=True, refit=True, verbose=0,
        ).fit(X_resampled, y_train_bin, groups=groups)

        champion = bscv.best_estimator_

        for idx, (mean_score, std_score) in enumerate(zip(bscv.cv_results_["mean_test_score"], bscv.cv_results_["std_test_score"])):
            dispersion_rows.append({
                "Algorithm": algo, "n_iter_budget": n_iter, "candidate_idx": idx,
                "mean_cv_kappa": mean_score, "std_cv_kappa": std_score,
                "is_best": mean_score == bscv.best_score_,
            })

        rep_scores = cross_validate(champion, X_resampled, y_train_bin, cv=cv_repetida, groups=groups,
                                     scoring=kappa_scorer, n_jobs=-1)["test_score"]
        boot_mean, boot_lo, boot_hi = bootstrap_ci_of_mean(rep_scores)

        y_test_pred = champion.predict(X_test)
        test_kappa = cohen_kappa_score(y_test_bin, y_test_pred)
        tn, fp, fn, tp = confusion_matrix(y_test_bin, y_test_pred).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)

        try:
            from sklearn.frozen import FrozenEstimator
            calibrador = __import__("sklearn.calibration", fromlist=["CalibratedClassifierCV"]).CalibratedClassifierCV(
                estimator=FrozenEstimator(champion), method="sigmoid")
        except ImportError:
            from sklearn.calibration import CalibratedClassifierCV
            calibrador = CalibratedClassifierCV(estimator=champion, method="sigmoid", cv="prefit")

        is_orig = (tracking_table["is_synthetic"] == False).to_numpy()
        X_orig = X_resampled.loc[is_orig].reset_index(drop=True)
        y_orig_bin = y_train_bin[is_orig]
        calibrador.fit(X_orig, y_orig_bin)
        y_test_score = calibrador.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_bin, y_test_score)

        row = {
            "Algorithm": algo, "n_iter_budget": n_iter,
            "search_best_kappa": bscv.best_score_,
            "repeated_cv_mean_kappa": rep_scores.mean(), "repeated_cv_sd_kappa": rep_scores.std(),
            "bootstrap_ci95_low": boot_lo, "bootstrap_ci95_high": boot_hi,
            "held_out_test_kappa": test_kappa, "held_out_test_acc": acc,
            "held_out_test_auc": test_auc,
            "best_params": str(dict(bscv.best_params_)),
        }
        all_results.append(row)
        print(f"  n_iter={n_iter}: search_kappa={bscv.best_score_:.4f}  repeated_cv={rep_scores.mean():.4f}+/-{rep_scores.std():.4f}  "
              f"test_kappa={test_kappa:.4f}  test_AUC={test_auc:.4f}")

        if n_iter == 5:
            champions[algo] = champion

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "tabela_niter_1_3_5_todos_algoritmos_G.csv", index=False)
dispersion_df = pd.DataFrame(dispersion_rows)
dispersion_df.to_csv(OUT_DIR / "tabela_niter_1_3_5_dispersao_todos_G.csv", index=False)
print(f"\n[tabela salva] {OUT_DIR / 'tabela_niter_1_3_5_todos_algoritmos_G.csv'}")

# Save the n_iter=5 champions (XGBoost's NEW official champion, others
# should match the already-saved n_iter=5 champions from before)
for algo, champion in champions.items():
    joblib.dump(champion, OUT_DIR / f"modelo_final_{algo}_n_iter5_G.pkl")
    print(f"Salvo: modelo_final_{algo}_n_iter5_G.pkl")

print()
print("=" * 70)
print("CONCLUIDO: sensibilidade n_iter=1,3,5 para os 4 algoritmos")
print("=" * 70)
