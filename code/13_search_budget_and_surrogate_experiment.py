"""Does the extra search effort buy anything, and does the surrogate do the work?

`BayesSearchCV` is a Gaussian-process optimizer, but it only starts fitting that
surrogate once its initialization quota of points has been spent. scikit-optimize
draws 10 initialization points by default, so at the retained budget of
`n_iter = 5` no surrogate is ever fitted and no acquisition function is ever
evaluated: the five evaluated configurations are draws from the ranges declared
in Table 2, scored by mother-child-grouped CV. This script measures that fact and
then asks what is lost by it.

Three conditions per algorithm, over the same space, folds and scorer:

  random5    n_iter=5,  n_initial_points=5   -- the retained protocol
  bayes15    n_iter=15, default init (10)    -- 10 draws, then 5 chosen by
                                                expected improvement over the GP
  random15   n_iter=15, n_initial_points=15  -- 15 draws, surrogate never fitted

`bayes15` against `random15` isolates the surrogate at a fixed budget; either
against `random5` isolates the budget. Repeated over 10 search seeds and the four
algorithms, giving 40 algorithm-seed pairs per contrast.

Held-out numbers are reported as an outcome, never as a selection criterion:
every champion here is chosen by internal CV alone, exactly as in step 01.

Output: results/regenerated/search_budget_surrogate_experiment.csv, one row per
algorithm-seed-condition, with the internal search score, how many surrogates
were actually fitted, repeated grouped CV with a bootstrap 95% CI, and held-out
kappa and AUC.

Runtime is roughly 40 minutes on 16 cores: 120 searches, each followed by a
100-fold repeated grouped cross-validation.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.utils import check_random_state
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
# Every regenerated file lands here, never on top of the deposited copies in
# results/ and models/, so a fresh run can be diffed against what was published.
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "results" / "regenerated"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Produced by step 01 into OUT_DIR; fall back to the copy deposited under
# models/ so this script also runs standalone from a fresh clone.
_ckpt_path = OUT_DIR / "training_partition_after_svmsmote.pkl"
if not _ckpt_path.exists():
    _ckpt_path = REPO_ROOT / "models" / "training_partition_after_svmsmote.pkl"
ckpt = joblib.load(_ckpt_path)

LABEL_MAP = {"Inactive": 0, "Active": 1}
X_resampled = ckpt["X_train_final"]
tracking_table = ckpt["tracking_table"]
y_train_bin = np.array([LABEL_MAP[c] for c in ckpt["y_train_final"]], dtype=np.int64)
X_test = ckpt["X_test"]
y_test_bin = np.array([LABEL_MAP[c] for c in ckpt["y_test"]["Activity"]], dtype=np.int64)

groups = tracking_table["mother_original_row_id"].copy()
is_original = tracking_table["is_synthetic"] == False
groups.loc[is_original] = tracking_table.loc[is_original, "original_row_id"]
groups = groups.astype(int)

# Platt calibration, where used below, is fitted on the original ligands only.
is_original_mask = is_original.to_numpy()
X_orig = X_resampled.loc[is_original_mask].reset_index(drop=True)
y_orig_bin = y_train_bin[is_original_mask]

KAPPA_SCORER = make_scorer(cohen_kappa_score)
CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=21)


class RepeatedStratifiedGroupKFold:
    def __init__(self, n_splits=5, n_repeats=20, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=rng)
            yield from cv.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


REPEATED_CV = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=20, random_state=21)

# Identical spaces and estimators to steps 01 and 02.
PIPELINES = {
    "MLP": (
        Pipeline([("NN", MLPClassifier(solver="lbfgs", max_iter=20000, random_state=23))]),
        {
            "NN__hidden_layer_sizes": Integer(5, 15),
            "NN__alpha": Real(1e-5, 1.0005965763586375e-05, "log-uniform"),
            "NN__activation": Categorical(["tanh"]),
            "NN__learning_rate_init": Real(1e-7, 1e-6, "log-uniform"),
        },
    ),
    "SVM": (
        Pipeline([("svm", svm.SVC(gamma="scale", max_iter=-1, probability=False))]),
        {
            "svm__C": Real(0.5, 1, prior="log-uniform"),
            "svm__gamma": Real(0.01, 1, prior="log-uniform"),
            "svm__kernel": Categorical(["rbf"]),
        },
    ),
    "XGBoost": (
        Pipeline([("xgb", XGBClassifier(random_state=0, booster="gbtree", objective="binary:logistic"))]),
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
        Pipeline([("LGR", LogisticRegression(max_iter=5000, random_state=23))]),
        {
            "LGR__C": Real(1e-3, 1e2, prior="log-uniform"),
            "LGR__penalty": Categorical(["l2"]),
            "LGR__solver": Categorical(["lbfgs"]),
        },
    ),
}

# (name, n_iter, n_initial_points); None keeps scikit-optimize's default of 10.
CONDITIONS = [("random5", 5, 5), ("bayes15", 15, None), ("random15", 15, 15)]
SEEDS = [21, 7, 101, 1234, 2718, 3, 42, 777, 31415, 99991]


def bootstrap_ci_of_mean(values, n_boot=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def platt_calibrated_auc(champion):
    try:
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(champion), method="sigmoid")
    except ImportError:                                   # scikit-learn < 1.6
        calibrator = CalibratedClassifierCV(estimator=champion, method="sigmoid", cv="prefit")
    calibrator.fit(X_orig, y_orig_bin)
    return roc_auc_score(y_test_bin, calibrator.predict_proba(X_test)[:, 1])


rows = []
out_path = OUT_DIR / "search_budget_surrogate_experiment.csv"
for seed in SEEDS:
    for algo, (pipe, space) in PIPELINES.items():
        for condition, n_iter, n_initial in CONDITIONS:
            kwargs = {} if n_initial is None else {"optimizer_kwargs": {"n_initial_points": n_initial}}
            search = BayesSearchCV(
                estimator=pipe, search_spaces=space, n_iter=n_iter, n_jobs=-1,
                cv=CV, scoring=KAPPA_SCORER, error_score="raise",
                random_state=seed, refit=True, **kwargs,
            ).fit(X_resampled, y_train_bin, groups=groups)

            champion = search.best_estimator_
            repeated = cross_validate(champion, X_resampled, y_train_bin, cv=REPEATED_CV,
                                      groups=groups, scoring=KAPPA_SCORER, n_jobs=-1)["test_score"]
            ci_low, ci_high = bootstrap_ci_of_mean(repeated)
            y_pred = champion.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()

            rows.append({
                "seed": seed, "algorithm": algo, "condition": condition, "n_iter": n_iter,
                # 0 means the Gaussian process was never fitted: every point is an
                # initialization draw. A value above 1 means points were proposed
                # by the acquisition function.
                "surrogates_fitted": len(search.optimizer_results_[0].models),
                "search_kappa": float(search.best_score_),
                "repeated_cv_mean": float(repeated.mean()),
                "repeated_cv_sd": float(repeated.std()),
                "ci95_low": ci_low, "ci95_high": ci_high,
                "heldout_kappa": float(cohen_kappa_score(y_test_bin, y_pred)),
                "heldout_auc": float(platt_calibrated_auc(champion)),
                "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
                "best_params": json.dumps({k: (v.item() if hasattr(v, "item") else v)
                                           for k, v in search.best_params_.items()}),
            })
            pd.DataFrame(rows).to_csv(out_path, index=False)   # checkpoint as we go
            print(f"seed={seed:<7}{algo:<20}{condition:<10}"
                  f"surrogates={rows[-1]['surrogates_fitted']:<3}"
                  f"search_kappa={rows[-1]['search_kappa']:.6f}  "
                  f"repeated_cv={repeated.mean():.4f} [{ci_low:.4f}, {ci_high:.4f}]  "
                  f"heldout_kappa={rows[-1]['heldout_kappa']:.4f}  "
                  f"AUC={rows[-1]['heldout_auc']:.4f}", flush=True)

print(f"\n[table saved] {out_path}")
print("Summarise with code/14_summarise_search_experiment.py")
