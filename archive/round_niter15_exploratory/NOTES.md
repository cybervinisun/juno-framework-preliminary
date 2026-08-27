# Archived round: XGBoost, n_iter=15 (exploratory Bayesian search)

**Status: superseded. Not used to produce any result reported in the article.**
Kept here only for historical traceability, per the model-selection discussion
in Section 3.4 of the article.

## What this is

An earlier XGBoost champion from the same version-G pipeline (same 320-ligand
training data, same preprocessing/SVMSMOTE/split), but selected from a
Bayesian hyperparameter search run with `n_iter=15` instead of the `n_iter=5`
budget ultimately adopted for all four algorithms (MLP, SVM-RBF, XGBoost,
Logistic Regression) in the published version.

## Why it was superseded

The article's final analysis (Section 3.4) settled on `n_iter=5` as the
search budget for every algorithm, for consistency across models and because
the sensitivity analysis (n_iter = 1, 3, 5) showed the extra search budget of
this exploratory round bought only a marginal, non-decisive gain in internal
CV performance relative to the added optimization cost. To keep the
comparison across the four algorithms on equal footing, the `n_iter=5`
XGBoost model in `models/modelo_final_XGBoost_G.pkl` is the one reported in
Tables 1, 3, 4, and 5 and used throughout the manuscript.

## Files

- `modelo_final_XGBoost_niter15_G.pkl` — the champion XGBoost `Pipeline`
  object from this round (raw, uncalibrated).
- `modelo_final_XGBoost_niter15_calibrado_G.pkl` — its `FrozenEstimator`-based
  calibrated counterpart (probability calibration fit on this same,
  superseded model — do **not** mix with the published n_iter=5 model).
- `tabela_hiperparametros_campeoes_G.csv` — the hyperparameter table this
  round was drawn from (see the `XGBoost` row).

## Hyperparameters of this round (verified via `get_params()`)

| Parameter | Value |
|---|---|
| `n_estimators` | 55 |
| `learning_rate` | 0.2 |
| `max_depth` | 6 |
| `max_leaves` | 100 |
| `colsample_bytree` | 0.4 |
| `min_child_weight` | 0.1 |
| `gamma` | 0.0 |
| `reg_alpha` | 0.004735893375157792 |
| `reg_lambda` | 0.007809462089649638 |
| `subsample` | 0.6 |
| **Internal CV kappa** | **0.8921577109262326** |

These do **not** match Table 1 of the published article (which reports the
`n_iter=5` champion's hyperparameters) — that mismatch is expected and is
exactly why this round lives in `archive/` rather than `models/`.
