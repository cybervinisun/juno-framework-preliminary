# Archived round: pre-version-G thesis models ("F", no pipeline)

**Status: superseded / divergent. Not used to produce any result reported in
the article.** Kept here only for historical traceability, per the
model-selection discussion in Section 3.4 of the article.

## What this is

SVC and XGBoost models from an earlier modeling round that predates the
version-G pipeline documented in this repository (different feature
engineering / descriptor set lineage, "descritores_F"). They are bare
estimators (`sklearn.svm.SVC`, `xgboost.sklearn.XGBClassifier`), not the
`Pipeline` objects (scaler + classifier) used everywhere else in this
repository, and were never wired into the version-G reproduction scripts
under `code/`.

## Why they are archived rather than included as champions

Their hyperparameters do not match Table 1 of the published article (verified
directly via `get_params()`). They represent a different, earlier tuning run
than the one reported in the manuscript, and are preserved here — rather than
discarded — purely for transparency about the model-selection history behind
the article, per the authors' own note in Section 3.4.

## Files

- `modelo_svc_descritores_F_sem_pipeline.pkl`
- `modelo_xgb_descritores_F_sem_pipeline.pkl`

## Hyperparameters of this round (verified via `get_params()`)

**SVC**

| Parameter | Value |
|---|---|
| `C` | 0.8187122804351649 |
| `gamma` | 0.1 |
| `kernel` | rbf |

**XGBoost**

| Parameter | Value |
|---|---|
| `n_estimators` | 400 |
| `learning_rate` | 0.01 |
| `max_depth` | 5 |
| `max_leaves` | 100 |
| `colsample_bytree` | 0.4 |
| `min_child_weight` | 0.1 |
| `gamma` | 0.0 |
| `reg_lambda` | 4.999999999999999 |
| `subsample` | 0.6 |

Compare against Table 1 of the published article and against
`models/modelo_final_SVM_G.pkl` / `models/modelo_final_XGBoost_G.pkl` — none
of the values above match the published champions.
