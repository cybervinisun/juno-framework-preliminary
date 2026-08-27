# results/ — table/figure provenance

Every file below corresponds to a table or figure in the published article
(version-G pipeline, `n_iter=5` champions — see `models/` and
`archive/round_niter15_exploratory/NOTES.md` for why `n_iter=5` and not an
earlier exploratory budget). Values were cross-checked line-by-line against
the numbers printed in `main.tex` before being included here.

| File | Article reference |
|---|---|
| `tab3_training_partition_metrics_G.csv` | Table 3 (training-partition performance) |
| `figG22_niter_1_3_5_cv_kappa_all_algorithms_G.csv` | Fig. G22 left panel (CV kappa vs. search budget) |
| `figG22_niter_1_3_5_candidate_dispersion_G.csv` | Fig. G22 left panel (non-selected candidate dispersion) |
| `figG22_repeated_cv_bootstrap_G.csv` | Fig. G22 right panel / Table 3 (repeated CV + bootstrap CI) |
| `tab4_heldout_test_metrics_G.csv` | Table 4 (held-out test performance, all 4 models) |
| `tab_calib_scenarios_G.csv` | Table "tab_calib_scenarios" (calibration scenarios A/B) |
| `fig7_roc_curve_points_mlp_svm_G.csv` | Fig. 7 (ROC curves) — **MLP and SVM only, see gap below** |
| `tab5_feature_importance_top_G.csv` | Table 5 (top feature importances) |
| `fig8_feature_importance_full57_G.csv` | Fig. 8 (full 57-descriptor importance) |
| `figG15_tanimoto_errors_G.csv` | Fig. G15 (Tanimoto similarity of misclassified compounds) |
| `figG15_pif_jaccard_errors_G.csv` | Fig. G15 (PIF/Jaccard similarity of misclassified compounds) |
| `figG12_pca_explained_variance_G.csv` | Fig. G12 (PCA scree) |
| `figG12_pca_loadings_G.csv` | Fig. G12 (PCA loadings) |
| `figG11_pif_prevalence_G.csv` | Fig. G11 (PIF prevalence by class) |
| `figG1_svmsmote_tracking_G.csv` | Fig. G1 (SVMSMOTE synthetic-sample tracking) |
| `logreg_baseline_summary_G.csv` | Logistic-regression baseline (Table 3/4 row + search summary) |
| `logreg_bayesian_search_candidates_G.csv` | Logistic-regression Bayesian-search candidate dispersion |
| `xgboost_niter5_full_metrics_G.csv` | XGBoost champion, train+test metrics in one row |
| `xgboost_niter5_test_errors_G.csv` | XGBoost champion, held-out test misclassifications |
| `juno_screening_713library_ranked_G.csv` | Prospective 713-compound screening ranking |

Redocking validation (Table 1 / Fig. G18) and native-contact-preservation
tables are shipped under `data/raw/` instead of here, since the pipeline
scripts (`code/redocking_analysis.py`, `code/native_contacts_analysis.py`)
treat them as ready-to-use inputs for anyone who doesn't want to re-derive
them from the raw GOLD population spreadsheets.

## How `tab3_training_partition_metrics_G.csv` and `tab4_heldout_test_metrics_G.csv` were built

No single pre-existing script output covered all four algorithms (MLP, SVM,
XGBoost, Logistic Regression) with the correct, final `n_iter=5` numbers in
one file. These two CSVs were assembled from verified per-algorithm sources
(MLP/SVM from the original train/test metrics tables, XGBoost from
`tabelas_round6_7/tabela_xgb_niter5_metricas_finais_G.csv`, Logistic
Regression from `tabela_logreg_resumo_G.csv`), with AUC taken from the Platt
calibration scenario "A2" (calibrated on the 224 original training ligands)
to match what Table 4's caption specifies. Every value was checked against
the numbers printed in `main.tex` Section 3.4/3.5 and matches exactly.

## Known gaps (not reproduced here)

- **`fig7_roc_curve_points_mlp_svm_G.csv` omits XGBoost.** The only ROC-points
  file found (`tabela_ROC_pontos_G.csv`) predates the switch to the
  `n_iter=5` XGBoost champion (Section 3.4) and was never regenerated
  afterward, so its XGBoost curve does not correspond to the published
  model. MLP and SVM are unaffected (same model artifact both before and
  after that switch) and are included. To reproduce XGBoost's ROC curve,
  rerun `code/pipeline_G.py` (or the calibration-scenarios script) against
  `models/modelo_final_XGBoost_G.pkl`.
- **Pairwise Tanimoto similarity among all consensus-error compounds** is not
  included: the only such file predates the round-7 update and is missing
  compound 55, which is part of the `n_iter=5` XGBoost champion's actual
  error set (see `figG15_tanimoto_errors_G.csv`). Rerun
  `code/pipeline_G_tanimoto_reassessment.py` to regenerate it in full.
