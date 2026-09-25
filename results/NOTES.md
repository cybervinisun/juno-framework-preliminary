# results/ - table/figure provenance

Every file below corresponds to a table or figure in the article (version-G
pipeline, `n_iter=5` champions - see `models/` and
`archive/round_niter15_exploratory/NOTES.md` for why `n_iter=5` and not an
earlier exploratory budget). Values were cross-checked line-by-line against the
numbers printed in `main.tex` before being included here.

Table and figure numbers below are those of the current manuscript. They changed
relative to earlier revisions, in which some figures were merged, removed or
reordered; the LaTeX label in the third column is the stable reference.

| File | Article reference | label |
|---|---|---|
| `tab3_training_partition_metrics_G.csv` | Table 4 (performance across search and validation stages) | `tab3` |
| `tab4_heldout_test_metrics_G.csv` | Table 5 (final held-out performance, all 4 models) | `tab4` |
| `tab_calib_scenarios_G.csv` | Table 6 (held-out calibration scenarios, Model A/B) | `tab_calib_scenarios` |
| `tab5_feature_importance_top_G.csv` | Table 7 (XGBoost feature importance) | `tab5` |
| `table_champion_hyperparameters_G.csv` | Table 2, "Final value" column (selected hyperparameters of the three champions; the search ranges in that table are the search spaces declared in `code/pipeline_G.py`) | `tab2` |
| `fig7_roc_curve_points_G.csv` | Figure 7 (held-out ROC curves, both panels) | `fig7` |
| `fig8_feature_importance_full57_G.csv` | Table 7, full ranking (XGBoost importance for all 57 descriptors; Table 7 prints the top entries) | `tab5` |
| `figG22_niter_1_3_5_cv_kappa_all_algorithms_G.csv` | Figure 6 left panel (CV kappa vs. search budget) | `figG22` |
| `figG22_niter_1_3_5_candidate_dispersion_G.csv` | Figure 6 left panel (non-selected candidate dispersion) | `figG22` |
| `figG22_repeated_cv_bootstrap_G.csv` | Figure 6 right panel / Table 4 (repeated CV + bootstrap CI) | `figG22` |
| `figG11_pif_prevalence_G.csv` | Supplementary Figure S6, Section S7 (PIF prevalence by class) | `fig:s7-prevalence` |
| `figG15_tanimoto_errors_G.csv` | Figure 8 (Tanimoto similarity of misclassified compounds) | `figG15` |
| `figG15_pif_jaccard_errors_G.csv` | Figure 8 (PIF/Jaccard similarity of misclassified compounds) | `figG15` |
| `figG15_tanimoto_pairs_among_errors_G.csv` | Figure 8 and Supplementary Section S4 (pairwise similarity among all 21 error compounds) | `figG15` |
| `figG12_pca_explained_variance_G.csv` | Supplementary Figure S8 (PCA scree) | - |
| `figG12_pca_loadings_G.csv` | Supplementary Figure S8 (PCA loadings) | - |
| `figG1_svmsmote_tracking_G.csv` | Supplementary Figure S2 (SVMSMOTE synthetic-sample tracking) | - |
| `logreg_baseline_summary_G.csv` | Logistic-regression baseline (Table 4/5 row + search summary) | - |
| `logreg_bayesian_search_candidates_G.csv` | Logistic-regression Bayesian-search candidate dispersion | - |
| `xgboost_niter5_full_metrics_G.csv` | XGBoost champion, train+test metrics in one row | - |
| `xgboost_niter5_test_errors_G.csv` | XGBoost champion, held-out test misclassifications | - |

Redocking validation (Table 3 / Supplementary Figure S1) and
native-contact-preservation tables are shipped under `data/raw/` instead of
here, since the pipeline scripts (`code/redocking_analysis.py`,
`code/native_contacts_analysis.py`) treat them as ready-to-use inputs for
anyone who doesn't want to re-derive them from the raw GOLD population
spreadsheets.

## How `tab3_training_partition_metrics_G.csv` and `tab4_heldout_test_metrics_G.csv` were built

No single pre-existing script output covered all four algorithms (MLP, SVM,
XGBoost, Logistic Regression) with the correct, final `n_iter=5` numbers in one
file. These two CSVs were assembled from verified per-algorithm sources
(MLP/SVM from the original train/test metrics tables, XGBoost from
the XGBoost round's own final-metrics table, and Logistic Regression from
`logreg_baseline_summary_G.csv`; the first two lived in a working folder that
is not part of this repository). Every value was checked against the numbers
printed in `main.tex` and matches exactly.

**AUC is the uncalibrated value** (column `AUC_uncalibrated`), computed from the
same champion and the same scores that produce the discrete metrics in the same
row. This follows the caption of Table 5, which states that all eight metrics,
AUC included, come from the champion without calibration. Platt-calibrated AUC
belongs to the scenario study only and is reported in
`tab_calib_scenarios_G.csv`.

Platt scaling is monotonic, so the choice is numerically immaterial for three
of the four models: SVM (0.929977), XGBoost (0.947917) and Logistic Regression
(0.939815) give the same AUC either way. Only the MLP differs - 0.903646
uncalibrated against 0.901620 calibrated - because the saturated sigmoid
collapses distinct scores into ties. An earlier revision of this file reported
the calibrated column (`AUC_platt_A2`), which is where that 0.901620 came from.

## How `fig7_roc_curve_points_G.csv` was built

Figure 7 of the article has two panels, and this file carries both, identified
by the `panel` column:

- `uncalibrated` - the scores behind the final held-out evaluation of Table 5;
- `platt_calibrated` - the same champions after Platt scaling fitted on the
  224 original, non-synthetic training ligands (calibration analysis,
  Section 3.6).

Regenerate with `code/pipeline_G_roc_panels.py`, which also renders the figure.
The SVM and XGBoost curves are identical in the two panels, as a monotonic
transformation cannot change a ROC curve; only the MLP curve differs.

## Loading the XGBoost champion

`models/final_model_XGBoost_G.pkl` stores the Booster as the byte buffer
produced by `XGBoosterSerializeToBuffer`, which is **not portable across
XGBoost builds**: on a different build, `joblib.load` fails with "input stream
corrupted" even at the same declared version.

`models/final_model_XGBoost_G.ubj` is the same champion exported in the
portable UBJSON format, and was verified to reproduce Table 5 exactly (AUC
0.947917, kappa 0.736842, TP/TN/FP/FN 65/21/3/7). Prefer it.
`code/pipeline_G_roc_panels.py` shows both paths, including a fallback that
extracts the model sub-document from the legacy buffer.

## Gaps found during curation, and how they were closed

Two files initially found in the source material predated the switch to the
`n_iter=5` XGBoost champion (Section 3.4) and were never regenerated afterward.
Rather than ship them as-is or leave them out, both were regenerated directly
against the published artifacts in this repository (`models/`,
`models/checkpoint_post_svmsmote_G.pkl`) and verified before inclusion:

- **`fig7_roc_curve_points_G.csv`** - the only pre-existing ROC-points file
  predated the champion switch, so its XGBoost curve didn't correspond to the
  published model (MLP/SVM were unaffected - same model artifact before and
  after the switch). Now regenerated by `code/pipeline_G_roc_panels.py`; the
  resulting AUCs match `main.tex` exactly in both panels.
- **`figG15_tanimoto_pairs_among_errors_G.csv`** - the only pre-existing
  pairwise-similarity file was missing compound 55, part of the `n_iter=5`
  XGBoost champion's actual error set. Regenerated by rerunning
  `code/pipeline_G_tanimoto_reassessment.py` (updated to load champions from
  `models/` rather than a local scratch folder) against the real published
  models; the resulting 21-compound error set is identical to
  `figG15_tanimoto_errors_G.csv`.

One further gap was closed later, in the revision that made Table 5
uncalibrated:

- **The calibrated champions were never deposited.**
  `code/pipeline_G_calibration_quality.py` loads
  `models/final_model_{MLP,SVM,XGBoost}_calibrated_G.pkl`, and those files were
  absent. They are now deposited, each fitted by Platt scaling on the 224
  original, non-synthetic training ligands, and each verified against the
  corresponding row of `tab_calib_scenarios_G.csv`.

## Reproducing these tables on a different software build

Everything in this folder was regenerated and checked against the article on the
software stack pinned in `environment.yml`. Re-running the pipeline on a
different build reproduces almost all of it exactly, but not quite all, and the
exception is worth stating plainly.

Reproduces exactly, verified on an independent build (scikit-learn 1.9.0,
XGBoost 3.4.1, NumPy 2.5.2, pandas 3.0.5):

- the champion hyperparameters of Table 2, including the XGBoost champion
  (`n_estimators=141`, `max_leaves=97`);
- every row of Table 5, the held-out evaluation - for XGBoost, AUC 0.947917,
  kappa 0.736842 and the 65/21/3/7 confusion matrix;
- `fig7_roc_curve_points_G.csv`, byte for byte;
- the Model A rows of Table 6, in all three calibration scenarios.

Does **not** reproduce exactly across builds:

- the three XGBoost **Model B** rows of Table 6 (B1, B2, B3). Model B is not a
  refit of a fixed model: `code/pipeline_G_calibration_scenarios.py` runs a
  fresh Bayesian search over the 224 original ligands, and for XGBoost that
  search settles on a different candidate on a different build. Its AUC came out
  0.943 rather than the 0.950 reported, with Brier, ECE and LogLoss shifting by
  up to 0.03. The MLP Model B rows reproduce exactly, so the sensitivity is
  specific to the XGBoost search, not to the Model B design.

This has a consequence worth stating, because it affects how Section 3.6 should
be read. On the environment of `environment.yml`, every proper score favours the
XGBoost Model B over Model A by a small margin. On the independent build above
that ordering does not survive: Model B still wins on all three proper scores in
the uncalibrated comparison (Brier 0.071 against 0.078, ECE 0.068 against 0.074,
LogLoss 0.252 against 0.260), but Model A wins once calibration is fitted on the
balanced partition, and AUC favours Model A in all three comparisons (0.948
against 0.943). So the *direction* of the XGBoost Model A/B difference is itself
within the noise of the re-optimisation, while its *magnitude* - small in every
scenario, on both builds - is the robust observation. The qualitative reading
that the choice between Model A and Model B is marginal for XGBoost holds on
both builds; a stronger claim, that Model B is uniformly better, would not.

The published values are those obtained on the environment of
`environment.yml`. A run that differs in those three rows is showing the build
sensitivity described here, not a failed reproduction.
