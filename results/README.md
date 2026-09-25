# results/ - table/figure provenance

Every file below corresponds to a table or figure in the article, from the
`n_iter=5` champions (see `archive/round_niter15_exploratory/README.md` for why
that budget and not a wider exploratory one). Values were cross-checked
line-by-line against the numbers printed in the manuscript before being
included here.

Files are named after their content rather than after a table or figure number,
because that numbering has already shifted once between revisions.
`MANIFEST.csv` in this folder is the machine-readable version of the table
below: it maps each file to its article element, its stable LaTeX label, and the
script under `code/` that regenerates it.

Table and figure numbers below are those of the current manuscript. They changed
relative to earlier revisions, in which some figures were merged, removed or
reordered; the LaTeX label in the third column is the stable reference.

| File | Article reference | label |
|---|---|---|
| `metrics_across_search_and_validation_stages.csv` | Table 4 (performance across search and validation stages) | `tab3` |
| `metrics_heldout_test.csv` | Table 5 (final held-out performance, all 4 models) | `tab4` |
| `calibration_scenarios.csv` | Table 6 (held-out calibration scenarios, Model A/B) | `tab_calib_scenarios` |
| `xgboost_feature_importance_top.csv` | Table 7 (XGBoost feature importance) | `tab5` |
| `champion_hyperparameters.csv` | Table 2, "Final value" column (selected hyperparameters of the three champions; the search ranges in that table are the search spaces declared in `code/01_split_balance_and_train_champions.py`) | `tab2` |
| `heldout_roc_curve_points.csv` | Figure 7 (held-out ROC curves, both panels) | `fig7` |
| `xgboost_feature_importance_all_57.csv` | Table 7, full ranking (XGBoost importance for all 57 descriptors; Table 7 prints the top entries) | `tab5` |
| `search_budget_cv_kappa_all_algorithms.csv` | Figure 6 left panel (CV kappa vs. search budget) | `figG22` |
| `search_budget_candidate_dispersion.csv` | Figure 6 left panel (non-selected candidate dispersion) | `figG22` |
| `repeated_cv_bootstrap_ci.csv` | Figure 6 right panel / Table 4 (repeated CV + bootstrap CI) | `figG22` |
| `pif_prevalence_by_class.csv` | Supplementary Figure S6, Section S7 (PIF prevalence by class) | `fig:s7-prevalence` |
| `error_compounds_tanimoto_to_training.csv` | Figure 8 (Tanimoto similarity of misclassified compounds) | `figG15` |
| `error_compounds_pif_jaccard.csv` | Figure 8 (PIF/Jaccard similarity of misclassified compounds) | `figG15` |
| `error_compounds_pairwise_tanimoto.csv` | Figure 8 and Supplementary Section S4 (pairwise similarity among all 21 error compounds) | `figG15` |
| `pca_explained_variance.csv` | Supplementary Figure S8 (PCA scree) | - |
| `pca_loadings.csv` | Supplementary Figure S8 (PCA loadings) | - |
| `svmsmote_synthetic_tracking.csv` | Supplementary Figure S2 (SVMSMOTE synthetic-sample tracking) | - |
| `logreg_baseline_summary.csv` | Logistic-regression baseline (Table 4/5 row + search summary) | - |
| `logreg_search_candidates.csv` | Logistic-regression Bayesian-search candidate dispersion | - |
| `xgboost_champion_train_and_test_metrics.csv` | XGBoost champion, train+test metrics in one row | - |
| `xgboost_champion_test_errors.csv` | XGBoost champion, held-out test misclassifications | - |
| `search_budget_surrogate_experiment.csv` | Discussion (search budget and surrogate; 120 runs over 10 seeds, produced by `code/13_search_budget_and_surrogate_experiment.py`) | - |

Redocking validation (Table 3 / Supplementary Figure S1) and
native-contact-preservation tables are shipped under `data/raw/` instead of
here, since the pipeline scripts (`code/11_redocking_rmsd_from_gold.py`,
`code/12_native_contact_preservation.py`) treat them as ready-to-use inputs for
anyone who doesn't want to re-derive them from the raw GOLD population
spreadsheets.

## How `metrics_across_search_and_validation_stages.csv` and `metrics_heldout_test.csv` were built

No single pre-existing script output covered all four algorithms (MLP, SVM,
XGBoost, Logistic Regression) with the correct, final `n_iter=5` numbers in one
file. These two CSVs were assembled from verified per-algorithm sources
(MLP/SVM from the original train/test metrics tables, XGBoost from
the XGBoost round's own final-metrics table, and Logistic Regression from
`logreg_baseline_summary.csv`; the first two lived in a working folder that
is not part of this repository). Every value was checked against the numbers
printed in `main.tex` and matches exactly.

**AUC is the uncalibrated value** (column `AUC_uncalibrated`), computed from the
same champion and the same scores that produce the discrete metrics in the same
row. This follows the caption of Table 5, which states that all eight metrics,
AUC included, come from the champion without calibration. Platt-calibrated AUC
belongs to the scenario study only and is reported in
`calibration_scenarios.csv`.

Platt scaling is monotonic, so the choice is numerically immaterial for three
of the four models: SVM (0.929977), XGBoost (0.947917) and Logistic Regression
(0.939815) give the same AUC either way. Only the MLP differs - 0.903646
uncalibrated against 0.901620 calibrated - because the saturated sigmoid
collapses distinct scores into ties. An earlier revision of this file reported
the calibrated column (`AUC_platt_A2`), which is where that 0.901620 came from.

## How `heldout_roc_curve_points.csv` was built

Figure 7 of the article has two panels, and this file carries both, identified
by the `panel` column:

- `uncalibrated` - the scores behind the final held-out evaluation of Table 5;
- `platt_calibrated` - the same champions after Platt scaling fitted on the
  224 original, non-synthetic training ligands (calibration analysis,
  Section 3.6).

Regenerate with `code/08_heldout_roc_panels.py`, which also renders the figure.
The SVM and XGBoost curves are identical in the two panels, as a monotonic
transformation cannot change a ROC curve; only the MLP curve differs.

## Loading the XGBoost champion

`models/champion_XGBoost.pkl` stores the Booster as the byte buffer
produced by `XGBoosterSerializeToBuffer`, which is **not portable across
XGBoost builds**: on a different build, `joblib.load` fails with "input stream
corrupted" even at the same declared version.

`models/champion_XGBoost.ubj` is the same champion exported in the
portable UBJSON format, and was verified to reproduce Table 5 exactly (AUC
0.947917, kappa 0.736842, TP/TN/FP/FN 65/21/3/7). Prefer it.
`code/08_heldout_roc_panels.py` shows both paths, including a fallback that
extracts the model sub-document from the legacy buffer.

## Gaps found during curation, and how they were closed

Two files initially found in the source material predated the switch to the
`n_iter=5` XGBoost champion (Section 3.4) and were never regenerated afterward.
Rather than ship them as-is or leave them out, both were regenerated directly
against the published artifacts in this repository (`models/`,
`models/training_partition_after_svmsmote.pkl`) and verified before inclusion:

- **`heldout_roc_curve_points.csv`** - the only pre-existing ROC-points file
  predated the champion switch, so its XGBoost curve didn't correspond to the
  published model (MLP/SVM were unaffected - same model artifact before and
  after the switch). Now regenerated by `code/08_heldout_roc_panels.py`; the
  resulting AUCs match `main.tex` exactly in both panels.
- **`error_compounds_pairwise_tanimoto.csv`** - the only pre-existing
  pairwise-similarity file was missing compound 55, part of the `n_iter=5`
  XGBoost champion's actual error set. Regenerated by rerunning
  `code/10_misclassification_similarity.py` (updated to load champions from
  `models/` rather than a local scratch folder) against the real published
  models; the resulting 21-compound error set is identical to
  `error_compounds_tanimoto_to_training.csv`.

One further gap was closed later, in the revision that made Table 5
uncalibrated:

- **The calibrated champions were never deposited.**
  `code/07_calibration_quality_leakfree.py` loads
  `models/final_model_{MLP,SVM,XGBoost}_calibrated_G.pkl`, and those files were
  absent. They are now deposited, each fitted by Platt scaling on the 224
  original, non-synthetic training ligands, and each verified against the
  corresponding row of `calibration_scenarios.csv`.

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
- `heldout_roc_curve_points.csv`, byte for byte;
- the Model A rows of Table 6, in all three calibration scenarios.

Does **not** reproduce exactly across builds:

- the three XGBoost **Model B** rows of Table 6 (B1, B2, B3). On the versions
  pinned in `environment.yml`, a fresh run of `code/06_calibration_scenarios.py`
  gives AUC 0.943 where the deposited table reports 0.950, with Brier, ECE and
  LogLoss differing by up to 0.03. The MLP and SVM Model B rows reproduce
  exactly, so whatever is going on is specific to the XGBoost search.

  The cause has not been identified, and the honest statement is that these
  three values are not reproducible from the deposited code as it stands. What
  was ruled out: it is not thread-count non-determinism (identical at 1, 4 and
  16 threads), and it is not a tie-break between the candidates this code
  evaluates - none of the five candidates a fresh search proposes yields the
  published numbers, and neither does refitting the Model A champion nor the
  superseded wider-budget champion on the 224 originals. Model B is
  re-optimized rather than refitted, so it depends on which points the search
  proposes, and the deposited row appears to come from a search that proposed a
  different set.

  Worth knowing when reading those rows: among the candidates a fresh run does
  evaluate, the best two are separated by 0.000117 in the internal CV kappa that
  selects them, yet differ by 0.008 in held-out AUC. The selection criterion
  cannot resolve candidates whose held-out behaviour differs materially, so the
  identity of "the" Model B champion is not a stable quantity at this budget.

This has a consequence worth stating, because it affects how Section 3.6 should
be read. The deposited table has every proper score favouring the XGBoost Model
B over Model A by a small margin. That ordering is not robust: on a fresh run
Model A wins on AUC in all three comparisons (0.948 against 0.943) and on the
proper scores once calibration is fitted on the balanced partition, while Model
B still wins on all three proper scores in the uncalibrated comparison (Brier
0.071 against 0.078, ECE 0.068 against 0.074, LogLoss 0.252 against 0.260). The
robust observation is the *magnitude*: the Model A/B difference is small in
every scenario and on both runs. The *direction*, for XGBoost, is not something
this design pins down, and a claim that Model B is uniformly better should not
be relied on.

The published values are those obtained on the environment of
`environment.yml`. A run that differs in those three rows is showing the build
sensitivity described here, not a failed reproduction.

## Comparing a fresh run against the deposited files

A run writes into `results/regenerated/` using the same file names as the
deposited copies here, so the two can be diffed directly. On the independent
build described above, the following came out identical to the deposited files:
`heldout_roc_curve_points.csv`, `pca_explained_variance.csv`, `pca_loadings.csv`,
`pif_prevalence_by_class.csv`, `svmsmote_synthetic_tracking.csv`,
`search_budget_candidate_dispersion.csv`, `logreg_baseline_summary.csv`,
`logreg_search_candidates.csv`, `xgboost_feature_importance_all_57.csv`,
`xgboost_champion_train_and_test_metrics.csv`,
`xgboost_champion_test_errors.csv` and
`error_compounds_pairwise_tanimoto.csv`.

Three deposited files are not a byte-for-byte target of any single script, and
that is by design rather than a defect:

- `metrics_heldout_test.csv` is Table 5, which covers four models. No single
  script produces all four rows: the three champions come from step 01, which
  writes them as `metrics_heldout_three_champions.csv`, and the
  logistic-regression row comes from step 05. Every value shared between the
  two files agrees exactly. The deposited table also carries the `Error` column
  the article prints, and names its AUC column `AUC_uncalibrated` to be explicit
  about which scenario it reports.
- `error_compounds_tanimoto_to_training.csv` drops two working columns that
  step 10 also emits (`error_type_by_model` and `smiles`); the SMILES of these
  compounds are already in `data/processed/smiles_320ligands_reference.xlsx`.
- `champion_hyperparameters.csv` records the XGBoost internal CV kappa rounded
  to `0.868`. That is a precision difference between two deposited files, not a
  disagreement: the full-precision value `0.868213` that a fresh run prints is
  itself deposited, in `search_budget_cv_kappa_all_algorithms.csv`. The
  hyperparameters, which are what Table 2 reports, match exactly.

And `calibration_scenarios.csv` differs in its three XGBoost Model B rows for
the build reason explained in the previous section.
