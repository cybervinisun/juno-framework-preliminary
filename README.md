# A multifaceted CADD architecture for GluN1–GluN2B NMDA receptor modulators

[![Smoke test](https://github.com/cybervinisun/juno-framework-preliminary/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/cybervinisun/juno-framework-preliminary/actions/workflows/smoke-test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22149634.svg)](https://doi.org/10.5281/zenodo.22149634)

Companion data/code repository for:

> Nunes da Rocha, V. & Rabello de Sant'Anna, C. M. "A multifaceted CADD
> architecture integrating molecular docking, pharmacophore interaction
> fingerprints, and machine learning to classify negative allosteric
> modulators of the GluN1–GluN2B NMDA receptor site." *Journal of
> Cheminformatics* (2026, submitted). Article DOI: assigned by the journal
> on acceptance/publication (not yet available).

This exact code/data release is permanently archived on Zenodo:
**[10.5281/zenodo.22149634](https://doi.org/10.5281/zenodo.22149634)**.

This repository accompanies the article's data/code availability statement.
It contains the curated training dataset, the published version-G modeling
pipeline, the four verified champion classifiers, and every result table
referenced by table/figure number in the manuscript.

## Article summary

The article builds a classifier for negative allosteric modulators (NAMs) of
the GluN1–GluN2B NMDA receptor amino-terminal domain, combining:

- A curated set of 320 literature ligands (239 Active / 81 Inactive at a
  $K_i < 10\,\mu$M threshold), docked and redocked against PDB 3QEL with GOLD
  (GoldScore, selected by redocking-accuracy comparison against alternative
  scoring functions).
- A 57-covariate representation per docked complex: 9 continuous descriptors
  (5 weighted GOLD terms, a mass-corrected rescoring term "corrScore",
  ligand-burial percentage, molecular weight, and an ordinal basicity
  descriptor from MolGpKa) plus 48 binary Pharmacophore Interaction
  Fingerprints (PIFs) extracted with PLIP, each indexed by receptor residue
  and interaction type.
- SVMSMOTE class balancing (applied to the training partition only, with a
  domain-aware correction protocol and mother–child grouping preserved
  through cross-validation).
- Four classifiers (MLP, SVM-RBF, XGBoost, and a tuned logistic-regression
  baseline), each selected by Bayesian hyperparameter search
  ($n_{\mathrm{iter}}=5$, scored on Cohen's $\kappa$, mother–child-grouped
  5-fold internal CV), Platt-calibrated, and evaluated on a held-out test
  set ($n=96$).

See `main.tex` (not included here — this repo is the data/code companion,
not the manuscript itself) for the full methodology.

## Repository structure

```
.
├── code/                     analysis scripts, numbered in run order (01-12)
│   └── pymol_scripts/        PyMOL scripts for the redocking pose-overlay figure
├── data/
│   ├── processed/            320-ligand descriptor matrix, labels, SMILES,
│   │                          compound table
│   └── raw/                  redocking population statistics, native contacts
├── models/                   the four published champions, their Platt-calibrated
│                              counterparts, the training scaler and the
│                              post-SVMSMOTE training partition
├── archive/                  superseded modeling rounds, kept for historical
│                              traceability only (see "Archive" below)
├── results/                  one CSV per table/figure, named after its content;
│                              MANIFEST.csv and README.md map each file to the
│                              article element and the script that makes it
│   └── regenerated/          created by a run; never overwrites the above
├── requirements.txt / environment.yml
├── LICENSE                   Apache-2.0 (code, models)
├── LICENSE-DATA              CC-BY-4.0 (data, results)
└── CITATION.cff
```

## Data dictionary (57-descriptor matrix, Section 2.4 of the article)

`data/processed/X_320ligands_57descriptors.xlsx` (320 rows × 57 columns):

| Group | Columns | Description |
|---|---|---|
| Continuous (9) | 5 weighted GOLD terms | Internal torsional cost, internal correction, external H-bond, internal van der Waals, external van der Waals (components of GoldScore fitness, Eq. 1) |
| | `corrScore` | Mass-corrected rescoring term, `normScore·(1−normMW)^0.25` (Eq. 2); already normalised, excluded from MinMax re-scaling |
| | `LBSAD %` | Ligand burial at the binding site: % of ligand SASA buried on complex formation (Eq. 3) |
| | `MW` | Molecular weight |
| | `bpKa` (ordinal) | 4-level basicity code (weakly/moderately/strongly/very strongly basic) from MolGpKa conjugate-acid pKa estimate |
| Binary PIFs (48) | `<chain>:<resnum>_<resname>__<interaction>` | e.g. `B:110_GLN__hbond`. 1 = PLIP detected that non-covalent contact (hydrogen bond, hydrophobic, halogen bond, salt bridge, π–π stacking, cation–π, or water bridge) at that residue for that ligand's docked pose; 0 = not detected. Structural, not statistical — see Section 2.4 for the important caveat about what "0" does and does not mean. |

Continuous descriptors are MinMax-normalised (fitted on the training
partition only, Section 2.6); binary PIFs and `corrScore` are left
unscaled. `y_320ligands_labels.xlsx` carries the Active/Inactive label;
`smiles_320ligands_reference.xlsx` carries each ligand's canonical SMILES
(index-aligned with the descriptor matrix);
`compound_table_320ligands.csv` is the full compound appendix (source,
PubChem CID where available).

## Models

`models/` contains only the four classifiers reported in the article
(hyperparameters verified by direct `get_params()` inspection against
Table 2 of the manuscript — an exact match, not the closest candidate):

- `champion_MLP.pkl`, `champion_SVM.pkl`,
  `champion_XGBoost.pkl`, `champion_LogisticRegression.pkl` —
  each a fitted `sklearn.pipeline.Pipeline`.
- `minmax_scaler_fitted_on_training.pkl` — the MinMaxScaler fitted on the training
  partition only (Section 2.6); required to preprocess new data before
  calling `.predict()` on the champions above.
- `training_partition_after_svmsmote.pkl` — the SVMSMOTE-balanced, corrected
  training partition used to fit these champions (Sections 2.6/S2).

Also included:

- `champion_MLP_platt_calibrated.pkl`, `champion_SVM_platt_calibrated.pkl`,
  `champion_XGBoost_platt_calibrated.pkl` — the three retained champions with
  Platt calibration fitted on the 224 original (non-synthetic) training
  ligands (calibration scenarios, Section 3.6).
- `champion_XGBoost.ubj` — the XGBoost champion's booster in XGBoost's
  portable UBJSON format. The booster inside the `.pkl` is not portable across
  XGBoost builds; see `code/08_heldout_roc_panels.py` for how to load either.

### Archive

`archive/` holds two earlier modeling rounds that are **not** used to
produce any number reported in the article — kept only for historical
traceability of the model-selection process described in Section 3.4, not
for reproducing published results:

- `round_niter15_exploratory/` — an earlier XGBoost champion from a wider
  ($n_{\mathrm{iter}}=15$) Bayesian search, superseded when every algorithm
  was capped at $n_{\mathrm{iter}}=5$ for consistency.
- `round_thesis_final_F_divergent/` — pre-version-G SVC/XGBoost models with
  hyperparameters that do not match Table 2, from an earlier, methodologically
  different pipeline.

Each subfolder has its own `README.md` with the exact hyperparameters and the
reason it was superseded.

## Results

`results/` holds one CSV per table/figure referenced in the manuscript,
named accordingly (`tab3_...`, `tab4_...`, `figG12_...`, etc.). See
`results/README.md` for the full file-to-table/figure mapping, including two
tables (`metrics_across_search_and_validation_stages.csv`,
`metrics_heldout_test.csv`) that were reconstructed from verified
per-algorithm source files (no single prior script output covered all four
algorithms with the correct, final $n_{\mathrm{iter}}=5$ numbers in one
file) — every value in them was cross-checked against `main.tex` and
matches exactly. `results/README.md` also documents two gaps found in the
source material during curation (a stale XGBoost ROC curve and an
incomplete pairwise Tanimoto comparison, both superseded by the champion
switch to $n_{\mathrm{iter}}=5$) and how they were closed by regenerating
directly against the published models in this repository.

Redocking validation (Table 3, Fig. 4) and native-contact-preservation
data live under `data/raw/` instead, since the corresponding scripts treat
them as ready-to-use inputs — see `data/raw/README.md`.

## Reproducing results

```bash
python -m venv .venv && source .venv/bin/activate   # or: conda env create -f environment.yml
pip install -r requirements.txt
cd code
```

The scripts under `code/` are numbered in the order they are meant to run. Each
one is self-contained: it reads the deposited inputs from `data/` and `models/`
and writes everything it produces into `results/regenerated/`, never on top of
the deposited copies, so a fresh run can be diffed against what was published.
Paths can be redirected with the `DATA_DIR`, `MODEL_DIR` and `OUT_DIR`
environment variables.

Steps 01-10 and 13-14 need nothing but this repository and the pinned environment.
Steps 11-12 and the PyMOL scripts additionally need the raw GOLD output, which
is not redistributable (see the caveat below); their extracted results are
already deposited under `data/raw/`, so skip them unless you hold that data.

| # | Script | Consumes | Produces | Article element |
|---|---|---|---|---|
| 01 | `01_split_balance_and_train_champions.py` | `data/processed/` | the four champions, the training scaler, `training_partition_after_svmsmote.pkl`, training/held-out metrics, champion hyperparameters, feature importance | Tables 2, 4, 5; Figs. 5, 6 |
| 02 | `02_search_budget_all_algorithms.py` | step 01 | search-budget sensitivity (n_iter 1/3/5) for all four algorithms | Table 4, Fig. 6 |
| 03 | `03_xgboost_budget_and_bootstrap.py` | step 01 | XGBoost budget sweep and bootstrap CIs of the internal CV | Fig. 6 |
| 04 | `04_apply_xgboost_champion_switch.py` | step 02 | recomputes the cascade that depended on the XGBoost champion | Tables 5, 7 |
| 05 | `05_logistic_regression_baseline.py` | step 01 | the tuned logistic-regression baseline | Tables 4, 5 |
| 06 | `06_calibration_scenarios.py` | step 01 | the Model A / Model B calibration scenarios | Table 6 |
| 07 | `07_calibration_quality_leakfree.py` | step 01 | leak-free calibration and Brier/ECE diagnostics | Suppl. S3 |
| 08 | `08_heldout_roc_panels.py` | `models/` | the two-panel held-out ROC curves and their points | Fig. 7 |
| 09 | `09_descriptor_eda_and_pca.py` | `data/processed/` | descriptor EDA, PIF prevalence, PCA | Fig. 5; Suppl. S6-S8 |
| 10 | `10_misclassification_similarity.py` | `data/processed/`, `models/` | Tanimoto/PIF similarity of the misclassified compounds | Table 8, Fig. 8 |
| 11 | `11_redocking_rmsd_from_gold.py` | raw GOLD populations | redocking RMSD statistics | Table 3, Fig. 4 |
| 12 | `12_native_contact_preservation.py` | raw GOLD poses + PLIP | native vs. redocked contact counts | Table 3 |
| -- | `pymol_scripts/render_*.pml` | raw GOLD `.mol2` | pose-overlay renderings (`pymol -cq render_3qel.pml`) | Fig. 4 |
| 13 | `13_search_budget_and_surrogate_experiment.py` | step 01 | search budget and surrogate contrasted over 10 seeds (~40 min) | Discussion |
| 14 | `14_summarise_search_experiment.py` | step 13 | the paired statistics quoted for that experiment | Discussion |

Step 04 exists for provenance. `01_split_balance_and_train_champions.py`
already searches at the retained budget (`n_iter=5`) for all three families, so
a fresh run reproduces the published champions directly; step 04 documents how
that switch was propagated to the dependent outputs when it was first made.

`results/README.md` maps every deposited file to the table or figure it backs,
names the script that regenerates it, and records what does and does not
reproduce exactly on a different software build.

### GOLD (proprietary software) caveat

Docking and redocking were performed with **GOLD** (Cambridge
Crystallographic Data Centre), which requires a paid license we cannot
redistribute. This repository ships GOLD's *extracted numerical outputs*
(RMSD populations, native contacts) but not GOLD itself, its license file,
or its raw per-system population spreadsheets/`.mol2` structures. Any
script that would need to re-run docking from scratch is marked as such in
its own header/docstring; everything downstream of docking (descriptor
matrix, PIF extraction, model training, evaluation) is fully reproducible
with the open-source stack in `requirements.txt`/`environment.yml`.

## Reproducibility anchors

- The exact commit corresponding to the published article is tagged
  `v1.0-published`.
- This repository is permanently archived on Zenodo:
  **[10.5281/zenodo.22149634](https://doi.org/10.5281/zenodo.22149634)**
  — cite this DOI for the code/data, not the GitHub URL, since GitHub
  content can change but the Zenodo archive cannot.

## License

- **Code, scripts, and model files** (`code/`, `models/`, `archive/`):
  Apache License 2.0 — see `LICENSE`.
- **Data and result tables** (`data/`, `results/`): CC BY 4.0 — see
  `LICENSE-DATA`.

## Citation

See `CITATION.cff` (author list: Vinícius Nunes da Rocha, corresponding
author; Carlos Maurício Rabello de Sant'Anna). Cite this repository via
its Zenodo DOI: **10.5281/zenodo.22149634**. The article's own DOI is a
separate identifier, assigned by the journal on acceptance, and is not
yet known.

Repository: https://github.com/cybervinisun/juno-framework-preliminary
