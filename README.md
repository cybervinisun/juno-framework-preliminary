# A multifaceted CADD architecture for GluN1–GluN2B NMDA receptor modulators

Companion data/code repository for:

> Nunes da Rocha, V. & Rabello de Sant'Anna, C. M. "A multifaceted CADD
> architecture integrating molecular docking, pharmacophore interaction
> fingerprints, and machine learning to classify negative allosteric
> modulators of the GluN1–GluN2B NMDA receptor site." *Journal of
> Cheminformatics* (2026, submitted). DOI: TBD.

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
- A prospective, in-house-synthesized 713-compound screening library scored
  by the retained champions (not used for training/tuning/calibration).

See `main.tex` (not included here — this repo is the data/code companion,
not the manuscript itself) for the full methodology.

## Repository structure

```
.
├── code/                     pipeline scripts (see "Reproducing results" below)
│   └── pymol_scripts/        PyMOL scripts for the redocking pose-overlay figure
├── data/
│   ├── processed/            320-ligand training matrix, labels, SMILES, appendix
│   └── raw/                  redocking population stats, native contacts,
│                              anonymized 713-compound prospective library
├── models/                   the 4 verified, published champion models + scaler
│                              + post-SVMSMOTE checkpoint
├── archive/                  superseded modeling rounds, kept for historical
│                              traceability only (see "Archive" below)
├── results/                  CSV tables/figures data, named by article
│                              table/figure number (see results/NOTES.md)
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
`compound_library_master_table.csv` is the full compound appendix (source,
PubChem CID where available).

## Models

`models/` contains only the four classifiers reported in the article
(hyperparameters verified by direct `get_params()` inspection against
Table 1 of the manuscript — an exact match, not the closest candidate):

- `modelo_final_MLP_G.pkl`, `modelo_final_SVM_G.pkl`,
  `modelo_final_XGBoost_G.pkl`, `modelo_final_LogisticRegression_G.pkl` —
  each a fitted `sklearn.pipeline.Pipeline`.
- `scaler_minmax_treino_G.pkl` — the MinMaxScaler fitted on the training
  partition only (Section 2.6); required to preprocess new data before
  calling `.predict()` on the champions above.
- `checkpoint_post_svmsmote_G.pkl` — the SVMSMOTE-balanced, corrected
  training partition used to fit these champions (Sections 2.6/S2).

**Calibrated (Platt-scaled) versions of these models are not included.**
`code/pipeline_G_juno_screening.py` looks for them optionally and falls
back to label-only predictions if absent — see that script's own note.

### Archive

`archive/` holds two earlier modeling rounds that are **not** used to
produce any number reported in the article — kept only for historical
traceability of the model-selection process described in Section 3.4, not
for reproducing published results:

- `round_niter15_exploratory/` — an earlier XGBoost champion from a wider
  ($n_{\mathrm{iter}}=15$) Bayesian search, superseded when every algorithm
  was capped at $n_{\mathrm{iter}}=5$ for consistency.
- `round_thesis_final_F_divergent/` — pre-version-G SVC/XGBoost models with
  hyperparameters that do not match Table 1, from an earlier, methodologically
  different pipeline.

Each subfolder has its own `NOTES.md` with the exact hyperparameters and the
reason it was superseded.

## Results

`results/` holds one CSV per table/figure referenced in the manuscript,
named accordingly (`tab3_...`, `tab4_...`, `figG12_...`, etc.). See
`results/NOTES.md` for the full file-to-table/figure mapping, including two
tables (`tab3_training_partition_metrics_G.csv`,
`tab4_heldout_test_metrics_G.csv`) that were reconstructed from verified
per-algorithm source files (no single prior script output covered all four
algorithms with the correct, final $n_{\mathrm{iter}}=5$ numbers in one
file) — every value in them was cross-checked against `main.tex` and
matches exactly. That file also documents two known, intentionally
unreproduced gaps (a stale XGBoost ROC curve and an incomplete pairwise
Tanimoto comparison — both superseded by the champion switch to
$n_{\mathrm{iter}}=5$ and never regenerated).

Redocking validation (Table 1, Fig. G18) and native-contact-preservation
data live under `data/raw/` instead, since the corresponding scripts treat
them as ready-to-use inputs — see `data/raw/NOTES.md`.

### Anonymized prospective screening library

The 713-compound prospective library (`data/raw/prospective_library_713_*`,
`results/juno_screening_713library_ranked_G.csv`) was synthesized in-house
and has no experimental validation yet against the target used here.
**SMILES and compound names have been removed** and replaced with a
sequential `compound_id`, to avoid publicly linking a specific real
compound to a target-specific activity prediction ahead of any future
patent filing. Descriptors and predictions are otherwise unmodified — see
`data/raw/NOTES.md` for the full rationale.

## Reproducing results

```bash
python -m venv .venv && source .venv/bin/activate   # or: conda env create -f environment.yml
pip install -r requirements.txt
```

Each script under `code/` is self-contained and reads from `data/` and
`models/` by default (override with the `DATA_DIR`, `MODEL_DIR`, `OUT_DIR`
environment variables — see each script's header). Suggested order:

1. `pipeline_G.py` — full pipeline: split, SVMSMOTE, Bayesian search, champion
   selection, calibration, held-out evaluation (Tables 3/4).
2. `pipeline_G_pca_eda.py`, `pipeline_G_tanimoto_reassessment.py` — EDA/PCA
   (Figs. G9–G12) and structural error analysis (Fig. G15).
3. `pipeline_G_calibration_scenarios.py`, `pipeline_G_calibration_quality.py`
   — calibration scenarios and Brier/ECE diagnostics.
4. `pipeline_G_niter_all_algorithms.py`, `pipeline_G_xgb_sensitivity.py`,
   `pipeline_G_logreg_baseline.py` — budget-sensitivity study (Fig. G22) and
   the logistic-regression baseline.
5. `redocking_analysis.py`, `native_contacts_analysis.py` — redocking
   validation (Table 1, Fig. G18) and native-contact preservation; these
   need the raw, per-system GOLD population spreadsheets (**not included**,
   see below) — `data/raw/` already ships their extracted output for anyone
   who just wants the numbers.
6. `pymol_scripts/render_*.pml` — pose-overlay rendering for Fig. 7; also
   need the raw GOLD `.mol2` outputs (**not included**), run with
   `pymol -cq render_XXXX.pml`.
7. `pipeline_G_juno_screening.py` — scores the anonymized 713-compound
   library with the four champions.
8. `rebuild_xgb_niter5_cascade.py` — utility script documenting how the
   XGBoost champion switch (Section 3.4) was cascaded through dependent
   outputs; provided for transparency, not needed for a fresh run.

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
- This repository is intended to be archived on Zenodo on publication to
  mint a citable DOI (to be added to `CITATION.cff` and this README once
  available).

## License

- **Code, scripts, and model files** (`code/`, `models/`, `archive/`):
  Apache License 2.0 — see `LICENSE`.
- **Data and result tables** (`data/`, `results/`): CC BY 4.0 — see
  `LICENSE-DATA`.

## Citation

See `CITATION.cff` (author list confirmed: Vinícius Nunes da Rocha,
corresponding author; Carlos Maurício Rabello de Sant'Anna). **Note:** DOI,
release date, and the GitHub URL in that file remain placeholders until the
article is accepted and this repository is archived on Zenodo.
