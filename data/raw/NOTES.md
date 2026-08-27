# data/raw/ — notes

## Prospective screening library (713 compounds) — anonymized

`prospective_library_713_descriptors_raw.xlsx` and
`prospective_library_713_labels_anonymized.xlsx` cover the 713-compound
prospective screening library ("Triagem JUNO") scored by the version-G
champions in `code/pipeline_G_juno_screening.py`.

Unlike the 320-ligand training set (drawn entirely from published
literature), this library was synthesized in-house and has no experimental
validation yet against the GluN1--GluN2B/NMDA target used in this article.
Publishing the real compound identity (SMILES/name) alongside this model's
Active/Inactive prediction for this specific target would constitute public
disclosure of exactly the kind of information that could support a future
new-use patent claim -- and, unlike in the US, most jurisdictions offer no
grace period, so that disclosure could pre-empt patentability outright.

To keep the pipeline fully reproducible without that risk, **SMILES and
compound names have been removed** from both files and replaced with a
sequential anonymous `compound_id` (0..712, same row order in both files and
in `results/juno_screening_713library_ranked_G.csv`). Descriptors and
predicted labels are otherwise unmodified. If you hold the original,
non-anonymized compound list, `compound_id` lets you re-link rows to real
structures yourself; this repository does not make that link public.

## Redocking population data

`redocking_rmsd_full_population_5systems.csv` and
`redocking_rmsd_summary_5systems.csv` are the pre-extracted RMSD statistics
for all five redocking reference systems (Table 1 / Fig. G18), provided as a
ready-to-use alternative to re-deriving them from the raw, per-system GOLD
population spreadsheets (not included here — see
`code/redocking_analysis.py`).

`tabela_contatos_nativos_vs_redocados_FINAL_G.csv` and
`..._detalhe_G.csv` are the corresponding native-contact-preservation tables
(see `code/native_contacts_analysis.py`).
