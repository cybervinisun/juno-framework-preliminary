"""
Version-G pipeline, part F: structural reassessment of the classification
errors (Tanimoto/MCS) against the REAL misclassifications of the version-G
champions, so that the analysis matches the models actually reported in
Tables 2/4/5 of Article 1 rather than an earlier pipeline's error set.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

RDLogger.DisableLog("rdApp.*")

import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "processed"))

X_PATH = DATA_DIR / "X_320ligands_57descriptors.xlsx"
Y_PATH = DATA_DIR / "y_320ligands_labels.xlsx"
SMILES_PATH = DATA_DIR / "smiles_320ligands_reference.xlsx"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models"))
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "results" / "regenerated"))
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================================
# 1. Rebuild the exact split (random_state=42), preserving the ORIGINAL
#    index (1..320) so each compound's real SMILES can be recovered.
# ====================================================================
df4 = pd.read_excel(X_PATH, index_col=0)
y = pd.read_excel(Y_PATH, index_col=0)
smiles_df = pd.read_excel(SMILES_PATH, index_col=0)

assert (df4.index == y.index).all()
assert (df4.index == smiles_df.index).all()

X_final = df4.copy()
y_final = y.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.30, stratify=y_final["Activity"], random_state=42,
)
# Do NOT reset the index this time -- the original ID (1..320) is needed to
# recover the SMILES of each test-set compound.

PRE_NORMALISED_DESCRIPTORS = ["corrScore"]
binary_cols = [c for c in X_train.columns if set(X_train[c].dropna().unique()) <= {0, 1}]
continuous_cols = [c for c in X_train.columns if c not in binary_cols and c not in PRE_NORMALISED_DESCRIPTORS]

scaler = joblib.load(MODEL_DIR / "minmax_scaler_fitted_on_training.pkl")
X_test_scaled = X_test.copy()
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

LABEL_MAP = {"Inactive": 0, "Active": 1}
y_test_bin = pd.Series([LABEL_MAP[c] for c in y_test["Activity"]], index=y_test.index)

# ====================================================================
# 2. Predictions of the 3 version-G champions on the test set (same order,
#    original index preserved)
# ====================================================================
print("=" * 70)
print("Identifying the REAL test-set errors of the version-G champions")
print("=" * 70)

errors_by_model = {}
for label in ["MLP", "SVM", "XGBoost"]:
    champion = joblib.load(MODEL_DIR / f"champion_{label}.pkl")
    y_pred = champion.predict(X_test_scaled)
    y_pred_series = pd.Series(y_pred, index=X_test_scaled.index)

    fp_idx = y_pred_series[(y_pred_series == 1) & (y_test_bin == 0)].index.tolist()
    fn_idx = y_pred_series[(y_pred_series == 0) & (y_test_bin == 1)].index.tolist()
    errors_by_model[label] = {"FP": fp_idx, "FN": fn_idx}
    print(f"{label}: {len(fp_idx)} false positives {fp_idx}, {len(fn_idx)} false negatives {fn_idx}")

all_error_ids = set()
for label, d in errors_by_model.items():
    all_error_ids.update(d["FP"])
    all_error_ids.update(d["FN"])
all_error_ids = sorted(all_error_ids)
print(f"\nUnion of all compounds misclassified by at least 1 model: {all_error_ids}")

shared_errors = set(errors_by_model["MLP"]["FP"] + errors_by_model["MLP"]["FN"])
for label in ["SVM", "XGBoost"]:
    shared_errors &= set(errors_by_model[label]["FP"] + errors_by_model[label]["FN"])
print(f"Compounds misclassified by all 3 models simultaneously: {sorted(shared_errors)}")

# ====================================================================
# 3. Morgan fingerprints (radius 2, 2048 bits) for the whole training set
#    (Active/Inactive) plus the misclassified compounds
# ====================================================================
def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

smiles_df["fp"] = smiles_df["Smiles"].apply(smiles_to_fp)
n_fingerprint_failures = smiles_df["fp"].isna().sum()
print(f"\nSMILES that failed to yield a fingerprint: {n_fingerprint_failures}/320")

train_ids = X_train.index
active_ids = [i for i in train_ids if smiles_df.loc[i, "Activity"] == "Active" and smiles_df.loc[i, "fp"] is not None]
inactive_ids = [i for i in train_ids if smiles_df.loc[i, "Activity"] == "Inactive" and smiles_df.loc[i, "fp"] is not None]
print(f"Reference population (training): {len(active_ids)} Active, {len(inactive_ids)} Inactive")

result_rows = []
for idx in all_error_ids:
    fp_error_compound = smiles_df.loc[idx, "fp"]
    if fp_error_compound is None:
        print(f"WARNING: compound {idx} has no valid fingerprint -- skipped.")
        continue

    sims_active = DataStructs.BulkTanimotoSimilarity(fp_error_compound, [smiles_df.loc[i, "fp"] for i in active_ids])
    sims_inactive = DataStructs.BulkTanimotoSimilarity(fp_error_compound, [smiles_df.loc[i, "fp"] for i in inactive_ids])

    misclassifying_models = [label for label, d in errors_by_model.items() if idx in d["FP"] or idx in d["FN"]]
    error_type_by_model = {
        label: ("FP" if idx in errors_by_model[label]["FP"] else "FN")
        for label in misclassifying_models
    }

    result_rows.append({
        "compound_idx": idx,
        "true_activity": smiles_df.loc[idx, "Activity"],
        "misclassified_by": ", ".join(misclassifying_models),
        "error_type_by_model": str(error_type_by_model),
        "median_tanimoto_active": float(np.median(sims_active)),
        "max_tanimoto_active": float(np.max(sims_active)),
        "median_tanimoto_inactive": float(np.median(sims_inactive)),
        "max_tanimoto_inactive": float(np.max(sims_inactive)),
        "smiles": smiles_df.loc[idx, "Smiles"],
    })

results_df = pd.DataFrame(result_rows).sort_values("compound_idx")
results_df.to_csv(OUT_DIR / "error_compounds_tanimoto_to_training.csv", index=False)
print(f"\n[table saved] {OUT_DIR / 'error_compounds_tanimoto_to_training.csv'}")
print(results_df[["compound_idx", "true_activity", "misclassified_by", "median_tanimoto_active", "max_tanimoto_active", "median_tanimoto_inactive", "max_tanimoto_inactive"]].to_string(index=False))

# ====================================================================
# 4. Pairwise similarity among the misclassified compounds themselves
# ====================================================================
print("\nPairwise similarity among the misclassified compounds:")
pair_rows = []
valid_error_ids = [idx for idx in all_error_ids if smiles_df.loc[idx, "fp"] is not None]
for a, b in combinations(valid_error_ids, 2):
    sim = DataStructs.TanimotoSimilarity(smiles_df.loc[a, "fp"], smiles_df.loc[b, "fp"])
    pair_rows.append({"compound_a": a, "compound_b": b, "tanimoto": sim})
    print(f"  {a} x {b}: {sim:.3f}")
pairs_df = pd.DataFrame(pair_rows)
pairs_df.to_csv(OUT_DIR / "error_compounds_pairwise_tanimoto.csv", index=False)

# ====================================================================
# 5. Figure: Tanimoto distribution of each misclassified compound against
#    the Active/Inactive training populations
# ====================================================================
n_err = len(valid_error_ids)
fig, axes = plt.subplots(1, n_err, figsize=(4.5 * n_err, 4.5), dpi=150, sharey=True)
if n_err == 1:
    axes = [axes]

for ax, idx in zip(axes, valid_error_ids):
    fp_error_compound = smiles_df.loc[idx, "fp"]
    sims_active = DataStructs.BulkTanimotoSimilarity(fp_error_compound, [smiles_df.loc[i, "fp"] for i in active_ids])
    sims_inactive = DataStructs.BulkTanimotoSimilarity(fp_error_compound, [smiles_df.loc[i, "fp"] for i in inactive_ids])
    ax.hist(sims_active, bins=20, alpha=0.6, color="#d62728", label="to Active", density=True)
    ax.hist(sims_inactive, bins=20, alpha=0.6, color="#1f77b4", label="to Inactive", density=True)
    model_names = [label for label, d in errors_by_model.items() if idx in d["FP"] or idx in d["FN"]]
    ax.set_title(f"Compound {idx} (true {smiles_df.loc[idx,'Activity']})\nmisclassified by: {', '.join(model_names)}", fontsize=9)
    ax.set_xlabel("Tanimoto similarity")
    if ax is axes[0]:
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

fig.suptitle("Tanimoto similarity of version-G misclassified test-set compounds\nto the training Active/Inactive populations", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "error_compounds_tanimoto.png", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {FIG_DIR / 'error_compounds_tanimoto.png'}")

print()
print("=" * 70)
print("COMPLETE: Tanimoto/MCS reassessment of the errors -- version G")
print("=" * 70)
