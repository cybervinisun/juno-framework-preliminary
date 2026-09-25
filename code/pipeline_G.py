"""
Version-G pipeline -- faithful reconstruction of the reference notebook
(Modelo_Classificacao_ML_QSAR_320_ligantes_GoldsScore_v_Tese_RECONSTRUIDO)
run over the real 320-ligand dataset, for Article 1 (Journal of
Cheminformatics).

Reproduces, with no change in logic, exactly what was confirmed
cell-by-cell in the original notebook:
  - Stratified 70/30 split (random_state=42)
  - Min-max normalisation (fitted on the training partition, corrScore excluded)
  - Diagnostic PCA(0.999) over the 9 continuous descriptors (it does NOT
    enter the modelling matrix -- it is study/figure material only)
  - SVMSMOTE parts A/B: mixed Gower-like distance (Manhattan on the
    continuous block + symmetric Hamming on the binary block, 50/50),
    raw candidate generation via SVMSMOTE, formal correction (reflection
    of continuous values outside [0,1], snapping of the ordinal pKa code,
    binary-bit assignment by point-biserial correlation with a
    noise_budget/temperature tolerance), and parent-balanced selection
    within the 0.83-0.90 similarity band
  - Bayesian optimisation (skopt.BayesSearchCV) with StratifiedGroupKFold
    (parent-child groups) and scoring = Cohen's kappa (kappa_scorer), for
    MLP, XGBoost and SVM -- confirmed as the ONLY real criterion for
    hyperparameter/champion selection in every effective call of the
    notebook (MCC is imported but never called; it appears only as an
    optional extra column in the repeated-CV report).
  - Champion selection via best_estimator_/best_score_ (internal CV),
    NEVER by a test-set metric.
  - Training/test metrics + a repeated-CV diagnostic (20x5) of the
    already-selected candidate.

Scope of this reconstruction: part A of the notebook (bootstrap of the
Active-Inactive/Inactive-Inactive similarity thresholds) is reproduced
only in its operational conclusion -- the notebook uses the fixed
thresholds 0.83 and 0.90 and never reads its bootstrap estimates back
into the part-B code, where they are printed as a diagnostic only. The
full bootstrap (coverage plots/tables) is therefore out of scope for
this script, because it does not affect the final model.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    make_scorer,
    pairwise_distances,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from xgboost import XGBClassifier
from imblearn.over_sampling import SVMSMOTE
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

warnings.filterwarnings("ignore")

# ====================================================================
# 0. Paths and output
#
# DATA_DIR can be overridden through an environment variable, e.g.:
#   DATA_DIR=/some/other/path python pipeline_G.py
# By default it points at data/processed/ in the repository root.
# ====================================================================
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "processed"))

X_PATH = DATA_DIR / "X_320ligands_57descriptors.xlsx"
Y_PATH = DATA_DIR / "y_320ligands_labels.xlsx"

OUT_DIR = Path(__file__).parent / "version_G_outputs"
OUT_DIR.mkdir(exist_ok=True)


def save_table(df: pd.DataFrame, nome: str) -> None:
    path = OUT_DIR / nome
    df.to_csv(path, index=False)
    print(f"[table saved] {path}")


# ====================================================================
# 1. Loading and split (notebook cells 91/93/94)
# ====================================================================
print("=" * 70)
print("1. Loading and stratified 70/30 split")
print("=" * 70)

df4 = pd.read_excel(X_PATH, index_col=0)
y = pd.read_excel(Y_PATH, index_col=0)

assert (df4.index == y.index).all(), "X and y are misaligned by index."

X_final = df4.copy()
y_final = y.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.30,
    stratify=y_final["Activity"],
    random_state=42,
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print("Training distribution:\n", y_train["Activity"].value_counts())
print("Test distribution:\n", y_test["Activity"].value_counts())

# ====================================================================
# 2. Min-max normalisation (notebook cell 96) -- fitted on train, corrScore excluded
# ====================================================================
print()
print("=" * 70)
print("2. Min-max normalisation (fitted on the training partition)")
print("=" * 70)

PRE_NORMALISED_DESCRIPTORS = ["corrScore"]

binary_cols_for_scaler = [
    col for col in X_train.columns
    if set(X_train[col].dropna().unique()) <= {0, 1}
]
continuous_cols_for_scaler = [
    col for col in X_train.columns
    if col not in binary_cols_for_scaler and col not in PRE_NORMALISED_DESCRIPTORS
]

scaler = MinMaxScaler()
scaler.fit(X_train[continuous_cols_for_scaler])

X_train[continuous_cols_for_scaler] = scaler.transform(X_train[continuous_cols_for_scaler])
X_test[continuous_cols_for_scaler] = scaler.transform(X_test[continuous_cols_for_scaler])

print(f"Continuous normalised: {len(continuous_cols_for_scaler)} | "
      f"Binary preserved: {len(binary_cols_for_scaler)} | "
      f"Already normalised, preserved: {PRE_NORMALISED_DESCRIPTORS}")

joblib.dump(scaler, OUT_DIR / "scaler_minmax_train_G.pkl")

# ====================================================================
# 3. Diagnostic PCA (notebook cells 56-57) -- 9 continuous descriptors;
#    does NOT enter the model, study/figure material only
# ====================================================================
print()
print("=" * 70)
print("3. Diagnostic PCA over the continuous descriptors (0.999 var.)")
print("=" * 70)

continuous_cols_pca = [c for c in X_train.columns if c not in binary_cols_for_scaler]
pca = PCA(n_components=0.999, svd_solver="full")
pca_scores = pca.fit_transform(X_train[continuous_cols_pca])

print(f"Continuous descriptors used in the PCA: {len(continuous_cols_pca)}")
print(f"Components retained (0.999 of the variance): {pca.n_components_}")
print(f"Explained variance per component: {np.round(pca.explained_variance_ratio_, 4)}")
print(f"Cumulative variance: {np.round(np.cumsum(pca.explained_variance_ratio_), 4)}")

loadings = pd.DataFrame(
    pca.components_.T,
    index=continuous_cols_pca,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
)
save_table(loadings.reset_index().rename(columns={"index": "descriptor"}), "table_pca_loadings_G.csv")

pca_var_df = pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(pca.n_components_)],
    "explained_variance": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
})
save_table(pca_var_df, "table_pca_explained_variance_G.csv")

# ====================================================================
# 4. SVMSMOTE parts A/B (notebook cells 103-106) -- faithful reconstruction
# ====================================================================
print()
print("=" * 70)
print("4. SVMSMOTE parts A/B -- balanced generation of synthetic Inactive samples")
print("=" * 70)


def infer_binary_columns(df: pd.DataFrame) -> list[str]:
    binary_cols = []
    for col in df.columns:
        values = df[col].dropna().unique()
        if len(values) > 0 and set(values).issubset({0, 1, False, True}):
            binary_cols.append(col)
    return binary_cols


def prepare_mixed_distance_columns(X, binary_cols=None):
    if binary_cols is None:
        binary_cols = infer_binary_columns(X)
    numeric_cols = [
        col for col in X.select_dtypes(include=[np.number]).columns
        if col not in binary_cols
    ]
    if not binary_cols and not numeric_cols:
        raise ValueError("No numeric or binary column was found.")
    return numeric_cols, binary_cols


def mixed_gower_like_distance_symmetric_binary(
    X_left, X_right, numeric_cols, binary_cols, numeric_min, numeric_range,
    numeric_weight=0.5, binary_weight=0.5,
):
    distance_parts = []
    weights = []
    if numeric_cols:
        Xn_left = (X_left[numeric_cols] - numeric_min) / numeric_range
        Xn_right = (X_right[numeric_cols] - numeric_min) / numeric_range
        Xn_left = Xn_left.fillna(0.0).to_numpy(dtype=float)
        Xn_right = Xn_right.fillna(0.0).to_numpy(dtype=float)
        numeric_dist = pairwise_distances(Xn_left, Xn_right, metric="manhattan") / len(numeric_cols)
        distance_parts.append(numeric_dist)
        weights.append(numeric_weight)
    if binary_cols:
        Xb_left = X_left[binary_cols].fillna(0).to_numpy(dtype=np.uint8)
        Xb_right = X_right[binary_cols].fillna(0).to_numpy(dtype=np.uint8)
        binary_dist = pairwise_distances(Xb_left, Xb_right, metric="hamming")
        distance_parts.append(binary_dist)
        weights.append(binary_weight)
    weights_array = np.array(weights, dtype=float)
    weights_array = weights_array / weights_array.sum()
    mixed_dist = np.zeros_like(distance_parts[0], dtype=float)
    for dist, weight in zip(distance_parts, weights_array):
        mixed_dist += weight * dist
    return mixed_dist


def compute_point_biserial_matrix(X_min_orig, binary_cols, numeric_cols):
    r_matrix = pd.DataFrame(index=binary_cols, columns=numeric_cols, dtype=float)
    for j in binary_cols:
        b = X_min_orig[j].to_numpy(dtype=float)
        if b.std() == 0:
            r_matrix.loc[j, :] = 0.0
            continue
        for k in numeric_cols:
            c = X_min_orig[k].to_numpy(dtype=float)
            if c.std() == 0:
                r_matrix.loc[j, k] = 0.0
                continue
            r, _ = pointbiserialr(b, c)
            r_matrix.loc[j, k] = 0.0 if np.isnan(r) else r
    return r_matrix


def enforce_formal_binary_assignment(
    X_synth_raw, X_min_orig, binary_cols, numeric_cols, r_matrix,
    noise_budget, temperature, random_state,
):
    X_out = X_synth_raw.copy()
    mean_k = X_min_orig[numeric_cols].mean()
    std_k = X_min_orig[numeric_cols].std().replace(0, 1e-8)
    Z = (X_out[numeric_cols] - mean_k) / std_k
    rng = np.random.default_rng(random_state)
    flips_per_column = {}
    for j in binary_cols:
        b_raw = X_out[j].to_numpy(dtype=float)
        b_binarised = (b_raw >= 0.5).astype(float)
        r_j = r_matrix.loc[j].to_numpy(dtype=float)
        bit_without_evidence = np.allclose(r_j, 0.0) and X_min_orig[j].std() == 0
        if bit_without_evidence:
            X_out[j] = b_binarised
            flips_per_column[j] = 0
            continue
        score = (Z.to_numpy() * r_j).sum(axis=1) * temperature
        p1 = 1.0 / (1.0 + np.exp(-score))
        disagree = np.abs(p1 - b_binarised)
        correction_probability = noise_budget * disagree
        draw = rng.random(len(X_out))
        to_correct = draw < correction_probability
        b_corrected = b_binarised.copy()
        b_corrected[to_correct] = 1 - b_binarised[to_correct]
        X_out[j] = b_corrected
        flips_per_column[j] = int(to_correct.sum())
    return X_out, flips_per_column


def enforce_continuous_bounds_by_reflection(X_synth_raw, numeric_cols, lower_bound=0.0, upper_bound=1.0):
    X_out = X_synth_raw.copy()
    width = upper_bound - lower_bound
    period = 2 * width
    n_corrected_per_column = {}
    for col in numeric_cols:
        values_array = X_out[col].to_numpy(dtype=float)
        out_of_bounds = (values_array < lower_bound) | (values_array > upper_bound)
        shifted = (values_array - lower_bound) % period
        reflected = np.where(shifted > width, period - shifted, shifted)
        X_out[col] = lower_bound + reflected
        n_corrected_per_column[col] = int(out_of_bounds.sum())
    return X_out, n_corrected_per_column


def enforce_ordinal_pka_category(X_synth_raw, X_orig, ordinal_pka_cols):
    X_out = X_synth_raw.copy()
    for col in ordinal_pka_cols:
        valid_categories = np.sort(X_orig[col].unique())
        values_array = X_out[col].to_numpy(dtype=float)
        nearest_category_idx = np.abs(values_array[:, None] - valid_categories[None, :]).argmin(axis=1)
        X_out[col] = valid_categories[nearest_category_idx]
    return X_out


def allocate_evenly(total, n_groups, rng):
    if total < 0:
        raise ValueError("total must be non-negative.")
    if n_groups <= 0:
        raise ValueError("n_groups must be positive.")
    quotas = np.full(n_groups, total // n_groups, dtype=int)
    remainder = total % n_groups
    if remainder > 0:
        selected = rng.permutation(n_groups)[:remainder]
        quotas[selected] += 1
    return quotas


def bin_quotas_for_mother(mother_index, mother_quota, n_bins):
    return allocate_evenly(total=int(mother_quota), n_groups=n_bins, rng=np.random.default_rng(mother_index))


def make_similarity_bins(similarity, similarity_min, similarity_max, n_bins):
    bin_edges = np.linspace(similarity_min, similarity_max, n_bins + 1)
    labels = [f"bin_{idx + 1}" for idx in range(n_bins)]
    return pd.cut(similarity, bins=bin_edges, labels=labels, include_lowest=True, right=False)


def assign_similarity_bins_with_clipping(similarity, similarity_min, similarity_max, n_bins):
    eps = np.finfo(float).eps
    clipped_similarity = np.clip(similarity, similarity_min, similarity_max - eps)
    sim_bins = make_similarity_bins(clipped_similarity, similarity_min, similarity_max, n_bins)
    return np.asarray(sim_bins, dtype=object)


def generate_svmsmote_candidates(X_orig, y_orig, minority_class, target_minority_count, random_state, k_neighbors):
    sampler = SVMSMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
        sampling_strategy={minority_class: target_minority_count},
    )
    X_res, _ = sampler.fit_resample(X_orig, y_orig)
    if len(X_res) <= len(X_orig):
        return pd.DataFrame(columns=X_orig.columns)
    return X_res.iloc[len(X_orig):].reset_index(drop=True)


def score_svmsmote_candidates(
    candidates, X_min_orig, numeric_cols, binary_cols, numeric_min, numeric_range,
    ref_max, similarity_min, similarity_max, n_bins,
):
    if candidates.empty:
        metadata = pd.DataFrame(columns=[
            "candidate_id", "mother_index", "similarity", "bin",
            "inside_interval", "interval_deviation", "rescue_assignment",
        ])
        return candidates, metadata

    dist_mat = mixed_gower_like_distance_symmetric_binary(
        candidates, X_min_orig, numeric_cols, binary_cols, numeric_min, numeric_range,
    )
    nearest_mother = dist_mat.argmin(axis=1)
    dist_min = dist_mat.min(axis=1)
    similarity = np.clip(1.0 - (dist_min / ref_max), 0.0, 1.0)
    sim_bins = make_similarity_bins(similarity, similarity_min, similarity_max, n_bins)
    sim_bins_array = np.asarray(sim_bins, dtype=object)
    inside_interval = (similarity >= similarity_min) & (similarity < similarity_max)
    interval_deviation = np.where(
        similarity < similarity_min, similarity_min - similarity,
        np.where(similarity >= similarity_max, similarity - similarity_max, 0.0),
    )
    fallback_bins_array = assign_similarity_bins_with_clipping(similarity, similarity_min, similarity_max, n_bins)
    final_bins_array = np.where(pd.notna(sim_bins_array), sim_bins_array, fallback_bins_array)

    candidates_valid = candidates.reset_index(drop=True)
    metadata = pd.DataFrame({
        "candidate_id": np.arange(len(candidates_valid)),
        "mother_index": nearest_mother,
        "similarity": similarity,
        "bin": final_bins_array,
        "inside_interval": inside_interval,
        "interval_deviation": interval_deviation,
        "rescue_assignment": False,
    })
    return candidates_valid, metadata


def has_minimum_pool_coverage(metadata, mother_quotas, n_bins, strict_bin_coverage):
    if metadata.empty:
        return False
    if "inside_interval" in metadata.columns:
        metadata = metadata[metadata["inside_interval"]].copy()
    if metadata.empty:
        return False
    counts_by_mother = metadata["mother_index"].value_counts()
    for mother_index, mother_quota in enumerate(mother_quotas):
        if counts_by_mother.get(mother_index, 0) < mother_quota:
            return False
    if not strict_bin_coverage:
        return True
    for mother_index, mother_quota in enumerate(mother_quotas):
        bin_quotas = bin_quotas_for_mother(mother_index, mother_quota, n_bins)
        mother_metadata = metadata[metadata["mother_index"] == mother_index]
        counts_by_bin = mother_metadata["bin"].value_counts()
        for bin_index, bin_quota in enumerate(bin_quotas):
            bin_label = f"bin_{bin_index + 1}"
            if counts_by_bin.get(bin_label, 0) < bin_quota:
                return False
    return True


def add_secondary_mother_assignments(
    candidates, metadata, X_min_orig, mother_quotas, numeric_cols, binary_cols,
    numeric_min, numeric_range, ref_max, similarity_min, similarity_max, n_bins,
):
    counts_by_mother = metadata["mother_index"].value_counts()
    missing_mothers = [
        mother_index for mother_index, mother_quota in enumerate(mother_quotas)
        if counts_by_mother.get(mother_index, 0) < mother_quota
    ]
    if not missing_mothers:
        return metadata

    rescue_frames = []
    candidate_global_ids = np.arange(len(candidates))
    for mother_index in missing_mothers:
        mother_df = X_min_orig.iloc[[mother_index]].reset_index(drop=True)
        dist_to_mother = mixed_gower_like_distance_symmetric_binary(
            candidates, mother_df, numeric_cols, binary_cols, numeric_min, numeric_range,
        ).ravel()
        similarity = np.clip(1.0 - (dist_to_mother / ref_max), 0.0, 1.0)
        inside_interval = (similarity >= similarity_min) & (similarity < similarity_max)
        interval_deviation = np.where(
            similarity < similarity_min, similarity_min - similarity,
            np.where(similarity >= similarity_max, similarity - similarity_max, 0.0),
        )
        sim_bins_array = assign_similarity_bins_with_clipping(similarity, similarity_min, similarity_max, n_bins)
        rescue_frame = pd.DataFrame({
            "candidate_id": candidate_global_ids,
            "mother_index": mother_index,
            "similarity": similarity,
            "bin": sim_bins_array,
            "inside_interval": inside_interval,
            "interval_deviation": interval_deviation,
            "candidate_global_id": candidate_global_ids,
            "rescue_assignment": True,
        })
        rescue_frames.append(rescue_frame)

    if not rescue_frames:
        return metadata

    enriched_metadata = pd.concat([metadata, *rescue_frames], ignore_index=True)
    enriched_metadata = enriched_metadata.sort_values(
        ["rescue_assignment", "interval_deviation", "similarity"], ascending=[True, True, False],
    )
    enriched_metadata = enriched_metadata.drop_duplicates(
        subset=["candidate_global_id", "mother_index"], keep="first",
    ).reset_index(drop=True)
    return enriched_metadata


def select_balanced_pool(candidates, metadata, mother_quotas, n_bins, similarity_min, similarity_max):
    selected_indices = []
    selected_metadata = []
    used_candidate_ids = set()

    bin_edges = np.linspace(similarity_min, similarity_max, n_bins + 1)
    bin_centers = {f"bin_{idx + 1}": (bin_edges[idx] + bin_edges[idx + 1]) / 2 for idx in range(n_bins)}

    mother_order = sorted(range(len(mother_quotas)), key=lambda idx: len(metadata[metadata["mother_index"] == idx]))

    for mother_index in mother_order:
        mother_quota = mother_quotas[mother_index]
        mother_metadata = metadata[metadata["mother_index"] == mother_index].copy()
        mother_metadata = mother_metadata[~mother_metadata["candidate_global_id"].isin(used_candidate_ids)].copy()

        if len(mother_metadata) < mother_quota:
            raise RuntimeError(f"Mother {mother_index} has only {len(mother_metadata)} valid candidates for quota {mother_quota}.")

        bin_quotas = bin_quotas_for_mother(mother_index, mother_quota, n_bins)
        mother_selected = []
        mother_selected_metadata = []

        for bin_index, bin_quota in enumerate(bin_quotas):
            if bin_quota == 0:
                continue
            bin_label = f"bin_{bin_index + 1}"
            bin_metadata = mother_metadata[mother_metadata["bin"] == bin_label].copy()
            bin_metadata = bin_metadata[~bin_metadata["candidate_global_id"].isin(used_candidate_ids)].copy()
            if bin_metadata.empty:
                continue
            center = bin_centers[bin_label]
            bin_metadata["center_distance"] = (bin_metadata["similarity"] - center).abs()
            bin_metadata = bin_metadata.sort_values(
                ["inside_interval", "interval_deviation", "center_distance", "similarity"],
                ascending=[False, True, True, True],
            )
            chosen = bin_metadata.head(int(bin_quota))
            mother_selected.extend(chosen["candidate_global_id"].tolist())
            used_candidate_ids.update(chosen["candidate_global_id"].tolist())
            mother_selected_metadata.append(chosen.drop(columns=["center_distance"]))

        still_needed = int(mother_quota) - len(mother_selected)
        if still_needed > 0:
            already_selected = set(mother_selected)
            remaining = mother_metadata[~mother_metadata["candidate_global_id"].isin(already_selected)].copy()
            remaining = remaining[~remaining["candidate_global_id"].isin(used_candidate_ids)].copy()
            if len(remaining) < still_needed:
                raise RuntimeError(f"Mother {mother_index} cannot fill quota after bin selection.")
            local_bin_counts = (
                pd.concat(mother_selected_metadata, ignore_index=True)["bin"].value_counts().to_dict()
                if mother_selected_metadata else {}
            )
            remaining["bin_count"] = remaining["bin"].map(lambda value: local_bin_counts.get(value, 0))
            remaining["distance_to_interval_center"] = (remaining["similarity"] - ((similarity_min + similarity_max) / 2)).abs()
            remaining = remaining.sort_values(
                ["inside_interval", "interval_deviation", "bin_count", "distance_to_interval_center"],
                ascending=[False, True, True, True],
            )
            chosen_extra = remaining.head(still_needed)
            mother_selected.extend(chosen_extra["candidate_global_id"].tolist())
            used_candidate_ids.update(chosen_extra["candidate_global_id"].tolist())
            mother_selected_metadata.append(chosen_extra.drop(columns=["bin_count", "distance_to_interval_center"]))

        selected_indices.extend(mother_selected)
        selected_metadata.append(pd.concat(mother_selected_metadata, ignore_index=True))

    final_metadata = pd.concat(selected_metadata, ignore_index=True)
    final_candidates = candidates.iloc[selected_indices].reset_index(drop=True)
    final_metadata = final_metadata.reset_index(drop=True)
    return final_candidates, final_metadata


def build_tracking_table(X_orig, y_orig, X_synth_final, synthetic_metadata, minority_class, original_source_index):
    original_tracking = pd.DataFrame({
        "final_row_id": np.arange(len(X_orig)),
        "sample_id": [f"orig_{idx}" for idx in range(len(X_orig))],
        "is_synthetic": False,
        "synthetic_id": pd.NA,
        "class_label": y_orig.to_numpy(),
        "original_row_id": np.arange(len(X_orig)),
        "original_source_index": original_source_index,
        "mother_index": pd.NA,
        "mother_original_row_id": pd.NA,
        "mother_source_index": pd.NA,
        "similarity_to_mother": np.nan,
        "similarity_bin": pd.NA,
        "inside_similarity_interval": pd.NA,
        "interval_deviation": np.nan,
        "rescue_assignment": pd.NA,
        "candidate_global_id": pd.NA,
    })

    synthetic_ids = [f"synth_inactive_{idx:05d}" for idx in range(len(X_synth_final))]
    minority_positions_in_X_orig = np.where(y_orig.to_numpy() == minority_class)[0]
    mother_index_in_minority = synthetic_metadata["mother_index"].to_numpy(dtype=int)
    mother_original_row_ids = minority_positions_in_X_orig[mother_index_in_minority]
    mother_source_indices = original_source_index[mother_original_row_ids]

    synthetic_tracking = pd.DataFrame({
        "final_row_id": np.arange(len(X_orig), len(X_orig) + len(X_synth_final)),
        "sample_id": synthetic_ids,
        "is_synthetic": True,
        "synthetic_id": synthetic_ids,
        "class_label": minority_class,
        "original_row_id": pd.NA,
        "original_source_index": pd.NA,
        "mother_index": mother_index_in_minority,
        "mother_original_row_id": mother_original_row_ids,
        "mother_source_index": mother_source_indices,
        "similarity_to_mother": synthetic_metadata["similarity"].to_numpy(float),
        "similarity_bin": synthetic_metadata["bin"].to_numpy(),
        "inside_similarity_interval": synthetic_metadata["inside_interval"].to_numpy(bool),
        "interval_deviation": synthetic_metadata["interval_deviation"].to_numpy(float),
        "rescue_assignment": synthetic_metadata["rescue_assignment"].to_numpy(bool),
        "candidate_global_id": synthetic_metadata["candidate_global_id"].to_numpy(int),
    })
    return pd.concat([original_tracking, synthetic_tracking], ignore_index=True)


# ---- Main parameters (identical to the notebook, cell 106) ----
SIMILARITY_MIN = 0.83
SIMILARITY_MAX = 0.90
N_SIMILARITY_BINS = 6
max_iter = 1000
pool_overshoot_factor = 4
strict_bin_coverage = False
RANDOM_STATE_BASE = 479
K_NEIGHBORS_BASE = 10
noise_budget = 0.05
temperature = 3.0

X_orig = X_train.copy().reset_index(drop=True)
y_orig = y_train["Activity"].copy().reset_index(drop=True)
original_source_index = np.arange(len(X_orig))

minority_class = y_orig.value_counts().idxmin()
majority_class = y_orig.value_counts().idxmax()

X_min_orig = X_orig[y_orig == minority_class].reset_index(drop=True)
n_min = len(X_min_orig)
n_maj = int((y_orig == majority_class).sum())
n_target = n_maj
n_needed = n_target - n_min

print(f"Minority class: {minority_class} (n={n_min}) | Majority class: {majority_class} (n={n_maj})")
print(f"Synthetic samples needed for a fully balanced partition: {n_needed}")

numeric_cols, binary_cols = prepare_mixed_distance_columns(X_orig)
numeric_min = X_orig[numeric_cols].min() if numeric_cols else pd.Series(dtype=float)
numeric_range = (X_orig[numeric_cols].max() - X_orig[numeric_cols].min()) if numeric_cols else pd.Series(dtype=float)
numeric_range = numeric_range.replace(0, 1.0)

ordinal_pka_cols = [c for c in X_orig.columns if "pka" in c.lower()]
r_matrix = compute_point_biserial_matrix(X_min_orig, binary_cols, numeric_cols)

distance_matrix_symmetric = mixed_gower_like_distance_symmetric_binary(
    X_orig, X_orig, numeric_cols=numeric_cols, binary_cols=binary_cols,
    numeric_min=numeric_min, numeric_range=numeric_range, numeric_weight=0.5, binary_weight=0.5,
)
unified_ref_max = float(distance_matrix_symmetric.max())
if unified_ref_max <= 0:
    raise ValueError("The unified reference distance must be greater than zero.")

print(f"Continuous numeric columns: {len(numeric_cols)} | Binary columns: {len(binary_cols)}")
print(f"Symmetric unified_ref_max: {unified_ref_max:.6f}")

rng = np.random.default_rng(RANDOM_STATE_BASE)

if n_needed <= 0:
    X_train_final = X_orig.copy()
    y_train_final = y_orig.copy()
    synthetic_metadata = pd.DataFrame(columns=["mother_index", "similarity", "bin"])
    tracking_table = build_tracking_table(
        X_orig, y_orig, pd.DataFrame(columns=X_orig.columns), synthetic_metadata, minority_class, original_source_index,
    )
    print("The minority class is already balanced with, or above, the majority class.")
else:
    if n_min < 2:
        raise ValueError("SVMSMOTE needs at least 2 minority samples.")

    k_neighbors = min(K_NEIGHBORS_BASE, n_min - 1)
    mother_quotas = allocate_evenly(total=n_needed, n_groups=n_min, rng=rng)

    candidate_frames = []
    metadata_frames = []
    next_candidate_global_id = 0
    minimum_pool_size = n_needed * pool_overshoot_factor
    flips_cumulative = {j: 0 for j in binary_cols}
    reflections_cumulative = {j: 0 for j in numeric_cols}

    for iteration in range(max_iter):
        X_synth_raw = generate_svmsmote_candidates(
            X_orig=X_orig, y_orig=y_orig, minority_class=minority_class,
            target_minority_count=n_target, random_state=RANDOM_STATE_BASE + iteration, k_neighbors=k_neighbors,
        )
        X_synth, reflections_this_iteration = enforce_continuous_bounds_by_reflection(X_synth_raw, numeric_cols)
        for col, n in reflections_this_iteration.items():
            reflections_cumulative[col] += n

        X_synth = enforce_ordinal_pka_category(X_synth, X_orig, ordinal_pka_cols)
        X_synth, flips_this_iteration = enforce_formal_binary_assignment(
            X_synth_raw=X_synth, X_min_orig=X_min_orig, binary_cols=binary_cols, numeric_cols=numeric_cols,
            r_matrix=r_matrix, noise_budget=noise_budget, temperature=temperature,
            random_state=RANDOM_STATE_BASE + iteration,
        )
        for col, n in flips_this_iteration.items():
            flips_cumulative[col] += n

        X_valid, metadata_valid = score_svmsmote_candidates(
            candidates=X_synth, X_min_orig=X_min_orig, numeric_cols=numeric_cols, binary_cols=binary_cols,
            numeric_min=numeric_min, numeric_range=numeric_range, ref_max=unified_ref_max,
            similarity_min=SIMILARITY_MIN, similarity_max=SIMILARITY_MAX, n_bins=N_SIMILARITY_BINS,
        )

        if len(X_valid) > 0:
            metadata_valid = metadata_valid.copy()
            metadata_valid["candidate_global_id"] = np.arange(next_candidate_global_id, next_candidate_global_id + len(X_valid))
            next_candidate_global_id += len(X_valid)
            candidate_frames.append(X_valid)
            metadata_frames.append(metadata_valid)

        if metadata_frames:
            pool_metadata_check = pd.concat(metadata_frames, ignore_index=True)
            enough_coverage = has_minimum_pool_coverage(
                metadata=pool_metadata_check, mother_quotas=mother_quotas,
                n_bins=N_SIMILARITY_BINS, strict_bin_coverage=strict_bin_coverage,
            )
            if enough_coverage and len(pool_metadata_check) >= minimum_pool_size:
                print(f"Converged at iteration {iteration} (pool={len(pool_metadata_check)}).")
                break

    if not candidate_frames:
        raise RuntimeError("SVMSMOTE generated no candidates under the configured flow.")

    X_pool = pd.concat(candidate_frames, ignore_index=True)
    pool_metadata = pd.concat(metadata_frames, ignore_index=True)

    print(f"Candidates in the pool: {len(pool_metadata)} (desired minimum: {minimum_pool_size})")

    if len(pool_metadata) < n_needed:
        raise RuntimeError(f"Insufficient pool: {len(pool_metadata)} candidates for {n_needed} synthetic samples needed.")

    pool_metadata = add_secondary_mother_assignments(
        candidates=X_pool, metadata=pool_metadata, X_min_orig=X_min_orig, mother_quotas=mother_quotas,
        numeric_cols=numeric_cols, binary_cols=binary_cols, numeric_min=numeric_min, numeric_range=numeric_range,
        ref_max=unified_ref_max, similarity_min=SIMILARITY_MIN, similarity_max=SIMILARITY_MAX,
        n_bins=N_SIMILARITY_BINS,
    )

    X_synth_final, synthetic_metadata = select_balanced_pool(
        candidates=X_pool, metadata=pool_metadata, mother_quotas=mother_quotas,
        n_bins=N_SIMILARITY_BINS, similarity_min=SIMILARITY_MIN, similarity_max=SIMILARITY_MAX,
    )

    y_synth_final = pd.Series([minority_class] * len(X_synth_final), dtype=y_orig.dtype)
    X_train_final = pd.concat([X_orig, X_synth_final], ignore_index=True)
    y_train_final = pd.concat([y_orig, y_synth_final], ignore_index=True)

    tracking_table = build_tracking_table(
        X_orig=X_orig, y_orig=y_orig, X_synth_final=X_synth_final, synthetic_metadata=synthetic_metadata,
        minority_class=minority_class, original_source_index=original_source_index,
    )

print()
print("Final class distribution (X_train_final):")
print(y_train_final.value_counts())

if len(synthetic_metadata) > 0:
    children_by_mother = synthetic_metadata["mother_index"].value_counts().sort_index()
    print(f"Children per parent -- min: {children_by_mother.min()} max: {children_by_mother.max()} "
          f"homogeneous: {(children_by_mother.max() - children_by_mother.min()) <= 1}")
    sim_final = synthetic_metadata["similarity"].to_numpy(dtype=float)
    print(f"Synthetic-sample similarity -- min:{sim_final.min():.4f} max:{sim_final.max():.4f} "
          f"mean:{sim_final.mean():.4f} median:{np.median(sim_final):.4f}")

save_table(tracking_table, "table_svmsmote_tracking_G.csv")

# The balanced training partition, the held-out partition and the parent-child
# tracking table are what every downstream script in this repository consumes,
# so persist them here: that is what makes the chain runnable end to end from a
# fresh clone. The same object is deposited under models/ for anyone who wants
# to run a downstream script without re-running this one.
joblib.dump(
    {
        "X_train_final": X_train_final,
        "y_train_final": y_train_final,
        "X_test": X_test,
        "y_test": y_test,
        "tracking_table": tracking_table,
    },
    OUT_DIR / "checkpoint_post_svmsmote_G.pkl",
)
print(f"Saved: {OUT_DIR / 'checkpoint_post_svmsmote_G.pkl'}")

# ====================================================================
# 5. Bayesian optimisation + champion selection (notebook cells 116-117)
# ====================================================================
print()
print("=" * 70)
print("5. Bayesian optimisation (BayesSearchCV, scoring=kappa) + champions")
print("=" * 70)

X_resampled = X_train_final.copy()
y_resampled = y_train_final.copy()

assert len(X_resampled) == len(tracking_table), "X_resampled and tracking_table have different lengths."
assert len(X_resampled) == len(y_resampled), "X_resampled and y_resampled have different lengths."

kappa_scorer = make_scorer(cohen_kappa_score)


def calculate_classification_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    p_fn = fn / (fn + tp) if (fn + tp) != 0 else 0
    p_fp = fp / (fp + tn) if (fp + tn) != 0 else 0
    kappa = cohen_kappa_score(y_true, y_pred)
    auc = None
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = None
    return accuracy, precision, recall, f1, tp, tn, fp, fn, p_fn, p_fp, kappa, auc


class RepeatedStratifiedGroupKFold:
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=rng)
            for train_idx, test_idx in cv.split(X, y, groups):
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


groups = tracking_table["mother_original_row_id"].copy()
mask = tracking_table["is_synthetic"] == False
groups.loc[mask] = tracking_table.loc[mask, "original_row_id"]
groups = groups.astype(int)

LABEL_MAP = {"Inactive": 0, "Active": 1}
y_test_bin = np.array([LABEL_MAP[c] for c in y_test["Activity"]], dtype=np.int64)
y_train_bin = np.array([LABEL_MAP[c] for c in y_resampled], dtype=np.int64)

pipe_mlp = Pipeline(steps=[("NN", MLPClassifier(solver="lbfgs", max_iter=20000, random_state=23, verbose=False))])
pipe_xgb = Pipeline(steps=[("xgb", XGBClassifier(random_state=0, booster="gbtree", objective="binary:logistic"))])
pipe_svm = Pipeline(steps=[("svm", svm.SVC(gamma="scale", max_iter=-1, probability=False))])

pair_grid_1 = {
    "NN__hidden_layer_sizes": Integer(5, 15),
    "NN__alpha": Real(1e-5, 1.0005965763586375e-05, "log-uniform"),
    "NN__activation": Categorical(["tanh"]),
    "NN__learning_rate_init": Real(1e-7, 1e-6, "log-uniform"),
}
pair_grid_2 = {
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
}
pair_grid_3 = {"svm__C": Real(0.5, 1, prior="log-uniform"), "svm__gamma": Real(0.01, 1, prior="log-uniform"), "svm__kernel": Categorical(["rbf"])}

pair_grid_list = [pair_grid_1, pair_grid_2, pair_grid_3]
labels = ["MLP", "XGBoost", "SVM"]
# Retained budget: n_iter=5 for all three families. The XGBoost champion was
# originally selected from a wider n_iter=15 search, but that candidate was
# superseded (see archive/round_niter15_exploratory/ and
# pipeline_G_niter_all_algorithms.py): the marginal gain in internal kappa was
# small next to the risk of the search itself overfitting one CV partition.
# This budget is what reproduces the published champions of Table 2.
n_iter_list = [5, 5, 5]
pipelines = [pipe_mlp, pipe_xgb, pipe_svm]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=21)

bscv_results = {}
bscv_objects = {}

for i in range(len(pipelines)):
    print(f"\n--- Bayesian search: {labels[i]} (n_iter={n_iter_list[i]}) ---")
    BSCV = BayesSearchCV(
        estimator=pipelines[i], search_spaces=pair_grid_list[i], n_iter=n_iter_list[i],
        n_jobs=-1, cv=cv, scoring=kappa_scorer, error_score="raise", random_state=21,
        return_train_score=True, refit=True, verbose=0,
    ).fit(X_resampled, y_train_bin, groups=groups)
    bscv_results[labels[i]] = BSCV.cv_results_
    bscv_objects[labels[i]] = BSCV

champions = {}
print("\nChampion model of each family (selected by internal CV kappa):")
for label, bscv in bscv_objects.items():
    champions[label] = bscv.best_estimator_
    print(f"{label}\n  best_params_: {bscv.best_params_}\n  best_score_ (CV, kappa): {bscv.best_score_:.4f}")

# ====================================================================
# 6. Training/test metrics of the already-fixed model
# ====================================================================
train_metrics = {k: [] for k in ["Model", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN", "Kappa", "AUC"]}
test_metrics = {k: [] for k in ["Model", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN", "Kappa", "AUC"]}

for label, model in champions.items():
    y_train_pred = model.predict(X_resampled)
    y_train_score = model.predict_proba(X_resampled)[:, 1] if hasattr(model, "predict_proba") else None
    acc, prec, rec, f1, tp, tn, fp, fn, p_fn, p_fp, kappa, auc = calculate_classification_metrics(y_train_bin, y_train_pred, y_train_score)
    for k, v in zip(train_metrics.keys(), [label, acc, prec, rec, f1, tp, tn, fp, fn, kappa, auc]):
        train_metrics[k].append(v)

    y_test_pred = model.predict(X_test)
    y_test_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc, prec, rec, f1, tp, tn, fp, fn, p_fn, p_fp, kappa, auc = calculate_classification_metrics(y_test_bin, y_test_pred, y_test_score)
    for k, v in zip(test_metrics.keys(), [label, acc, prec, rec, f1, tp, tn, fp, fn, kappa, auc]):
        test_metrics[k].append(v)

train_df = pd.DataFrame(train_metrics)
test_df = pd.DataFrame(test_metrics)

print("\nTraining performance (model already fixed by CV):")
print(train_df)
print("\nTest performance -- external validation (model already fixed by CV):")
print(test_df)

save_table(train_df, "table_training_metrics_G.csv")
save_table(test_df, "table_heldout_test_metrics_G.csv")

# Champion hyperparameters per model
hp_rows = []
for label, bscv in bscv_objects.items():
    row = {"Model": label, "best_score_cv_kappa": bscv.best_score_}
    row.update({k: v for k, v in bscv.best_params_.items()})
    hp_rows.append(row)
save_table(pd.DataFrame(hp_rows), "table_champion_hyperparameters_G.csv")

# ====================================================================
# 7. Diagnostic: repeated CV (20x5) of the already-selected candidate
# ====================================================================
print()
print("=" * 70)
print("7. Repeated CV (20x5) of the already-selected candidate (diagnostic)")
print("=" * 70)

N_REPEATS_DIAGNOSTIC = 20
repeated_cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=N_REPEATS_DIAGNOSTIC, random_state=21)

diagnostic_rows = []
for label, model in champions.items():
    scores = cross_validate(model, X_resampled, y_train_bin, cv=repeated_cv, groups=groups, scoring=kappa_scorer, n_jobs=-1)
    vals = scores["test_score"]
    print(f"  {label:<10} n={len(vals)}  mean={vals.mean():.4f}  sd={vals.std():.4f}  "
          f"IC95%=[{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]")
    diagnostic_rows.append({
        "Model": label, "n": len(vals), "kappa_mean": vals.mean(), "kappa_sd": vals.std(),
        "kappa_ci95_low": np.percentile(vals, 2.5), "kappa_ci95_high": np.percentile(vals, 97.5),
    })

save_table(pd.DataFrame(diagnostic_rows), "table_repeated_cv_diagnostic_G.csv")

# ====================================================================
# 8. Save the final models
# ====================================================================
for label, model in champions.items():
    filename = OUT_DIR / f"final_model_{label}_G.pkl"
    joblib.dump(model, filename)
    print(f"Saved: {filename}")

# ====================================================================
# 9. Minimal Platt/sigmoid calibration of the champions, only to obtain
#    probability scores usable for AUC/ROC -- needed in particular for
#    the SVM (pipe_svm uses probability=False, identical to the
#    notebook). It does NOT touch the predicted labels (predict()) used
#    in the Accuracy/Precision/Recall/F1/kappa metrics above -- it only
#    affects the continuous score used for AUC/ROC. The full
#    multi-scenario calibration study (Model A vs B, per-bin ECE
#    bootstrap) lives in pipeline_G_calibration_scenarios.py.
# ====================================================================
print()
print("=" * 70)
print("9. Platt/sigmoid calibration of the champions for AUC/ROC")
print("=" * 70)

from sklearn.calibration import CalibratedClassifierCV

# Calibration set: ONLY the original training ligands, deliberately excluding
# the SVMSMOTE synthetic instances the champion itself saw while being fitted
# (Section 2.6 of Article 1; the "calib. original" scenario of Table 6).
is_original_for_calibration = (tracking_table["is_synthetic"] == False).to_numpy()
X_cal = X_resampled.loc[is_original_for_calibration].reset_index(drop=True)
y_cal = y_train_bin[is_original_for_calibration]
print(f"  calibration set: {len(X_cal)} original ligands "
      f"({int((~is_original_for_calibration).sum())} synthetic instances excluded)")

try:
    from sklearn.frozen import FrozenEstimator

    def fit_platt_calibration(model):
        return CalibratedClassifierCV(estimator=FrozenEstimator(model), method="sigmoid").fit(X_cal, y_cal)
except ImportError:
    def fit_platt_calibration(model):
        return CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit").fit(X_cal, y_cal)

calibrated_champions = {label: fit_platt_calibration(model) for label, model in champions.items()}

roc_rows = []
for label, model in champions.items():
    y_train_score_cal = calibrated_champions[label].predict_proba(X_resampled)[:, 1]
    y_test_score_cal = calibrated_champions[label].predict_proba(X_test)[:, 1]

    auc_train_cal = roc_auc_score(y_train_bin, y_train_score_cal)
    auc_test_cal = roc_auc_score(y_test_bin, y_test_score_cal)

    train_df.loc[train_df["Model"] == label, "AUC"] = auc_train_cal
    test_df.loc[test_df["Model"] == label, "AUC"] = auc_test_cal

    fpr, tpr, _ = roc_curve(y_test_bin, y_test_score_cal)
    for f, t in zip(fpr, tpr):
        roc_rows.append({"Model": label, "fpr": f, "tpr": t})

    print(f"  {label:<10} AUC train (calibrated)={auc_train_cal:.4f}  AUC test (calibrated)={auc_test_cal:.4f}")

save_table(train_df, "table_training_metrics_G.csv")
save_table(test_df, "table_heldout_test_metrics_G.csv")
save_table(pd.DataFrame(roc_rows), "table_roc_points_G.csv")

for label, model in calibrated_champions.items():
    joblib.dump(model, OUT_DIR / f"final_model_{label}_calibrated_G.pkl")

# ====================================================================
# 10. Feature importance of the XGBoost champion (gain/cover), using the
#     same extraction technique as the notebook (booster.get_score) but
#     applied to the REAL champion of the kappa-scored Bayesian search
#     (cells 116-117), not to the separate log-loss/early-stopping D1
#     model, which belongs to an exploratory branch outside this scope.
# ====================================================================
print()
print("=" * 70)
print("10. Feature importance (gain/cover) of the XGBoost champion")
print("=" * 70)

xgb_champion = champions["XGBoost"].named_steps["xgb"]
booster = xgb_champion.get_booster()

importance_types = ["gain", "total_gain", "cover", "total_cover"]
importance_data = {imp: booster.get_score(importance_type=imp) for imp in importance_types}

importance_df = pd.DataFrame.from_dict(importance_data)
importance_df.index.name = "Feature"
importance_df = importance_df.reset_index()
importance_df[importance_types] = importance_df[importance_types].fillna(0)
importance_df = importance_df.sort_values(by="total_gain", ascending=False).reset_index(drop=True)

# Guarantees that ALL 57 descriptors are present (even those used in no
# split, importance = 0) for the full-set figure.
all_features = pd.DataFrame({"Feature": X_resampled.columns})
importance_full = all_features.merge(importance_df, on="Feature", how="left").fillna(0)
importance_full = importance_full.sort_values(by="gain", ascending=False).reset_index(drop=True)

top20_df = importance_df.head(20)
save_table(top20_df, "table_feature_importance_top20_G.csv")
save_table(importance_full, "table_feature_importance_all57_G.csv")

print(top20_df.to_string(index=False))

# ====================================================================
# 11. Figures, directly usable in Article 1
# ====================================================================
print()
print("=" * 70)
print("11. Figures for Article 1")
print("=" * 70)

FIG_DIR = OUT_DIR / "figures_G"
FIG_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {"MLP": "#d62728", "SVM": "#2ca02c", "XGBoost": "#1f77b4"}

# --- Fig. G1: SVMSMOTE synthetic-sample diagnostics (4-panel layout) ---
if len(synthetic_metadata) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    axes = axes.ravel()

    sim_values = synthetic_metadata["similarity"].to_numpy(dtype=float)
    bins = np.linspace(SIMILARITY_MIN - 0.01, SIMILARITY_MAX + 0.01, 28)

    axes[0].hist(sim_values, bins=bins, color="#6a3d9a", edgecolor="white", alpha=0.9)
    axes[0].axvspan(SIMILARITY_MIN, SIMILARITY_MAX, color="green", alpha=0.08)
    axes[0].axvline(SIMILARITY_MIN, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(SIMILARITY_MAX, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Selected synthetic-sample similarities")
    axes[0].set_xlabel("Mixed-distance similarity to nearest Inactive parent")
    axes[0].set_ylabel("Frequency")

    axes[1].boxplot(sim_values, showfliers=False, patch_artist=True,
                     boxprops=dict(facecolor="#cab2d6"))
    axes[1].axhspan(SIMILARITY_MIN, SIMILARITY_MAX, color="green", alpha=0.08)
    axes[1].set_title("Similarity summary")
    axes[1].set_ylabel("Similarity")
    axes[1].set_xticks([])

    children_by_bin = synthetic_metadata["bin"].value_counts().sort_index()
    axes[2].bar(children_by_bin.index.astype(str), children_by_bin.values, color="#6a3d9a")
    axes[2].set_title("Synthetics per similarity bin")
    axes[2].set_xlabel("Similarity bin")
    axes[2].set_ylabel("Frequency")
    axes[2].tick_params(axis="x", rotation=45)

    children_by_mother = synthetic_metadata["mother_index"].value_counts().sort_index()
    axes[3].hist(
        children_by_mother.values,
        bins=np.arange(children_by_mother.min(), children_by_mother.max() + 2) - 0.5,
        color="#1f77b4", edgecolor="white",
    )
    axes[3].set_title("Synthetic children per Inactive parent")
    axes[3].set_xlabel("Number of children per parent")
    axes[3].set_ylabel("Number of parents")
    axes[3].set_xticks(np.arange(children_by_mother.min(), children_by_mother.max() + 1))

    fig.suptitle("SVMSMOTE synthetic Inactive samples: similarity and balance diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figG1_svmsmote_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'figG1_svmsmote_diagnostics.png'}")

# --- Fig. G2: Bayesian-search candidate dispersion per model family
#     (replaces the 20-candidate MLP/SVM/XGBoost comparison figures;
#     n_iter = 5 for every family, the retained budget) ---
dispersion_rows = []
for label, bscv in bscv_objects.items():
    cvr = bscv.cv_results_
    for idx, (mean_score, std_score) in enumerate(zip(cvr["mean_test_score"], cvr["std_test_score"])):
        dispersion_rows.append({
            "Model": label, "candidate_idx": idx, "mean_cv_kappa": mean_score,
            "std_cv_kappa": std_score, "is_best": mean_score == bscv.best_score_,
        })
dispersion_df = pd.DataFrame(dispersion_rows)
save_table(dispersion_df, "table_bayes_search_candidate_dispersion_G.csv")

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
order = ["MLP", "SVM", "XGBoost"]
data_by_model = [dispersion_df.loc[dispersion_df["Model"] == m, "mean_cv_kappa"].to_numpy() for m in order]
bp = ax.boxplot(data_by_model, tick_labels=order, patch_artist=True, showmeans=True)
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor(MODEL_COLORS[m])
    patch.set_alpha(0.5)
for m, x in zip(order, range(1, len(order) + 1)):
    vals = dispersion_df.loc[dispersion_df["Model"] == m, "mean_cv_kappa"]
    ax.scatter(np.full(len(vals), x) + np.random.default_rng(0).uniform(-0.05, 0.05, len(vals)),
               vals, color="black", alpha=0.6, s=18, zorder=3)
    best_val = dispersion_df.loc[(dispersion_df["Model"] == m) & (dispersion_df["is_best"]), "mean_cv_kappa"]
    if len(best_val):
        ax.scatter([x], best_val.values[:1], color="gold", edgecolor="black", s=140, zorder=4, marker="*",
                   label="Selected candidate" if m == order[0] else None)
ax.set_ylabel("Internal 5-fold CV Cohen's $\\kappa$ (mother-child grouped)")
ax.set_title("Bayesian-search candidate dispersion by model family\n(n_iter = 5 for MLP, SVM and XGBoost)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "figG2_bayes_search_dispersion.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'figG2_bayes_search_dispersion.png'}")

# --- Fig. G3: Repeated group-CV (20x5) robustness of the chosen
#     candidate -- supports "the optimisation itself did not overfit
#     to CV noise" (repeated estimate close to the search's best_score_) --
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
rep_data = []
rep_labels = []
for label in order:
    scores = cross_validate(champions[label], X_resampled, y_train_bin, cv=repeated_cv, groups=groups,
                             scoring=kappa_scorer, n_jobs=-1)["test_score"]
    rep_data.append(scores)
    rep_labels.append(label)
bp = ax.boxplot(rep_data, tick_labels=rep_labels, patch_artist=True, showmeans=True)
for patch, m in zip(bp["boxes"], rep_labels):
    patch.set_facecolor(MODEL_COLORS[m])
    patch.set_alpha(0.5)
for label, bscv in bscv_objects.items():
    x = rep_labels.index(label) + 1
    ax.scatter([x], [bscv.best_score_], color="gold", edgecolor="black", s=140, zorder=4, marker="*",
               label="Bayesian-search best_score_" if x == 1 else None)
ax.set_ylabel("Cohen's $\\kappa$ (mother-child grouped CV)")
ax.set_title("Repeated cross-validation (20$\\times$5 = 100 folds) of the\nselected champion, vs. the search's internal best_score_")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "figG3_repeated_cv_robustness.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'figG3_repeated_cv_robustness.png'}")

# --- Fig. G4 (replaces Fig. 7): ROC curves on the held-out test set,
#     calibrated scores, all three retained classifiers ---
fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
for label in order:
    y_score = calibrated_champions[label].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    auc_val = roc_auc_score(y_test_bin, y_score)
    ax.plot(fpr, tpr, color=MODEL_COLORS[label], linewidth=2, label=f"{label} (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the held-out test set (n = 96)\nfinal retained MLP, SVM and XGBoost classifiers")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "figG4_roc_curves.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'figG4_roc_curves.png'}")

# --- Fig. G5 (replaces Fig. 8): global XGBoost feature importance
#     (mean gain), full 57-descriptor set, champion model ---
fig, ax = plt.subplots(figsize=(9, 14), dpi=150)
plot_df = importance_full.sort_values("gain", ascending=True)
ax.barh(plot_df["Feature"], plot_df["gain"], color="#1f77b4")
ax.set_xlabel("Mean gain")
ax.set_title("Global XGBoost feature importance (mean gain)\nacross the full 57-descriptor set -- selected champion model")
ax.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG5_feature_importance_full.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'figG5_feature_importance_full.png'}")

# --- Fig. G6 (replaces Fig. 4): class distribution across the 70/30
#     split AND the SVMSMOTE-balanced training partition, in English ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
split_counts = pd.DataFrame({
    "Train (pre-resampling)": y_orig.value_counts(),
    "Test": y_test["Activity"].value_counts(),
    "Train (post-SVMSMOTE)": y_train_final.value_counts(),
}).reindex(["Active", "Inactive"])
split_counts.T.plot(kind="bar", stacked=True, ax=axes[0], color=["#d62728", "#1f77b4"])
axes[0].set_ylabel("Number of compounds")
axes[0].set_title("Class counts across the pipeline")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(title="Class")

(split_counts.T.div(split_counts.T.sum(axis=1), axis=0) * 100).plot(
    kind="bar", stacked=True, ax=axes[1], color=["#d62728", "#1f77b4"], legend=False,
)
axes[1].set_ylabel("Proportion (%)")
axes[1].set_title("Class proportions across the pipeline")
axes[1].tick_params(axis="x", rotation=20)

fig.suptitle("Class distribution: 70/30 split and SVMSMOTE-balanced training partition", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG6_class_distribution.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'figG6_class_distribution.png'}")

print()
print("=" * 70)
print("VERSION-G PIPELINE COMPLETE")
print("=" * 70)
