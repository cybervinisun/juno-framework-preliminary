"""
Pipeline "versao G" -- reconstrucao fiel do notebook RECONSTRUIDO
(Modelo_Classificacao_ML_QSAR_320_ligantes_GoldsScore_v_Tese_RECONSTRUIDO)
rodando sobre o banco real de 320 ligantes, para o Artigo 1
(Journal of Cheminformatics).

Reaproveita, sem alteracao de logica, exatamente o que foi confirmado
celula-a-celula no notebook original:
  - Split 70/30 estratificado (random_state=42)
  - Normalizacao Min-Max (fit no treino, corrScore excluido)
  - PCA(0.999) diagnostico sobre os 9 descritores continuos (nao entra
    na matriz de modelagem -- e so estudo/figura, igual na tese)
  - SVMSMOTE PARTE A/B: distancia mista Gower-like (Manhattan continuas +
    Hamming simetrico binarias, 50/50), geracao de candidatos crus via
    SVMSMOTE, correcao formal (reflexao de continuas fora de [0,1],
    snap de pKa ordinal, atribuicao de bits binarios por correlacao
    ponto-bisserial com tolerancia noise_budget/temperatura), selecao
    balanceada por mae dentro da faixa de similaridade 0.83-0.90
  - Otimizacao bayesiana (skopt.BayesSearchCV) com StratifiedGroupKFold
    (grupos mae-filha) e scoring = Cohen's Kappa (kappa_scorer), para
    MLP, XGBoost e SVM -- confirmado como o UNICO criterio real de
    selecao de hiperparametros/campeao em todas as chamadas efetivas
    do notebook (MCC e importado mas nunca chamado; so aparece como
    coluna extra opcional no relatorio de CV repetida).
  - Selecao do campeao via best_estimator_/best_score_ (CV interna),
    NUNCA por metrica de teste.
  - Metricas de treino/teste + diagnostico de CV repetida (20x5) do
    candidato ja escolhido.

Escopo desta reconstrucao (decisao explicita, documentada para o
usuario): a PARTE A do notebook (bootstrap dos limiares de
similaridade Ativo-Inativo/Inativo-Inativo) e reproduzida apenas na
sua conclusao operacional -- o notebook usa os limiares fixos 0.83 e
0.90 (nao le programaticamente `limiar_equilibrado`/`limiar_conservador`
de volta no codigo da PARTE B, sao so diagnostico impresso). O
bootstrap completo (graficos/tabelas de cobertura) fica de fora deste
script porque nao afeta o modelo final -- pode ser adicionado depois
se for necessario para a secao de Metodos.
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
# 0. Caminhos e saida
#
# DATA_DIR pode ser sobrescrito via variavel de ambiente, ex.:
#   DATA_DIR=/caminho/outro python pipeline_G.py
# Por padrao aponta para data/processed/ na raiz do repositorio.
# ====================================================================
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "processed"))

X_PATH = DATA_DIR / "X_320ligands_57descriptors.xlsx"
Y_PATH = DATA_DIR / "y_320ligands_labels.xlsx"

OUT_DIR = Path(__file__).parent / "versao_G_outputs"
OUT_DIR.mkdir(exist_ok=True)


def salvar_tabela(df: pd.DataFrame, nome: str) -> None:
    path = OUT_DIR / nome
    df.to_csv(path, index=False)
    print(f"[tabela salva] {path}")


# ====================================================================
# 1. Carregamento e split (celulas 91/93/94 do notebook)
# ====================================================================
print("=" * 70)
print("1. Carregamento e split 70/30 estratificado")
print("=" * 70)

df4 = pd.read_excel(X_PATH, index_col=0)
y = pd.read_excel(Y_PATH, index_col=0)

assert (df4.index == y.index).all(), "X e y desalinhados pelo indice."

X_final = df4.copy()
y_final = y.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.30,
    stratify=y_final["Atividade"],
    random_state=42,
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print("Distribuicao treino:\n", y_train["Atividade"].value_counts())
print("Distribuicao teste:\n", y_test["Atividade"].value_counts())

# ====================================================================
# 2. Normalizacao Min-Max (celula 96) -- fit no treino, corrScore fora
# ====================================================================
print()
print("=" * 70)
print("2. Normalizacao Min-Max (fit no treino)")
print("=" * 70)

descritores_ja_normalizados = ["corrScore"]

col_binarias_scaler = [
    col for col in X_train.columns
    if set(X_train[col].dropna().unique()) <= {0, 1}
]
col_continuas_scaler = [
    col for col in X_train.columns
    if col not in col_binarias_scaler and col not in descritores_ja_normalizados
]

scaler = MinMaxScaler()
scaler.fit(X_train[col_continuas_scaler])

X_train[col_continuas_scaler] = scaler.transform(X_train[col_continuas_scaler])
X_test[col_continuas_scaler] = scaler.transform(X_test[col_continuas_scaler])

print(f"Continuos normalizados: {len(col_continuas_scaler)} | "
      f"Binarios preservados: {len(col_binarias_scaler)} | "
      f"Ja normalizados preservados: {descritores_ja_normalizados}")

joblib.dump(scaler, OUT_DIR / "scaler_minmax_treino_G.pkl")

# ====================================================================
# 3. PCA diagnostico (celulas 56-57) -- 9 continuas, NAO entra no
#    modelo, so estudo/figura (igual na tese)
# ====================================================================
print()
print("=" * 70)
print("3. PCA diagnostico sobre os descritores continuos (0.999 var.)")
print("=" * 70)

continuous_cols_pca = [c for c in X_train.columns if c not in col_binarias_scaler]
pca = PCA(n_components=0.999, svd_solver="full")
pca_scores = pca.fit_transform(X_train[continuous_cols_pca])

print(f"Descritores continuos usados na ACP: {len(continuous_cols_pca)}")
print(f"Componentes retidas (0.999 da variancia): {pca.n_components_}")
print(f"Variancia explicada por componente: {np.round(pca.explained_variance_ratio_, 4)}")
print(f"Variancia acumulada: {np.round(np.cumsum(pca.explained_variance_ratio_), 4)}")

loadings = pd.DataFrame(
    pca.components_.T,
    index=continuous_cols_pca,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
)
salvar_tabela(loadings.reset_index().rename(columns={"index": "descritor"}), "tabela_ACP_loadings_G.csv")

pca_var_df = pd.DataFrame({
    "componente": [f"PC{i+1}" for i in range(pca.n_components_)],
    "variancia_explicada": pca.explained_variance_ratio_,
    "variancia_acumulada": np.cumsum(pca.explained_variance_ratio_),
})
salvar_tabela(pca_var_df, "tabela_ACP_variancia_G.csv")

# ====================================================================
# 4. SVMSMOTE PARTE A/B (celulas 103-106) -- reconstrucao fiel
# ====================================================================
print()
print("=" * 70)
print("4. SVMSMOTE PARTE A/B -- geracao balanceada de sinteticas Inativo")
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
        raise ValueError("Nenhuma coluna numerica ou binaria foi encontrada.")
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
    X_sint_raw, X_min_orig, binary_cols, numeric_cols, r_matrix,
    noise_budget, temperatura, random_state,
):
    X_out = X_sint_raw.copy()
    mean_k = X_min_orig[numeric_cols].mean()
    std_k = X_min_orig[numeric_cols].std().replace(0, 1e-8)
    Z = (X_out[numeric_cols] - mean_k) / std_k
    rng = np.random.default_rng(random_state)
    flips_por_coluna = {}
    for j in binary_cols:
        b_bruto = X_out[j].to_numpy(dtype=float)
        b_binario = (b_bruto >= 0.5).astype(float)
        r_j = r_matrix.loc[j].to_numpy(dtype=float)
        bit_sem_evidencia = np.allclose(r_j, 0.0) and X_min_orig[j].std() == 0
        if bit_sem_evidencia:
            X_out[j] = b_binario
            flips_por_coluna[j] = 0
            continue
        score = (Z.to_numpy() * r_j).sum(axis=1) * temperatura
        p1 = 1.0 / (1.0 + np.exp(-score))
        disagree = np.abs(p1 - b_binario)
        prob_correcao = noise_budget * disagree
        sorteio = rng.random(len(X_out))
        corrigir = sorteio < prob_correcao
        b_final = b_binario.copy()
        b_final[corrigir] = 1 - b_binario[corrigir]
        X_out[j] = b_final
        flips_por_coluna[j] = int(corrigir.sum())
    return X_out, flips_por_coluna


def enforce_continuous_bounds_by_reflection(X_sint_raw, numeric_cols, limite_inferior=0.0, limite_superior=1.0):
    X_out = X_sint_raw.copy()
    largura = limite_superior - limite_inferior
    periodo = 2 * largura
    contagem_corrigidos = {}
    for col in numeric_cols:
        valores = X_out[col].to_numpy(dtype=float)
        fora = (valores < limite_inferior) | (valores > limite_superior)
        deslocado = (valores - limite_inferior) % periodo
        refletido = np.where(deslocado > largura, periodo - deslocado, deslocado)
        X_out[col] = limite_inferior + refletido
        contagem_corrigidos[col] = int(fora.sum())
    return X_out, contagem_corrigidos


def enforce_ordinal_pka_category(X_sint_raw, X_orig, col_pka_ordinal):
    X_out = X_sint_raw.copy()
    for col in col_pka_ordinal:
        categorias_validas = np.sort(X_orig[col].unique())
        valores = X_out[col].to_numpy(dtype=float)
        idx_mais_proximo = np.abs(valores[:, None] - categorias_validas[None, :]).argmin(axis=1)
        X_out[col] = categorias_validas[idx_mais_proximo]
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


def generate_svmsmote_candidates(X_orig, y_orig, classe_min, target_minority_count, random_state, k_neighbors):
    sampler = SVMSMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
        sampling_strategy={classe_min: target_minority_count},
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


def build_tracking_table(X_orig, y_orig, X_sint_final, synthetic_metadata, classe_min, original_source_index):
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

    synthetic_ids = [f"synth_inactive_{idx:05d}" for idx in range(len(X_sint_final))]
    minority_positions_in_X_orig = np.where(y_orig.to_numpy() == classe_min)[0]
    mother_index_in_minority = synthetic_metadata["mother_index"].to_numpy(dtype=int)
    mother_original_row_ids = minority_positions_in_X_orig[mother_index_in_minority]
    mother_source_indices = original_source_index[mother_original_row_ids]

    synthetic_tracking = pd.DataFrame({
        "final_row_id": np.arange(len(X_orig), len(X_orig) + len(X_sint_final)),
        "sample_id": synthetic_ids,
        "is_synthetic": True,
        "synthetic_id": synthetic_ids,
        "class_label": classe_min,
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


# ---- Parametros principais (identicos ao notebook, celula 106) ----
similaridade_minima = 0.83
similaridade_maxima = 0.90
n_bins_uniforme = 6
max_iter = 1000
pool_overshoot_factor = 4
strict_bin_coverage = False
random_state_base = 479
k_neighbors_base = 10
noise_budget = 0.05
temperatura = 3.0

X_orig = X_train.copy().reset_index(drop=True)
y_orig = y_train["Atividade"].copy().reset_index(drop=True)
original_source_index = np.arange(len(X_orig))

classe_min = y_orig.value_counts().idxmin()
classe_maj = y_orig.value_counts().idxmax()

X_min_orig = X_orig[y_orig == classe_min].reset_index(drop=True)
n_min = len(X_min_orig)
n_maj = int((y_orig == classe_maj).sum())
n_target = n_maj
n_needed = n_target - n_min

print(f"Classe minoritaria: {classe_min} (n={n_min}) | Classe majoritaria: {classe_maj} (n={n_maj})")
print(f"Sinteticas necessarias para balancear 100%: {n_needed}")

numeric_cols, binary_cols = prepare_mixed_distance_columns(X_orig)
numeric_min = X_orig[numeric_cols].min() if numeric_cols else pd.Series(dtype=float)
numeric_range = (X_orig[numeric_cols].max() - X_orig[numeric_cols].min()) if numeric_cols else pd.Series(dtype=float)
numeric_range = numeric_range.replace(0, 1.0)

col_pka_ordinal = [c for c in X_orig.columns if "pka" in c.lower()]
r_matrix = compute_point_biserial_matrix(X_min_orig, binary_cols, numeric_cols)

D_all_symmetric = mixed_gower_like_distance_symmetric_binary(
    X_orig, X_orig, numeric_cols=numeric_cols, binary_cols=binary_cols,
    numeric_min=numeric_min, numeric_range=numeric_range, numeric_weight=0.5, binary_weight=0.5,
)
ref_max_unificado = float(D_all_symmetric.max())
if ref_max_unificado <= 0:
    raise ValueError("A distancia de referencia unificada deve ser maior que zero.")

print(f"Colunas numericas continuas: {len(numeric_cols)} | Colunas binarias: {len(binary_cols)}")
print(f"ref_max_unificado simetrico: {ref_max_unificado:.6f}")

rng = np.random.default_rng(random_state_base)

if n_needed <= 0:
    X_train_final = X_orig.copy()
    y_train_final = y_orig.copy()
    synthetic_metadata = pd.DataFrame(columns=["mother_index", "similarity", "bin"])
    tracking_table = build_tracking_table(
        X_orig, y_orig, pd.DataFrame(columns=X_orig.columns), synthetic_metadata, classe_min, original_source_index,
    )
    print("A classe minoritaria ja esta balanceada ou acima da majoritaria.")
else:
    if n_min < 2:
        raise ValueError("SVMSMOTE precisa de pelo menos 2 amostras minoritarias.")

    k_neighbors = min(k_neighbors_base, n_min - 1)
    mother_quotas = allocate_evenly(total=n_needed, n_groups=n_min, rng=rng)

    candidate_frames = []
    metadata_frames = []
    next_candidate_global_id = 0
    minimum_pool_size = n_needed * pool_overshoot_factor
    flips_acumulados = {j: 0 for j in binary_cols}
    reflexoes_acumuladas = {j: 0 for j in numeric_cols}

    for iteration in range(max_iter):
        X_sint_raw = generate_svmsmote_candidates(
            X_orig=X_orig, y_orig=y_orig, classe_min=classe_min,
            target_minority_count=n_target, random_state=random_state_base + iteration, k_neighbors=k_neighbors,
        )
        X_sint, reflexoes_iteracao = enforce_continuous_bounds_by_reflection(X_sint_raw, numeric_cols)
        for col, n in reflexoes_iteracao.items():
            reflexoes_acumuladas[col] += n

        X_sint = enforce_ordinal_pka_category(X_sint, X_orig, col_pka_ordinal)
        X_sint, flips_iteracao = enforce_formal_binary_assignment(
            X_sint_raw=X_sint, X_min_orig=X_min_orig, binary_cols=binary_cols, numeric_cols=numeric_cols,
            r_matrix=r_matrix, noise_budget=noise_budget, temperatura=temperatura,
            random_state=random_state_base + iteration,
        )
        for col, n in flips_iteracao.items():
            flips_acumulados[col] += n

        X_valid, metadata_valid = score_svmsmote_candidates(
            candidates=X_sint, X_min_orig=X_min_orig, numeric_cols=numeric_cols, binary_cols=binary_cols,
            numeric_min=numeric_min, numeric_range=numeric_range, ref_max=ref_max_unificado,
            similarity_min=similaridade_minima, similarity_max=similaridade_maxima, n_bins=n_bins_uniforme,
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
                n_bins=n_bins_uniforme, strict_bin_coverage=strict_bin_coverage,
            )
            if enough_coverage and len(pool_metadata_check) >= minimum_pool_size:
                print(f"Convergiu na iteracao {iteration} (pool={len(pool_metadata_check)}).")
                break

    if not candidate_frames:
        raise RuntimeError("SVMSMOTE nao gerou candidatos dentro do fluxo configurado.")

    X_pool = pd.concat(candidate_frames, ignore_index=True)
    pool_metadata = pd.concat(metadata_frames, ignore_index=True)

    print(f"Candidatos no pool: {len(pool_metadata)} (minimo desejado: {minimum_pool_size})")

    if len(pool_metadata) < n_needed:
        raise RuntimeError(f"Pool insuficiente: {len(pool_metadata)} candidatos para {n_needed} sinteticas necessarias.")

    pool_metadata = add_secondary_mother_assignments(
        candidates=X_pool, metadata=pool_metadata, X_min_orig=X_min_orig, mother_quotas=mother_quotas,
        numeric_cols=numeric_cols, binary_cols=binary_cols, numeric_min=numeric_min, numeric_range=numeric_range,
        ref_max=ref_max_unificado, similarity_min=similaridade_minima, similarity_max=similaridade_maxima,
        n_bins=n_bins_uniforme,
    )

    X_sint_final, synthetic_metadata = select_balanced_pool(
        candidates=X_pool, metadata=pool_metadata, mother_quotas=mother_quotas,
        n_bins=n_bins_uniforme, similarity_min=similaridade_minima, similarity_max=similaridade_maxima,
    )

    y_sint_final = pd.Series([classe_min] * len(X_sint_final), dtype=y_orig.dtype)
    X_train_final = pd.concat([X_orig, X_sint_final], ignore_index=True)
    y_train_final = pd.concat([y_orig, y_sint_final], ignore_index=True)

    tracking_table = build_tracking_table(
        X_orig=X_orig, y_orig=y_orig, X_sint_final=X_sint_final, synthetic_metadata=synthetic_metadata,
        classe_min=classe_min, original_source_index=original_source_index,
    )

print()
print("Distribuicao final das classes (X_train_final):")
print(y_train_final.value_counts())

if len(synthetic_metadata) > 0:
    children_by_mother = synthetic_metadata["mother_index"].value_counts().sort_index()
    print(f"Filhos/mae -- min: {children_by_mother.min()} max: {children_by_mother.max()} "
          f"homogeneo: {(children_by_mother.max() - children_by_mother.min()) <= 1}")
    sim_final = synthetic_metadata["similarity"].to_numpy(dtype=float)
    print(f"Similaridade das sinteticas -- min:{sim_final.min():.4f} max:{sim_final.max():.4f} "
          f"media:{sim_final.mean():.4f} mediana:{np.median(sim_final):.4f}")

salvar_tabela(tracking_table, "tabela_rastreio_SVMSMOTE_G.csv")

# ====================================================================
# 5. Otimizacao bayesiana + selecao do campeao (celulas 116-117)
# ====================================================================
print()
print("=" * 70)
print("5. Otimizacao bayesiana (BayesSearchCV, scoring=Kappa) + campeoes")
print("=" * 70)

X_resampled = X_train_final.copy()
y_resampled = y_train_final.copy()

assert len(X_resampled) == len(tracking_table), "X_resampled e tracking_table com tamanhos diferentes."
assert len(X_resampled) == len(y_resampled), "X_resampled e y_resampled com tamanhos diferentes."

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

mapeamento = {"Inativo": 0, "Ativo": 1}
y_test_bin = np.array([mapeamento[c] for c in y_test["Atividade"]], dtype=np.int64)
y_train_bin = np.array([mapeamento[c] for c in y_resampled], dtype=np.int64)

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
n_iter_list = [5, 15, 5]
pipelines = [pipe_mlp, pipe_xgb, pipe_svm]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=21)

bscv_results = {}
bscv_objects = {}

for i in range(len(pipelines)):
    print(f"\n--- Busca bayesiana: {labels[i]} (n_iter={n_iter_list[i]}) ---")
    BSCV = BayesSearchCV(
        estimator=pipelines[i], search_spaces=pair_grid_list[i], n_iter=n_iter_list[i],
        n_jobs=-1, cv=cv, scoring=kappa_scorer, error_score="raise", random_state=21,
        return_train_score=True, refit=True, verbose=0,
    ).fit(X_resampled, y_train_bin, groups=groups)
    bscv_results[labels[i]] = BSCV.cv_results_
    bscv_objects[labels[i]] = BSCV

melhores_modelos = {}
print("\nModelo campeao de cada familia (escolhido pela CV interna, kappa):")
for label, bscv in bscv_objects.items():
    melhores_modelos[label] = bscv.best_estimator_
    print(f"{label}\n  best_params_: {bscv.best_params_}\n  best_score_ (CV, kappa): {bscv.best_score_:.4f}")

# ====================================================================
# 6. Metricas de treino/teste do modelo ja fixado
# ====================================================================
train_metrics = {k: [] for k in ["Model", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN", "Kappa", "AUC"]}
test_metrics = {k: [] for k in ["Model", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN", "Kappa", "AUC"]}

for label, modelo in melhores_modelos.items():
    y_train_pred = modelo.predict(X_resampled)
    y_train_score = modelo.predict_proba(X_resampled)[:, 1] if hasattr(modelo, "predict_proba") else None
    acc, prec, rec, f1, tp, tn, fp, fn, p_fn, p_fp, kappa, auc = calculate_classification_metrics(y_train_bin, y_train_pred, y_train_score)
    for k, v in zip(train_metrics.keys(), [label, acc, prec, rec, f1, tp, tn, fp, fn, kappa, auc]):
        train_metrics[k].append(v)

    y_test_pred = modelo.predict(X_test)
    y_test_score = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
    acc, prec, rec, f1, tp, tn, fp, fn, p_fn, p_fp, kappa, auc = calculate_classification_metrics(y_test_bin, y_test_pred, y_test_score)
    for k, v in zip(test_metrics.keys(), [label, acc, prec, rec, f1, tp, tn, fp, fn, kappa, auc]):
        test_metrics[k].append(v)

train_df = pd.DataFrame(train_metrics)
test_df = pd.DataFrame(test_metrics)

print("\nDesempenho no treino (modelo ja fixado pela CV):")
print(train_df)
print("\nDesempenho no teste -- validacao externa (modelo ja fixado pela CV):")
print(test_df)

salvar_tabela(train_df, "tabela_metricas_treino_G.csv")
salvar_tabela(test_df, "tabela_metricas_teste_externo_G.csv")

# Hiperparametros do campeao por modelo
hp_rows = []
for label, bscv in bscv_objects.items():
    row = {"Model": label, "best_score_cv_kappa": bscv.best_score_}
    row.update({k: v for k, v in bscv.best_params_.items()})
    hp_rows.append(row)
salvar_tabela(pd.DataFrame(hp_rows), "tabela_hiperparametros_campeoes_G.csv")

# ====================================================================
# 7. Diagnostico: CV repetida (20x5) do candidato ja escolhido
# ====================================================================
print()
print("=" * 70)
print("7. CV repetida (20x5) do candidato ja escolhido (diagnostico)")
print("=" * 70)

N_REPEATS_DIAGNOSTICO = 20
cv_repetida = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=N_REPEATS_DIAGNOSTICO, random_state=21)

diag_rows = []
for label, modelo in melhores_modelos.items():
    scores = cross_validate(modelo, X_resampled, y_train_bin, cv=cv_repetida, groups=groups, scoring=kappa_scorer, n_jobs=-1)
    vals = scores["test_score"]
    print(f"  {label:<10} n={len(vals)}  media={vals.mean():.4f}  desvio={vals.std():.4f}  "
          f"IC95%=[{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]")
    diag_rows.append({
        "Model": label, "n": len(vals), "kappa_media": vals.mean(), "kappa_desvio": vals.std(),
        "kappa_IC95_inf": np.percentile(vals, 2.5), "kappa_IC95_sup": np.percentile(vals, 97.5),
    })

salvar_tabela(pd.DataFrame(diag_rows), "tabela_CV_repetida_diagnostico_G.csv")

# ====================================================================
# 8. Salvar modelos finais
# ====================================================================
for label, modelo in melhores_modelos.items():
    filename = OUT_DIR / f"modelo_final_{label}_G.pkl"
    joblib.dump(modelo, filename)
    print(f"Salvo: {filename}")

# ====================================================================
# 9. Calibracao minima dos campeoes (Platt/sigmoid), so para obter
#    scores de probabilidade utilizaveis em AUC/ROC -- necessario em
#    especial para o SVM (pipe_svm usa probability=False, identico ao
#    notebook). Nao mexe nos rotulos preditos (predict()) usados nas
#    metricas de Accuracy/Precision/Recall/F1/Kappa acima -- so afeta
#    o score continuo usado para AUC/ROC. O estudo multi-cenario
#    completo de calibracao (Modelo A vs B, bootstrap de ECE por bin)
#    e parte da secao "Extra a metodologia" do notebook, deliberadamente
#    adiada por instrucao do usuario -- nao reproduzida aqui.
# ====================================================================
print()
print("=" * 70)
print("9. Calibracao (Platt/sigmoid) dos campeoes para AUC/ROC")
print("=" * 70)

from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator

    def calibrar(modelo):
        return CalibratedClassifierCV(estimator=FrozenEstimator(modelo), method="sigmoid").fit(X_resampled, y_train_bin)
except ImportError:
    def calibrar(modelo):
        return CalibratedClassifierCV(estimator=modelo, method="sigmoid", cv="prefit").fit(X_resampled, y_train_bin)

calibrados = {label: calibrar(modelo) for label, modelo in melhores_modelos.items()}

roc_rows = []
for label, modelo in melhores_modelos.items():
    y_train_score_cal = calibrados[label].predict_proba(X_resampled)[:, 1]
    y_test_score_cal = calibrados[label].predict_proba(X_test)[:, 1]

    auc_train_cal = roc_auc_score(y_train_bin, y_train_score_cal)
    auc_test_cal = roc_auc_score(y_test_bin, y_test_score_cal)

    train_df.loc[train_df["Model"] == label, "AUC"] = auc_train_cal
    test_df.loc[test_df["Model"] == label, "AUC"] = auc_test_cal

    fpr, tpr, _ = roc_curve(y_test_bin, y_test_score_cal)
    for f, t in zip(fpr, tpr):
        roc_rows.append({"Model": label, "fpr": f, "tpr": t})

    print(f"  {label:<10} AUC treino (calibrado)={auc_train_cal:.4f}  AUC teste (calibrado)={auc_test_cal:.4f}")

salvar_tabela(train_df, "tabela_metricas_treino_G.csv")
salvar_tabela(test_df, "tabela_metricas_teste_externo_G.csv")
salvar_tabela(pd.DataFrame(roc_rows), "tabela_ROC_pontos_G.csv")

for label, modelo in calibrados.items():
    joblib.dump(modelo, OUT_DIR / f"modelo_final_{label}_calibrado_G.pkl")

# ====================================================================
# 10. Importancia de features do campeao XGBoost (gain/cover), mesma
#     tecnica de extracao do notebook (booster.get_score), mas aplicada
#     ao CAMPEAO real da busca bayesiana por kappa (cell 116-117), nao
#     ao modelo D1 log-loss/early-stopping separado (esse pertence ao
#     ramo exploratorio "Modelo C/D1", fora do escopo pedido agora).
# ====================================================================
print()
print("=" * 70)
print("10. Importancia de features (gain/cover) do campeao XGBoost")
print("=" * 70)

xgb_champion = melhores_modelos["XGBoost"].named_steps["xgb"]
booster = xgb_champion.get_booster()

importance_types = ["gain", "total_gain", "cover", "total_cover"]
importance_data = {imp: booster.get_score(importance_type=imp) for imp in importance_types}

importance_df = pd.DataFrame.from_dict(importance_data)
importance_df.index.name = "Feature"
importance_df = importance_df.reset_index()
importance_df[importance_types] = importance_df[importance_types].fillna(0)
importance_df = importance_df.sort_values(by="total_gain", ascending=False).reset_index(drop=True)

# Garante presenca de TODOS os 57 descritores (mesmo os nao usados em
# nenhum split, importancia = 0) para a figura do conjunto completo.
todas_features = pd.DataFrame({"Feature": X_resampled.columns})
importance_full = todas_features.merge(importance_df, on="Feature", how="left").fillna(0)
importance_full = importance_full.sort_values(by="gain", ascending=False).reset_index(drop=True)

top20_df = importance_df.head(20)
salvar_tabela(top20_df, "tabela5_feature_importance_top20_G.csv")
salvar_tabela(importance_full, "tabela_feature_importance_completa_57_G.csv")

print(top20_df.to_string(index=False))

# ====================================================================
# 11. Figuras em ingles, aproveitaveis diretamente no Artigo 1
# ====================================================================
print()
print("=" * 70)
print("11. Figuras (ingles) para o Artigo 1")
print("=" * 70)

FIG_DIR = OUT_DIR / "figuras_ingles_G"
FIG_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {"MLP": "#d62728", "SVM": "#2ca02c", "XGBoost": "#1f77b4"}

# --- Fig. G1: SVMSMOTE synthetic-sample diagnostics (replaces the
#     Portuguese diagnostic figure from the notebook; 4-panel layout) ---
if len(synthetic_metadata) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    axes = axes.ravel()

    sim_values = synthetic_metadata["similarity"].to_numpy(dtype=float)
    bins = np.linspace(similaridade_minima - 0.01, similaridade_maxima + 0.01, 28)

    axes[0].hist(sim_values, bins=bins, color="#6a3d9a", edgecolor="white", alpha=0.9)
    axes[0].axvspan(similaridade_minima, similaridade_maxima, color="green", alpha=0.08)
    axes[0].axvline(similaridade_minima, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(similaridade_maxima, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Selected synthetic-sample similarities")
    axes[0].set_xlabel("Mixed-distance similarity to nearest Inactive parent")
    axes[0].set_ylabel("Frequency")

    axes[1].boxplot(sim_values, showfliers=False, patch_artist=True,
                     boxprops=dict(facecolor="#cab2d6"))
    axes[1].axhspan(similaridade_minima, similaridade_maxima, color="green", alpha=0.08)
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
    print(f"Salvo: {FIG_DIR / 'figG1_svmsmote_diagnostics.png'}")

# --- Fig. G2: Bayesian-search candidate dispersion per model family
#     (replaces the 20-candidate MLP/SVM/XGBoost comparison figures;
#     here n_iter differs by family: 5/15/5, per the refined pipeline) ---
dispersion_rows = []
for label, bscv in bscv_objects.items():
    cvr = bscv.cv_results_
    for idx, (mean_score, std_score) in enumerate(zip(cvr["mean_test_score"], cvr["std_test_score"])):
        dispersion_rows.append({
            "Model": label, "candidate_idx": idx, "mean_cv_kappa": mean_score,
            "std_cv_kappa": std_score, "is_best": mean_score == bscv.best_score_,
        })
dispersion_df = pd.DataFrame(dispersion_rows)
salvar_tabela(dispersion_df, "tabela3_dispersao_candidatos_bayesianos_G.csv")

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
ax.set_title("Bayesian-search candidate dispersion by model family\n(n_iter = 5 / 5 / 15 for MLP / SVM / XGBoost)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "figG2_bayes_search_dispersion.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG2_bayes_search_dispersion.png'}")

# --- Fig. G3: Repeated group-CV (20x5) robustness of the chosen
#     candidate -- supports "the optimisation itself did not overfit
#     to CV noise" (repeated estimate close to the search's best_score_) --
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
rep_data = []
rep_labels = []
for label in order:
    scores = cross_validate(melhores_modelos[label], X_resampled, y_train_bin, cv=cv_repetida, groups=groups,
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
print(f"Salvo: {FIG_DIR / 'figG3_repeated_cv_robustness.png'}")

# --- Fig. G4 (replaces Fig. 7): ROC curves on the held-out test set,
#     calibrated scores, all three retained classifiers ---
fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
for label in order:
    y_score = calibrados[label].predict_proba(X_test)[:, 1]
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
print(f"Salvo: {FIG_DIR / 'figG4_roc_curves.png'}")

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
print(f"Salvo: {FIG_DIR / 'figG5_feature_importance_full.png'}")

# --- Fig. G6 (replaces Fig. 4): class distribution across the 70/30
#     split AND the SVMSMOTE-balanced training partition, in English ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
split_counts = pd.DataFrame({
    "Train (pre-resampling)": y_orig.value_counts(),
    "Test": y_test["Atividade"].value_counts(),
    "Train (post-SVMSMOTE)": y_train_final.value_counts(),
}).reindex(["Ativo", "Inativo"]).rename(index={"Ativo": "Active", "Inativo": "Inactive"})
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
print(f"Salvo: {FIG_DIR / 'figG6_class_distribution.png'}")

print()
print("=" * 70)
print("PIPELINE VERSAO G CONCLUIDO")
print("=" * 70)
