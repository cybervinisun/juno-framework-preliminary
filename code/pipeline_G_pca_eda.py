"""
Pipeline G - Parte C: Analise exploratoria de dados (EDA) e Analise de
Componentes Principais (ACP), para a nova secao do Artigo 1 sobre a
natureza dos dados (descritores continuos + fingerprints PLIP).

Reusa exatamente a mesma matriz canonica de 320 ligantes e o mesmo
split/normalizacao do pipeline_G.py (mesmos seeds), sem re-treinar
nenhum modelo -- e puramente descritivo.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "processed"))

X_PATH = DATA_DIR / "X_320ligands_57descriptors.xlsx"
Y_PATH = DATA_DIR / "y_320ligands_labels.xlsx"

OUT_DIR = Path(__file__).parent / "versao_G_outputs"
FIG_DIR = OUT_DIR / "figuras_ingles_G"
FIG_DIR.mkdir(exist_ok=True)

df4 = pd.read_excel(X_PATH, index_col=0)
y = pd.read_excel(Y_PATH, index_col=0)

X_final = df4.copy()
y_final = y.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.30, stratify=y_final["Atividade"], random_state=42,
)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

descritores_ja_normalizados = ["corrScore"]
col_binarias = [c for c in X_train.columns if set(X_train[c].dropna().unique()) <= {0, 1}]
col_continuas = [c for c in X_train.columns if c not in col_binarias and c not in descritores_ja_normalizados]

scaler = MinMaxScaler()
scaler.fit(X_train[col_continuas])
X_train[col_continuas] = scaler.transform(X_train[col_continuas])
X_test[col_continuas] = scaler.transform(X_test[col_continuas])

print(f"Continuous descriptors: {len(col_continuas)} | Binary PLIP fingerprints: {len(col_binarias)}")

# ====================================================================
# EDA 1: class-conditional distributions of the 9 continuous descriptors
# ====================================================================
continuous_cols_pca = [c for c in X_train.columns if c not in col_binarias]
X_eda = pd.concat([X_train.reset_index(drop=True), y_train["Atividade"].reset_index(drop=True)], axis=1)

n_cont = len(continuous_cols_pca)
ncols = 3
nrows = int(np.ceil(n_cont / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), dpi=150)
axes = np.atleast_1d(axes).ravel()
palette = {"Ativo": "#d62728", "Inativo": "#1f77b4"}
for i, col in enumerate(continuous_cols_pca):
    ax = axes[i]
    for cls, color in palette.items():
        vals = X_eda.loc[X_eda["Atividade"] == cls, col]
        ax.hist(vals, bins=20, alpha=0.55, color=color, label=cls, density=True)
    ax.set_title(col, fontsize=9)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)
for j in range(n_cont, len(axes)):
    axes[j].axis("off")
fig.suptitle("Class-conditional distributions of the nine continuous descriptors\n(training partition, min-max normalised)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG9_descriptor_distributions.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG9_descriptor_distributions.png'}")

# ====================================================================
# EDA 2: correlation heatmap of the 9 continuous descriptors
# ====================================================================
corr = X_train[continuous_cols_pca].corr()
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(continuous_cols_pca)))
ax.set_xticklabels(continuous_cols_pca, rotation=90, fontsize=8)
ax.set_yticks(range(len(continuous_cols_pca)))
ax.set_yticklabels(continuous_cols_pca, fontsize=8)
for i in range(len(continuous_cols_pca)):
    for j in range(len(continuous_cols_pca)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Correlation structure of the nine continuous descriptors\n(training partition)")
fig.tight_layout()
fig.savefig(FIG_DIR / "figG10_continuous_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG10_continuous_correlation_heatmap.png'}")

# ====================================================================
# EDA 3: binary PLIP fingerprint prevalence (overall and by class)
# ====================================================================
binary_prevalence = pd.DataFrame({
    "Feature": col_binarias,
    "Overall_prevalence": X_train[col_binarias].mean().to_numpy(),
    "Active_prevalence": X_train.loc[y_train["Atividade"] == "Ativo", col_binarias].mean().to_numpy(),
    "Inactive_prevalence": X_train.loc[y_train["Atividade"] == "Inativo", col_binarias].mean().to_numpy(),
}).sort_values("Overall_prevalence", ascending=False)
binary_prevalence.to_csv(OUT_DIR / "tabela_plip_fingerprint_prevalence_G.csv", index=False)
print(f"[tabela salva] {OUT_DIR / 'tabela_plip_fingerprint_prevalence_G.csv'}")

fig, ax = plt.subplots(figsize=(9, 12), dpi=150)
plot_df = binary_prevalence.sort_values("Overall_prevalence")
y_pos = np.arange(len(plot_df))
ax.barh(y_pos - 0.2, plot_df["Active_prevalence"], height=0.4, color="#d62728", label="Active")
ax.barh(y_pos + 0.2, plot_df["Inactive_prevalence"], height=0.4, color="#1f77b4", label="Inactive")
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["Feature"], fontsize=6)
ax.set_xlabel("Prevalence (fraction of training-set ligands with this contact)")
ax.set_title("PLIP interaction-fingerprint prevalence by class\n(training partition, 48 binary descriptors)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "figG11_plip_prevalence_by_class.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG11_plip_prevalence_by_class.png'}")

# ====================================================================
# PCA: scree plot + PC1xPC2 scatter coloured by class + loadings
# ====================================================================
pca = PCA(n_components=0.999, svd_solver="full")
pca_scores = pca.fit_transform(X_train[continuous_cols_pca])

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
axes[0].bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, color="#4292c6", label="Individual")
axes[0].plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), "o-", color="#08519c", label="Cumulative")
axes[0].axhline(0.999, color="grey", linestyle="--", linewidth=1)
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Explained variance ratio")
axes[0].set_title("Scree plot (9 continuous descriptors)")
axes[0].legend(fontsize=8)

for cls, color in palette.items():
    mask = (y_train["Atividade"] == cls).to_numpy()
    axes[1].scatter(pca_scores[mask, 0], pca_scores[mask, 1], s=18, alpha=0.65, color=color, label=cls)
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
axes[1].set_title("PC1 vs. PC2 scores, coloured by class")
axes[1].legend(fontsize=8)

fig.suptitle("Principal component analysis of the nine continuous descriptors (training partition)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG12_pca_scree_scores.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG12_pca_scree_scores.png'}")

loadings = pd.DataFrame(pca.components_.T, index=continuous_cols_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
im = ax.imshow(loadings[["PC1", "PC2", "PC3"]].to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(3))
ax.set_xticklabels(["PC1", "PC2", "PC3"])
ax.set_yticks(range(len(continuous_cols_pca)))
ax.set_yticklabels(continuous_cols_pca, fontsize=8)
for i in range(len(continuous_cols_pca)):
    for j in range(3):
        ax.text(j, i, f"{loadings.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="Loading")
ax.set_title("PCA loadings (PC1-PC3) of the nine continuous descriptors")
fig.tight_layout()
fig.savefig(FIG_DIR / "figG13_pca_loadings.png", bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {FIG_DIR / 'figG13_pca_loadings.png'}")

pca_var_df = pd.DataFrame({
    "componente": [f"PC{i+1}" for i in range(pca.n_components_)],
    "variancia_explicada": pca.explained_variance_ratio_,
    "variancia_acumulada": np.cumsum(pca.explained_variance_ratio_),
})
pca_var_df.to_csv(OUT_DIR / "tabela_ACP_variancia_G.csv", index=False)
loadings.reset_index().rename(columns={"index": "descritor"}).to_csv(OUT_DIR / "tabela_ACP_loadings_G.csv", index=False)

print()
print("PCA summary:")
print(pca_var_df.to_string(index=False))
print()
print("CONCLUIDO: EDA + ACP")
