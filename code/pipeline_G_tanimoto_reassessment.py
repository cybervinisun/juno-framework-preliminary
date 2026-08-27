"""
Pipeline G - Parte F: reavaliacao da analise estrutural de erros
(Tanimoto/MCS) contra as classificacoes ERRADAS REAIS dos campeoes da
versao G -- substitui a narrativa antiga (composto 55/130/243, do
pipeline pre-versao-G) por uma analise fiel ao modelo atualmente
reportado nas Tabelas 2/4/5 do Artigo 1.
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

OUT_DIR = Path(__file__).parent / "versao_G_outputs"
FIG_DIR = OUT_DIR / "figuras_ingles_G"

# ====================================================================
# 1. Reconstruir o split exato (random_state=42), preservando o indice
#    ORIGINAL (1..320) para recuperar os SMILES reais dos compostos.
# ====================================================================
df4 = pd.read_excel(X_PATH, index_col=0)
y = pd.read_excel(Y_PATH, index_col=0)
smiles_df = pd.read_excel(SMILES_PATH, index_col=0)

assert (df4.index == y.index).all()
assert (df4.index == smiles_df.index).all()

X_final = df4.copy()
y_final = y.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.30, stratify=y_final["Atividade"], random_state=42,
)
# NAO resetar o indice desta vez -- precisamos do ID original (1..320)
# para recuperar os SMILES de cada composto do conjunto de teste.

descritores_ja_normalizados = ["corrScore"]
col_binarias = [c for c in X_train.columns if set(X_train[c].dropna().unique()) <= {0, 1}]
col_continuas = [c for c in X_train.columns if c not in col_binarias and c not in descritores_ja_normalizados]

scaler = joblib.load(OUT_DIR / "scaler_minmax_treino_G.pkl")
X_test_scaled = X_test.copy()
X_test_scaled[col_continuas] = scaler.transform(X_test[col_continuas])

mapeamento = {"Inativo": 0, "Ativo": 1}
y_test_bin = pd.Series([mapeamento[c] for c in y_test["Atividade"]], index=y_test.index)

# ====================================================================
# 2. Previsoes dos 3 campeoes versao G no conjunto de teste (mesma
#    ordem/indice original preservado)
# ====================================================================
print("=" * 70)
print("Identificando os erros REAIS dos campeoes versao G no teste")
print("=" * 70)

erros_por_modelo = {}
for label in ["MLP", "SVM", "XGBoost"]:
    champion = joblib.load(OUT_DIR / f"modelo_final_{label}_G.pkl")
    y_pred = champion.predict(X_test_scaled)
    y_pred_series = pd.Series(y_pred, index=X_test_scaled.index)

    fp_idx = y_pred_series[(y_pred_series == 1) & (y_test_bin == 0)].index.tolist()
    fn_idx = y_pred_series[(y_pred_series == 0) & (y_test_bin == 1)].index.tolist()
    erros_por_modelo[label] = {"FP": fp_idx, "FN": fn_idx}
    print(f"{label}: {len(fp_idx)} falsos positivos {fp_idx}, {len(fn_idx)} falsos negativos {fn_idx}")

todos_erros = set()
for label, d in erros_por_modelo.items():
    todos_erros.update(d["FP"])
    todos_erros.update(d["FN"])
todos_erros = sorted(todos_erros)
print(f"\nUniao de todos os compostos mal classificados por ao menos 1 modelo: {todos_erros}")

compartilhados = set(erros_por_modelo["MLP"]["FP"] + erros_por_modelo["MLP"]["FN"])
for label in ["SVM", "XGBoost"]:
    compartilhados &= set(erros_por_modelo[label]["FP"] + erros_por_modelo[label]["FN"])
print(f"Compostos mal classificados pelos 3 modelos simultaneamente: {sorted(compartilhados)}")

# ====================================================================
# 3. Fingerprints de Morgan (raio 2, 2048 bits) para toda a base de
#    treino (Ativo/Inativo) + compostos mal classificados
# ====================================================================
def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

smiles_df["fp"] = smiles_df["Smiles"].apply(smiles_to_fp)
n_falhas = smiles_df["fp"].isna().sum()
print(f"\nSMILES que falharam ao gerar fingerprint: {n_falhas}/320")

train_ids = X_train.index
active_ids = [i for i in train_ids if smiles_df.loc[i, "Atividade"] == "Ativo" and smiles_df.loc[i, "fp"] is not None]
inactive_ids = [i for i in train_ids if smiles_df.loc[i, "Atividade"] == "Inativo" and smiles_df.loc[i, "fp"] is not None]
print(f"Populacao de referencia (treino): {len(active_ids)} Ativos, {len(inactive_ids)} Inativos")

resultados_rows = []
for idx in todos_erros:
    fp_erro = smiles_df.loc[idx, "fp"]
    if fp_erro is None:
        print(f"AVISO: composto {idx} sem fingerprint valido -- pulado.")
        continue

    sims_active = DataStructs.BulkTanimotoSimilarity(fp_erro, [smiles_df.loc[i, "fp"] for i in active_ids])
    sims_inactive = DataStructs.BulkTanimotoSimilarity(fp_erro, [smiles_df.loc[i, "fp"] for i in inactive_ids])

    modelos_que_erram = [label for label, d in erros_por_modelo.items() if idx in d["FP"] or idx in d["FN"]]
    tipo_erro = {
        label: ("FP" if idx in erros_por_modelo[label]["FP"] else "FN")
        for label in modelos_que_erram
    }

    resultados_rows.append({
        "compound_idx": idx,
        "true_activity": smiles_df.loc[idx, "Atividade"],
        "misclassified_by": ", ".join(modelos_que_erram),
        "error_type_by_model": str(tipo_erro),
        "median_tanimoto_active": float(np.median(sims_active)),
        "max_tanimoto_active": float(np.max(sims_active)),
        "median_tanimoto_inactive": float(np.median(sims_inactive)),
        "max_tanimoto_inactive": float(np.max(sims_inactive)),
        "smiles": smiles_df.loc[idx, "Smiles"],
    })

resultados_df = pd.DataFrame(resultados_rows).sort_values("compound_idx")
resultados_df.to_csv(OUT_DIR / "tabela_tanimoto_erros_versaoG.csv", index=False)
print(f"\n[tabela salva] {OUT_DIR / 'tabela_tanimoto_erros_versaoG.csv'}")
print(resultados_df[["compound_idx", "true_activity", "misclassified_by", "median_tanimoto_active", "max_tanimoto_active", "median_tanimoto_inactive", "max_tanimoto_inactive"]].to_string(index=False))

# ====================================================================
# 4. Similaridade par-a-par entre os proprios compostos mal classificados
# ====================================================================
print("\nSimilaridade par-a-par entre os compostos mal classificados:")
pares_rows = []
validos = [idx for idx in todos_erros if smiles_df.loc[idx, "fp"] is not None]
for a, b in combinations(validos, 2):
    sim = DataStructs.TanimotoSimilarity(smiles_df.loc[a, "fp"], smiles_df.loc[b, "fp"])
    pares_rows.append({"compound_a": a, "compound_b": b, "tanimoto": sim})
    print(f"  {a} x {b}: {sim:.3f}")
pares_df = pd.DataFrame(pares_rows)
pares_df.to_csv(OUT_DIR / "tabela_tanimoto_pares_erros_versaoG.csv", index=False)

# ====================================================================
# 5. Figura: distribuicao Tanimoto para cada composto mal classificado
#    vs. populacoes Ativo/Inativo (mesmo estilo da figura antiga)
# ====================================================================
n_err = len(validos)
fig, axes = plt.subplots(1, n_err, figsize=(4.5 * n_err, 4.5), dpi=150, sharey=True)
if n_err == 1:
    axes = [axes]

for ax, idx in zip(axes, validos):
    fp_erro = smiles_df.loc[idx, "fp"]
    sims_active = DataStructs.BulkTanimotoSimilarity(fp_erro, [smiles_df.loc[i, "fp"] for i in active_ids])
    sims_inactive = DataStructs.BulkTanimotoSimilarity(fp_erro, [smiles_df.loc[i, "fp"] for i in inactive_ids])
    ax.hist(sims_active, bins=20, alpha=0.6, color="#d62728", label="to Active", density=True)
    ax.hist(sims_inactive, bins=20, alpha=0.6, color="#1f77b4", label="to Inactive", density=True)
    modelos = [label for label, d in erros_por_modelo.items() if idx in d["FP"] or idx in d["FN"]]
    ax.set_title(f"Compound {idx} (true {smiles_df.loc[idx,'Atividade']})\nmisclassified by: {', '.join(modelos)}", fontsize=9)
    ax.set_xlabel("Tanimoto similarity")
    if ax is axes[0]:
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

fig.suptitle("Tanimoto similarity of version-G misclassified test-set compounds\nto the training Active/Inactive populations", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "figG15_tanimoto_errors_versionG.png", bbox_inches="tight")
plt.close(fig)
print(f"\nSalvo: {FIG_DIR / 'figG15_tanimoto_errors_versionG.png'}")

print()
print("=" * 70)
print("CONCLUIDO: reavaliacao Tanimoto/MCS dos erros -- versao G")
print("=" * 70)
