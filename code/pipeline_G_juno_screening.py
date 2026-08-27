"""
Pipeline G - Parte D: aplica os modelos curados/calibrados da versao G
(MLP/SVM/XGBoost, treinados no banco de 320 ligantes) sobre a
quimioteca prospectiva de 713 compostos ("v6"), atualizando a "Triagem
JUNO" para usar os modelos versao G em vez dos antigos.

Correcao metodologica em relacao ao notebook original da Triagem JUNO:
o notebook antigo re-ajustava (fit) um novo MinMaxScaler sobre a propria
quimioteca a cada inferencia, em vez de reutilizar o scaler ajustado no
treino. Aqui, o scaler `scaler_minmax_treino_G.pkl` (ajustado apenas no
treino, Secao 2.6 do Artigo 1) e aplicado via transform() apenas --
nunca re-ajustado -- exatamente como e feito para o conjunto de teste
interno do artigo.

Nao reproduz a cascata de filtros ADME/Tox (SwissADME/ADMETlab3) do
notebook original -- produz a lista ranqueada de "Ativo" previsto pelos
3 modelos versao G, pronta para alimentar essa cascata (ja implementada
e nao repetida aqui) como proximo passo.

NOTA (anonimizacao): a quimioteca prospectiva de 713 compostos foi
sintetizada em laboratorio (nao e apenas literatura publica) e ainda
nao possui validacao experimental para o alvo GluN1-GluN2B/NMDA. Para
evitar divulgar publicamente o par composto-real -> predicao-de-atividade
para esse alvo (o que poderia comprometer a novidade de um eventual
pedido de patente de novo uso), os arquivos deste repositorio sob
data/raw/prospective_library_713_* tem SMILES/nomes reais removidos e
substituidos por um identificador sequencial anonimo `compound_id`
(0..712, mesma ordem de linhas). Descritores e predicoes permanecem
inalterados -- o pipeline e totalmente reprodutivel, apenas a identidade
quimica real de cada composto nao e publica.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "raw"))
DATA_PROCESSED_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "processed"))

LIB_DIR = DATA_RAW_DIR
X_LIB_PATH = LIB_DIR / "prospective_library_713_descriptors_raw.xlsx"
LABELS_LIB_PATH = LIB_DIR / "prospective_library_713_labels_anonymized.xlsx"

TRAIN_X_PATH = DATA_PROCESSED_DIR / "X_320ligands_57descriptors.xlsx"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models"))
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "results" / "regenerated"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Carregamento da quimioteca prospectiva (713 compostos, v6)")
print("=" * 70)

X_lib = pd.read_excel(X_LIB_PATH)
labels_lib = pd.read_excel(LABELS_LIB_PATH)

assert len(X_lib) == len(labels_lib) == 713, "Tamanho inesperado da quimioteca."

# Alinhamento por posicao (ambos os arquivos carregam a mesma coluna
# `compound_id`, 0..712, na mesma ordem de linhas -- a quimioteca
# publicada neste repositorio e anonimizada: SMILES/nomes reais foram
# removidos e substituidos por esse identificador sequencial, ver
# data/raw/NOTES.md).
compound_id = X_lib["compound_id"].reset_index(drop=True)
X_lib = X_lib.drop(columns=["compound_id"]).reset_index(drop=True)
labels_lib = labels_lib.drop(columns=["compound_id"]).reset_index(drop=True)

train_cols = pd.read_excel(TRAIN_X_PATH, index_col=0, nrows=0).columns.tolist()
missing = set(train_cols) - set(X_lib.columns)
extra = set(X_lib.columns) - set(train_cols)
print(f"Colunas faltantes na quimioteca em relacao ao treino: {missing or 'nenhuma'}")
print(f"Colunas extras na quimioteca em relacao ao treino: {extra or 'nenhuma'}")

X_lib = X_lib.reindex(columns=train_cols)
print(f"Quimioteca reindexada para as {len(train_cols)} colunas do treino (versao G).")

# ====================================================================
# Normalizacao: reutiliza o scaler ajustado SOMENTE no treino (Secao 2.6
# do Artigo 1) -- nunca re-ajustado sobre a quimioteca.
# ====================================================================
scaler = joblib.load(MODEL_DIR / "scaler_minmax_treino_G.pkl")
col_binarias = [c for c in X_lib.columns if set(X_lib[c].dropna().unique()) <= {0, 1}]
descritores_ja_normalizados = ["corrScore"]
col_continuas = [c for c in X_lib.columns if c not in col_binarias and c not in descritores_ja_normalizados]

col_continuas_scaler_fit = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else col_continuas
assert set(col_continuas) == set(col_continuas_scaler_fit), (
    f"Descompasso de colunas continuas entre scaler de treino e quimioteca: "
    f"{set(col_continuas) ^ set(col_continuas_scaler_fit)}"
)

X_lib_scaled = X_lib.copy()
X_lib_scaled[col_continuas] = scaler.transform(X_lib[col_continuas])

fora_dos_limites = ((X_lib_scaled[col_continuas] < 0) | (X_lib_scaled[col_continuas] > 1)).sum()
print("Descritores continuos fora de [0,1] apos aplicar o scaler de treino (extrapolacao esperada, quimioteca != treino):")
print(fora_dos_limites[fora_dos_limites > 0])

# ====================================================================
# Predicao com os 3 campeoes versao G (rotulo) + versoes calibradas
# (probabilidade), reutilizando exatamente os modelos salvos/treinados
# no banco de 320 ligantes.
# ====================================================================
print()
print("=" * 70)
print("Predicao com os modelos curados versao G (MLP, SVM, XGBoost)")
print("=" * 70)

resultados = labels_lib.copy()
resultados.insert(0, "compound_id", compound_id)

# NOTA: este repositorio disponibiliza apenas os 3 modelos-base (rotulo
# Ativo/Inativo). As versoes calibradas (probabilidade) usadas no
# notebook original nao sao incluidas aqui -- carregadas apenas se
# `modelo_final_{label}_calibrado_G.pkl` existir em MODEL_DIR (por
# exemplo, se voce mesmo re-treinar/calibrar os modelos).
has_proba = []
for label in ["MLP", "SVM", "XGBoost"]:
    champion = joblib.load(MODEL_DIR / f"modelo_final_{label}_G.pkl")
    pred = champion.predict(X_lib_scaled)
    resultados[f"{label}_predicao"] = np.where(pred == 1, "Ativo", "Inativo")

    calibrado_path = MODEL_DIR / f"modelo_final_{label}_calibrado_G.pkl"
    if calibrado_path.exists():
        calibrado = joblib.load(calibrado_path)
        resultados[f"{label}_probabilidade_ativo"] = calibrado.predict_proba(X_lib_scaled)[:, 1]
        has_proba.append(label)

    n_ativo = int((pred == 1).sum())
    print(f"  {label:<10} previsto Ativo: {n_ativo}/{len(pred)} ({100*n_ativo/len(pred):.1f}%)")

resultados["Consenso_3_modelos_Ativo"] = (
    (resultados["MLP_predicao"] == "Ativo").astype(int)
    + (resultados["SVM_predicao"] == "Ativo").astype(int)
    + (resultados["XGBoost_predicao"] == "Ativo").astype(int)
)
if has_proba:
    resultados["Media_probabilidade_ativo"] = resultados[
        [f"{label}_probabilidade_ativo" for label in has_proba]
    ].mean(axis=1)
else:
    print("\n[aviso] Nenhum modelo calibrado encontrado em MODEL_DIR -- ranking por consenso apenas.")

sort_cols = ["Consenso_3_modelos_Ativo"] + (["Media_probabilidade_ativo"] if has_proba else [])
resultados_ranked = resultados.sort_values(sort_cols, ascending=False).reset_index(drop=True)

out_path = OUT_DIR / "tabela_triagem_JUNO_quimioteca713_versaoG.csv"
resultados_ranked.to_csv(out_path, index=False)
print(f"\n[tabela salva] {out_path}")

n_consenso_3 = int((resultados["Consenso_3_modelos_Ativo"] == 3).sum())
n_consenso_0 = int((resultados["Consenso_3_modelos_Ativo"] == 0).sum())
print(f"\nConsenso dos 3 modelos -- unanime Ativo: {n_consenso_3}/713 ({100*n_consenso_3/713:.1f}%)")
print(f"Consenso dos 3 modelos -- unanime Inativo: {n_consenso_0}/713 ({100*n_consenso_0/713:.1f}%)")
print(f"Discordancia entre os 3 modelos (1 ou 2 votos Ativo): {713 - n_consenso_3 - n_consenso_0}/713")

# ====================================================================
# Comparacao de concordancia com as previsoes do XGBoost/SVC "antigos"
# (pipeline pre-versao-G), ja salvas localmente, se disponiveis.
# ====================================================================
print()
print("=" * 70)
print("Concordancia com as previsoes do pipeline anterior (pre-versao-G)")
print("=" * 70)

# Comparacao opcional, apenas para contexto de escala -- as previsoes do
# pipeline pre-versao-G nao fazem parte deste repositorio. Aponte
# OLD_PIPELINE_DIR para sua propria copia se quiser reproduzir esta secao;
# caso contrario, o bloco abaixo simplesmente reporta "nao encontrado".
OLD_DIR = Path(
    os.environ.get(
        "OLD_PIPELINE_DIR",
        str(REPO_ROOT / "data" / "raw" / "old_pipeline_v6_screening"),
    )
)
try:
    old_xgb_ativos = pd.read_excel(OLD_DIR / "XGB" / "xgb_rotulos_ativos_previstos_v6Fc.xlsx", index_col=0)
    old_svc_ativos = pd.read_excel(OLD_DIR / "SVC" / "svc_rotulos_ativos_previstos_v6Fc.xlsx", index_col=0)
    print(f"Pipeline anterior -- XGBoost previu Ativo: {len(old_xgb_ativos)}/713")
    print(f"Pipeline anterior -- SVC previu Ativo: {len(old_svc_ativos)}/713")
    print(
        "(comparacao index-a-index nao realizada aqui -- os arquivos antigos "
        "usam apenas os SMILES/rotulos dos previstos-ativos, sem indice "
        "posicional comum verificado; reportado apenas para contexto de escala.)"
    )
except FileNotFoundError as e:
    print(f"Arquivos do pipeline anterior nao encontrados para comparacao: {e}")

print()
print("=" * 70)
print("CONCLUIDO: Triagem JUNO versao G (sem cascata ADME/Tox)")
print("=" * 70)
