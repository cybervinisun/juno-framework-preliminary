"""
Pipeline G - Parte E (revisada): cenarios de calibracao Modelo A (treinado
COM sinteticas SVMSMOTE, X_resampled) vs Modelo B (treinado SEM sinteticas,
apenas X_orig), cada um comparado sem calibracao, calibrado em X_orig, e
calibrado em X_resampled -- reproduzindo fielmente o design experimental
do notebook RECONSTRUIDO (`build_scenarios`/`build_metrics_table`/
`plot_calibration_flow`, secao "Extra a metodologia"), agora aplicado aos
3 algoritmos da versao G (o notebook original so fazia isso para o MLP).

Substitui a comparacao anterior (calibracao leak-free via CV agrupada de
5 folds pequenos) -- que o usuario corretamente apontou como
metodologicamente fragil neste tamanho de amostra (cerca de 45-67
instancias por fold de calibracao). Aqui, os conjuntos de calibracao sao
muito maiores (224 ou 334 instancias, o conjunto inteiro), evitando esse
problema de variancia, ao custo de reintroduzir alguma sobreposicao entre
dados de ajuste e de calibracao em alguns cenarios -- exatamente o
trade-off que este estudo foi desenhado para expor, nao esconder.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "versao_G_outputs"
FIG_DIR = OUT_DIR / "figuras_ingles_G"
FIG_DIR.mkdir(exist_ok=True)

ckpt = joblib.load(OUT_DIR / "checkpoint_post_svmsmote_G.pkl")
X_train_final = ckpt["X_train_final"]
y_train_final = ckpt["y_train_final"]
X_test = ckpt["X_test"]
y_test = ckpt["y_test"]
tracking_table = ckpt["tracking_table"]

X_resampled = X_train_final.copy()
mapeamento = {"Inativo": 0, "Ativo": 1}
y_train_bin = np.array([mapeamento[c] for c in y_train_final], dtype=np.int64)
y_test_bin = np.array([mapeamento[c] for c in y_test["Atividade"]], dtype=np.int64)

is_orig = (tracking_table["is_synthetic"] == False).to_numpy()
X_orig = X_resampled.loc[is_orig].reset_index(drop=True)
y_orig_bin = y_train_bin[is_orig]
print(f"X_orig (sem sinteticas): {X_orig.shape}  |  X_resampled (com sinteticas): {X_resampled.shape}")

hp = pd.read_csv(OUT_DIR / "tabela_hiperparametros_campeoes_G.csv").set_index("Model")


def fresh_pipeline(label: str):
    row = hp.loc[label]
    if label == "MLP":
        return Pipeline(steps=[("NN", MLPClassifier(
            solver="lbfgs", max_iter=20000, random_state=23,
            activation=row["NN__activation"], alpha=float(row["NN__alpha"]),
            hidden_layer_sizes=int(row["NN__hidden_layer_sizes"]),
            learning_rate_init=float(row["NN__learning_rate_init"]),
        ))])
    if label == "XGBoost":
        return Pipeline(steps=[("xgb", XGBClassifier(
            random_state=0, booster="gbtree", objective="binary:logistic",
            colsample_bytree=float(row["xgb__colsample_bytree"]), gamma=float(row["xgb__gamma"]),
            learning_rate=float(row["xgb__learning_rate"]), max_depth=int(row["xgb__max_depth"]),
            max_leaves=int(row["xgb__max_leaves"]), min_child_weight=float(row["xgb__min_child_weight"]),
            n_estimators=int(row["xgb__n_estimators"]), reg_alpha=float(row["xgb__reg_alpha"]),
            reg_lambda=float(row["xgb__reg_lambda"]), subsample=float(row["xgb__subsample"]),
        ))])
    if label == "SVM":
        return Pipeline(steps=[("svm", svm.SVC(
            gamma=float(row["svm__gamma"]), C=float(row["svm__C"]),
            kernel=row["svm__kernel"], max_iter=-1, probability=False,
        ))])
    raise ValueError(label)


def calibrar(modelo_ajustado, X_c, y_c):
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(estimator=FrozenEstimator(modelo_ajustado), method="sigmoid").fit(X_c, y_c)
    except ImportError:
        return CalibratedClassifierCV(estimator=modelo_ajustado, method="sigmoid", cv="prefit").fit(X_c, y_c)


def native_score(modelo_ajustado, X):
    """Score nativo (sem calibracao). MLP/XGBoost: predict_proba (link
    logistico nativo). SVM (probability=False): decision_function
    min-max normalizada -- e um escore, NAO uma probabilidade; usado
    apenas para completar o cenario "1" (sem calibracao) de forma
    honesta e claramente identificada como tal."""
    step = modelo_ajustado.steps[-1][1]
    if hasattr(step, "predict_proba"):
        return modelo_ajustado.predict_proba(X)[:, 1]
    raw = modelo_ajustado.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min())


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask_bin = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask_bin.sum() == 0:
            continue
        ece += (mask_bin.sum() / n) * abs(y_prob[mask_bin].mean() - y_true[mask_bin].mean())
    return float(ece)


all_rows = []
reliability_by_model = {}

for label in ["MLP", "SVM", "XGBoost"]:
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    # Modelo A: ja treinado (campeao versao G), sobre X_resampled.
    modelo_a = joblib.load(OUT_DIR / f"modelo_final_{label}_G.pkl")

    # Modelo B: MESMOS hiperparametros vencedores, mas treinado SOMENTE
    # em X_orig (sem sinteticas SVMSMOTE) -- novo fit necessario aqui.
    modelo_b = fresh_pipeline(label)
    modelo_b.fit(X_orig, y_orig_bin)

    scenarios = {}
    is_svm = label == "SVM"

    # A1 / B1: sem calibracao (score nativo)
    scenarios["A1"] = {"model": "A", "label": "A1 (sem calibracao)",
                        "scores": native_score(modelo_a, X_test), "is_probability": not is_svm}
    scenarios["B1"] = {"model": "B", "label": "B1 (sem calibracao)",
                        "scores": native_score(modelo_b, X_test), "is_probability": not is_svm}

    # A2 / A3: modelo A calibrado em X_orig / X_resampled
    scenarios["A2"] = {"model": "A", "label": "A2 (calibrado em originais)",
                        "scores": calibrar(modelo_a, X_orig, y_orig_bin).predict_proba(X_test)[:, 1], "is_probability": True}
    scenarios["A3"] = {"model": "A", "label": "A3 (calibrado em balanceada)",
                        "scores": calibrar(modelo_a, X_resampled, y_train_bin).predict_proba(X_test)[:, 1], "is_probability": True}

    # B2 / B3: modelo B calibrado em X_resampled / X_orig
    scenarios["B2"] = {"model": "B", "label": "B2 (calibrado em balanceada)",
                        "scores": calibrar(modelo_b, X_resampled, y_train_bin).predict_proba(X_test)[:, 1], "is_probability": True}
    scenarios["B3"] = {"model": "B", "label": "B3 (calibrado em originais)",
                        "scores": calibrar(modelo_b, X_orig, y_orig_bin).predict_proba(X_test)[:, 1], "is_probability": True}

    for sid, item in scenarios.items():
        scores = np.asarray(item["scores"], dtype=float)
        row = {
            "Algorithm": label, "Scenario": sid, "Description": item["label"],
            "AUC": roc_auc_score(y_test_bin, scores),
            "Brier": brier_score_loss(y_test_bin, scores) if item["is_probability"] else np.nan,
            "ECE": expected_calibration_error(y_test_bin, scores) if item["is_probability"] else np.nan,
            "LogLoss": log_loss(y_test_bin, np.clip(scores, 1e-7, 1 - 1e-7)) if item["is_probability"] else np.nan,
            "is_probability": item["is_probability"],
        }
        all_rows.append(row)
        flag = "" if item["is_probability"] else "  [decision score, not a probability]"
        print(f"  {sid} {item['label']:<32} AUC={row['AUC']:.4f} Brier={row['Brier']:.4f} "
              f"ECE={row['ECE']:.4f} LogLoss={row['LogLoss']:.4f}{flag}")

    reliability_by_model[label] = scenarios

results_df = pd.DataFrame(all_rows)
results_df.to_csv(OUT_DIR / "tabela_cenarios_calibracao_G.csv", index=False)
print(f"\n[tabela salva] {OUT_DIR / 'tabela_cenarios_calibracao_G.csv'}")

# ====================================================================
# Figuras: reliability diagram, Modelo A (com sinteticas) vs Modelo B
# (sem sinteticas), por algoritmo -- 1 figura por algoritmo, 2 paineis
# (A, B) cada, replicando o design do notebook original.
# ====================================================================
COLORS = {"A1": "#d62728", "A2": "#1f77b4", "A3": "#ff7f0e",
          "B1": "#2ca02c", "B2": "#9467bd", "B3": "#8c564b"}

for label, scenarios in reliability_by_model.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    for ax, model_id, title in zip(axes, ["A", "B"],
                                    [f"{label}: trained WITH synthetics (Model A)",
                                     f"{label}: trained WITHOUT synthetics (Model B)"]):
        for sid, item in scenarios.items():
            if item["model"] != model_id or not item["is_probability"]:
                continue
            scores = np.asarray(item["scores"], dtype=float)
            frac, pred = calibration_curve(y_test_bin, scores, n_bins=5, strategy="quantile")
            ax.plot(pred, frac, "o-", label=item["label"], color=COLORS[sid], linewidth=2)
        ax.plot([0, 1], [0, 1], ":", color="grey", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency of Active")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle(f"Calibration scenarios on the held-out test set (n = 96): {label}", fontsize=12)
    fig.tight_layout()
    fname = f"figG20_calibration_scenarios_{label.lower()}.png"
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {FIG_DIR / fname}")

print()
print("=" * 70)
print("CONCLUIDO: cenarios de calibracao Modelo A/B (3 algoritmos)")
print("=" * 70)
