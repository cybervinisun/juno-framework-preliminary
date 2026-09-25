"""
Reassessment of the REAL GOLD redocking data (GoldScore, N=1000 solutions per
system) for Article 1 -- replaces "visual inspection" language with genuine
quantitative statistics taken directly from the full population of solutions
GOLD generated (the population-statistics spreadsheet's RMSD column, ordered
by fitness rank, "r.m.s.d. (S)").

NOTE: this script re-derives the data from the RAW GOLD population
spreadsheets (one per system), which are NOT part of this repository (an
idiosyncratic per-system GOLD output, not redistributed here for size/format
reasons). If you only need the already-extracted data (the RMSD of each of
the ~1000 solutions per system), use
data/raw/redocking_rmsd_full_population_5systems.csv directly -- this script
only needs to be re-run if you have access to the original GOLD spreadsheets
and want to re-derive or audit the extraction.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directory holding the raw GOLD redocking folders, one subfolder per system
# (see SYSTEMS below). Override it through the GOLD_RAW_DIR environment
# variable if your copy lives elsewhere.
#
# The literal "redocagem"/"Goldscore" path segments below are the folder names
# GOLD wrote on the original workstation and are kept verbatim so the script
# still finds them; they are paths, not translatable text.
D = os.environ.get(
    "GOLD_RAW_DIR",
    str(REPO_ROOT / "data" / "raw" / "gold_redocking_raw"),
)

SYSTEMS = {
    "Ifenprodil (3QEL)": f"{D}/3qel/redocagem/goldscore/gold_(N1000)",
    "Ro25-6981 (3QEM)": f"{D}/3qem/redocagem/Goldscore/gold_(N1000)",
    "EVT-101 (5EWM)": f"{D}/5ewm/redocagem/Goldscore/goldscore(N1000)",
    "Hybrid 93 (6E7R)": f"{D}/6e7r/redocagem/goldscore/gold_(N1000)",
    "Hybrid 93 (6E7U)": f"{D}/6e7u/goldscore/gold_(N1000)",
}

OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "results" / "regenerated"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

RMSD_THRESHOLD = 2.0  # A, standard redocking pose-recovery convention

rows_summary = []
rows_full = []

for name, folder in SYSTEMS.items():
    matches = [m for m in glob.glob(os.path.join(folder, "*.xlsx")) if "popula" in m.lower() or "estat" in m.lower()]
    assert matches, f"Not found for {name}: {folder}"
    xlsx_path = matches[0]

    sheet_names = pd.ExcelFile(xlsx_path).sheet_names
    stat_sheet = None
    for s in sheet_names:
        probe = pd.read_excel(xlsx_path, sheet_name=s, header=None, nrows=3)
        if (probe.iloc[:3, :] == "r.m.s.d. (S)").any().any():
            stat_sheet = s
            break
    assert stat_sheet is not None, f"No sheet containing 'r.m.s.d. (S)' was found in {xlsx_path}"

    df = pd.read_excel(xlsx_path, sheet_name=stat_sheet, header=None)
    # Columns (0-indexed): 13 = fitness ordered by score (S), 14 = RMSD ordered by score (S)
    scores_by_rank = pd.to_numeric(df.iloc[2:, 13], errors="coerce").dropna().to_numpy()
    rmsd_by_rank = pd.to_numeric(df.iloc[2:, 14], errors="coerce").dropna().to_numpy()

    n = min(len(scores_by_rank), len(rmsd_by_rank))
    scores_by_rank = scores_by_rank[:n]
    rmsd_by_rank = rmsd_by_rank[:n]

    top1_score = scores_by_rank[0]
    top1_rmsd = rmsd_by_rank[0]
    mean_rmsd = float(np.mean(rmsd_by_rank))
    median_rmsd = float(np.median(rmsd_by_rank))
    sd_rmsd = float(np.std(rmsd_by_rank))
    frac_within_threshold = float((rmsd_by_rank <= RMSD_THRESHOLD).mean()) * 100
    n_within_threshold = int((rmsd_by_rank <= RMSD_THRESHOLD).sum())

    print(f"{name}: N={n}  top1_score={top1_score:.2f}  top1_RMSD={top1_rmsd:.3f} A  "
          f"mean={mean_rmsd:.3f}  median={median_rmsd:.3f}  SD={sd_rmsd:.3f}  "
          f"pose-recovery(<= {RMSD_THRESHOLD} A)={n_within_threshold}/{n} ({frac_within_threshold:.1f}%)")

    rows_summary.append({
        "System": name, "N_solutions": n, "Top1_fitness_score": top1_score,
        "Top1_RMSD_A": top1_rmsd, "Mean_RMSD_A": mean_rmsd, "Median_RMSD_A": median_rmsd,
        "SD_RMSD_A": sd_rmsd, "N_within_2A": n_within_threshold,
        "PoseRecovery_pct_2A": frac_within_threshold,
    })
    for r in rmsd_by_rank:
        rows_full.append({"System": name, "RMSD_A": r})

summary_df = pd.DataFrame(rows_summary)
full_df = pd.DataFrame(rows_full)

summary_df.to_csv(OUT_DIR / "redocking_rmsd_summary.csv", index=False)
full_df.to_csv(OUT_DIR / "redocking_rmsd_full_population.csv", index=False)
print(f"\n[table saved] {OUT_DIR / 'redocking_rmsd_summary.csv'}")

# ====================================================================
# Figure: violin/box of the full RMSD population per system, with the
# top-1 (best-fitness) pose highlighted and the 2A threshold marked --
# replaces qualitative "visual inspection" language with real, N=1000
# quantitative distributions.
# ====================================================================
order = list(SYSTEMS.keys())
colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
data = [full_df.loc[full_df["System"] == s, "RMSD_A"].to_numpy() for s in order]
parts = ax.violinplot(data, showmedians=False, showextrema=False)
for pc, color in zip(parts["bodies"], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.45)

bp = ax.boxplot(data, widths=0.15, patch_artist=True, showfliers=False)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

for i, s in enumerate(order):
    top1 = summary_df.loc[summary_df["System"] == s, "Top1_RMSD_A"].values[0]
    ax.scatter([i + 1], [top1], color="gold", edgecolor="black", s=160, zorder=5, marker="*",
               label="Top-ranked (best-fitness) pose" if i == 0 else None)

ax.axhline(RMSD_THRESHOLD, color="grey", linestyle="--", linewidth=1, label=f"{RMSD_THRESHOLD} Å pose-recovery threshold")
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=15, ha="right")
ax.set_ylabel("RMSD to crystallographic pose (Å, heavy atoms)")
ax.set_title("GoldScore redocking accuracy: full population of $N=1000$\ngenetic-algorithm solutions per reference ligand")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "redocking_rmsd_population.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'redocking_rmsd_population.png'}")

print()
print(summary_df.to_string(index=False))
