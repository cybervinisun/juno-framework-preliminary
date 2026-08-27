"""
Comparacao de contatos nativos (cristal) vs pose redocada (melhor
fitness) para os 5 sistemas de referencia da redocagem (Tabela 1),
usando o PLIP real sobre as estruturas reais.

Para cada sistema:
1. Identifica o arquivo gold_soln_*.mol2 da pose de melhor fitness
   (cruzando o RMSD "melhor pose" ja reportado na Tabela 1 com a aba
   "Worksheet" da planilha de populacao, que traz o numero da solucao).
2. Funde receptor (gold_protein.mol2) + essa pose em um unico PDB
   (removendo atomos fantasma tipo par-isolado "Lp"/"*", que o
   OpenBabel nao consegue tipar a partir do mol2 do GOLD).
3. Roda o PLIP na estrutura cristalografica original (contatos
   nativos) e na pose redocada fundida (contatos redocados).
4. Reporta contagens de interacao por tipo para ambas.
"""
from __future__ import annotations

import glob
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

# Diretorio com os PDBs cristalograficos e as pastas brutas de
# redocagem do GOLD (ver nota em redocking_analysis.py). Sobrescreva
# via variavel de ambiente GOLD_RAW_DIR se sua copia estiver em outro
# lugar; os resultados ja extraidos desta analise estao em
# data/raw/tabela_contatos_nativos_vs_redocados_*.csv.
BASE = os.environ.get(
    "GOLD_RAW_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "raw" / "gold_redocking_raw"),
)
D = f"{BASE}/cristais_selecionados_ensemble/ligantes_cocristalizados_docagem"

SYSTEMS = {
    "3QEL": {"folder": f"{D}/3qel/redocagem/goldscore/gold_(N1000)", "crystal": f"{BASE}/3qel.pdb",
             "ligand_prefix": "gold_soln_ifenprodil_protonado_exp_m1_", "best_rmsd": 0.301},
    "3QEM": {"folder": f"{D}/3qem/redocagem/Goldscore/gold_(N1000)", "crystal": f"{BASE}/3qem.pdb",
             "ligand_prefix": None, "best_rmsd": 0.309},
    "5EWM": {"folder": f"{D}/5ewm/redocagem/Goldscore/goldscore(N1000)", "crystal": f"{BASE}/5ewm.pdb",
             "ligand_prefix": None, "best_rmsd": 0.527},
    "6E7R": {"folder": f"{D}/6e7r/redocagem/goldscore/gold_(N1000)", "crystal": f"{BASE}/6e7r.pdb",
             "ligand_prefix": None, "best_rmsd": 0.987},
    "6E7U": {"folder": f"{D}/6e7u/goldscore/gold_(N1000)", "crystal": f"{BASE}/6e7u.pdb",
             "ligand_prefix": None, "best_rmsd": 1.070},
}

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("WORK_DIR", REPO_ROOT / "results" / "regenerated" / "plip_native"))
WORK.mkdir(parents=True, exist_ok=True)


import re


def _find_pose_by_number(folder: str, sol_number: int):
    all_soln = glob.glob(os.path.join(folder, f"gold_soln_*_{sol_number}.mol2"))
    all_soln = [f for f in all_soln if f.rsplit("_", 1)[-1].replace(".mol2", "") == str(sol_number)]
    return all_soln[0] if all_soln else None


def find_top_pose_file(folder: str, best_rmsd: float, ligand_prefix_hint: str | None):
    matches = [m for m in glob.glob(os.path.join(folder, "*.xlsx")) if "popula" in m.lower() or "estat" in m.lower()]
    assert matches, f"stats xlsx not found in {folder}"
    xlsx_path = matches[0]

    sheet_names = pd.ExcelFile(xlsx_path).sheet_names
    id_sheet = None
    for s in sheet_names:
        probe = pd.read_excel(xlsx_path, sheet_name=s, header=None, nrows=2)
        first_cell = str(probe.iloc[0, 0]) if probe.shape[0] else ""
        if "|" in first_cell and "dock" in first_cell:
            id_sheet = s
            break
    assert id_sheet is not None, f"No per-solution ID sheet found in {xlsx_path} (sheets: {sheet_names})"

    df = pd.read_excel(xlsx_path, sheet_name=id_sheet, header=None)
    rmsd_col = pd.to_numeric(df[3], errors="coerce")
    idx = (rmsd_col - best_rmsd).abs().idxmin()
    diff = abs(rmsd_col.loc[idx] - best_rmsd)
    id_string = str(df.loc[idx, 0])
    sol_number_col1 = int(df.loc[idx, 1])

    dock_match = re.search(r"dock(\d+)", id_string)
    sol_number_dock = int(dock_match.group(1)) if dock_match else None

    for candidate_num in [sol_number_col1, sol_number_dock]:
        if candidate_num is None:
            continue
        if ligand_prefix_hint:
            candidate = os.path.join(folder, f"{ligand_prefix_hint}{candidate_num}.mol2")
            if os.path.exists(candidate):
                return candidate, candidate_num, diff
        found = _find_pose_by_number(folder, candidate_num)
        if found:
            return found, candidate_num, diff

    raise FileNotFoundError(
        f"No pose file found for solution col1={sol_number_col1} or dock={sol_number_dock} "
        f"in {folder} (id={id_string})"
    )


def find_receptor_file(folder: str):
    for name in ["gold_protein.mol2", "3QEL_protein.mol2"]:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(folder, "*protein*.mol2"))
    assert cands, f"No receptor file found in {folder}"
    return cands[0]


def merge_receptor_pose(receptor_mol2: str, pose_mol2: str, out_pdb: str, ligand_resname: str = "LIG"):
    receptor_pdb = out_pdb.replace(".pdb", "_receptor.pdb")
    pose_pdb = out_pdb.replace(".pdb", "_pose.pdb")
    subprocess.run(["obabel", receptor_mol2, "-O", receptor_pdb], capture_output=True, text=True)
    subprocess.run(["obabel", pose_mol2, "-O", pose_pdb], capture_output=True, text=True)

    lines_receptor = [l for l in open(receptor_pdb) if l.startswith("ATOM")]
    # The ligand pose file contains ONLY the ligand -- OpenBabel sometimes
    # labels its records ATOM instead of HETATM depending on residue-name
    # heuristics, so grab both and treat everything in this file as ligand.
    lines_ligand_raw = [l for l in open(pose_pdb) if l.startswith("ATOM") or l.startswith("HETATM")]

    lines_ligand = []
    for l in lines_ligand_raw:
        elem = l[76:78].strip()
        name = l[12:16].strip()
        if elem in ("", "*") or name.upper().startswith("LP") or name == "*":
            continue
        lines_ligand.append(l)

    with open(out_pdb, "w") as out:
        out.writelines(lines_receptor)
        out.write("TER\n")
        for l in lines_ligand:
            l2 = "HETATM" + l[6:17] + f"{ligand_resname:<3}" + l[20:21] + "X" + l[22:]
            out.write(l2)
        out.write("END\n")
    return len(lines_ligand), len(lines_ligand_raw)


def run_plip(pdb_path: str, outdir: str):
    subprocess.run(
        ["python3", "-m", "plip.plipcmd", "-f", pdb_path, "-x", "-o", outdir],
        capture_output=True, text=True, cwd=os.path.dirname(pdb_path) or ".",
    )
    base = os.path.splitext(os.path.basename(pdb_path))[0]
    report = os.path.join(os.path.dirname(pdb_path) or ".", outdir, f"{base}_report.xml")
    return report if os.path.exists(report) else None


def parse_interactions(report_xml: str, target_longnames: set[str] | None = None):
    """Retorna lista de dicts {longname, chain, type, resname, resnr, reschain}."""
    if report_xml is None:
        return []
    tree = ET.parse(report_xml)
    root = tree.getroot()
    rows = []
    for bs in root.iter("bindingsite"):
        ln = bs.find("identifiers/longname")
        longname = ln.text if ln is not None else "?"
        if target_longnames and longname not in target_longnames:
            continue
        bs_chain = bs.find("identifiers/chain")
        for interactions in bs.iter("interactions"):
            for tag in ["hydrogen_bond", "pi_stack", "hydrophobic_interaction", "salt_bridge", "water_bridge", "halogen_bond"]:
                for hb in interactions.iter(tag):
                    rows.append({
                        "longname": longname,
                        "bs_chain": bs_chain.text if bs_chain is not None else "?",
                        "type": tag,
                        "restype": hb.find("restype").text,
                        "resnr": hb.find("resnr").text,
                        "reschain": hb.find("reschain").text,
                    })
    return rows


summary_rows = []
detail_rows = []

for sys_name, cfg in SYSTEMS.items():
    print(f"\n{'='*70}\n{sys_name}\n{'='*70}")
    try:
        pose_file, sol_number, rmsd_diff = find_top_pose_file(cfg["folder"], cfg["best_rmsd"], cfg["ligand_prefix"])
        receptor_file = find_receptor_file(cfg["folder"])
        print(f"Top pose solution: {sol_number}  (RMSD match diff: {rmsd_diff:.4f})  file: {os.path.basename(pose_file)}")
        print(f"Receptor: {os.path.basename(receptor_file)}")

        merged_pdb = str(WORK / f"merged_{sys_name}.pdb")
        n_kept, n_raw = merge_receptor_pose(receptor_file, pose_file, merged_pdb)
        print(f"Ligand atoms kept after lone-pair filtering: {n_kept}/{n_raw}")

        redocked_report = run_plip(merged_pdb, f"out_{sys_name}_redocked")
        crystal_pdb = cfg["crystal"]
        crystal_local = str(WORK / f"crystal_{sys_name}.pdb")
        import shutil
        shutil.copy(crystal_pdb, crystal_local)
        crystal_report = run_plip(crystal_local, f"out_{sys_name}_crystal")

        redocked_rows = parse_interactions(redocked_report)
        crystal_rows_all = parse_interactions(crystal_report)

        print(f"Redocked interactions found: {len(redocked_rows)}")
        print(f"Crystal interactions found (all ligand copies): {len(crystal_rows_all)}")

        for r in redocked_rows:
            r["system"] = sys_name
            r["source"] = "redocked"
            detail_rows.append(r)
        for r in crystal_rows_all:
            r["system"] = sys_name
            r["source"] = "crystal"
            detail_rows.append(r)

        redocked_types = pd.Series([r["type"] for r in redocked_rows]).value_counts().to_dict()
        crystal_types = pd.Series([r["type"] for r in crystal_rows_all]).value_counts().to_dict()

        summary_rows.append({
            "System": sys_name,
            "Top_pose_solution_number": sol_number,
            "RMSD_match_diff": rmsd_diff,
            "Crystal_total_interactions": len(crystal_rows_all),
            "Redocked_total_interactions": len(redocked_rows),
            "Crystal_hbond": crystal_types.get("hydrogen_bond", 0),
            "Redocked_hbond": redocked_types.get("hydrogen_bond", 0),
            "Crystal_hydrophobic": crystal_types.get("hydrophobic_interaction", 0),
            "Redocked_hydrophobic": redocked_types.get("hydrophobic_interaction", 0),
            "Crystal_pistack": crystal_types.get("pi_stack", 0),
            "Redocked_pistack": redocked_types.get("pi_stack", 0),
            "Crystal_waterbridge": crystal_types.get("water_bridge", 0),
            "Redocked_waterbridge": redocked_types.get("water_bridge", 0),
        })
    except Exception as e:
        print(f"FALHOU para {sys_name}: {e}")
        summary_rows.append({"System": sys_name, "error": str(e)})

summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(detail_rows)
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "results" / "regenerated"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(OUT_DIR / "tabela_contatos_nativos_vs_redocados_G.csv", index=False)
detail_df.to_csv(OUT_DIR / "tabela_contatos_nativos_vs_redocados_detalhe_G.csv", index=False)
print("\n\n=== RESUMO FINAL ===")
print(summary_df.to_string(index=False))
