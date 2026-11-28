#!/usr/bin/env python3
"""Builds the Colab-ready Jupyter notebook for Experiment 3 - Modules 4 & 5:
Interfacial Biophysical Proxies, Decoupled Z-Score Triage & 48-Target Wet-Lab Panel.
"""
import json
import pathlib

OUTPUT_NB_PATH = pathlib.Path(__file__).parent / "exp3_module_4_and_5_triage_and_panel.ipynb"
cells = []


def add_md(source: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    })


def add_code(source: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    })


add_md("""# Experiment 3 — Modules 4 & 5: Interfacial Proxies, Decoupled Triage & 48-Target Wet-Lab Panel
### Colab Pipeline for Post-Docking Biophysical Filtering & Experimental Panel Assembly

This notebook runs **Modules 4 & 5** directly on top of your **Module 3** outputs (`relaxed_pdbs.zip` and TSV reports synced to Google Drive or present in `/content/exp3_module_3_workspace`):
1. **Component 4.1 (Interfacial Adsorption Proxies):** Computes solvent-accessible hydrophobic cleft area ($A_{\\text{cleft\\_hp}}$ for Trp/Phe/Tyr/Leu/Ile/Val within $8\\text{ \\AA}$ of catalytic Ser $O\\gamma$ via FreeSASA / Biopython `ShrakeRupley`) and local electrostatic cleft charge.
2. **Component 4.2 (Disulfide Topology Audit):** Audits C-terminal cutinase disulfides ($T_m > 70^\\circ\\text{C}$ determinant) and active-site `DS1`/`DS2` pairs.
3. **Component 4.3 (Thermostability Prediction):** Runs **TemStaPro** (ESM-2 GPU embeddings) when available and combines it with a Ridge $T_m$ calibrator trained on the 139 experimentally annotated PETase pool enzymes.
4. **Component 4.4 (*E. coli* Solubility Proxy):** Evaluates soluble expression probability (`solubility_score >= 0.45`).
5. **Module 5 — Stage 1 (Hard Binary Elimination Gate):**
   $$\\text{Gate} = (\\text{pLDDT} \\ge 80.0) \\land (\\text{SeqTriad}_{\\text{Ser,Asp,His}} = \\text{True}) \\land (241 \\le L \\le 306) \\land (\\text{Solubility} \\ge 0.45)$$
6. **Module 5 — Stage 2 (Continuous Z-Score Composite Ranking):**
   $$\\text{Fitness Score} = 0.25\\,Z(\\text{pLDDT}) + 0.20\\,Z(\\text{TM}_{\\text{ref}}) + 0.20\\,Z(-\\Delta G_{\\text{bind}}) + 0.15\\,Z(\\text{NAC}) + 0.10\\,Z(\\hat{T}_m) + 0.10\\,Z(A_{\\text{cleft\\_hp}})$$
7. **Controlled 48-Target Wet-Lab Panel (Panels A–F):**
   - **Panel A (12):** Thermostable Structural Leads ($\\hat{T}_m \\ge 70^\\circ\\text{C}$, intact triad, $d_{\\text{attack}} \\le 3.5\\text{ \\AA}$ priority)
   - **Panel B (12):** Deep Sequence Novelty (farthest-point clustering, $50\\%\\text{–}65\\%$ pool identity priority)
   - **Panel C (8):** *Is*PETase Groove Rescues (`Trp159` groove + thermophilic core packing)
   - **Panel D (6):** Random Gate-Passers (in silico null control)
   - **Panel E (4):** Catalytic $\\text{Ser} \\to \\text{Ala}$ Knockouts of top 4 Panel A leads
   - **Panel F (6):** Natural & Engineered Reference Standards (`LCC`, `ICCG`, `PHL7`, `TfCut2`, *Is*PETase, `Fast-PETase`)""")

add_md("## 1. Environment, Drive Mount & Workspace Setup")
add_code("""import os, sys, pathlib, subprocess, shutil, zipfile

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')

WORKSPACE = pathlib.Path("exp3_module_4_5_workspace")
WORKSPACE.mkdir(exist_ok=True)
RELAX_DIR = WORKSPACE / "relaxed_pdbs"
RELAX_DIR.mkdir(exist_ok=True)
OUT_DIR = WORKSPACE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
except Exception:
    print("CPU runtime active.")
print("Workspace:", WORKSPACE.resolve())""")

add_md("## 2. Install Dependencies (Biopython, FreeSASA, TemStaPro / ESM-2, Plotting)")
add_code("""!pip install -q biopython pandas numpy scipy matplotlib seaborn freesasa
# Optional GPU thermostability predictor (TemStaPro via PyPI / GitHub)
!pip install -q temstapro || echo "TemStaPro optional install skipped; using 139-pool calibrated Ridge Tm predictor."

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
print("Core dependencies loaded.")""")

add_md("""## 3. Locate Module 2 & Module 3 Artifacts (Drive / Local / Uploaded)
Automatically finds your Module 3 Drive archive (`relaxed_pdbs.zip`, TSV reports, `foldseek_vs_references.m8`) and Module 2/pool files.""")
add_code("""SEARCH_ROOTS = [
    pathlib.Path("."),
    pathlib.Path("outputs"),
    pathlib.Path("../outputs"),
    pathlib.Path("../../data"),
    pathlib.Path("/content"),
    pathlib.Path("/content/exp3_module_3_workspace"),
    pathlib.Path("/content/drive/MyDrive/experiment_3/module_3_results"),
    pathlib.Path("/content/drive/MyDrive/evo_2_try"),
]

def find_file(filename):
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        direct = root / filename
        if direct.exists():
            return direct
        hits = list(root.glob(f"**/{filename}"))
        if hits:
            return hits[0]
    return None

# Extract relaxed_pdbs.zip if needed
if not list(RELAX_DIR.glob("*.pdb")):
    m3_relax_dir = pathlib.Path("/content/exp3_module_3_workspace/relaxed_pdbs")
    if m3_relax_dir.exists() and list(m3_relax_dir.glob("*.pdb")):
        for p in m3_relax_dir.glob("*.pdb"):
            shutil.copy(p, RELAX_DIR / p.name)
    else:
        rzip = find_file("relaxed_pdbs.zip")
        if rzip:
            print(f"Extracting {rzip} -> {RELAX_DIR}...")
            with zipfile.ZipFile(rzip, "r") as zf:
                for m in zf.namelist():
                    if m.endswith(".pdb"):
                        dest = RELAX_DIR / pathlib.Path(m).name
                        if not dest.exists():
                            with zf.open(m) as src, open(dest, "wb") as dst:
                                shutil.copyfileobj(src, dst)

FASTA_PATH = find_file("designs_exp3_screened_novel.faa")
GRID_TSV = find_file("generated_continuations_grid.tsv")
RERANK_TSV = find_file("contrastive_reranked_designs.tsv")
POOL_FASTA = find_file("petase_pool.fasta")
POOL_META = find_file("metadata_pool_annotated.tsv")
PLDDT_TSV = find_file("esmfold_plddt_summary.tsv")
TRIAD_TSV = find_file("catalytic_triad_geometry_report.tsv")
FOLDSEEK_M8 = find_file("foldseek_vs_references.m8")
DOCKING_TSV = find_file("smina_flexible_docking_results.tsv")
DISULFIDE_TSV = find_file("disulfide_topology_report.tsv")

for label, p in [
    ("FASTA", FASTA_PATH), ("Grid TSV", GRID_TSV), ("Reranked TSV", RERANK_TSV),
    ("Pool FASTA", POOL_FASTA), ("Pool Meta", POOL_META), ("pLDDT TSV", PLDDT_TSV),
    ("Triad TSV", TRIAD_TSV), ("Foldseek M8", FOLDSEEK_M8), ("Docking TSV", DOCKING_TSV),
    ("Disulfide TSV", DISULFIDE_TSV),
]:
    print(f"  {label:<14}: {p if p else 'MISSING (will use fallback if optional)'}")
print(f"  Relaxed PDBs  : {len(list(RELAX_DIR.glob('*.pdb')))} files")""")

add_md("""## 4. Module 4 — Optional TemStaPro GPU Thermostability Run
Runs TemStaPro on GPU if installed in Colab to obtain ESM-2 $P(T_m \\ge 65^\\circ\\text{C})$ predictions; seamlessly falls back to the 139-enzyme PETase pool Ridge calibrator if TemStaPro is unavailable.""")
add_code("""TEMSTAPRO_TSV = WORKSPACE / "temstapro_results.tsv"
temstapro_exe = shutil.which("temstapro")
if temstapro_exe and FASTA_PATH and not TEMSTAPRO_TSV.exists():
    print("Running TemStaPro on GPU...")
    tp_out = WORKSPACE / "temstapro_raw"
    tp_out.mkdir(exist_ok=True)
    res = subprocess.run([temstapro_exe, "-f", str(FASTA_PATH), "-d", str(tp_out), "--more-thresholds"], capture_output=True, text=True)
    print("TemStaPro returncode:", res.returncode)
    mean_files = list(tp_out.glob("*mean*.tsv")) + list(tp_out.glob("*.tsv"))
    if mean_files:
        df_tp = pd.read_csv(mean_files[0], sep="\\t")
        id_c = df_tp.columns[0]
        col65 = [c for c in df_tp.columns if "65" in str(c)]
        if col65:
            pd.DataFrame({"id": df_tp[id_c], "temstapro_tm65_prob": df_tp[col65[0]]}).to_csv(TEMSTAPRO_TSV, sep="\\t", index=False)
            print("Saved TemStaPro predictions to", TEMSTAPRO_TSV)
else:
    print("Using 139-pool Ridge-calibrated Tm estimator (TemStaPro CLI not present or already cached).")""")

add_md("""## 5. Run Module 4: 3D Cleft Hydrophobic SASA ($A_{\\text{cleft\\_hp}}$), Disulfides, $\\hat{T}_m$, & Solubility""")
add_code("""# Add module_4 and module_5 to sys.path if repo is mounted, or define inline helpers
for mod_dir in [
    pathlib.Path("module_4_biophysical_proxies"),
    pathlib.Path("../module_4_biophysical_proxies"),
    pathlib.Path("/content/drive/MyDrive/evo_2_try/experiments/experiment_3_deep_exploration/module_4_biophysical_proxies"),
    pathlib.Path("module_5_triage_and_panel"),
    pathlib.Path("../module_5_triage_and_panel"),
    pathlib.Path("/content/drive/MyDrive/evo_2_try/experiments/experiment_3_deep_exploration/module_5_triage_and_panel"),
]:
    if mod_dir.exists() and str(mod_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(mod_dir.resolve()))

from mod_04_biophysical_proxies import run_module_4
from mod_05_decoupled_triage import run_module_5

df_m4 = run_module_4(
    fasta_path=FASTA_PATH,
    reranked_tsv=RERANK_TSV,
    grid_tsv=GRID_TSV,
    pool_fasta=POOL_FASTA,
    pool_metadata_tsv=POOL_META,
    out_dir=OUT_DIR,
    relaxed_pdb_dir=RELAX_DIR,
    plddt_tsv=PLDDT_TSV,
    triad_tsv=TRIAD_TSV,
    disulfide_tsv=DISULFIDE_TSV,
    external_tm_tsv=TEMSTAPRO_TSV if TEMSTAPRO_TSV.exists() else None,
)
df_m4.head(5)""")

add_md("""## 6. Run Module 5: Stage 1 Binary Gate, Stage 2 Z-Score Ranking & 48-Target Wet-Lab Panel""")
add_code("""summary_m5 = run_module_5(
    fasta_path=FASTA_PATH,
    reranked_tsv=RERANK_TSV,
    grid_tsv=GRID_TSV,
    mod4_features_tsv=OUT_DIR / "interfacial_and_stability_features.tsv",
    pool_fasta=POOL_FASTA,
    out_dir=OUT_DIR,
    plddt_tsv=PLDDT_TSV,
    triad_tsv=TRIAD_TSV,
    foldseek_m8=FOLDSEEK_M8,
    docking_tsv=DOCKING_TSV,
    seed=42,
)

df_ranked = pd.read_csv(OUT_DIR / "ranked_exp3_candidates.tsv", sep="\\t")
df_panel = pd.read_csv(OUT_DIR / "wetlab_panel_exp3_48targets.tsv", sep="\\t")
print("\\n48-Target Wet-Lab Panel Breakdown:")
print(df_panel["panel"].value_counts())
df_panel[df_panel["panel"] == "A_thermostable_lead"][["id", "fitness_score", "mean_plddt", "tm_ref", "combined_affinity", "min_attack_dist", "predicted_tm", "solubility_score"]]""")

add_md("""## 7. Visualization: Stage 1 Elimination, Z-Score Pareto Space & 48-Target Panel""")
add_code("""df_audit = pd.read_csv(OUT_DIR / "triage_elimination_audit.tsv", sep="\\t")
passers = df_ranked[df_ranked["passed_stage1_gate"]].copy()

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# 1. Stage 1 Funnel
funnel_counts = [
    len(df_audit),
    int(df_audit["gate_plddt80"].sum()),
    int((df_audit["gate_plddt80"] & df_audit["gate_seq_triad"]).sum()),
    int((df_audit["gate_plddt80"] & df_audit["gate_seq_triad"] & df_audit["gate_length"]).sum()),
    int(df_audit["passed_stage1_gate"].sum()),
]
funnel_labels = ["Input Novel", "pLDDT >= 80", "+ Seq Triad", "+ Len 241-306", "+ Sol >= 0.45"]
axes[0, 0].bar(funnel_labels, funnel_counts, color=["#4c72b0", "#55a868", "#8172b3", "#ccb974", "#2ca02c"])
axes[0, 0].set_title("Stage 1: Hard Binary Elimination Funnel")
axes[0, 0].set_ylabel("Candidate Count")
axes[0, 0].tick_params(axis="x", rotation=20)

# 2. Fitness Score vs Novelty (Max PID) highlighting Panel Arms
axes[0, 1].scatter(passers["max_pid"], passers["fitness_score"], c="lightgray", alpha=0.6, label="Gate-Passers")
arm_colors = {
    "A_thermostable_lead": ("crimson", "Panel A: Thermo Leads (12)"),
    "B_deep_novelty": ("royalblue", "Panel B: Deep Novelty (12)"),
    "C_ispetase_groove_rescue": ("darkorange", "Panel C: Groove Rescues (8)"),
    "D_random_control": ("seagreen", "Panel D: Random Control (6)"),
}
for arm, (col, lbl) in arm_colors.items():
    sub_arm = df_panel[df_panel["panel"] == arm]
    axes[0, 1].scatter(sub_arm["max_pid"], sub_arm["fitness_score"], c=col, s=75, edgecolors="black", label=lbl, zorder=5)
axes[0, 1].axvline(65.0, color="navy", linestyle="--", alpha=0.5)
axes[0, 1].set_title("Stage 2 Z-Score Fitness vs. Sequence Novelty")
axes[0, 1].set_xlabel("Max % Identity to Natural Pool")
axes[0, 1].set_ylabel("Composite Z-Score Fitness")
axes[0, 1].legend(fontsize=8)

# 3. Predicted Tm vs E. coli Solubility
sc = axes[1, 0].scatter(df_ranked["solubility_score"], df_ranked["predicted_tm"], c=df_ranked["mean_plddt"], cmap="viridis", alpha=0.8)
plt.colorbar(sc, ax=axes[1, 0], label="ESMFold pLDDT")
axes[1, 0].axvline(0.45, color="red", linestyle="--", label="Solubility Gate (0.45)")
axes[1, 0].axhline(65.0, color="orange", linestyle="--", label="Thermostable Target (65 C)")
axes[1, 0].set_title("Predicted Tm vs. E. coli Solubility Proxy")
axes[1, 0].set_xlabel("Solubility Score")
axes[1, 0].set_ylabel("Predicted Tm (C)")
axes[1, 0].legend()

# 4. Cleft Hydrophobic SASA vs Binding Affinity
axes[1, 1].scatter(passers["cleft_hydrophobic_sasa"], passers["combined_affinity"], c=passers["predicted_tm"], cmap="plasma", alpha=0.85)
axes[1, 1].axhline(-5.0, color="red", linestyle="--", label="Affinity Cutoff (-5.0 kcal/mol)")
axes[1, 1].set_title("Cleft Hydrophobic SASA vs. Binding Affinity")
axes[1, 1].set_xlabel("Cleft Hydrophobic SASA (A^2)")
axes[1, 1].set_ylabel("Combined Smina Affinity (kcal/mol)")
axes[1, 1].legend()

plt.tight_layout()
fig_path = OUT_DIR / "module_4_5_triage_summary.png"
plt.savefig(fig_path, dpi=300)
plt.show()""")

add_md("## 8. Export & Sync Modules 4 & 5 Artifacts to Google Drive")
add_code("""if 'google.colab' in sys.modules:
    drive_out = pathlib.Path("/content/drive/MyDrive/experiment_3/module_4_5_results")
    drive_out.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("*"):
        if f.is_file():
            shutil.copy(f, drive_out / f.name)
    print(f"Synced all Modules 4 & 5 artifacts to: {drive_out}")
else:
    print(f"Local artifacts saved in: {OUT_DIR.resolve()}")""")

nb_json = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open(OUTPUT_NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb_json, f, indent=1)
print(f"Generated notebook: {OUTPUT_NB_PATH} ({len(cells)} cells)")
