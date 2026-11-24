# Module 3: Structural Relaxation, Active-Site Architecture & Induced-Fit Docking

## Overview

Module 3 takes the **140 novel screened de novo PETase designs** generated in **Module 2** (`designs_exp3_screened_novel.faa`) and implements the structural validation, local energy relaxation, and reactive docking pipeline planned in `experiment_3_modular_plan.md`.

### Core Engineering Breakthroughs Over Experiment 2
1. **OpenMM Constrained Energy Minimization (AMBER14SB):**
   - In Experiment 2, static ESMFold rigid-receptor docking caused a **98.5% Near-Attack Conformation (NAC) false-negative rate** due to unminimized side-chain rotamers blocking catalytic Ser $O\gamma$.
   - Module 3 applies harmonic position restraints ($k = 10\text{ kcal/mol}\cdot\text{\AA}^2$) on all heavy backbone atoms ($N, C\alpha, C, O$), allowing side chains to resolve steric strain into physical minima while preserving fold geometry ($< 0.5\text{ \AA}$ backbone RMSD).
2. **Dual-Ligand Induced-Fit Docking:**
   - Docks both **2-HE(MHET)** (monomer intermediate) and **3PET** (trimeric PET chain representing the solid plastic surface interface).
   - Flexible active-site side chains centered on catalytic Ser $O\gamma$.
3. **Bürgi-Dunitz Trajectory Angle:**
   - Evaluates the physical attack trajectory angle $\theta = \angle(O\gamma \cdots C_{\text{ester}} = O_{\text{carbonyl}}) \in [95^\circ, 115^\circ]$.
4. **Disulfide Topology Audit:**
   - Scans for conserved C-terminal disulfides ($T_m > 70^\circ\text{C}$ cutinase determinants) and active-site disulfides (DS1/DS2).

---

## File Structure

```
module_3_structural_relaxation_and_docking/
├── exp3_module_3_structural_relaxation_and_docking.ipynb  # Primary Colab GPU notebook
├── mod_03_induced_fit_docking.py                         # Standalone Linux/HPC CLI runner
├── docking_relaxation_utils.py                           # Core biophysical algorithms & utilities
├── create_colab_notebook.py                              # Notebook generator script
├── README.md                                             # This guide
└── inputs/
    ├── designs_exp3_screened_novel.faa                   # 140 novel de novo candidates from Module 2
    └── contrastive_reranked_designs.tsv                  # Contrastive log-likelihood scores (Delta-LL)
```

---

## Instructions for Google Colab Execution

Because ESMFold folding requires 16–24 GB GPU VRAM and structural tools (OpenMM, Foldseek, Smina) are optimized for Linux, run the pipeline on Google Colab:

1. **Open Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/).
   - Click **File > Upload notebook** and select `exp3_module_3_structural_relaxation_and_docking.ipynb`.
2. **Select GPU Accelerator:**
   - Go to **Runtime > Change runtime type**.
   - Select **T4**, **L4**, or **A100 GPU** and **High-RAM**.
3. **Upload Inputs or Mount Google Drive:**
   - If using Google Drive, place the `evo_2_try` repository in your Drive (`/content/drive/MyDrive/evo_2_try`), and Cell 1 will automatically link it.
   - Alternatively, upload `designs_exp3_screened_novel.faa` directly to the Colab files panel.
4. **Execute Cells Sequentially:**
   - **Cell 1–3:** Verifies GPU (`nvidia-smi`), installs OpenMM, Foldseek, Smina, RDKit, Meeko, and downloads reference structures (`6EQE` and `4EB0`).
   - **Cell 4–5 (ESMFold):** Folds all candidate sequences and generates PDB files in `esmfold_pdbs/`, evaluating mean pLDDT ($\ge 80.0$).
   - **Cell 6 (OpenMM):** Relaxes the PDBs into `relaxed_pdbs/` resolving side-chain rotamers.
   - **Cell 7–8 (Foldseek & Active Site):** Evaluates TM-scores, catalytic triad distances (Ser-His, His-Asp), oxyanion donor, and substrate clamp.
   - **Cell 9–10 (Smina Docking):** Performs flexible docking with 2-HE(MHET) and 3PET, computing attack distances and Bürgi-Dunitz angles.
   - **Cell 11–13 (Export):** Automatically creates `esmfold_pdbs.zip` and `relaxed_pdbs.zip` for instant download back to your local workspace, and syncs all TSV reports to Google Drive.
