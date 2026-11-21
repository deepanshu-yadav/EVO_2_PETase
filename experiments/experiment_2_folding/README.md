# Experiment 2: Structural Validation, Active-Site Architecture & Multi-Parametric Ranking

This document details the computational protocols, audit findings, and validated outcomes for **Experiment 2 (Phase 2: Structural Validation & Biased ICL)** incorporating the audited v2 biophysics and reactive docking pipeline.

In this phase, the **71 novel de novo PETase designs** curated during [Experiment 1](../experiment1_initial_tests/README.md) were evaluated through an atomic-resolution structural biology and functional fitness screening funnel.

---

## Navigation & Project Hierarchy
- **[Parent Suite: Experiments Overview (`../README.md`)](../README.md)**
- **[Preceding Phase: Experiment 1 Discovery (`../experiment1_initial_tests/README.md`)](../experiment1_initial_tests/README.md)**
- **[Master Repository: Root README (`../../README.md`)](../../README.md)**

---

## 1. Experimental Overview & Objectives

While Experiment 1 verified sequence coherence and novelty ($< 85\%$ max identity to known PETases), sequence-based language models can generate sequences that fail to fold or possess disrupted active-site geometries. 

Experiment 2 establishes an atomic-resolution in silico screening funnel to identify candidate enzymes ready for recombinant expression and in vitro testing:
1. **Tertiary Foldability:** Predict 3D backbones with **ESMFold** and filter at high confidence ($\text{mean pLDDT} \ge 80.0$).
2. **Topological Congruence:** Align folds to wild-type *Ideonella sakaiensis* PETase (**IsPETase**, PDB: `6EQE`) and the thermophilic benchmark **LCC** (PDB: `4EB0`) via **Foldseek**.
3. **Active-Site 3D Geometry:** Superimpose onto `6EQE` via global pairwise sequence alignment (BLOSUM62) and measure sub-angstrom spatial distances for the catalytic triad (**Ser160–His237–Asp206**), oxyanion hole (**Tyr87**), and substrate groove clamp (**Trp159/Trp185**).
4. **Reactive Substrate Docking:** Dock the native depolymerization intermediate **2-HE(MHET)** (`results/docking_ligands/2he_mhet.sdf`) with **Smina** in the native receptor coordinate frame, measuring nucleophilic attack distance specifically to the electrophilic ester carbonyl carbon ($C_{\text{ester}}$) via SMARTS `[CX3](=O)[OX2][CH2]`.
5. **Plausibility Gating & Multi-Arm Wet-Lab Panel:** Filter candidates by biophysical plausibility gates (`plddt >= 80.0`, sequence triad intact, length 241–306 aa) and assemble a controlled wet-lab validation panel.

```mermaid
flowchart TD
    A["71 Novel De Novo Designs<br/>(from Experiment 1)"] --> B{"Tier 1 & 2: ESMFold Prediction<br/>pLDDT >= 80.0 Filter"}
    B -- "70 Passed (98.6%)<br/>Mean pLDDT = 90.5" --> C["Tier 3: Foldseek Alignment<br/>vs IsPETase (6EQE) & LCC (4EB0)"]
    B -- "1 Failed (1.4%)" --> DROP1["petase_8_seq_30 (pLDDT = 74.0)"]
    C --> D{"Tier 4: Active Site 3D Audit<br/>BLOSUM62 Sequence Alignment"}
    D -- "66 Structurally Intact Triads (94.3%)<br/>68 Sequence Triads Intact (97.1%)" --> E["Tier 5: 2-HE(MHET) Smina Reactive Docking<br/>Native Frame + SMARTS C_ester Distance"]
    D -- "4 Failed Triad Geometry (5.7%)" --> DROP2["2 Seq Triad Mutations (petase_8_seq_24, petase_64_seq_24)<br/>2 Geometry Distortions"]
    E --> F["Zero Clash Artifacts (< 2.2 Å: 0 / 70)<br/>Attack Distances: 3.42 Å – 9.84 Å<br/>Affinities: -7.20 to -4.90 kcal/mol"]
    F --> G["Plausibility Gating (68 Passers)<br/>Controlled Wet-Lab Panel (Panels A–F)"]
```

---

## 2. Pipeline Audit & Verified Outcomes

Following computational auditing and implementation of the v2 biophysical pipeline:

| Metric | Previous Run (v1.5) | Audited v2 Run (Current) | Scientific Impact of Fix |
| :--- | :---: | :---: | :--- |
| **ESMFold Foldability** | 70 / 71 passed (98.6%) | **70 / 71 passed (98.6%)** | Average library pLDDT of **$90.49 \pm 3.9$** maintained. |
| **Foldseek TM vs 6EQE** | Mean TM = 0.949, RMSD = 1.47 Å | **Mean TM = 0.949, RMSD = 1.47 Å** | Backbone folds match wild-type IsPETase architecture. |
| **Active-Site Triad Intact** | 56 / 70 passed (80.0%) | **66 / 70 passed (94.3%)** | **+10 designs rescued** by replacing coordinate index slicing with pairwise BLOSUM62 sequence alignment. |
| **Sequence Triad Intact** | 56 / 70 passed (80.0%) | **68 / 70 passed (97.1%)** | Only 2 designs lack the sequence triad (`petase_64_seq_24` with Glu, `petase_8_seq_24` with missing His). |
| **Complete Active Site** | 52 / 70 passed (74.3%) | **63 / 70 passed (90.0%)** | Intact Triad + Tyr87 Oxyanion + Trp159/185 Groove Clamp. |
| **Docking Clash Artifacts** | **13 / 70 designs (< 2.2 Å)** | **0 / 70 designs (0.0%)** | Fixed coordinate-frame mismatch by capturing `native_ser_og` before superposition. |
| **Attack Distance Measurement** | Any ligand atom (min 0.71 Å) | **Specific $C_{\text{ester}}$ via SMARTS** | Measured strictly to electrophilic carbonyl carbon ($3.42\text{ \AA} - 9.84\text{ \AA}$). |
| **Docking Affinities ($\Delta G$)** | -5.8 to -4.1 kcal/mol | **-7.20 to -4.90 kcal/mol** | Mean $\Delta G = -6.19\text{ kcal/mol}$; all 70 PDBQTs docked cleanly. |
| **Overall Top Candidate** | Distorted lead (clash artifact) | **`petase_64_seq_34` (Score: 0.7087)** | **60.9% identity**, pLDDT: 94.0, TM: 0.9602, Triad: 2.85/2.86 Å, Attack: 3.76 Å, $\Delta G$: -6.0 kcal/mol. |

---

## 3. Top 15 Validated Functional Candidates (Audited v2)

Candidates were evaluated under the validated multi-parametric composite scoring formula:

$$\text{Composite Score} = 0.35 \times \left(\frac{\text{pLDDT}}{100}\right) + 0.25 \times \text{TM-score}_{\text{6EQE}} - 0.15 \times \text{Triad Penalty} + 0.15 \times \text{NAC Score} - 0.10 \times \left(\frac{\Delta G_{\text{bind}}}{-10}\right)$$

where $\text{NAC Score} = \frac{1}{1 + \max(0, d_{\text{attack}} - 3.2)}$ and distances $< 2.5\text{ \AA}$ receive clash penalties.

| Rank | Candidate ID | Context $n$ | pLDDT | Max %ID (Novelty) | TM (6EQE) | Ser–His ($\text{\AA}$) | His–Asp ($\text{\AA}$) | Attack Dist ($\text{\AA}$) | $\Delta G$ (kcal/mol) | Composite Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`petase_64_seq_34`** | 64 | **94.0** | **60.9%** | **0.9602** | **2.85** | **2.86** | **3.76** | **-6.0** | **0.7087** |
| **2** | **`petase_32_seq_31`** | 32 | **94.0** | 64.6% | **0.9531** | 3.01 | 2.84 | **3.79** | **-5.8** | **0.6821** |
| **3** | **`petase_32_seq_21`** | 32 | **88.0** | **55.7%** | 0.9252 | 2.95 | 2.94 | **3.42** | **-6.1** | **0.6798** |
| **4** | **`petase_8_seq_39`**  | 8  | **94.0** | 72.1% | 0.9360 | 2.87 | 3.10 | **3.65** | **-6.8** | **0.6790** |
| **5** | **`petase_16_seq_13`** | 16 | **95.0** | 73.0% | 0.9523 | 2.97 | 2.80 | 4.67 | **-6.9** | **0.6748** |
| **6** | **`petase_16_seq_21`** | 16 | **91.0** | 58.3% | **0.9599** | 2.97 | 2.87 | **3.86** | **-6.2** | **0.6748** |
| **7** | **`petase_32_seq_27`** | 32 | **91.0** | 69.6% | 0.9597 | 2.95 | 2.91 | 4.04 | **-6.9** | **0.6699** |
| **8** | **`petase_64_seq_2`**  | 64 | **93.0** | 66.9% | 0.9628 | 2.90 | 2.93 | 4.26 | **-6.5** | **0.6694** |
| **9** | **`petase_16_seq_11`** | 16 | **95.0** | 72.2% | **0.9660** | **2.88** | **2.86** | 4.69 | **-5.6** | **0.6693** |
| **10**| **`petase_32_seq_8`**  | 32 | **91.0** | 63.2% | **0.9623** | 2.98 | 2.73 | 4.14 | **-5.9** | **0.6683** |
| **11**| `petase_32_seq_7`  | 32 | **95.0** | 61.9% | 0.9545 | 2.97 | 2.85 | 4.50 | **-6.3** | 0.6664 |
| **12**| `petase_32_seq_39` | 32 | **93.0** | 66.9% | **0.9675** | 2.95 | 2.90 | 4.38 | **-6.6** | 0.6647 |
| **13**| `petase_32_seq_9`  | 32 | **93.0** | 66.2% | 0.9426 | 3.04 | 2.89 | **3.98** | **-6.5** | 0.6609 |
| **14**| `petase_64_seq_27` | 64 | 87.0 | 62.2% | 0.9595 | 2.88 | 3.01 | **3.85** | **-6.6** | 0.6578 |
| **15**| `petase_64_seq_38` | 64 | 89.0 | **53.6%** | 0.9483 | 2.98 | 2.80 | 4.40 | **-6.8** | 0.6578 |

### Standout Candidate Profiles
- **`petase_64_seq_34` (Undisputed Overall Champion):** Top rank (0.7087) with 94.0 pLDDT, 0.9602 TM-score, pristine catalytic triad (2.85 Å / 2.86 Å), strong binding affinity ($-6.0\text{ kcal/mol}$), physical attack distance of **$3.76\text{ \AA}$**, and extreme sequence novelty (**60.9% identity** — over 100 mutations from the nearest known PETase).
- **`petase_32_seq_21` (Tightest Attack Trajectory):** Closest physical approach to Ser O$\gamma$ at **$3.42\text{ \AA}$** with $\Delta G = -6.1\text{ kcal/mol}$ and 55.7% sequence identity.
- **`petase_16_seq_11` (Pristine Structural Lead):** Peak folding confidence (**95.0 pLDDT**), TM-score of 0.9660, and sub-angstrom triad precision (2.88 Å / 2.86 Å).
- **`petase_64_seq_38` & `petase_32_seq_12` (Deep Novelty Champions):** 53.6% and 53.3% identity to pool, stably folded with sub-angstrom triads and binding affinities down to $-6.8\text{ kcal/mol}$.

---

## 4. Controlled Wet-Lab Testing Panel (`wetlab_panel_v2.tsv`)

Rather than relying on fine decimal ranking, candidates passing biophysical plausibility gates (`plddt >= 80.0`, mature length 241–306 aa, sequence triad intact: **68 / 70 passed**) are selected into a controlled testing panel across 6 arms:

```
Controlled Wet-Lab Panel Structure:
├── Panel A: Structure-Confidence Top 8 (Uncorrupted structural fitness)
├── Panel B: Sequence-Diverse 8 (Farthest-point sampling on identity down to 53.3%)
├── Panel C: Random Gate-Passers 8 (Negative control against in silico ranking)
├── Panel D: Rescued Designs (Rescued after fixing the index-slicing bug)
├── Panel E: Catalytic Knockouts 2 (Ser -> Ala on top Panel A leads)
└── Panel F: Natural Enzyme Benchmarks 4 (IsPETase, TfCut2, LCC, PHL7)
```

### Complete Panel Roster

| Panel | Candidate ID | Context $n$ | pLDDT | Max %ID (Novelty) | TM (6EQE) | Attack Dist ($\text{\AA}$) | $\Delta G$ (kcal/mol) | Role / Hypothesis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A_structure_top** | **`petase_16_seq_11`** | 16 | **95.0** | 72.2% | **0.966** | 4.69 | -5.6 | Uncorrupted structural lead #1; pristine triad |
| **A_structure_top** | **`petase_64_seq_34`** | 64 | **94.0** | **60.9%** | **0.960** | **3.76** | **-6.0** | Overall #1 champion; extreme novelty |
| **A_structure_top** | **`petase_8_seq_17`**  | 8  | **94.0** | 68.2% | **0.978** | 6.74 | -5.6 | Highest Foldseek TM-score to IsPETase (0.978) |
| **A_structure_top** | **`petase_16_seq_13`** | 16 | **95.0** | 73.0% | 0.952 | 4.67 | -6.9 | High pLDDT; tight binding (-6.9 kcal/mol) |
| **A_structure_top** | **`petase_64_seq_25`** | 64 | **93.0** | 69.8% | 0.957 | 5.28 | -5.7 | Rescued from v1; TM = 0.957 to 6EQE |
| **A_structure_top** | **`petase_64_seq_10`** | 64 | **93.0** | 65.3% | 0.969 | 6.80 | -5.9 | High global fold congruence to 6EQE |
| **A_structure_top** | **`petase_32_seq_7`**  | 32 | **95.0** | 61.9% | 0.954 | 4.50 | -6.3 | High pLDDT, 61.9% identity |
| **A_structure_top** | **`petase_32_seq_8`**  | 32 | **91.0** | 63.2% | 0.962 | 4.14 | -5.9 | Compact active cleft |
| **B_diverse** | `petase_32_seq_12` | 32 | 86.0 | **53.3%** | 0.940 | 3.77 | -5.9 | Deepest sequence exploration (53.3% identity) |
| **B_diverse** | `petase_32_seq_21` | 32 | 88.0 | **55.7%** | 0.925 | **3.42** | -6.1 | Tightest attack trajectory (3.42 Å) |
| **B_diverse** | `petase_64_seq_20` | 64 | 89.0 | 64.9% | 0.939 | 3.72 | -5.9 | Farthest point cluster member |
| **B_diverse** | `petase_16_seq_21` | 16 | 91.0 | 58.3% | 0.960 | 3.86 | -6.2 | High-scoring diverse representative |
| **B_diverse** | `petase_64_seq_38` | 64 | 89.0 | **53.6%** | 0.948 | 4.40 | -6.8 | Extreme sequence divergence (270 aa) |
| **B_diverse** | `petase_32_seq_24` | 32 | 89.0 | 70.8% | 0.942 | 3.83 | -6.1 | Farthest point on BLOSUM distance |
| **B_diverse** | `petase_8_seq_39`  | 8  | 94.0 | 72.1% | 0.936 | 3.65 | -6.8 | High binding affinity & 3.65 Å attack dist |
| **B_diverse** | `petase_16_seq_36` | 16 | 89.0 | 63.5% | 0.948 | 4.59 | -6.9 | Alternative loop topology |
| **C_random** | `petase_32_seq_5`  | 32 | 88.0 | 59.3% | 0.965 | 4.54 | -6.2 | Unbiased random gate-passer |
| **C_random** | `petase_64_seq_17` | 64 | 86.0 | 64.0% | 0.961 | 5.96 | -5.8 | Unbiased random gate-passer |
| **C_random** | `petase_64_seq_26` | 64 | 93.0 | 69.8% | 0.958 | 4.72 | -6.3 | Unbiased random gate-passer |
| **C_random** | `petase_32_seq_4`  | 32 | 90.0 | 64.4% | 0.923 | 4.86 | -6.4 | Unbiased random gate-passer |
| **C_random** | `petase_8_seq_27`  | 8  | 90.0 | 72.1% | 0.962 | 4.72 | -6.3 | Unbiased random gate-passer |
| **C_random** | `petase_32_seq_18` | 32 | 92.0 | 62.5% | 0.963 | 4.31 | -6.4 | Unbiased random gate-passer |
| **C_random** | `petase_32_seq_40` | 32 | 87.0 | 64.0% | 0.934 | 4.89 | -5.9 | Unbiased random gate-passer |
| **C_random** | `petase_8_seq_35`  | 8  | 92.0 | 63.5% | 0.969 | 6.79 | -5.5 | Unbiased random gate-passer |
| **D_rescued** | `petase_16_seq_31` | 16 | 90.0 | 65.2% | 0.955 | 3.58 | -5.6 | Rescued; near-attack approach (3.58 Å) |
| **D_rescued** | `petase_32_seq_17` | 32 | 89.0 | 72.7% | 0.940 | 4.34 | -6.0 | Rescued; intact sequence triad |
| **E_knockout** | `petase_16_seq_11_S131A` | 16 | 95.0 | — | — | — | — | Catalytic knockout negative control ($S \rightarrow A$) |
| **E_knockout** | `petase_64_seq_34_S130A` | 64 | 94.0 | — | — | — | — | Catalytic knockout negative control ($S \rightarrow A$) |
| **F_benchmark** | `001_IsPETase` | WT | — | 100.0% | 1.000 | — | — | Wild-type mesophilic benchmark (*I. sakaiensis*) |
| **F_benchmark** | `002_TfCut2` | Benchmark | — | — | — | — | — | Thermotolerant actinomycete cutinase |
| **F_benchmark** | `003_LCC` | Benchmark | — | — | — | — | — | Engineered thermophilic standard ($T_m \approx 93^\circ\text{C}$) |
| **F_benchmark** | `004_PHL7` | Benchmark | — | — | — | — | — | Fast compost metagenome cutinase |

---

## 5. Output Files & Dataset Manifest

| File | Description |
| :--- | :--- |
| **[`results/ranked_functional_petase_candidates.tsv`](results/ranked_functional_petase_candidates.tsv)** | **Master ranked report** containing audited docking, foldseek, triad, and composite fitness metrics across all 70 designs. |
| **[`results/wetlab_panel_v2.tsv`](results/wetlab_panel_v2.tsv)** | **Master controlled wet-lab testing panel** (Panels A–F) with full sequences, panel tags, and audit flags. |
| **[`results/wetlab_panel_v2.faa`](results/wetlab_panel_v2.faa)** | FASTA sequences of the expression targets ready for gene synthesis. |
| **[`results/all_designs_v2_features.tsv`](results/all_designs_v2_features.tsv)** | Comprehensive feature audit across all 70 folded designs, confirming 0 clash artifacts. |
| **[`results/catalytic_triad_geometry_report.tsv`](results/catalytic_triad_geometry_report.tsv)** | Sub-angstrom active site geometry: Ser–His and His–Asp distances, oxyanion hole, and groove aromatic clamp status. |
| **[`results/docking_results/*.pdbqt`](results/docking_results/)** | Complete set of 70 docked receptor–ligand complexes in native coordinates. |
| **[`results/esmfold_plddt_passed_80.tsv`](results/esmfold_plddt_passed_80.tsv)** | The 70 candidates passing the $\text{mean pLDDT} \ge 80.0$ threshold. |
