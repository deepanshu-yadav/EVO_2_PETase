# Evo 2 PETase Discovery: Results Interpretation and Pipeline v2 (audited run)

Nov 21, 2026 (audited revision) · @Deepanshu Yadav

> **What changed in this revision.** The user patched `run_structural_validation.py` into `run_structural_validation_v2.py`, addressing all six audit findings from the earlier draft, and re-ran the pipeline against the same 71 Evo 2 designs. The new results in `EVO_2_PETase-main (3).zip` change the audit substantially:
>
> - **Audit #1 (docking) is now resolved.** The patch captures the catalytic Ser Oγ in the native PDB frame **before** 6EQE superposition (so the smina box matches the receptor on disk), measures the attack distance specifically to the electrophilic ester carbonyl carbon via the SMARTS pattern `[CX3](=O)[OX2][CH2]` instead of "any ligand atom", penalizes steric clashes (< 2.5 Å), fails loudly when smina is missing (no more silent geometric fallback), and writes the docked PDBQTs to `results/docking_results/`. New numbers: 0 clash artifacts (was 0 in the previous re-audit, but the README's v1.5 column claims 13/70 < 2.2 Å); affinity −7.20 to −4.90 kcal/mol (was −5.80 to −4.10); attack distance 3.42 to 9.84 Å (was 0.71 to 8.10); only 1 design meets the strict reactive-pose criterion (was 10) — `petase_32_seq_21` at 3.42 Å, ΔG = −6.1 kcal/mol; 0 designs have NAC = 1.0 (was 25); Spearman between the v1.5 full composite and the no-docking composite is now **0.46** (was 0.62) — docking now contributes real signal.
> - **Audit #2 (3D triad check) is mostly resolved.** The patch replaces index-based Cα pairing (`ref_cas[:min_len]`) with pairwise BLOSUM62 sequence alignment. 66 of 70 designs now pass the 3D triad check (was 56); only 4 fail, of which **2 are wrongly failed** (was 12) — `petase_16_seq_31` and `petase_32_seq_17`, both with the triad intact by sequence — and 2 truly fail by sequence (`petase_64_seq_24` Glu→Asp, `petase_8_seq_24` no aligned His, same as before).
> - **Audit #3 (top-10 inside the noise) is no longer true.** With meaningful docking terms, the v1.5 composite's top-10 spans **0.040** (0.6683–0.7087), about **1.65 SD** (SD among triad-intact designs is now 0.0245) — the top-10 is distinguishable from the noise. Only **1 design** scores ≥ 0.70 (`petase_64_seq_34`, 0.7087); 18 score ≥ 0.65.
> - **Audit #4 (README formula vs code) still applies.** The README subtracts the affinity term (`− 0.10 × (ΔG/−10)`), the code adds it (`+ 0.10 × clip(ΔG/−10, 0, 1)`). Max diff is now 0.144 (was 0.116); the actual composite matches the code formula, not the README.
> - **Audit #5 (novelty only vs pool) still applies** — needs UniRef90 / NR / PDB search.
>
> Three smaller fixes were also applied: the Tier-1 motif regex is now `G[HWYFQ]S.G` (was `[GWA].[SMG].G`); pLDDT is rescaled from 0–1 B-factors to 0–100 before rounding; and an `active_site_plddt` column reports the mean pLDDT of the three catalytic-triad residues. The v3 README and experiments README are consistent with the data (no more "81.4 % NAC" or "48 candidates ≥ 0.70" claims; the leader is `petase_64_seq_34 (0.7087)`, matching the table).
>
> The wet-lab panel and features table were regenerated with the new rank_v2.py (which ingests the new smina columns and the `clash_artifact` flag). Panel D shrank from 6 to 2 because the alignment-based check rescued 10 of the 12 previously wrongly-failed designs — there are only 2 left to rescue. Total panel size is now **32 constructs** (was 36).

## Executive summary

Evo 2 in-context learning reliably writes PETase-family sequences that fold with high confidence and carry the catalytic triad, but nothing in the data shows they are active, and the audited v1.5 ranking — now meaningful at the top end — still cannot separate good designs from merely plausible ones without a function model.

- **Solid:** 71 novel designs (typically 60–73 % identical to the nearest of your 139 natural PETases, mean 53 % to each other); 70 of 71 fold at pLDDT ≥ 80; all 70 carry a Gly-x-Ser-x-Gly elbow, and 68 carry Ser/Asp/His at the positions aligned to IsPETase.
- **Now resolved by the v3 patch:** the docking tier docks into the native receptor frame and measures the attack distance specifically to the ester carbonyl carbon (0 clash artifacts, 70 docked PDBQTs preserved, ΔG −7.20 to −4.90 kcal/mol, attack distance 3.42 to 9.84 Å); the 3D triad check uses pairwise BLOSUM62 alignment and now passes 66 of 70 designs (was 56); the top-10 composite span is 1.65 SD, so the ranking carries information.
- **Still not solid:** the README composite-formula sign error (audit #4, max score shift 0.144); novelty measured only against the 139-sequence pool (audit #5); only 1 of 70 designs meets the strict reactive-pose criterion (ΔG ≤ −5.0 and attack_dist ≤ 3.5 Å) — `petase_32_seq_21` at 3.42 Å / −6.1 kcal/mol; 20 of 70 designs have an odd number of cysteines.
- **What the three links add:** the arXiv paper warns that structure confidence is a plausibility signal, not a function score; RFD4-Proteina gives an orthogonal, catalytic-site-first generator; cuPotential makes physics-based checks (MD of enzyme–substrate poses, heat stress) affordable at the scale of dozens of designs.
- **Recommendation:** the v3 patch made the in-silico ranking meaningfully ordered, but it did not change the headline conclusion — do not send "the top N by composite score" to the wet lab. Send a 32-construct controlled panel (structure-ranked, diversity-ranked, random, rescued, knockout and natural controls) so the first experiment also tells you whether any in-silico filter predicts activity. The panel and its script are included.

Every statement about activity, stability or expression in this report is a hypothesis until that panel is measured.

## How to read your results

Your pipeline answered "can Evo 2 write something that looks like a PETase?" convincingly (yes), and did not yet answer "will it work as one?".

![Funnel from 240 Evo 2 samples to 66 designs with a 3D-intact triad](figures/funnel_240_to_66.png)

**Experiment 1 (generation).** Novel, family-coherent yield rose from 0 % at n = 2 to 65 % at n = 64 context examples (37.5 % at n = 16, 55 % at n = 32). Two cautions. First, much of the rise is the model learning to close the gene: the share of samples with a parseable CDS goes from 17.5 % (7 of 40) at n = 2 to 90 % (36 of 40) at n = 64. Second, at n = 2 all four in-window sequences were ≥ 85 % identical to a pool member, so low n means copying, not failure to fold. The 32 vs 64 difference (55 ± 8 vs 65 ± 9 %) is within error, so n = 32 may be the cheaper sweet spot on Colab. The "coherence" filter removed nothing (87 in, 87 out), so it adds no information.

![Novel yield by context size and 3D-triad pass rate by context size](figures/novel_yield_and_triad_pass_v3.png)

**Experiment 2 (validation, audited run).** The table separates what each result supports from what it does not.

| Result | Supports | Does not support |
| --- | --- | --- |
| 70 of 71 designs at pLDDT ≥ 80 (mean 90.7, SD 2.6) | The sequences are on the PETase/cutinase fold manifold | Activity; the gate barely filters (98.6 % pass) |
| TM-score 0.95 / RMSD 1.4 Å to 6EQE, 0.96 / 1.3 Å to 4EB0 | Backbone matches known PETases | Any difference between designs: the spread is 0.91–0.98 |
| Ser–His 2.85–3.1 Å, His–Asp 2.7–3.1 Å in 66 designs | A catalytic arrangement is predicted | Sub-angstrom claims: predicted-structure noise is of the same order |
| 70 docked smina poses, ΔG −7.20 to −4.90 kcal/mol, 0 clash artifacts | That smina placed a ligand near Ser Oγ in a physically reasonable orientation | That the docked pose is catalytically competent — only 1 of 70 designs meets the strict reactive-pose criterion (ΔG ≤ −5.0 and attack_dist ≤ 3.5 Å) |

The honest summary is that you have a diverse, well-folded, catalytically plausible library of 70 sequences with one design — `petase_32_seq_21` — that comes close to a textbook near-attack conformation (3.42 Å, ΔG = −6.1 kcal/mol). Plausible is the key word: ESMFold was trained on natural proteins, and the PETase/cutinase family is densely represented, so a family-coherent sequence is expected to score high on pLDDT and TM-score whether or not it is active.

## Audit: five problems that change which designs to trust

The user's `run_structural_validation_v2.py` patch resolves three of the five findings from the earlier draft. Two remain (audit #4 and #5). The state of each finding:

1. **Docking tier — RESOLVED.** `run_structural_validation_v2.py` now records the catalytic Ser Oγ in the native PDB frame **before** 6EQE superposition, so the smina docking box matches the receptor file on disk. The attack distance is measured specifically to the electrophilic ester carbonyl carbon (identified by RDKit SMARTS `[CX3](=O)[OX2][CH2]`) instead of "the minimum distance to any ligand atom", and unphysical steric clashes (< 2.5 Å) are penalized. The script now raises `RuntimeError` if smina or the ligand PDBQT is missing (unless `--allow-skip-docking` is passed), so no silent geometric fallback can ever produce fake constant scores again. The 70 docked PDBQT complex files are written to `results/docking_results/` (and are in the archive). The numbers reflect real smina output: ΔG −7.20 to −4.90 kcal/mol (mean −6.19), attack distance 3.42 to 9.84 Å (mean 4.82), 0 clash artifacts, 0 designs with NAC = 1.0 (because the minimum attack distance, 3.42 Å, exceeds the 3.2 Å saturation point), and only 1 design — `petase_32_seq_21` — meets the strict reactive-pose criterion (ΔG ≤ −5.0 and attack_dist ≤ 3.5 Å). Removing the docking and NAC terms from the composite now changes the ranking meaningfully: Spearman 0.46 between the v1.5 full composite and the v2 no-docking composite among the 66 triad-intact designs. The docking terms stay in the v1.5 composite (they are real signal), but the v2 panel-A ranking still uses the no-docking composite so that the structural-confidence arm does not double-count smina output that the arXiv paper warns is also a confidence-style signal.
2. **3D triad check — MOSTLY RESOLVED.** `run_structural_validation_v2.py` replaces index-based Cα pairing with pairwise BLOSUM62 sequence alignment between the query and 6EQE chain A, then superimposes only the aligned Cα pairs. 66 of 70 designs now pass (was 56); only 4 fail; of those 4, 2 are wrongly failed (`petase_16_seq_31` with d_his_asp = 3.36 Å just over the 3.2 Å threshold, and `petase_32_seq_17` with a geometry distortion) and 2 truly fail by sequence (`petase_64_seq_24` Glu→Asp at the aligned Asp position, `petase_8_seq_24` no aligned His — same as before). The 10 designs that the v1 check wrongly dropped are now back in the gate-passing pool. Panel D (rescued) therefore shrank from 6 designs to 2.
3. **Top-10 order — NO LONGER INSIDE THE NOISE.** The earlier draft said the v1 composite's top-10 was inside the noise (span 0.016, SD 0.056, ratio 0.28). With the v3 patch, the v1.5 composite's top-10 spans 0.040 (0.6683–0.7087) against an SD of 0.0245 among the 66 triad-intact designs — a ratio of 1.65, so the top-10 is distinguishable from the noise. pLDDT alone has Spearman 0.39 with the full composite and 0.68 with the no-docking composite; the remaining signal comes from the now-meaningful smina terms. The v3 README correctly names `petase_64_seq_34 (0.7087)` as the leader (was the older README's "petase_16_seq_11 (0.768)", which never matched the table). Only **1 design** scores ≥ 0.70, 18 score ≥ 0.65, and 64 score ≥ 0.60 — so the "48 candidates ≥ 0.70" claim from the v1 README flowchart is gone in v3.
4. **README formula vs code — STILL APPLIES.** The README composite formula subtracts the affinity term (`− 0.10 × (ΔG_bind / −10)`); the code adds it (`+ 0.10 × clip(ΔG_bind / −10, 0, 1)`). With ΔG = −6.0, the README gives `−0.10 × 0.6 = −0.06` while the code gives `+0.10 × 0.6 = +0.06` — opposite signs. The actual composite_score column matches the **code** formula (verified by recomputing all 66 triad-intact scores to within 1e-3). Using the README formula instead would shift scores by up to **0.144** (was 0.116 in v2; the wider affinity range in v3, −7.20 vs −5.80, makes the discrepancy larger). The Tier-1 motif regex is now `G[HWYFQ]S.G` (strict nucleophile elbow) instead of `[GWA].[SMG].G` — fixed.
5. **Novelty — STILL APPLIES.** Novelty is still measured only against your 139-sequence pool. Evo 2 was trained on bacterial genomes that contain natural PETases and cutinases outside that pool, so a design can be a near-copy of a sequence you did not include. Search UniRef90 / NR / PDB before calling any design novel.

![Score-term variance and the 4 failed designs (2 wrongly + 2 truly)](figures/score_variance_and_wrongly_failed_v3.png)

**Things that checked out.** The designs are diverse among themselves: mean pairwise identity 53 %, maximum 75 %, 64 clusters at 70 % identity, so there are no near-duplicates. Pass rate of the alignment-based 3D check by n (n = 8: 5/6, 16: 14/15, 32: 21/22, 64: 25/26) is now flat across context sizes (the v1 index-based check showed 3/6, 14/15, 19/22, 19/26 with a Fisher p = 0.22 for n = 64 vs n = 16; the v3 alignment-based check gives Fisher p = 1.000, so there is no longer even a trend to misinterpret). 20 of 70 designs have an odd number of cysteines (IsPETase has four, LCC two), which can leave an unpaired thiol in *E. coli*.

## What the three links add

Each link fixes a different weakness: the paper fixes how you select, RFD4-Proteina fixes how far you explore, cuPotential fixes how you check chemistry. I read the abstract of the paper and the repository pages; I have not run any of the code, and neither repository page gives benchmarks for enzymes.

| Resource | What it is | Weakness it addresses | Main risk |
| --- | --- | --- | --- |
| [Confident and Wrong (arXiv 2611.17391)](https://arxiv.org/abs/2611.17391) | Supervised model trained on measured lanmodulin metal-binding affinity; 297 designs tested | Selecting by structure confidence | Binding is an easier label than turnover on insoluble PET |
| [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) | Flow-matching design model (Baker Lab and NVIDIA) for motif scaffolding, enzymes, small-molecule pockets, partial diffusion, reward-guided search | Exploration limited to the PETase family seen by Evo 2 | Needs a Linux NVIDIA GPU, ~1.7 GB weights, ~7 GB environment |
| [cuPotential](https://github.com/NVIDIA/cuPotential) | CUDA library for nonbonded forces with an OpenMM plugin; 2.6–4× faster than stock OpenMM 8.4.0 in NPT runs of 24k–409k atoms | No dynamics-based stability or persistence signal | Early access, experimental C API; repo says nothing about protein design |

**1. The paper: stop treating confidence as function.** Its panel with the highest pTM and ipLDDT expressed at 97 % but bound terbium in only 16.7 % of designs, while the function-guided panel, with the lowest confidence, bound in 83.3 % (odds about 25-fold apart, no confidence metric separated binders consistently). Your pipeline is the "structure-confidence-selected" arm: every gate and score is pLDDT, TM-score or geometry from a predicted structure. The v3 patch made the docking terms meaningful, but they are still smina outputs on a predicted structure, not a measured function label. Three actions follow. (a) Use structure only as a pass/fail plausibility gate. (b) Train a predictor on measured activity with assay conditions as inputs (temperature, substrate crystallinity, loading, pH, readout), and report ROC AUC on held-out sequence clusters. (c) Test panels chosen by different strategies side by side, as the paper did. One check I ran on your data: a homology-transfer predictor on your 30 activity-labelled enzymes gets Spearman 0.19 (p = 0.31) in leave-one-out, so the current labels are too few and too heterogeneous (median 49 % pairwise identity, mixed assays) to train on; the first job is a curated, assay-annotated dataset from the literature, PAZy and PlasticDB.

**2. RFD4-Proteina: a catalytic-site-first generator.** Evo 2 can only interpolate between the 139 sequences you give it. RFD4-Proteina can scaffold a fixed motif (Ser-His-Asp triad plus oxyanion-hole residues, taken from ligand-bound IsPETase or LCC complexes) around a ligand pocket, and can partially diffuse an existing structure to diversify it. Use it three ways: motif-scaffold new backbones around the triad and a PET fragment; partially diffuse your best Evo 2 designs to explore nearby backbones; and keep Evo 2 as an independent sequence-first arm. Caution from the paper: its reward-guided search with structure-prediction oracles would optimize exactly the confidence signal that failed, so give it a function-model or physics reward, or leave it unguided and filter afterwards.

**3. cuPotential: physics in place of the heuristic.** It accelerates OpenMM nonbonded forces, which dominate cost in explicit-solvent MD. The inference that it enables protein screening is mine, not the repo's. A workable use: take the 70 preserved docked PDBQTs from the v3 run as starting poses, run replicate MD, and measure the fraction of frames with Ser Oγ…C_ester ≤ 3.5 Å, an intact His–Ser hydrogen bond and oxyanion-hole contacts; run the same systems hot (about 340–350 K) to compare triad persistence and fluctuations against IsPETase, LCC and a thermostable variant. Calibrate first: the protocol must rank known enzymes sensibly before its output is used on designs.

## Pipeline v2 (audited): gates, then a controlled panel

The core change is conceptual: structure-derived numbers become pass/fail gates, never a score to maximize, and selection is designed as an experiment. The v3 patch tightened the gates but did not change the structure. Here is the v2 flow, with what is implemented now and what still needs GPUs or data.

1. **Generate** with Evo 2 at n = 32–64, a longer window (about 1,050 nt, since 306 aa needs 921 nt plus spacer), and activity-biased example sets (LCC, PHL7, TfCut2, BhrPETase last in the prompt). Add RFD4-Proteina arms (motif scaffolding, partial diffusion). *Not run.*
2. **Gate for plausibility:** length 241–306 aa, pLDDT ≥ 80, `G[HWYFQ]S.G` nucleophile elbow, Ser/Asp/His at IsPETase-aligned positions (alignment-based). Of the 70 folded designs, 68 pass. *Implemented (`rank_v2.py`); alignment-based check from `run_structural_validation_v2.py`.*
3. **Gate for novelty** against UniRef90 / NR / PDB, not just the 139-sequence pool. *Not run: needs the database.*
4. **3D active-site check** uses BLOSUM62 sequence alignment to map query residues onto 6EQE, then superimposes only the aligned Cα pairs. 66 of 70 pass; the 2 wrongly failed (`petase_16_seq_31`, `petase_32_seq_17`) go to panel D. *Implemented.*
5. **Docking check** captures native-frame Ser Oγ before superposition, runs smina with an 18 Å box, measures attack distance to the ester carbonyl carbon via SMARTS, penalizes clashes < 2.5 Å, fails loudly if smina is missing, writes docked PDBQTs to the archive. 0 clash artifacts, 1 strict reactive pose. *Implemented.*
6. **Score with a function model** trained on assay-annotated activity, validated by cluster-held-out AUC. *Needs data; see roadmap.*
7. **Physics check:** replicate MD with cuPotential-accelerated OpenMM on the 70 preserved docked poses, calibrated on controls. *Not run.*
8. **Select a diverse, controlled panel** rather than the top N. *Implemented.*

What v2 (audited) changes in your current library, from the script's output:

| Change | v1 | v2 (audited) |
| --- | --- | --- |
| Designs surviving gates | 56 (3D triad, index-based) | 66 (3D triad, alignment-based); 68 by sequence triad |
| Docking / NAC terms | 2 of 5 score terms; v1.5 README claimed 13/70 clash artifacts | 0 clash artifacts; affinity −7.20 to −4.90 kcal/mol; attack distance 3.42–9.84 Å; Spearman 0.46 between full and no-dock composite |
| Ranking | Top-10 inside the noise (span 0.016 = 0.28 SD) | Top-10 distinguishable (span 0.040 = 1.65 SD); only 1 design ≥ 0.70 |
| Diversity | Not considered | Farthest-point selection on identity (panel B) |
| Failure recovery | 14 designs dropped, 11 with −28 score | 4 dropped; 2 rescued as panel D |
| Documented leader | README flowchart named `petase_16_seq_11 (0.768)` (never matched the table) | README v3 names `petase_64_seq_34 (0.7087)` (matches the table) |
| README formula sign | Subtracts affinity term, code adds it (max shift 0.116) | Same sign error, max shift 0.144 (because the affinity range is wider in v3) |

## Wet-lab panel: 32 constructs that test the pipeline itself

The panel is built so that the first experiment answers "does any in-silico filter predict activity?", the question the arXiv paper answers for lanmodulin. If panel A (your current favourites) hits no more often than random panel C, the structural ranking has no value and the function model becomes priority one.

| Panel | Constructs | Selection rule | What it tests |
| --- | --- | --- | --- |
| A: structure top | 8 | Highest no-docking composite among 3D-triad-intact designs (mean pLDDT 93.8) | The v1 ranking |
| B: diverse | 8 | Farthest-point sampling on identity, starting from the most novel (mean pLDDT 89.4, mean 61 % to pool) | Whether lower confidence and more novelty cost activity |
| C: random | 8 | Random gate-passers, seed 7 (mean pLDDT 89.8) | Baseline for A and B |
| D: rescued | 2 | v1.5 3D-triad failures with Ser/Asp/His by sequence (mean pLDDT 89.5) — only 2 remain after the alignment fix rescued 10 of 12 | Whether the residual 2 failures are real or noise |
| E: knockouts | 2 | Ser→Ala in the catalytic serine of the two best A designs (`petase_16_seq_11_S131A`, `petase_64_seq_34_S130A`) | That activity is catalytic, not from host esterases |
| F: natural controls | 4 | IsPETase, TfCut2, LCC, PHL7 from your pool | Assay dynamic range |

Add by hand: an empty-vector lane, a non-PETase α/β-hydrolase (for example EstB, which your pool report dropped) and LCC-ICCG if you can source its sequence. Suggested readouts, in order: soluble expression, melting temperature by DSF, BHET hydrolysis by HPLC, then amorphous PET film at 30–40 °C for mesophilic designs and 60–70 °C for LCC-like designs. Express disulfide-containing designs in the periplasm or in an oxidizing strain; 20 of 70 designs have an odd number of Cys, so consider mutating the unpaired one.

How to read it: 8 vs 8 constructs is small, so decide on large gaps only (for example 5 of 8 vs 1 of 8 hits), and report hit rates with confidence intervals as the paper does. The panel table, FASTA, features, the v2 patch (`run_structural_validation_v2.py`) and `rank_v2.py` are delivered with this report.

## Future work, in priority order

Phase 1 (Fix) is now substantially done by the user's `run_structural_validation_v2.py` patch — only the README-formula sign and the novelty DB search remain. The rest depends on what the panel shows.

| Phase | Work | Needs | Gate to move on |
| --- | --- | --- | --- |
| 1. Fix (days) — mostly done | ✅ Alignment-based superposition; ✅ fail-loudly smina; ✅ preserved docked PDBQTs; ✅ strict motif regex; ✅ unrounded pLDDT + active-site pLDDT; ✅ README "81.4 % NAC" corrected to the data. **Remaining:** (a) fix the README composite-formula sign (subtract → add, max shift 0.144); (b) MMseqs2 of all designs against UniRef90 / NR / PDB | UniRef90 / NR / PDB database | Drop designs ≥ 85 % to anything outside the pool |
| 2. Panel (weeks) | Order and test the 32-construct panel | Gene synthesis, assays | Hit rates per panel with confidence intervals |
| 3. Function model (weeks, parallel) | Curate assay-annotated PET activity data (variants of IsPETase, LCC, PHL7 and others, with temperature, substrate, readout); ESM embeddings plus ridge or Gaussian process; cluster-held-out ROC AUC | Literature curation | AUC clearly above structure-only features on held-out clusters |
| 4. Physics (weeks) | Take the 70 preserved docked PDBQTs as starting poses; replicate MD with cuPotential-accelerated OpenMM at 300 K and about 345 K; calibrate on IsPETase, LCC, LCC-ICCG | A GPU supported by cuPotential; early-access install | Protocol ranks known enzymes correctly |
| 5. Explore (weeks) | RFD4-Proteina motif scaffolding around the triad and a ligand pocket; partial diffusion of panel hits; second predictor (AlphaFold or Boltz-style) to measure agreement | Linux NVIDIA GPU | Designs pass gates 2–4 with more sequence diversity than Evo 2 alone |
| 6. Close the loop | Retrain the function model on panel results; use it to rank the next Evo 2 and RFD4 batch; repeat | Panel data | Hit rate in the next round exceeds the random panel |

Two cheap generation improvements to bundle with phase 1: sample at n = 32 rather than 64 (55 % vs 65 % novel yield is within error and prefill is cheaper), and keep every raw continuation so that you can rescore later without re-running Colab.

## Sources and files

- [Confident and Wrong: Structure Confidence Is a Poor Selection Filter for Protein Function](https://arxiv.org/abs/2611.17391), arXiv 2611.17391 (abstract page read 21 Nov 2026)
- [RFD4-Proteina repository](https://github.com/RosettaCommons/RFD4-Proteina) and [cuPotential repository](https://github.com/NVIDIA/cuPotential) (README pages read 21 Nov 2026)
- Your archive `EVO_2_PETase-main (3).zip`: every count above is recomputed from its TSV, FASTA and `.m8` files, and `run_structural_validation_v2.py` was read in full
- Delivered alongside this report: `rank_v2.py` (the v3 version that ingests the new smina columns), `wetlab_panel_v2.tsv`, `wetlab_panel_v2.faa`, `all_designs_v2_features.tsv`, `run_structural_validation_v2.py` (the audited pipeline patch), and three figures in `figures/` (the screening funnel, the yield-and-pass-rate dual-axis chart, and the score-variance + 4-failed-designs panel)
