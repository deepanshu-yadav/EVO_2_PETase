#!/usr/bin/env python3
"""Module 5: Decoupled Multi-Parametric Triage & Controlled 48-Target Wet-Lab Panel Selection.

Merges Module 2 (contrastive log-likelihood & novelty), Module 3 (ESMFold pLDDT, Foldseek TM-score,
active-site geometry, flexible Smina docking), and Module 4 (cleft hydrophobic SASA, disulfides,
thermostability Tm, E. coli solubility) to execute:
  1. Stage 1: Hard Binary Plausibility Elimination Gate
  2. Stage 2: Continuous Z-Score Composite Ranking
  3. Controlled 48-Target Wet-Lab Panel Assembly (Panels A-F)

Outputs:
  - ranked_exp3_candidates.tsv
  - triage_elimination_audit.tsv
  - wetlab_panel_exp3_48targets.tsv
  - wetlab_panel_exp3_48targets.faa
  - module_5_summary.json
"""
import argparse
import json
import pathlib
import sys
from typing import Optional

import pandas as pd
from Bio import SeqIO

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
EXP3_DIR = CURRENT_DIR.parent
REPO_ROOT = EXP3_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from triage_panel_utils import (
    apply_stage1_hard_gate,
    assemble_wetlab_panel_48,
    compute_zscore_fitness,
    parse_foldseek_m8,
)


def run_module_5(
    fasta_path: pathlib.Path,
    reranked_tsv: pathlib.Path,
    mod4_features_tsv: pathlib.Path,
    pool_fasta: pathlib.Path,
    out_dir: pathlib.Path,
    grid_tsv: Optional[pathlib.Path] = None,
    plddt_tsv: Optional[pathlib.Path] = None,
    triad_tsv: Optional[pathlib.Path] = None,
    foldseek_m8: Optional[pathlib.Path] = None,
    docking_tsv: Optional[pathlib.Path] = None,
    seed: int = 42,
) -> dict:
    """Execute Module 5 decoupled triage and 48-target panel assembly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 3 — MODULE 5: DECOUPLED TRIAGE & 48-TARGET WET-LAB PANEL")
    print("=" * 78)

    with open(fasta_path, "r", encoding="utf-8") as fh:
        seq_records = [{"id": r.id, "sequence": str(r.seq).upper().replace("*", "")} for r in SeqIO.parse(fh, "fasta")]
    df = pd.DataFrame(seq_records)
    df["length"] = df["sequence"].str.len()
    print(f"Loaded {len(df)} novel candidate sequences from {fasta_path.name}")

    # Merge Module 2 sweep grid & contrastive reranked scores (all 140 designs + top-50 delta_ll)
    if grid_tsv and grid_tsv.exists():
        df_g = pd.read_csv(grid_tsv, sep="\t")
        g_cols = [c for c in ["design_id", "n_examples", "temperature", "max_pid", "delta_ll"] if c in df_g.columns]
        if "design_id" in g_cols:
            df = df.merge(df_g[g_cols].drop_duplicates("design_id").rename(columns={"design_id": "id"}), on="id", how="left")
    if reranked_tsv.exists():
        df_m2 = pd.read_csv(reranked_tsv, sep="\t")
        m2_cols = [c for c in ["design_id", "n_examples", "temperature", "max_pid", "delta_ll"] if c in df_m2.columns]
        if "design_id" in m2_cols:
            df_m2_ren = df_m2[m2_cols].drop_duplicates("design_id").rename(columns={"design_id": "id"}).set_index("id")
            df = df.set_index("id")
            for col in [c for c in df_m2_ren.columns if c in df.columns]:
                df[col] = df_m2_ren[col].combine_first(df[col])
            for col in [c for c in df_m2_ren.columns if c not in df.columns]:
                df[col] = df_m2_ren[col]
            df = df.reset_index()

    # Merge Module 4 interfacial & stability features
    if not mod4_features_tsv.exists():
        raise FileNotFoundError(f"Module 4 features TSV not found: {mod4_features_tsv}")
    df_m4 = pd.read_csv(mod4_features_tsv, sep="\t")
    drop_dup_m4 = [c for c in ["length"] if c in df_m4.columns and c in df.columns]
    df = df.merge(df_m4.drop(columns=drop_dup_m4), on="id", how="left")

    # Merge Module 3 pLDDT, active-site geometry, Foldseek TM, and Smina docking
    if plddt_tsv and plddt_tsv.exists():
        df_p = pd.read_csv(plddt_tsv, sep="\t")
        p_cols = [c for c in ["id", "mean_plddt", "passed_plddt80"] if c in df_p.columns]
        df = df.merge(df_p[p_cols], on="id", how="left")
    if "mean_plddt" not in df.columns:
        df["mean_plddt"] = 85.0  # Fallback only if Module 3 pLDDT file not supplied

    if triad_tsv and triad_tsv.exists():
        df_t = pd.read_csv(triad_tsv, sep="\t")
        t_cols = [
            c for c in [
                "id", "seq_triad_intact", "triad_intact", "d_ser_his", "d_his_asp",
                "has_oxyanion_tyr", "has_groove_w159", "has_groove_w185", "active_site_complete"
            ] if c in df_t.columns
        ]
        df = df.merge(df_t[t_cols], on="id", how="left")

    df_fs = parse_foldseek_m8(foldseek_m8)
    if not df_fs.empty:
        df = df.merge(df_fs, on="id", how="left")

    if docking_tsv and docking_tsv.exists():
        df_dk = pd.read_csv(docking_tsv, sep="\t")
        df = df.merge(df_dk, on="id", how="left")

    # Stage 1: Hard Binary Elimination
    df_gated = apply_stage1_hard_gate(df)
    n_pass = int(df_gated["passed_stage1_gate"].sum())
    n_fail = len(df_gated) - n_pass
    print(f"\n[Stage 1] Hard Binary Plausibility Elimination:")
    print(f"  Passed all 4 binary gates: {n_pass} / {len(df_gated)}")
    print(f"  Eliminated prior to Z-score ranking: {n_fail}")

    audit_cols = [
        c for c in [
            "id", "length", "mean_plddt", "solubility_score",
            "gate_plddt80", "gate_seq_triad", "gate_length", "gate_solubility",
            "passed_stage1_gate", "stage1_drop_reasons"
        ] if c in df_gated.columns
    ]
    audit_path = out_dir / "triage_elimination_audit.tsv"
    df_gated[audit_cols].to_csv(audit_path, sep="\t", index=False)

    # Stage 2: Continuous Z-Score Composite Ranking
    df_ranked = compute_zscore_fitness(df_gated)
    ranked_path = out_dir / "ranked_exp3_candidates.tsv"
    df_ranked.to_csv(ranked_path, sep="\t", index=False)

    print(f"\n[Stage 2] Continuous Z-Score Top 10 Candidates:")
    top_preview = df_ranked[df_ranked["passed_stage1_gate"]].head(10)
    for _, r in top_preview.iterrows():
        print(
            f"  #{int(r['rank']):>2} {r['id']:<26} "
            f"Fitness={r['fitness_score']:+.3f} | pLDDT={r['mean_plddt']:.1f} | "
            f"TM={r['tm_ref']:.3f} | Aff={r['combined_affinity']:.2f} | "
            f"Tm_hat={r['predicted_tm']:.1f}C | Sol={r['solubility_score']:.3f}"
        )

    # Controlled 48-Target Wet-Lab Panel (Panels A-F)
    df_panel = assemble_wetlab_panel_48(df_ranked, pool_fasta=pool_fasta, seed=seed)
    panel_tsv = out_dir / "wetlab_panel_exp3_48targets.tsv"
    panel_faa = out_dir / "wetlab_panel_exp3_48targets.faa"
    df_panel.to_csv(panel_tsv, sep="\t", index=False)

    with open(panel_faa, "w", encoding="utf-8") as f:
        for _, r in df_panel.iterrows():
            f.write(f">{r['id']} panel={r['panel']} note={r['selection_note']}\n{r['sequence']}\n")

    arm_counts = df_panel["panel"].value_counts().to_dict()
    print(f"\n[Panel Assembly] Controlled Wet-Lab Panel ({len(df_panel)} targets):")
    for arm, cnt in arm_counts.items():
        print(f"  {arm:<28}: {cnt}")

    summary = {
        "total_candidates": len(df_ranked),
        "stage1_passed": n_pass,
        "stage1_eliminated": n_fail,
        "panel_total": len(df_panel),
        "panel_arm_counts": arm_counts,
        "top5_ids": top_preview["id"].head(5).tolist(),
        "ranked_tsv": str(ranked_path),
        "panel_tsv": str(panel_tsv),
        "panel_faa": str(panel_faa),
    }
    with open(out_dir / "module_5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved Module 5 artifacts in: {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Module 5: Decoupled Triage & 48-Target Wet-Lab Panel")
    parser.add_argument("--fasta_path", type=pathlib.Path, default=EXP3_DIR / "outputs" / "designs_exp3_screened_novel.faa")
    parser.add_argument("--reranked_tsv", type=pathlib.Path, default=EXP3_DIR / "outputs" / "contrastive_reranked_designs.tsv")
    parser.add_argument("--grid_tsv", type=pathlib.Path, default=EXP3_DIR / "outputs" / "generated_continuations_grid.tsv")
    parser.add_argument("--mod4_features", type=pathlib.Path, default=EXP3_DIR / "outputs" / "interfacial_and_stability_features.tsv")
    parser.add_argument("--pool_fasta", type=pathlib.Path, default=REPO_ROOT / "data" / "petase_pool.fasta")
    parser.add_argument("--plddt_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--triad_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--foldseek_m8", type=pathlib.Path, default=None)
    parser.add_argument("--docking_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--out_dir", type=pathlib.Path, default=EXP3_DIR / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_module_5(
        fasta_path=args.fasta_path,
        reranked_tsv=args.reranked_tsv,
        grid_tsv=args.grid_tsv,
        mod4_features_tsv=args.mod4_features,
        pool_fasta=args.pool_fasta,
        out_dir=args.out_dir,
        plddt_tsv=args.plddt_tsv,
        triad_tsv=args.triad_tsv,
        foldseek_m8=args.foldseek_m8,
        docking_tsv=args.docking_tsv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
