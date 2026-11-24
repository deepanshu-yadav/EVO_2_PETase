#!/usr/bin/env python3
"""Module 1: Thermally Stratified & Cluster-Balanced Prompt Engineering.

Pipeline Executable:
1. Annotates all 139 pool sequences with verified biophysical thermal metadata (Tm, Topt).
2. Performs 70% sequence identity clustering to balance taxonomic representations.
3. Generates stratified prompt suites:
   - prompts_thermo_curriculum.json (ascending Tm ordering, thermophile recency bias)
   - prompts_thermo.json (pure thermophilic cutinases)
   - prompts_meso.json (pure mesophilic IsPETase homologs)
   - prompts_contrastive.json (paired thermo vs general background for delta-LL scoring)
4. Emits detailed token prefill, memory, and cluster balancing audit reports.

Usage:
  python mod_01_stratified_prompts.py \
      --fasta ../../../data/petase_pool.fasta \
      --codons ../../../data/petase_codons.json \
      --panda ../../../data/panda.xlsx \
      --report ../../../data/petase_pool_report.tsv \
      --out ../outputs
"""
import argparse
import json
import pathlib
import sys
from typing import Dict, Any

from Bio import SeqIO

# Support direct execution from any directory
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from thermal_metadata import annotate_petase_pool, export_metadata_tsv
from cluster_balancer import cluster_pool, verify_cluster_balance
from prompt_sampler import generate_all_prompt_suites


def run_module_1(
    fasta_path: pathlib.Path,
    codons_path: pathlib.Path,
    panda_path: pathlib.Path,
    report_path: pathlib.Path,
    out_dir: pathlib.Path,
    k_draws: int = 50,
    seed: int = 1337,
    force_recompute_clusters: bool = False,
) -> Dict[str, Any]:
    """Execute the end-to-end prompt engineering workflow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 3 — MODULE 1: THERMALLY STRATIFIED PROMPT ENGINEERING")
    print("=" * 78)

    # 1. Load codon cache and verify alignment with FASTA
    print(f"\n[1/4] Loading FASTA and codon cache...")
    cache = json.loads(codons_path.read_text(encoding="utf-8"))
    pool_records = list(SeqIO.parse(fasta_path, "fasta"))
    assert len(pool_records) == len(cache["proteins"]), (
        f"Mismatch: {len(pool_records)} FASTA sequences vs {len(cache['proteins'])} codon cache proteins"
    )
    assert all(str(r.seq) == p for r, p in zip(pool_records, cache["proteins"])), (
        "FASTA protein sequences do not match codon cache proteins exactly!"
    )
    dna_of = {r.id: d for r, d in zip(pool_records, cache["dna"])}
    print(f"  Loaded {len(pool_records)} sequences from {fasta_path.name}")
    print(f"  Codon cache verified: {len(dna_of)} DNA sequences ready")

    # 2. Annotate thermal metadata
    print(f"\n[2/4] Annotating thermal metadata (Tm, Topt, taxonomy)...")
    metadata = annotate_petase_pool(fasta_path, panda_path, report_path)
    meta_tsv = out_dir / "metadata_pool_annotated.tsv"
    export_metadata_tsv(metadata, meta_tsv)

    counts = {"thermo": 0, "meso": 0, "intermediate": 0}
    for v in metadata.values():
        counts[v["thermal_class"]] += 1
    print(f"  Thermal Stratification: {counts['thermo']} Thermo | {counts['meso']} Meso | {counts['intermediate']} Intermediate")

    # 3. Cluster balancing (MMseqs2 70% identity or Python fallback)
    print(f"\n[3/4] Performing 70% sequence identity clustering...")
    clusters_cache_path = out_dir / "clusters_70pct.json"
    clustering_res = cluster_pool(
        fasta_path=fasta_path,
        min_seq_id=0.70,
        min_cov=0.80,
        cache_path=clusters_cache_path,
        force_recompute=force_recompute_clusters,
    )
    print(f"  Clustering completed: {clustering_res['num_clusters']} clusters from {len(pool_records)} sequences "
          f"(Method: {clustering_res['method']})")

    # 4. Generate prompt suites
    print(f"\n[4/4] Generating stratified prompt suites (K={k_draws} draws/cell, seed={seed})...")
    suites = generate_all_prompt_suites(
        metadata=metadata,
        clustering_info=clustering_res,
        dna_of=dna_of,
        k_draws=k_draws,
        base_seed=seed,
    )

    manifest = {}
    for suite_key, suite_data in suites.items():
        out_file = out_dir / f"{suite_key}.json"
        out_file.write_text(json.dumps(suite_data, indent=2), encoding="utf-8")
        manifest[suite_key] = str(out_file)
        if "cells" in suite_data:
            total_prompts = sum(len(c["draws"]) for c in suite_data["cells"])
            max_ctx = max(max(d["prompt_length_nt"] for d in c["draws"]) for c in suite_data["cells"])
            print(f"  Wrote {out_file.name:<32} | {len(suite_data['cells'])} cells | {total_prompts:4d} prompts | max ctx {max_ctx:,} nt")
        elif "pairs" in suite_data:
            print(f"  Wrote {out_file.name:<32} | {len(suite_data['pairs']):4d} contrastive pairs")

    # 5. Export summary audit
    summary = {
        "pool_size": len(pool_records),
        "thermal_counts": counts,
        "num_clusters_70pct": clustering_res["num_clusters"],
        "clustering_method": clustering_res["method"],
        "k_draws": k_draws,
        "base_seed": seed,
        "manifest": manifest,
    }
    (out_dir / "module_1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nModule 1 complete. All outputs saved to {out_dir}")
    print("=" * 78)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3 - Module 1: Prompt Engineering")
    parser.add_argument("--fasta", default="data/petase_pool.fasta", help="Path to petase_pool.fasta")
    parser.add_argument("--codons", default="data/petase_codons.json", help="Path to petase_codons.json")
    parser.add_argument("--panda", default="data/panda.xlsx", help="Path to panda.xlsx")
    parser.add_argument("--report", default="data/petase_pool_report.tsv", help="Path to petase_pool_report.tsv")
    parser.add_argument("--out", default="experiments/experiment_3_deep_exploration/outputs", help="Output directory")
    parser.add_argument("--k_draws", type=int, default=50, help="Number of draws per cell")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    parser.add_argument("--recluster", action="store_true", help="Force recomputation of clustering")
    args = parser.parse_args()

    run_module_1(
        fasta_path=pathlib.Path(args.fasta).resolve(),
        codons_path=pathlib.Path(args.codons).resolve(),
        panda_path=pathlib.Path(args.panda).resolve(),
        report_path=pathlib.Path(args.report).resolve(),
        out_dir=pathlib.Path(args.out).resolve(),
        k_draws=args.k_draws,
        seed=args.seed,
        force_recompute_clusters=args.recluster,
    )


if __name__ == "__main__":
    main()
