#!/usr/bin/env python3
"""Stratified, Cluster-Balanced Prompt Sampler for Experiment 3.

Enforces:
1. Cluster balancing: at most 1 exemplar per 70% sequence cluster.
2. Thermal stratification: partitions pool into thermo, meso, and general background.
3. Curriculum / Recency ordering: sorts exemplars by ascending thermostability (T_m / T_opt)
   so the most thermotolerant exemplars occupy the privileged positions immediately
   preceding the continuation prefix.
4. DNA prompt formatting: standard GGGGGGGG spacers.
"""
import copy
import json
import pathlib
import random
from typing import Dict, List, Any, Optional, Tuple

SPACER = "GGGGGGGG"


def compose_prompt(dna_seqs: List[str], spacer: str = SPACER) -> str:
    """Format prompt with spacer before each exemplar and a trailing spacer for continuation."""
    return "".join(spacer + s for s in dna_seqs) + spacer


def sample_cluster_balanced(
    candidate_ids: List[str],
    id_to_cluster: Dict[str, int],
    n: int,
    rng: random.Random,
) -> List[str]:
    """Select n exemplars from candidate_ids such that no two share the same 70% cluster.

    If n > number of available clusters in candidates, raises ValueError.
    """
    # Group candidates by cluster
    clusters_to_cands: Dict[int, List[str]] = {}
    for cid_str in candidate_ids:
        c_idx = id_to_cluster[cid_str]
        clusters_to_cands.setdefault(c_idx, []).append(cid_str)

    available_clusters = list(clusters_to_cands.keys())
    if n > len(available_clusters):
        raise ValueError(
            f"Requested n={n} exemplars, but only {len(available_clusters)} distinct 70% "
            f"clusters are available in the candidate pool without cluster duplication."
        )

    # Sample n distinct clusters
    selected_clusters = rng.sample(available_clusters, n)

    # From each chosen cluster, pick 1 exemplar
    chosen_exemplars = []
    for c_idx in selected_clusters:
        member = rng.choice(clusters_to_cands[c_idx])
        chosen_exemplars.append(member)

    return chosen_exemplars


def sample_curriculum_stratified(
    candidate_ids: List[str],
    id_to_cluster: Dict[str, int],
    metadata: Dict[str, Dict[str, Any]],
    n: int,
    rng: random.Random,
) -> List[str]:
    """Sample n cluster-balanced exemplars ensuring thermophilic representation for curriculum.

    Guarantees that at least one (and up to n//4 + 1) distinct thermophilic clusters are sampled,
    with the remaining drawn from mesophilic/intermediate clusters, with zero cluster overlap.
    """
    # Group candidates strictly by cluster
    clusters_to_cands: Dict[int, List[str]] = {}
    for cid_str in candidate_ids:
        c_idx = id_to_cluster[cid_str]
        clusters_to_cands.setdefault(c_idx, []).append(cid_str)

    # Partition cluster IDs into thermo vs other (disjoint partition)
    thermo_cluster_ids = []
    other_cluster_ids = []
    for c_idx, members in clusters_to_cands.items():
        has_thermo = any(metadata[m].get("thermal_class") == "thermo" for m in members)
        if has_thermo:
            thermo_cluster_ids.append(c_idx)
        else:
            other_cluster_ids.append(c_idx)

    # How many thermo clusters to sample
    n_thermo = max(1, min(max(1, n // 4), len(thermo_cluster_ids)))
    n_other = n - n_thermo

    if n_other > len(other_cluster_ids):
        n_other = len(other_cluster_ids)
        n_thermo = min(n - n_other, len(thermo_cluster_ids))

    selected_thermo = rng.sample(thermo_cluster_ids, n_thermo)
    selected_other = rng.sample(other_cluster_ids, n_other)

    chosen_exemplars = []
    for c_idx in selected_other:
        # Choose from non-thermo members if possible
        cands = [m for m in clusters_to_cands[c_idx] if metadata[m].get("thermal_class") != "thermo"]
        chosen_exemplars.append(rng.choice(cands if cands else clusters_to_cands[c_idx]))

    for c_idx in selected_thermo:
        # Choose specifically a thermo member from this cluster
        cands = [m for m in clusters_to_cands[c_idx] if metadata[m].get("thermal_class") == "thermo"]
        chosen_exemplars.append(rng.choice(cands if cands else clusters_to_cands[c_idx]))

    return chosen_exemplars


def sort_by_curriculum(
    exemplar_ids: List[str],
    metadata: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Sort exemplars in ascending order of thermostability (T_m, with T_opt fallback).

    T_m(e_1) <= T_m(e_2) <= ... <= T_m(e_n)
    This places the highest melting temperature exemplars in the recency positions.
    """
    def get_temp_key(eid: str) -> float:
        info = metadata.get(eid, {})
        tm = info.get("t_m")
        if tm is not None:
            return float(tm)
        t_opt = info.get("t_opt")
        if t_opt is not None:
            return float(t_opt) + 6.0
        return 35.0

    return sorted(exemplar_ids, key=get_temp_key)


def build_stratified_prompt_cell(
    cell_name: str,
    family_label: str,
    n_examples: int,
    k_draws: int,
    mode: str,
    candidate_pool_ids: List[str],
    id_to_cluster: Dict[str, int],
    metadata: Dict[str, Dict[str, Any]],
    dna_of: Dict[str, str],
    base_seed: int = 1337,
    spacer: str = SPACER,
) -> Dict[str, Any]:
    """Construct an experimental prompt cell containing K independent stratified draws."""
    draws = []
    for p in range(k_draws):
        rng = random.Random(base_seed + p * 1000 + n_examples)

        # 1. Cluster-balanced draw
        if mode == "curriculum":
            selected_ids = sample_curriculum_stratified(
                candidate_ids=candidate_pool_ids,
                id_to_cluster=id_to_cluster,
                metadata=metadata,
                n=n_examples,
                rng=rng,
            )
        else:
            selected_ids = sample_cluster_balanced(
                candidate_ids=candidate_pool_ids,
                id_to_cluster=id_to_cluster,
                n=n_examples,
                rng=rng,
            )

        # 2. Curriculum ordering (if requested or in thermo/curriculum mode)
        if mode in ("curriculum", "thermo", "ascending"):
            selected_ids = sort_by_curriculum(selected_ids, metadata)

        # 3. Fetch DNA sequences
        dna_seqs = [dna_of[eid] for eid in selected_ids]
        prompt = compose_prompt(dna_seqs, spacer=spacer)

        # Invariant assertions
        assert prompt.startswith(spacer) and prompt.endswith(spacer), "Prompt spacer boundary error"
        assert prompt.count(spacer) == n_examples + 1, f"Expected {n_examples+1} spacers, found {prompt.count(spacer)}"

        # Verify cluster balance on the draw
        chosen_clusters = [id_to_cluster[eid] for eid in selected_ids]
        assert len(set(chosen_clusters)) == len(chosen_clusters), "Cluster collision in draw!"

        draws.append({
            "draw_idx": p,
            "seed": base_seed + p * 1000 + n_examples,
            "exemplar_ids": selected_ids,
            "temperatures": [metadata[eid].get("t_m") or metadata[eid].get("t_opt") for eid in selected_ids],
            "cluster_ids": chosen_clusters,
            "prompt": prompt,
            "prompt_length_nt": len(prompt),
            "max_exemplar_aa_len": max(metadata[eid]["length_aa"] for eid in selected_ids),
        })

    max_aa = max(d["max_exemplar_aa_len"] for d in draws)
    gen_length = max_aa * 3 + 30

    return {
        "cell_name": cell_name,
        "family": family_label,
        "mode": mode,
        "n_examples": n_examples,
        "k_draws": k_draws,
        "gen_length": gen_length,
        "draws": draws,
    }


def generate_all_prompt_suites(
    metadata: Dict[str, Dict[str, Any]],
    clustering_info: Dict[str, Any],
    dna_of: Dict[str, str],
    k_draws: int = 50,
    base_seed: int = 1337,
    n_curriculum_grid: Optional[List[int]] = None,
    n_thermo_grid: Optional[List[int]] = None,
    n_meso_grid: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Generate the complete set of stratified prompt suites for Experiment 3.

    Outputs:
      1. thermo_curriculum: Deep context scaling with ascending Tm ordering.
      2. thermo: Pure thermophilic cutinase prompt suite.
      3. meso: Pure mesophilic IsPETase prompt suite.
      4. contrastive: Paired thermo vs general background suites for contrastive LL scoring.
    """
    id_to_cluster = clustering_info["id_to_cluster"]
    all_pool_ids = list(metadata.keys())

    # Sub-pool partitioning
    thermo_ids = [k for k, v in metadata.items() if v["thermal_class"] == "thermo"]
    meso_ids = [k for k, v in metadata.items() if v["thermal_class"] == "meso"]
    general_ids = [k for k, v in metadata.items() if v["thermal_class"] in ("meso", "intermediate")]

    # Count distinct clusters per pool
    thermo_clusters = len(set(id_to_cluster[k] for k in thermo_ids))
    meso_clusters = len(set(id_to_cluster[k] for k in meso_ids))
    all_clusters = len(set(id_to_cluster[k] for k in all_pool_ids))

    print(f"Available 70% clusters: Total={all_clusters}, Thermo={thermo_clusters}, Meso={meso_clusters}")

    # Set default grids bounded by cluster availability
    if n_curriculum_grid is None:
        n_curriculum_grid = [n for n in [8, 16, 32, 64] if n <= all_clusters]
    if n_thermo_grid is None:
        n_thermo_grid = [n for n in [4, 8, 12, 16] if n <= thermo_clusters]
    if n_meso_grid is None:
        n_meso_grid = [n for n in [4, 8, 16, 32] if n <= meso_clusters]

    suites: Dict[str, Any] = {}

    # 1. Thermo Curriculum Suite (Ascending Tm across pool, thermophiles in recency)
    curriculum_cells = []
    for n in n_curriculum_grid:
        cell = build_stratified_prompt_cell(
            cell_name=f"curriculum_n{n:03d}",
            family_label="petase_curriculum",
            n_examples=n,
            k_draws=k_draws,
            mode="curriculum",
            candidate_pool_ids=all_pool_ids,
            id_to_cluster=id_to_cluster,
            metadata=metadata,
            dna_of=dna_of,
            base_seed=base_seed,
        )
        curriculum_cells.append(cell)
    suites["prompts_thermo_curriculum"] = {
        "suite_name": "prompts_thermo_curriculum",
        "description": "Thermally stratified prompt suite with ascending Tm curriculum ordering",
        "cells": curriculum_cells,
    }

    # 2. Pure Thermophilic Suite
    thermo_cells = []
    for n in n_thermo_grid:
        cell = build_stratified_prompt_cell(
            cell_name=f"thermo_n{n:03d}",
            family_label="petase_thermo",
            n_examples=n,
            k_draws=k_draws,
            mode="thermo",
            candidate_pool_ids=thermo_ids,
            id_to_cluster=id_to_cluster,
            metadata=metadata,
            dna_of=dna_of,
            base_seed=base_seed,
        )
        thermo_cells.append(cell)
    suites["prompts_thermo"] = {
        "suite_name": "prompts_thermo",
        "description": "Pure thermophilic cutinase exemplars (Tm >= 70 C, Topt >= 65 C)",
        "cells": thermo_cells,
    }

    # 3. Pure Mesophilic Suite
    meso_cells = []
    for n in n_meso_grid:
        cell = build_stratified_prompt_cell(
            cell_name=f"meso_n{n:03d}",
            family_label="petase_meso",
            n_examples=n,
            k_draws=k_draws,
            mode="meso",
            candidate_pool_ids=meso_ids,
            id_to_cluster=id_to_cluster,
            metadata=metadata,
            dna_of=dna_of,
            base_seed=base_seed,
        )
        meso_cells.append(cell)
    suites["prompts_meso"] = {
        "suite_name": "prompts_meso",
        "description": "Pure mesophilic PETase exemplars (IsPETase scaffold, Tm <= 57 C)",
        "cells": meso_cells,
    }

    # 4. Contrastive Evaluation Suite (Paired thermo vs general background for delta-LL)
    # Pairs each thermo prompt with a background general prompt of identical n
    contrastive_pairs = []
    n_contrast = min(thermo_clusters, 8)  # standard n=8 contrastive probe
    for p in range(min(k_draws, 25)):
        rng_t = random.Random(base_seed + p * 1000 + n_contrast)
        rng_g = random.Random(base_seed + 9999 + p * 1000 + n_contrast)

        t_ids = sample_cluster_balanced(thermo_ids, id_to_cluster, n_contrast, rng_t)
        t_ids = sort_by_curriculum(t_ids, metadata)
        g_ids = sample_cluster_balanced(general_ids, id_to_cluster, n_contrast, rng_g)

        t_prompt = compose_prompt([dna_of[i] for i in t_ids])
        g_prompt = compose_prompt([dna_of[i] for i in g_ids])

        contrastive_pairs.append({
            "pair_idx": p,
            "n_examples": n_contrast,
            "thermo_exemplars": t_ids,
            "thermo_prompt": t_prompt,
            "general_exemplars": g_ids,
            "general_prompt": g_prompt,
        })
    suites["prompts_contrastive"] = {
        "suite_name": "prompts_contrastive",
        "description": "Paired thermophilic vs general background prompts for delta-LL reranking",
        "n_examples": n_contrast,
        "pairs": contrastive_pairs,
    }

    return suites
