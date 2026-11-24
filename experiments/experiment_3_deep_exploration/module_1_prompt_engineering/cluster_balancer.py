#!/usr/bin/env python3
"""70% Sequence Identity Clustering & Cluster Balancing for Experiment 3.

Implements MMseqs2 easy-cluster integration with a pure-Python fallback (via Biopython
BLOSUM62 global alignment) for cross-platform portability.
Provides verification utilities to ensure at most 1 exemplar is sampled per 70% cluster.
"""
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices


def make_pairwise_aligner() -> Align.PairwiseAligner:
    """Instantiate standard global aligner matching MMseqs2 / Biopython conventions."""
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def compute_global_identity(aligner: Align.PairwiseAligner, seq_a: str, seq_b: str) -> Tuple[float, float]:
    """Compute global sequence identity and query coverage between two sequences."""
    aln = aligner.align(seq_a, seq_b)[0]
    q, t = aln.aligned
    matches = sum(
        x == y
        for (a0, a1), (b0, b1) in zip(q, t)
        for x, y in zip(seq_a[a0:a1], seq_b[b0:b1])
    )
    ident = matches / max(min(len(seq_a), len(seq_b)), 1)
    cov = sum(a1 - a0 for a0, a1 in q) / max(len(seq_a), 1)
    return ident, cov


def cluster_with_mmseqs(
    fasta_path: pathlib.Path,
    min_seq_id: float = 0.70,
    min_cov: float = 0.80,
    mmseqs_bin: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Execute MMseqs2 easy-cluster if binary is found on system."""
    binary = mmseqs_bin or shutil.which("mmseqs") or shutil.which("mmseqs/bin/mmseqs")
    if not binary or not os.path.exists(binary):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        out_prefix = tmp / "cluster_res"
        cmd = [
            str(binary), "easy-cluster",
            str(fasta_path),
            str(out_prefix),
            str(tmp / "tmp_mmseqs"),
            "--min-seq-id", str(min_seq_id),
            "-c", str(min_cov),
            "--cov-mode", "0",
            "-v", "0",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except Exception as e:
            print(f"MMseqs2 execution failed ({e}), using python fallback.")
            return None

        cluster_tsv = out_prefix.with_name(out_prefix.name + "_cluster.tsv")
        if not cluster_tsv.exists():
            return None

        # Format: <rep_id>\t<member_id>
        clusters_by_rep: Dict[str, List[str]] = {}
        with open(cluster_tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep, member = parts[0], parts[1]
                    clusters_by_rep.setdefault(rep, []).append(member)

        # Build cluster mapping
        id_to_cluster: Dict[str, int] = {}
        cluster_list = []
        for c_idx, (rep, members) in enumerate(clusters_by_rep.items()):
            cluster_list.append({
                "cluster_id": c_idx,
                "representative": rep,
                "size": len(members),
                "members": members,
            })
            for m in members:
                id_to_cluster[m] = c_idx

        return {
            "method": "mmseqs2",
            "min_seq_id": min_seq_id,
            "min_cov": min_cov,
            "num_clusters": len(cluster_list),
            "clusters": cluster_list,
            "id_to_cluster": id_to_cluster,
        }


def cluster_with_python(
    fasta_path: pathlib.Path,
    min_seq_id: float = 0.70,
    min_cov: float = 0.80,
) -> Dict[str, Any]:
    """Pure-Python greedy sequence clustering (CD-HIT style) at specified identity & coverage."""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    aligner = make_pairwise_aligner()

    # Sort descending by length so longer sequences serve as cluster seeds
    ordered_records = sorted(records, key=lambda r: len(r.seq), reverse=True)

    clusters: List[Dict[str, Any]] = []
    id_to_cluster: Dict[str, int] = {}

    for rec in ordered_records:
        seq_str = str(rec.seq)
        placed = False
        for c in clusters:
            rep_seq = c["rep_seq"]
            ident, cov = compute_global_identity(aligner, seq_str, rep_seq)
            if ident >= min_seq_id and cov >= min_cov:
                c["members"].append(rec.id)
                id_to_cluster[rec.id] = c["cluster_id"]
                placed = True
                break

        if not placed:
            new_cid = len(clusters)
            clusters.append({
                "cluster_id": new_cid,
                "representative": rec.id,
                "rep_seq": seq_str,
                "members": [rec.id],
            })
            id_to_cluster[rec.id] = new_cid

    # Clean rep_seq for export
    clean_clusters = []
    for c in clusters:
        clean_clusters.append({
            "cluster_id": c["cluster_id"],
            "representative": c["representative"],
            "size": len(c["members"]),
            "members": c["members"],
        })

    return {
        "method": "python_greedy",
        "min_seq_id": min_seq_id,
        "min_cov": min_cov,
        "num_clusters": len(clean_clusters),
        "clusters": clean_clusters,
        "id_to_cluster": id_to_cluster,
    }


def cluster_pool(
    fasta_path: pathlib.Path,
    min_seq_id: float = 0.70,
    min_cov: float = 0.80,
    cache_path: Optional[pathlib.Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Retrieve or compute 70% clustering for petase_pool.fasta."""
    if cache_path and cache_path.exists() and not force_recompute:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Verify cache matches input sequence count
            records = list(SeqIO.parse(fasta_path, "fasta"))
            if len(data.get("id_to_cluster", {})) == len(records):
                return data

    # Attempt MMseqs2 first, then fall back to Python
    res = cluster_with_mmseqs(fasta_path, min_seq_id=min_seq_id, min_cov=min_cov)
    if res is None:
        print(f"MMseqs2 not found. Running pure-Python 70% sequence clustering on {fasta_path.name}...")
        res = cluster_with_python(fasta_path, min_seq_id=min_seq_id, min_cov=min_cov)
    else:
        print(f"Clustered {fasta_path.name} via MMseqs2.")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved clustering cache to {cache_path}")

    return res


def verify_cluster_balance(exemplar_ids: List[str], id_to_cluster: Dict[str, int]) -> Tuple[bool, Optional[str]]:
    """Assert that no two exemplars in the exemplar list share the same 70% identity cluster."""
    seen_clusters: Dict[int, str] = {}
    for eid in exemplar_ids:
        cid = id_to_cluster.get(eid)
        if cid is None:
            return False, f"Exemplar {eid} missing from cluster mapping"
        if cid in seen_clusters:
            prev_eid = seen_clusters[cid]
            return False, f"Cluster clash: {eid} and {prev_eid} both belong to 70% cluster {cid}"
        seen_clusters[cid] = eid
    return True, None
