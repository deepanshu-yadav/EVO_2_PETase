#!/usr/bin/env python3
"""Decoding and post-processing utilities for Module 2.

Handles:
1. CDS extraction from continuations using the GGGGGGGG spacer.
2. In-frame translation and stop-codon validation.
3. Coherence and length-window filtering (240 - 306 aa).
4. Alignment against pool to calculate maximum percent identity (max PID).
5. Memorization cliff detection.
6. Contrastive log-likelihood (delta-LL) scoring.
"""
import csv
import json
import pathlib
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq

SPACER = "GGGGGGGG"
MIN_LEN_AA = 241
MAX_LEN_AA = 306
NOVELTY_MAX_ID_THRESHOLD = 85.0
MEMORIZATION_CLIFF_PID_THRESHOLD = 80.0


def extract_cds(continuation: str, spacer: str = SPACER) -> Optional[str]:
    """Extract CDS before the first closing spacer delimiter."""
    clean = continuation.upper().strip()
    if spacer in clean:
        return clean.split(spacer)[0]
    return None


def validate_and_translate_cds(
    cds_nt: str,
    min_len: int = MIN_LEN_AA,
    max_len: int = MAX_LEN_AA,
) -> Dict[str, Any]:
    """Translate CDS and evaluate frame, stop codons, and length plausibility."""
    if not cds_nt or len(cds_nt) < 3:
        return {"valid": False, "reason": "empty_or_short", "aa": "", "mature_aa": "", "length": 0}

    # Frame check
    remainder = len(cds_nt) % 3
    if remainder != 0:
        # Partial codon at terminus
        cds_in_frame = cds_nt[:-remainder]
    else:
        cds_in_frame = cds_nt

    try:
        aa = str(Seq(cds_in_frame).translate())
    except Exception as e:
        return {"valid": False, "reason": f"translation_error_{e}", "aa": "", "mature_aa": "", "length": 0}

    has_terminal_stop = aa.endswith("*")
    mature_aa = aa.rstrip("*")
    internal_stop = "*" in mature_aa
    length_aa = len(mature_aa)

    in_window = (min_len <= length_aa <= max_len)
    valid = in_window and not internal_stop and len(mature_aa) > 0

    reason = "valid"
    if internal_stop:
        reason = "premature_internal_stop"
    elif not in_window:
        reason = f"length_out_of_window_{length_aa}"

    return {
        "valid": valid,
        "reason": reason,
        "aa_raw": aa,
        "mature_aa": mature_aa,
        "length_aa": length_aa,
        "has_terminal_stop": has_terminal_stop,
        "has_internal_stop": internal_stop,
    }


def compute_pool_max_pid(
    query_aa: str,
    pool_records: List[Tuple[str, str]],
    aligner: Optional[Align.PairwiseAligner] = None,
) -> Tuple[float, str]:
    """Compute maximum percent identity (max PID) of query against pool."""
    if not query_aa or not pool_records:
        return 0.0, ""

    if aligner is None:
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.mode = "global"
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5

    best_pid = 0.0
    best_hit = ""

    for pid, pseq in pool_records:
        aln = aligner.align(query_aa, pseq)[0]
        q, t = aln.aligned
        matches = sum(
            x == y
            for (a0, a1), (b0, b1) in zip(q, t)
            for x, y in zip(query_aa[a0:a1], pseq[b0:b1])
        )
        ident = (matches / min(len(query_aa), len(pseq))) * 100.0
        if ident > best_pid:
            best_pid = ident
            best_hit = pid

    return best_pid, best_hit


def evaluate_memorization_cliff(
    results_by_depth: Dict[int, List[Dict[str, Any]]],
    novelty_threshold: float = NOVELTY_MAX_ID_THRESHOLD,
    cliff_pid_threshold: float = MEMORIZATION_CLIFF_PID_THRESHOLD,
) -> Dict[str, Any]:
    """Detect whether sequence memorization spikes as a function of context depth n.

    Memorization cliff condition (from modular plan):
      mean(max_PID) > 80.0% AND novel yield drops by > 25% from its peak.
    """
    cliff_analysis = {}
    peak_novel_yield = 0.0
    cliff_detected_depth = None

    sorted_depths = sorted(results_by_depth.keys())
    metrics_per_depth = []

    for n in sorted_depths:
        items = results_by_depth[n]
        valid_items = [x for x in items if x.get("valid_cds", False)]
        if not valid_items:
            continue

        pids = [x["max_pid"] for x in valid_items if "max_pid" in x]
        mean_pid = float(np.mean(pids)) if pids else 0.0
        novel_count = sum(1 for p in pids if p < novelty_threshold)
        novel_yield = (novel_count / len(valid_items)) * 100.0 if valid_items else 0.0

        if novel_yield > peak_novel_yield:
            peak_novel_yield = novel_yield

        drop_from_peak = peak_novel_yield - novel_yield
        is_cliff = (mean_pid >= cliff_pid_threshold and drop_from_peak >= 25.0)
        if is_cliff and cliff_detected_depth is None:
            cliff_detected_depth = n

        metrics_per_depth.append({
            "n_examples": n,
            "total_samples": len(items),
            "valid_samples": len(valid_items),
            "mean_max_pid": round(mean_pid, 2),
            "novel_count": novel_count,
            "novel_yield_pct": round(novel_yield, 2),
            "drop_from_peak_pct": round(drop_from_peak, 2),
            "is_cliff": is_cliff,
        })

    return {
        "cliff_detected": cliff_detected_depth is not None,
        "cliff_depth": cliff_detected_depth,
        "peak_novel_yield_pct": round(peak_novel_yield, 2),
        "metrics_per_depth": metrics_per_depth,
    }


def compute_contrastive_delta_ll(
    logp_thermo: float,
    logp_general: float,
) -> float:
    """Compute Delta-LL = log P(x | Prompt_thermo) - log P(x | Prompt_general)."""
    return float(logp_thermo - logp_general)
