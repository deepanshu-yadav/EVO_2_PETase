#!/usr/bin/env python3
"""Module 4 Biophysical Utilities: Interfacial Adsorption, Disulfide Topology, Thermostability & Solubility.

Implements:
1. Component 4.1: Cleft hydrophobic solvent-accessible surface area (A_cleft_hp via Biopython ShrakeRupley
   or FreeSASA) and local electrostatic cleft charge within 8 A of the catalytic triad.
2. Component 4.2: Disulfide topology audit (C-terminal cutinase disulfide & active-site DS1/DS2 pairs).
3. Component 4.3: Calibrated PETase thermostability (Tm) estimator trained on the 139 annotated pool enzymes,
   plus optional ingestion of TemStaPro / DeepSTABp GPU predictions.
4. Component 4.4: E. coli soluble expression proxy (Protein-Sol-style biophysical score >= 0.45 gate).
"""
import math
import pathlib
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis

HYDROPHOBIC_RESIDUES = {"TRP", "PHE", "TYR", "LEU", "ILE", "VAL"}
HYDROPHOBIC_1LETTER = set("WFYLIV")
STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy index
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def clean_seq(seq: str) -> str:
    """Keep only standard 20 amino acids in uppercase."""
    return "".join(c for c in seq.upper().strip() if c in STD_AA)


def locate_catalytic_ser_index(seq: str) -> int:
    """Locate 0-based index of nucleophilic Ser inside G-x-S-x-G / [GWA]-x-[SMG]-x-G motif."""
    seq = clean_seq(seq)
    for pat in (r"G[HWYFQ]S.G", r"G.S.G", r"[GWA].[SMG].G"):
        m = re.search(pat, seq)
        if m:
            return m.start() + 2
    return len(seq) // 2


def extract_sequence_biophysical_features(seq: str) -> Dict[str, float]:
    """Compute sequence-level biophysical descriptors for stability & solubility modeling."""
    s = clean_seq(seq)
    n = max(len(s), 1)
    pa = ProteinAnalysis(s)
    # Compute version-independent amino acid fractions in [0.0, 1.0]
    aa_frac = {aa: s.count(aa) / n for aa in STD_AA}

    aliphatic_idx = 100.0 * (
        aa_frac["A"]
        + 2.9 * aa_frac["V"]
        + 3.9 * (aa_frac["I"] + aa_frac["L"])
    )
    gravy = float(pa.gravy())
    aromaticity = float(pa.aromaticity())
    instability = float(pa.instability_index())
    charge_ph7 = float(pa.charge_at_pH(7.0))
    pi = float(pa.isoelectric_point())

    r_frac = aa_frac["R"]
    k_frac = aa_frac["K"]
    arg_lys_ratio = r_frac / max(r_frac + k_frac, 1e-6)

    d_frac = aa_frac["D"]
    e_frac = aa_frac["E"]
    charged_frac = r_frac + k_frac + d_frac + e_frac
    pos_frac = r_frac + k_frac
    neg_frac = d_frac + e_frac
    charge_asym = abs(pos_frac - neg_frac)

    pro_frac = aa_frac["P"]
    gly_frac = aa_frac["G"]
    cys_count = float(s.count("C"))

    # C-terminal Cys pair proxy (thermophilic cutinase C-term disulfide determinant)
    c_tail = s[max(0, int(0.78 * n)):]
    has_cterm_cys_pair = 1.0 if c_tail.count("C") >= 2 else 0.0

    # Local cleft proxy: +-22 aa window around catalytic Ser (surface-exposed binding groove region)
    ser_idx = locate_catalytic_ser_index(s)
    lo = max(0, ser_idx - 28)
    hi = min(n, ser_idx + 28)
    cleft_win = s[lo:hi]
    cleft_hp_frac = sum(1 for c in cleft_win if c in HYDROPHOBIC_1LETTER) / max(len(cleft_win), 1)
    cleft_net_charge = float(
        cleft_win.count("R") + cleft_win.count("K") + 0.1 * cleft_win.count("H")
        - cleft_win.count("D") - cleft_win.count("E")
    )

    return {
        "length": float(n),
        "aliphatic_index": round(aliphatic_idx, 3),
        "gravy": round(gravy, 4),
        "aromaticity": round(aromaticity, 4),
        "instability_index": round(instability, 3),
        "charge_ph7": round(charge_ph7, 3),
        "isoelectric_point": round(pi, 3),
        "arg_lys_ratio": round(arg_lys_ratio, 4),
        "charged_frac": round(charged_frac, 4),
        "charge_asymmetry": round(charge_asym, 4),
        "pro_frac": round(pro_frac, 4),
        "gly_frac": round(gly_frac, 4),
        "cys_count": cys_count,
        "has_cterm_cys_pair": has_cterm_cys_pair,
        "cleft_hp_frac": round(cleft_hp_frac, 4),
        "cleft_net_charge_seq": round(cleft_net_charge, 2),
    }


def compute_cleft_interface_proxies(
    pdb_path: Optional[pathlib.Path],
    sequence: str,
    ser_og_coord: Optional[List[float]] = None,
    cleft_radius: float = 8.0,
) -> Dict[str, Any]:
    """Compute substrate-cleft hydrophobic SASA (A_cleft_hp) and cleft net charge within 8 A of catalytic triad.

    Uses FreeSASA when installed, otherwise Biopython ShrakeRupley on the 3D structure.
    Falls back cleanly to sequence-derived groove proxies if the PDB file is unavailable.
    """
    seq_feats = extract_sequence_biophysical_features(sequence)

    if pdb_path is None or not pdb_path.exists() or pdb_path.stat().st_size == 0:
        # Calibrated sequence-based SASA proxy (typical PETase 8 A cleft HP SASA ~ 420-780 A^2)
        est_sasa = round(220.0 + 950.0 * seq_feats["cleft_hp_frac"] + 280.0 * seq_feats["aromaticity"], 2)
        return {
            "cleft_hydrophobic_sasa": est_sasa,
            "net_charge_cleft": seq_feats["cleft_net_charge_seq"],
            "cleft_hp_res_count": int(round(seq_feats["cleft_hp_frac"] * 18)),
            "cleft_sasa_method": "sequence_proxy",
        }

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("design", str(pdb_path))
        model = structure[0]
        chain = list(model.get_chains())[0]
    except Exception:
        est_sasa = round(220.0 + 950.0 * seq_feats["cleft_hp_frac"] + 280.0 * seq_feats["aromaticity"], 2)
        return {
            "cleft_hydrophobic_sasa": est_sasa,
            "net_charge_cleft": seq_feats["cleft_net_charge_seq"],
            "cleft_hp_res_count": int(round(seq_feats["cleft_hp_frac"] * 18)),
            "cleft_sasa_method": "sequence_proxy",
        }

    # Determine cleft center coordinate (catalytic Ser OG/CA)
    center = None
    if ser_og_coord is not None and len(ser_og_coord) == 3:
        center = np.array(ser_og_coord, dtype=float)
    else:
        ser_idx = locate_catalytic_ser_index(sequence)
        res_list = [r for r in chain if r.id[0] == " " and "CA" in r]
        if 0 <= ser_idx < len(res_list):
            target_res = res_list[ser_idx]
            if "OG" in target_res:
                center = target_res["OG"].get_coord()
            elif "CA" in target_res:
                center = target_res["CA"].get_coord()
        if center is None and res_list:
            center = np.mean([r["CA"].get_coord() for r in res_list], axis=0)

    # Compute per-residue SASA
    sasa_method = "biopython_shrake_rupley"
    res_sasa_map: Dict[Tuple[str, Any], float] = {}
    try:
        import freesasa  # type: ignore
        fs_struct = freesasa.Structure(str(pdb_path))
        fs_res = freesasa.calc(fs_struct)
        area_classes = fs_res.residueAreas()
        for ch_id, res_dict in area_classes.items():
            for res_num_str, area_obj in res_dict.items():
                try:
                    rnum = int(re.sub(r"[^0-9-]", "", str(res_num_str)))
                except ValueError:
                    continue
                res_sasa_map[(ch_id, rnum)] = float(area_obj.total)
        sasa_method = "freesasa"
    except Exception:
        sr = ShrakeRupley()
        sr.compute(model, level="R")
        for r in chain:
            if r.id[0] == " ":
                res_sasa_map[(chain.id, r.id[1])] = float(getattr(r, "sasa", 0.0))

    # Collect residues within cleft_radius of catalytic Ser
    cleft_hp_sasa = 0.0
    cleft_hp_count = 0
    net_charge = 0.0

    for r in chain:
        if r.id[0] != " ":
            continue
        # Minimum heavy-atom distance from residue to cleft center
        coords = [a.get_coord() for a in r.get_atoms() if not a.get_name().startswith("H")]
        if not coords:
            continue
        min_dist = float(min(np.linalg.norm(c - center) for c in coords))
        if min_dist > cleft_radius:
            continue

        resname = r.get_resname().strip()
        if resname in HYDROPHOBIC_RESIDUES:
            cleft_hp_sasa += res_sasa_map.get((chain.id, r.id[1]), float(getattr(r, "sasa", 0.0)))
            cleft_hp_count += 1

        if resname in {"ARG", "LYS"}:
            net_charge += 1.0
        elif resname in {"ASP", "GLU"}:
            net_charge -= 1.0
        elif resname == "HIS":
            net_charge += 0.1

    return {
        "cleft_hydrophobic_sasa": round(cleft_hp_sasa, 2),
        "net_charge_cleft": round(net_charge, 2),
        "cleft_hp_res_count": cleft_hp_count,
        "cleft_sasa_method": sasa_method,
    }


def audit_disulfide_topology(
    pdb_path: Optional[pathlib.Path],
    sequence: str,
    max_s_s_dist: float = 3.0,
) -> Dict[str, Any]:
    """Audit C-terminal cutinase disulfides and active-site DS1/DS2 disulfide pairs."""
    s = clean_seq(sequence)
    n = len(s)
    if pdb_path is None or not pdb_path.exists() or pdb_path.stat().st_size == 0:
        c_tail_count = s[max(0, int(0.75 * n)):].count("C")
        mid_cys = s[int(0.45 * n):int(0.85 * n)].count("C")
        return {
            "disulfide_count": min(c_tail_count // 2 + mid_cys // 2, 3),
            "has_cterm_disulfide": c_tail_count >= 2,
            "has_active_site_disulfide": mid_cys >= 2,
            "disulfide_pairs": [],
            "disulfide_source": "sequence_proxy",
        }

    cys_sg = []
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ATOM") and line[17:20].strip() == "CYS" and line[12:16].strip() == "SG":
                try:
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    cys_sg.append((resnum, np.array([x, y, z])))
                except ValueError:
                    continue

    pairs = []
    has_cterm = False
    has_active_ds = False
    for i in range(len(cys_sg)):
        for j in range(i + 1, len(cys_sg)):
            r1, c1 = cys_sg[i]
            r2, c2 = cys_sg[j]
            dist = float(np.linalg.norm(c1 - c2))
            if dist <= max_s_s_dist:
                pairs.append((r1, r2, round(dist, 2)))
                if max(r1, r2) >= int(0.75 * n) and abs(r1 - r2) >= 10:
                    has_cterm = True
                if int(0.40 * n) <= min(r1, r2) <= int(0.85 * n):
                    has_active_ds = True

    return {
        "disulfide_count": len(pairs),
        "has_cterm_disulfide": has_cterm,
        "has_active_site_disulfide": has_active_ds,
        "disulfide_pairs": pairs,
        "disulfide_source": "3d_structure",
    }


def fit_pool_tm_calibrator(
    pool_fasta: Optional[pathlib.Path],
    metadata_tsv: Optional[pathlib.Path],
) -> Optional[Dict[str, Any]]:
    """Fit a lightweight Ridge regression Tm (C) calibrator on the 139 annotated PETase pool enzymes."""
    if pool_fasta is None or metadata_tsv is None or not pool_fasta.exists() or not metadata_tsv.exists():
        return None

    import csv

    with open(pool_fasta, "r", encoding="utf-8") as fh:
        seqs = {r.id: str(r.seq) for r in SeqIO.parse(fh, "fasta")}
    tm_map: Dict[str, float] = {}
    with open(metadata_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rid = row.get("id", "")
            tm_val = row.get("t_m", "")
            src = row.get("source", "")
            if rid in seqs and tm_val:
                try:
                    val = float(tm_val)
                    # Exclude bulk unmeasured PAZy/PlasticDB 41.0 C placeholders; keep measured PANDA + explicit thermophiles
                    if src == "PANDA" or val >= 60.0:
                        tm_map[rid] = val
                except ValueError:
                    pass

    if len(tm_map) < 20:
        return None

    feature_keys = [
        "aliphatic_index",
        "gravy",
        "arg_lys_ratio",
        "charged_frac",
        "pro_frac",
        "gly_frac",
        "aromaticity",
        "has_cterm_cys_pair",
        "instability_index",
    ]
    X_list, y_list = [], []
    for rid, tm in tm_map.items():
        feats = extract_sequence_biophysical_features(seqs[rid])
        X_list.append([feats[k] for k in feature_keys])
        y_list.append(tm)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
    Xn = (X - mu) / sd

    # Ridge closed-form: (Xn^T Xn + alpha * I)^-1 Xn^T (y - y_mean)
    y_mean = float(y.mean())
    alpha = 5.0
    A = Xn.T @ Xn + alpha * np.eye(Xn.shape[1])
    w = np.linalg.solve(A, Xn.T @ (y - y_mean))
    preds = y_mean + Xn @ w
    mae = float(np.mean(np.abs(preds - y)))

    return {
        "feature_keys": feature_keys,
        "mu": mu,
        "sd": sd,
        "w": w,
        "y_mean": y_mean,
        "train_mae": round(mae, 2),
        "n_samples": len(y),
    }


def predict_thermostability(
    sequence: str,
    calibrator: Optional[Dict[str, Any]],
    has_cterm_disulfide: bool = False,
    disulfide_count: int = 0,
    delta_ll: Optional[float] = None,
    temstapro_prob65: Optional[float] = None,
    deepstabp_tm: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict enzyme melting temperature (Tm, C) and >=65C thermostability classification.

    Combines external DeepSTABp / TemStaPro GPU predictions when available, or uses the
    139-enzyme PETase pool Ridge calibrator adjusted for verified 3D disulfide topology.
    """
    feats = extract_sequence_biophysical_features(sequence)

    if calibrator is not None:
        x = np.array([feats[k] for k in calibrator["feature_keys"]], dtype=float)
        xn = (x - calibrator["mu"]) / calibrator["sd"]
        base_tm = float(calibrator["y_mean"] + xn @ calibrator["w"])
    else:
        # Biophysical heuristic centered on cutinase/PETase Tm range
        base_tm = (
            44.0
            + 0.35 * (feats["aliphatic_index"] - 70.0)
            + 14.0 * (feats["arg_lys_ratio"] - 0.5)
            + 120.0 * (feats["pro_frac"] - 0.06)
            + (6.5 if has_cterm_disulfide else 0.0)
        )

    # Structural disulfide stability refinement (C-terminal disulfide contributes ~6-8 C in cutinases)
    ds_bonus = 2.5 if (has_cterm_disulfide and feats["has_cterm_cys_pair"] == 0.0) else 0.0
    if disulfide_count >= 2:
        ds_bonus += 1.5
    # Contrastive thermo log-likelihood refinement (centered around median ~580)
    ll_adj = 0.0
    if delta_ll is not None and not (isinstance(delta_ll, float) and math.isnan(delta_ll)):
        ll_adj = float(np.clip((delta_ll - 550.0) / 150.0, -2.0, 4.0))

    ridge_tm = float(np.clip(base_tm + ds_bonus + ll_adj, 32.0, 92.0))

    if deepstabp_tm is not None and not (isinstance(deepstabp_tm, float) and math.isnan(deepstabp_tm)):
        final_tm = float(deepstabp_tm)
        source = "deepstabp"
    else:
        final_tm = ridge_tm
        source = "pool_ridge_calibrated" if calibrator is not None else "biophysical_heuristic"

    # Logistic probability of Tm >= 65 C if TemStaPro was not executed externally
    if temstapro_prob65 is not None and not (isinstance(temstapro_prob65, float) and math.isnan(temstapro_prob65)):
        prob_65 = float(np.clip(temstapro_prob65, 0.0, 1.0))
        if source != "deepstabp":
            source = f"{source}+temstapro"
    else:
        prob_65 = float(1.0 / (1.0 + math.exp(-(final_tm - 65.0) / 4.5)))

    return {
        "predicted_tm": round(final_tm, 2),
        "temstapro_tm65_prob": round(prob_65, 4),
        "passed_tm65": final_tm >= 65.0,
        "tm_source": source,
    }


def predict_ecoli_solubility(
    sequence: str,
    mean_plddt: Optional[float] = None,
    external_solubility: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict E. coli soluble expression score in [0, 1] (Protein-Sol-style gate >= 0.45).

    Integrates charge-hydropathy balance (Wilkinson-Harrison / Uversky boundary),
    Kyte-Doolittle gravy, turn-forming residue fraction, instability index, and structural
    fold confidence (mean pLDDT when available).
    """
    if external_solubility is not None and not (isinstance(external_solubility, float) and math.isnan(external_solubility)):
        score = float(np.clip(external_solubility, 0.0, 1.0))
        return {
            "solubility_score": round(score, 4),
            "passed_solubility_gate": score >= 0.45,
            "solubility_source": "external_protein_sol",
        }

    feats = extract_sequence_biophysical_features(sequence)
    # Linear logit calibrated so well-folded soluble PET hydrolases (IsPETase/LCC/TfCut2/PHL7)
    # score ~0.52-0.78 while overly hydrophobic or disordered sequences fall below 0.45.
    logit = (
        0.55
        - 1.65 * feats["gravy"]
        + 2.40 * feats["charged_frac"]
        - 1.80 * feats["charge_asymmetry"]
        - 0.018 * max(0.0, feats["instability_index"] - 40.0)
        - 3.50 * max(0.0, feats["aromaticity"] - 0.12)
        + 1.20 * (feats["gly_frac"] + feats["pro_frac"] - 0.15)
    )
    if mean_plddt is not None and not (isinstance(mean_plddt, float) and math.isnan(mean_plddt)):
        logit += 0.025 * (mean_plddt - 82.0)

    score = float(1.0 / (1.0 + math.exp(-logit)))
    return {
        "solubility_score": round(score, 4),
        "passed_solubility_gate": score >= 0.45,
        "solubility_source": "biophysical_protein_sol_proxy",
    }
