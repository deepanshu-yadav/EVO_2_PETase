#!/usr/bin/env python3
"""Module 5 Utilities: Two-Stage Decoupled Triage, Z-Score Composite Ranking & 48-Target Panel Assembly.

Implements:
1. Stage 1 (Hard Binary Plausibility Gate):
     (pLDDT >= 80.0) & (SeqTriad_Ser_Asp_His == True) & (241 <= L <= 306) & (Solubility >= 0.45)
2. Stage 2 (Continuous Z-Score Composite Ranking across Gate-Passers):
     Fitness = 0.25*Z(pLDDT) + 0.20*Z(TM_ref) + 0.20*Z(-dG_bind) + 0.15*Z(NAC) + 0.10*Z(Tm_hat) + 0.10*Z(A_cleft_hp)
3. Controlled 48-Target Wet-Lab Panel (Panels A-F):
     - Panel A (12): Thermostable Structural Leads (Tm >= 70 C, intact triad, d_attack <= 3.5 A priority)
     - Panel B (12): Deep Sequence Novelty (farthest-point clustering, 50%-65% identity priority)
     - Panel C (8):  IsPETase Groove Rescues (Trp159 groove + thermophilic core packing)
     - Panel D (6):  Random Gate-Passers (in silico null control)
     - Panel E (4):  Catalytic Ser -> Ala Knockouts of top 4 Panel A leads
     - Panel F (6):  Natural & Engineered Reference Standards (LCC, ICCG, PHL7, TfCut2, IsPETase, Fast-PETase)
"""
import pathlib
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices

# Fallback canonical mature benchmark sequences (matching data/petase_pool.fasta)
DEFAULT_BENCHMARKS = {
    "001_IsPETase": (
        "MQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPG"
        "YTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIY"
        "GKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACEND"
        "SIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDT"
        "RYSTFACENPNSTRVSDFRTANCS"
    ),
    "002_TfCut2": (
        "MANPYERGPNPTDALLEASSGPFSVSEENVSRLSASGFGGGTIYYPRENNTYGAVAISPG"
        "YTGTEASIAWLGERIASHGFVVITIDTITTLDQPDSRAEQLNAALNHMINRASSTVRSRI"
        "DSSRLAVMGHSMGGGGTLRLASQRPDLKAAIPLTPWHLNKNWSSVTVPTLIIGADLDTIA"
        "PVATHAKPFYNSLPSSISKAYLELDGATHFAPNIPNKIIGKYSVAWLKRFVDNDTRYTQF"
        "LCPGPRDGLFGEVEEYRSTCPF"
    ),
    "003_LCC": (
        "MQSNPYQRGPNPTRSALTADGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSP"
        "GYTADASSLAWLGRRLASHGFVVLVINTNSRFDYPDSRASQLSAALNYLRTSSPSAVRAR"
        "LDANRLAVAGHSMGGGGTLRIAEQNPSLKAAVPLTPWHTDKTFNTSVPVLIVGAEADTVA"
        "PVSQHAIPFYQNLPSTTPKVYVELDNASHFAPNSNNAAISVYTISWMKLWVDNDTRYRQF"
        "LCNVNDPALSDFRTNNRHCQ"
    ),
    "004_PHL7": (
        "MMANPYERGPDPTESSIEAVRGPFAVAQTTVSRLQADGFGGGTIYYPTDTSQGTFGAVAI"
        "SPGFTAGQESIAWLGPRIASQGFVVITIDTITRLDQPDSRGRQLQAALDHLRTNSVVRNR"
        "IDPNRMAVMGHSMGGGGALSAAANNTSLEAAIPLQGWHTRKNWSSVRTPTLVVGAQLDTI"
        "APVSSHSEAFYNSLPSDLDKAYMELRGASHLVSNTPDTTTAKYSIAWLKRFVDDDLRYEQ"
        "FLCPAPDDFAISEYRSTCPF"
    ),
}


def make_global_aligner() -> Align.PairwiseAligner:
    """Create BLOSUM62 global aligner matching pool clustering conventions."""
    al = Align.PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.mode = "global"
    al.open_gap_score = -10.0
    al.extend_gap_score = -0.5
    al.end_gap_score = 0.0
    return al


def compute_seq_identity(seq_a: str, seq_b: str, aligner: Align.PairwiseAligner) -> float:
    """Global pairwise identity normalized by shorter sequence length."""
    if not seq_a or not seq_b:
        return 0.0
    aln = aligner.align(seq_a, seq_b)[0]
    matches = sum(
        sum(1 for k in range(ea - sa) if seq_a[sa + k] == seq_b[sb + k])
        for (sa, ea), (sb, eb) in zip(*aln.aligned)
    )
    return matches / max(min(len(seq_a), len(seq_b)), 1)


def has_sequence_triad(seq: str) -> bool:
    """Verify nucleophilic elbow motif (G-x-S-x-G / [GWA]-x-[SMG]-x-G) plus Asp and His."""
    s = seq.upper().replace("*", "").strip()
    motif = re.search(r"G[HWYFQ]S.G", s) or re.search(r"[GWA].[SMG].G", s) or re.search(r"G.S.G", s)
    return (motif is not None) and ("D" in s) and ("H" in s)


def locate_catalytic_ser_via_ispetase(seq: str, ref_ispetase: str, aligner: Align.PairwiseAligner) -> int:
    """Find 0-based index of the catalytic Ser equivalent to IsPETase Ser160 (pool idx 133)."""
    s = seq.upper().replace("*", "").strip()
    ref_ser_0 = 133  # 0-based index of Ser160 in 001_IsPETase (GWSMG)
    try:
        aln = aligner.align(ref_ispetase, s)[0]
        for (rs, re_), (qs, qe) in zip(*aln.aligned):
            if rs <= ref_ser_0 < re_:
                cand_idx = qs + (ref_ser_0 - rs)
                if 0 <= cand_idx < len(s) and s[cand_idx] == "S":
                    return cand_idx
    except Exception:
        pass

    for pat in (r"G[HWYFQ]S.G", r"G.S.G", r"[GWA].[SMG].G"):
        m = re.search(pat, s)
        if m and s[m.start() + 2] == "S":
            return m.start() + 2
    raise ValueError("Could not locate catalytic Ser residue for knockout mutagenesis.")


def make_ser_to_ala_knockout(seq: str, ref_ispetase: str, aligner: Align.PairwiseAligner) -> Tuple[str, int]:
    """Mutate exact catalytic Ser -> Ala and return (mutated_seq, 1_based_position)."""
    idx = locate_catalytic_ser_via_ispetase(seq, ref_ispetase, aligner)
    chars = list(seq.upper().replace("*", "").strip())
    assert chars[idx] == "S", f"Expected Ser at 0-based index {idx}, found {chars[idx]}"
    chars[idx] = "A"
    return "".join(chars), idx + 1


def build_engineered_benchmarks(pool_fasta: Optional[pathlib.Path] = None) -> List[Dict[str, Any]]:
    """Build the 6 Panel F natural and engineered reference standards."""
    seqs = dict(DEFAULT_BENCHMARKS)
    if pool_fasta and pool_fasta.exists():
        with open(pool_fasta, "r", encoding="utf-8") as fh:
            for rec in SeqIO.parse(fh, "fasta"):
                if rec.id in seqs:
                    seqs[rec.id] = str(rec.seq).upper().replace("*", "")

    # LCC-ICCG: Y127G / D238C / F243I / S283C (full-length LCC offset = -33 -> 0-based 93, 204, 209, 249)
    lcc_chars = list(seqs["003_LCC"])
    iccg_muts = [(93, "Y", "G"), (204, "D", "C"), (209, "F", "I"), (249, "S", "C")]
    for idx0, expected, new_aa in iccg_muts:
        if lcc_chars[idx0] != expected:
            raise ValueError(f"LCC-ICCG residue mismatch at 0-based {idx0}: expected {expected}, got {lcc_chars[idx0]}")
        lcc_chars[idx0] = new_aa
    lcc_iccg_seq = "".join(lcc_chars)

    # Fast-PETase: S121E / D186H / R224Q / N233K / R280A (full-length IsPETase offset = -26 -> 0-based 94, 159, 197, 206, 253)
    isp_chars = list(seqs["001_IsPETase"])
    fast_muts = [(94, "S", "E"), (159, "D", "H"), (197, "R", "Q"), (206, "N", "K"), (253, "R", "A")]
    for idx0, expected, new_aa in fast_muts:
        if isp_chars[idx0] != expected:
            raise ValueError(f"Fast-PETase residue mismatch at 0-based {idx0}: expected {expected}, got {isp_chars[idx0]}")
        isp_chars[idx0] = new_aa
    fast_petase_seq = "".join(isp_chars)

    return [
        {"id": "003_LCC", "label": "LCC (WT)", "sequence": seqs["003_LCC"], "tm_exp": 71.0},
        {"id": "003_LCC_ICCG", "label": "LCC-ICCG (Y127G/D238C/F243I/S283C)", "sequence": lcc_iccg_seq, "tm_exp": 94.0},
        {"id": "004_PHL7", "label": "PHL7 (WT)", "sequence": seqs["004_PHL7"], "tm_exp": 79.0},
        {"id": "002_TfCut2", "label": "TfCut2 (WT)", "sequence": seqs["002_TfCut2"], "tm_exp": 71.0},
        {"id": "001_IsPETase", "label": "IsPETase (WT)", "sequence": seqs["001_IsPETase"], "tm_exp": 48.0},
        {"id": "001_IsPETase_FastPETase", "label": "Fast-PETase (S121E/D186H/R224Q/N233K/R280A)", "sequence": fast_petase_seq, "tm_exp": 56.0},
    ]


def parse_foldseek_m8(m8_path: Optional[pathlib.Path]) -> pd.DataFrame:
    """Parse Foldseek tabular alignment output vs 6EQE and 4EB0 into per-design tm_6eqe, tm_4eb0, tm_ref."""
    if m8_path is None or not m8_path.exists() or m8_path.stat().st_size == 0:
        return pd.DataFrame(columns=["id", "tm_6eqe", "tm_4eb0", "tm_ref", "rmsd_6eqe"])

    fields = [
        "query", "target", "fident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits", "alntmscore", "rmsd", "lddt"
    ]
    df_fs = pd.read_csv(m8_path, sep="\t", names=fields)
    df_fs["id"] = df_fs["query"].astype(str).str.replace(".pdb", "", regex=False).str.replace("_relaxed", "", regex=False)
    df_fs["target_clean"] = df_fs["target"].astype(str).str.replace(".pdb", "", regex=False)

    f6 = (
        df_fs[df_fs["target_clean"].str.contains("6EQE", na=False)]
        .sort_values("alntmscore", ascending=False)
        .drop_duplicates("id")
        [["id", "alntmscore", "rmsd"]]
        .rename(columns={"alntmscore": "tm_6eqe", "rmsd": "rmsd_6eqe"})
    )
    f4 = (
        df_fs[df_fs["target_clean"].str.contains("4EB0", na=False)]
        .sort_values("alntmscore", ascending=False)
        .drop_duplicates("id")
        [["id", "alntmscore"]]
        .rename(columns={"alntmscore": "tm_4eb0"})
    )
    merged = pd.merge(f6, f4, on="id", how="outer")
    merged["tm_ref"] = merged[["tm_6eqe", "tm_4eb0"]].max(axis=1)
    return merged


def standardize_z(series: pd.Series) -> pd.Series:
    """Compute sample Z-scores across passing cohort; returns 0.0 when variance is zero."""
    vals = pd.to_numeric(series, errors="coerce")
    mu = vals.mean()
    sd = vals.std(ddof=0)
    if pd.isna(sd) or sd < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (vals - mu) / sd


def apply_stage1_hard_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: Hard binary plausibility elimination before continuous Z-score ranking.

    Gate = (pLDDT >= 80.0) & (SeqTriad == True) & (241 <= L <= 306) & (Solubility >= 0.45)
    """
    out = df.copy()
    out["gate_plddt80"] = pd.to_numeric(out["mean_plddt"], errors="coerce").fillna(0.0) >= 80.0

    if "seq_triad_intact" in out.columns:
        seq_triad = out["seq_triad_intact"].fillna(out["sequence"].map(has_sequence_triad)).astype(bool)
    else:
        seq_triad = out["sequence"].map(has_sequence_triad).astype(bool)
    out["gate_seq_triad"] = seq_triad

    lengths = pd.to_numeric(out["length"], errors="coerce").fillna(out["sequence"].str.len())
    out["length"] = lengths.astype(int)
    out["gate_length"] = out["length"].between(241, 306)

    out["gate_solubility"] = pd.to_numeric(out["solubility_score"], errors="coerce").fillna(0.0) >= 0.45
    out["passed_stage1_gate"] = (
        out["gate_plddt80"] & out["gate_seq_triad"] & out["gate_length"] & out["gate_solubility"]
    )

    reasons = []
    for _, r in out.iterrows():
        fails = []
        if not r["gate_plddt80"]:
            fails.append("plddt<80")
        if not r["gate_seq_triad"]:
            fails.append("seq_triad_missing")
        if not r["gate_length"]:
            fails.append("length_out_of_241_306")
        if not r["gate_solubility"]:
            fails.append("solubility<0.45")
        reasons.append(";".join(fails) if fails else "PASSED")
    out["stage1_drop_reasons"] = reasons
    return out


def compute_zscore_fitness(df_gated: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: Continuous Z-score composite ranking strictly across Stage 1 gate-passers.

    Fitness = 0.25*Z(pLDDT) + 0.20*Z(TM_ref) + 0.20*Z(-dG_bind) + 0.15*Z(NAC)
              + 0.10*Z(Tm_hat) + 0.10*Z(A_cleft_hp)
    """
    out = df_gated.copy()
    pass_mask = out["passed_stage1_gate"].astype(bool)

    # Derive unified docking metrics from 2-HE(MHET) & 3PET
    aff_2he = pd.to_numeric(out.get("2he_mhet_affinity", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    aff_3pet = pd.to_numeric(out.get("3pet_affinity", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    aff_combined = np.where(
        (aff_2he < 0.0) & (aff_3pet < 0.0),
        0.5 * aff_2he + 0.5 * aff_3pet,
        np.where(aff_2he < 0.0, aff_2he, aff_3pet),
    )
    out["combined_affinity"] = np.round(aff_combined, 3)
    out["neg_dg_bind"] = -out["combined_affinity"]

    nac_2he = pd.to_numeric(out.get("2he_mhet_nac_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    nac_3pet = pd.to_numeric(out.get("3pet_nac_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    out["combined_nac_score"] = np.round(0.5 * nac_2he + 0.5 * nac_3pet, 4)

    dist_2he = pd.to_numeric(out.get("2he_mhet_attack_dist", pd.Series(99.0, index=out.index)), errors="coerce").fillna(99.0)
    dist_3pet = pd.to_numeric(out.get("3pet_attack_dist", pd.Series(99.0, index=out.index)), errors="coerce").fillna(99.0)
    out["min_attack_dist"] = np.minimum(dist_2he, dist_3pet)
    out["any_strict_nac"] = (
        out.get("2he_mhet_is_nac", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        | out.get("3pet_is_nac", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )

    if "tm_ref" not in out.columns:
        out["tm_ref"] = 0.85
    out["tm_ref"] = pd.to_numeric(out["tm_ref"], errors="coerce").fillna(out["tm_ref"].median() if out["tm_ref"].notna().any() else 0.85)

    for z_col in ["z_plddt", "z_tm_ref", "z_neg_dg", "z_nac", "z_tm_hat", "z_cleft_hp", "fitness_score"]:
        out[z_col] = np.nan

    if pass_mask.sum() > 0:
        sub = out.loc[pass_mask]
        z_plddt = standardize_z(sub["mean_plddt"])
        z_tm_ref = standardize_z(sub["tm_ref"])
        z_neg_dg = standardize_z(sub["neg_dg_bind"])
        z_nac = standardize_z(sub["combined_nac_score"])
        z_tm_hat = standardize_z(sub["predicted_tm"])
        z_cleft_hp = standardize_z(sub["cleft_hydrophobic_sasa"])

        fitness = (
            0.25 * z_plddt
            + 0.20 * z_tm_ref
            + 0.20 * z_neg_dg
            + 0.15 * z_nac
            + 0.10 * z_tm_hat
            + 0.10 * z_cleft_hp
        )

        out.loc[pass_mask, "z_plddt"] = np.round(z_plddt, 4)
        out.loc[pass_mask, "z_tm_ref"] = np.round(z_tm_ref, 4)
        out.loc[pass_mask, "z_neg_dg"] = np.round(z_neg_dg, 4)
        out.loc[pass_mask, "z_nac"] = np.round(z_nac, 4)
        out.loc[pass_mask, "z_tm_hat"] = np.round(z_tm_hat, 4)
        out.loc[pass_mask, "z_cleft_hp"] = np.round(z_cleft_hp, 4)
        out.loc[pass_mask, "fitness_score"] = np.round(fitness, 4)

    out = out.sort_values(by=["passed_stage1_gate", "fitness_score", "mean_plddt"], ascending=[False, False, False]).reset_index(drop=True)
    out["rank"] = np.where(out["passed_stage1_gate"], np.arange(1, len(out) + 1), np.nan)
    return out


def _to_panel_entry(panel_name: str, row: pd.Series, note: str = "") -> Dict[str, Any]:
    return {
        "panel": panel_name,
        "id": row["id"],
        "sequence": row["sequence"],
        "length": int(row["length"]),
        "mean_plddt": row.get("mean_plddt", np.nan),
        "tm_ref": row.get("tm_ref", np.nan),
        "max_pid": row.get("max_pid", np.nan),
        "delta_ll": row.get("delta_ll", np.nan),
        "triad_intact": row.get("triad_intact", np.nan),
        "has_groove_w159": row.get("has_groove_w159", np.nan),
        "combined_affinity": row.get("combined_affinity", np.nan),
        "min_attack_dist": row.get("min_attack_dist", np.nan),
        "combined_nac_score": row.get("combined_nac_score", np.nan),
        "predicted_tm": row.get("predicted_tm", np.nan),
        "solubility_score": row.get("solubility_score", np.nan),
        "cleft_hydrophobic_sasa": row.get("cleft_hydrophobic_sasa", np.nan),
        "has_cterm_disulfide": row.get("has_cterm_disulfide", np.nan),
        "fitness_score": row.get("fitness_score", np.nan),
        "selection_note": note,
    }


def assemble_wetlab_panel_48(
    df_ranked: pd.DataFrame,
    pool_fasta: Optional[pathlib.Path] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Assemble the controlled 48-target wet-lab panel across Arms A-F (12+12+8+6+4+6 = 48)."""
    passers = df_ranked[df_ranked["passed_stage1_gate"]].copy()
    if len(passers) < 38:
        raise ValueError(f"Need at least 38 Stage-1 gate-passers to populate Arms A-D; found {len(passers)}.")

    # Ensure required flags exist with safe defaults
    if "triad_intact" not in passers.columns:
        passers["triad_intact"] = True
    passers["triad_intact"] = passers["triad_intact"].fillna(False).astype(bool)
    if "has_groove_w159" not in passers.columns:
        passers["has_groove_w159"] = False
    passers["has_groove_w159"] = passers["has_groove_w159"].fillna(False).astype(bool)
    if "max_pid" not in passers.columns:
        passers["max_pid"] = 70.0
    if "delta_ll" not in passers.columns:
        passers["delta_ll"] = 0.0

    aligner = make_global_aligner()
    benchmarks = build_engineered_benchmarks(pool_fasta)
    ref_ispetase = next(b["sequence"] for b in benchmarks if b["id"] == "001_IsPETase")

    taken_ids = set()
    panel_rows: List[Dict[str, Any]] = []

    # --- Panel A (12): Thermostable Structural Leads ---
    # Priority: Tm >= 70 C & triad_intact & d_attack <= 3.5 A; then relax d_attack, then Tm >= 65 C
    pa_tiers = [
        (passers["triad_intact"] & (passers["predicted_tm"] >= 70.0) & (passers["min_attack_dist"] <= 3.5), "Tm>=70C + triad + d<=3.5A"),
        (passers["triad_intact"] & (passers["predicted_tm"] >= 70.0), "Tm>=70C + triad (d_attack relaxed)"),
        (passers["triad_intact"] & (passers["predicted_tm"] >= 65.0), "Tm>=65C + triad fallback"),
        (pd.Series(True, index=passers.index), "Top fitness fallback"),
    ]
    panel_a_rows = []
    for cond, note in pa_tiers:
        if len(panel_a_rows) >= 12:
            break
        sub = passers[cond & (~passers["id"].isin(taken_ids))].sort_values("fitness_score", ascending=False)
        for _, r in sub.iterrows():
            if len(panel_a_rows) >= 12:
                break
            taken_ids.add(r["id"])
            entry = _to_panel_entry("A_thermostable_lead", r, note=note)
            panel_a_rows.append(entry)
    panel_rows.extend(panel_a_rows)

    # --- Panel B (12): Deep Sequence Novelty (Farthest-Point Clustering, 50%-65% PID priority) ---
    rem_b = passers[~passers["id"].isin(taken_ids)].copy()
    strict_novel = rem_b[rem_b["triad_intact"] & rem_b["max_pid"].between(50.0, 65.0)].sort_values("max_pid")
    backup_novel = rem_b[~rem_b["id"].isin(set(strict_novel["id"]))].sort_values(["triad_intact", "max_pid"], ascending=[False, True])
    pool_b = pd.concat([strict_novel, backup_novel], ignore_index=True)

    # Farthest-point sampling seeded by the lowest-PID viable candidate
    cand_b = [r for _, r in pool_b.head(max(36, 12)).iterrows()]
    chosen_b: List[pd.Series] = []
    if cand_b:
        chosen_b.append(cand_b[0])
        taken_ids.add(cand_b[0]["id"])
        while len(chosen_b) < 12:
            best_r = None
            best_min_dist = -1.0
            for c in cand_b:
                if c["id"] in taken_ids:
                    continue
                min_dist = min(1.0 - compute_seq_identity(c["sequence"], x["sequence"], aligner) for x in chosen_b)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_r = c
            if best_r is None:
                break
            chosen_b.append(best_r)
            taken_ids.add(best_r["id"])

    for r in chosen_b:
        in_band = 50.0 <= float(r["max_pid"]) <= 65.0
        note = "FPS novelty (50-65% PID)" if in_band else f"FPS novelty (closest available PID={r['max_pid']:.1f}%)"
        panel_rows.append(_to_panel_entry("B_deep_novelty", r, note=note))

    # --- Panel C (8): IsPETase Groove Rescues (Trp159 groove + thermophilic core packing) ---
    rem_c = passers[~passers["id"].isin(taken_ids)].copy()
    pc_tiers = [
        (rem_c["has_groove_w159"] & rem_c["triad_intact"] & (rem_c["predicted_tm"] >= 65.0), "Trp159 groove + Tm>=65C"),
        (rem_c["has_groove_w159"] & rem_c["triad_intact"], "Trp159 groove + triad"),
        (rem_c["has_groove_w159"], "Trp159 groove"),
        (rem_c["triad_intact"], "Thermophilic core packing fallback"),
    ]
    panel_c_count = 0
    for cond, note in pc_tiers:
        if panel_c_count >= 8:
            break
        sub = rem_c[cond & (~rem_c["id"].isin(taken_ids))].sort_values(["fitness_score", "delta_ll"], ascending=[False, False])
        for _, r in sub.iterrows():
            if panel_c_count >= 8:
                break
            taken_ids.add(r["id"])
            panel_rows.append(_to_panel_entry("C_ispetase_groove_rescue", r, note=note))
            panel_c_count += 1

    # --- Panel D (6): Random Gate-Passers (In Silico Null Control) ---
    rem_d = passers[~passers["id"].isin(taken_ids)]
    panel_d = rem_d.sample(n=6, random_state=seed)
    for _, r in panel_d.iterrows():
        taken_ids.add(r["id"])
        panel_rows.append(_to_panel_entry("D_random_control", r, note=f"Uniform random gate-passer (seed={seed})"))

    # --- Panel E (4): Catalytic Ser -> Ala Knockouts of Top 4 Panel A Leads ---
    for entry_a in panel_a_rows[:4]:
        ko_seq, ser_pos1 = make_ser_to_ala_knockout(entry_a["sequence"], ref_ispetase, aligner)
        panel_rows.append({
            "panel": "E_catalytic_ko_S_to_A",
            "id": f"{entry_a['id']}_KO_S{ser_pos1}A",
            "sequence": ko_seq,
            "length": len(ko_seq),
            "mean_plddt": entry_a["mean_plddt"],
            "tm_ref": np.nan,
            "max_pid": np.nan,
            "delta_ll": np.nan,
            "triad_intact": False,
            "has_groove_w159": entry_a["has_groove_w159"],
            "combined_affinity": np.nan,
            "min_attack_dist": np.nan,
            "combined_nac_score": np.nan,
            "predicted_tm": entry_a["predicted_tm"],
            "solubility_score": entry_a["solubility_score"],
            "cleft_hydrophobic_sasa": entry_a["cleft_hydrophobic_sasa"],
            "has_cterm_disulfide": entry_a["has_cterm_disulfide"],
            "fitness_score": np.nan,
            "selection_note": f"Catalytic knockout of {entry_a['id']} at S{ser_pos1}A",
        })

    # --- Panel F (6): Natural & Engineered Reference Standards ---
    for bm in benchmarks:
        panel_rows.append({
            "panel": "F_natural_standard",
            "id": bm["id"],
            "sequence": bm["sequence"],
            "length": len(bm["sequence"]),
            "mean_plddt": 100.0,
            "tm_ref": 1.0,
            "max_pid": 100.0,
            "delta_ll": np.nan,
            "triad_intact": True,
            "has_groove_w159": True,
            "combined_affinity": np.nan,
            "min_attack_dist": np.nan,
            "combined_nac_score": np.nan,
            "predicted_tm": bm["tm_exp"],
            "solubility_score": 1.0,
            "cleft_hydrophobic_sasa": np.nan,
            "has_cterm_disulfide": bm["id"] != "001_IsPETase",
            "fitness_score": np.nan,
            "selection_note": bm["label"],
        })

    return pd.DataFrame(panel_rows)
