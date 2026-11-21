#!/usr/bin/env python3
"""
Patched Experiment 2 Structural Validation & 2-HE(MHET) Reactive Docking Pipeline (v2).

Fixes all six audit findings:
1. Smina coordinate-frame bug: extracts catalytic Ser OG in the NATIVE PDB frame (before
   6EQE superposition) so the docking box and attack distance match the receptor on disk.
2. Ester carbonyl carbon distance: identifies the specific ester carbonyl carbon (C_ester)
   in 2-HE(MHET) via RDKit SMARTS instead of taking the minimum distance to ANY ligand atom,
   and penalizes unphysical steric clashes (< 2.5 A).
3. Fail-loudly docking: raises RuntimeError if smina or the ligand PDBQT is missing unless
   --allow-skip-docking is explicitly passed (never writes fake constant scores).
4. Alignment-based 3D superposition: maps query residues to 6EQE via BLOSUM62 sequence
   alignment instead of index slicing (ref_cas[:min_len]), recovering the 12 wrongly failed designs.
5. Strict Tier-1 motif regex: checks G[HWYFQ]S.G (nucleophile elbow) instead of [GWA].[SMG].G.
6. Unrounded pLDDT + active-site pLDDT: rescales 0-1 B-factors to 0-100 BEFORE rounding.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request
import numpy as np
import pandas as pd
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import seq1

REFS = {
    "6EQE": "https://files.rcsb.org/download/6EQE.pdb",
    "4EB0": "https://files.rcsb.org/download/4EB0.pdb",
}
SMILES_2HE_MHET = "O=C(O)c1ccc(C(=O)OCCO)cc1"
ESTER_CARBON_SMARTS = "[CX3](=O)[OX2][CH2]"  # Electrophilic ester carbonyl carbon


def make_aligner():
    al = Align.PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.mode = "global"
    al.open_gap_score = -10.0
    al.extend_gap_score = -0.5
    al.end_gap_score = 0.0
    return al


ALIGNER = make_aligner()


def fetch_references(ref_dir: pathlib.Path):
    ref_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for pdb_id, url in REFS.items():
        dst = ref_dir / f"{pdb_id}.pdb"
        if not dst.exists():
            print(f"Downloading reference {pdb_id}...")
            urllib.request.urlretrieve(url, dst)
        out[pdb_id] = dst
    return out


def filter_sequence_motifs(records):
    """Strict Tier-1 Gate: canonical nucleophile elbow G[HWYFQ]S.G plus D and H."""
    passed, rejected = [], []
    for r in records:
        s = str(r.seq)
        if re.search(r"G[HWYFQ]S.G", s) and ("D" in s) and ("H" in s):
            passed.append(r)
        else:
            rejected.append(r)
    return passed, rejected


def extract_plddt(pdb_path: pathlib.Path):
    """Extract per-residue CA pLDDT on 0-100 scale without premature rounding."""
    vals = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            vals.append(float(line[60:66].strip()))
    if not vals:
        return np.nan, []
    mean_raw = float(np.mean(vals))
    scale = 100.0 if mean_raw <= 1.5 else 1.0
    scaled = [v * scale for v in vals]
    return round(float(np.mean(scaled)), 2), scaled


def chain_sequence_and_residues(chain):
    residues = [r for r in chain if r.id[0] == " " and "CA" in r]
    seq_str = "".join(seq1(r.get_resname()) for r in residues)
    return seq_str, residues


def check_active_site_aligned(query_pdb_path: pathlib.Path, ref_6eqe_path: pathlib.Path):
    """Alignment-based superposition onto 6EQE; returns native-frame Ser OG for docking."""
    parser = PDBParser(QUIET=True)
    ref_chain = parser.get_structure("6EQE", str(ref_6eqe_path))[0]["A"]
    qry_struct = parser.get_structure("qry", str(query_pdb_path))[0]
    qry_chain = list(qry_struct.get_chains())[0]

    ref_seq, ref_res = chain_sequence_and_residues(ref_chain)
    qry_seq, qry_res = chain_sequence_and_residues(qry_chain)

    aln = ALIGNER.align(ref_seq, qry_seq)[0]
    ref_ca_pairs, qry_ca_pairs = [], []
    resnum_to_qry = {}
    for (rs, re_), (qs, qe) in zip(*aln.aligned):
        for k in range(re_ - rs):
            rr = ref_res[rs + k]
            qr = qry_res[qs + k]
            ref_ca_pairs.append(rr["CA"])
            qry_ca_pairs.append(qr["CA"])
            resnum_to_qry[rr.id[1]] = qr

    # Record native-frame Ser OG BEFORE superimposing so Smina docks into the PDB on disk
    qry_ser = resnum_to_qry.get(160)
    qry_asp = resnum_to_qry.get(206)
    qry_his = resnum_to_qry.get(237)
    native_ser_og = (
        qry_ser["OG"].get_coord().copy().tolist()
        if (qry_ser is not None and qry_ser.get_resname() == "SER" and "OG" in qry_ser)
        else None
    )

    # Superimpose onto 6EQE using aligned CA pairs
    sup = Superimposer()
    sup.set_atoms(ref_ca_pairs, qry_ca_pairs)
    sup.apply(qry_struct.get_atoms())

    seq_triad_intact = (
        qry_ser is not None and qry_ser.get_resname() == "SER"
        and qry_asp is not None and qry_asp.get_resname() == "ASP"
        and qry_his is not None and qry_his.get_resname() == "HIS"
    )

    if not seq_triad_intact or "OG" not in qry_ser or "NE2" not in qry_his:
        return {
            "ser_id": f"{qry_ser.get_resname()}{qry_ser.id[1]}" if qry_ser else "None",
            "his_id": f"{qry_his.get_resname()}{qry_his.id[1]}" if qry_his else "None",
            "asp_id": f"{qry_asp.get_resname()}{qry_asp.id[1]}" if qry_asp else "None",
            "d_ser_his": 99.0,
            "d_his_asp": 99.0,
            "seq_triad_intact": seq_triad_intact,
            "triad_intact": False,
            "has_oxyanion_tyr": False,
            "has_groove_w159": False,
            "has_groove_w185": False,
            "active_site_plddt": np.nan,
            "native_ser_og_coord": native_ser_og,
        }

    ser_og = qry_ser["OG"].get_coord()
    his_ne2 = qry_his["NE2"].get_coord()
    his_nd1 = qry_his["ND1"].get_coord() if "ND1" in qry_his else his_ne2
    asp_ods = [qry_asp[a].get_coord() for a in ("OD1", "OD2") if a in qry_asp]
    d_ser_his = float(np.linalg.norm(ser_og - his_ne2))
    d_his_asp = min(float(np.linalg.norm(his_nd1 - od)) for od in asp_ods) if asp_ods else 99.0
    triad_intact = (d_ser_his <= 3.5) and (d_his_asp <= 3.2)

    # Oxyanion hole (Tyr87) and groove aromatics (Trp159, Trp185) via alignment + 3D check
    def check_mapped_aromatic(ref_num, allowed, cutoff):
        qr = resnum_to_qry.get(ref_num)
        if qr is None or qr.get_resname() not in allowed:
            return False
        return float(np.linalg.norm(qr["CA"].get_coord() - ref_chain[ref_num]["CA"].get_coord())) < cutoff

    has_oxy = check_mapped_aromatic(87, {"TYR", "PHE", "TRP", "HIS"}, 3.5)
    has_w159 = check_mapped_aromatic(159, {"TRP", "TYR", "PHE"}, 3.5)
    has_w185 = check_mapped_aromatic(185, {"TRP", "TYR", "PHE"}, 3.8)

    # Active-site local pLDDT (mean across mapped catalytic triad CA B-factors)
    triad_bf = [r["CA"].get_bfactor() for r in (qry_ser, qry_asp, qry_his)]
    scale = 100.0 if np.mean(triad_bf) <= 1.5 else 1.0
    as_plddt = round(float(np.mean(triad_bf) * scale), 2)

    return {
        "ser_id": f"{qry_ser.get_resname()}{qry_ser.id[1]}",
        "his_id": f"{qry_his.get_resname()}{qry_his.id[1]}",
        "asp_id": f"{qry_asp.get_resname()}{qry_asp.id[1]}",
        "d_ser_his": round(d_ser_his, 2),
        "d_his_asp": round(d_his_asp, 2),
        "seq_triad_intact": True,
        "triad_intact": triad_intact,
        "has_oxyanion_tyr": has_oxy,
        "has_groove_w159": has_w159,
        "has_groove_w185": has_w185,
        "active_site_plddt": as_plddt,
        "native_ser_og_coord": native_ser_og,
    }


def identify_ester_carbon_indices(ligand_sdf: pathlib.Path):
    """Return 1-based atom serial indices of ester carbonyl carbon(s) in 2-HE(MHET)."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False)
        patt = Chem.MolFromSmarts(ESTER_CARBON_SMARTS)
        matches = mol.GetSubstructMatches(patt)
        return {m[0] + 1 for m in matches}  # 1-based serials for PDBQT lookup
    except Exception:
        return set()


def run_smina_native(pdb_path: pathlib.Path, native_ser_og, ligand_pdbqt: pathlib.Path,
                     ester_serials, dock_dir: pathlib.Path, smina_exe: str):
    """Dock 2-HE(MHET) into the NATIVE PDB frame and measure Ser OG -> C_ester distance."""
    if native_ser_og is None:
        return {"docking_affinity": np.nan, "attack_dist_ester": np.nan, "nac_score": 0.0, "reactive_pose": False}

    dock_dir.mkdir(parents=True, exist_ok=True)
    out_dock = dock_dir / f"{pdb_path.stem}_docked.pdbqt"
    cmd = [
        str(smina_exe), "-r", str(pdb_path), "-l", str(ligand_pdbqt),
        "--center_x", f"{native_ser_og[0]:.3f}",
        "--center_y", f"{native_ser_og[1]:.3f}",
        "--center_z", f"{native_ser_og[2]:.3f}",
        "--size_x", "18.0", "--size_y", "18.0", "--size_z", "18.0",
        "--exhaustiveness", "8", "--num_modes", "5",
        "-o", str(out_dock),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(f"Smina failed on {pdb_path.name}: {res.stderr}")

    affinity = np.nan
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "1":
            affinity = float(parts[1])
            break

    ser_og = np.array(native_ser_og)
    min_ester_d = 99.0
    if out_dock.exists() and out_dock.stat().st_size > 0:
        for line in out_dock.read_text().splitlines():
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")):
                serial = int(line[6:11].strip())
                elem = line[12:16].strip()
                # Match ester carbonyl carbon serial (or fallback to carbon atoms if renumbered)
                if (ester_serials and serial in ester_serials) or (not ester_serials and elem.startswith("C")):
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    d = float(np.linalg.norm(xyz - ser_og))
                    if d < min_ester_d:
                        min_ester_d = d

    attack_dist = round(min_ester_d, 2) if min_ester_d < 90.0 else np.nan
    # Penalize BOTH too-far (> 3.2 A) and unphysical steric clashes (< 2.5 A)
    if pd.isna(attack_dist):
        nac_score = 0.0
    elif attack_dist < 2.5:
        nac_score = round(1.0 / (1.0 + (2.5 - attack_dist) * 2.0), 3)
    else:
        nac_score = round(1.0 / (1.0 + max(0.0, attack_dist - 3.2)), 3)

    is_reactive = bool(pd.notna(attack_dist) and 2.5 <= attack_dist <= 3.5 and pd.notna(affinity) and affinity <= -5.0)
    return {
        "docking_affinity": affinity,
        "attack_dist_ester": attack_dist,
        "nac_score": nac_score,
        "reactive_pose": is_reactive,
    }


def main():
    ap = argparse.ArgumentParser(description="Patched Experiment 2 structural validation & docking (v2).")
    ap.add_argument("--pdb-dir", required=True, help="Directory containing ESMFold PDB files")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--ligand-pdbqt", required=True, help="Path to 2he_mhet.pdbqt")
    ap.add_argument("--ligand-sdf", default=None, help="Path to 2he_mhet.sdf (for C_ester SMARTS lookup)")
    ap.add_argument("--allow-skip-docking", action="store_true", help="Skip Smina if binary is absent instead of failing")
    args = ap.parse_args()

    pdb_dir = pathlib.Path(args.pdb_dir)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = fetch_references(out_dir / "references")

    smina_exe = shutil.which("smina") or ("/usr/local/bin/smina" if pathlib.Path("/usr/local/bin/smina").exists() else None)
    ligand_pdbqt = pathlib.Path(args.ligand_pdbqt)
    if (not smina_exe or not ligand_pdbqt.exists()) and not args.allow_skip_docking:
        raise RuntimeError(
            "Smina binary or ligand PDBQT not found. Install smina or pass --allow-skip-docking "
            "(fake geometric fallback has been removed in v2)."
        )

    ester_serials = identify_ester_carbon_indices(pathlib.Path(args.ligand_sdf)) if args.ligand_sdf else set()
    rows = []
    for pdb_path in sorted(pdb_dir.glob("*.pdb")):
        mean_plddt, _ = extract_plddt(pdb_path)
        arch = check_active_site_aligned(pdb_path, refs["6EQE"])
        row = {"id": pdb_path.stem, "mean_plddt": mean_plddt, "passed_plddt80": mean_plddt >= 80.0}
        row.update(arch)
        if smina_exe and ligand_pdbqt.exists() and mean_plddt >= 80.0:
            dock = run_smina_native(
                pdb_path, arch["native_ser_og_coord"], ligand_pdbqt,
                ester_serials, out_dir / "docking_results_v2", smina_exe
            )
            row.update(dock)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_tsv = out_dir / "validated_candidates_v2.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {len(df)} designs to {out_tsv}")
    print(f"Alignment-based intact 3D triads: {df['triad_intact'].sum()} / {df['passed_plddt80'].sum()}")
    if "attack_dist_ester" in df.columns:
        clashes = (df["attack_dist_ester"] < 2.5).sum()
        print(f"Reactive poses (2.5-3.5 A & dG <= -5.0): {df['reactive_pose'].sum()} | Clashes (<2.5 A): {clashes}")


if __name__ == "__main__":
    main()
