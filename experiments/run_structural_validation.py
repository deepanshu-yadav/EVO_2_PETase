#!/usr/bin/env python3
"""
Structural Validation and Catalytic Triad 3D Check for De Novo PETase Designs.

Workflow:
1. Load designs from `experiments/experiment1_initial_tests/designs_71_novel.faa`
2. Predict 3D structure with ESMFold (local GPU or ESM Atlas API fallback)
3. Extract pLDDT from PDB B-factors and filter with threshold >= 80.0
4. Align structures to IsPETase (6EQE) and LCC (4EB0) via Foldseek
5. Verify 3D catalytic triad geometry (Ser-His-Asp spatial distances)
6. Export ranked functional candidate report
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.PDB import PDBParser

# Default URLs & References
FOLDSEEK_URL = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
REFS = {
    "6EQE": "https://files.rcsb.org/download/6EQE.pdb",
    "4EB0": "https://files.rcsb.org/download/4EB0.pdb"
}


def setup_foldseek(tools_dir):
    tools_dir = pathlib.Path(tools_dir)
    fs_bin = tools_dir / "foldseek" / "bin" / "foldseek"
    if fs_bin.exists():
        return fs_bin

    if sys.platform.startswith("linux"):
        print("Downloading static Foldseek binary for Linux...")
        tar_path = tools_dir / "foldseek.tar.gz"
        urllib.request.urlretrieve(FOLDSEEK_URL, tar_path)
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=tools_dir)
        subprocess.run(["chmod", "+x", str(fs_bin)], check=True)
        return fs_bin
    else:
        # On Windows or Mac, check if foldseek is in PATH
        which_fs = shutil.which("foldseek")
        if which_fs:
            return pathlib.Path(which_fs)
        print("Note: Foldseek binary not found locally. Skipping CLI alignment or run inside Colab/Linux.")
        return None


def fetch_references(ref_dir):
    ref_dir = pathlib.Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    out_files = {}
    for pdb_id, url in REFS.items():
        dst = ref_dir / f"{pdb_id}.pdb"
        if not dst.exists():
            print(f"Fetching reference {pdb_id} from RCSB PDB...")
            urllib.request.urlretrieve(url, dst)
        out_files[pdb_id] = dst
    return out_files


def run_esmfold_batch(records, out_pdb_dir, device="cuda"):
    out_pdb_dir = pathlib.Path(out_pdb_dir)
    out_pdb_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing ESMFold on {device}...")
    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        model = model.to(device)
        model.eval()
        use_hf = True
    except Exception as e:
        print(f"Transformers ESMFold loading error: {e}. Falling back to ESM Atlas API...")
        use_hf = False

    results = []
    t0 = time.time()
    for i, r in enumerate(records, 1):
        qid = r.id
        seq = str(r.seq)
        pdb_file = out_pdb_dir / f"{qid}.pdb"

        if not pdb_file.exists():
            if use_hf:
                with torch.no_grad():
                    inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out = model(**inputs)
                    pdb_str = model.output_to_pdb(out)[0]
                    pdb_file.write_text(pdb_str)
            else:
                # ESM Atlas REST API fallback
                req = urllib.request.Request(
                    "https://api.esmatlas.com/v1/fold/",
                    data=seq.encode("utf-8"),
                    headers={"Content-Type": "text/plain"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        pdb_str = resp.read().decode("utf-8")
                        pdb_file.write_text(pdb_str)
                except Exception as err:
                    print(f"Error folding {qid}: {err}")
                    continue

        # Extract pLDDT from CA B-factors
        bfactors = []
        for line in pdb_file.read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    bfactors.append(float(line[60:66].strip()))
                except:
                    pass
        mean_plddt = sum(bfactors) / max(len(bfactors), 1)

        results.append({
            "id": qid,
            "sequence": seq,
            "length": len(seq),
            "mean_plddt": round(mean_plddt, 2),
            "passed_plddt80": mean_plddt >= 80.0,
            "pdb_path": str(pdb_file)
        })

        if i % 10 == 0 or i == len(records):
            print(f"  [{i:>2}/{len(records)}] {qid:<20} len={len(seq)} mean_pLDDT={mean_plddt:.1f} ({time.time() - t0:.1f}s)")

    return pd.DataFrame(results)


def check_active_site_triad(pdb_path, sequence):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", str(pdb_path))
    except:
        return None

    # Find Serine via motif: [GWA][A-Z]S[A-Z]G
    motif_match = re.search(r'[GWA].[SMG].G', sequence)
    ser_idx = motif_match.start() + 2 if motif_match else None
    if ser_idx is None:
        m2 = re.search(r'G.S', sequence)
        ser_idx = m2.start() + 2 if m2 else None

    model = structure[0]
    chain = list(model.get_chains())[0]
    residues = [r for r in chain if r.id[0] == ' ']

    if ser_idx is None or ser_idx >= len(residues):
        return None

    ser_res = residues[ser_idx]
    if ser_res.get_resname() != "SER" or "OG" not in ser_res:
        return None

    ser_og = ser_res["OG"].get_coord()

    # Search His
    best_his = None
    min_ser_his_dist = 999.0
    for r in residues:
        if r.get_resname() == "HIS" and "NE2" in r and "ND1" in r:
            dist = np.linalg.norm(ser_og - r["NE2"].get_coord())
            if dist < min_ser_his_dist:
                min_ser_his_dist = dist
                best_his = r

    if best_his is None or min_ser_his_dist > 8.0:
        return None

    his_res = best_his
    his_ne2 = his_res["NE2"].get_coord()
    his_nd1 = his_res["ND1"].get_coord()

    # Search Asp
    best_asp = None
    min_his_asp_dist = 999.0
    for r in residues:
        if r.get_resname() == "ASP":
            d1 = np.linalg.norm(his_nd1 - r["OD1"].get_coord()) if "OD1" in r else 999
            d2 = np.linalg.norm(his_nd1 - r["OD2"].get_coord()) if "OD2" in r else 999
            d_min = min(d1, d2)
            if d_min < min_his_asp_dist:
                min_his_asp_dist = d_min
                best_asp = r

    if best_asp is None:
        return None

    asp_res = best_asp
    asp_coord = asp_res["OD1"].get_coord() if "OD1" in asp_res else asp_res["OD2"].get_coord()
    asp_cg = asp_res["CG"].get_coord() if "CG" in asp_res else asp_coord
    ser_asp_dist = np.linalg.norm(ser_og - asp_cg)

    return {
        "ser_id": f"{ser_res.get_resname()}{ser_res.id[1]}",
        "his_id": f"{his_res.get_resname()}{his_res.id[1]}",
        "asp_id": f"{asp_res.get_resname()}{asp_res.id[1]}",
        "d_ser_his": round(float(min_ser_his_dist), 2),
        "d_his_asp": round(float(min_his_asp_dist), 2),
        "d_ser_asp": round(float(ser_asp_dist), 2),
        "triad_intact": (min_ser_his_dist <= 3.5) and (min_his_asp_dist <= 3.2)
    }


def main():
    parser = argparse.ArgumentParser(description="Fold designs and verify catalytic triad.")
    parser.add_argument("--fasta", default="experiments/experiment1_initial_tests/designs_71_novel.faa", help="Path to FASTA with novel designs")
    parser.add_argument("--out", default="experiments/experiment2_structural_validation", help="Output directory")
    parser.add_argument("--device", default="cuda" if sys.platform.startswith("linux") else "cpu", help="Device for ESMFold")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = out_dir / "esmfold_pdbs"
    ref_dir = out_dir / "references"

    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"Loaded {len(records)} sequences from {args.fasta}")

    # 1. Fold
    df_folding = run_esmfold_batch(records, pdb_dir, device=args.device)
    df_folding.to_csv(out_dir / "esmfold_plddt_summary.tsv", sep="\t", index=False)

    passed_df = df_folding[df_folding["passed_plddt80"]].copy()
    print(f"\nDesigns with pLDDT >= 80.0: {len(passed_df)} / {len(df_folding)}")

    # 2. Check references & Foldseek
    refs = fetch_references(ref_dir)
    fs_bin = setup_foldseek(out_dir / "tools")
    if fs_bin:
        fs_out = out_dir / "foldseek_alignments.m8"
        tmp_dir = out_dir / "tmp_fs"
        tmp_dir.mkdir(exist_ok=True)
        fields = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore,rmsd,lddt"
        cmd = [str(fs_bin), "easy-search", str(pdb_dir), str(ref_dir), str(fs_out), str(tmp_dir),
               "--format-output", fields, "-e", "1.0", "--alignment-type", "1"]
        print("Running Foldseek structural alignment...")
        subprocess.run(cmd, check=True)
        print(f"Wrote Foldseek results to {fs_out}")

    # 3. Check 3D active site triad
    print("\nChecking 3D active site catalytic triad geometry...")
    triad_rows = []
    for _, row in passed_df.iterrows():
        triad = check_active_site_triad(row["pdb_path"], row["sequence"])
        t_data = {
            "id": row["id"],
            "length": row["length"],
            "mean_plddt": row["mean_plddt"]
        }
        if triad:
            t_data.update(triad)
        else:
            t_data.update({"triad_intact": False, "ser_id": "None", "his_id": "None", "asp_id": "None",
                           "d_ser_his": 99.0, "d_his_asp": 99.0, "d_ser_asp": 99.0})
        triad_rows.append(t_data)

    df_triads = pd.DataFrame(triad_rows)
    df_triads.to_csv(out_dir / "catalytic_triad_report.tsv", sep="\t", index=False)
    intact = df_triads["triad_intact"].sum()
    print(f"Catalytic Triad Intact: {intact} / {len(df_triads)}")
    print(f"Results saved in {out_dir}")


if __name__ == "__main__":
    main()
