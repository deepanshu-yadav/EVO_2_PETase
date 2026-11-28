#!/usr/bin/env python3
"""Module 4: Solid-Polymer Interfacial Adsorption & Thermostability Proxies.

Computes for every screened Experiment 3 candidate:
1. Component 4.1: Substrate-cleft hydrophobic SASA (A_cleft_hp) and local cleft charge.
2. Component 4.2: Disulfide bond topology (C-terminal disulfide & active-site DS1/DS2).
3. Component 4.3: Thermostability Tm prediction (TemStaPro/DeepSTABp ingestion or 139-pool calibrated model).
4. Component 4.4: E. coli bacterial solubility proxy (gate >= 0.45).

Outputs:
  - interfacial_and_stability_features.tsv
  - module_4_summary.json
"""
import argparse
import json
import pathlib
import sys
from typing import Dict, Any, Optional

import pandas as pd
from Bio import SeqIO

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
EXP3_DIR = CURRENT_DIR.parent
REPO_ROOT = EXP3_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from biophysical_proxies_utils import (
    audit_disulfide_topology,
    compute_cleft_interface_proxies,
    extract_sequence_biophysical_features,
    fit_pool_tm_calibrator,
    predict_ecoli_solubility,
    predict_thermostability,
)


def _parse_coord(val: Any) -> Optional[list]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return [float(x) for x in val]
    if isinstance(val, str) and val.strip().startswith("["):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list) and len(parsed) == 3:
                return [float(x) for x in parsed]
        except Exception:
            return None
    return None


def run_module_4(
    fasta_path: pathlib.Path,
    reranked_tsv: pathlib.Path,
    pool_fasta: pathlib.Path,
    pool_metadata_tsv: pathlib.Path,
    out_dir: pathlib.Path,
    grid_tsv: Optional[pathlib.Path] = None,
    relaxed_pdb_dir: Optional[pathlib.Path] = None,
    esmfold_pdb_dir: Optional[pathlib.Path] = None,
    plddt_tsv: Optional[pathlib.Path] = None,
    triad_tsv: Optional[pathlib.Path] = None,
    disulfide_tsv: Optional[pathlib.Path] = None,
    external_tm_tsv: Optional[pathlib.Path] = None,
    external_sol_tsv: Optional[pathlib.Path] = None,
) -> pd.DataFrame:
    """Execute Module 4 biophysical proxy computation and save summary artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 3 — MODULE 4: INTERFACIAL ADSORPTION & THERMOSTABILITY PROXIES")
    print("=" * 78)

    with open(fasta_path, "r", encoding="utf-8") as fh:
        records = {r.id: str(r.seq).upper().replace("*", "") for r in SeqIO.parse(fh, "fasta")}
    print(f"Loaded {len(records)} candidate sequences from {fasta_path.name}")

    rerank_map: Dict[str, Dict[str, Any]] = {}
    if grid_tsv and grid_tsv.exists():
        df_g = pd.read_csv(grid_tsv, sep="\t")
        if "design_id" in df_g.columns:
            rerank_map.update(df_g.drop_duplicates("design_id").set_index("design_id").to_dict(orient="index"))
    if reranked_tsv.exists():
        df_rerank = pd.read_csv(reranked_tsv, sep="\t")
        if "design_id" in df_rerank.columns:
            for k, v in df_rerank.drop_duplicates("design_id").set_index("design_id").to_dict(orient="index").items():
                rerank_map.setdefault(k, {}).update(v)

    plddt_map: Dict[str, float] = {}
    if plddt_tsv and plddt_tsv.exists():
        df_p = pd.read_csv(plddt_tsv, sep="\t")
        if {"id", "mean_plddt"}.issubset(df_p.columns):
            plddt_map = dict(zip(df_p["id"], df_p["mean_plddt"]))

    triad_map: Dict[str, Dict[str, Any]] = {}
    if triad_tsv and triad_tsv.exists():
        df_t = pd.read_csv(triad_tsv, sep="\t")
        if "id" in df_t.columns:
            triad_map = df_t.set_index("id").to_dict(orient="index")

    ds_map: Dict[str, Dict[str, Any]] = {}
    if disulfide_tsv and disulfide_tsv.exists():
        df_d = pd.read_csv(disulfide_tsv, sep="\t")
        if "id" in df_d.columns:
            ds_map = df_d.set_index("id").to_dict(orient="index")

    ext_tm_map: Dict[str, Dict[str, Any]] = {}
    if external_tm_tsv and external_tm_tsv.exists():
        df_tm = pd.read_csv(external_tm_tsv, sep="\t")
        id_col = "id" if "id" in df_tm.columns else ("design_id" if "design_id" in df_tm.columns else None)
        if id_col:
            ext_tm_map = df_tm.set_index(id_col).to_dict(orient="index")

    ext_sol_map: Dict[str, float] = {}
    if external_sol_tsv and external_sol_tsv.exists():
        df_sol = pd.read_csv(external_sol_tsv, sep="\t")
        id_col = "id" if "id" in df_sol.columns else ("design_id" if "design_id" in df_sol.columns else None)
        if id_col and "solubility_score" in df_sol.columns:
            ext_sol_map = dict(zip(df_sol[id_col], df_sol["solubility_score"]))

    calibrator = fit_pool_tm_calibrator(pool_fasta, pool_metadata_tsv)
    if calibrator:
        print(f"Calibrated PETase Tm predictor on {calibrator['n_samples']} pool enzymes (train MAE={calibrator['train_mae']} C)")

    rows = []
    for qid, seq in records.items():
        pdb_path: Optional[pathlib.Path] = None
        if relaxed_pdb_dir and (relaxed_pdb_dir / f"{qid}_relaxed.pdb").exists():
            pdb_path = relaxed_pdb_dir / f"{qid}_relaxed.pdb"
        elif esmfold_pdb_dir and (esmfold_pdb_dir / f"{qid}.pdb").exists():
            pdb_path = esmfold_pdb_dir / f"{qid}.pdb"

        ser_coord = _parse_coord(triad_map.get(qid, {}).get("ser_og_coord"))
        cleft = compute_cleft_interface_proxies(pdb_path, seq, ser_og_coord=ser_coord, cleft_radius=8.0)

        if qid in ds_map and pdb_path is None:
            ds = {
                "disulfide_count": int(ds_map[qid].get("disulfide_count", 0)),
                "has_cterm_disulfide": bool(ds_map[qid].get("has_cterm_disulfide", False)),
                "has_active_site_disulfide": bool(ds_map[qid].get("has_active_site_disulfide", False)),
                "disulfide_source": "module_3_report",
            }
        else:
            ds = audit_disulfide_topology(pdb_path, seq)

        rr = rerank_map.get(qid, {})
        delta_ll = rr.get("delta_ll")
        mean_plddt = plddt_map.get(qid)
        ext_tm = ext_tm_map.get(qid, {})
        tm_res = predict_thermostability(
            sequence=seq,
            calibrator=calibrator,
            has_cterm_disulfide=bool(ds["has_cterm_disulfide"]),
            disulfide_count=int(ds["disulfide_count"]),
            delta_ll=float(delta_ll) if delta_ll is not None else None,
            temstapro_prob65=ext_tm.get("temstapro_tm65_prob"),
            deepstabp_tm=ext_tm.get("deepstabp_tm") or ext_tm.get("predicted_tm"),
        )

        sol_res = predict_ecoli_solubility(
            sequence=seq,
            mean_plddt=float(mean_plddt) if mean_plddt is not None else None,
            external_solubility=ext_sol_map.get(qid),
        )
        seq_feats = extract_sequence_biophysical_features(seq)

        rows.append({
            "id": qid,
            "length": int(seq_feats["length"]),
            "cleft_hydrophobic_sasa": cleft["cleft_hydrophobic_sasa"],
            "net_charge_cleft": cleft["net_charge_cleft"],
            "cleft_hp_res_count": cleft["cleft_hp_res_count"],
            "cleft_sasa_method": cleft["cleft_sasa_method"],
            "disulfide_count": ds["disulfide_count"],
            "has_cterm_disulfide": ds["has_cterm_disulfide"],
            "has_active_site_disulfide": ds["has_active_site_disulfide"],
            "disulfide_source": ds["disulfide_source"],
            "predicted_tm": tm_res["predicted_tm"],
            "temstapro_tm65_prob": tm_res["temstapro_tm65_prob"],
            "passed_tm65": tm_res["passed_tm65"],
            "tm_source": tm_res["tm_source"],
            "solubility_score": sol_res["solubility_score"],
            "passed_solubility_gate": sol_res["passed_solubility_gate"],
            "solubility_source": sol_res["solubility_source"],
            "aliphatic_index": seq_feats["aliphatic_index"],
            "gravy": seq_feats["gravy"],
            "isoelectric_point": seq_feats["isoelectric_point"],
            "instability_index": seq_feats["instability_index"],
        })

    df_out = pd.DataFrame(rows)
    out_tsv = out_dir / "interfacial_and_stability_features.tsv"
    df_out.to_csv(out_tsv, sep="\t", index=False)

    summary = {
        "total_evaluated": len(df_out),
        "passed_tm65_count": int(df_out["passed_tm65"].sum()),
        "passed_solubility_gate_count": int(df_out["passed_solubility_gate"].sum()),
        "cterm_disulfide_count": int(df_out["has_cterm_disulfide"].sum()),
        "mean_predicted_tm": round(float(df_out["predicted_tm"].mean()), 2),
        "mean_solubility_score": round(float(df_out["solubility_score"].mean()), 4),
        "mean_cleft_hydrophobic_sasa": round(float(df_out["cleft_hydrophobic_sasa"].mean()), 2),
        "sasa_methods": df_out["cleft_sasa_method"].value_counts().to_dict(),
        "output_tsv": str(out_tsv),
    }
    with open(out_dir / "module_4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nModule 4 Summary ({len(df_out)} designs):")
    print(f"  Mean Cleft Hydrophobic SASA (A_cleft_hp): {summary['mean_cleft_hydrophobic_sasa']:.1f} A^2")
    print(f"  Intact C-Terminal Disulfide:              {summary['cterm_disulfide_count']} / {len(df_out)}")
    print(f"  Predicted Tm >= 65.0 C:                   {summary['passed_tm65_count']} / {len(df_out)} (mean {summary['mean_predicted_tm']:.1f} C)")
    print(f"  Soluble Expression Gate (>= 0.45):        {summary['passed_solubility_gate_count']} / {len(df_out)} (mean {summary['mean_solubility_score']:.3f})")
    print(f"  Saved: {out_tsv}")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Module 4: Interfacial & Thermostability Proxies")
    parser.add_argument("--fasta_path", type=pathlib.Path, default=EXP3_DIR / "outputs" / "designs_exp3_screened_novel.faa")
    parser.add_argument("--reranked_tsv", type=pathlib.Path, default=EXP3_DIR / "outputs" / "contrastive_reranked_designs.tsv")
    parser.add_argument("--grid_tsv", type=pathlib.Path, default=EXP3_DIR / "outputs" / "generated_continuations_grid.tsv")
    parser.add_argument("--pool_fasta", type=pathlib.Path, default=REPO_ROOT / "data" / "petase_pool.fasta")
    parser.add_argument("--pool_metadata", type=pathlib.Path, default=EXP3_DIR / "outputs" / "metadata_pool_annotated.tsv")
    parser.add_argument("--relaxed_pdb_dir", type=pathlib.Path, default=None)
    parser.add_argument("--esmfold_pdb_dir", type=pathlib.Path, default=None)
    parser.add_argument("--plddt_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--triad_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--disulfide_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--external_tm_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--external_sol_tsv", type=pathlib.Path, default=None)
    parser.add_argument("--out_dir", type=pathlib.Path, default=EXP3_DIR / "outputs")
    args = parser.parse_args()

    run_module_4(
        fasta_path=args.fasta_path,
        reranked_tsv=args.reranked_tsv,
        grid_tsv=args.grid_tsv,
        pool_fasta=args.pool_fasta,
        pool_metadata_tsv=args.pool_metadata,
        out_dir=args.out_dir,
        relaxed_pdb_dir=args.relaxed_pdb_dir,
        esmfold_pdb_dir=args.esmfold_pdb_dir,
        plddt_tsv=args.plddt_tsv,
        triad_tsv=args.triad_tsv,
        disulfide_tsv=args.disulfide_tsv,
        external_tm_tsv=args.external_tm_tsv,
        external_sol_tsv=args.external_sol_tsv,
    )


if __name__ == "__main__":
    main()
