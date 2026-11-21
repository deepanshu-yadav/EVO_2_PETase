#!/usr/bin/env python3
"""Pipeline v2 selection: plausibility GATES (not scores) -> diversity-aware, controlled wet-lab panel.
Updated for EVO_2_PETase-main (2).zip: ingests and audits the new Smina docking columns,
flags unphysical < 2.2 A coordinate-frame/clash artifacts, and keeps Panel A docking-free
until run_structural_validation_v2.py is re-run on the ESMFold PDBs.

Usage: python -I rank_v2.py <EVO_2_PETase-main dir> <out dir>
Requires: biopython pandas numpy scipy
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices

R = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
E = R / "experiments/experiment_2_folding/results"

rk = pd.read_csv(E / "ranked_functional_petase_candidates.tsv", sep="\t", keep_default_na=False)
pl = pd.read_csv(E / "esmfold_plddt_all_71.tsv", sep="\t")
seq = dict(zip(pl.id, pl.sequence))
pool_path = R / "prepared_data/three_datasets_16nov2026/petase_pool.fasta"
if not pool_path.exists():
    pool_path = R / "data/petase_pool.fasta"
pool = {r.id: str(r.seq) for r in SeqIO.parse(pool_path, "fasta")}
ispet_matches = [v for k, v in pool.items() if "IsPETase" in k or "ispetase" in k.lower()]
ispet = ispet_matches[0] if ispet_matches else list(pool.values())[0]

cols = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore,rmsd,lddt".split(",")
fs = pd.read_csv(E / "foldseek_vs_references.m8", sep="\t", names=cols)
fs["query"] = fs["query"].astype(str).str.replace(".pdb", "", regex=False)
fs["target"] = fs["target"].astype(str).str.replace(".pdb", "", regex=False)
tm = fs.pivot_table(index="query", columns="target", values="alntmscore", aggfunc="max")

al = Align.PairwiseAligner()
al.substitution_matrix = substitution_matrices.load("BLOSUM62")
al.open_gap_score = -10
al.extend_gap_score = -0.5
al.mode = "global"
al.end_gap_score = 0


def amap(ref, q):
    a = al.align(ref, q)[0]
    mp = {}
    for (rs, re_), (qs, qe) in zip(*a.aligned):
        for k in range(re_ - rs):
            mp[rs + k] = qs + k
    return mp


def ident(a, b):
    x = al.align(a, b)[0]
    return sum(
        sum(1 for k in range(e - s) if a[s + k] == b[qs + k])
        for (s, e), (qs, qe) in zip(*x.aligned)
    ) / min(len(a), len(b))


def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


S_I, D_I, H_I = 133, 179, 210  # IsPETase pool-sequence indices of Ser/Asp/His (0-based)
score_col = "composite_score" if "composite_score" in rk.columns else "composite_fitness_score"

rows = []
for _, r in rk.iterrows():
    q = seq[r.id]
    mp = amap(ispet, q)
    s, d, h = (mp.get(i) for i in (S_I, D_I, H_I))
    seq_triad = all(x is not None for x in (s, d, h)) and q[s] == "S" and q[d] == "D" and q[h] == "H"
    att = to_float(r.get("attack_dist", np.nan))
    aff = to_float(r.get("docking_affinity", np.nan))
    nac = to_float(r.get("nac_score", np.nan))
    comp = to_float(r.get(score_col, np.nan))
    # Flag unphysical non-covalent distances (< 2.2 A) caused by the 6EQE-vs-native PDB frame mismatch
    dock_clash_flag = bool(pd.notna(att) and att < 2.2)
    rows.append(dict(
        id=r.id,
        n_examples=int(r.n_examples) if "n_examples" in rk.columns and str(r.n_examples) != "" else np.nan,
        plddt=to_float(r.mean_plddt),
        max_pid=to_float(r.max_pid) if "max_pid" in rk.columns else np.nan,
        tm_6eqe=tm.loc[r.id, "6EQE"] if r.id in tm.index and "6EQE" in tm.columns else np.nan,
        tm_4eb0=tm.loc[r.id, "4EB0"] if r.id in tm.index and "4EB0" in tm.columns else np.nan,
        struct_triad=str(r.triad_intact).strip().lower() == "true",
        seq_triad=seq_triad,
        ser_idx=s,
        length=len(q),
        cys=q.count("C"),
        smina_affinity=aff,
        smina_attack_dist=att,
        smina_nac=nac,
        smina_clash_artifact=dock_clash_flag,
        score_v1_5_smina=comp,
        sequence=q,
    ))

df = pd.DataFrame(rows)

# No-docking composite: docking/NAC terms remain excluded because the v1.5 Smina run
# docked into unsuperimposed PDBs using 6EQE-superimposed Ser OG coordinates.
v1 = rk.set_index("id")
pen = lambda a, b: max(0.0, a - 2.8) + max(0.0, b - 2.8)
df["score_v1_nodock"] = [
    0.35 * to_float(v1.loc[i, "mean_plddt"]) / 100.0
    + 0.25 * to_float(v1.loc[i, "tm_score_6eqe"])
    - 0.15 * pen(to_float(v1.loc[i, "d_ser_his"]), to_float(v1.loc[i, "d_his_asp"]))
    if str(v1.loc[i, "triad_intact"]).strip().lower() == "true"
    else np.nan
    for i in df.id
]

# Audit summary of the updated Smina run
n_clash = int(df["smina_clash_artifact"].sum())
print(f"Smina audit: {n_clash} designs have unphysical attack_dist < 2.2 A (coordinate-frame mismatch).")
top10_smina = df.sort_values("score_v1_5_smina", ascending=False).head(10).id.tolist()

# GATES
gate = (df.plddt >= 80.0) & df.seq_triad & (df.length.between(241, 306))
P = df[gate].copy()
print("gate-passers:", len(P), "of", len(df))

ids = list(P.id)
idx = {i: k for k, i in enumerate(ids)}
M = np.eye(len(ids))
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        M[i, j] = M[j, i] = ident(seq[ids[i]], seq[ids[j]])

rng = np.random.default_rng(7)
taken = []

# Panel A: functional top arm (top-8 by audited composite score among structurally intact)
A = P[P.struct_triad].sort_values("score_v1_5_smina", ascending=False).head(8).id.tolist()
taken += A
print("Panel A (audited composite score top 8):", A)

# Panel B: diversity arm, farthest-point sampling on identity, decorrelated from pLDDT
rest = [i for i in ids if i not in taken]
cand = [i for i in rest if P.set_index("id").loc[i, "struct_triad"]]
B = [max(cand, key=lambda i: P.set_index("id").loc[i, "max_pid"] * -1)]  # start from most novel
while len(B) < 8:
    nxt = max([i for i in cand if i not in B], key=lambda i: min(1 - M[idx[i], idx[b]] for b in B + A))
    B.append(nxt)
taken += B

# Panel C: random gate-passers (tests whether ANY in-silico ranking beats random)
rest = [i for i in ids if i not in taken and P.set_index("id").loc[i, "struct_triad"]]
C = list(rng.choice(rest, size=8, replace=False))
taken += C

# Panel D: "rescued" = v1 structural-triad FAIL but sequence triad intact; top by TM to 6EQE, diverse
D0 = P[~P.struct_triad].sort_values("tm_6eqe", ascending=False)
D = []
for i in D0.id:
    if all(M[idx[i], idx[d]] < 0.7 for d in D):
        D.append(i)
    if len(D) == 6:
        break

panel = []
for name, lst in [("A_structure_top", A), ("B_diverse", B), ("C_random", C), ("D_rescued_v1_fail", D)]:
    for i in lst:
        panel.append((i, name))

pp = df.set_index("id")
out = []
for i, name in panel:
    r = pp.loc[i]
    out.append(dict(
        panel=name,
        id=i,
        n_examples=r.n_examples,
        plddt=r.plddt,
        max_pid_to_pool=r.max_pid,
        tm_6eqe=round(r.tm_6eqe, 3) if pd.notna(r.tm_6eqe) else np.nan,
        tm_4eb0=round(r.tm_4eb0, 3) if pd.notna(r.tm_4eb0) else np.nan,
        struct_triad=r.struct_triad,
        smina_affinity=r.smina_affinity,
        smina_attack_dist=r.smina_attack_dist,
        smina_clash_artifact=r.smina_clash_artifact,
        cys=r.cys,
        length=r.length,
        sequence=r.sequence,
    ))

# Panel E: knock-out controls (Ser->Ala at catalytic serine of the two best Panel A designs)
for i in A[:2]:
    q = pp.loc[i, "sequence"]
    s = int(pp.loc[i, "ser_idx"])
    out.append(dict(
        panel="E_knockout_S_to_A",
        id=f"{i}_S{s+1}A",
        n_examples=pp.loc[i, "n_examples"],
        plddt=np.nan,
        max_pid_to_pool=np.nan,
        tm_6eqe=np.nan,
        tm_4eb0=np.nan,
        struct_triad=np.nan,
        smina_affinity=np.nan,
        smina_attack_dist=np.nan,
        smina_clash_artifact=np.nan,
        cys=q.count("C"),
        length=len(q),
        sequence=q[:s] + "A" + q[s + 1:],
    ))

# Panel F: natural controls
for k, v in pool.items():
    if any(t in k for t in ["IsPETase", "LCC", "TfCut2", "PHL7"]):
        out.append(dict(
            panel="F_natural_control",
            id=k,
            n_examples=np.nan,
            plddt=np.nan,
            max_pid_to_pool=np.nan,
            tm_6eqe=np.nan,
            tm_4eb0=np.nan,
            struct_triad=np.nan,
            smina_affinity=np.nan,
            smina_attack_dist=np.nan,
            smina_clash_artifact=np.nan,
            cys=v.count("C"),
            length=len(v),
            sequence=v,
        ))

O = pd.DataFrame(out)
O.to_csv(OUT / "wetlab_panel_v2.tsv", sep="\t", index=False)
with open(OUT / "wetlab_panel_v2.faa", "w") as f:
    for _, r in O.iterrows():
        f.write(f">{r.id}|{r.panel}\n{r.sequence}\n")

df.drop(columns="sequence").to_csv(OUT / "all_designs_v2_features.tsv", sep="\t", index=False)
print(O.drop(columns="sequence").to_string())
print(O.panel.value_counts().to_dict())

for p, g in O[O.panel.str.match("[ABCD]_")].groupby("panel"):
    print(
        p,
        "mean pLDDT", round(g.plddt.mean(), 1),
        "mean maxPID", round(g.max_pid_to_pool.mean(), 1),
        "n-values", sorted(g.n_examples.dropna().astype(int).unique()),
    )
