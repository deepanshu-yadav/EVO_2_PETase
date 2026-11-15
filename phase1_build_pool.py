#!/usr/bin/env python3
"""Phase 1 -- build a single-family PETase pool for the small_protein_ICL notebook.

Inputs : PANDA supplementary xlsx (Proteins pro.70282), PAZy FASTA, PlasticDB FASTA
Outputs: <out>/petase_pool.fasta        mature domains, Met-started (what the notebook reads)
         <out>/petase_pool_report.tsv   provenance + why each sequence was kept / dropped

Logic
  1. Seeds   = PANDA unique sequences in SSN "Cluster 1" (experimentally validated PET
               hydrolases, signal peptide already removed).
  2. Candidates = PAZy + PlasticDB. A candidate is kept only if it aligns to a seed with
               >= MIN_ID identity and >= MIN_COV of the seed length (a homology gate, not a
               name regex), so PHB depolymerases / laccases / etc. are rejected.
  3. Trim    = project the seed's boundaries onto the candidate -> removes signal peptide.
  4. Filter  = 240-305 aa, 20 standard amino acids only.
  5. Cluster = greedy, <= MAX_ID identity to anything already kept; PANDA goes first.
  6. Prepend M (the notebook's bla cache does this too: n_forced_start_met).

Usage:  python3 phase1_build_pool.py --panda X.xlsx --pazy pazy.fasta --plasticdb db.fasta --out data
Needs : biopython, openpyxl
"""
import argparse, collections, csv, pathlib, re
import openpyxl
from Bio import Align
from Bio.Align import substitution_matrices

LEN_LO, LEN_HI = 240, 305
MIN_ID, MIN_COV = 0.35, 0.70     # homology gate vs a PANDA Cluster-1 seed
MAX_ID = 0.90                    # redundancy cut inside the pool
CLUSTER = "Cluster 1"
STD = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    out, h = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:]
            out.append([h, ""])
        elif h is not None:
            out[-1][1] += line
    return [(h, s.upper().replace("*", "")) for h, s in out if s]


def read_panda(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb["PANDA"].iter_rows(values_only=True))
    ix = {k: i for i, k in enumerate(rows[0])}
    uniq = {}
    for r in rows[1:]:
        s = r[ix["clean_AAseq_resolved"]]
        if not s or s == "None":
            continue
        s = s.strip().upper()
        uniq.setdefault(s, dict(name=str(r[ix["UniName"]]), cluster=str(r[ix["SSN_clustering"]])))
    return uniq


def make_aligner(mode):
    a = Align.PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.mode = mode
    a.open_gap_score, a.extend_gap_score = -11, -1
    return a


LOCAL = make_aligner("local")
GLOBAL = make_aligner("global")
GLOBAL.open_gap_score, GLOBAL.extend_gap_score = -10, -0.5


def identity(aln):
    """matches / aligned columns, plus (q_start, q_end, t_start, t_end) of the aligned span."""
    q, t = aln.aligned
    m = cols = 0
    # Biopython: aln.target is the FIRST sequence passed to align(), aln.query the second,
    # and aln.aligned returns blocks in that same (first, second) order.
    qs, ts = aln.target, aln.query
    for (a0, a1), (b0, b1) in zip(q, t):
        cols += a1 - a0
        m += sum(x == y for x, y in zip(qs[a0:a1], ts[b0:b1]))
    return m / max(cols, 1), q[0][0], q[-1][1], t[0][0], t[-1][1]


def best_seed_hit(cand, seeds):
    """Align candidate to each seed; return the best (score-wise) hit that passes the gate."""
    best = None
    for name, s in seeds:
        sc = LOCAL.score(cand, s)
        if best is None or sc > best[0]:
            best = (sc, name, s)
    sc, name, s = best
    aln = LOCAL.align(cand, s)[0]
    ident, q0, q1, t0, t1 = identity(aln)
    cov = (t1 - t0) / len(s)
    return dict(seed=name, ident=ident, cov=cov, score=sc,
                trimmed=cand[max(q0 - t0, 0): min(q1 + (len(s) - t1), len(cand))])


def global_identity(a, b):
    aln = GLOBAL.align(a, b)[0]
    q, t = aln.aligned
    m = sum(x == y for (a0, a1), (b0, b1) in zip(q, t)
            for x, y in zip(a[a0:a1], b[b0:b1]))
    return m / min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panda", required=True)
    ap.add_argument("--pazy", required=True)
    ap.add_argument("--plasticdb", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log = []   # (id, source, status, detail)

    # 1 seeds ---------------------------------------------------------------------
    panda = read_panda(a.panda)
    seeds = [(v["name"], s) for s, v in panda.items() if v["cluster"] == CLUSTER]
    print(f"PANDA: {len(panda)} unique sequences, {len(seeds)} in {CLUSTER}")

    ordered = []   # (name, source, mature_seq)
    for name, s in seeds:
        if not (LEN_LO <= len(s) <= LEN_HI):
            log.append((name, "PANDA", "dropped", f"length {len(s)} outside {LEN_LO}-{LEN_HI}"))
        elif not set(s) <= STD:
            log.append((name, "PANDA", "dropped", "non-standard residues"))
        else:
            ordered.append((name, "PANDA", s))

    # 2-3 candidates -----------------------------------------------------------------
    seen = {s for _, _, s in ordered} | {s for _, s in seeds}
    cands = [(f"PAZy|{h}", "PAZy", s) for h, s in read_fasta(a.pazy)] + \
            [(f"PlasticDB|{h}", "PlasticDB", s) for h, s in read_fasta(a.plasticdb)]
    print(f"candidates: {len(cands)} (PAZy + PlasticDB), aligning to {len(seeds)} seeds ...")
    for name, src, s in cands:
        if s in seen:
            log.append((name, src, "dropped", "exact duplicate"))
            continue
        seen.add(s)
        if not set(s) <= STD:
            log.append((name, src, "dropped", "non-standard residues"))
            continue
        hit = best_seed_hit(s, seeds)
        d = f"best seed {hit['seed']} id={hit['ident']:.2f} cov={hit['cov']:.2f}"
        if hit["ident"] < MIN_ID or hit["cov"] < MIN_COV:
            log.append((name, src, "dropped", "not a PETase-class homolog: " + d))
            continue
        t = hit["trimmed"]
        if not (LEN_LO <= len(t) <= LEN_HI) or not set(t) <= STD:
            log.append((name, src, "dropped", f"trimmed length {len(t)}: " + d))
            continue
        ordered.append((name, src, t))
        log.append((name, src, "candidate", f"trimmed {len(s)}->{len(t)}; " + d))

    # 5 greedy redundancy cut (PANDA first, so validated sequences win) ---------------
    kept = []
    for name, src, s in ordered:
        hi = max((global_identity(s, k[2]) for k in kept), default=0.0)
        if hi > MAX_ID:
            log.append((name, src, "dropped", f"{hi:.2f} identical to a kept sequence"))
            continue
        kept.append((name, src, s))
        log.append((name, src, "KEPT", ""))

    # 6 write ---------------------------------------------------------------------------
    with open(out / "petase_pool.fasta", "w") as f:
        for i, (name, src, s) in enumerate(kept, 1):
            tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60]
            f.write(f">{i:03d}_{tag}\n")
            seq = "M" + s
            f.write("\n".join(seq[j:j + 60] for j in range(0, len(seq), 60)) + "\n")
    with open(out / "petase_pool_report.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "source", "status", "detail"])
        w.writerows(log)

    lens = [len(s) + 1 for _, _, s in kept]
    print(f"\nKEPT {len(kept)} sequences, {min(lens)}-{max(lens)} aa (with Met)")
    print("by source:", dict(collections.Counter(src for _, src, _ in kept)))
    print("dropped by reason:")
    for k, v in collections.Counter(re.split(r"[:0-9]", d)[0].strip() for _, _, st, d in log
                                    if st == "dropped").most_common():
        print(f"   {v:4d}  {k}")
    print(f"wrote {out/'petase_pool.fasta'}  and  {out/'petase_pool_report.tsv'}")
    if len(kept) < 64:
        print(f"NOTE: pool has {len(kept)} < 64 sequences; the notebook skips n=64 automatically "
              f"(it prints 'skipping n=64').")


if __name__ == "__main__":
    main()
