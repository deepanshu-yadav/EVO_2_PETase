#!/usr/bin/env python3
"""Phase 2 -- codon-optimise the PETase pool for E. coli and write the notebook's cache.

Reads  data/petase_pool.fasta           (from phase 1)
Writes data/petase_codons.json          same schema as the notebook's petase_codons.json:
         {family, fasta, organism, model, n_forced_start_met, proteins[], dna[]}
       - proteins[i] is exactly the FASTA sequence i (the notebook asserts this)
       - dna[i] = codons for proteins[i] + TGA stop  (translation is verified)

Run on Colab (GPU optional; CodonTransformer is small and also works on CPU):
    !pip install -q CodonTransformer biopython
    !python phase2_codon_cache.py --fasta data/petase_pool.fasta --out data/petase_codons.json

--dry-run skips the model (uses the most common E. coli codon per residue) so the plumbing
can be tested anywhere. Do NOT use a dry-run cache for the real experiment.
"""
import argparse, json, pathlib
from Bio.Seq import Seq

ORGANISM = "Escherichia coli general"      # same string as the notebook's cache
STOP = "TGA"

# most frequent E. coli codon per residue -- only for --dry-run
_DRY = dict(A="GCG", C="TGC", D="GAT", E="GAA", F="TTT", G="GGC", H="CAT", I="ATT", K="AAA",
            L="CTG", M="ATG", N="AAC", P="CCG", Q="CAG", R="CGT", S="AGC", T="ACC", V="GTG",
            W="TGG", Y="TAT")


def load_pool(path):
    """Identical to the notebook's load_pool: wrapped FASTA, order preserved."""
    seqs, cur = [], None
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs.append(cur)
            cur = ""
        elif cur is not None:
            cur += line.strip()
    if cur:
        seqs.append(cur)
    return [s for s in (x.upper().replace("*", "") for x in seqs) if s]


def make_predictor(dry_run):
    if dry_run:
        return lambda prot: "".join(_DRY[a] for a in prot)
    import torch
    from transformers import AutoTokenizer, BigBirdForMaskedLM
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(device).eval()
    print(f"CodonTransformer loaded on {device}")

    def predict(prot):
        # Don't pass attention_type / deterministic: older CodonTransformer releases lack
        # `deterministic`, newer ones renamed attention_type "original" -> "original_full".
        # Each version's own defaults are valid for it and decode deterministically.
        out = predict_dna_sequence(protein=prot, organism=ORGANISM, device=device,
                                   tokenizer=tok, model=model)
        return out.predicted_dna
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/petase_pool.fasta")
    ap.add_argument("--out", default="data/petase_codons.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    proteins = load_pool(a.fasta)
    assert len(set(proteins)) == len(proteins), "duplicate sequences in pool"
    assert all(p[0] == "M" for p in proteins), "pool sequences should start with M"
    print(f"{len(proteins)} proteins, {min(map(len, proteins))}-{max(map(len, proteins))} aa")

    predict = make_predictor(a.dry_run)
    dna = []
    for i, p in enumerate(proteins, 1):
        d = predict(p).upper()
        if d[-3:] in ("TAA", "TAG", "TGA") and len(d) == 3 * len(p) + 3:
            d = d[:-3]                                # drop any stop the model added
        assert len(d) == 3 * len(p), f"#{i}: DNA length {len(d)} != 3*{len(p)}"
        assert str(Seq(d).translate()) == p, f"#{i}: DNA does not translate back to protein"
        dna.append(d + STOP)
        if i % 20 == 0 or i == len(proteins):
            print(f"  {i}/{len(proteins)}", flush=True)

    # final cross-checks, same invariants the notebook relies on
    assert all(set(d) <= set("ACGT") for d in dna)
    assert all(str(Seq(d).translate()).rstrip("*") == p for d, p in zip(dna, proteins))
    cache = dict(family="petase", fasta=pathlib.Path(a.fasta).name,
                 organism=ORGANISM + (" (DRY RUN - not codon optimised)" if a.dry_run else ""),
                 model="CodonTransformer" if not a.dry_run else "most-frequent-codon",
                 n_forced_start_met=len(proteins), proteins=proteins, dna=dna)
    pathlib.Path(a.out).write_text(json.dumps(cache))
    gc = sum(c in "GC" for d in dna for c in d) / sum(map(len, dna))
    print(f"wrote {a.out}  ({len(dna)} sequences, mean GC {100*gc:.1f}%)")


if __name__ == "__main__":
    main()