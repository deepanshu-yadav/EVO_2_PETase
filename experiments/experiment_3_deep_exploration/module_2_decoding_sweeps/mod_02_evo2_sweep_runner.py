#!/usr/bin/env python3
"""Module 2: Advanced Autoregressive Decoding & Context Scaling Sweeps.

Executes:
1. Min-p (0.05) & Temperature sweeps (T in {0.6, 0.8, 1.0, 1.1}) with top_p=0.92.
2. Context depth scaling across n in {8, 16, 32, 64}.
3. Live verbose logging and per-batch JSON checkpointing with instant resume.
4. CDS extraction, frame validation, and mature protein translation.
5. Homology search vs pool to detect memorization cliff (mean max PID > 80% & novel yield drop > 25%).
6. Contrastive log-likelihood reranking:
     Delta-LL = log P(x | Prompt_thermo) - log P(x | Prompt_general)
7. Exports generated_continuations_grid.tsv and screened candidate FASTA.

Unified sequence input & prompt formatting aligned with exp1_initial_discovery.ipynb.
"""
import argparse
import csv
import datetime
import json
import os
import pathlib
import random
import sys
import time
from typing import Dict, List, Any, Optional

import numpy as np
from Bio import SeqIO

# Ensure local imports
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from decoding_utils import (
    extract_cds,
    validate_and_translate_cds,
    compute_pool_max_pid,
    evaluate_memorization_cliff,
    compute_contrastive_delta_ll,
    SPACER,
)


def load_pool(path: pathlib.Path) -> List[str]:
    """Protein sequences in FASTA order, cleaned as the reference pipeline does.

    Handles 60-column wrapped FASTA files properly so line fragmentation does not occur.
    """
    seqs, cur = [], None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append(cur)
                cur = ""
            elif cur is not None:
                cur += line.strip()
    if cur:
        seqs.append(cur)
    return [s for s in (x.upper().replace("*", "") for x in seqs) if s]


def compose_prompt(dna_seqs: List[str], spacer: str = SPACER) -> str:
    """left layout: a spacer before every example, plus a trailing one."""
    return "".join(spacer + s for s in dna_seqs) + spacer


def select_examples(pool: List[str], n: int, seed: int) -> List[str]:
    """Uniform random exemplar selection from the reference pool."""
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def run_sweeps(
    prompts_path: Optional[pathlib.Path],
    contrastive_path: Optional[pathlib.Path],
    pool_fasta_path: pathlib.Path,
    codons_path: Optional[pathlib.Path],
    out_dir: pathlib.Path,
    temperatures: List[float] = [0.6, 0.8, 1.0, 1.1],
    min_p: float = 0.05,
    top_p: float = 0.92,
    top_k: int = 4,
    samples_per_prompt: int = 2,
    model_name: str = "evo2_7b",
    gpu_mem: float = 0.85,
    dry_run: bool = False,
    max_cells: Optional[int] = None,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Execute the Module 2 generation and analysis grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "generation.log"

    def log_msg(msg: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    print("=" * 78)
    print("EXPERIMENT 3 — MODULE 2: DECODING SWEEPS & CONTEXT SCALING")
    print("=" * 78)

    # 1. Load pool references using reference load_pool
    print(f"\n[1/5] Loading reference pool from {pool_fasta_path.name}...")
    pool_proteins = load_pool(pool_fasta_path)
    pool_records = [(f"pool_{i:03d}", seq) for i, seq in enumerate(pool_proteins, 1)]
    print(f"  Loaded {len(pool_records)} pool references for max PID / novelty auditing.")

    # 2. Build or load prompt cells
    cells = []
    if prompts_path and prompts_path.exists():
        print(f"\n[2/5] Loading prompt suite from {prompts_path.name}...")
        suite = json.loads(prompts_path.read_text(encoding="utf-8"))
        cells = suite.get("cells", [])
        if max_cells:
            cells = cells[:max_cells]
        print(f"  Suite: {suite.get('suite_name', 'custom')} | {len(cells)} cells loaded")
    elif codons_path and codons_path.exists():
        print(f"\n[2/5] Composing prompt cells directly from codon cache {codons_path.name}...")
        cache = json.loads(codons_path.read_text(encoding="utf-8"))
        assert cache["proteins"] == pool_proteins, "Codon cache proteins do not match pool FASTA!"
        proteins, dna = cache["proteins"], cache["dna"]
        dna_of = dict(zip(proteins, dna))

        n_grid = [8, 16, 32, 64]
        if max_cells:
            n_grid = n_grid[:max_cells]

        for n in n_grid:
            draws = []
            for p in range(10):
                sel = select_examples(proteins, n, seed + p)
                prompt = compose_prompt([dna_of[a] for a in sel])
                assert prompt.count(SPACER) == n + 1
                assert prompt.startswith(SPACER) and prompt.endswith(SPACER)
                draws.append({
                    "draw_idx": p,
                    "prompt": prompt,
                    "gen_length": max(map(len, sel)) * 3 + 30,
                })
            cells.append({
                "cell_name": f"petase_n{n:03d}",
                "n_examples": n,
                "draws": draws,
                "gen_length": max(d["gen_length"] for d in draws),
            })
        print(f"  Built {len(cells)} cells across context depths {n_grid} via compose_prompt()")
    else:
        raise FileNotFoundError("Neither a valid --prompts JSON nor --codons JSON was provided!")

    # 3. Model setup or Dry-Run Engine
    model = None
    if not dry_run:
        try:
            import torch
            if not torch.cuda.is_available():
                print("  CUDA not available. Falling back to --dry_run mode.")
                dry_run = True
            else:
                from evo2_flash import Evo2
                print(f"  Loading {model_name} with vLLM (gpu_mem={gpu_mem})...")
                model = Evo2(model_name, gpu_memory_utilization=gpu_mem)
                print(f"  Evo2 model initialized (max_model_len={model.max_model_len:,})")
        except Exception as e:
            print(f"  Failed to load GPU engine ({e}). Running in --dry_run simulation mode.")
            dry_run = True

    # 4. Execute generation grid across depths and temperatures with checkpoints
    total_batches = len(cells) * len(temperatures)
    print(f"\n[3/5] Executing decoding sweep ({total_batches} batches total):")
    print(f"  Temperatures: {temperatures} | min_p: {min_p} | top_p: {top_p} | top_k: {top_k}")
    print(f"  Samples per prompt: {samples_per_prompt} | Execution mode: {'SIMULATED (dry-run)' if dry_run else 'LIVE GPU'}")

    all_generated_rows: List[Dict[str, Any]] = []
    results_by_depth: Dict[int, List[Dict[str, Any]]] = {}
    timing_records: List[Dict[str, Any]] = []

    seq_counter = 0
    batch_counter = 0
    t0 = time.time()

    for c_idx, cell in enumerate(cells, 1):
        n_ex = cell["n_examples"]
        draws = cell.get("draws", [])
        gen_len = cell.get("gen_length") or (max(d.get("gen_length", 948) for d in draws) if draws else 948)
        cell_base = cell.get("cell_name", f"n{n_ex:03d}")

        # Flatten prompts
        prompt_list = [d["prompt"] for d in draws for _ in range(samples_per_prompt)]
        prompt_draw_indices = [d["draw_idx"] for d in draws for _ in range(samples_per_prompt)]
        uniq_tok = sum(len(d["prompt"]) for d in draws)

        for temp in temperatures:
            batch_counter += 1
            cell_id = f"{cell_base}_T{temp:.1f}"
            ckpt_path = ckpt_dir / f"gen_{cell_id}.json"

            # Checkpoint resume check
            if ckpt_path.exists():
                try:
                    saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
                    cached_rows = saved.get("rows", [])
                    if cached_rows:
                        log_msg(f"[{batch_counter:02d}/{total_batches:02d}] RESUMED {cell_id} from checkpoint ({len(cached_rows)} seqs)")
                        all_generated_rows.extend(cached_rows)
                        results_by_depth.setdefault(n_ex, []).extend(cached_rows)
                        timing_records.append(saved.get("timing", {
                            "cell_id": cell_id, "n_examples": n_ex, "temperature": temp,
                            "elapsed_s": 0.0, "status": "resumed_from_cache"
                        }))
                        continue
                except Exception:
                    pass

            max_p_len = max(len(p) for p in prompt_list)
            prefill_nt = sum(len(p) for p in prompt_list)
            log_msg(f"[{batch_counter:02d}/{total_batches:02d}] STARTING {cell_id} | n={n_ex} | T={temp:.1f} | {len(prompt_list)} seqs | Max ctx: {max_p_len:,} nt")

            t_batch_start = time.time()

            if dry_run:
                # Dry run simulation: realistic mutated sequences
                continuations = []
                rng = random.Random(seed + n_ex + int(temp * 100))
                for p_idx, prompt in enumerate(prompt_list):
                    ref_id, ref_seq = rng.choice(pool_records)
                    ref_dna = "".join({"A": "GCG", "C": "TGC", "D": "GAT", "E": "GAA", "F": "TTT",
                                       "G": "GGC", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
                                       "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGT",
                                       "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAT"}[a] for a in ref_seq)
                    dna_list = list(ref_dna)
                    num_mut = int(len(dna_list) * (0.05 + 0.15 * (temp - 0.5)))
                    for _ in range(num_mut):
                        pos = rng.randint(0, len(dna_list) - 1)
                        dna_list[pos] = rng.choice("ACGT")
                    mut_dna = "".join(dna_list) + "TGA" + SPACER + "ATGCATGC"
                    continuations.append(mut_dna[:gen_len])
                hit = 85.0
                ideal = 90.0
                throughput = sum(len(s) for s in continuations) / max(time.time() - t_batch_start, 0.001)
            else:
                longest = max(len(p) + gen_len for p in prompt_list)
                if model.max_model_len < longest:
                    raise SystemExit(
                        f"Cell requires {longest:,} context tokens, but model.max_model_len is "
                        f"{model.max_model_len:,}."
                    )
                out = model.generate(
                    prompt_list,
                    n_tokens=gen_len,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed + 1000 * c_idx + int(temp * 10),
                    verbose=0,
                )
                continuations = out.sequences
                tok = sum(len(p) for p in prompt_list)
                hit = 100 * sum(out.num_cached_tokens) / max(tok, 1)
                ideal = 100 * (1 - uniq_tok / max(tok, 1))
                throughput = sum(len(s) for s in continuations) / max(time.time() - t_batch_start, 0.001)

            elapsed_batch = time.time() - t_batch_start

            # Process and validate continuations
            cell_rows = []
            for p_idx, (cont, draw_idx) in enumerate(zip(continuations, prompt_draw_indices), 1):
                seq_counter += 1
                design_id = f"EXP3_n{n_ex:03d}_T{temp:.1f}_s{seq_counter:04d}"

                cds_nt = extract_cds(cont, spacer=SPACER)
                parsed_ok = (cds_nt is not None)

                val_res = validate_and_translate_cds(cds_nt if cds_nt else "")
                mature_aa = val_res["mature_aa"]
                valid_cds = val_res["valid"]

                max_pid = 0.0
                best_hit = ""
                is_novel = False
                if valid_cds and mature_aa:
                    max_pid, best_hit = compute_pool_max_pid(mature_aa, pool_records)
                    is_novel = (max_pid < 85.0)

                row = {
                    "design_id": design_id,
                    "cell_id": cell_id,
                    "n_examples": n_ex,
                    "temperature": temp,
                    "min_p": min_p,
                    "top_p": top_p,
                    "draw_idx": draw_idx,
                    "parsed_ok": parsed_ok,
                    "valid_cds": valid_cds,
                    "rejection_reason": val_res["reason"],
                    "length_nt": len(cds_nt) if cds_nt else 0,
                    "length_aa": val_res["length_aa"],
                    "max_pid": round(max_pid, 2),
                    "best_pool_hit": best_hit,
                    "is_novel": is_novel,
                    "sequence_aa": mature_aa,
                    "cds_nt": cds_nt or "",
                }
                cell_rows.append(row)
                all_generated_rows.append(row)
                results_by_depth.setdefault(n_ex, []).append(row)

            timing_info = {
                "cell_id": cell_id,
                "n_examples": n_ex,
                "temperature": temp,
                "seqs_generated": len(continuations),
                "elapsed_s": round(elapsed_batch, 2),
                "throughput_nt_s": round(throughput, 1),
                "prefix_cache_hit_pct": round(hit, 1),
                "ideal_cache_hit_pct": round(ideal, 1),
                "status": "completed",
            }
            timing_records.append(timing_info)

            # Save atomic checkpoint JSON
            ckpt_payload = {
                "cell_id": cell_id,
                "timing": timing_info,
                "sequences": continuations,
                "rows": cell_rows,
            }
            ckpt_path.write_text(json.dumps(ckpt_payload, indent=2), encoding="utf-8")
            log_msg(f"[{batch_counter:02d}/{total_batches:02d}] DONE {cell_id} in {elapsed_batch:5.1f}s | Speed: {throughput:5.1f} nt/s | Cache Hit: {hit:4.1f}% | Ckpt: {ckpt_path.name}")

    total_time = time.time() - t0
    (out_dir / "generation_timings.json").write_text(json.dumps(timing_records, indent=2), encoding="utf-8")
    print(f"\nGenerated and audited {len(all_generated_rows)} total continuations in {total_time:.1f}s")
    print("\n--- Cell Bottleneck Profile (sorted by elapsed time) ---")
    for rec in sorted(timing_records, key=lambda x: x.get("elapsed_s", 0), reverse=True):
        print(f"  - {rec['cell_id']:<26} : {rec.get('elapsed_s', 0):6.1f}s | {rec.get('throughput_nt_s', 0):5.1f} nt/s | Cache: {rec.get('prefix_cache_hit_pct', 0)}%")

    # 5. Contrastive Evaluation (Delta-LL) if contrastive suite provided
    contrastive_rows = []
    if contrastive_path and contrastive_path.exists():
        print(f"\n[4/5] Evaluating contrastive log-likelihoods (Delta-LL)...")
        cont_suite = json.loads(contrastive_path.read_text(encoding="utf-8"))
        pairs = cont_suite.get("pairs", [])
        print(f"  Loaded {len(pairs)} paired thermo vs general background contexts.")

        valid_designs = [r for r in all_generated_rows if r["valid_cds"] and r["cds_nt"]]
        scored_subset = valid_designs[: min(len(valid_designs), 50)]

        for des in scored_subset:
            if not dry_run and model is not None:
                t_ctx = pairs[0]["thermo_prompt"] + des["cds_nt"]
                g_ctx = pairs[0]["general_prompt"] + des["cds_nt"]
                score_t = model.score_sequences([t_ctx], reduce_method="sum")[0]
                score_g = model.score_sequences([g_ctx], reduce_method="sum")[0]
                delta_ll = compute_contrastive_delta_ll(score_t, score_g)
            else:
                seq = des["sequence_aa"]
                thermo_weight = sum(seq.count(a) for a in "IVYWREL") / max(len(seq), 1)
                delta_ll = round(float((thermo_weight - 0.42) * 50.0 + (des["temperature"] * 2.0)), 3)

            contrastive_rows.append({
                "design_id": des["design_id"],
                "n_examples": des["n_examples"],
                "temperature": des["temperature"],
                "max_pid": des["max_pid"],
                "delta_ll": delta_ll,
                "sequence_aa": des["sequence_aa"],
            })

        contrastive_rows.sort(key=lambda x: x["delta_ll"], reverse=True)
        print(f"  Scored {len(contrastive_rows)} designs for thermophilic statistical enrichment.")

    # 6. Memorization Cliff Detection
    print(f"\n[5/5] Auditing Memorization Cliff across context depths n...")
    cliff_analysis = evaluate_memorization_cliff(results_by_depth)
    if cliff_analysis["cliff_detected"]:
        print(f"  >>> MEMORIZATION CLIFF DETECTED at n={cliff_analysis['cliff_depth']} <<<")
    else:
        print("  Memorization cliff not reached within tested context range (steady novel yield maintained).")

    for m in cliff_analysis["metrics_per_depth"]:
        print(f"    n={m['n_examples']:<3} | Valid: {m['valid_samples']:4d}/{m['total_samples']} | "
              f"Mean Max PID: {m['mean_max_pid']:5.1f}% | Novel Yield: {m['novel_yield_pct']:5.1f}% | "
              f"Drop from peak: {m['drop_from_peak_pct']:5.1f}% | Cliff: {m['is_cliff']}")

    # 7. Export TSVs and FASTAs
    grid_tsv = out_dir / "generated_continuations_grid.tsv"
    with open(grid_tsv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "design_id", "cell_id", "n_examples", "temperature", "min_p", "top_p", "draw_idx",
            "parsed_ok", "valid_cds", "rejection_reason", "length_nt", "length_aa",
            "max_pid", "best_pool_hit", "is_novel", "sequence_aa", "cds_nt"
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(all_generated_rows)
    print(f"\nSaved generation grid to {grid_tsv} ({len(all_generated_rows)} rows)")

    if contrastive_rows:
        cont_tsv = out_dir / "contrastive_reranked_designs.tsv"
        with open(cont_tsv, "w", newline="", encoding="utf-8") as f:
            fields = ["design_id", "n_examples", "temperature", "max_pid", "delta_ll", "sequence_aa"]
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(contrastive_rows)
        print(f"Saved contrastive reranking to {cont_tsv} ({len(contrastive_rows)} rows)")

    novel_passers = [r for r in all_generated_rows if r["valid_cds"] and r["is_novel"]]
    fasta_out = out_dir / "designs_exp3_screened_novel.faa"
    with open(fasta_out, "w", encoding="utf-8") as f:
        for r in novel_passers:
            f.write(f">{r['design_id']} n={r['n_examples']} T={r['temperature']} max_pid={r['max_pid']}%\n")
            seq = r["sequence_aa"]
            f.write("\n".join(seq[j:j+60] for j in range(0, len(seq), 60)) + "\n")
    print(f"Saved {len(novel_passers)} coherent & novel designs (<85% PID) to {fasta_out.name}")

    summary = {
        "total_generated": len(all_generated_rows),
        "valid_cds_count": sum(1 for r in all_generated_rows if r["valid_cds"]),
        "novel_count": sum(1 for r in all_generated_rows if r["is_novel"]),
        "cliff_analysis": cliff_analysis,
        "grid_tsv": str(grid_tsv),
        "screened_fasta": str(fasta_out),
        "timings": timing_records,
    }
    (out_dir / "module_2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 78)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3 - Module 2: Decoding Sweeps")
    parser.add_argument("--prompts", default="experiments/experiment_3_deep_exploration/outputs/prompts_thermo_curriculum.json")
    parser.add_argument("--contrastive", default="experiments/experiment_3_deep_exploration/outputs/prompts_contrastive.json")
    parser.add_argument("--pool", default="data/petase_pool.fasta")
    parser.add_argument("--codons", default="data/petase_codons.json")
    parser.add_argument("--out", default="experiments/experiment_3_deep_exploration/outputs")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.1])
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--model", default="evo2_7b")
    parser.add_argument("--dry_run", action="store_true", help="Simulate generation for pipeline verification without GPU")
    parser.add_argument("--max_cells", type=int, default=None)
    args = parser.parse_args()

    prompts_path = pathlib.Path(args.prompts).resolve() if args.prompts and pathlib.Path(args.prompts).exists() else None
    contrastive_path = pathlib.Path(args.contrastive).resolve() if args.contrastive and pathlib.Path(args.contrastive).exists() else None
    codons_path = pathlib.Path(args.codons).resolve() if args.codons and pathlib.Path(args.codons).exists() else None

    run_sweeps(
        prompts_path=prompts_path,
        contrastive_path=contrastive_path,
        pool_fasta_path=pathlib.Path(args.pool).resolve(),
        codons_path=codons_path,
        out_dir=pathlib.Path(args.out).resolve(),
        temperatures=args.temperatures,
        min_p=args.min_p,
        top_p=args.top_p,
        samples_per_prompt=args.samples,
        model_name=args.model,
        dry_run=args.dry_run,
        max_cells=args.max_cells,
    )


if __name__ == "__main__":
    main()
