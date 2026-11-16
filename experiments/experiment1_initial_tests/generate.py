
# Generation stage -- runs in the evo2env venv, not the Colab kernel.
import json, pathlib, time
from evo2_flash import Evo2

WORK = pathlib.Path("work")
cfg = json.loads((WORK / "cells.json").read_text())
cells = cfg["cells"]
longest = max(len(p) + c["gen_length"] for c in cells for p in c["prompts"])

model = Evo2(cfg["model"], gpu_memory_utilization=cfg["gpu_memory_utilization"])
print(f"max_model_len = {model.max_model_len:,}", flush=True)
if model.max_model_len < longest:
    raise SystemExit(
        f"the longest cell needs {longest:,} positions but max_model_len is "
        f"{model.max_model_len:,}. Drop the largest n from a family's grid, or pass "
        f"Evo2(..., kv_cache_dtype='fp8') to double KV capacity.")

t0 = time.time()
for i, c in enumerate(cells, 1):
    t = time.time()
    out = model.generate(
        c["prompts"], n_tokens=c["gen_length"],
        temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=cfg["top_p"],
        seed=cfg["seed"] + 1000 * i,   # a distinct sampling stream per cell
        verbose=0,                     # verbose >= 1 echoes every 54 kb prompt
    )
    (WORK / f"gen_{c['family']}_{c['n_examples']:03d}.json").write_text(
        json.dumps({"sequences": out.sequences}))
    tok = sum(len(p) for p in c["prompts"])
    hit = 100 * sum(out.num_cached_tokens) / max(tok, 1)
    ideal = 100 * (1 - c["uniq_tok"] / max(tok, 1))
    print(f"[{i:>2}/{len(cells)}] {c['family']:>5} n={c['n_examples']:<4} "
          f"{len(c['prompts'])} seqs x {c['gen_length']} nt in {time.time() - t:6.1f}s   "
          f"prefix cache {hit:4.1f}% (ideal {ideal:4.1f}%)", flush=True)
print(f"\ntotal {time.time() - t0:.0f}s", flush=True)
