#!/usr/bin/env python3
"""Aggregate p1-post-baseline jsonl: median/spread per label x prompt."""
import json, statistics, sys, collections

rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    key = (r["label"], r["file"].split("/")[-1])
    rows[key].append(r)

def med(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0: return 0.0
    if n % 2: return vals[n//2]
    return (vals[n//2-1]+vals[n//2])/2

print(f"{'label':26s} {'prompt':16s} {'n':>2s} {'pp_tok':>6s} {'pp_med':>7s} {'pp_min':>7s} {'pp_max':>7s} {'tg_med':>7s} {'tg_min':>7s} {'tg_max':>7s} {'gen_n':>10s}")
for (label, pf) in sorted(rows):
    rs = rows[(label, pf)]
    pp = [r["prompt_per_second"] for r in rs]
    # decode only meaningful when >=64 tokens were actually generated (EOS-short runs pollute tg)
    tg_all = [(r["predicted_per_second"], r["predicted_n"]) for r in rs]
    tg = [t for t, n in tg_all if n >= 64]
    gen = [r["predicted_n"] for r in rs]
    tg_med = med(tg) if tg else float("nan")
    tg_min = min(tg) if tg else float("nan")
    tg_max = max(tg) if tg else float("nan")
    print(f"{label:26s} {pf:16s} {len(rs):2d} {rs[0]['prompt_n']:6d} {med(pp):7.1f} {min(pp):7.1f} {max(pp):7.1f} {tg_med:7.2f} {tg_min:7.2f} {tg_max:7.2f} {str(gen):>10s}")
