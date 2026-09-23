"""Collect runs/<tag>/*/results.json + bench.json into markdown tables."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/q17b")
rows = []
base = root / "baselines" / "results.json"
if base.exists():
    b = json.loads(base.read_text())
    for name in ("popularity", "sasrec"):
        rows.append((name, b[name]["test"], None))
order = ["zeroshot_pointwise", "zeroshot_isolated", "isolated", "pointwise_from_isolated", "listwise"]
for name in order:
    f = root / name / "results.json"
    if f.exists():
        r = json.loads(f.read_text())
        rows.append((name, r["test"], r["eval_cost"]))

cols = ["hr@1", "hr@5", "ndcg@10", "mrr", "nll", "brier", "ece", "coverage@p95"]
print("| run | " + " | ".join(cols) + " | tokens/record | GPU-s / 1k records |")
print("|---" * (len(cols) + 3) + "|")
for name, m, cost in rows:
    extra = f"{cost['tokens_per_record']:.0f} | {1000 * cost['gpu_seconds'] / cost['records']:.1f}" if cost else "– | –"
    print(f"| {name} | " + " | ".join(f"{m[c]:.4f}" for c in cols) + f" | {extra} |")

bench = root / "bench.json"
if bench.exists():
    b = json.loads(bench.read_text())
    print(f"\nLatency, batch = 1 request, {b['gpu']}, {b['model']}\n")
    ks = sorted({r["K"] for r in b["rows"]})
    print("| mode | " + " | ".join(f"K={k} p50 ms (tokens)" for k in ks) + " |")
    print("|---" * (len(ks) + 1) + "|")
    for mode in ("pointwise", "isolated", "listwise"):
        cells = {r["K"]: r for r in b["rows"] if r["mode"] == mode}
        print(f"| {mode} | " + " | ".join(f"{cells[k]['p50_ms']:.1f} ({cells[k]['tokens']:.0f})" for k in ks) + " |")
