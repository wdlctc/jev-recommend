"""Slice analysis + figures from saved logits.

  python scripts/analyze.py runs/q17b figures/

Outputs <run>/analysis.json and PNG figures:
  latency.png       per-request latency vs K for the three scoring modes
  popularity.png    HR@1 by train-popularity bucket of the positive item
  reliability.png   top-1 confidence vs accuracy (calibrated)
  transfer.png      ML-1M-trained scorers on ml-latest-small, seen vs new items
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevrec.data import load, make_records  # noqa: E402
from jevrec.metrics import evaluate  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/q17b")
FIG = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
FIG.mkdir(parents=True, exist_ok=True)

# Fixed identity colours (reference categorical palette, light mode).
C = {"isolated": "#2a78d6", "listwise": "#eb6834", "sasrec": "#1baf7a", "pointwise": "#4a3aa7",
     "zeroshot": "#9b9a94", "popularity": "#c9c8c2"}
LABEL = {"isolated": "Jev-style, isolated", "listwise": "Jev-style, listwise", "sasrec": "SASRec (item IDs)",
         "pointwise": "pointwise (Open-Jev)", "zeroshot": "zero-shot LLM", "popularity": "popularity"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 160, "savefig.bbox": "tight"})


def run_logits(root):
    out = {}
    base = root / "baselines" / "logits.pt"
    if base.exists():
        b = torch.load(base)
        out["popularity"], out["sasrec"] = b["popularity"], b["sasrec"]
    for key, name in (("zeroshot", "zeroshot_isolated"), ("isolated", "isolated"), ("listwise", "listwise")):
        f = root / name / "logits.pt"
        if f.exists():
            t = torch.load(f)
            out[key] = {"valid": t["valid_logits"], "test": t["test_logits"]}
    return out


def temps(root):
    t = {}
    b = root / "baselines" / "results.json"
    if b.exists():
        r = json.loads(b.read_text())
        t.update(popularity=r["popularity"]["test"]["temperature"], sasrec=r["sasrec"]["test"]["temperature"])
    for key, name in (("zeroshot", "zeroshot_isolated"), ("isolated", "isolated"), ("listwise", "listwise")):
        f = root / name / "results.json"
        if f.exists():
            t[key] = json.loads(f.read_text())["test"]["temperature"]
    return t


def bars(ax, groups, series, values, ylabel):
    width = min(0.8 / len(series), 0.22)
    for i, s in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * (width + 0.02) for g in range(len(groups))]
        ax.bar(xs, values[s], width, color=C[s], label=LABEL[s])
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


analysis = {}

# ---- popularity buckets on ML-1M --------------------------------------------------------------
ml = load("data", "ml-1m")
test = make_records(ml, "test")
labels = torch.tensor([r["label"] for r in test])
counts = {}
for seq in ml.sequences.values():
    for m, _ in seq[:-2]:
        counts[m] = counts.get(m, 0) + 1
pos_pop = torch.tensor([counts.get(r["candidates"][r["label"]], 0) for r in test])
edges = [0, 50, 200, 500, 1000, 10 ** 9]
names = ["<50", "50-200", "200-500", "500-1k", ">=1k"]
L = run_logits(RUN)
T = temps(RUN)
series = [s for s in ("popularity", "sasrec", "zeroshot", "isolated", "listwise") if s in L]
buckets = {s: [] for s in series}
sizes = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (pos_pop >= lo) & (pos_pop < hi)
    sizes.append(int(sel.sum()))
    for s in series:
        m = evaluate(L[s]["test"][sel], labels[sel], T[s])
        buckets[s].append(m["hr@1"])
analysis["popularity_buckets"] = {"edges": edges, "names": names, "n": sizes, "hr@1": buckets}
if series:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars(ax, [f"{n}\n(n={k})" for n, k in zip(names, sizes)], series, buckets, "HR@1 (K=20)")
    ax.set_xlabel("training interactions of the positive movie")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=8.5)
    ax.set_title("HR@1 by training popularity of the positive movie", loc="left", color=INK)
    fig.savefig(FIG / "popularity.png")
    plt.close(fig)

# ---- reliability -------------------------------------------------------------------------------
rel = {}
fig, ax = plt.subplots(figsize=(4.2, 4.0))
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
for s in [s for s in ("sasrec", "isolated", "listwise") if s in L]:
    p = (L[s]["test"].float() / T[s]).softmax(-1)
    conf, pred = p.max(-1)
    correct = (pred == labels).float()
    xs, ys = [], []
    for lo in torch.linspace(0, 0.9, 10):
        sel = (conf > lo) & (conf <= lo + 0.1)
        if sel.sum() >= 30:
            xs.append(conf[sel].mean().item()), ys.append(correct[sel].mean().item())
    rel[s] = {"conf": xs, "acc": ys}
    ax.plot(xs, ys, color=C[s], linewidth=2, marker="o", markersize=4.5, markeredgecolor="white",
            markeredgewidth=1.5, label=LABEL[s])
ax.set_xlabel("predicted top-1 probability")
ax.set_ylabel("observed accuracy")
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Calibration (temperature-scaled)", loc="left", color=INK)
fig.savefig(FIG / "reliability.png")
plt.close(fig)
analysis["reliability"] = rel

# ---- latency -----------------------------------------------------------------------------------
bench = RUN / "bench.json"
if bench.exists():
    b = json.loads(bench.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for mode in ("pointwise", "isolated", "listwise"):
        rows = sorted((r for r in b["rows"] if r["mode"] == mode), key=lambda r: r["K"])
        ks = [r["K"] for r in rows]
        for ax, key in zip(axes, ("min_ms" if "min_ms" in rows[0] else "p50_ms", "tokens")):
            ax.plot(ks, [r[key] for r in rows], color=C[mode], linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=1.5, label=LABEL[mode])
    lat = "min latency of 60 requests (ms)" if "min_ms" in b["rows"][0] else "p50 latency per request (ms)"
    for ax, lab in zip(axes, (lat, "tokens processed per request")):
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel("candidates K"), ax.set_ylabel(lab)
        ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle(f"One request, batch 1, {b['gpu']}", x=0.01, ha="left", color=INK, fontsize=10)
    fig.savefig(FIG / "latency.png")
    plt.close(fig)
    analysis["latency"] = b["rows"]

# ---- transfer to ml-latest-small ---------------------------------------------------------------
TR = RUN / "transfer"
if (TR / "isolated" / "logits.pt").exists():
    small = load("data", "ml-latest-small")
    stest = make_records(small, "test")
    slabels = torch.tensor([r["label"] for r in stest])
    in_ml1m = torch.tensor([r["candidates"][r["label"]] in ml.titles for r in stest])
    TL, TT = run_logits(TR), temps(TR)
    tseries = [s for s in ("popularity", "sasrec", "zeroshot", "isolated", "listwise") if s in TL]
    groups = {"all": torch.ones_like(in_ml1m), "movie in ML-1M": in_ml1m, "movie new since ML-1M": ~in_ml1m}
    vals = {s: [evaluate(TL[s]["test"][g], slabels[g], TT[s])["hr@1"] for g in groups.values()] for s in tseries}
    analysis["transfer"] = {"groups": list(groups), "n": [int(g.sum()) for g in groups.values()], "hr@1": vals,
                            "full": {s: evaluate(TL[s]["test"], slabels, TT[s]) for s in tseries}}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    LABEL.update(sasrec="SASRec (trained on target)", popularity="popularity (target)",
                 isolated="Jev-style isolated (ML-1M only)", listwise="Jev-style listwise (ML-1M only)")
    bars(ax, [f"{g}\n(n={int(m.sum())})" for g, m in groups.items()], tseries, vals, "HR@1 (K=20)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5)
    ax.set_title("Downstream: ML-1M-trained scorer on ml-latest-small (no retraining)", loc="left", color=INK)
    fig.savefig(FIG / "transfer.png")
    plt.close(fig)

(RUN / "analysis.json").write_text(json.dumps(analysis, indent=2))
print(json.dumps({k: v for k, v in analysis.items() if k != "latency"}, indent=1)[:3000])
