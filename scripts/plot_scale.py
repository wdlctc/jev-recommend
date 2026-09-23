"""In-domain vs transfer HR@1 for both backbone sizes (runs/q17b, runs/q8b)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures") / "scale.png"
R = Path("runs")
get = lambda p: json.loads((R / p).read_text())  # noqa: E731
series = [  # (label, colour, in-domain HR@1, transfer HR@1)
    ("SASRec (trained on each target)", "#1baf7a", get("q17b/baselines/results.json")["sasrec"]["test"]["hr@1"],
     get("q17b/transfer/baselines/results.json")["sasrec"]["test"]["hr@1"]),
    ("zero-shot Qwen3-1.7B", "#d6d5cf", get("q17b/zeroshot_isolated/results.json")["test"]["hr@1"],
     get("q17b/transfer/zeroshot_isolated/results.json")["test"]["hr@1"]),
    ("zero-shot Qwen3-8B", "#9b9a94", get("q8b/zeroshot_isolated/results.json")["test"]["hr@1"],
     get("q8b/transfer/zeroshot_isolated/results.json")["test"]["hr@1"]),
    ("Jev-style Qwen3-1.7B", "#86b6ef", get("q17b/isolated/results.json")["test"]["hr@1"],
     get("q17b/transfer/isolated/results.json")["test"]["hr@1"]),
    ("Jev-style Qwen3-8B", "#1c5cab", get("q8b/isolated/results.json")["test"]["hr@1"],
     get("q8b/transfer/isolated/results.json")["test"]["hr@1"]),
]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#e4e3df", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.15
for i, (label, color, a, b) in enumerate(series):
    xs = [g + (i - 2) * (w + 0.02) for g in (0, 1)]
    ax.bar(xs, [a, b], w, color=color, label=label)
    for x, v in zip(xs, (a, b)):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color="#52514e")
ax.set_xticks([0, 1], ["ML-1M (in-domain, n=6040)", "ml-latest-small (transfer, n=610)"])
ax.set_ylabel("HR@1 (K=20)")
ax.grid(axis="x", visible=False)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5, frameon=False)
ax.set_title("Backbone scale: in-domain vs. transfer", loc="left", color="#0b0b0b")
fig.savefig(out, dpi=160, bbox_inches="tight")
print(out)
