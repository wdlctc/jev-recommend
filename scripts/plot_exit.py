"""Figure for the early-exit section: per-layer HR@1 and the valid-chosen test operating points."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

runs = {"Qwen3-1.7B retrofit": "runs/exit/q17b_retrofit", "Qwen3-1.7B joint": "runs/exit/q17b_joint",
        "Qwen3-8B retrofit": "runs/exit/q8b_retrofit"}
colors = {"Qwen3-1.7B retrofit": "#2a6fdb", "Qwen3-1.7B joint": "#8fb3ee", "Qwen3-8B retrofit": "#d9534f"}
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
for name, run in runs.items():
    r = json.loads((Path(run) / "exit_policy.json").read_text())
    curve = r["per_layer_test_hr1"]
    L = len(curve)
    a.plot([(l + 1) / L for l in range(L)], curve, color=colors[name], label=name)
    for fam, marker in (("static", "s"), ("combined", "o")):
        pts = sorted({(v[fam]["layer_fraction"], v[fam]["test"]["hr@1"]) for v in r["budgets"].values()})
        b.plot(*zip(*pts), marker=marker, ls="-" if fam == "combined" else ":", color=colors[name],
               label=f"{name}, {'adaptive' if fam == 'combined' else 'static cut'}")
a.set_xlabel("depth (fraction of layers)"), a.set_ylabel("test HR@1 of that layer's head")
a.set_ylim(0.1, 0.66), a.grid(alpha=.3), a.legend(fontsize=8), a.set_title("decision quality by layer")
b.set_xlabel("mean fraction of layers run"), b.set_ylabel("test HR@1")
b.set_ylim(0.57, 0.64), b.grid(alpha=.3), b.legend(fontsize=7), b.set_title("exit rules chosen on valid, scored on test")
fig.tight_layout()
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "figures/exit.png", dpi=140)
