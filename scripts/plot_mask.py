"""Draw the actual block masks produced by jevrec.model for a toy request."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from jevrec.model import block_mask, pack_setwise  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures") / "mask.png"
rec = {"state": [1, 2, 3, 4], "cands": [[5, 6], [7, 8], [9, 10]], "decide": [11, 12]}
labels = ["s"] * 4 + ["c1"] * 2 + ["c2"] * 2 + ["c3"] * 2 + ["d"] * 2
fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
for ax, (title, with_decide) in zip(axes, (("isolated", False), ("listwise (+ decide)", True))):
    ids, pos, seg, _, _ = pack_setwise([rec], with_decide, 0)
    allowed = (block_mask(seg, torch.float32)[0, 0] == 0).float().numpy()
    n = allowed.shape[0]
    ax.imshow(allowed, cmap=ListedColormap(["#f0efec", "#2a78d6"]), vmin=0, vmax=1)
    ax.set_xticks(range(n), labels[:n], fontsize=8)
    ax.set_yticks(range(n), [f"{l} p{p}" for l, p in zip(labels[:n], pos[0].tolist())], fontsize=8)
    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("key", color="#52514e"), ax.set_ylabel("query (segment, position id)", color="#52514e")
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
fig.savefig(out, dpi=160, bbox_inches="tight")
print(out)
