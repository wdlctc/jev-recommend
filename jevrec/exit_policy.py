"""Pick early-exit rules on validation, report them on test (offline, from saved layer logits).

  python -m jevrec.exit_policy runs/exit/q17b_retrofit

For every accuracy budget (allowed HR@1 drop vs the full network, measured on
valid) the cheapest rule of each family is chosen on valid and applied once to
test. Families:
  static      always stop at layer l (no adaptivity)
  confidence  p_max >= tau_hi
  patience    same argmax for m consecutive layers (optionally p_max >= tau_lo)
  combined    confidence OR patience        (the "settled many times / very sure" rule)
Also reports the oracle: stop at the first layer after which the argmax never
changes again (upper bound for any rule that keeps the final decision).
Cost = mean number of decoder layers run, as a fraction of all layers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from .exit import ExitPolicy, simulate
from .metrics import evaluate

BUDGETS = (0.0, 0.25, 0.5, 1.0, 2.0)   # allowed HR@1 drop in points


def grid(L: int):
    taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    mins = sorted({0, L // 8, L // 4, 3 * L // 8, L // 2})
    fams = {
        "confidence": [ExitPolicy(tau_hi=t, min_layer=m) for t in taus for m in mins],
        "patience": [ExitPolicy(patience=p, tau_lo=t, min_layer=m)
                     for p in range(2, 11) for t in [0.0] + taus[:7] for m in mins],
    }
    fams["combined"] = [ExitPolicy(tau_hi=hi, patience=p, tau_lo=lo, min_layer=m)
                        for hi, p, lo, m in itertools.product(taus[4:], range(2, 9), [0.0] + taus[:5], mins)
                        if lo < hi]
    return fams


def hr1(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    args = p.parse_args()
    run = Path(args.run)
    saved = torch.load(run / "layer_logits.pt")
    V, T = saved["valid"].float(), saved["test"].float()
    yv, yt, temps = saved["valid_labels"], saved["test_labels"], saved["temperatures"]
    L = V.shape[0]
    full_v, full_t = hr1(V[-1], yv), hr1(T[-1], yt)

    def cost(exit_at):
        return (exit_at.float() + 1).mean().item() / L

    # Evaluate every candidate rule once on valid.
    table = {"static": [(ExitPolicy(tau_hi=0.0, min_layer=l), hr1(V[l], yv), (l + 1) / L) for l in range(L)]}
    for fam, pols in grid(L).items():
        rows = []
        for pol in pols:
            chosen, ex = simulate(V, temps, pol)
            rows.append((pol, hr1(chosen, yv), cost(ex)))
        table[fam] = rows

    def report(pol):
        chosen, ex = simulate(T, temps, pol)
        m = evaluate(chosen, yt)
        hist = torch.bincount(ex, minlength=L).tolist()
        return {"policy": pol.as_dict(), "test": {k: m[k] for k in ("hr@1", "hr@5", "ndcg@10", "nll", "ece",
                                                                      "coverage@p95")},
                "layer_fraction": cost(ex), "speedup_layers": 1 / cost(ex), "exit_histogram": hist}

    out = {"layers": L, "full_valid_hr1": full_v, "full_test": evaluate(T[-1] / temps[-1], yt), "budgets": {}}
    for b in BUDGETS:
        out["budgets"][str(b)] = {}
        for fam, rows in table.items():
            ok = [r for r in rows if r[1] >= full_v - b / 100]
            pol, v_hr, v_cost = min(ok, key=lambda r: (r[2], -r[1]))
            out["budgets"][str(b)][fam] = {**report(pol), "valid_hr1": v_hr, "valid_layer_fraction": v_cost}

    # Oracle: first layer after which the argmax equals the final argmax for good.
    arg = T.argmax(-1)                                          # [L, N]
    stable = torch.flip(torch.cumprod(torch.flip((arg == arg[-1]).int(), [0]), 0), [0])
    first = stable.float().argmax(0)
    out["oracle_keep_final"] = {"layer_fraction": cost(first), "speedup_layers": 1 / cost(first),
                                "test_hr1": full_t}
    # Per-layer accuracy curve and Pareto frontier on valid for plotting.
    out["per_layer_test_hr1"] = [hr1(T[l], yt) for l in range(L)]
    out["frontier_valid"] = {}
    for fam, rows in table.items():
        pts = sorted(((c, h) for _, h, c in rows))
        front, best = [], -1.0
        for c, h in pts:
            if h > best:
                front.append((c, h))
                best = h
        out["frontier_valid"][fam] = front
    (run / "exit_policy.json").write_text(json.dumps(out, indent=2))

    print(f"{run}: L={L} full HR@1 valid={full_v:.4f} test={full_t:.4f}; "
          f"oracle speedup {out['oracle_keep_final']['speedup_layers']:.2f}x")
    print(f"{'budget':>6} {'family':>10} {'test HR@1':>9} {'ndcg@10':>8} {'ece':>6} {'layers':>7} {'speedup':>7}  policy")
    for b, fams in out["budgets"].items():
        for fam, r in fams.items():
            print(f"{b:>6} {fam:>10} {r['test']['hr@1']:9.4f} {r['test']['ndcg@10']:8.4f} {r['test']['ece']:6.3f} "
                  f"{r['layer_fraction']:7.3f} {r['speedup_layers']:6.2f}x  {r['policy']}")


if __name__ == "__main__":
    main()
