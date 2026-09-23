"""Build demo/data.json for the Jev Decision Desk from saved runs.

Run from the repo root: PYTHONPATH=. python demo/extract_demo_data.py
The page (demo/index.html) inlines this JSON.
"""
import json, random, torch
from pathlib import Path
from jevrec.data import load, make_records
R = Path("runs")
ml, small = load("data", "ml-1m"), load("data", "ml-latest-small")

def temp(path, key=None):
    r = json.loads((R / path).read_text())
    return r[key]["test"]["temperature"] if key else r["test"]["temperature"]

def lg(path, key=None):
    t = torch.load(R / path)
    return t[key]["test"] if key else t["test_logits"]

SC = {
 "uniform": dict(data=ml, neg="uniform", title="MovieLens-1M", models={
   "jev8b": ("q8b/isolated/logits.pt", None, "q8b/isolated/results.json", None),
   "jev17b": ("q17b/isolated/logits.pt", None, "q17b/isolated/results.json", None),
   "sasrec": ("q17b/baselines/logits.pt", "sasrec", "q17b/baselines/results.json", "sasrec"),
   "zs8b": ("q8b/zeroshot_isolated/logits.pt", None, "q8b/zeroshot_isolated/results.json", None)}),
 "hard": dict(data=ml, neg="popular", title="Hard negatives", models={
   "jev17b": ("q17b_hard/isolated/logits.pt", None, "q17b_hard/isolated/results.json", None),
   "sasrec": ("q17b_hard/baselines/logits.pt", "sasrec", "q17b_hard/baselines/results.json", "sasrec"),
   "zs17b": ("q17b_hard/zeroshot_isolated/logits.pt", None, "q17b_hard/zeroshot_isolated/results.json", None)}),
 "transfer": dict(data=small, neg="uniform", title="New catalogue", models={
   "jev8b": ("q8b/transfer/isolated/logits.pt", None, "q8b/transfer/isolated/results.json", None),
   "jev17b": ("q17b/transfer/isolated/logits.pt", None, "q17b/transfer/isolated/results.json", None),
   "sasrec": ("q17b/transfer/baselines/logits.pt", "sasrec", "q17b/transfer/baselines/results.json", "sasrec"),
   "zs8b": ("q8b/transfer/zeroshot_isolated/logits.pt", None, "q8b/transfer/zeroshot_isolated/results.json", None)}),
}
out = {"scenarios": {}}
for name, sc in SC.items():
    d = sc["data"]
    recs = make_records(d, "test", negatives=sc["neg"])
    y = torch.tensor([r["label"] for r in recs])
    P, meta = {}, {}
    for m, (lp, lk, rp, rk) in sc["models"].items():
        T = temp(rp, rk)
        p = (lg(lp, lk).float() / T).softmax(-1)
        P[m] = p
        conf, pred = p.max(-1)
        corr = (pred == y).float()
        order = conf.argsort(descending=True)
        prec = corr[order].cumsum(0) / torch.arange(1, len(y) + 1)
        ok = (prec >= 0.95).nonzero()
        thr = conf[order][ok.max()].item() if len(ok) else 1.01
        meta[m] = {"hr1": corr.mean().item(), "thr95": thr, "auto": ((conf >= thr).float().mean().item() if len(ok) else 0.0),
                   "T": T}
    jev = "jev8b" if "jev8b" in P else "jev17b"
    jc = P[jev].argmax(-1) == y
    sc_ = P["sasrec"].argmax(-1) == y
    groups = {"jev_only": jc & ~sc_, "sasrec_only": ~jc & sc_, "both": jc & sc_, "neither": ~jc & ~sc_}
    rng = random.Random(7)
    quota = {"jev_only": 7, "sasrec_only": 5, "both": 6, "neither": 4}
    chosen = []
    jconf = P[jev].max(-1).values
    auto = jconf >= meta[jev]["thr95"]
    for g, mask in groups.items():
        idx = mask.nonzero().flatten().tolist()
        if g == "jev_only":  # 2 of them drawn from decisions the scorer would automate
            sure = (mask & auto).nonzero().flatten().tolist()
            pick = rng.sample(sure, min(2, len(sure)))
            rest = [i for i in idx if i not in set(pick)]
            pick += rng.sample(rest, quota[g] - len(pick))
        else:
            pick = rng.sample(idx, min(quota[g], len(idx)))
        chosen += [(i, g) for i in pick]
    rng.shuffle(chosen)
    ex = []
    for i, g in chosen:
        r = recs[i]
        ex.append({"user": r["user"], "group": g, "label": r["label"],
                   "history": [[d.titles[m], d.genres[m], rt] for m, rt in r["history"]],
                   "cands": [[d.titles[m], d.genres[m], (m not in ml.titles) if name == "transfer" else False] for m in r["candidates"]],
                   "probs": {m: [round(x, 4) for x in P[m][i].tolist()] for m in P}})
    out["scenarios"][name] = {"title": sc["title"], "n": len(recs), "jev": jev, "models": meta,
                              "group_sizes": {g: int(m.sum()) for g, m in groups.items()}, "examples": ex}
b = json.loads((R / "q17b/bench.json").read_text())
out["bench"] = [{k: r[k] for k in ("mode", "K", "min_ms", "p50_ms", "tokens")} for r in b["rows"]]
p = "demo/data.json"
Path(p).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
print(len(Path(p).read_text()) // 1024, "KB")
for n, s in out["scenarios"].items():
    print(n, s["n"], s["group_sizes"], {m: (round(v["hr1"], 3), round(v["thr95"], 3), round(v["auto"], 3)) for m, v in s["models"].items()})
