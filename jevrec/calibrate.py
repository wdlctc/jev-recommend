"""Fit one temperature for a DecisionEngine on the calibration split (never on JevBench items).

  python -m jevrec.calibrate --model Qwen/Qwen3-4B-Instruct-2507 [--adapter DIR] --data data/decisions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .decide import DecisionEngine, gold_index, options_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--readout", default="letters")
    p.add_argument("--data", default="data/decisions")
    p.add_argument("--out", required=True)
    p.add_argument("--max-chars", type=int, default=12000)
    args = p.parse_args()
    eng = DecisionEngine(args.model, args.readout, adapter=args.adapter)
    logits, golds, fams = [], [], []
    for line in open(Path(args.data) / "calib.jsonl"):
        r = json.loads(line)
        if len(r["state"]) > args.max_chars:
            continue
        _, lg, _ = eng.logits(r["state"], r["question"])
        logits.append(lg.cpu())
        golds.append(gold_index(r, options_of(r["question"])))
        fams.append(r["family"])
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        nll = sum(torch.nn.functional.cross_entropy((lg / log_t.exp())[None], torch.tensor([g]))
                  for lg, g in zip(logits, golds)) / len(golds)
        nll.backward()
        return nll
    opt.step(closure)
    T = log_t.exp().item()
    correct = [int(lg.argmax()) == g for lg, g in zip(logits, golds)]
    by = {}
    for f, c in zip(fams, correct):
        by.setdefault(f, []).append(c)
    res = {"model": args.model, "adapter": args.adapter, "temperature": T, "n": len(golds),
           "accuracy": sum(correct) / len(correct), "by_family": {k: sum(v) / len(v) for k, v in by.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
