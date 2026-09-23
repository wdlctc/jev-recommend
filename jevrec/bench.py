"""Per-request latency of the three scoring modes (batch = 1 request, warmed).

Sweeps the number of candidates K: pointwise re-encodes the state K times,
isolated/listwise encode it once. All modes share one backbone and are timed
interleaved request by request (rotating the order), so drift in host load
on a shared machine hits every mode equally.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .data import load, make_records
from .model import MODES, DecisionScorer, build
from .prompts import Encoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/bench.json")
    p.add_argument("--candidates", default="5,10,20,50,100")
    p.add_argument("--history-len", type=int, default=20)
    p.add_argument("--requests", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    data = load(args.data)
    base, tok = build(args.model, "isolated", lora_rank=0, device="cuda")
    hidden = base.head.weight.shape[1]
    scorers = {m: DecisionScorer(base.backbone, hidden, base.head.weight.detach()[0], mode=m,
                                 pad_id=base.pad_id).to("cuda").eval() for m in MODES}
    encode = Encoder(tok, data)
    rows = []
    for K in map(int, args.candidates.split(",")):
        recs = [encode(r) for r in make_records(data, "test", num_candidates=K, history_len=args.history_len,
                                                max_users=args.requests + args.warmup)]
        times = {m: [] for m in MODES}
        tokens = {m: [] for m in MODES}
        with torch.inference_mode():
            for i, rec in enumerate(recs):
                order = MODES[i % 3:] + MODES[:i % 3]
                for mode in order:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    scorers[mode]([rec])
                    torch.cuda.synchronize()
                    if i >= args.warmup:
                        times[mode].append((time.perf_counter() - t) * 1000)
                        tokens[mode].append(scorers[mode].tokens_processed)
        for mode in MODES:
            ts = sorted(times[mode])
            row = {"mode": mode, "K": K, "p50_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * (len(ts) - 1))],
                   "min_ms": ts[0], "tokens": statistics.mean(tokens[mode])}
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "gpu": torch.cuda.get_device_name(),
                                          "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
