"""Per-request latency of the three scoring modes (batch = 1 request, warmed).

Sweeps the number of candidates K to show how cost grows: pointwise
re-encodes the state K times, isolated/listwise encode it once.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .data import load, make_records
from .model import MODES, build
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
    rows = []
    for mode in MODES:
        model, tok = build(args.model, mode, lora_rank=0, device="cuda")
        model.eval()
        encode = Encoder(tok, data)
        for K in map(int, args.candidates.split(",")):
            recs = [encode(r) for r in make_records(data, "test", num_candidates=K, history_len=args.history_len,
                                                    max_users=args.requests + args.warmup)]
            times, tokens = [], []
            with torch.inference_mode():
                for i, rec in enumerate(recs):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    model([rec])
                    torch.cuda.synchronize()
                    if i >= args.warmup:
                        times.append((time.perf_counter() - t) * 1000)
                        tokens.append(model.tokens_processed)
            row = {"mode": mode, "K": K, "p50_ms": statistics.median(times),
                   "p95_ms": sorted(times)[int(0.95 * (len(times) - 1))], "tokens": statistics.mean(tokens)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "gpu": torch.cuda.get_device_name(),
                                          "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
