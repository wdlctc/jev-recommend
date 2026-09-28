"""Wall-clock of real early exit vs the full network, on the same GPU and weights.

  python -m jevrec.bench_exit runs/exit/q17b_retrofit --budget 0.5 --family combined

The full network is the unmodified isolated scorer (HF forward). Early exit is
``ExitScorer.decide`` with the rule that ``jevrec.exit_policy`` chose on valid;
requests that settle are dropped from the batch. Each batch is timed for both
paths back to back, alternating which goes first; also checks that the exit
layers and HR@1 match the offline simulation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .data import load, make_records
from .exit import ExitPolicy, ExitScorer, simulate
from .model import build
from .prompts import Encoder


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--budget", default="0.5")
    p.add_argument("--family", default="combined")
    p.add_argument("--batch-sizes", default="1,16,64")
    p.add_argument("--requests", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--no-cudnn-sdpa", action="store_true",
                   help="cuDNN attention builds a plan per new (batch, length) shape; exit makes many shapes")
    args = p.parse_args()
    if args.no_cudnn_sdpa:
        torch.backends.cuda.enable_cudnn_sdp(False)
    run = Path(args.run)
    cfg = json.loads((run / "args.json").read_text())
    chosen = json.loads((run / "exit_policy.json").read_text())["budgets"][args.budget][args.family]
    policy = ExitPolicy(**chosen["policy"])
    saved = torch.load(run / "layer_logits.pt")
    temps = saved["temperatures"].cuda()

    data = load(cfg.get("data", "data"))
    scorer, tok = build(cfg["model"], "isolated", lora_rank=cfg["lora_rank"], device="cuda",
                        gradient_checkpointing=False)
    model = ExitScorer(scorer).cuda().eval()
    state = torch.load(run / "trainable.pt")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and not [m for m in missing if "lora_" in m or m.startswith("heads.")], (missing, unexpected)
    if hasattr(scorer.backbone, "merge_and_unload"):   # serve with merged LoRA, like production
        scorer.backbone = scorer.backbone.merge_and_unload()
        model.base = scorer.backbone
    encode = Encoder(tok, data)
    recs = [encode(r) for r in make_records(data, "test", num_candidates=cfg["num_candidates"],
                                            history_len=cfg["history_len"], negatives=cfg["negatives"],
                                            max_users=args.requests)]
    labels = torch.tensor([r["label"] for r in recs])
    sim_logits, sim_exit = simulate(saved["test"][:, :len(recs)].float(), saved["temperatures"], policy)

    rows = []
    with torch.inference_mode():
        for bs in map(int, args.batch_sizes.split(",")):
            t_full, t_exit, exits, hits_exit, hits_full = [], [], [], [], []
            batches = [recs[i:i + bs] for i in range(0, len(recs) - bs + 1, bs)]
            for i, batch in enumerate(batches):
                runs = [("full", lambda b=batch: scorer(b)), ("exit", lambda b=batch: model.decide(b, policy, temps))]
                for name, fn in (runs if i % 2 == 0 else runs[::-1]):
                    out, ms = timed(fn)
                    if i < args.warmup:
                        continue
                    if name == "full":
                        t_full.append(ms)
                        hits_full.append(out.argmax(-1).cpu())
                    else:
                        t_exit.append(ms)
                        exits.append(out[1].cpu())
                        hits_exit.append(out[0].argmax(-1).cpu())
            n0 = args.warmup * bs
            y = labels[n0:n0 + len(t_full) * bs]
            ex = torch.cat(exits)
            row = {"batch": bs, "requests": len(y),
                   "full_ms_per_batch": statistics.median(t_full), "exit_ms_per_batch": statistics.median(t_exit),
                   "full_total_s": sum(t_full) / 1000, "exit_total_s": sum(t_exit) / 1000,
                   "hr1_full": (torch.cat(hits_full) == y).float().mean().item(),
                   "hr1_exit": (torch.cat(hits_exit) == y).float().mean().item(),
                   "mean_layers": ex.float().add(1).mean().item(),
                   "exit_layers_match_sim": (ex == sim_exit[n0:n0 + len(ex)]).float().mean().item()}
            row["speedup_total"] = row["full_total_s"] / row["exit_total_s"]
            row["speedup_median"] = row["full_ms_per_batch"] / row["exit_ms_per_batch"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    res = {"run": str(run), "gpu": torch.cuda.get_device_name(), "policy": policy.as_dict(),
           "budget": args.budget, "family": args.family, "layers": model.num_layers, "rows": rows}
    res["cudnn_sdpa"] = not args.no_cudnn_sdpa
    (run / f"bench_{args.family}_{args.budget}{'_nocudnn' if args.no_cudnn_sdpa else ''}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
