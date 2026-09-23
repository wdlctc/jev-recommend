"""Fine-tune a Jev-style decision scorer on MovieLens-1M Choice records.

Launch with torchrun for data parallelism, e.g.
  torchrun --nproc-per-node 4 -m jevrec.train --mode isolated --out runs/iso
``--epochs 0`` evaluates the zero-shot Yes/No readout.

Loss: soft-max cross-entropy over the K candidates (listwise objective,
whatever the scoring mode) + ``brier_weight`` x multiclass Brier. After
training one temperature is fitted on validation and applied to test.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .data import load, make_records
from .metrics import evaluate, fit_temperature, fmt
from .model import MODES, build
from .prompts import Encoder


def setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, world, f"cuda:{os.environ['LOCAL_RANK']}"
    return 0, 1, "cuda" if torch.cuda.is_available() else "cpu"


def log(rank, *msg):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *msg, flush=True)


@torch.no_grad()
def score(model, encoded, batch_size, rank, world):
    """Score records sharded across ranks; return logits for all records on every rank."""
    model.eval()
    mine = list(range(rank, len(encoded), world))
    out, tokens = [], 0
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for i in range(0, len(mine), batch_size):
        out.append(model([encoded[j] for j in mine[i:i + batch_size]]).float())
        tokens += model.tokens_processed
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    K = len(encoded[0]["cands"])
    logits = torch.cat(out) if out else torch.empty(0, K, device=model.device)
    stats = torch.tensor([tokens, elapsed], device=model.device, dtype=torch.float64)
    if world > 1:
        pad = torch.full((len(range(0, len(encoded), world)), K), float("nan"), device=model.device)
        pad[:len(logits)] = logits
        gathered = [torch.empty_like(pad) for _ in range(world)]
        dist.all_gather(gathered, pad)
        full = torch.empty(len(encoded), K, device=model.device)
        for r, g in enumerate(gathered):
            n = len(range(r, len(encoded), world))
            full[r::world] = g[:n]
        logits = full
        dist.all_reduce(stats)
    model.train()
    return logits.cpu(), {"tokens": stats[0].item(), "gpu_seconds": stats[1].item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--mode", choices=MODES, default="isolated")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--dataset", default="ml-1m", choices=["ml-1m", "ml-latest-small"])
    p.add_argument("--history-len", type=int, default=20)
    p.add_argument("--num-candidates", type=int, default=20)
    p.add_argument("--negatives", default="uniform", choices=["uniform", "popular"])
    p.add_argument("--train-candidates", type=int, default=20)
    p.add_argument("--per-user", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4, help="records per GPU per step")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--brier-weight", type=float, default=0.1)
    p.add_argument("--max-eval-users", type=int, default=None)
    p.add_argument("--max-train-users", type=int, default=None)
    p.add_argument("--init-from", help="trainable.pt of a finished run (LoRA + head); use with --epochs 0")
    p.add_argument("--init-lora-rank", type=int, default=16, help="LoRA rank of --init-from")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rank, world, device = setup()
    torch.manual_seed(args.seed)

    data = load(args.data, args.dataset)
    rank_ = args.init_lora_rank if args.init_from else (args.lora_rank if args.epochs else 0)
    model, tokenizer = build(args.model, args.mode, lora_rank=rank_, device=device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        # listwise-only parameters are absent from isolated/pointwise runs and vice versa.
        missing = [n for n, p_ in model.named_parameters() if p_.requires_grad and n not in state]
        extra = [n for n in state if n not in dict(model.named_parameters())]
        model.load_state_dict(state, strict=False)
        log(rank, f"loaded {len(state)} tensors from {args.init_from}; missing={missing} unused={extra}")
    encode = Encoder(tokenizer, data)
    kw = dict(history_len=args.history_len, negatives=args.negatives)
    evals = {s: [encode(r) for r in make_records(data, s, num_candidates=args.num_candidates,
                                                    max_users=args.max_eval_users, **kw)]
             for s in ("valid", "test")}
    labels = {s: torch.tensor([r["label"] for r in recs]) for s, recs in evals.items()}
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps({**vars(args), "world": world}, indent=2))

    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    head_params = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and not n.startswith("backbone")]
    body_params = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and n.startswith("backbone")]
    log(rank, f"mode={args.mode} world={world} trainable={sum(p_.numel() for p_ in trainable)/1e6:.2f}M")
    history = []
    train_seconds = 0.0
    if args.epochs > 0:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]) \
            if world > 1 else model
        opt = torch.optim.AdamW([{"params": body_params, "lr": args.lr},
                                 {"params": head_params, "lr": args.head_lr}], weight_decay=0.0)
        epoch_records = len(make_records(data, "train", per_user=args.per_user, seed=0,
                                         max_users=args.max_train_users, **kw))
        total_steps = int(args.epochs * epoch_records / (args.batch_size * world))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / max(1, total_steps)))
        log(rank, f"train records/epoch={epoch_records} steps={total_steps}")
        step, epoch, start = 0, 0, time.time()
        model.train()
        while step < total_steps:
            recs = make_records(data, "train", per_user=args.per_user, seed=epoch,
                                num_candidates=args.train_candidates, max_users=args.max_train_users, **kw)
            order = torch.randperm(len(recs), generator=torch.Generator().manual_seed(epoch)).tolist()
            per_step = args.batch_size * world
            for i in range(0, len(order) - per_step + 1, per_step):
                if step >= total_steps:
                    break
                chunk = order[i + rank * args.batch_size:i + (rank + 1) * args.batch_size]
                batch = [encode(recs[j]) for j in chunk]
                target = torch.tensor([b["label"] for b in batch], device=device)
                logits = ddp(batch)
                probs = logits.softmax(-1)
                onehot = torch.nn.functional.one_hot(target, logits.shape[-1]).float()
                loss = torch.nn.functional.cross_entropy(logits, target) + \
                    args.brier_weight * ((probs - onehot) ** 2).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 20 == 0 or step == total_steps:
                    acc = (logits.argmax(-1) == target).float().mean().item()
                    rate = step * per_step / (time.time() - start)
                    log(rank, f"step {step}/{total_steps} loss={loss.item():.4f} acc={acc:.3f} "
                              f"records/s={rate:.1f}")
                    history.append({"step": step, "loss": loss.item(), "acc": acc})
            epoch += 1
        train_seconds = time.time() - start

    valid, _ = score(model, evals["valid"], args.eval_batch_size, rank, world)
    test, cost = score(model, evals["test"], args.eval_batch_size, rank, world)
    if rank == 0:
        temp = fit_temperature(valid, labels["valid"])
        result = {"mode": args.mode, "model": args.model, "epochs": args.epochs, "dataset": args.dataset,
                  "init_from": args.init_from,
                  "test": evaluate(test, labels["test"], temp),
                  "test_uncalibrated": evaluate(test, labels["test"]),
                  "eval_cost": {**cost, "records": len(evals["test"]),
                                "tokens_per_record": cost["tokens"] / len(evals["test"])},
                  "train_seconds": train_seconds, "history": history}
        (out / "results.json").write_text(json.dumps(result, indent=2))
        torch.save({"test_logits": test, "valid_logits": valid}, out / "logits.pt")
        log(rank, "test", fmt(result["test"]), f"T={temp:.3f}",
            f"tokens/record={result['eval_cost']['tokens_per_record']:.0f}")
        if args.epochs > 0:
            state = {n: t.detach().cpu() for n, t in model.named_parameters() if t.requires_grad}
            torch.save(state, out / "trainable.pt")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
