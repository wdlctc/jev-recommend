"""Train per-layer decision heads for early exit (see ``jevrec.exit``).

  # retrofit: freeze a finished isolated run, fit only the intermediate heads
  torchrun --nproc-per-node 2 -m jevrec.train_exit --variant retrofit \
      --init-from runs/q17b/isolated/trainable.pt --out runs/q17b_exit/retrofit
  # joint: LoRA + all heads from scratch, deep supervision
  torchrun --nproc-per-node 4 -m jevrec.train_exit --variant joint --out runs/q17b_exit/joint

Saves per-layer logits for valid/test ([L, N, K]) and per-layer temperatures;
``jevrec.exit_policy`` picks the exit rule on valid and reports test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import load, make_records
from .exit import ExitScorer
from .metrics import evaluate, fit_temperature
from .model import build
from .prompts import Encoder
from .train import log, setup


def layer_loss(logits: torch.Tensor, target: torch.Tensor, brier: float) -> torch.Tensor:
    """CE + brier x multiclass Brier for every layer: [L, B, K] -> [L]."""
    L, B, K = logits.shape
    ce = F.cross_entropy(logits.reshape(L * B, K), target.repeat(L), reduction="none").view(L, B).mean(1)
    onehot = F.one_hot(target, K).float()
    return ce + brier * ((logits.softmax(-1) - onehot) ** 2).sum(-1).mean(1)


@torch.no_grad()
def score_layers(model: ExitScorer, encoded, batch_size, rank, world):
    """[L, N, K] logits for all records on every rank."""
    model.eval()
    mine = list(range(rank, len(encoded), world))
    out = [model([encoded[j] for j in mine[i:i + batch_size]]).float() for i in range(0, len(mine), batch_size)]
    L, K = model.num_layers, len(encoded[0]["cands"])
    logits = torch.cat(out, 1) if out else torch.empty(L, 0, K, device=model.device)
    if world > 1:
        per = len(range(0, len(encoded), world))
        pad = torch.full((L, per, K), float("nan"), device=model.device)
        pad[:, :logits.shape[1]] = logits
        gathered = [torch.empty_like(pad) for _ in range(world)]
        dist.all_gather(gathered, pad)
        full = torch.empty(L, len(encoded), K, device=model.device)
        for r, g in enumerate(gathered):
            full[:, r::world] = g[:, :len(range(r, len(encoded), world))]
        logits = full
    model.train()
    return logits.cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--variant", choices=["retrofit", "joint"], required=True)
    p.add_argument("--init-from", help="trainable.pt of a finished isolated run (required for retrofit)")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--history-len", type=int, default=20)
    p.add_argument("--num-candidates", type=int, default=20)
    p.add_argument("--negatives", default="uniform", choices=["uniform", "popular"])
    p.add_argument("--per-user", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--brier-weight", type=float, default=0.1)
    p.add_argument("--aux-weight", type=float, default=1.0,
                   help="joint: weight of the (depth-weighted) intermediate-layer loss vs the last layer")
    p.add_argument("--max-eval-users", type=int, default=None)
    p.add_argument("--max-train-users", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rank, world, device = setup()
    torch.manual_seed(args.seed)

    data = load(args.data)
    retro = args.variant == "retrofit"
    if retro and not args.init_from:
        raise SystemExit("--variant retrofit needs --init-from")
    scorer, tokenizer = build(args.model, "isolated", lora_rank=args.lora_rank, device=device,
                              gradient_checkpointing=not retro)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        missing, _ = scorer.load_state_dict(state, strict=False)
        log(rank, f"loaded {len(state)} tensors from {args.init_from}")
    if retro:
        scorer.requires_grad_(False)                   # backbone, LoRA and the last head stay frozen
    model = ExitScorer(scorer).to(device)
    L = model.num_layers
    if retro:
        for h in model.heads[:-1]:
            h.requires_grad_(True)
    # Depth weights for the intermediate layers: deeper layers matter more, sum to 1.
    depth = torch.arange(1, L, dtype=torch.float, device=device)
    depth = depth / depth.sum()

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
    body = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and ".backbone." in f".{n}"]
    heads = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and ".backbone." not in f".{n}"]
    log(rank, f"variant={args.variant} layers={L} world={world} "
              f"trainable={sum(p_.numel() for p_ in trainable)/1e6:.2f}M (heads {len(heads)} tensors)")
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]) \
        if world > 1 else model
    groups = [{"params": heads, "lr": args.head_lr}] + ([{"params": body, "lr": args.lr}] if body else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    epoch_records = len(make_records(data, "train", per_user=args.per_user, seed=0,
                                     max_users=args.max_train_users, **kw))
    total = int(args.epochs * epoch_records / (args.batch_size * world))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / max(1, total)))
    log(rank, f"train records/epoch={epoch_records} steps={total}")
    history, step, epoch, start = [], 0, 0, time.time()
    model.train()
    while step < total:
        recs = make_records(data, "train", per_user=args.per_user, seed=epoch,
                            num_candidates=args.num_candidates, max_users=args.max_train_users, **kw)
        order = torch.randperm(len(recs), generator=torch.Generator().manual_seed(epoch)).tolist()
        per_step = args.batch_size * world
        for i in range(0, len(order) - per_step + 1, per_step):
            if step >= total:
                break
            batch = [encode(recs[j]) for j in order[i + rank * args.batch_size:i + (rank + 1) * args.batch_size]]
            target = torch.tensor([b["label"] for b in batch], device=device)
            logits = ddp(batch)                                   # [L, B, K]
            per_layer = layer_loss(logits, target, args.brier_weight)
            aux = (depth * per_layer[:-1]).sum()
            loss = aux if retro else per_layer[-1] + args.aux_weight * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0 or step == total:
                acc = (logits.argmax(-1) == target).float().mean(1)       # [L]
                probe = {f"acc@{l}": round(acc[l].item(), 3) for l in (L // 4, L // 2, 3 * L // 4, L - 1)}
                log(rank, f"step {step}/{total} loss={loss.item():.4f} last={per_layer[-1].item():.4f} "
                          f"{probe} records/s={step * per_step / (time.time() - start):.1f}")
                history.append({"step": step, "loss": loss.item(), **probe})
        epoch += 1
    train_seconds = time.time() - start

    valid = score_layers(model, evals["valid"], args.eval_batch_size, rank, world)
    test = score_layers(model, evals["test"], args.eval_batch_size, rank, world)
    if rank == 0:
        temps = torch.tensor([fit_temperature(valid[l], labels["valid"]) for l in range(L)])
        per_layer = [evaluate(test[l], labels["test"], temps[l].item()) for l in range(L)]
        result = {"variant": args.variant, "model": args.model, "init_from": args.init_from,
                  "temperatures": temps.tolist(), "per_layer_test": per_layer,
                  "train_seconds": train_seconds, "history": history}
        (out / "results.json").write_text(json.dumps(result, indent=2))
        torch.save({"valid": valid.half(), "test": test.half(), "valid_labels": labels["valid"],
                    "test_labels": labels["test"], "temperatures": temps}, out / "layer_logits.pt")
        torch.save({n: t.detach().cpu() for n, t in model.named_parameters(remove_duplicate=False)
                    if n.startswith("heads.") or "lora_" in n}, out / "trainable.pt")
        for l in range(L):
            m = per_layer[l]
            log(rank, f"layer {l:2d} hr@1={m['hr@1']:.4f} ndcg@10={m['ndcg@10']:.4f} "
                      f"nll={m['nll']:.3f} ece={m['ece']:.3f} T={temps[l]:.2f}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
