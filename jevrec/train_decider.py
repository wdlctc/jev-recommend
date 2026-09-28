"""LoRA-train a general typed-decision model with the letters readout.

  torchrun --nproc-per-node 4 -m jevrec.train_decider --model Qwen/Qwen3-4B-Instruct-2507 \
      --data data/decisions --out runs/decider/q4b

The loss is cross-entropy over the option letters only (plus a Brier term), read at
the last prompt token, i.e. exactly what the server returns. After training, one
temperature is fitted on the calibration split (never on JevBench items).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from .decide import LETTERS, gold_index, letters_text


def load(path, tok, max_len):
    rows, skipped = [], 0
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    for line in open(path):
        r = json.loads(line)
        text, options = letters_text(tok, r["state"], r["question"])
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > max_len or len(options) < 2:
            skipped += 1
            continue
        rows.append({"ids": ids, "cand": letter_ids[:len(options)], "gold": gold_index(r, options),
                     "family": r.get("family", "")})
    return rows, skipped


def batch_logits(model, head, batch, device, pad):
    width = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["ids"])] = torch.tensor(b["ids"])
        mask[i, :len(b["ids"])] = 1
    ids, mask = ids.to(device), mask.to(device)
    hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    last = hidden[torch.arange(len(batch), device=device), mask.sum(-1) - 1]
    out = []
    for i, b in enumerate(batch):
        w = head.weight[b["cand"]]                                     # [k, H]
        out.append((last[i:i + 1].float() @ w.float().T).squeeze(0))    # [k]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--data", default="data/decisions")
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=3072)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--brier-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
    log = (lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)) if rank == 0 else (lambda *m: None)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    full = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    head = full.get_output_embeddings()
    head.weight.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    full.requires_grad_(False)
    full = get_peft_model(full, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"]))
    full.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    full.enable_input_require_grads()
    full.to(device)
    body = full.base_model.model.model                                   # the decoder stack (with LoRA)
    train, skipped = load(Path(args.data) / "train.jsonl", tok, args.max_len)
    calib, _ = load(Path(args.data) / "calib.jsonl", tok, args.max_len)
    log(f"train={len(train)} (skipped {skipped} over {args.max_len} tokens) calib={len(calib)} world={world}")
    params = [q for q in full.parameters() if q.requires_grad]
    wrapped = torch.nn.parallel.DistributedDataParallel(body, device_ids=[device]) if ddp else body
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    per_step = args.batch_size * world
    steps = int(args.epochs * len(train) / per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / max(1, steps)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    full.train()
    start, step, epoch = time.time(), 0, 0
    while step < steps:
        order = list(range(len(train)))
        random.Random(epoch).shuffle(order)
        for i in range(0, len(order) - per_step + 1, per_step):
            if step >= steps:
                break
            batch = [train[j] for j in order[i + rank * args.batch_size:i + (rank + 1) * args.batch_size]]
            logits = batch_logits(wrapped, head, batch, device, pad)
            loss = 0.0
            for lg, b in zip(logits, batch):
                target = torch.tensor(b["gold"], device=device)
                probs = lg.softmax(-1)
                onehot = torch.nn.functional.one_hot(target, len(lg)).float()
                loss = loss + torch.nn.functional.cross_entropy(lg[None], target[None]) \
                    + args.brier_weight * ((probs - onehot) ** 2).sum()
            loss = loss / len(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0 or step == steps:
                acc = sum(int(lg.argmax()) == b["gold"] for lg, b in zip(logits, batch)) / len(batch)
                log(f"step {step}/{steps} loss={loss.item():.4f} acc={acc:.2f} rec/s={step * per_step / (time.time() - start):.1f}")
        epoch += 1

    if rank == 0:
        full.eval()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        full.save_pretrained(out / "adapter")
        with torch.no_grad():
            cl = [batch_logits(body, head, [c], device, pad)[0].cpu() for c in calib]
        # Options differ in count, so fit T on per-record NLL directly.
        log_t = torch.zeros((), requires_grad=True)
        lbfgs = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def closure():
            lbfgs.zero_grad()
            nll = sum(torch.nn.functional.cross_entropy((lg / log_t.exp())[None], torch.tensor([c["gold"]]))
                      for lg, c in zip(cl, calib)) / len(calib)
            nll.backward()
            return nll
        lbfgs.step(closure)
        T = log_t.exp().item()
        acc = sum(int(lg.argmax()) == c["gold"] for lg, c in zip(cl, calib)) / len(calib)
        fam = {}
        for lg, c in zip(cl, calib):
            fam.setdefault(c["family"], []).append(int(lg.argmax()) == c["gold"])
        summary = {"model": args.model, "temperature": T, "calib_accuracy": acc, "steps": steps,
                   "train_records": len(train), "calib_by_family": {k: sum(v) / len(v) for k, v in fam.items()},
                   "train_seconds": time.time() - start, "args": vars(args)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"done: calib acc={acc:.3f} T={T:.3f}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
