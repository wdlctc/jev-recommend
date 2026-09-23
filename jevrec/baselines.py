"""Classic ID-based baselines on the exact same candidate sets.

popularity  Score = global training interaction count.
sasrec      Kang & McAuley 2018: causal self-attention over item ids, BCE with
            one sampled negative per position, candidates scored by dot product.
            It sees the same ``history_len`` items as the LLM models.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from .data import load, make_records
from .metrics import evaluate, fit_temperature, fmt


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_len: int, dim: int = 64, layers: int = 2, heads: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.item = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm, self.drop, self.max_len = nn.LayerNorm(dim), nn.Dropout(dropout), max_len

    def encode(self, seq: torch.Tensor) -> torch.Tensor:          # [B, L] left-padded ids
        L = seq.shape[1]
        x = self.item(seq) * self.item.embedding_dim ** 0.5 + self.pos(torch.arange(L, device=seq.device))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        x = self.encoder(self.drop(x), mask=causal, src_key_padding_mask=seq == 0)
        return self.norm(x)

    def score(self, seq: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        h = self.encode(seq)[:, -1]                                   # last position
        return (self.item(cands) * h[:, None]).sum(-1)


def _left_pad(rows, L):
    return torch.tensor([[0] * (L - len(r[-L:])) + r[-L:] for r in rows], dtype=torch.long)


def run_sasrec(data, index, train_seqs, eval_sets, args, device):
    torch.manual_seed(args.seed)
    L = args.history_len
    model = SASRec(len(index), L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_items = len(index)
    # Training windows: every user's sequence (excluding valid/test items), chunked to L+1.
    windows = []
    for seq in train_seqs:
        for end in range(len(seq), 1, -L):
            windows.append(seq[max(0, end - L - 1):end])
    for epoch in range(args.sasrec_epochs):
        model.train()
        random.Random(epoch).shuffle(windows)
        for i in range(0, len(windows), 256):
            chunk = windows[i:i + 256]
            inp = _left_pad([w[:-1] for w in chunk], L).to(device)
            tgt = _left_pad([w[1:] for w in chunk], L).to(device)
            neg = torch.randint(1, num_items + 1, tgt.shape, device=device)
            h = model.encode(inp)
            pos_logit = (model.item(tgt) * h).sum(-1)
            neg_logit = (model.item(neg) * h).sum(-1)
            valid = tgt != 0
            loss = (nn.functional.softplus(-pos_logit) + nn.functional.softplus(neg_logit))[valid].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for name, recs in eval_sets.items():
            logits = []
            for i in range(0, len(recs), 512):
                chunk = recs[i:i + 512]
                seq = _left_pad([[index[m] for m, _ in r["history"]] for r in chunk], L).to(device)
                cands = torch.tensor([[index[m] for m in r["candidates"]] for r in chunk], device=device)
                logits.append(model.score(seq, cands).cpu())
            out[name] = torch.cat(logits)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/baselines")
    p.add_argument("--history-len", type=int, default=20)
    p.add_argument("--num-candidates", type=int, default=20)
    p.add_argument("--sasrec-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(args.data)
    kw = dict(history_len=args.history_len, num_candidates=args.num_candidates)
    sets = {s: make_records(data, s, **kw) for s in ("valid", "test")}
    labels = {s: torch.tensor([r["label"] for r in recs]) for s, recs in sets.items()}
    index = {m: i + 1 for i, m in enumerate(data.items)}
    train_seqs = [[index[m] for m, _ in seq[:-2]] for seq in data.sequences.values()]

    # Popularity from training interactions only.
    counts = {}
    for seq in data.sequences.values():
        for m, _ in seq[:-2]:
            counts[m] = counts.get(m, 0) + 1
    results = {}
    pop = {s: torch.tensor([[float(counts.get(m, 0)) for m in r["candidates"]] for r in recs]).log1p()
           for s, recs in sets.items()}
    t0 = time.time()
    sas = run_sasrec(data, index, train_seqs, sets, args, device)
    sas_time = time.time() - t0
    for name, logits in (("popularity", pop), ("sasrec", sas)):
        temp = fit_temperature(logits["valid"], labels["valid"])
        results[name] = {"test": evaluate(logits["test"], labels["test"], temp),
                         "test_uncalibrated": evaluate(logits["test"], labels["test"])}
        print(name, fmt(results[name]["test"]), flush=True)
    results["sasrec"]["train_seconds"] = sas_time
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results.json").write_text(json.dumps({"args": vars(args), **results}, indent=2))


if __name__ == "__main__":
    main()
