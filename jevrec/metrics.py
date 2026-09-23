"""Ranking and calibration metrics over [N, K] candidate logits."""
from __future__ import annotations

import math

import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single temperature minimising NLL on held-out data (LBFGS on log T)."""
    logits, labels = logits.float(), labels.long()
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def evaluate(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0, bins: int = 15) -> dict:
    logits, labels = logits.float() / temperature, labels.long()
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite logits")
    probs = logits.softmax(-1)
    N, K = probs.shape
    # Rank of the positive (0 = top); ties count against the model.
    pos = logits.gather(1, labels[:, None])
    rank = (logits >= pos).sum(1) - 1
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.float().mean().item() * abs(conf[sel].mean().item() - correct[sel].mean().item())
    onehot = torch.nn.functional.one_hot(labels, K).float()
    out = {
        "n": N, "K": K,
        "hr@1": (rank < 1).float().mean().item(),
        "hr@5": (rank < 5).float().mean().item(),
        "hr@10": (rank < 10).float().mean().item(),
        "ndcg@10": torch.where(rank < 10, 1 / torch.log2(rank.float() + 2), torch.zeros(())).mean().item(),
        "mrr": (1 / (rank.float() + 1)).mean().item(),
        "nll": torch.nn.functional.cross_entropy(logits, labels).item(),
        "brier": ((probs - onehot) ** 2).sum(-1).mean().item(),
        "ece": ece,
        "temperature": temperature,
    }
    # Auto-decide at a 95% precision target: how many decisions can be automated?
    order = conf.argsort(descending=True)
    hits = correct[order].cumsum(0) / torch.arange(1, N + 1)
    ok = (hits >= 0.95).nonzero()
    out["coverage@p95"] = (ok.max().item() + 1) / N if len(ok) else 0.0
    return out


def fmt(metrics: dict) -> str:
    keys = ["hr@1", "hr@5", "ndcg@10", "mrr", "nll", "brier", "ece", "coverage@p95"]
    return " ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics and not math.isnan(metrics[k]))
