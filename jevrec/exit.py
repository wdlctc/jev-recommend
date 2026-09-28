"""Jev-Exit: a decision readout after every layer, stop as soon as the decision is settled.

The isolated scorer reads one scalar per candidate from the last layer. Here
every decoder layer l gets its own readout ``head_l(norm(h_l))`` (the backbone's
final RMSNorm, frozen, then a linear head initialised like the last one), so
each layer emits a full distribution over the K candidates. At inference a
request leaves the network at the first layer where the decision is settled:

    p_max(l) >= tau_hi                                  (one very confident read)
 or streak(l) >= m  and  p_max(l) >= tau_lo              (same argmax m layers in a row)

Probabilities use a per-layer temperature fitted on validation. The decision
covers the whole Choice (all K candidates), so a request exits as a unit and
the batch simply shrinks: no KV cache to patch, unlike token-level early exit
in generation.

Two ways to get the heads:
  retrofit  load a finished isolated run, freeze everything, fit only the
            intermediate heads; the last layer is bit-identical to the original.
  joint     LoRA + all heads from scratch with deep supervision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import PAD, DecisionScorer, block_mask, pack_setwise


def base_model(backbone: nn.Module) -> nn.Module:
    return backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone


@dataclass(frozen=True)
class ExitPolicy:
    tau_hi: float = 1.01        # > 1 disables the single-read rule
    patience: int = 99          # consecutive layers with the same argmax
    tau_lo: float = 0.0
    min_layer: int = 0

    def as_dict(self):
        return {"tau_hi": self.tau_hi, "patience": self.patience, "tau_lo": self.tau_lo,
                "min_layer": self.min_layer}


class ExitScorer(nn.Module):
    """Wrap an isolated DecisionScorer; its head becomes the last layer's head."""

    def __init__(self, scorer: DecisionScorer, init: torch.Tensor | None = None):
        super().__init__()
        if scorer.mode != "isolated":
            raise ValueError("early exit is implemented for the isolated layout")
        self.scorer = scorer
        self.base = base_model(scorer.backbone)
        L = len(self.base.layers)
        w = (scorer.head.weight.detach()[0] if init is None else init).float()
        b = float(scorer.head.bias.detach()[0]) if init is None else 0.0
        heads = []
        for _ in range(L - 1):
            h = nn.Linear(w.numel(), 1)
            with torch.no_grad():
                h.weight.copy_(w.view(1, -1))
                h.bias.fill_(b)
            heads.append(h)
        self.heads = nn.ModuleList(heads + [scorer.head])  # heads[l] reads the output of layer l
        self.num_layers = L
        self.tokens_processed = 0

    @property
    def device(self):
        return self.scorer.device

    def _prepare(self, batch):
        dev = self.device
        ids, pos, seg, cand_last, _ = pack_setwise(batch, False, self.scorer.pad_id)
        self.tokens_processed = int((seg != PAD).sum())
        mask = block_mask(seg.to(dev), next(self.base.parameters()).dtype)
        ids, pos = ids.to(dev), pos.to(dev)
        h = self.base.embed_tokens(ids)
        cos, sin = self.base.rotary_emb(h, pos)
        return h, mask, pos, (cos, sin), cand_last.to(dev)

    def _read(self, l, h, cand_last):
        rows = torch.arange(h.shape[0], device=h.device).unsqueeze(1)
        x = self.base.norm(h[rows, cand_last]).float()                  # [B, K, H]
        return self.heads[l](x).squeeze(-1)

    def forward(self, batch: list[dict]) -> torch.Tensor:
        """Logits of every layer: [L, B, K]."""
        h, mask, pos, pe, cand_last = self._prepare(batch)
        out = []
        for l, layer in enumerate(self.base.layers):
            h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
            out.append(self._read(l, h, cand_last))
        return torch.stack(out)

    @torch.no_grad()
    def decide(self, batch: list[dict], policy: ExitPolicy, temps: torch.Tensor):
        """Real early exit: finished requests are dropped from the batch.

        Returns (temperature-scaled logits [B, K] of each request's exit layer, exit layer [B]).
        """
        h, mask, pos, (cos, sin), cand_last = self._prepare(batch)
        B, last = h.shape[0], self.num_layers - 1
        live = torch.arange(B, device=h.device)
        logits = torch.empty(B, cand_last.shape[1], device=h.device)
        exit_at = torch.full((B,), last, device=h.device, dtype=torch.long)
        streak = torch.zeros(B, dtype=torch.long, device=h.device)
        prev = torch.full((B,), -1, dtype=torch.long, device=h.device)
        for l, layer in enumerate(self.base.layers):
            h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
            if l < policy.min_layer and l != last:
                continue
            lg = self._read(l, h, cand_last) / temps[l]
            if l == last:
                logits[live] = lg
                break
            conf, arg = lg.softmax(-1).max(-1)
            streak = torch.where(arg == prev, streak + 1, torch.ones_like(streak))
            prev = arg
            done = (conf >= policy.tau_hi) | ((streak >= policy.patience) & (conf >= policy.tau_lo))
            if bool(done.any()):
                logits[live[done]] = lg[done]
                exit_at[live[done]] = l
                keep = ~done
                if not bool(keep.any()):
                    break
                live, h, mask, pos = live[keep], h[keep], mask[keep], pos[keep]
                cos, sin, cand_last = cos[keep], sin[keep], cand_last[keep]
                streak, prev = streak[keep], prev[keep]
        return logits, exit_at


def simulate(layer_logits: torch.Tensor, temps: torch.Tensor, policy: ExitPolicy):
    """Offline replay of ``ExitScorer.decide`` on saved [L, N, K] logits.

    Returns (logits [N, K] of each request's exit layer, divided by that layer's
    temperature, and the exit layer [N]).
    """
    L, N, _ = layer_logits.shape
    temps = temps.float()
    conf, arg = (layer_logits.float() / temps.view(L, 1, 1)).softmax(-1).max(-1)   # [L, N]
    streak = torch.ones_like(arg)            # decide() starts counting at min_layer
    for l in range(policy.min_layer + 1, L):
        streak[l] = torch.where(arg[l] == arg[l - 1], streak[l - 1] + 1, 1)
    done = (conf >= policy.tau_hi) | ((streak >= policy.patience) & (conf >= policy.tau_lo))
    done[:policy.min_layer] = False
    done[-1] = True
    exit_at = done.float().argmax(0)
    chosen = layer_logits[exit_at, torch.arange(N)].float() / temps[exit_at].view(N, 1)
    return chosen, exit_at
