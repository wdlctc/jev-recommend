"""Three ways to score K candidates against one shared state with an LLM.

pointwise  Open-Jev style. One sequence per candidate (state + candidate),
           plain causal attention, scalar readout at the last token. The state
           is recomputed K times; candidates never see each other.
isolated   Same function, one sequence per record. A block mask lets every
           candidate attend to the state and to itself only, and position ids
           restart after the state, so each candidate's logit equals the
           pointwise one while the state is encoded once (training included).
listwise   kev style. Isolated candidates plus a trailing "decide" segment
           that attends to everything; a low-rank bilinear term between the
           decide state and each candidate state lets candidates compete.
           Initialised to zero, so at step 0 it equals ``isolated``.

All modes read a scalar from the final hidden state with a head initialised
to ``W_yes - W_no`` of the pretrained unembedding, i.e. the zero-shot Yes/No
logit margin. A pure-attention backbone (e.g. Qwen3) is required for the
mask-based modes: linear-attention/SSM layers ignore attention masks.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODES = ("pointwise", "isolated", "listwise")

# Segment ids used to build the block mask.
STATE, DECIDE, PAD = 0, -1, -2


def pack_setwise(batch: list[dict], with_decide: bool, pad_id: int):
    """Pack each record into one row: state | cand_1 | ... | cand_K [| decide].

    Returns input ids, position ids, segment ids (all [B, T]), the index of each
    candidate's last token [B, K] and of the decide token [B].
    """
    rows, positions, segments, cand_last, decide_last = [], [], [], [], []
    for record in batch:
        ids, pos, seg, last = list(record["state"]), list(range(len(record["state"]))), \
            [STATE] * len(record["state"]), []
        start = len(record["state"])
        for i, cand in enumerate(record["cands"]):
            ids += cand
            pos += range(start, start + len(cand))
            seg += [i + 1] * len(cand)
            last.append(len(ids) - 1)
        if with_decide:
            tail = start + max(map(len, record["cands"]))
            ids += record["decide"]
            pos += range(tail, tail + len(record["decide"]))
            seg += [DECIDE] * len(record["decide"])
        rows.append(ids), positions.append(pos), segments.append(seg)
        cand_last.append(last), decide_last.append(len(ids) - 1)
    width = max(map(len, rows))
    for ids, pos, seg in zip(rows, positions, segments):
        extra = width - len(ids)
        ids += [pad_id] * extra
        pos += [0] * extra
        seg += [PAD] * extra
    as_long = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
    return as_long(rows), as_long(positions), as_long(segments), as_long(cand_last), as_long(decide_last)


def block_mask(segments: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive [B, 1, T, T] mask from segment ids.

    query q may attend key k iff k <= q and (k is state, or same segment, or q
    is a decide token). Padding attends only to itself so softmax stays finite.
    """
    q, k = segments.unsqueeze(-1), segments.unsqueeze(-2)
    T = segments.shape[-1]
    idx = torch.arange(T, device=segments.device)
    causal = idx.view(1, T, 1) >= idx.view(1, 1, T)
    allowed = causal & ((k == STATE) | (k == q) | (q == DECIDE)) & (k != PAD) & (q != PAD)
    allowed |= torch.eye(T, dtype=torch.bool, device=segments.device).unsqueeze(0) & (q == PAD)
    mask = torch.zeros(allowed.shape, dtype=dtype, device=segments.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(1)


def pack_pointwise(batch: list[dict], pad_id: int):
    """One right-padded row per (record, candidate); returns ids, 2D mask, last index."""
    rows = [r["state"] + c for r in batch for c in r["cands"]]
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = 1
    return ids, mask, mask.sum(-1) - 1


class DecisionScorer(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, yes_minus_no: torch.Tensor,
                 mode: str = "isolated", pad_id: int = 0, rank: int = 64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.backbone, self.mode, self.pad_id = backbone, mode, pad_id
        self.head = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.head.weight.copy_(yes_minus_no.float().view(1, -1))
            self.head.bias.zero_()
        if mode == "listwise":
            self.query = nn.Linear(hidden_size, rank, bias=False)
            self.key = nn.Linear(hidden_size, rank, bias=False)
            nn.init.zeros_(self.query.weight)
            self.rank = rank
        self.tokens_processed = 0

    @property
    def device(self):
        return self.head.weight.device

    def _dtype(self):
        return next(self.backbone.parameters()).dtype

    def forward(self, batch: list[dict]) -> torch.Tensor:
        """Return [B, K] candidate logits (K must be equal across the batch)."""
        K = len(batch[0]["cands"])
        if any(len(r["cands"]) != K for r in batch):
            raise ValueError("all records in a batch need the same number of candidates")
        dev = self.device
        if self.mode == "pointwise":
            ids, mask, last = pack_pointwise(batch, self.pad_id)
            self.tokens_processed = int(mask.sum())
            hidden = self.backbone(input_ids=ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state
            h = hidden[torch.arange(len(ids), device=dev), last.to(dev)]
            return self.head(h.float()).view(len(batch), K)

        ids, pos, seg, cand_last, decide_last = pack_setwise(batch, self.mode == "listwise", self.pad_id)
        self.tokens_processed = int((seg != PAD).sum())
        mask = block_mask(seg.to(dev), self._dtype())
        hidden = self.backbone(input_ids=ids.to(dev), position_ids=pos.to(dev),
                               attention_mask=mask).last_hidden_state
        rows = torch.arange(len(batch), device=dev)
        h = hidden[rows.unsqueeze(1), cand_last.to(dev)].float()      # [B, K, H]
        logits = self.head(h).squeeze(-1)
        if self.mode == "listwise":
            d = hidden[rows, decide_last.to(dev)].float()                # [B, H]
            logits = logits + torch.einsum("br,bkr->bk", self.query(d), self.key(h)) / math.sqrt(self.rank)
        return logits


def build(model_id: str, mode: str, *, lora_rank: int = 16, device: str = "cuda",
          dtype: torch.dtype = torch.bfloat16, attn: str = "sdpa", gradient_checkpointing: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    full = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
    yes = tokenizer.encode(" Yes", add_special_tokens=False)
    no = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("' Yes' / ' No' must be single tokens for the head initialisation")
    unembed = full.get_output_embeddings().weight
    init = (unembed[yes[0]] - unembed[no[0]]).detach().float().clone()
    backbone = full.model
    hidden = full.config.hidden_size
    del full
    backbone.config.use_cache = False
    backbone.requires_grad_(False)
    if lora_rank:
        from peft import LoraConfig, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = DecisionScorer(backbone, hidden, init, mode=mode, pad_id=pad)
    return model.to(device), tokenizer
