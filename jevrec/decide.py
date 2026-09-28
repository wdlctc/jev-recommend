"""General typed decisions (TypeSafe System One wire format) with a Qwen3 backbone.

A request carries a ``state`` (string or JSON) and one or more questions:
``{"type": "choice" | "noul" | "score", "instructions": str, "criteria": ...}``.
choice criteria map option keys to descriptions, noul criteria describe
true/false, score criteria are an ordered list of levels.

Two readouts, both one forward pass per question:

letters   The prompt lists every option as a lettered line; the answer is the
          next-token distribution over the valid letters (options can be
          compared directly).
branches  The state, task and full option list form a shared prefix encoded
          once; each option then gets an isolated branch ("Is option C
          correct? Answer Yes or No.") under the block mask from
          ``jevrec.model``, read as the Yes-minus-No logit, softmaxed over
          options.
"""
from __future__ import annotations

import json
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import PAD, block_mask, pack_setwise

LETTERS = string.ascii_uppercase


def render_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def options_of(question: dict) -> list[tuple[str, str]]:
    """[(answer key, description)] in the order the answer is reported."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        criteria = criteria or {}
        return [("true", criteria.get("true", "Yes, the statement holds.")),
                ("false", criteria.get("false", "No, the statement does not hold."))]
    if kind == "score":
        return [(str(i), level) for i, level in enumerate(criteria)]
    if isinstance(criteria, dict):
        return list(criteria.items())
    return [(str(c), str(c)) for c in criteria]


def user_prompt(state, question: dict, options) -> str:
    lines = [f"{LETTERS[i]}. {key}: {desc}" if key != desc else f"{LETTERS[i]}. {key}"
             for i, (key, desc) in enumerate(options)]
    return (f"{render_state(state)}\n\n---\nTask: {question['instructions']}\n"
            f"Options:\n" + "\n".join(lines) + "\n")


LETTER_SUFFIX = "Answer with the letter of the correct option only."


def letters_text(tok, state, question: dict) -> tuple[str, list[tuple[str, str]]]:
    """Chat-formatted prompt whose next token is the answer letter (shared by training and serving)."""
    options = options_of(question)
    if len(options) > len(LETTERS):
        raise ValueError("at most 26 options are supported")
    user = user_prompt(state, question, options) + LETTER_SUFFIX
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return text, options


def gold_index(record: dict, options: list[tuple[str, str]]) -> int:
    """Position of a JevBench-style ``expected`` label among ``options_of`` keys."""
    kind, expected = record["question"]["type"], str(record["expected"])
    if kind == "noul":
        return 0 if expected in ("yes", "true", "True") else 1
    return [k for k, _ in options].index(expected)


class DecisionEngine:
    def __init__(self, model_id: str, readout: str = "letters", device: str = "cuda",
                 dtype=torch.bfloat16, temperature: float = 1.0, adapter: str | None = None):
        if readout not in ("letters", "branches"):
            raise ValueError("readout must be 'letters' or 'branches'")
        self.readout, self.device, self.temperature = readout, device, temperature
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.to(device).eval()
        self.letter_ids = [self._one(c) for c in LETTERS]
        self.yes, self.no = self._one("Yes"), self._one("No")
        self.model_id = model_id

    def _one(self, text: str) -> int:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token")
        return ids[0]

    def _chat(self, user: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    @torch.inference_mode()
    def logits(self, state, question: dict) -> tuple[list[tuple[str, str]], torch.Tensor, int]:
        """Return options, one logit per option, and input tokens processed."""
        options = options_of(question)
        if len(options) > len(LETTERS):
            raise ValueError("at most 26 options are supported")
        prompt = user_prompt(state, question, options)
        if self.readout == "letters":
            text, _ = letters_text(self.tok, state, question)
            ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
            out = self.model(input_ids=ids).logits[0, -1].float()
            return options, out[self.letter_ids[:len(options)]], ids.shape[1]
        # branches: split the chat text at the end of the shared prefix
        marker = "⁣"  # invisible separator, only used to locate the split point
        text = self._chat(prompt + marker)
        head, tail = text.split(marker)
        prefix = self.tok.encode(head, add_special_tokens=False)
        cands = [self.tok.encode(f"Is option {LETTERS[i]} ({key}) correct? Answer Yes or No." + tail,
                                 add_special_tokens=False) for i, (key, _) in enumerate(options)]
        record = {"state": prefix, "cands": cands, "decide": []}
        ids, pos, seg, last, _ = pack_setwise([record], False, self.tok.pad_token_id or 0)
        mask = block_mask(seg.to(self.device), next(self.model.parameters()).dtype)
        out = self.model(input_ids=ids.to(self.device), position_ids=pos.to(self.device),
                         attention_mask=mask).logits[0]
        at = out[last[0].to(self.device)].float()
        return options, at[:, self.yes] - at[:, self.no], int((seg != PAD).sum())

    def answer(self, state, question: dict) -> tuple[dict, int]:
        options, logit, tokens = self.logits(state, question)
        probs = (logit / self.temperature).softmax(-1).tolist()
        keys = [k for k, _ in options]
        top = max(range(len(keys)), key=probs.__getitem__)
        k = len(keys)
        confidence = (probs[top] - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
        kind = question["type"]
        if kind == "noul":
            return {"type": "noul", "noul": probs[0], "confidence": abs(2 * probs[0] - 1)}, tokens
        ans = {"type": kind, "probabilities": dict(zip(keys, probs)), "confidence": confidence}
        if kind == "choice":
            ans["choice"] = keys[top]
        else:
            ans["score"] = sum(i * p for i, p in enumerate(probs))
        return ans, tokens
