"""Public datasets rendered as typed decisions (JevBench / System One record format).

Each record: {"id", "family", "state", "question": {"type", "instructions", "criteria"},
"labels", "expected"}. Option order is shuffled with a fixed seed. No JevBench item is
used anywhere: every source is an independent public dataset.

  python -m jevrec.decision_data --out data/decisions --train-cap 8000 --calib-cap 300
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

NLI3 = {"entailment": "The statement must be true given the context.",
        "neutral": "The statement may or may not be true; the context does not settle it.",
        "contradiction": "The statement must be false given the context."}


def choice(rid, family, state, instructions, options: dict, gold: str, rng):
    keys = list(options)
    rng.shuffle(keys)
    return {"id": rid, "family": family, "state": state,
            "question": {"type": "choice", "instructions": instructions,
                         "criteria": {k: options[k] for k in keys}},
            "labels": keys, "expected": gold}


def noul(rid, family, state, instructions, truth: bool, true_desc=None, false_desc=None):
    crit = {"true": true_desc or "Yes.", "false": false_desc or "No."}
    return {"id": rid, "family": family, "state": state,
            "question": {"type": "noul", "instructions": instructions, "criteria": crit},
            "labels": ["no", "yes"], "expected": "yes" if truth else "no"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "option"


def mc(rid, family, state, question, texts: list[str], gold_idx: int, rng):
    """Multiple choice with free-text options: keys are option_a.. and descriptions the text."""
    opts = {f"option_{'abcdefgh'[i]}": t for i, t in enumerate(texts)}
    return choice(rid, family, state, question, opts, f"option_{'abcdefgh'[gold_idx]}", rng)


# ---- sources: each yields records for a split ("train" or "calib") -------------------------------

def anli(split, rng):
    for r in (1, 2, 3):
        ds = load_dataset("facebook/anli", split=f"train_r{r}" if split == "train" else f"dev_r{r}")
        for x in ds:
            if x["label"] < 0:
                continue
            yield choice(f"anli{r}-{x['uid']}", "nli", x["premise"],
                         f"Does the context support this statement: \"{x['hypothesis']}\"",
                         NLI3, ["entailment", "neutral", "contradiction"][x["label"]], rng)


def wanli(split, rng):
    ds = load_dataset("alisawuffles/WANLI", split="train" if split == "train" else "test")
    for x in ds:
        yield choice(f"wanli-{x['id']}", "nli", x["premise"],
                     f"Does the context support this statement: \"{x['hypothesis']}\"", NLI3, x["gold"], rng)


def logiqa(split, rng):
    ds = load_dataset("tasksource/logiqa-2.0-nli", split="train" if split == "train" else "validation")
    for i, x in enumerate(ds):
        yield noul(f"logiqa-{i}", "logic", x["premise"],
                   f"Does the passage logically entail this conclusion: \"{x['hypothesis']}\"",
                   x["label"] == "entailment", "The conclusion follows from the passage.",
                   "The conclusion does not follow from the passage.")


def boolq(split, rng):
    ds = load_dataset("google/boolq", split="train" if split == "train" else "validation")
    for i, x in enumerate(ds):
        yield noul(f"boolq-{i}", "reading", x["passage"], x["question"].capitalize() + "?", bool(x["answer"]))


def race(split, rng):
    for level in ("high", "middle"):
        ds = load_dataset("ehovy/race", level, split="train" if split == "train" else "validation")
        for x in ds:
            yield mc(f"race-{x['example_id']}-{hash(x['question']) % 10**6}", "reading", x["article"],
                     x["question"], x["options"], "ABCD".index(x["answer"]), rng)


def quality(split, rng):
    ds = load_dataset("emozilla/quality", split="train" if split == "train" else "validation")
    for i, x in enumerate(ds):
        yield mc(f"quality-{i}", "long_doc", x["article"], x["question"], x["options"], x["answer"], rng)


def intents(name, cfg, text_key, label_fn, rid, family, split, rng, k=6):
    ds = load_dataset(name, cfg, split="train" if split == "train" else ("validation" if name.startswith("clinc") else "test"))
    labels = sorted({label_fn(ds, x) for x in ds})
    for i, x in enumerate(ds):
        gold = label_fn(ds, x)
        others = rng.sample([l for l in labels if l != gold], k - 1)
        opts = {slug(l): l.replace("_", " ") for l in others + [gold]}
        yield choice(f"{rid}-{i}", family, x[text_key], "Which intent does the user's message express?",
                     opts, slug(gold), rng)


def clinc(split, rng):
    feat = None

    def lab(ds, x):
        nonlocal feat
        feat = feat or ds.features["intent"]
        return feat.int2str(x["intent"])
    yield from intents("clinc/clinc_oos", "plus", "text", lab, "clinc", "intent", split, rng)


def banking(split, rng):
    yield from intents("mteb/banking77", None, "text", lambda ds, x: x["label_text"], "banking", "intent", split, rng)


def mmlu(split, rng):
    ds = load_dataset("cais/mmlu", "auxiliary_train" if split == "train" else "all",
                      split="train" if split == "train" else "validation")
    for i, x in enumerate(ds):
        x = x.get("train", x) if isinstance(x.get("train"), dict) else x
        yield mc(f"mmlu-{i}", "knowledge", f"Subject: {x.get('subject', 'general')}", x["question"],
                 x["choices"], x["answer"], rng)


def arc(split, rng):
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        ds = load_dataset("allenai/ai2_arc", cfg, split="train" if split == "train" else "validation")
        for x in ds:
            labels = x["choices"]["label"]
            if x["answerKey"] not in labels:
                continue
            yield mc(f"arc-{x['id']}", "knowledge", "Science question.", x["question"], x["choices"]["text"],
                     labels.index(x["answerKey"]), rng)


def csqa(split, rng):
    ds = load_dataset("tau/commonsense_qa", split="train" if split == "train" else "validation")
    for x in ds:
        yield mc(f"csqa-{x['id']}", "commonsense", "Commonsense question.", x["question"], x["choices"]["text"],
                 x["choices"]["label"].index(x["answerKey"]), rng)


def obqa(split, rng):
    ds = load_dataset("allenai/openbookqa", "main", split="train" if split == "train" else "validation")
    for x in ds:
        yield mc(f"obqa-{x['id']}", "knowledge", "Elementary science question.", x["question_stem"],
                 x["choices"]["text"], x["choices"]["label"].index(x["answerKey"]), rng)


def proofwriter(split, rng):
    ds = load_dataset("tasksource/proofwriter", split="train" if split == "train" else "validation")
    opts = {"true": "The statement follows from the facts and rules.",
            "false": "The negation of the statement follows from the facts and rules.",
            "unknown": "Neither the statement nor its negation can be derived."}
    for x in ds:
        ans = str(x["answer"]).lower()
        if ans not in opts:
            continue
        yield choice(f"pw-{x['id']}", "multi_hop", x["theory"],
                     f"Using only these facts and rules, what is the status of: \"{x['question']}\"", opts, ans, rng)


def fld(split, rng):
    ds = load_dataset("hitachi-nlp/FLD.v2", split="train" if split == "train" else "validation")
    opts = {"proved": "The hypothesis can be proved from the facts.",
            "disproved": "The negation of the hypothesis can be proved from the facts.",
            "unknown": "Neither can be proved from the facts."}
    for i, x in enumerate(ds):
        label = str(x.get("world_assump_label") or x.get("proof_label") or "").lower()
        label = {"__proved__": "proved", "__disproved__": "disproved", "__unknown__": "unknown"}.get(label, label)
        if label not in opts:
            continue
        yield choice(f"fld-{i}", "multi_hop", x["context"], f"Hypothesis: \"{x['hypothesis']}\". Which holds?",
                     opts, label, rng)


def defeasible(split, rng):
    ds = load_dataset("metaeval/defeasible-nli", "atomic", split="train" if split == "train" else "validation")
    for i, x in enumerate(ds):
        yield choice(f"dnli-{i}", "tradeoff",
                     f"Premise: {x['Premise']}\nHypothesis: {x['Hypothesis']}\nNew information: {x['Update']}",
                     "Does the new information make the hypothesis more or less likely?",
                     {"strengthener": "It makes the hypothesis more likely.",
                      "weakener": "It makes the hypothesis less likely."}, x["UpdateType"], rng)


def gsm8k(split, rng):
    ds = load_dataset("openai/gsm8k", "main", split="train" if split == "train" else "test")
    for i, x in enumerate(ds):
        gold = x["answer"].split("####")[-1].strip().replace(",", "")
        try:
            g = float(gold)
        except ValueError:
            continue
        cands = {g}
        for delta in rng.sample([1, 2, 5, 10, -1, -2, -5, -10, g, -g / 2, g * 0.1, 3], 8):
            v = g + delta
            if v >= 0 and v not in cands:
                cands.add(round(v, 2))
            if len(cands) == 4:
                break
        texts = [f"{v:g}" for v in cands]
        rng.shuffle(texts)
        yield mc(f"gsm8k-{i}", "numeric", x["question"], "What is the correct final answer?", texts,
                 texts.index(f"{g:g}"), rng)


SOURCES = {"anli": anli, "wanli": wanli, "logiqa": logiqa, "boolq": boolq, "race": race, "quality": quality,
           "clinc": clinc, "banking": banking, "mmlu": mmlu, "arc": arc, "csqa": csqa, "obqa": obqa,
           "proofwriter": proofwriter, "fld": fld, "defeasible": defeasible, "gsm8k": gsm8k}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/decisions")
    p.add_argument("--train-cap", type=int, default=8000)
    p.add_argument("--calib-cap", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=16000)
    p.add_argument("--sources", default=",".join(SOURCES))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, cap in (("train", args.train_cap), ("calib", args.calib_cap)):
        rows = []
        for name in args.sources.split(","):
            rng = random.Random(f"{name}-{split}")
            try:
                recs = [r for r in SOURCES[name](split, rng) if len(r["state"]) <= args.max_chars]
            except Exception as e:  # a source that fails to load is reported, not silently dropped
                print(f"[skip] {name}/{split}: {e}", flush=True)
                continue
            rng.shuffle(recs)
            rows += recs[:cap]
            stats.setdefault(name, {})[split] = min(cap, len(recs))
            print(f"{name}/{split}: {min(cap, len(recs))} of {len(recs)}", flush=True)
        random.Random(split).shuffle(rows)
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
