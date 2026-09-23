# jev-recommend

**Jev-style typed decision models used as recommenders.** A user's history is the
shared *state*, K candidate items are the options of one `Choice` question, and an
LLM returns a calibrated probability for every candidate in a single forward pass,
with no generated text. We compare three ways to score candidates on the same
backbone, train them on MovieLens-1M, and test them downstream on a newer
catalogue they have never seen.

Blog post: <https://wdlctc.github.io/jev-recommend.html>

| scoring mode | what it is | state encoded | candidates see each other |
|---|---|---|---|
| `pointwise` | [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) style: one sequence per (state, candidate), scalar readout at the last token | K times | no |
| `isolated` | one sequence per request, **block attention mask** + position ids restarting after the state | once | no |
| `listwise` | `isolated` + a trailing *decide* segment that attends to all candidates, low-rank bilinear term (zero-initialised) | once | through the decide token |

`isolated` computes **exactly the same function** as `pointwise`
(`tests/test_model.py` checks this in fp32 for eager and SDPA attention), so it
is a drop-in replacement that encodes the state once, during training too, not
only at inference with a prefix cache.

All modes read a scalar with a head initialised to `W_yes − W_no` of the
pretrained unembedding (the zero-shot Yes/No margin), fine-tune LoRA (r=16) on
the backbone, and optimise softmax cross-entropy over the K candidates plus
0.1 × Brier. One temperature is fitted on validation afterwards.

## Results (Qwen3-1.7B, 1 epoch, 48k training decisions)

MovieLens-1M, leave-one-out test (6,040 users), K = 20 candidates (1 positive +
19 random unseen movies), identical candidate sets for every model. All numbers
after validation-fitted temperature scaling.

| model | HR@1 | HR@5 | NDCG@10 | MRR | NLL | ECE | auto-decidable @95% precision | tokens / request |
|---|---|---|---|---|---|---|---|---|
| popularity | 0.274 | 0.695 | 0.558 | 0.460 | 2.33 | 0.025 | 0.0% | – |
| SASRec (item IDs) | **0.623** | **0.899** | **0.799** | **0.746** | **1.25** | **0.013** | **22.9%** | – |
| zero-shot LLM, pointwise | 0.084 | 0.357 | 0.303 | 0.236 | 2.95 | 0.002 | 0.0% | 9,124 |
| zero-shot LLM, isolated | 0.087 | 0.365 | 0.304 | 0.237 | 2.95 | 0.005 | 0.0% | 910 |
| **Jev-style, isolated** | 0.604 | 0.894 | 0.785 | 0.731 | 1.29 | 0.023 | 20.5% | **910** |
| same weights, pointwise | 0.605 | 0.894 | 0.786 | 0.732 | 1.29 | 0.023 | 20.6% | 9,124 |
| Jev-style, listwise | 0.604 | 0.894 | 0.785 | 0.732 | 1.29 | 0.026 | 18.6% | 923 |

*auto-decidable @95%*: the largest share of requests, sorted by confidence,
whose top-1 is still ≥ 95% correct. This is the "automate the confident ones,
escalate the rest" operating point that calibrated decisions are meant for.

**Downstream transfer**: scorers trained only on ML-1M, applied unchanged to
`ml-latest-small` (2018 catalogue, 610 users). 245 of the 610 test positives
are movies that do not exist in ML-1M. Popularity and SASRec are trained on the
target dataset itself.

| model | HR@1 all (n=610) | movie in ML-1M (n=365) | movie new since ML-1M (n=245) |
|---|---|---|---|
| popularity (target) | 0.385 | 0.466 | 0.265 |
| SASRec (trained on target) | 0.380 | 0.466 | 0.253 |
| zero-shot LLM | 0.070 | 0.047 | 0.106 |
| **Jev-style isolated (ML-1M only)** | **0.523** | **0.564** | **0.461** |
| Jev-style listwise (ML-1M only) | 0.523 | 0.564 | 0.461 |

**Latency**, one request, batch 1, B200, HF eager + SDPA, p50 ms (tokens):

| mode | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---|---|---|---|---|
| pointwise | 74 (2,294) | 98 (4,588) | 210 (9,169) | 210 (22,926) | 360 (45,851) |
| isolated | 74 (555) | 73 (675) | 73 (909) | 74 (1,622) | 95 (2,809) |
| listwise | 71 (568) | 72 (688) | 72 (922) | 74 (1,635) | 85 (2,822) |

Below ~2k tokens the model is launch-bound (~72 ms floor for 28 layers in eager
PyTorch), so tokens are the more portable cost measure.

![latency](figures/latency.png)
![popularity](figures/popularity.png)
![transfer](figures/transfer.png)
![reliability](figures/reliability.png)

### What we learned

1. **The block mask is free accuracy-wise and ~10× cheaper.** Same weights,
   same metrics to bf16 noise, 10× fewer tokens at K=20 and 16× at K=100. The
   Open-Jev layout pays for re-reading the user state once per candidate.
2. **Letting candidates compete did not help here.** The listwise term learns
   non-trivial weights (it flips 8% of top-1 decisions vs isolated) but gains
   nothing on average. With random negatives each candidate can be judged on
   its own; the comparison is more likely to matter with hard, similar
   negatives.
3. **In-domain, ID models remain strong.** SASRec edges the 1.7B LLM by ~2
   points HR@1 overall. The LLM wins on the least popular positives (<50
   training interactions) and on the most popular ones, and loses in the
   middle band where ID embeddings are well trained.
4. **Text transfers, IDs do not.** Trained on ML-1M only, the LLM scorer beats a
   SASRec trained on the target data by 14 points, and by 21 points on movies
   that did not exist when the training data was collected.
5. **Calibration is cheap.** Raw ECE 0.07 → 0.023 with one temperature (T≈1.2).
   Both SASRec and the LLM are well calibrated after that; neither is a reason
   to prefer the other.

## Reproduce

```bash
pip install -e .            # torch, transformers>=4.56, peft
python -m pytest -q tests   # mask/position equivalence checks, CPU, ~5 s

# full suite on 4 GPUs (downloads MovieLens + Qwen3 weights)
GA=0,1 GB=2,3 bash scripts/run_all.sh          # train isolated + listwise, zero-shot, baselines, bench
bash scripts/run_downstream.sh                  # transfer to ml-latest-small
python scripts/summarize.py runs/q17b           # tables
python scripts/analyze.py runs/q17b figures     # slices + figures
```

Single run: `torchrun --nproc-per-node 2 -m jevrec.train --mode isolated --out runs/iso`.
Evaluate saved weights in another mode: `--epochs 0 --init-from runs/iso/trainable.pt --mode pointwise`.

Trained LoRA + head weights (`trainable.pt`) and all logits are attached to the
[GitHub release](https://github.com/wdlctc/jev-recommend/releases).

## Layout

```
jevrec/data.py       MovieLens loaders, leave-one-out Choice records, fixed-seed negatives
jevrec/prompts.py    state / candidate / decide text segments, tokenised once
jevrec/model.py      packing, block mask, DecisionScorer (pointwise | isolated | listwise)
jevrec/train.py      DDP training, sharded eval, temperature fit
jevrec/metrics.py    HR/NDCG/MRR, NLL, Brier, ECE, auto-decidable coverage
jevrec/baselines.py  popularity and SASRec on the same candidate sets
jevrec/bench.py      per-request latency sweep over K
runs/                results.json per run (logits in the release)
```

## Caveats

- Random negatives, K=20: an easier protocol than full ranking; all models share it.
- One seed per configuration. With 6,040 test users the 95% CI on HR@1 is about ±0.012;
  on the 610-user transfer set about ±0.04.
- The mask-based modes need pure-attention backbones (Qwen3). Linear-attention / SSM
  layers (e.g. Qwen3.5 Gated DeltaNet) ignore attention masks; there you need one row
  per candidate plus a forked state cache, as Open-Jev and kev do.
- This is an independent study inspired by TypeSafe's Jev. It does not use or
  reproduce Jev's weights, data or RLCD training.

## License

Apache-2.0. MovieLens data is subject to the
[GroupLens terms](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt) and is
downloaded, not redistributed.
