#!/usr/bin/env bash
# Downstream transfer: ML-1M-trained scorers applied zero-shot to ml-latest-small
# (2018 catalogue, mostly unseen movies), vs. zero-shot LLM and in-domain baselines.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
TAG=${TAG:-q17b}
G=(${GPUS:-0 1 6 7})
R=runs/$TAG
D=$R/transfer
mkdir -p "$D"
EV="--model $MODEL --epochs 0 --dataset ml-latest-small"
CUDA_VISIBLE_DEVICES=${G[0]} python3 -m jevrec.train $EV --mode isolated \
  --init-from $R/isolated/trainable.pt --out $D/isolated > $D/isolated.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G[1]} python3 -m jevrec.train $EV --mode listwise \
  --init-from $R/listwise/trainable.pt --out $D/listwise > $D/listwise.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G[2]} python3 -m jevrec.train $EV --mode isolated \
  --out $D/zeroshot_isolated > $D/zeroshot_isolated.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G[3]} python3 -m jevrec.baselines --dataset ml-latest-small \
  --out $D/baselines > $D/baselines.log 2>&1 &
wait
echo done > $D/DONE
