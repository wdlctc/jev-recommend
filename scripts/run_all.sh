#!/usr/bin/env bash
# Full experiment on 4 free GPUs (default 0,1,6,7 of the shared B200 box).
# Stage 1: train isolated (GPUs A) and listwise (GPUs B) in parallel.
# Stage 2: zero-shot for all modes, baselines, pointwise eval of the isolated
#          weights (same function, K x the tokens), latency sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
TAG=${TAG:-q17b}
GA=${GA:-0,1}
GB=${GB:-6,7}
G=(${GA//,/ } ${GB//,/ })
R=runs/$TAG
mkdir -p "$R"
COMMON="--model $MODEL --batch-size ${BS:-8} --per-user ${PER_USER:-8} --epochs ${EPOCHS:-1}"

train() {  # gpus mode port
  CUDA_VISIBLE_DEVICES=$1 torchrun --nproc-per-node $(( $(tr -cd , <<<"$1" | wc -c) + 1 )) --master-port $3 \
    -m jevrec.train $COMMON --mode $2 --out $R/$2 > $R/$2.log 2>&1
}
train "$GA" isolated 29511 &
train "$GB" listwise 29512 &
wait

CUDA_VISIBLE_DEVICES=${G[0]} python3 -m jevrec.train --model $MODEL --mode pointwise --epochs 0 \
  --init-from $R/isolated/trainable.pt --out $R/pointwise_from_isolated > $R/pointwise_from_isolated.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G[1]} python3 -m jevrec.train --model $MODEL --mode isolated --epochs 0 \
  --out $R/zeroshot_isolated > $R/zeroshot_isolated.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G[2]} python3 -m jevrec.train --model $MODEL --mode pointwise --epochs 0 \
  --out $R/zeroshot_pointwise > $R/zeroshot_pointwise.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G[3]} python3 -m jevrec.baselines --out $R/baselines > $R/baselines.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=${G[3]} python3 -m jevrec.bench --model $MODEL --out $R/bench.json > $R/bench.log 2>&1
echo done > $R/DONE
