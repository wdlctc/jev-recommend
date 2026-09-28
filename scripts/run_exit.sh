#!/usr/bin/env bash
# Jev-Exit: per-layer decision heads. Three runs in parallel on 8 GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
R=runs/exit
mkdir -p $R
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 29521 -m jevrec.train_exit --variant retrofit \
  --init-from runs/q17b/isolated/trainable.pt --out $R/q17b_retrofit > $R/q17b_retrofit.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc-per-node 4 --master-port 29522 -m jevrec.train_exit --variant joint \
  --out $R/q17b_joint > $R/q17b_joint.log 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc-per-node 2 --master-port 29523 -m jevrec.train_exit --variant retrofit \
  --model Qwen/Qwen3-8B --init-from runs/q8b/isolated/trainable.pt --out $R/q8b_retrofit > $R/q8b_retrofit.log 2>&1 &
wait
echo done > $R/DONE
