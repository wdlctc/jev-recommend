#!/usr/bin/env bash
# After the 8B runs: hard (popularity-sampled) negatives for 1.7B, then a clean latency re-measure.
cd "$(dirname "$0")/.."
until [ -f runs/q8b/transfer/DONE ]; do sleep 30; done
TAG=q17b_hard NEGATIVES=popular SKIP_POINTWISE=1 SKIP_BENCH=1 bash scripts/run_all.sh
CUDA_VISIBLE_DEVICES=${BENCH_GPU:-0} python3 -m jevrec.bench --requests 60 --warmup 10 \
  --out runs/q17b/bench.json > runs/q17b/bench_rerun.log 2>&1
echo done > runs/CHAIN2_DONE
