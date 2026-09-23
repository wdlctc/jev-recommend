#!/usr/bin/env bash
# After the 1.7B suite: downstream transfer for 1.7B, then the 8B scale point + its transfer.
cd "$(dirname "$0")/.."
until [ -f runs/q17b/DONE ]; do sleep 30; done
TAG=q17b bash scripts/run_downstream.sh
MODEL=Qwen/Qwen3-8B TAG=q8b SKIP_POINTWISE=1 SKIP_BASELINES=1 SKIP_BENCH=1 bash scripts/run_all.sh
mkdir -p runs/q8b/transfer && cp -r runs/q17b/transfer/baselines runs/q8b/transfer/ 2>/dev/null
cp -r runs/q17b/baselines runs/q8b/ 2>/dev/null
MODEL=Qwen/Qwen3-8B TAG=q8b bash scripts/run_downstream.sh
