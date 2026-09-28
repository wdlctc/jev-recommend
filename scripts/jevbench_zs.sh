#!/usr/bin/env bash
# Zero-shot JevBench public-set baselines: one server per GPU, official CLI as client.
cd /data00/home/cheng.luo/jev-recommend
B=/data00/home/cheng.luo/jevbench
T=$B/datasets/public/easy.jsonl,$B/datasets/public/original.jsonl,$B/datasets/public/hard.jsonl
R=runs/jevbench/zs; mkdir -p $R
python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B')" > $R/dl.log 2>&1
i=0
for m in 1.7B 4B 8B; do for ro in letters branches; do
  port=$((8800+i)); name=qwen3-$m-$ro
  CUDA_VISIBLE_DEVICES=$i python3 -m jevrec.server --model Qwen/Qwen3-$m --readout $ro --port $port --name $name > $R/$name.server.log 2>&1 &
  i=$((i+1))
done; done
for p in $(seq 8800 8805); do until curl -s localhost:$p/health >/dev/null; do sleep 5; done; done
i=0
for m in 1.7B 4B 8B; do for ro in letters branches; do
  port=$((8800+i)); name=qwen3-$m-$ro
  ( cd $B && TYPESAFE_PRICE_INPUT_PER_M=0 TYPESAFE_PRICE_OUTPUT_PER_M=0 python3 -m jevbench.cli run --tasks $T --adapter typesafe \
      --endpoint http://127.0.0.1:$port --model $name --key-env "" --results /data00/home/cheng.luo/jev-recommend/$R/$name.jsonl \
      --ledger /data00/home/cheng.luo/jev-recommend/$R/$name.ledger.jsonl --raw-dir /data00/home/cheng.luo/jev-recommend/$R/raw_$name \
      > /data00/home/cheng.luo/jev-recommend/$R/$name.run.log 2>&1 && \
    python3 -m jevbench.cli summarize --tasks $T --results /data00/home/cheng.luo/jev-recommend/$R/$name.jsonl \
      > /data00/home/cheng.luo/jev-recommend/$R/$name.summary.txt 2>&1 ) &
  i=$((i+1))
done; done
wait %7 %8 %9 %10 %11 %12 2>/dev/null; wait
echo done > $R/DONE
