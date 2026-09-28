#!/usr/bin/env bash
# Zero-shot run for one extra backbone; args: model_id name gpu port
cd /data00/home/cheng.luo/jev-recommend
B=/data00/home/cheng.luo/jevbench
T=$B/datasets/public/easy.jsonl,$B/datasets/public/original.jsonl,$B/datasets/public/hard.jsonl
R=runs/jevbench/zs; mkdir -p $R
M=$1; N=$2; G=$3; P=$4; RO=${5:-letters}
python3 -c "from huggingface_hub import snapshot_download as s; s('$M')" > $R/$N.dl.log 2>&1
CUDA_VISIBLE_DEVICES=$G python3 -m jevrec.server --model $M --readout $RO --temperature ${TEMP:-1.0} --port $P --name $N > $R/$N.server.log 2>&1 &
SP=$!
until curl -s localhost:$P/health >/dev/null; do sleep 5; kill -0 $SP 2>/dev/null || exit 1; done
cd $B && TYPESAFE_PRICE_INPUT_PER_M=0 TYPESAFE_PRICE_OUTPUT_PER_M=0 python3 -m jevbench.cli run --tasks $T --adapter typesafe \
  --endpoint http://127.0.0.1:$P --model $N --key-env "" --results /data00/home/cheng.luo/jev-recommend/$R/$N.jsonl \
  --ledger /data00/home/cheng.luo/jev-recommend/$R/$N.ledger.jsonl --raw-dir /data00/home/cheng.luo/jev-recommend/$R/raw_$N \
  > /data00/home/cheng.luo/jev-recommend/$R/$N.run.log 2>&1
python3 -m jevbench.cli summarize --tasks $T --results /data00/home/cheng.luo/jev-recommend/$R/$N.jsonl > /data00/home/cheng.luo/jev-recommend/$R/$N.summary.txt 2>&1
kill $SP
