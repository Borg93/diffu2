#!/bin/bash
# Speed and peak memory of minicpm-all's step on one GPU (GPU 1), batch and gradient checkpointing varied.
cd /home/gabriel/Desktop/diffu2
export HTR_TRAIN_LOCAL_GPUS=1
for v in b4_ckpt b4_nockpt b8_nockpt; do
  uv run htr-train launch runs/htr/bench/$v.yaml --where local > runs/htr/bench/$v.log 2>&1
  echo "$v exit $?"
done
echo BENCH DONE
