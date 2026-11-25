#!/bin/bash
# Baselines on the Swedish Lion's out-of-domain lines, before the trainings take GPUs 1 and 2.
cd /home/gabriel/Desktop/diffu2
E=packages/htr_train/envs/minicpm/.venv/bin/python
L=data/htr/eval/ood.jsonl
O=runs/htr/eval
mkdir -p $O
T=Riksarkivet/trocr-large-handwritten-hist-swe-3-char
( CUDA_VISIBLE_DEVICES=2 $E -m diffu_htr_train.evaluate read --lines $L --reader trocr --model $T --beams 4 --batch 16 --out $O/ood_trocr_beam4.jsonl
  CUDA_VISIBLE_DEVICES=2 $E -m diffu_htr_train.evaluate read --lines $L --reader trocr --model $T --beams 1 --batch 32 --out $O/ood_trocr_greedy.jsonl
  CUDA_VISIBLE_DEVICES=2 $E -m diffu_htr_train.evaluate read --lines $L --reader trocr --model $T --beams 1 --batch 32 --dtype bfloat16 --out $O/ood_trocr_greedy_bf16.jsonl ) > $O/trocr.log 2>&1 &
M=$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM-OCR-1.0/snapshots/2697d4e93dca82784abf858ae57206a49629cb6b
CUDA_VISIBLE_DEVICES=1 packages/htr_train/envs/vllm/.venv/bin/vllm serve $M --served-model-name minicpm --port 8001 \
  --gpu-memory-utilization 0.85 --max-model-len 8192\
  --allowed-local-media-path /mnt/work/gabriel/htr --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
  --default-chat-template-kwargs '{"enable_thinking": false}' > $O/vllm_base.log 2>&1 &
V=$!
until curl -sf localhost:8001/health > /dev/null; do kill -0 $V 2>/dev/null || { echo "vLLM died"; exit 1; }; sleep 5; done
$E -m diffu_htr_train.evaluate read --lines $L --reader minicpm --model minicpm --endpoint http://localhost:8001 --concurrency 64 --out $O/ood_minicpm_base.jsonl > $O/minicpm_base.log 2>&1
kill $V; wait
$E -m diffu_htr_train.evaluate cer $O/ood_trocr_beam4.jsonl $O/ood_trocr_greedy.jsonl $O/ood_trocr_greedy_bf16.jsonl $O/ood_minicpm_base.jsonl
echo EVAL DONE
