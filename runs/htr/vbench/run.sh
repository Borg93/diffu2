#!/bin/bash
# vLLM throughput of base MiniCPM-OCR on GPU 2 (the fine-tuned model has its architecture): each serve configuration
# read at concurrency 1 (300 lines), 16, 64 and 256 (2,000 out-of-domain lines), its CER kept to see what FP8 costs.
cd /home/gabriel/Desktop/diffu2
E=packages/htr_train/envs/minicpm/.venv/bin/python
V=packages/htr_train/envs/vllm/.venv/bin/vllm
M=$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM-OCR-1.0/snapshots/2697d4e93dca82784abf858ae57206a49629cb6b
O=runs/htr/vbench
L=$O/ood2000.jsonl
COMMON=(--served-model-name minicpm --port 8010 --allowed-local-media-path /mnt/work/gabriel/htr
  --limit-mm-per-prompt '{"image": 4, "video": 0}' --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}'
  --trust-request-mm-kwargs --default-chat-template-kwargs '{"enable_thinking": false}')
OPENBMB=(--attention-backend FLASH_ATTN --mm-encoder-attn-backend FLASH_ATTN --max-num-batched-tokens 40960)
declare -A CONFIGS=(
  [A_ours]="--gpu-memory-utilization 0.85 --max-model-len 8192"
  [B_openbmb]="--gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 256 ${OPENBMB[*]}"
  [C_fp8]="--gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 256 ${OPENBMB[*]} --quantization fp8"
  [D_fp8_kv]="--gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 256 ${OPENBMB[*]} --quantization fp8 --kv-cache-dtype fp8"
)
until ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -qi "A3:00"; do sleep 20; done
for name in A_ours B_openbmb C_fp8 D_fp8_kv; do
  echo "[$(date '+%T')] $name: serving"
  CUDA_VISIBLE_DEVICES=2 $V serve $M "${COMMON[@]}" ${CONFIGS[$name]} > $O/$name.vllm.log 2>&1 &
  SERVER=$!
  until curl -sf localhost:8010/health > /dev/null; do kill -0 $SERVER 2>/dev/null || { echo "$name: vLLM died"; continue 2; }; sleep 5; done
  $E -m diffu_htr_train.evaluate read --lines $L --reader minicpm --model minicpm --endpoint http://localhost:8010 --concurrency 8 --limit 50 --out $O/warmup.jsonl > /dev/null 2>&1
  $E -m diffu_htr_train.evaluate read --lines $L --reader minicpm --model minicpm --endpoint http://localhost:8010 --concurrency 1 --limit 300 --out $O/${name}_c1.jsonl > /dev/null 2>&1
  for c in 16 64 256; do
    $E -m diffu_htr_train.evaluate read --lines $L --reader minicpm --model minicpm --endpoint http://localhost:8010 --concurrency $c --out $O/${name}_c$c.jsonl > /dev/null 2>&1
  done
  kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
  for f in $O/${name}_c*.speed.json; do echo "  $(basename $f .speed.json): $(grep -oE '"lines_per_second": [0-9.]+' $f)"; done
done
$E -m diffu_htr_train.evaluate cer $O/A_ours_c64.jsonl $O/B_openbmb_c64.jsonl $O/C_fp8_c64.jsonl $O/D_fp8_kv_c64.jsonl
echo VBENCH DONE
