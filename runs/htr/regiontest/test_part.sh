until [ -f $C/trainer_state.json ] && grep -q "'global_step/max_steps': '3010/" runs/htr/pipeline.log; do sleep 30; done
log "checkpoint-3000 written and training past it: stopping minicpm-all"
pkill -u gabriel -f "minicpm-all/output"
until ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -qiE "75:00|A3:00"; do sleep 5; done
log "GPUs 1 and 2 free"
E=packages/htr_train/envs/minicpm/.venv/bin/python
V=packages/htr_train/envs/vllm/.venv/bin/vllm
M=$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM-OCR-1.0/snapshots/2697d4e93dca82784abf858ae57206a49629cb6b
FLAGS=(--served-model-name minicpm --allowed-local-media-path /mnt/work/gabriel/htr --gpu-memory-utilization 0.9
  --max-model-len 16384 --max-num-seqs 256 --attention-backend FLASH_ATTN --mm-encoder-attn-backend FLASH_ATTN
  --max-num-batched-tokens 40960 --limit-mm-per-prompt '{"image": 4, "video": 0}'
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs
  --default-chat-template-kwargs '{"enable_thinking": false}')
CUDA_VISIBLE_DEVICES=1 $V serve $M --port 8021 "${FLAGS[@]}" > $O/vllm_regions.log 2>&1 &
S1=$!
CUDA_VISIBLE_DEVICES=2 $V serve $M --port 8022 "${FLAGS[@]}" > $O/vllm_lines.log 2>&1 &
S2=$!
until curl -sf localhost:8021/health > /dev/null && curl -sf localhost:8022/health > /dev/null; do
  kill -0 $S1 2>/dev/null && kill -0 $S2 2>/dev/null || { log "vLLM died: see $O/vllm_*.log"; break; }; sleep 5; done
$E -m diffu_htr_train.evaluate read --lines $O/regions.jsonl --reader minicpm --model minicpm --endpoint http://localhost:8021 --concurrency 64 --max-tokens 4096 --out $O/read_regions.jsonl > $O/read_regions.log 2>&1 &
R1=$!
$E -m diffu_htr_train.evaluate read --lines $O/lines.jsonl --reader minicpm --model minicpm --endpoint http://localhost:8022 --concurrency 64 --out $O/read_lines.jsonl > $O/read_lines.log 2>&1
wait $R1
kill $S1 $S2; wait $S1 $S2 2>/dev/null
$E $O/compare.py > $O/result.txt 2>&1; cat $O/result.txt
