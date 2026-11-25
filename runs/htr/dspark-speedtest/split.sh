#!/bin/bash
# The scaled case's data, vLLM on GPU 2 (4 API servers), the trainer on GPU 1 (its compute idle while the draft run
# prepares its data; the run's vLLM there keeps its own memory), GPU load logged once a second.
cd /home/gabriel/Desktop/diffu2 || exit 1
T=runs/htr/dspark-speedtest
VLLM=packages/htr_train/envs/vllm/.venv/bin/vllm
SPEC=packages/htr_train/envs/speculators/.venv/bin/speculators
CK=/home/gabriel/Desktop/diffu2/runs/htr/minicpm-all/output/checkpoint-28000
CUDA_VISIBLE_DEVICES=2 $VLLM serve $CK --enable-scale-out --served-model-name $CK --port 8001 \
  --gpu-memory-utilization 0.35 --allowed-local-media-path /mnt/work/gabriel/htr --max-model-len 4097 \
  --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [4, 8, 12, 16, 20, 24]}}}' \
  --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "'"$PWD/$T/hidden_states_split"'"}}' \
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
  --default-chat-template-kwargs '{"enable_thinking": false}' --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --api-server-count 4 > $T/serve_split.log 2>&1 &
served=$!
for _ in $(seq 120); do curl -sf localhost:8001/health > /dev/null && break; sleep 5; done
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -i 1,2 -l 1 > $T/gpu_split.csv &
watch=$!
start=$(date +%s)
CUDA_VISIBLE_DEVICES=1 $SPEC train --verifier-name-or-path $CK --data-path $PWD/$T/data_scaled \
  --vllm-endpoint http://localhost:8001/v1 --on-missing generate --save-path $PWD/$T/output_split \
  --speculator-type dspark --num-layers 5 --block-size 15 --target-layer-ids 4 8 12 16 20 --draft-vocab-size 32000 \
  --mask-token-id 248076 --markov-rank 256 --markov-head-type vanilla --enable-confidence-head \
  --confidence-head-with-markov --confidence-head-alpha 1.0 --max-anchors 3072 --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --lr 0.0003 --scheduler-type cosine --scheduler-warmup-ratio 0.04 --epochs 1 --total-seq-len 4096 \
  --num-workers 8 --logger trackio --run-name speedtest-split > $T/train_split.log 2>&1
echo "[$(date '+%T')] split: train exit $?, $(( $(date +%s) - start )) s" >> $T/times.log
kill $watch $served; wait $served 2>/dev/null
