#!/bin/bash
# The draft run's own serve, prepare-data and train commands on 2,400 of its samples (seeded draw), on GPU 2 (idle,
# held for the check) and port 8001, twice: as the run has them (1 API server, 12 preparation workers, 2 dataloader
# workers) and with 4 API servers, 48 preparation workers and 8 dataloader workers. Wall times in times.log.
cd /home/gabriel/Desktop/diffu2 || exit 1
T=runs/htr/dspark-speedtest
VLLM=packages/htr_train/envs/vllm/.venv/bin/vllm
SPEC=packages/htr_train/envs/speculators/.venv/bin/speculators
CK=/home/gabriel/Desktop/diffu2/runs/htr/minicpm-all/output/checkpoint-28000
python3 - <<'EOF'
import random
rows = open("data/htr/dspark_responses.jsonl", encoding="utf-8").readlines()
pick = sorted(random.Random(1).sample(range(len(rows)), 2400))
open("runs/htr/dspark-speedtest/responses.jsonl", "w", encoding="utf-8").writelines(rows[i] for i in pick)
EOF
note() { echo "[$(date '+%T')] $*" | tee -a $T/times.log; }
case_run() {  # case_run NAME API_SERVERS PREP_WORKERS LOADER_WORKERS
  local name=$1 servers=$2 prep=$3 loaders=$4 start
  CUDA_VISIBLE_DEVICES=2 $VLLM serve $CK --enable-scale-out --served-model-name $CK --port 8001 \
    --gpu-memory-utilization 0.35 --allowed-local-media-path /mnt/work/gabriel/htr --max-model-len 4097 \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [4, 8, 12, 16, 20, 24]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "'"$PWD/$T/hidden_states_$name"'"}}' \
    --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
    --default-chat-template-kwargs '{"enable_thinking": false}' --limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --api-server-count "$servers" > $T/serve_$name.log 2>&1 &
  local served=$!
  for _ in $(seq 120); do curl -sf localhost:8001/health > /dev/null && break; sleep 5; done
  curl -sf localhost:8001/health > /dev/null || { note "$name: server did not come up"; kill $served; return 1; }
  start=$(date +%s)
  $SPEC prepare-data --model $CK --data $T/responses.jsonl --output $PWD/$T/data_$name --seq-length 4096 \
    --render-endpoint http://localhost:8001 --no-skip-token-freq --overwrite --num-preprocessing-workers "$prep" \
    > $T/prep_$name.log 2>&1
  note "$name: prepare-data exit $?, $(( $(date +%s) - start )) s for 2,400 samples"
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES=2 $SPEC train --verifier-name-or-path $CK --data-path $PWD/$T/data_$name \
    --vllm-endpoint http://localhost:8001/v1 --on-missing generate --save-path $PWD/$T/output_$name \
    --speculator-type dspark --num-layers 5 --block-size 15 --target-layer-ids 4 8 12 16 20 --draft-vocab-size 32000 \
    --mask-token-id 248076 --markov-rank 256 --markov-head-type vanilla --enable-confidence-head \
    --confidence-head-with-markov --confidence-head-alpha 1.0 --max-anchors 3072 --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --lr 0.0003 --scheduler-type cosine --scheduler-warmup-ratio 0.04 --epochs 1 --total-seq-len 4096 \
    --num-workers "$loaders" --logger trackio --run-name speedtest-$name > $T/train_$name.log 2>&1
  note "$name: train exit $?, $(( $(date +%s) - start )) s for 1 epoch of 2,400 samples (start-up included)"
  kill $served; wait $served 2>/dev/null
}
case_run asis 1 12 2
case_run scaled 4 48 8
note "done"
