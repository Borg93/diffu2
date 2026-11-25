#!/bin/bash
# The reader served from the Hub exactly as its card says (only the port differs: 8000 is the draft's target server)
cd /home/gabriel/Desktop/diffu2
export CUDA_VISIBLE_DEVICES=1
exec packages/htr_train/envs/vllm/.venv/bin/vllm serve Gabriel/diffu-minicpm-ocr-htr --served-model-name htr --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 8045
