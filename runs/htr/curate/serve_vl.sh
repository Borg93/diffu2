#!/bin/bash
# Nemotron 3.5 Super VL (NVFP4) served on one GPU for describing scans (CURATION task 9), with the flags of NVIDIA's
# cookbook for one RTX Pro 6000 (NVIDIA-NeMo/Nemotron usage-cookbook/Nemotron-3.5-Super-VL/vllm_cookbook.ipynb). GPU and port: serve_vl.sh [GPU] [PORT]. Logs to runs/htr/curate/vl/serve.log.
cd /home/gabriel/Desktop/diffu2 || exit 1
GPU=${1:-1}; PORT=${2:-8011}; mkdir -p runs/htr/curate/vl
# In the card's container (vllm/vllm-openai:v0.32.0, 33 GB): our own vLLM environment holds no built flashinfer
# kernels and cannot build them (its nvcc and CUDA headers disagree).
docker rm -f nemotron-vl 2>/dev/null
exec docker run --name nemotron-vl --gpus "\"device=$GPU\"" --ipc=host -p 127.0.0.1:"$PORT":8000 \
  -v /mnt/scratch/gabriel/models/nemotron-3.5-super-vl-nvfp4:/model:ro -v /mnt/scratch/gabriel/models/vllm_cache:/root/.cache \
  vllm/vllm-openai:v0.32.0 /model \
  --served-model-name nemotron-3.5-super-vl --trust-remote-code \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 16384 --max-num-seqs 128 \
  --speculative_config.method mtp --speculative_config.num_speculative_tokens 5 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 > runs/htr/curate/vl/serve.log 2>&1
