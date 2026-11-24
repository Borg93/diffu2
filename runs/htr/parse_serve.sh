#!/bin/bash
# CURATION E50: our Nemotron Parse checkpoint served by vLLM as the model's repository documents (native Parse, DFlash2
# draft, its repetition-stop processor). GPU=1|2 PORT=8011 FRACTION=0.45 SEQS=64 DRAFT=16 (0: no draft) bash runs/htr/parse_serve.sh [MODEL_DIR]
cd /home/gabriel/Desktop/diffu2 || exit 1
GPU=${GPU:-2}; case "$GPU" in 2) ;; *) echo "GPU must be 2 (from 20 Nov GPU 1 is another person's)"; exit 1;; esac
D=${1:-/mnt/scratch/gabriel/htr/serve/parse-more}; DRAFT=${DRAFT:-16}
SPEC=(); [ "$DRAFT" != 0 ] && SPEC=(--speculative-config "{\"method\": \"dflash\", \"model\": \"$D/dflash2\", \"num_speculative_tokens\": $DRAFT}")
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 PYTHONPATH=/mnt/scratch/gabriel/models/nemotron-3.5-parse-repo/logitsprocs
# nvcc lives inside the vLLM environment (a pip package), not on the path: with CUDA_HOME set, vLLM stops reporting the
# FlashInfer and DeepGEMM kernels as disabled (18 Nov)
E=$PWD/packages/htr_train/envs/vllm/.venv; export CUDA_HOME=$E/lib/python3.12/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$E/bin:$PATH
exec packages/htr_train/envs/vllm/.venv/bin/vllm serve $D --served-model-name nemotron-parse --trust-remote-code --host 127.0.0.1 --port ${PORT:-8011} --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs ${SEQS:-64} --max-num-batched-tokens 32768 --gpu-memory-utilization ${FRACTION:-0.45} \
  --no-enable-prefix-caching --mm-processor-cache-gb 4 --enable-chunked-prefill --async-scheduling \
  --attention-config "${ATTN:-{\"backend\": \"FLASH_ATTN\", \"flash_attn_version\": 3\}}" \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 17, 32, 34, 64, 68, 128, 136, 256, 272, 512, 544, 1024, 1088]}' \
  "${SPEC[@]}"  # the repository's repetition-stop processor is written for the older runner: this vLLM refuses it (E51)
