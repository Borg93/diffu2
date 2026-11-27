#!/bin/bash
# 27 Nov: the DiT's projections in MXFP8 and row-wise FP8 inside SGLang (the plugin's DIFFU_SGLANG_DIT_PRECISION, added
# today), on GPU 2 after the RL evaluations (rl_eval_after_moss.sh has started the pool read, which is paused for this
# and started again after): the kernel check, then the same 12 captured pages as moss_bench_set40.sh's SGLang runs,
# eager and with regional compile, for each precision. Log: runs/htr/moss_bench_quant.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
B=/mnt/scratch/gabriel/moss_bench; O=$B/set40/out
until grep -q "pool read started" runs/htr/rl_eval_after_moss.log 2>/dev/null; do sleep 60; done
sleep 120; log "pausing the pool read: pool_v4 $(wc -l < runs/htr/curate/pool_read/pool_v4_vllm.jsonl) pages"
tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 45
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:/mnt/scratch/gabriel/envs/sglang_moss/bin:$PATH
log "check_b_quantized_dit"
PYTHONPATH=$PWD/packages/sglang_moss/src:$PWD/runs/htr/moss_bench timeout 3600 /mnt/scratch/gabriel/envs/sglang_moss/bin/python runs/htr/moss_bench/check_b_quantized_dit.py > $O/check_b_quantized_dit.log 2>&1 \
  && log "check_b done: $(tail -n 8 $O/check_b_quantized_dit.log | tr '\n' ' ' | cut -c1-700)" || log "check_b FAILED: $(tail -n 4 $O/check_b_quantized_dit.log | cut -c1-400)"
base='"disable_autocast": true, "attention_backend": "torch_cudnn_sdpa", "component_attention_backends": {"text_encoder": "torch_sdpa"}'
for p in mxfp8 fp8_rowwise; do
  for mode in eager regional; do
    [ $mode = eager ] && { server="{$base}"; ext=diffu_sglang_moss; } || { server="{$base, \"enable_torch_compile\": true, \"regional_compile\": true}"; ext=moss_bench_ext; }
    kind=quant_${p}_$mode; log "sglang $kind"
    ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $O/sglang_$kind.mem ) & watch=$!
    DIFFU_SGLANG_DIT_PRECISION=$p PYTHONPATH=$PWD/runs/htr/moss_bench SGLANG_EXTERNAL_MODEL_PACKAGE=$ext timeout 5400 /mnt/scratch/gabriel/envs/sglang_moss/bin/python $B/sglang_bench.py \
        /mnt/scratch/gabriel/models/moss/hf-diffu-moss-above $O/captured12 $O/sglang_$kind "$server" '{}' > $O/sglang_$kind.log 2>&1 \
      && log "sglang $kind done; peak $(sort -n $O/sglang_$kind.mem | tail -n 1) MiB; seconds $(grep '^{"page' $O/sglang_$kind.log | python3 -c "import sys,json;print([json.loads(l).get('seconds') for l in sys.stdin])")" \
      || log "sglang $kind FAILED: $(grep -v '^ *$' $O/sglang_$kind.log | tail -n 5 | cut -c1-400)"
    kill $watch 2>/dev/null; sleep 10
  done
done
log "finished (quant)"
sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
