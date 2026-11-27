#!/bin/bash
# 27 Nov: the generator's speed options judged on 40 clean pages (set40: 20 openings of 20 volumes, each page its own
# text, four samples; runs/htr/moss_bench/build_set40.py), on GPU 2, after the RL pilot's queue (rlvr_queue4.sh) has
# started the pool read, which is paused for this and started again after. In order: the attention kernels timed
# alone; our pipeline (R the reference: compiled bfloat16; F MXFP8; K06, K12, K24 the first-block cache with warm-up 4
# and at most 3 skips in a row, compiled; K12F the cache with MXFP8; D UniPC in 16 steps); SGLang with the plugin on
# the first 12 of the same pages (exact; CFG gating; Cache-DiT and its calibrators; FP8 attention; autotune; cuBLASLt;
# compiled; breakable CUDA graph: E75's audit).
# Log: runs/htr/moss_bench_set40.log; outputs /mnt/scratch/gabriel/moss_bench/set40/out.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
B=/mnt/scratch/gabriel/moss_bench; S40=$B/set40; O=$S40/out; M=/mnt/scratch/gabriel/models/moss/moss-diffu-moss-above; HOME_PATH=$PATH
mkdir -p $O
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
toolkit() { export CUDA_HOME=$1/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:${1%/lib/python*}/bin:$HOME_PATH; }
watch_memory() { ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $1 ) & watch=$!; }
if [ "${WAIT:-1}" = 1 ]; then
  until grep -q "pool read started" runs/htr/rlvr_queue4.log 2>/dev/null; do
    tmux has-session -t =queue4 2>/dev/null || { log "the RL queue is gone without starting the pool read: going on"; break; }
    sleep 60
  done
  sleep 120; log "pausing the pool read: pool_v4 $(wc -l < runs/htr/curate/pool_read/pool_v4_vllm.jsonl) pages"
  tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 45
fi
export CUDA_VISIBLE_DEVICES=2
log "GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"

toolkit /mnt/scratch/gabriel/envs/sglang_moss/lib/python3.12
log "attention kernels"
/mnt/scratch/gabriel/envs/sglang_moss/bin/python -I $B/bench_attn_sm120.py 15700 > $O/attention.log 2>&1 && log "attention done: $(grep -c ms $O/attention.log) lines" || log "attention FAILED: $(tail -n 3 $O/attention.log | cut -c1-300)"

toolkit $PWD/packages/htr_train/envs/vllm/.venv/lib/python3.12
for t in ${OURS-R F K12 K06 K24 K12F D}; do
  log "ours $t: $(cat $B/settings40/$t.json)"; watch_memory $O/$t.mem; rm -rf $O/$t
  [ $t = R ] && run="env CAPTURE_DIR=$O/captured uv run python $B/capture_generate.py" || run="uv run diffu-generate"
  [ $t = R ] && rm -rf $O/captured
  $run --model $M --jobs $S40/jobs.jsonl --draw-settings $B/settings40/$t.json --out $O/$t > $O/$t.log 2>&1 \
    && log "ours $t done; peak $(sort -n $O/$t.mem | tail -n 1) MiB; median s $(python3 -c "import json,statistics;c=json.load(open('$O/$t/summary.json'))['crops'];print(round(statistics.median(x['seconds'] for x in c[1:]),1), 'pages', len(c), 'mean cer', round(statistics.fmean(x['cer'] for x in c),4))")" \
    || log "ours $t FAILED: $(grep -v '^ *$' $O/$t.log | tail -n 5 | cut -c1-400)"
  kill $watch 2>/dev/null
done

# SGLang builds kernels against a toolkit laid out as /usr/local/cuda is (lib64/libcudart.so): cuda13 is the
# environment's pip toolkit in that shape (the CUDA graph run failed without it: "cannot find -lcudart").
export CUDA_HOME=/mnt/scratch/gabriel/envs/sglang_moss/cuda13; export PATH=$CUDA_HOME/bin:/mnt/scratch/gabriel/envs/sglang_moss/bin:$HOME_PATH
mkdir -p $O/captured12; for k in $(seq 0 11); do cp $O/captured/page_$k.* $O/captured12/ 2>/dev/null; done
# 27 Nov 12:30: the audit of every SGLang Diffusion option (E75) adds CFG gating (SGLang's stand-in for our guidance
# stop), Cache-DiT's calibrators and step mask, attention autotune and cuBLASLt; cuDNN attention under all of them.
for kind in ${SGLANG-exact cfg_gate05 cfg_gate075 cache_default cache_012 cache_012_taylorseer cache_012_dmd cache_scm_slow fp8_attention autotune cublaslt compiled cuda_graph}; do
  server='{"disable_autocast": true, "attention_backend": "torch_cudnn_sdpa", "component_attention_backends": {"text_encoder": "torch_sdpa"}}'; ext=diffu_sglang_moss; blas=
  case $kind in
    exact) extra='{"quality": "exact"}';;
    cfg_gate05) extra='{"cfg_gate_step": 0.5}';;
    cfg_gate075) extra='{"cfg_gate_step": 0.75}';;
    cache_012_taylorseer) extra='{"enable_cache_dit": true, "cache_dit_params": {"residual_diff_threshold": 0.12, "enable_taylorseer": true, "taylorseer_order": 1}}';;
    cache_012_dmd) extra='{"enable_cache_dit": true, "cache_dit_params": {"residual_diff_threshold": 0.12, "enable_dmd": true, "dmd_history": 6}}';;
    cache_scm_slow) extra='{"enable_cache_dit": true, "cache_dit_params": {"scm_preset": "slow"}}';;
    autotune) extra='{}'; server='{"disable_autocast": true, "enable_attention_backend_autotune": true}';;
    cublaslt) extra='{}'; blas=1;;
    cache_default) extra='{"enable_cache_dit": true}';;
    cache_012) extra='{"enable_cache_dit": true, "cache_dit_params": {"residual_diff_threshold": 0.12}}';;
    compiled) extra='{}'; server='{"disable_autocast": true, "attention_backend": "torch_cudnn_sdpa", "component_attention_backends": {"text_encoder": "torch_sdpa"}, "enable_torch_compile": true}'; ext=moss_bench_ext;;  # SGLang's refusal lifted
    cuda_graph) extra='{}'; server='{"disable_autocast": true, "enable_breakable_cuda_graph": true}';;
    fp8_attention) extra='{}'; server='{"disable_autocast": true, "attention_backend": "fp8_fa_sm120", "component_attention_backends": {"text_encoder": "torch_sdpa"}}';;
  esac
  log "sglang $kind"; watch_memory $O/sglang_$kind.mem
  ${blas:+env TORCH_BLAS_PREFER_CUBLASLT=1} env PYTHONPATH=$PWD/runs/htr/moss_bench SGLANG_EXTERNAL_MODEL_PACKAGE=$ext timeout 5400 /mnt/scratch/gabriel/envs/sglang_moss/bin/python $B/sglang_bench.py \
      /mnt/scratch/gabriel/models/moss/hf-diffu-moss-above $O/captured12 $O/sglang_$kind "$server" "$extra" > $O/sglang_$kind.log 2>&1 \
    && log "sglang $kind done; peak $(sort -n $O/sglang_$kind.mem | tail -n 1) MiB; seconds $(grep '^{"page' $O/sglang_$kind.log | python3 -c "import sys,json;print([json.loads(l).get('seconds') for l in sys.stdin])")" \
    || log "sglang $kind FAILED: $(grep -v '^ *$' $O/sglang_$kind.log | tail -n 5 | cut -c1-400)"
  kill $watch 2>/dev/null; sleep 10
done
log "finished (set40); GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
if [ "${WAIT:-1}" = 1 ]; then
  sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
fi
