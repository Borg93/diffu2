#!/bin/bash
# 27 Nov 18:10, in place of moss_bench_quant.sh: SGLang's fastest stacks, on GPU 2 after the RL evaluations
# (rl_eval_after_moss.sh has started the pool read, which is paused for this and started again after), on the same 12
# captured pages as moss_bench_set40.sh's SGLang runs. In order: the MXFP8 kernel check; fp8_fa_sm120 again (it failed
# at 17:39 on SGLang's meta-device import, mended in the plugin's __init__); then the stacks a production run would use,
# all with guidance stopped at half the steps (our setting) and regional compile: bf16, MXFP8, row-wise FP8, MXFP8 with
# FP8 attention, MXFP8 with SageAttention 2, and MXFP8 without compile, every page from our pipeline's own noise seed
# (18:40: the earlier SGLang runs all used seed 7, so their pages could not be set beside ours; seeded_bf16_eager is
# the plain anchor); last the read-back of every SGLang run
# (score_sglang.py, the TrOCR judge as diffu-generate scores). Log: runs/htr/moss_bench_quant.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
B=/mnt/scratch/gabriel/moss_bench; O=$B/set40/out
if [ "${NOWAIT:-0}" != 1 ]; then  # NOWAIT=1 (after_reboot.sh): GPU 2 is free, start at once and leave the pool read to the caller
  until grep -q "pool read started" runs/htr/rl_eval_after_moss.log 2>/dev/null; do sleep 60; done
  sleep 120; log "pausing the pool read: pool_v4 $(wc -l < runs/htr/curate/pool_read/pool_v4_vllm.jsonl) pages"
  tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 45
fi
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:/mnt/scratch/gabriel/envs/sglang_moss/bin:$PATH
log "check_b_quantized_dit"
PYTHONPATH=$PWD/packages/sglang_moss/src:$PWD/runs/htr/moss_bench timeout 3600 /mnt/scratch/gabriel/envs/sglang_moss/bin/python runs/htr/moss_bench/check_b_quantized_dit.py > $O/check_b_quantized_dit.log 2>&1 \
  && log "check_b done: $(tail -n 8 $O/check_b_quantized_dit.log | tr '\n' ' ' | cut -c1-700)" || log "check_b FAILED: $(tail -n 4 $O/check_b_quantized_dit.log | cut -c1-400)"
cudnn='"attention_backend": "torch_cudnn_sdpa"'; fp8a='"attention_backend": "fp8_fa_sm120"'; sage='"attention_backend": "sage_attn"'
regional=', "enable_torch_compile": true, "regional_compile": true'; gs='{"guidance_stop": 0.5, "page_seeds": true}'  # each page from our pipeline's own noise seed
# kind | precision (none = bf16) | attention | compile | sampling | python env
for row in "fp8_attention|none|$fp8a||{}|sglang_moss" \
           "seeded_bf16_eager|none|$cudnn||$gs|sglang_moss" \
           "stack_bf16|none|$cudnn|$regional|$gs|sglang_moss" \
           "stack_mxfp8|mxfp8|$cudnn|$regional|$gs|sglang_moss" \
           "stack_fp8_rowwise|fp8_rowwise|$cudnn|$regional|$gs|sglang_moss" \
           "stack_mxfp8_fp8attn|mxfp8|$fp8a|$regional|$gs|sglang_moss" \
           "stack_mxfp8_sage|mxfp8|$sage|$regional|$gs|attn_sm120" \
           "stack_mxfp8_eager|mxfp8|$cudnn||$gs|sglang_moss"; do
  IFS='|' read -r kind p attn comp extra env <<< "$row"
  server="{\"disable_autocast\": true, $attn, \"component_attention_backends\": {\"text_encoder\": \"torch_sdpa\"}$comp}"
  [ -n "$comp" ] && ext=moss_bench_ext || ext=diffu_sglang_moss
  [ $p = none ] && prec= || prec=$p
  log "sglang $kind: $server $extra ${prec:+precision $prec}"
  ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $O/sglang_$kind.mem ) & watch=$!
  ${prec:+env DIFFU_SGLANG_DIT_PRECISION=$prec} env PYTHONPATH=$PWD/runs/htr/moss_bench SGLANG_EXTERNAL_MODEL_PACKAGE=$ext timeout 5400 /mnt/scratch/gabriel/envs/$env/bin/python $B/sglang_bench.py \
      /mnt/scratch/gabriel/models/moss/hf-diffu-moss-above $O/captured12 $O/sglang_$kind "$server" "$extra" > $O/sglang_$kind.log 2>&1 \
    && log "sglang $kind done; peak $(sort -n $O/sglang_$kind.mem | tail -n 1) MiB; seconds $(grep '^{"page' $O/sglang_$kind.log | python3 -c "import sys,json;print([json.loads(l).get('seconds') for l in sys.stdin])")" \
    || log "sglang $kind FAILED: $(grep -v '^ *$' $O/sglang_$kind.log | tail -n 5 | cut -c1-400)"
  kill $watch 2>/dev/null; sleep 10
done
log "read-back of every SGLang run"
timeout 3600 uv run python runs/htr/moss_bench/score_sglang.py cuda > $O/scores.log 2>&1 && log "scores: $(grep CER $O/scores.log | tr '\n' ';')" || log "scores FAILED: $(tail -n 4 $O/scores.log | cut -c1-400)"
log "finished (quant2)"
[ "${NOWAIT:-0}" = 1 ] && exit 0
sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
