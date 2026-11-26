#!/bin/bash
# 26 Nov 21:50: the parts of the evening's queue that failed for one reason, no CUDA toolkit on the path (nvcc is a pip
# package inside each environment: CUDA_HOME must name it), run again: the joint block's package and test C, SGLang's
# eight runs and its trace, our pipeline's two traces (--whole-page is refused with a job file: its lines give
# page_pixels), then the RL temperature test. Only GPU 2. Log: runs/htr/moss_bench_gpu2.log (appended).
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
B=/mnt/scratch/gabriel/moss_bench; M=/mnt/scratch/gabriel/models/moss/moss-diffu-moss-above
T=/mnt/scratch/gabriel/htr/pool_v3/C0074421/alto; HOME_PATH=$PATH
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
toolkit() { export CUDA_HOME=$1/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$HOME_PATH; }
watch_memory() { ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $1 ) & watch=$!; }

toolkit $PWD/packages/htr_train/envs/vllm/.venv/lib/python3.12   # the workspace's environment has no nvcc of its own
log "package (for C), CUDA_HOME=$CUDA_HOME"
[ -f $B/speed/C/summary.json ] || uv run python -m diffu_moss.aoti --model $M --out $B/moss_bf16_sm120.pt2 > $B/aoti.log 2>&1 && log "package built" || log "package FAILED: $(tail -n 4 $B/aoti.log | cut -c1-400)"
if [ -f $B/moss_bf16_sm120.pt2 ] && [ ! -f $B/speed/C/summary.json ]; then
  log "test C"; watch_memory $B/speed_C.mem; rm -rf $B/speed/C
  uv run diffu-generate --model $M --templates $T/C0074421_00154.xml --templates $T/C0074421_00156.xml \
      --text-from $T/C0074421_00154.xml --text-from $T/C0074421_00156.xml --title "the court records of a häradsrätt" --years 1829 \
      --reference $B/pagedemo/samples/gota-hovratt-1701.jpg --reference-caption sample --whole-page --page-megapixels 2 \
      --seed 7 --draw-settings $B/settings/C.json --out $B/speed/C > $B/speed_C.log 2>&1 \
    && log "test C done; peak $(sort -n $B/speed_C.mem | tail -n 1) MiB" || log "test C FAILED: $(tail -n 4 $B/speed_C.log | cut -c1-400)"
  kill $watch 2>/dev/null
fi

toolkit /mnt/scratch/gabriel/envs/sglang_moss/lib/python3.12
# 22:10: every SGLang run goes through moss_bench_ext, which lays the sample above the page as the model was trained
# (the plugin lays it over the page: SGLang then drew a copy of the sample, PSNR 16.6 dB to ours; with it 31.9 dB).
export MOSS_REFERENCE_PLACE=above
for kind in ${KINDS:-fa_lossless fa_exact torch_sdpa cudnn_sdpa fa_high batch2 cuda_graph compiled trace}; do
  extra='{}'; ext=moss_bench_ext; unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR
  case $kind in
    fa_lossless) server='{"disable_autocast": true}';;
    fa_exact) server='{"disable_autocast": true}'; extra='{"quality": "exact"}';;
    torch_sdpa) server='{"disable_autocast": true, "attention_backend": "torch_sdpa"}';;
    cudnn_sdpa) server='{"disable_autocast": true, "attention_backend": "torch_cudnn_sdpa"}';;
    fa_high) server='{"disable_autocast": true}'; extra='{"quality": "high"}';;
    batch2) server='{"disable_autocast": true}'; extra='{"num_outputs_per_prompt": 2}';;
    cuda_graph) server='{"disable_autocast": true, "enable_breakable_cuda_graph": true}';;
    compiled) server='{"disable_autocast": true, "enable_torch_compile": true}'; ext=moss_bench_ext;;
    trace) server='{"disable_autocast": true}'; extra='{"profile": true, "num_profiled_timesteps": 3}'; export SGLANG_DIFFUSION_TORCH_PROFILER_DIR=$B/trace_sglang;;
  esac
  log "sglang $kind"; watch_memory $B/sglang_$kind.mem
  PYTHONPATH=$PWD/runs/htr/moss_bench SGLANG_EXTERNAL_MODEL_PACKAGE=$ext timeout 3600 /mnt/scratch/gabriel/envs/sglang_moss/bin/python $B/sglang_bench.py \
      /mnt/scratch/gabriel/models/moss/hf-diffu-moss-above $B/captured $B/sglang_$kind "$server" "$extra" > $B/sglang_$kind.log 2>&1 \
    && log "sglang $kind done; peak $(sort -n $B/sglang_$kind.mem | tail -n 1) MiB: $(grep '^{' $B/sglang_$kind.log | tr '\n' ' ' | cut -c1-600)" || log "sglang $kind FAILED: $(tail -n 5 $B/sglang_$kind.log | cut -c1-500)"
  kill $watch 2>/dev/null; sleep 10
done

toolkit $PWD/packages/htr_train/envs/vllm/.venv/lib/python3.12
for name in eager compiled; do
  [ $name = compiled ] && flag=--compile-blocks || flag=
  log "trace $name"
  uv run diffu-generate --jobs $B/run.jsonl --model $M --out $B/$name --steps 28 --guidance 2 --guidance-stop 0.5 \
      --profile $B/trace_$name $flag > $B/$name.log 2>&1 && log "trace $name done" || log "trace $name FAILED: $(tail -n 4 $B/$name.log | cut -c1-400)"
done

toolkit /mnt/scratch/gabriel/envs/parse_rlvr/venv/lib/python3.13
E=/mnt/scratch/gabriel/htr/rlvr/eval; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
for name in sft_dev974 pilot2b_s256_dev974; do
  log "headroom $name"
  HTR_RLVR_CONFIG=$PWD/runs/htr/rlvr/grpo_htr_v2.yaml PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:runs/htr/rlvr \
    $PY runs/htr/rlvr/headroom.py $E/$name/view $E/headroom_${name%_dev974} --temperatures 0.3,0.6,1.0 --samples 8 > $E/headroom_${name%_dev974}.log 2>&1 \
    && log "headroom $name done: $(grep -c '^temperature' $E/headroom_${name%_dev974}.log) temperatures" || log "headroom $name FAILED: $(grep -n 'Error' $E/headroom_${name%_dev974}.log | tail -n 3 | cut -c1-300)"
done
log "finished (second round); GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
