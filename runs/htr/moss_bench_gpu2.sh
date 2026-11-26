#!/bin/bash
# 26 Nov: the generator's drawing speed on GPU 2 (RTX PRO 6000 Blackwell), once the reader's work of the evening is
# done (runs/htr/rlvr_headroom.sh with RESUME=0 has logged "GPU 2 left free"). INFERENCE.md section 12, "Trying every
# option on one workstation": tests A to H, each `diffu-generate` with its draw.json on the same real pages (two ALTO
# openings of our pool, drawn as four 2 MP pages, the Space's sample gota-hovratt-1701.jpg, seed 7), the peak memory
# of GPU 2 sampled each second. Then the same pages by SGLang Diffusion (section 13), and, apart (the profiler slows a
# page), traces of an eager and a compiled run. Only GPU 2. Log: runs/htr/moss_bench_gpu2.log; outputs /mnt/scratch/gabriel/moss_bench.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
B=/mnt/scratch/gabriel/moss_bench; M=/mnt/scratch/gabriel/models/moss/moss-diffu-moss-above
T=/mnt/scratch/gabriel/htr/pool_v3/C0074421/alto
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
until grep -q "GPU 2 left free" runs/htr/rlvr_headroom.log 2>/dev/null; do
  tmux has-session -t =headroom 2>/dev/null || { log "the evening's queue is gone: going on"; break; }
  sleep 60
done
sleep 20; log "GPU 2: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader -i 2)"
log "package (for C)"
[ -f $B/moss_bf16_sm120.pt2 ] || uv run python -m diffu_moss.aoti --model $M --out $B/moss_bf16_sm120.pt2 > $B/aoti.log 2>&1 && log "package built" || log "package FAILED: $(tail -n 3 $B/aoti.log | cut -c1-300)"
for t in ${TESTS:-A B C D E F G H}; do
  log "test $t: $(cat $B/settings/$t.json)"
  ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $B/speed_$t.mem ) & watch=$!
  [ $t = A ] && run="env CAPTURE_DIR=$B/captured uv run python $B/capture_generate.py" || run="uv run diffu-generate"
  $run --model $M --templates $T/C0074421_00154.xml --templates $T/C0074421_00156.xml \
      --text-from $T/C0074421_00154.xml --text-from $T/C0074421_00156.xml --title "the court records of a häradsrätt" --years 1829 \
      --reference $B/pagedemo/samples/gota-hovratt-1701.jpg --reference-caption sample --whole-page --page-megapixels 2 \
      --seed 7 --draw-settings $B/settings/$t.json --out $B/speed/$t > $B/speed_$t.log 2>&1 \
    && log "test $t done; peak $(sort -n $B/speed_$t.mem | tail -n 1) MiB" || log "test $t FAILED: $(tail -n 4 $B/speed_$t.log | cut -c1-400)"
  kill $watch 2>/dev/null
done
for name in eager compiled; do
  [ $name = compiled ] && flag=--compile-blocks || flag=
  log "trace $name"
  uv run diffu-generate --jobs $B/run.jsonl --model $M --out $B/$name --steps 28 --guidance 2 --guidance-stop 0.5 --whole-page \
      --profile $B/trace_$name $flag > $B/$name.log 2>&1 && log "trace $name done" || log "trace $name FAILED: $(tail -n 3 $B/$name.log | cut -c1-300)"
done
# The target is SGLang Diffusion inside Ray Data (INFERENCE.md section 13), so: the same pages (test A's prompts and
# samples, captured as it drew) drawn by SGLang with our plugin, its attention kernels and quality tiers; compare with
# test H (guided every step, as SGLang guides). The demo Space is not measured.
# SGLang's MOSS-Image (sglang 0.5.23) has validated recipes for the attention kernel, the quality tier and batching;
# compiling and breakable CUDA graphs are "not validated", which is a reason to try them and compare the pages with
# the exact run's, not to leave them out. Multi-GPU levers (CFG and sequence parallelism) need more than GPU 2.
for kind in fa_lossless fa_exact torch_sdpa cudnn_sdpa fa_high batch2 cuda_graph compiled; do
  extra='{}'; ext=diffu_sglang_moss
  case $kind in
    fa_lossless) server='{"disable_autocast": true}';;
    fa_exact) server='{"disable_autocast": true}'; extra='{"quality": "exact"}';;
    torch_sdpa) server='{"disable_autocast": true, "attention_backend": "torch_sdpa"}';;
    cudnn_sdpa) server='{"disable_autocast": true, "attention_backend": "torch_cudnn_sdpa"}';;
    fa_high) server='{"disable_autocast": true}'; extra='{"quality": "high"}';;                      # Q, K, V as one GEMM
    batch2) server='{"disable_autocast": true}'; extra='{"num_outputs_per_prompt": 2}';;             # two pages a call
    cuda_graph) server='{"disable_autocast": true, "enable_breakable_cuda_graph": true}';;           # not validated for MOSS-Image: tried
    compiled) server='{"disable_autocast": true, "enable_torch_compile": true}'; ext=moss_bench_ext;; # refused by SGLang: its check lifted (runs/htr/moss_bench/moss_bench_ext)
  esac
  log "sglang $kind"
  ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $B/sglang_$kind.mem ) & watch=$!
  PYTHONPATH=$PWD/runs/htr/moss_bench SGLANG_EXTERNAL_MODEL_PACKAGE=$ext /mnt/scratch/gabriel/envs/sglang_moss/bin/python $B/sglang_bench.py \
      /mnt/scratch/gabriel/models/moss/hf-diffu-moss-above $B/captured $B/sglang_$kind "$server" "$extra" > $B/sglang_$kind.log 2>&1 \
    && log "sglang $kind done; peak $(sort -n $B/sglang_$kind.mem | tail -n 1) MiB: $(grep '^{' $B/sglang_$kind.log | tr '\n' ' ' | cut -c1-500)" || log "sglang $kind FAILED: $(tail -n 4 $B/sglang_$kind.log | cut -c1-400)"
  kill $watch 2>/dev/null; sleep 10
done
# SGLang's own profiler trace (PyTorch profiler, 3 denoising steps after a warm-up step a page), apart from the timed
# runs: a trace slows a page. Read with the pytorch profile-model skill, beside our pipeline's two traces.
log "sglang trace"
PYTHONPATH=$PWD/runs/htr/moss_bench SGLANG_EXTERNAL_MODEL_PACKAGE=diffu_sglang_moss SGLANG_DIFFUSION_TORCH_PROFILER_DIR=$B/trace_sglang \
  /mnt/scratch/gabriel/envs/sglang_moss/bin/python $B/sglang_bench.py /mnt/scratch/gabriel/models/moss/hf-diffu-moss-above $B/captured $B/sglang_trace \
  '{"disable_autocast": true}' '{"profile": true, "num_profiled_timesteps": 3}' > $B/sglang_trace.log 2>&1 \
  && log "sglang trace done: $(ls $B/trace_sglang 2>/dev/null | wc -l) files" || log "sglang trace FAILED: $(tail -n 4 $B/sglang_trace.log | cut -c1-400)"
log "finished; GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
