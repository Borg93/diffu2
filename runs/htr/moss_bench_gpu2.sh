#!/bin/bash
# 26 Nov: the generator's drawing speed on GPU 2 (RTX PRO 6000 Blackwell), INFERENCE.md section 11, once the reader's
# work of the evening is done (runs/htr/rlvr_headroom.sh with RESUME=0 has logged "GPU 2 left free"). Three 2 MP pages
# with a sample page (/mnt/scratch/gabriel/moss_bench/run.jsonl), guided at the demo's settings (Euler 28 steps,
# guidance 2 over the first half): eager and compiled, each with a profiler trace of pages 2-3; then the joint block's
# package built for this card and diffu_moss.bench_inference at a page of the same token count (about 14,000).
# Only GPU 2. Log: runs/htr/moss_bench_gpu2.log; outputs under /mnt/scratch/gabriel/moss_bench.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
B=/mnt/scratch/gabriel/moss_bench; M=/mnt/scratch/gabriel/models/moss/moss-diffu-moss-above
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
until grep -q "GPU 2 left free" runs/htr/rlvr_headroom.log 2>/dev/null; do
  tmux has-session -t =headroom 2>/dev/null || { log "the evening's queue is gone: going on"; break; }
  sleep 60
done
sleep 20; log "GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
draw() {  # name, extra flags
  name=$1; shift; log "draw $name"
  ( time uv run diffu-generate --jobs $B/run.jsonl --model $M --out $B/$name --steps 28 --guidance 2 --guidance-stop 0.5 \
      --whole-page --profile $B/trace_$name "$@" ) > $B/$name.log 2>&1 && log "draw $name done" || log "draw $name FAILED: $(tail -n 3 $B/$name.log | cut -c1-300)"
}
draw eager
draw compiled --compile-blocks
log "package"
uv run python -m diffu_moss.aoti --model $M --precision bf16 --tokens 14000 --out $B/moss_bf16_sm120.pt2 > $B/aoti.log 2>&1 && log "package built" || log "package FAILED: $(tail -n 3 $B/aoti.log | cut -c1-300)"
log "bench_inference"
uv run python -m diffu_moss.bench_inference --model $M --size 1568x2272 --steps 8 --package-bf16 $B/moss_bf16_sm120.pt2 \
  --config bf16_bf16_eager --config bf16_bf16_compiled --config bf16_bf16_packaged --config mxfp8_bf16_compiled \
  --out $B/inference.jsonl --images $B/bench_pages > $B/bench_inference.log 2>&1 && log "bench_inference done" || log "bench_inference FAILED: $(tail -n 3 $B/bench_inference.log | cut -c1-300)"
log "finished; GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
