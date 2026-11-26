#!/bin/bash
# 26 Nov: the generator's drawing speed on GPU 2 (RTX PRO 6000 Blackwell), once the reader's work of the evening is
# done (runs/htr/rlvr_headroom.sh with RESUME=0 has logged "GPU 2 left free"). INFERENCE.md section 12, "Trying every
# option on one workstation": tests A to H, each `diffu-generate` with its draw.json on the same real pages (two ALTO
# openings of our pool, drawn as four 2 MP pages, the Space's sample gota-hovratt-1701.jpg, seed 7), the peak memory
# of GPU 2 sampled each second. Then, apart (the profiler slows a page): traces of an eager and a compiled run, and
# diffu_moss.bench_inference. Only GPU 2. Log: runs/htr/moss_bench_gpu2.log; outputs /mnt/scratch/gabriel/moss_bench.
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
  uv run diffu-generate --model $M --templates $T/C0074421_00154.xml --templates $T/C0074421_00156.xml \
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
# INFERENCE.md section 11: the demo Space on this card, compiled and not, guided and fast, through its API (the local
# model folders in place of its downloads; bound to 127.0.0.1; nothing of the Space is changed or published).
for kind in compiled eager; do
  [ $kind = compiled ] && c=1 || c=0
  log "space $kind"
  ( cd $B/pagedemo && PAGEDEMO_COMPILE=$c PAGEDEMO_BASE=/mnt/scratch/gabriel/models/moss/MOSS-Image-Base PAGEDEMO_DIT=/mnt/scratch/gabriel/models/moss/dit/diffu-moss-above \
      GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7872 exec /mnt/scratch/gabriel/envs/pagedemo/bin/python app.py ) > $B/space_$kind.app.log 2>&1 & app=$!
  ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $B/space_$kind.mem ) & watch=$!
  for _ in $(seq 180); do curl -s -m 2 http://127.0.0.1:7872/ > /dev/null && break; kill -0 $app 2>/dev/null || break; sleep 10; done
  /mnt/scratch/gabriel/envs/pagedemo/bin/python $B/space_client.py http://127.0.0.1:7872 $B/space_$kind > $B/space_$kind.log 2>&1 \
    && log "space $kind done; peak $(sort -n $B/space_$kind.mem | tail -n 1) MiB: $(tr '\n' ' ' < $B/space_$kind.log | cut -c1-600)" || log "space $kind FAILED: $(tail -n 3 $B/space_$kind.log | cut -c1-300) | app: $(tail -n 2 $B/space_$kind.app.log | cut -c1-300)"
  kill $watch $app 2>/dev/null; wait $app 2>/dev/null; sleep 15
done
log "finished; GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
