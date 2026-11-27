#!/bin/bash
# 27 Nov: the DiT's GEMMs (48 % of a compiled pass by the traces) in MXFP8 and NVFP4 by torchao, PyTorch's recipe for
# Blackwell, on GPU 2: the same four pages and settings as test B (compiled, 28 steps, guidance 2 over the first half),
# in the torchao environment (torch and torchao nightlies): bfloat16 as its baseline, then each format.
# Log: runs/htr/moss_bench_gpu2.log (appended); outputs /mnt/scratch/gabriel/moss_bench/speed/ao_*.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
B=/mnt/scratch/gabriel/moss_bench; M=/mnt/scratch/gabriel/models/moss/moss-diffu-moss-above
T=/mnt/scratch/gabriel/htr/pool_v3/C0074421/alto; E=/mnt/scratch/gabriel/envs/torchao_bw
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
export CUDA_HOME=$E/lib/python3.12/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$E/bin:$PATH
for f in ${FORMATS:-none mxfp8 nvfp4}; do
  log "torchao $f (compiled)"
  ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2; done > $B/speed_ao_$f.mem ) & watch=$!
  rm -rf $B/speed/ao_$f
  TORCHAO=$f timeout ${LIMIT:-2400} $E/bin/python $B/torchao_generate.py --model $M --templates $T/C0074421_00154.xml --templates $T/C0074421_00156.xml \
      --text-from $T/C0074421_00154.xml --text-from $T/C0074421_00156.xml --title "the court records of a häradsrätt" --years 1829 \
      --reference $B/pagedemo/samples/gota-hovratt-1701.jpg --reference-caption sample --whole-page --page-megapixels 2 \
      --seed 7 --draw-settings $B/settings/${SETTINGS:-B}.json --out $B/speed/ao_$f > $B/speed_ao_$f.log 2>&1 \
    && log "torchao $f done; peak $(sort -n $B/speed_ao_$f.mem | tail -n 1) MiB; seconds $(python3 -c "import json;print([round(c['seconds'],1) for c in json.load(open('$B/speed/ao_$f/summary.json'))['crops']])")" \
    || log "torchao $f FAILED: $(grep -v '^ *$' $B/speed_ao_$f.log | tail -n 6 | cut -c1-500)"
  kill $watch 2>/dev/null
done
log "finished (torchao)"
