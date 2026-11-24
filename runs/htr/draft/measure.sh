#!/bin/bash
# A draft measured on a cache as the report gives it: the trainer's own validation, then vLLM's acceptance worked out
# by serve_accept.py for 8, 16 and 32 drafted tokens. bash runs/htr/draft/measure.sh DRAFT.safetensors CACHE [CACHE...]
cd /home/gabriel/Desktop/diffu2 || exit 1
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
PY=/mnt/scratch/gabriel/envs/automodel/bin/python; T=$(mktemp -d /mnt/scratch/gabriel/htr/draft/runs/measure.XXXX); W=$1; shift
for C in "$@"; do
  A=(-c runs/htr/draft/dflash2_parse.yaml --recipe_args.init_draft_path "$W" --recipe_args.output_dir "$T" --checkpoint.checkpoint_dir "$T/checkpoints")
  echo "== $W on $C"
  $PY runs/htr/draft/train_draft.py "${A[@]}" --eval-only --recipe_args.train_cache_path "$C" --recipe_args.val_cache_path "$C" 2>&1 | grep -E "^VALIDATION|Error|memory"
  for K in 8 16 32; do $PY runs/htr/draft/serve_accept.py "${A[@]}" --k $K --chunk ${CHUNK:-32} --cache "$C" 2>&1 | grep -E "^SERVING|Error|memory"; done
done
rm -rf "$T"
