#!/bin/bash
# The first real draft for the reader of E60 (nemotron-parse-selftrain-finetune2 checkpoint-3000): 4,000 pool_v3 pages
# of 4,000 volumes (pick_pages.py, make_cache.py), validated after each epoch on 200 pages of 95 other volumes, warm
# from NVIDIA's draft. One arm: NAME=a LR=1e-4 EPOCHS=3 bash runs/htr/draft/train_v1.sh  (ANCHORS=64 blocks a page)
cd /home/gabriel/Desktop/diffu2 || exit 1
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
D=/mnt/scratch/gabriel/htr/draft; O=$D/runs/v1_${NAME:?}
/mnt/scratch/gabriel/envs/automodel/bin/python runs/htr/draft/train_draft.py -c runs/htr/draft/dflash2_parse.yaml --export $O/dflash2 \
  --recipe_args.train_cache_path $D/cache/v1_train4000 --recipe_args.val_cache_path $D/cache/v1_val200 \
  --recipe_args.output_dir $O --checkpoint.checkpoint_dir $O/checkpoints --recipe_args.save_checkpoint_every_epoch true \
  --recipe_args.num_epochs ${EPOCHS:-3} --optimizer.lr ${LR:-1e-4} --recipe_args.num_anchors ${ANCHORS:-64} \
  --recipe_args.log_every_steps 200 > $O.log 2>&1
grep -E "Finished epoch|exported|Error|memory" $O.log | sed 's/.*| INFO | [a-z_.0-9]* | //' | cut -c1-330
