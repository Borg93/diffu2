#!/bin/bash
# GPU 1: minicpm_all's step (4 a GPU, checkpointing off, one GPU) as it is, with torch.compile, grouped by length, both.
cd /home/gabriel/Desktop/diffu2
for n in base compile grouped compile_grouped; do
  timeout 1500 bash runs/htr/opt/$n.sh > runs/htr/opt/$n.log 2>&1
  echo "[$(date '+%T')] $n exit $?"
done
echo OPT DONE
