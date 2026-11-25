#!/bin/bash
# At checkpoint-3000: stop minicpm-all, test base MiniCPM-OCR on 300 whole test regions against their lines, wait for
# the masked regions, then resume from checkpoint-3000 on the lines and the masked regions if the regions read about
# as well as their lines (corpus CER at most 1.5 x the lines', at most 5 % cut off at 4,096 tokens), else on the lines
# only; the data order starts afresh (--ignore_data_skip). Then after_minicpm.sh.
cd /home/gabriel/Desktop/diffu2
O=runs/htr/regiontest
C=$PWD/runs/htr/minicpm-all/output/checkpoint-3000
log() { echo "[$(date '+%F %T')] $*"; }
sed -n '/^until \[ -f \$C/,/^\$E \$O\/compare.py/p' $O/run.sh > $O/test_part.sh
source <(sed -n '1,/^log()/p' $O/run.sh | grep -E '^(O|C)=') ; C=runs/htr/minicpm-all/output/checkpoint-3000
E=packages/htr_train/envs/minicpm/.venv/bin/python
V=packages/htr_train/envs/vllm/.venv/bin/vllm
M=$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM-OCR-1.0/snapshots/2697d4e93dca82784abf858ae57206a49629cb6b
source $O/test_part.sh
cat $O/result.txt
until grep -q "MASKED REGIONS DONE" runs/htr/cut_regions_masked.log; do
  grep -qiE "Traceback|Error" runs/htr/cut_regions_masked.log && { log "masked region cut failed"; break; }; sleep 30; done
DECISION=$($E -I -c "
import re; t=open('$O/result.txt').read()
r=float(re.search(r'whole regions\s+corpus CER\s+([\d.]+)%', t).group(1)); l=float(re.search(r'their lines, joined\s+corpus CER\s+([\d.]+)%', t).group(1))
cut=int(re.search(r'cut off at 4,096 tokens: (\d+)', t).group(1))
print('regions' if r <= 1.5 * l and cut <= 15 else 'lines')" 2>/dev/null || echo lines)
grep -q "MASKED REGIONS DONE" runs/htr/cut_regions_masked.log || DECISION=lines
log "decision: train on $DECISION"
echo "--- paused at checkpoint-3000; resumed on: $DECISION ---" >> runs/htr/pipeline.log
if [ "$DECISION" = regions ]; then
  NEW=/mnt/work/gabriel/htr/minicpm_regions_masked.train.jsonl
  sed -e "\$ s#/mnt/work/gabriel/htr/minicpm_regions.train.jsonl#$NEW#" runs/htr/minicpm-all/steps.sh > runs/htr/minicpm-all/resume_3000.sh
else
  sed -e "\$ s# /mnt/work/gabriel/htr/minicpm_regions.train.jsonl##" runs/htr/minicpm-all/steps.sh > runs/htr/minicpm-all/resume_3000.sh
fi
sed -i "\$ s#\$# --resume_from_checkpoint $PWD/$C --ignore_data_skip true#" runs/htr/minicpm-all/resume_3000.sh
tail -c 330 runs/htr/minicpm-all/resume_3000.sh; echo
until ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -qiE "75:00|A3:00"; do sleep 5; done
log "resuming minicpm-all from checkpoint-3000"
if bash runs/htr/minicpm-all/resume_3000.sh >> runs/htr/pipeline.log 2>&1; then
  touch runs/htr/pipeline_markers/minicpm_train.done; log "minicpm-all finished"
  exec env GPUS=1,2 bash packages/htr_train/lion_data_prep/after_minicpm.sh >> runs/htr/pipeline.log 2>&1
else
  log "minicpm-all stopped with an error after resuming: see runs/htr/pipeline.log"
fi
