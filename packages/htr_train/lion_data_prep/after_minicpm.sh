#!/bin/bash
# After minicpm-all (train.sh execs this, reading it only then): its best checkpoint served on $GPUS, scored on the
# out-of-domain and test_0050 lines against the TrOCR baseline, reading a sample of the training lines and every test
# line for Jev's pairs (real errors beside the rule ones, as OpenBMB's finetune_jev_en.md asks); then Jev.
# Being written while minicpm-all trains: until then it stops here, the trained model kept.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
echo "[$(date '+%F %T')] minicpm-all finished; after_minicpm.sh has no steps yet: stopping"
