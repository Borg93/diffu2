#!/bin/bash
# Submit N chained slurm/train.sbatch jobs: each starts once the previous one ends (whatever its exit — a
# walltime TIMEOUT included) and resumes from the newest checkpoint, so together they run past the 24 h limit.
# Arguments after N go to diffu-train, as for train.sbatch.
#
#   OUT_DIR=$WORK/runs/canvas slurm/chain.sh 4 --data-dir $FAST/data_iam --style-canvas --glyph-pad ...

set -euo pipefail
: "${OUT_DIR:?set OUT_DIR (the run directory the chained jobs share)}"
n="$1"
shift
prev=""
for _ in $(seq "$n"); do
    prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} "$(dirname "$0")/train.sbatch" "$@")
    echo "submitted job $prev"
done
