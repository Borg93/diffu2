#!/bin/bash
#SBATCH --job-name=diffu-train
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_bprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
# Diffu training on one whole Booster node (4x A100 64 GB, 32 cores; --mem=0 = all of its memory). It resumes
# from the newest complete step_N in OUT_DIR, so slurm/chain.sh can run a training longer than the walltime as
# a series of jobs. Everything after the script name goes to diffu-train.
#
#   OUT_DIR=$WORK/runs/canvas sbatch slurm/train.sbatch --data-dir $FAST/data_iam --style-canvas --glyph-pad ...

set -euo pipefail
source "${DIFFU_ROOT:-$WORK/diffu2}/slurm/env.sh"
: "${OUT_DIR:?set OUT_DIR (the run directory; chained jobs resume from it)}"

gpus="${SLURM_GPUS_ON_NODE:-4}"
uv run --no-sync accelerate launch --multi_gpu --num_processes "$gpus" --num_machines 1 --mixed_precision bf16 \
    -m diffu.train --out-dir "$OUT_DIR" --resume auto --num-workers $((${SLURM_CPUS_PER_TASK:-32} / gpus)) "$@"
