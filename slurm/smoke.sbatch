#!/bin/bash
#SBATCH --job-name=diffu-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
# The checks first run on the GB10, on one Leonardo A100 (debug QOS): the test suite, the dino and canvas
# recipes, and the compile path. The account comes from $SBATCH_ACCOUNT (see slurm/README.md).
#
#   sbatch slurm/smoke.sbatch

set -euo pipefail
source "${DIFFU_ROOT:-$WORK/diffu2}/slurm/env.sh"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest -q  # the suite is CPU-sized
uv run --no-sync diffu-smoke --steps 3 --batch-size 2
uv run --no-sync diffu-smoke --steps 3 --batch-size 2 --style-canvas --glyph-pad
uv run --no-sync diffu-smoke --steps 4 --batch-size 2 --compile
