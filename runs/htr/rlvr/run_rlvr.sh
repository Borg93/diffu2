#!/usr/bin/env bash
# RLVR (GRPO) on our handwritten pages with NVIDIA's Parse recipe, unchanged, on GPU 2 only.
#
#   runs/htr/rlvr/run_rlvr.sh NAME                       the pilot: 128 updates, 256 pages of 1,024 candidates
#   PROFILE=tiny runs/htr/rlvr/run_rlvr.sh NAME          the smoke: 2 updates, 16 pages
#   CONFIG=/path/other.yaml runs/htr/rlvr/run_rlvr.sh NAME   another training YAML (defaults: grpo_htr.yaml)
#   DRY=1 runs/htr/rlvr/run_rlvr.sh NAME                 print the resolved stage and training configuration, no GPU
#
# Writes /mnt/scratch/gabriel/htr/rlvr/runs/NAME (must not exist: the stage never appends to a run) and NAME.log
# beside it. The recipe's CLI would run `uv run --locked --project stage2_rlvr python train.py --config ...`, which
# builds the environment inside the Nemotron clone; this runs the same train.py with the same locked environment,
# built from a copy of the stage's pyproject.toml and uv.lock in /mnt/scratch/gabriel/envs/parse_rlvr.
set -euo pipefail

NAME=${1:?usage: run_rlvr.sh NAME}
PROFILE=${PROFILE:-pilot}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NEMOTRON=/mnt/scratch/gabriel/src/Nemotron
PYTHON=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
ROOT=/mnt/scratch/gabriel/htr/rlvr
CONFIG=$(realpath "${CONFIG:-$HERE/grpo_htr.yaml}")
STAGE=$HERE/stage_$PROFILE.yaml
OUT=$ROOT/runs/$NAME

export CUDA_VISIBLE_DEVICES=2            # the only GPU that is ours
export PYTHONPATH=$NEMOTRON/src:$HERE    # the recipe, and our task and reward environment (Ray workers get it too)
export HTR_RLVR_CONFIG=$CONFIG           # the task's hooks read the same training YAML (htr_settings.py)
export HTR_RLVR_DATA=$ROOT/data
export VLLM_USE_FLASHINFER_SAMPLER=0     # as the recipe's CLI sets for this stage
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0   # as the recipe's CLI sets for this stage
export OMP_NUM_THREADS=1                 # dmlpai01: more threads stall GPU-feeding jobs
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT

if [[ ${DRY:-0} == 1 ]]; then
  exec "$PYTHON" "$HERE/resolve_config.py" --stage "$STAGE" --training "$CONFIG" --out "$OUT"
fi

[[ -f $ROOT/data/artifact.json ]] || { echo "No dataset bundle; run build_dataset.py first" >&2; exit 1; }
[[ -f $ROOT/sft/artifact.json ]] || "$PYTHON" "$HERE/record_sft_lineage.py" --out "$ROOT/sft"
[[ ! -e $OUT ]] || { echo "$OUT exists; a run is never appended to, choose another NAME" >&2; exit 1; }
mkdir -p "$ROOT/runs"

cd "$ROOT"   # Ray and vLLM write their scratch files where they start
"$PYTHON" "$NEMOTRON/src/nemotron/recipes/parse/stage2_rlvr/train.py" \
  --config "$STAGE" "output_dir=$OUT" "training_config=$CONFIG" 2>&1 | tee "$ROOT/runs/$NAME.log"
