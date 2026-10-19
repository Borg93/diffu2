#!/bin/bash
# One-time setup on a Leonardo LOGIN node (they have internet; compute nodes don't): uv, the venv, and every Hub
# model and dataset the jobs read offline, all under $WORK. Re-run after changing dependencies.
#
#   cd $WORK && git clone https://github.com/Borg93/diffu2.git && cd diffu2
#   uv run hf auth login          # once: DINOv3 is gated (accept its licence on the Hub first)
#   bash slurm/setup_login.sh

set -euo pipefail
source "$(dirname "$0")/env.sh"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh  # the official installer
uv sync --extra dev --extra hf --extra track
uv run --no-sync hf auth whoami >/dev/null || { echo "run 'uv run hf auth login' first (DINOv3 is gated)"; exit 1; }

fetch() { uv run --no-sync hf download "$@" >/dev/null && echo "  cached $1"; }
fetch Qwen/Qwen-Image --include "vae/*"                     # the VAE
fetch facebook/dinov3-vitl16-pretrain-lvd1689m              # dino-mode style encoder + the REPA target
fetch Riksarkivet/trocr-large-handwritten-hist-swe-3-char   # the standard gen_CER / eval recognizer
fetch microsoft/trocr-base-handwritten                      # the gauge for English sets like IAM
fetch konnik/DiffusionPen --include "saved_iam_data/*" --local-dir data_raw/diffusionpen  # IAM writer ids
uv run --no-sync python -c "from datasets import load_dataset; load_dataset('Teklia/IAM-line')"
uv run --no-sync python -c "from torchvision.models import resnet18; resnet18(weights='IMAGENET1K_V1')"  # glyph CNN init
echo "setup done: submit slurm/smoke.sbatch next"
