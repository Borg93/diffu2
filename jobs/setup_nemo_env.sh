#!/bin/bash
# Install this repo's diffu-page, diffu-gen and diffu-train, editable, into NeMo AutoModel's environment, where training
# (the dataloader configs in diffu_train.nemo) and the GPU jobs (diffu_gen.cli.generate / evaluate) run. --no-deps:
# NeMo's environment owns torch, diffusers (the same FLUX.3 commit as this repo's uv.lock) and transformers. Rerun
# after a fresh NeMo checkout.
#
#   jobs/setup_nemo_env.sh [~/Desktop/repos/Automodel]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTOMODEL="${1:-$HOME/Desktop/repos/Automodel}"
PY="$AUTOMODEL/.venv-managed/bin/python"

uv pip uninstall --python "$PY" diffu page-synth 2>/dev/null || true  # the packages of earlier layouts
for member in diffu-page diffu-gen diffu-train; do
    uv pip install --python "$PY" --no-deps -e "$REPO/packages/$member"
done
"$PY" -c "import diffu_gen.generator, diffu_train.nemo; print('diffu-page, diffu-gen, diffu-train installed')"
