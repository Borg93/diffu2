#!/bin/bash
# Install this repo's diffu, diffu-page and diffu-train, editable, into NeMo AutoModel's environment, where training
# (the dataloader configs in diffu_train.nemo, diffu's init repos) and the GPU jobs (diffu_page.cli.generate / evaluate)
# run. --no-deps: NeMo's
# environment owns torch, diffusers (the same FLUX.3 commit as this repo's uv.lock) and transformers. Rerun after a
# fresh NeMo checkout.
#
#   jobs/setup_nemo_env.sh [~/Desktop/repos/Automodel]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTOMODEL="${1:-$HOME/Desktop/repos/Automodel}"
PY="$AUTOMODEL/.venv-managed/bin/python"

uv pip uninstall --python "$PY" diffu page-synth diffu-gen 2>/dev/null || true  # earlier layouts (the old diffu too)
for member in diffu diffu-page diffu-train; do
    uv pip install --python "$PY" --no-deps -e "$REPO/packages/$member"
done
"$PY" -c "import diffu.init, diffu_page.pipeline, diffu_train.nemo; print('diffu, diffu-page, diffu-train installed')"
