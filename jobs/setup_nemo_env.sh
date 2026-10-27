#!/bin/bash
# Install this repo's diffu, diffu-moss, diffu-page and diffu-train, editable, into NeMo AutoModel's environment, where
# training (the dataloader configs in diffu_train.nemo, diffu's init repos) and the GPU jobs (diffu_page.cli.generate /
# evaluate) run; diffu-page draws with MOSS-Image's blocks too, so it needs diffu-moss. --no-deps: NeMo's
# environment owns torch, diffusers (the same FLUX.3 commit as this repo's uv.lock) and transformers. Rerun after a
# fresh NeMo checkout.
#
#   jobs/setup_nemo_env.sh [~/Desktop/repos/Automodel]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTOMODEL="${1:-$HOME/Desktop/repos/Automodel}"
PY="$AUTOMODEL/.venv-managed/bin/python"

uv pip uninstall --python "$PY" diffu page-synth diffu-gen 2>/dev/null || true  # earlier layouts (the old diffu too)
# diffu (the root), diffu-moss, diffu-page, diffu-train
for project in "$REPO" "$REPO/packages/moss" "$REPO/packages/page" "$REPO/packages/train"; do
    uv pip install --python "$PY" --no-deps -e "$project"
done
"$PY" -c "import diffu.init, diffu_moss.init, diffu_page.pipeline, diffu_train.nemo; print('diffu, diffu-moss, diffu-page, diffu-train installed')"
