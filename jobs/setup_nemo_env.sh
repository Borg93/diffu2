#!/bin/bash
# Install this repo's page-synth package, editable, into NeMo AutoModel's environment, where training (its dataloader
# configs, page_synth.data.nemo) and the GPU jobs (page_synth.cli.generate / evaluate) run. --no-deps: NeMo's environment owns
# torch, diffusers (the same FLUX.3 commit as this repo's uv.lock) and transformers. Rerun after a fresh NeMo checkout.
#
#   jobs/setup_nemo_env.sh [~/Desktop/repos/Automodel]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTOMODEL="${1:-$HOME/Desktop/repos/Automodel}"
PY="$AUTOMODEL/.venv-managed/bin/python"

uv pip uninstall --python "$PY" diffu diffu-page 2>/dev/null || true  # the pre-2026-10-22 packages
uv pip install --python "$PY" --no-deps -e "$REPO/packages/page-synth"
"$PY" -c "import page_synth.generator, page_synth.data.nemo; print('page-synth installed from', page_synth.__path__[0])"
