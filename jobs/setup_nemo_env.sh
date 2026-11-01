#!/bin/bash
# Install this repo's diffu-core, diffu-moss, diffu-page and diffu-flux3, editable, into NeMo AutoModel's environment,
# where FLUX.3's LoRA training (the dataloader configs in diffu_flux3.nemo, the style-pair cache) and its GPU jobs
# (diffu_page.cli.generate / evaluate) run; diffu-page draws with MOSS-Image's blocks too, so it needs diffu-moss, and
# both read model directories with diffu-core. --no-deps: NeMo's environment owns torch, diffusers (the same FLUX.3
# commit as this repo's uv.lock) and transformers. Rerun after a fresh NeMo checkout.
#
#   jobs/setup_nemo_env.sh [~/Desktop/repos/Automodel]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTOMODEL="${1:-$HOME/Desktop/repos/Automodel}"
PY="$AUTOMODEL/.venv-managed/bin/python"

# diffu-core, diffu-moss, diffu-page, diffu-flux3
for project in "$REPO/packages/core" "$REPO/packages/moss" "$REPO/packages/page" "$REPO/packages/flux3"; do
    uv pip install --python "$PY" --no-deps -e "$project"
done
"$PY" -c "import diffu_core.repo, diffu_moss.init, diffu_page.pipeline, diffu_flux3.nemo; print('diffu-core, diffu-moss, diffu-page, diffu-flux3 installed')"
