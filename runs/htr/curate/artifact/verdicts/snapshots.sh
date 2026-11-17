#!/bin/bash
# The two archify diagrams of the verdicts page: delivered as interactive HTML, then photographed in both themes for
# build.py to embed. Run in this folder; needs the archify skill and a headless Chromium (no sandbox on this machine).
#
#   bash snapshots.sh && python3 build.py "17 November 2026, 20:15" "step 3,620 of 4,374, 83 %" "1.75 to 1.79"
set -euo pipefail
ARCHIFY=${ARCHIFY:-$HOME/.claude/skills/archify/bin/archify.mjs}
CHROME=${CHROME:-$(ls "$HOME"/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell | sort -V | tail -n 1)}
node "$ARCHIFY" deliver dataflow pipeline.dataflow.json pipeline.html --quality showcase --json > pipeline.receipt.json
node "$ARCHIFY" deliver workflow cut.workflow.json cut.html --quality showcase --json > cut.receipt.json
for name in pipeline cut; do
  for theme in light dark; do
    "$CHROME" --no-sandbox --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=1200,720 \
      --virtual-time-budget=8000 --screenshot="$PWD/snap_${name}_${theme}.png" "file://$PWD/${name}.html?embed=1&theme=${theme}" 2> /dev/null
  done
done
# Cut each pair of snapshots below the legend, at the same height in both themes.
uv run python - <<'PY'
import numpy as np
from PIL import Image

for name in ("pipeline", "cut"):
    last = []
    for theme in ("light", "dark"):
        pixels = np.asarray(Image.open(f"snap_{name}_{theme}.png").convert("RGB")).astype(int)
        marked = (np.abs(pixels - pixels[3, 3]).sum(axis=2) > 90).sum(axis=1) > 3  # rows that hold more than the faint grid
        last.append(int(np.where(marked)[0].max()))
    for theme in ("light", "dark"):
        image = Image.open(f"snap_{name}_{theme}.png").convert("RGB")
        image.crop((0, 0, image.width, min(max(last) + 56, image.height))).save(f"snap_{name}_{theme}.png", optimize=True)
PY
