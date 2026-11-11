#!/bin/bash
# The Space that shows the three models (Gabriel/diffu-htr-speed, ZeroGPU), staged in $STAGE/<space name> and uploaded:
# space/'s files; the Jev prompt and renderer from diffu_htr_train, copied as jev_prompt.py and jev_render.py (a Space
# cannot install the package); the renderer's font; and the example regions build_examples.py cuts from the dev sets
# and the GT's PAGE XML. Refused if a staged file names a local path. DRY_RUN=1 stages and lists, without uploading.
# The Space reads the private models with its HF_TOKEN secret (a read-only token, set in the Space's settings).
#
#   GT=/mnt/work/GT-Lejonet_HTR_202602 bash packages/htr_train/lion_data_prep/publish_space.sh
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
here=packages/htr_train/lion_data_prep
SPACE=${SPACE:-Gabriel/diffu-htr-speed}
D=${SETS:-data/htr}
GT=${GT:-$HOME/Desktop/data/GT-Lejonet_HTR_202602}
FONT=${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf}  # the font Jev's pairs were rendered in
HF=${HF:-packages/htr_train/envs/minicpm/.venv/bin/hf}
PY=${PY:-packages/htr_train/envs/minicpm/.venv/bin/python}
stage=${STAGE:-runs/htr/final/hub}/${SPACE##*/}
rm -rf "$stage" && mkdir -p "$stage"
for name in app.py dspark_hf.py race_view.py race_view.js race_view.css check_decode.py HOW_IT_WORKS.md README.md \
    requirements.txt; do
  cp "$here/space/$name" "$stage/"
done
cp packages/htr_train/src/diffu_htr_train/jev/prompt.py "$stage/jev_prompt.py"
cp packages/htr_train/src/diffu_htr_train/jev/render.py "$stage/jev_render.py"
cp "$FONT" "$stage/DejaVuSans.ttf"
$PY "$here/space/build_examples.py" --sets "$D" --gt "$GT" --out "$stage/examples"
if leaks=$(grep -rlE "/home/|/mnt/" "$stage" --include='*.py' --include='*.md' --include='*.json' --include='*.js'); then
  echo "refused: local paths in $leaks" >&2; exit 1
fi
[ -n "${DRY_RUN:-}" ] && { ls -la "$stage"; exit 0; }
$HF upload "$SPACE" "$stage" . --repo-type space --exclude "**/__pycache__/**" \
  --commit-message "${MESSAGE:-From diffu2 at $(git rev-parse --short HEAD)}"
