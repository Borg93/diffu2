#!/bin/bash
# The demo Space (Gabriel/diffu-archive-pages, ZeroGPU), staged in $STAGE/<space name> and uploaded: space/'s files
# without its tests; the sample pages of $SAMPLES (scans, kept outside the repo); the distilled student's checkpoint
# of $STUDENT, if there is one (diffu-moss-distill's adapter.safetensors, adapter.json and distill.json: the Space's
# fast mode; without it the Space draws guided only); and requirements.txt with the diffu packages pinned to this
# checkout's commit, which the Space installs from GitHub, so the commit must be on origin/main. Refused if it is
# not, if a staged file names a local path, or if there is no sample page.
# DRY_RUN=1 stages and lists, without uploading.
#
#   bash packages/studio/publish_space.sh
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
here=packages/studio
SPACE=${SPACE:-Gabriel/diffu-archive-pages}
SAMPLES=${SAMPLES:-$HOME/diffu_queue/space/samples}
STUDENT=${STUDENT:-$HOME/diffu_queue/space/student}
HF=${HF:-hf}
stage=${STAGE:-runs/space}/${SPACE##*/}
commit=$(git rev-parse HEAD)
git fetch -q origin main
git merge-base --is-ancestor "$commit" origin/main || { echo "refused: $commit is not on origin/main" >&2; exit 1; }
ls "$SAMPLES"/*.jpg > /dev/null 2>&1 || { echo "refused: no sample pages in $SAMPLES" >&2; exit 1; }
rm -rf "$stage" && mkdir -p "$stage/samples"
rsync -a --exclude tests --exclude __pycache__ --exclude ruff.toml "$here/space/" "$stage/"
cp "$SAMPLES"/*.jpg "$stage/samples/"
if [ -f "$STUDENT/adapter.safetensors" ]; then
  mkdir -p "$stage/student"
  cp "$STUDENT/adapter.safetensors" "$STUDENT/adapter.json" "$STUDENT/distill.json" "$stage/student/"
  # the record names the teacher's folder where the student was trained: the Space needs only how it is drawn
  sed -i -E 's#"teacher": *"[^"]*"#"teacher": "Gabriel/diffu-moss-above"#' "$stage/student/distill.json"
fi
sed -i "s/DIFFU_COMMIT/$commit/" "$stage/requirements.txt"
if leaks=$(grep -rlE "/home/|/mnt/|/leonardo" "$stage" --include='*.py' --include='*.md' --include='*.js' --include='*.txt' --include='*.json'); then
  echo "refused: local paths in $leaks" >&2; exit 1
fi
[ -n "${DRY_RUN:-}" ] && { find "$stage" -type f | sort; grep diffu "$stage/requirements.txt"; exit 0; }
$HF upload "$SPACE" "$stage" . --repo-type space --exclude "**/__pycache__/**" \
  --commit-message "${MESSAGE:-From diffu2 at $(git rev-parse --short HEAD)}"
