#!/bin/bash
# A trained model to the Hugging Face Hub as one commit: its weights and configs (no optimizer, trainer or RNG state)
# staged in $STAGE/<repo name>, its card rendered from model_cards/<KIND>.md, refused if a staged file names a local
# path. A new repo is private unless HF_PUBLIC=1 (an existing repo keeps its visibility); DRY_RUN=1 stages and renders
# and prints the upload instead of running it.
#
#   bash publish_model.sh KIND CHECKPOINT REPO [NAME=VALUE | NAME=file:PATH]...
#
# KIND: reader (MiniCPM-OCR), jev (MiniCPM-OCR-Jev) or dspark (the draft: READER_REPO=<repo> names the reader it
# drafts for, written into its config in place of the local path it was trained against). Each NAME fills @NAME@ in
# the card (file:PATH with that file's text); @REPO@ and @NAME@ are the repo and its name.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,12p' "$0"; exit 2; }
kind=$1 src=${2%/} repo=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
HF=${HF:-$here/../envs/minicpm/.venv/bin/hf}
stage=${STAGE:-runs/htr/final/hub}/${repo##*/}
case $kind in
  reader | jev) files=(config.json generation_config.json chat_template.jinja preprocessor_config.json
    processor_config.json tokenizer.json tokenizer_config.json) ;;
  dspark) files=(config.json config.py) ;;
  *) echo "unknown kind: $kind (reader, jev or dspark)" >&2; exit 2 ;;
esac
shopt -s nullglob
weights=("$src"/*.safetensors "$src"/*.safetensors.index.json)
[ ${#weights[@]} -gt 0 ] || { echo "no weights in $src" >&2; exit 1; }
rm -rf "$stage" && mkdir -p "$stage"
for path in "${weights[@]}"; do ln "$path" "$stage/" 2>/dev/null || cp "$path" "$stage/"; done  # linked: no copy
for name in "${files[@]}"; do [ -f "$src/$name" ] && cp "$src/$name" "$stage/"; done
if [ "$kind" = dspark ]; then
  python3 - "$stage/config.json" "${READER_REPO:?READER_REPO=<repo> names the reader the draft is for}" <<'EOF'
import json
import sys

path, reader = sys.argv[1:]
with open(path, encoding="utf-8") as file:
    config = json.load(file)
config["speculators_config"]["verifier"]["name_or_path"] = reader
with open(path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
    file.write("\n")
EOF
fi
python3 - "$here/model_cards/$kind.md" "$stage/README.md" "REPO=$repo" "NAME=${repo##*/}" "$@" <<'EOF'
import re
import sys
from pathlib import Path

template, out, *pairs = sys.argv[1:]
text = Path(template).read_text(encoding="utf-8")
for pair in pairs:
    name, value = pair.split("=", 1)
    if value.startswith("file:"):
        value = Path(value.removeprefix("file:")).read_text(encoding="utf-8").rstrip("\n")
    if not value.strip():
        sys.exit(f"{template}: an empty value for @{name}@")
    text = text.replace(f"@{name}@", value)
if left := sorted(set(re.findall(r"@[A-Z_]+@", text))):
    sys.exit(f"{template}: no value for {', '.join(left)}")
Path(out).write_text(text, encoding="utf-8")
EOF
if leaks=$(grep -lE "/home/|/mnt/|/tmp/" "$stage"/*.json "$stage"/*.py "$stage"/README.md 2>/dev/null); then
  echo "refused: local paths in $leaks" >&2; exit 1
fi
args=(upload "$repo" "$stage" . --commit-message "$kind: $(basename "$src")")
[ "${HF_PUBLIC:-0}" = 1 ] || args+=(--private)
if [ "${DRY_RUN:-0}" = 1 ]; then echo "dry run: $HF ${args[*]}"; ls -la "$stage"; exit 0; fi
"$HF" "${args[@]}"
