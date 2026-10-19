"""Deploy the Diffu Studio Hugging Face Space FROM the local repo — the single, drift-proof push path.

The Space (``Gabriel/diffu_test``) is a FLATTENED copy of three workspace packages (a Space has no uv
workspace, so their `src/` trees are copied to the repo root next to ``app.py``). This script is the ONLY
way that copy is made: it assembles a staging tree from the CURRENT local sources every time, so the Space
is always exactly local `main` — there is no separately-editable copy to drift.

    uv run python packages/diffu-studio/space/push.py            # stage + upload
    uv run python packages/diffu-studio/space/push.py --dry-run  # stage only, print the tree, don't upload

What ships:
  app.py, requirements.txt, packages.txt, README.md, assets/  — tracked HERE (edit these in the repo)
  diffu/, diffu_page/, diffu_studio/                          — copied fresh from the package `src/` trees

Everything else (the pinned PiD git dep, gemma, checkpoints) is pulled by the Space at build/run time per
requirements.txt + app.py. Editable installs / __pycache__ are never shipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

SPACE_ID = "Gabriel/diffu_test"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # packages/diffu-studio/space -> repo root

# (package import name, its src/ directory) — copied to the Space root as flat top-level packages.
PACKAGES = {
    "diffu": REPO / "src" / "diffu",
    "diffu_page": REPO / "packages" / "diffu-page" / "src" / "diffu_page",
    "diffu_studio": REPO / "packages" / "diffu-studio" / "src" / "diffu_studio",
}
# Files tracked in this dir that ship verbatim (assets/ is copied as a tree separately).
SPACE_FILES = ["app.py", "requirements.txt", "packages.txt", "README.md"]


def _stage(dst: Path) -> None:
    """Assemble the exact tree the Space will run, into ``dst``."""
    for name in SPACE_FILES:
        shutil.copy2(HERE / name, dst / name)
    shutil.copytree(HERE / "assets", dst / "assets")
    for import_name, src in PACKAGES.items():
        if not src.is_dir():
            sys.exit(f"missing package source: {src}")
        # ignore __pycache__ so stale bytecode never ships and can't shadow the copied source
        shutil.copytree(src, dst / import_name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="stage + print the tree, do not upload")
    ap.add_argument("--space-id", default=SPACE_ID, help=f"target Space (default {SPACE_ID})")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="diffu-space-") as tmp:
        dst = Path(tmp)
        _stage(dst)
        files = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file())
        print(f"staged {len(files)} files from local source:")
        for f in files:
            if "/" not in f or f.count("/") <= 1:  # show top-level + one level (skip deep package internals)
                print(f"  {f}")
        if args.dry_run:
            print("\n--dry-run: not uploading.")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        commit = api.upload_folder(
            folder_path=str(dst),
            repo_id=args.space_id,
            repo_type="space",
            commit_message="deploy from local repo (packages/diffu-studio/space/push.py)",
            delete_patterns=[  # scrub anything not in this upload: stale flattened pkgs, the old PiD clone, pyc
                "diffu/**",
                "diffu_page/**",
                "diffu_studio/**",
                "assets/**",
                "PiD/**",
                "**/__pycache__/**",
                "*.py",
                "*.txt",
            ],  # fmt: skip
        )
        print(f"\npushed -> {commit.commit_url}")


if __name__ == "__main__":
    main()
