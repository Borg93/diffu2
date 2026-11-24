"""Record our fine-tuned reader as the SFT model the RLVR stage starts from.

The RLVR stage (``stage2_rlvr/train.py``) starts from ``<sft_output>/artifact.json`` and refuses one without the
lineage its own SFT stage writes (``training/lineage.py``, ``read_lineage``): the base and the SFT model, each with
the SHA-256 of its weight, configuration and processor files, and the identity of the dataset. Our reader was trained
by ``diffu_htr_train``, not by that stage. Nothing in the check needs the recipe's trainer, only true file hashes, so
this writes the same record with the recipe's own functions (``model_reference``, ``artifact``) for

* base: the NVIDIA snapshot our training started from,
* sft: our checkpoint (its own processor files),

and says in the record who trained it and on what. The stage then runs unchanged, and its checks (the files have not
changed, the dataset is the one recorded) keep working. What the record cannot say: that the SFT model was trained on
this bundle's ``train.jsonl``. It was trained on all 49,491 pages of ``sft_train_set``; the record states that.

    PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python \\
        runs/htr/rlvr/record_sft_lineage.py --out /mnt/scratch/gabriel/htr/rlvr/sft
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO / "runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000"
BASE = Path.home() / ".cache/huggingface/hub/models--nvidia--Nemotron-3.5-Parse/snapshots"
BASE_REVISION = "72ad4d71dd827135ee581502363e14748008900a"
DATA = Path("/mnt/scratch/gabriel/htr/rlvr/data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the folder to pass as sft_output")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--base", type=Path, default=BASE / BASE_REVISION)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()

    from nemotron.recipes.parse.common import artifact, validate_dataset
    from nemotron.recipes.parse.training.lineage import model_reference, read_lineage

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"A record is not rewritten; choose a new, empty directory: {out}")
    identity = validate_dataset(args.data)
    checkpoint = args.checkpoint.resolve()
    lineage = {
        "schema_version": 1,
        "dataset_identity": identity,
        "models": {
            "base": model_reference(args.base, source="nvidia/Nemotron-3.5-Parse", revision=args.base_revision),
            "sft": model_reference(checkpoint),
        },
    }
    result = artifact(
        out,
        "parse/model",
        checkpoint,
        stage="sft",
        base_model="nvidia/Nemotron-3.5-Parse",
        lineage=lineage,
        dataset_manifests=identity["manifests"],
        processor_source=str(checkpoint),
        recorded_by="runs/htr/rlvr/record_sft_lineage.py",
        trained_by="diffu_htr_train (runs/htr/nemotron-parse-selftrain-finetune2), not the recipe's SFT stage",
        sft_train_set=str(REPO / "data/htr/nemotron_parse_pages.train.jsonl"),
        note=(
            "The SFT model was trained on sft_train_set (49,491 pages), not on this dataset's train.jsonl, which is "
            "the part of it RLVR may draw from. The hashes are of the files as they are."
        ),
    )
    read_lineage(result, "sft", identity)  # the stage's own check, so a record that would be refused fails here
    print(f"SFT record: {result}")


if __name__ == "__main__":
    main()
