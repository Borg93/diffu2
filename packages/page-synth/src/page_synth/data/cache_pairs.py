"""Cache the v3 reference pairs (``page_synth.data.style_pairs``'s manifest) with NeMo AutoModel's FLUX.3 image-edit encoder.

NeMo's ``image-edit`` preprocessing CLI reads a Hugging Face dataset and materializes it into a manifest; ours is
already one, so this hands it straight to the encoder that CLI calls (``Flux3ImageEditCacheEncoder``). Targets keep
their exact bucket size (none is over ``TARGET_PIXELS``); the references were shrunk by ``page_synth.data.style_pairs``.
Runs in the NeMo AutoModel environment, on the GPU:

    cd ~/Desktop/repos/Automodel && python -m page_synth.data.cache_pairs \
        --manifest ~/Desktop/repos/diffu2/data_flux3/style_pairs_v3/train.jsonl --out ~/Desktop/repos/diffu2/data_flux3/cache_style_v3
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from nemo_automodel.components.models.flux3.preprocessing import (
    MAX_SEQUENCE_LENGTH,
    Flux3ImageEditCacheEncoder,
)

from page_synth.pipelines import FLUX3_KLEIN

TARGET_PIXELS = 1024 * 1024  # the largest v2 bucket (1024 x 1024)


def first_rows(manifest: Path, limit: int) -> Path:
    """A manifest of the first ``limit`` rows, beside the full one so its media paths still resolve."""
    subset = manifest.with_name(f"{manifest.stem}_first{limit}.jsonl")
    with manifest.open(encoding="utf-8") as rows:
        subset.write_text("".join(islice(rows, limit)), encoding="utf-8")
    return subset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=str(FLUX3_KLEIN))
    ap.add_argument("--limit", type=int, default=None, help="cache only the first N pairs (a smoke test)")
    args = ap.parse_args()

    manifest = first_rows(args.manifest, args.limit) if args.limit else args.manifest
    encoder = Flux3ImageEditCacheEncoder(model_name=args.model, max_sequence_length=MAX_SEQUENCE_LENGTH)
    metadata = encoder.encode_manifest(
        manifest_path=manifest,
        output_dir=args.out,
        max_pixels=TARGET_PIXELS,
        resolution_preset=None,
        num_gpus=1,
        verify=False,
    )
    print(f"cached {manifest} -> {metadata}")


if __name__ == "__main__":
    main()
