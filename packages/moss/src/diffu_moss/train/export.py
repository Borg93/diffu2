"""diffu-moss-export-ref: a reference LoRA's checkpoint merged into the Base DiT, as a diffusers-format transformer
folder that ``diffu-moss-init --transformer`` draws with.

    diffu-moss-export-ref --transformer <MOSS-Image-Base snapshot>/transformer \\
        --checkpoint runs/moss_ref_pilot/lora/checkpoints/step_001500 --out runs/moss_ref_pilot/export_step_001500

The Base DiT is loaded in bfloat16, the LoRA (``adapter.json``, ``adapter.safetensors``) added in float32 as it was
trained and fused into its weights (each merged weight rounded to bfloat16 once), the adapter layers removed, and the
DiT saved as
``config.json`` and safetensors beside ``export.json``: the export's record, which says the DiT was trained to take a
reference image, so the model directory ``diffu-moss-init`` writes for it takes one. A full fine-tune's checkpoint
(``diffu-moss-train-ref --full``) needs no export: it is such a folder already.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import typer
from peft import set_peft_model_state_dict
from rich.console import Console
from safetensors.torch import load_file

from diffu_moss.train.loop import add_lora
from diffu_moss.train.records import ADAPTER_RECORD, ADAPTER_WEIGHTS, EXPORT_RECORD, AdapterRecord, ExportRecord
from diffu_moss.transformer import MossTransformer

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()


def export_merged(base: Path, checkpoint: Path, out: Path, *, device: str = "cpu") -> ExportRecord:
    """The LoRA of ``checkpoint`` merged into the DiT at ``base``, saved into ``out`` with its record.

    Returns:
        The export's record.

    Raises:
        ValueError: If ``out`` holds a DiT already, or the checkpoint's weights are not its LoRA's.
    """
    if (out / "config.json").exists():
        raise ValueError(f"{out} holds a DiT already; give a new directory")
    adapter = AdapterRecord.read(checkpoint / ADAPTER_RECORD)
    transformer = MossTransformer.from_pretrained(str(base), torch_dtype=torch.bfloat16).to(device)
    add_lora(transformer, adapter.lora)  # float32, as trained: each merged weight is rounded once, to bfloat16
    loaded = set_peft_model_state_dict(transformer, load_file(str(checkpoint / ADAPTER_WEIGHTS), device=device))
    if loaded.unexpected_keys:
        raise ValueError(f"{checkpoint}'s weights hold {len(loaded.unexpected_keys)} that are not its LoRA's")
    transformer.fuse_lora()
    transformer.unload_lora()
    transformer.save_pretrained(str(out))
    record = ExportRecord(
        takes_reference=True, base=str(base.resolve()), checkpoint=str(checkpoint.resolve()), adapter=adapter
    )
    record.write(out / EXPORT_RECORD)
    return record


@app.command()
def main(
    transformer: Annotated[Path, typer.Option(help="the Base DiT's folder the LoRA was trained on")],
    checkpoint: Annotated[Path, typer.Option(help="a checkpoint of diffu-moss-train-ref (checkpoints/step_NNNNNN)")],
    out: Annotated[Path, typer.Option(help="the merged DiT's folder (new)")],
    device: Annotated[str, typer.Option(help="where the weights are merged")] = "cpu",
) -> None:
    """Merge a reference LoRA into the Base DiT, as a folder diffu-moss-init --transformer draws with."""
    try:
        export_merged(transformer, checkpoint, out, device=device)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"the LoRA of {checkpoint} merged into {transformer} -> {out}")


if __name__ == "__main__":
    app()
