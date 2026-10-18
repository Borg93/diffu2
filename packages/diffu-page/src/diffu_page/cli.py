"""``diffu-page`` — synthesize full document pages from generated (or placeholder) lines.

    # placeholder ink (no model/GPU) — proves the layout + compositor:
    diffu-page --template multi_column --n 3

    # the whole run from a YAML config (page geometry + renderer + every scan knob):
    diffu-page --config run.yaml

    # real handwriting via a Diffu checkpoint (batched), overriding a couple of fields:
    diffu-page --renderer diffu --ckpt run/final/model.safetensors --style ref_line.png --bed-prob 0.5

The run is a :class:`~diffu_page.config.GenConfig`; ``--config`` loads it from YAML and the flags below
override common fields (CLI > YAML > default). Each page writes ``page_XXXX.png`` + one label file per
format (json / page / alto). The orchestration itself lives in :mod:`diffu_page.generation`; this module
is only the typer command.
"""

from pathlib import Path

import typer

from .config import Renderer, Template, load_gen_config
from .generation import generate_pages, with_overrides


def generate(
    config: Path | None = typer.Option(
        None, "--config", help="YAML GenConfig (full control); flags override"
    ),
    out: Path = typer.Option(Path("out/pages"), help="output directory"),
    n: int = typer.Option(4, help="number of pages"),
    seed: int = typer.Option(0),
    text_file: Path | None = typer.Option(None, "--text-file", help="one paragraph per line"),
    xml: Path | None = typer.Option(None, help="replay layout from an ALTO/PAGE XML instead of a template"),
    spec: Path | None = typer.Option(
        None, "--spec", help="full PageSpec JSON (one object or an array, cycled) — the whole control surface"
    ),
    renderer: Renderer | None = typer.Option(None, help="override: line generator"),
    ckpt: Path | None = typer.Option(None, help="override: Diffu checkpoint"),
    lora: str | None = typer.Option(
        None, "--lora", help="Stage-3 LoRA adapter dir or Hub id, applied on top of ckpt (renderer=diffu)"
    ),
    lora_scale: float | None = typer.Option(
        None, "--lora-scale", help="override: LoRA strength in [0, 1+] (1.0 = full)"
    ),
    style: Path | None = typer.Option(None, help="override: writer style reference line"),
    paper_dir: Path | None = typer.Option(None, "--paper-dir", help="override: blank paper scans"),
    binding_prob: float | None = typer.Option(None, "--binding-prob", min=0.0, max=1.0, help="override"),
    bed_prob: float | None = typer.Option(
        None, "--bed-prob", min=0.0, max=1.0, help="override: scanner-bed fraction"
    ),
    clasp_prob: float | None = typer.Option(
        None, "--clasp-prob", min=0.0, max=1.0, help="override: clasp fraction"
    ),
    fast: bool | None = typer.Option(
        None, "--fast/--no-fast", help="override: batched diffu rendering (faster, slightly softer strokes)"
    ),
    furniture_prob: float | None = typer.Option(
        None, "--furniture-prob", min=0.0, max=1.0, help="override: pages with header + marginalia + page no."
    ),
    crops: bool | None = typer.Option(
        None, "--crops/--no-crops", help="override: also export per-line image+text crops + lines.jsonl"
    ),
    width: int | None = typer.Option(None, help="override: page width in px"),
    height: int | None = typer.Option(None, help="override: page height in px"),
    template: Template | None = typer.Option(None, help="override: layout template"),
) -> None:
    """Synthesize full handwritten document pages + ground-truth labels.

    Pass ``--config run.yaml`` for full control over geometry, renderer and every scan knob; the flags
    above override common fields (CLI > YAML > default).
    """
    cfg = with_overrides(
        load_gen_config(config),
        renderer=renderer,
        ckpt=ckpt,
        lora=lora,
        lora_scale=lora_scale,
        style=style,
        paper_dir=paper_dir,
        binding_prob=binding_prob,
        bed_prob=bed_prob,
        clasp_prob=clasp_prob,
        fast=fast,
        furniture_prob=furniture_prob,
        crop_lines=crops,
        width=width,
        height=height,
        template=template,
    )
    generate_pages(cfg, out=out, n=n, seed=seed, text_file=text_file, xml=xml, spec=spec)


def main() -> None:
    typer.run(generate)


if __name__ == "__main__":
    main()
