"""``diffu-runner`` CLI — forge a synthetic HTR line dataset at scale (generate + CER gate)."""

from __future__ import annotations

from pathlib import Path

import typer

from .gate import GateConfig
from .pipeline import RunConfig, run, run_target

app = typer.Typer(add_completion=False, help="Forge a synthetic HTR line dataset with Diffu + a TrOCR gate.")


@app.command()
def main(
    ckpt: str = typer.Option(..., help="Diffu checkpoint (.safetensors)"),
    train_jsonl: str = typer.Option(..., help="TRAIN split (text + style) — NOT val.jsonl (held-out test)"),
    out: str = typer.Option("synth_shards", help="output dir for the Lance dataset of clean lines"),
    n: int = typer.Option(
        1_000_000, help="candidate count before gating (--stream), or exact clean target (--exact-target)"
    ),
    gen_gpus: int = typer.Option(2, help="generator (DiffuActor) GPU replicas"),
    gen_gpu_frac: float = typer.Option(
        1.0, help="GPU fraction per DiffuActor replica (lower to co-locate a TrOCR reader on the same GPU)"
    ),
    trocr_replicas: int = typer.Option(4, help="TrOCR gate replicas (share GPUs)"),
    trocr_gpu_frac: float = typer.Option(0.25, help="GPU fraction per TrOCRActor replica"),
    cer_max: float = typer.Option(0.30, help="drop lines read back worse than this CER"),
    compile_model: bool = typer.Option(
        True,
        "--compile/--no-compile",
        help="torch.compile the DiffuActor's MMDiT blocks (regional compile; same win used at train/inference)",
    ),
    gen_dtype: str = typer.Option(
        "bfloat16", help="DiffuActor inference dtype (bf16 is the validated default)"
    ),
    num_steps: int = typer.Option(
        24,
        help="denoise steps per line; 12 is benchmark-validated equal-quality on the 128px model "
        "(paired ΔCER ~0 vs 24) for ~1.5x generation speed — 8 visibly degrades letterforms",
    ),
    bucket_px: int = typer.Option(
        0,
        help="width-bucket coarseness; 0 (default) = OFF, fixed-width canvas — the only mode the current "
        "checkpoint's glyph conditioning supports (bucketed/homogeneous-width batches make it stutter); "
        "set only for a future conditioning-fixed checkpoint",
    ),
    lora: str | None = typer.Option(
        None, help="LoRA adapter dir or Hub id to load on top of the base checkpoint (Stage-3 adaptation)"
    ),
    lora_scale: float = typer.Option(1.0, help="LoRA adapter strength in [0, 1+] (1.0 = full)"),
    exact_target: bool = typer.Option(
        False,
        "--exact-target/--stream",
        help="--exact-target regenerates rejects with a fresh writer style until EXACTLY n clean lines exist "
        "(pipeline.run_target); --stream (default) is one pass over n candidates and may undershoot n if the "
        "reject rate is high (pipeline.run).",
    ),
    coherent: bool = typer.Option(
        False,
        "--coherent/--random",
        help="--coherent takes WHOLE documents ((doc_id, line_idx)-keyed contiguous lines, one writer per "
        "doc) so a line bank can replay coherent prose pages; --random (default) pairs independent text x "
        "style. Only --stream honors it — --exact-target's text-keyed retry always uses the random list.",
    ),
) -> None:
    """Run the Ray-Data generate→gate→write pipeline. Requires a running Ray cluster (or starts a local one)."""
    # Absolutize here, at the CLI boundary: the actors run in Ray workers whose working dir is the
    # runtime-env package, not the caller's — a relative ckpt path FileNotFounds inside every worker.
    cfg = RunConfig(
        ckpt=str(Path(ckpt).resolve()),
        train_jsonl=str(Path(train_jsonl).resolve()),
        out=str(Path(out).resolve()),
        n=n,
        gen_gpus=gen_gpus,
        gen_gpu_frac=gen_gpu_frac,
        trocr_replicas=trocr_replicas,
        trocr_gpu_frac=trocr_gpu_frac,
        compile_model=compile_model,
        gen_dtype=gen_dtype,
        num_steps=num_steps,
        bucket_px=bucket_px or None,  # CLI 0 -> None (bucketing off)
        # a LOCAL adapter dir must be absolutized for the Ray workers (like ckpt); a Hub id passes through
        lora=(str(Path(lora).resolve()) if lora and Path(lora).exists() else lora),
        lora_scale=lora_scale,
        coherent=coherent,
        gate=GateConfig(cer_max=cer_max),
    )
    if exact_target:
        if coherent:  # documented in --help, but say it at runtime too — the bank silently loses its keys
            typer.echo(
                "warning: --coherent is ignored by --exact-target; the bank will be random-keyed", err=True
            )
        path, stats = run_target(cfg, target=n)
        typer.echo(f"wrote {stats['kept']} clean lines -> {path}  ({stats})")
    else:
        path = run(cfg)
        typer.echo(f"wrote clean lines -> {path}")


if __name__ == "__main__":
    app()
