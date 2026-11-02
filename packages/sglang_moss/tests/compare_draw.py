"""One page drawn by our pipeline and by SGLang Diffusion with diffu_sglang_moss, from the same model, prompt, sample
page and noise, on one device; prints how far apart they are and saves both, side by side, to look at.

Run with the workspace's environment; SGLang runs in its own (``--sglang-python``), one after the other, so the two
models are never in memory together:

    python packages/sglang_moss/tests/compare_draw.py --moss <diffu-moss-init dir> --hf <diffu-moss-export-hf dir> \\
        --sample <sample page.png> --prompt-file <caption.txt> --width 1024 --height 1024 --steps 50 \\
        --sglang-python <SGLang env>/bin/python --out runs/sglang_compare
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from PIL import Image

from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.denoise import GUIDANCE_SCALE
from diffu_moss.modular_pipelines.moss.encoders import NEGATIVE_PROMPT

app = typer.Typer(add_completion=False)
DRAW = Path(__file__).parent / "sglang_draw.py"
PLUGIN = Path(__file__).parents[1] / "src"


def _image(pixels: np.ndarray) -> Image.Image:
    return Image.fromarray((pixels.transpose(1, 2, 0) * 255).round().astype(np.uint8))


@app.command()
def main(
    moss: Annotated[Path, typer.Option(help="the MOSS model directory our pipeline draws from")],
    hf: Annotated[Path, typer.Option(help="the same model exported by diffu-moss-export-hf, for SGLang")],
    prompt_file: Annotated[Path, typer.Option(help="the prompt, as a text file")],
    sglang_python: Annotated[Path, typer.Option(help="the SGLang environment's python")],
    out: Annotated[Path, typer.Option(help="where both images and the comparison are written")],
    sample: Annotated[Path | None, typer.Option(help="the sample page (on the 16-px grid); none: text only")] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 50,
    seed: int = 0,
    device: str = "cuda",
) -> None:
    """Draw one page with both and compare them."""
    out.mkdir(parents=True, exist_ok=True)
    prompt = prompt_file.read_text(encoding="utf-8")
    pipe = MossModularPipeline.from_pretrained(str(moss))
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    pipe.to(device)
    skip = list(pipe.config.uncond_skip_layers)
    ours = (
        pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="pt",
            output="images",
            **({} if sample is None else {"image": Image.open(sample)}),
        )[0]
        .float()
        .cpu()
        .numpy()
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    params: dict[str, object] = {"prompt": prompt, "negative_prompt": NEGATIVE_PROMPT, "height": height}
    params |= {"width": width, "num_inference_steps": steps, "seed": seed, "generator_device": "cpu"}
    params |= {"guidance_scale": GUIDANCE_SCALE, "uncond_skip_layers": skip, "slg_start": 0.0}
    if sample is not None:
        params["image_path"] = str(sample.resolve())
    server = {"num_gpus": 1, "attention_backend": "torch_sdpa", "disable_autocast": True}
    request = out / "request.json"
    request.write_text(json.dumps({"model": str(hf), "params": params, "server": server}), encoding="utf-8")
    env = {**os.environ, "SGLANG_EXTERNAL_MODEL_PACKAGE": "diffu_sglang_moss", "PYTHONPATH": str(PLUGIN)}
    subprocess.run([str(sglang_python), str(DRAW), str(request), str(out / "sglang.npy")], env=env, check=True)
    theirs = np.load(out / "sglang.npy")

    error = np.abs(ours.astype(np.float64) - theirs)
    mse = float((error**2).mean())
    report = {
        "identical": bool(np.array_equal(ours, theirs)),
        "max_abs_error": float(error.max()),
        "psnr_db": None if mse == 0 else float(10 * np.log10(1 / mse)),
    }
    (out / "compare.json").write_text(f"{json.dumps(report, indent=1)}\n", encoding="utf-8")
    side = Image.new("RGB", (2 * width, height))
    side.paste(_image(ours), (0, 0))
    side.paste(_image(theirs), (width, 0))
    side.save(out / "ours_left_sglang_right.png")
    typer.echo(json.dumps(report))


if __name__ == "__main__":
    app()
