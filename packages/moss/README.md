# diffu-moss

MOSS-Image (OpenMOSS, Apache-2.0) as a model family of the page pipeline: our own implementation, written from the
published design, around the released weights, and our own reference training. How it draws and trains:
[docs/MOSS.md](../../docs/MOSS.md).

- `transformer.py`: `MossTransformer`, the DiT; the released Base transformer, a merged LoRA or a full fine-tune's
  checkpoint loads into it by name. It also reads reference images (sample pages) once trained to.
- `modular_pipelines/moss/`: `MossAutoBlocks` and `MossModularPipeline`: the Layout prompt (`layout_prompt.py`,
  compiled from a crop plan's caption), the Qwen3.5-9B text encoder, the reference encoder, the denoise loop (laid out
  as FLUX.3's, so the page pipeline's line repair runs in it) and the FLUX.2 VAE (`latents.py`).
- `sample_caption.py`: a reference-drawn caption in the forms MOSS draws from and trains on (the sample image named,
  dropped, or its look words dropped).
- `init.py`: `diffu-moss-init`, a model directory of the Base snapshot's components around its own or a trained DiT.
- `train/`: the reference training: `pairs.py` (style and page pairs, the prompt forms), `cache.py`
  (`diffu-moss-cache-pairs`), `restyle.py` (look changes for the sample form), `loop.py` (`diffu-moss-train-ref`: a
  LoRA, or `--full` with `full.py` and `muon.py`), `sigmas.py`, `distributed.py` (FSDP2 over several GPUs),
  `export.py` (`diffu-moss-export-ref`, a LoRA merged into Base) and `records.py` (what caches, checkpoints and exports
  record).
