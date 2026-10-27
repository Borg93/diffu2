# diffu-moss

MOSS-Image (OpenMOSS, Apache-2.0) as the page pipeline's third model family, beside FLUX.3 Klein and diffu: our own
implementation, written from the published design, around the released weights.

- **The DiT** (`transformer.py`): `MossTransformer`, 42 single-stream blocks over the image's and the prompt's tokens;
  the released Base transformer (or a merged LoRA or full fine-tune exported in diffusers format) loads into it by name.
- **The blocks** (`modular_pipelines/moss/`): `MossAutoBlocks` and `MossModularPipeline`. A crop plan's layout caption
  compiled as MOSS's Layout prompt (`layout_prompt.py`: a summary and one text item per line, its exact text in its
  pixel box), encoded by the frozen Qwen3.5-9B text encoder, denoised with classifier-free guidance whose
  unconditional pass skips a block, decoded by the FLUX.2 VAE (`latents.py`). The denoise loop has FLUX.3's step
  names, so the page pipeline's line repair runs in it.
- **A model repo** (`diffu-moss-init`): a `modular_model_index.json` naming `MossAutoBlocks`, the components loading
  from the Base snapshot's folders, the DiT from the snapshot or a trained export.

      diffu-moss-init --base <MOSS-Image-Base snapshot> --out runs/moss/base
      diffu-moss-init --base <MOSS-Image-Base snapshot> --transformer <merged export> --out runs/moss/hand_1715
