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
- `export_hf.py`: `diffu-moss-export-hf`, a model directory or checkpoint written in the official Hub layout
  (`model_index.json`, `MOSSImageTransformer`, `rope.bands`), the folder SGLang Diffusion loads; SGLang's sample-page
  input is `packages/sglang_moss`.
- `train/`: the reference training: `pairs.py` (style and page pairs, the prompt forms), `cache.py`
  (`diffu-moss-cache-pairs`), `restyle.py` (look changes for the sample form), `loop.py` (`diffu-moss-train-ref`: a
  LoRA, or `--full` with `full.py` and `muon.py`), `sigmas.py`, `distributed.py` (FSDP2 over several GPUs),
  `export.py` (`diffu-moss-export-ref`, a LoRA merged into Base) and `records.py` (what caches, checkpoints and exports
  record).
- `mxfp8.py`, `inference.py`, `step_cache.py`: faster drawing, options of the page pipeline's `DrawSettings`, each
  off by default: `precision` / `text_encoder_precision` `"mxfp8"` (on Blackwell GPUs, the DiT's joint-block
  projections and the text encoder's in MXFP8, OCP MX v1.0 on torch's own `scaled_mm`), `compile_blocks` (the joint
  blocks compiled), `sampler` `"unipc"` (diffusers' multistep UniPC solver stepping MOSS's snapped grid, for fewer
  `steps`) and `cache` (diffusers' step caches, FirstBlockCache, TaylorSeer or MagCache, which skip the joint blocks at
  some steps; `magcache_calibration` measures MagCache's ratios).
- `aoti.py`: `diffu-moss-aoti`, the joint block packaged ahead of time with AOTInductor for one GPU kind and torch
  version (`DrawSettings.aoti_package`), so a worker starts without compiling; `bench_inference.py`: the options'
  speed, memory and agreement with bfloat16 on a GPU. How to use them: [docs/INFERENCE.md](../../docs/INFERENCE.md).
