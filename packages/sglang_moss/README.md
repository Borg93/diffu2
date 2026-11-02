# diffu-sglang-moss

SGLang Diffusion's MOSS-Image (sglang 0.5.23, sgl-project/sglang#44325) drawing from a reference image, the sample
page, as our own pipeline (`diffu_moss`) does. An out-of-tree plugin: SGLang imports it in every process it starts
when `SGLANG_EXTERNAL_MODEL_PACKAGE=diffu_sglang_moss` is set, and nothing of SGLang is patched or copied.

- `dit.py`: `DiffuMOSSImageTransformer`, SGLang's `MOSSImageTransformer` whose forward also takes
  `reference_latents`: each reference's tokens after the prompt's, on the rotary reference axis at `k + 1`, modulated
  at timestep 0. Without references it is SGLang's forward.
- `reference.py`: the stage that VAE-encodes the request's reference (`image_path`) as `diffu_moss.latents` does,
  on its own 16-px grid, never resized; a DiT whose `transformer/export.json` does not say `takes_reference` is
  refused one.
- `pipeline.py`: the config (a second task, TI2I; the reference latents in both passes of guidance) and the pipeline,
  registered in the place of SGLang's `MOSSImagePipeline` by `__init__.py`.

It lives in its own environment, not the workspace's (SGLang pins diffusers 0.37 and transformers 5.19), so it is
not a workspace member. The model folder it loads is written by `diffu-moss-export-hf` (packages/moss). How to set
it up, serve, and run it inside Ray Data: [docs/INFERENCE.md](../../docs/INFERENCE.md), section 13.
