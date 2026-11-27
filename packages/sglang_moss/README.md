# diffu-sglang-moss

SGLang Diffusion's MOSS-Image (sglang 0.5.23, sgl-project/sglang#44325) drawing from a reference image, the sample
page, as our own pipeline (`diffu_moss`) does. An out-of-tree plugin: SGLang imports it in every process it starts
when `SGLANG_EXTERNAL_MODEL_PACKAGE=diffu_sglang_moss` is set, and nothing of SGLang is patched or copied.

- `dit.py`: `DiffuMOSSImageTransformer`, SGLang's `MOSSImageTransformer` whose forward also takes
  `reference_latents`: each reference's tokens after the prompt's, on the rotary reference axis at `k + 1`, modulated
  at timestep 0. Without references it is SGLang's forward. Its joint block runs SGLang's fused RMSNorm + modulation
  and gated residual on the two parts of the tokens that share a modulation; a request's `fused_qkv` computes Q, K
  and V by one GEMM (equal within rounding); `_compile_conditions` lets `regional_compile` compile the joint blocks
  alone (SGLang's MOSS-Image config still refuses torch.compile: lift it as `runs/htr/moss_bench/moss_bench_ext` does).
- `precision.py`: the precision of the DiT's joint-block projections for the whole server,
  `DIFFU_SGLANG_DIT_PRECISION`: `bf16` (default, the DiT as loaded), `mxfp8` (`diffu_moss.mxfp8`'s OCP MXFP8,
  copied here; cuBLAS's block-scaled GEMM, Blackwell) or `fp8_rowwise` (E4M3 with a scale a row and a token,
  `torch._scaled_mm`). SGLang refuses its own quantization for MOSS-Image, so `DiffuMOSSImageTransformer` swaps the
  six projections of every joint block after loading (`post_load_weights`); embedders, text refiner, time embedding,
  modulations, norms and the head stay bf16. A quantized DiT runs every request through the plugin's joint blocks;
  `fused_qkv` is refused on one. The input is quantized by plain ops, fused by Inductor under compile and run as a
  small compiled function outside it on a GPU; on CPU MXFP8 is emulated as `diffu_moss.mxfp8` does (SGLang draws
  bit for bit what our pipeline draws with `quantize_linears(transformer, DIT_LINEARS)`).
- `denoising.py`: the denoising stage with our guidance stop: a request's `guidance_stop` (1 by default) runs the
  conditional pass alone after that fraction of the steps and takes its prediction as it is, as our pipeline's guider
  (`stop`) does; not SGLang's `cfg_gate_step`, which reuses the last guidance difference.
- `reference.py`: the stage that VAE-encodes the request's reference (`image_path`) as `diffu_moss.latents` does,
  on its own 16-px grid, never resized; a DiT whose `transformer/export.json` does not say `takes_reference` is
  refused one.
- `pipeline.py`: the config (a second task, TI2I; the reference latents in both passes of guidance) and the pipeline,
  registered in the place of SGLang's `MOSSImagePipeline` by `__init__.py`.

It lives in its own environment, not the workspace's (SGLang pins diffusers 0.37 and transformers 5.19), so it is
not a workspace member. The model folder it loads is written by `diffu-moss-export-hf` (packages/moss). How to set
it up, serve, and run it inside Ray Data: [docs/INFERENCE.md](../../docs/INFERENCE.md), section 13.
