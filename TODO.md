# TODO

What is known to be wrong or unfinished. Remove an item when it is done; add the commit that did it.

## Direction (decided 2026-10-21)

- FLUX.3 Klein (with our LoRAs) is the **research reference**: what quality is reachable, and the eval target.
- **diffu is the deliverable**: our own weights, trained only on our data, using FLUX.3's architecture and
  training recipe. The FLUX Kommunity License (§1a(iii), §4b) forbids distilling from FLUX.3, training on its
  outputs, or reusing its weights or autoencoder for another image model. Allowed: diffusers'
  `Flux3Transformer2DModel` (Apache-2.0 code) instantiated from a config with random init; Apache-2.0 text
  encoders and VAEs.
- Everything is diffusers-native (model classes, Modular Diffusers blocks and pipelines) and trained with NeMo
  AutoModel. No custom model code where diffusers has the class; no backward compatibility with the old code.

## Repo structure

Done 2026-10-21.

- [x] `diffu_page.layout` and `diffu_page.labels` moved into page-synth (`page_synth.layout`, `page_synth.labels`)
      with their tests; `layout/__init__.py` exports only public names. Checked: 331 crop plans (4,488 lines) from 60
      random GT pages are byte-identical before and after.
- [x] Removed the line-pasting era: the rest of diffu-page, `diffu-runner`, `diffu-studio`, the old model `src/diffu`
      with `tests/`, `Makefile`, `slurm/`, diffu-era `scripts/` and `docs/`. All under the tag
      `diffu-mmdit-v1`.
- [x] Root `pyproject.toml` is a virtual workspace root; diffusers pinned to `e04f0c5` (the NeMo environment's FLUX.3
      commit); ruff bans `dataclasses.dataclass` (NeMo's dataloader configs are the one exception).
- [x] `jobs/setup_nemo_env.sh` installs page-synth into the NeMo environment (and removes the old diffu, diffu-page).
- [x] README says what the repo is and how it is laid out.
- [ ] Delete `experiments/` once the running v2 job's evals have used it (~16:45 2026-10-21). Kept only for that
      run, which started before the restructure.
- [ ] Run `jobs/setup_nemo_env.sh` once to drop the stale diffu / diffu-page installs from the NeMo environment
      (page-synth's editable install already points at this checkout).
- [ ] Tests for plan and restyle (the generator, layout, labels, refill, sampler and XML replay are covered).
- [ ] `docs/PIPELINE.md` from the pipeline page, once the generator has run on the GPU.

## Modular Diffusers (checked against huggingface.co/blog/modular-diffusers and the diffusers source, 2026-10-21)

Tested end to end on CPU with diffusers' tiny FLUX.3 test model (`test_generator.py`, each test
red-checked); not yet run on the GPU with FLUX.3 Klein.

- [x] Text encoder as its own pipeline: `pipelines.text_blocks()` (crop prompt + FLUX.3's text encoder) and
      `pipelines.crop_blocks()` (everything else), `CropGenerator.encode` / `draw` as the two Ray stages.
- [x] One `ComponentsManager` for both pipelines. One crop pipeline serves v2 and v3: FLUX.3's auto blocks choose
      reference-conditioned denoising when an `image` is passed.
- [x] TrOCR loads through `load_components` from the read-back block's `ComponentSpec`;
      `readback.materialize_position_tables` fixes its meta table. `load_pipelines` fails fast when a component
      did not load (`load_components` only logs).
- [x] LoRAs load with diffusers' FLUX.3 LoRA loader (`pipelines.load_nemo_lora`, alpha/r folded into lora_B), named
      adapters, `set_adapters`. NeMo is no longer needed at inference. All 21 module kinds of the real v2 checkpoint
      take the adapter.
- [x] The base model is a constructor argument (`CropGenerator(model, ...)`), so diffu plugs in the same way.
- [ ] Publish the blocks to the Hub (`diffusers-cli custom_blocks`) if others should use them; they import page-synth.
- [x] Training and inference see the same conditioning (checked 2026-10-21 on the real FLUX.3 text encoder, CPU): NeMo's
      FLUX.3 cache encoder and our text pipeline give bit-identical prompt embeddings on the same device (chat template,
      80-token padding, position ids, layers 4…32). Latents: same VAE `mode()`, patchify, batch-norm normalization.
      RoPE ids and the reference stream: identical in NeMo's flux3 adapter and diffusers' blocks (image
      (0, h·0.5, w·0.5, 0), reference l = -1, text (0, 0, 0, i), references at timestep 0 inside the transformer).
- [x] Fixed: v3's references (10,299 of 11,152 not 16-px aligned) were squashed onto the grid by NeMo's cache encoder
      in training but would have been reflect-padded by FLUX.3's preprocessing at inference. `ReferenceGridStep` now
      resizes a reference exactly as training did, before FLUX.3's VAE encoder (test red-checked).
- [x] CFG null: NeMo's dropout zeroes the text embeddings, FLUX.3 (BFL, diffusers) uses the empty prompt's. Our LoRA
      configs set `cfg_dropout_prob: 0.0`, so v2/v3 are consistent; diffu must pick one null for training and inference.
- [ ] Guidance: we use 4.0, BFL's own code's default (the model card says 3.5). Sampling shift: diffusers 6.93, BFL's
      code 16.0 since 2026-10-17. Compare both on the eval set.
- [ ] The text-encoder layer choice is `GeneratorSettings.text_layers` (FLUX.3's default when unset); diffu's encoder
      will need its own.

## diffu: FLUX.3 architecture, our weights (plan only, nothing implemented)

Research 2026-10-21 (reports: FLUX.3 recipe, NeMo feasibility; both read-only).

- BFL published no technical report for FLUX.3 / Klein: the architecture is fully in the code, most of the training
  recipe is not public. What is: "FLUX 3 builds on Self-Flow" (bfl.ai/blog/flux-3; arXiv 2603.06507): two noise
  levels per sample, an EMA teacher (0.9999) on the cleaner input, a cosine feature loss (student layer 0.3·depth to
  teacher layer 0.7·depth, weight 0.8); it reports clearly better text rendering than plain flow matching. Candidate
  for diffu (needs an EMA teacher: a NeMo recipe subclass).
- Architecture details that matter: the 5 early blocks are per mode (image, reference, text each attend only within
  their mode), then 28 shared joint blocks (62% of the weights); one modulation per mode per stage; no biases; no
  pooled text vector; CFG null = the empty caption. Captions: Klein trained on its upsampler's output (200–350-word
  English prose, quoted text verbatim, JSON layout boxes [y0, x0, y1, x1] on 0–1000).
- Timestep shift: FLUX.2-latent experiments found logit-normal with shift 4.63 best for training at 256², scaled by
  √(token ratio) for other resolutions; NeMo has only a constant `flow_shift` (no resolution-dependent shift).
- The autoencoder: FLUX.3's `ae.safetensors` is byte-identical (sha256) to the autoencoder BFL also publishes under
  Apache-2.0 (32 channels, semantically regularized like REPA-E). Legal question: if that license holds, diffu can
  use FLUX.3's own autoencoder instead of Qwen-Image's 16-channel VAE.
- NeMo trains diffu without a NeMo patch: the flux3 adapter and `Flux3Transformer2DModel` take channel count and
  text width from the config. Use an init directory (modular index + our initialized weights) with `mode: finetune`
  and no `peft:` block, so every rank loads identical weights (pretrain mode inits per rank and HSDP never resyncs).
  Our caching code writes NeMo's cache layout (Qwen-Image VAE: `vae.encode(x.unsqueeze(2))`, per-channel
  mean/std, patchify to 64 channels; Qwen3.5-4B: with 8 stacked layers NeMo's rule picks exactly its 8
  full-attention layers). No EMA in the diffusion recipe; validation is held-out loss only; consolidated export with
  `diffusers_compatible: true` should load with `from_pretrained` (verify in the smoke run).
- Biggest risk: the text cache. v2's FLUX.3 cache is 524 GB for 11,329 crops (layout prompts ~1,150 tokens). For diffu
  at 8 / 4 / 2 / 1 stacked layers: ~47 / 24 / 12 / 6 MB per crop, 4.7 / 2.4 / 1.2 / 0.6 TB per 100k crops (Leonardo:
  1 TB `$WORK`, 20 TB `$SCRATCH` with a 40-day purge). Options: ≤ 2 layers, shorter prompts, or encoding text during
  training (~+45% compute, 9 GB per GPU).
- Order: measure Qwen3.5 token counts and latent statistics on ~1k crops → init-directory script → caching script →
  YAML (finetune mode, flux3 adapter, logit-normal shift ≈ 3, fp32 weights / bf16 compute, `dp_replicate_size: 8`,
  our interleaved dataloader, consolidated diffusers export) → GB10 smoke run → diffu inference blocks → EMA /
  Self-Flow recipe subclass → Leonardo.

- [ ] **Model**: `Flux3Transformer2DModel.from_config(...)`, random init. Sizes measured on the meta device:
      diffu-S 0.66B (hidden 1536, 3 double + 10 single blocks), diffu-M 1.66B (2048, 4 + 16); Klein is 5.54B
      (3072, 5 + 28). Start with S.
- [ ] **Text encoder**: Qwen3.5-4B (Apache-2.0, Feb 2026, hidden 2560 like FLUX.3's Qwen3-VL-4B, so FLUX.3's 8
      stacked layers = 20,480 carry over); Qwen3.5-2B as the cheaper alternative. Open: which layers to stack
      (Qwen3.5 mixes linear- and full-attention layers). Exact spelling (å ä ö, historical forms) through a BPE
      encoder is a risk; fallback: rendered glyphs as a reference image (old diffu's glyph strip, in 2D).
- [ ] **VAE**: Qwen-Image's 16-channel VAE (Apache-2.0; passed our å ä ö reconstruction gate). Needs our own
      encode / decode blocks: the Flux3 VAE blocks assume FLUX's 32-channel autoencoder and its batch-norm stats.
      `in_channels` = 16 × 2 × 2 = 64.
- [ ] **References**: up to N reference images in the sequence with the 4-axis RoPE (FLUX.3's in-context
      conditioning) replace the old pooled DINO style vector, which gave generic style.
- [ ] **Training**: NeMo AutoModel's flux3 recipe in pretrain mode. Check it accepts a random-init transformer
      and a different VAE / text encoder; otherwise add a diffu processor. Progressive resolution 256² → 512² →
      1024², aspect buckets, our captions. Recipe details (timestep shift, sampling, schedule, data mix) from
      BFL's FLUX.3 technical report: find and read it.
- [ ] **Data**: GT-Lejonet, 103,285 pages (~3M lines). Never FLUX.3 outputs.
- [ ] **Compute**: estimated ~8×10¹⁹ FLOPs for diffu-S (10M samples at 256², 5M at 512², 2M at 1024²; GB10
      measured at ~25 TFLOPS effective on the v2 LoRA). GB10 alone ~40 days: too slow, use it for smoke runs and
      short ablations. Leonardo, 8 nodes (32 A100s): roughly 6–30 h. First a short GB10 run at 256² to see loss
      and read-back CER fall, then book compute.
- [ ] **Eval**: the same held-out crops and metrics as the FLUX.3 LoRAs (per-line CER, layout, style score), so
      diffu and the reference are compared head to head.

## Labels (ALTO / PAGE are only output formats; the label data is what must be right)

- [x] Reading order across columns: labels follow the source blocks in page order, top to bottom in each
      (`CropPlan.reading_order`).
- [x] Line polygons: checked against the GT, not a defect (GT bands are 1.6x the line pitch; our slots 1.0x;
      slot IoU 0.72 vs traced ink 0.37). To match the GT convention, widen the slot.
- [ ] Narrow columns get no crops (590 px < the 640 px minimum when a margin note narrows a multi_column page).
      Decide: crop narrow columns side by side, or keep such templates out.

## Layout (from diffu-page): verify before relying on it

- [ ] Tune the template sampler to the GT before using synthetic templates (2026-10-21, 300 vs 300 pages:
      running-text lines per page 30 real vs 9 sampled; line pitch 0.029 vs 0.090 of page height; line width 0.82
      vs 1.00 of the block). Until tuned: real templates only.
- [ ] The `landskap` template gives no crops: the windowing rejects its short label lines.

## FLUX.3 reference runs

- [ ] Tonight's results: v2 eval, steering test (style by name), first generated crops (`runs/flux3_pages/smoke`).
- [ ] Decide on v3 (style by example) from the steering results; its results land tomorrow morning.
- [ ] Read tonight's profile (`runs/flux3_pages/smoke/profile/trace.json`; pytorch-skills:profile-model): SDPA
      backend on sm_121, GPU idle vs busy, top kernels, text-encoder vs denoise vs VAE vs TrOCR share.
- [ ] Full pages: generate at 2–4 MP, or stitch windows that share a reference.
- [ ] Style score: a classifier on real crops that says whether a crop matches the requested archive or reference.

## Inference at scale

- [ ] Ray Data entrypoint in page-synth: layout / refill / plan as CPU map steps, text encoding as its own
      stage, the generator as a stateful GPU actor (`map_batches`, concurrency = GPUs), TrOCR read-back
      co-located, Lance sink.
- [ ] Runs on our Ray cluster's Blackwell GPUs (96 GB each, sm_120). Re-profile there before optimizing; measure
      FP8 / NVFP4, fewer steps, a compiled transformer, several actors per card.
