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

- [x] diffu-page's layout and labels kept, the rest of the old diffu-page removed
      with their tests; `layout/__init__.py` exports only public names. Checked: 331 crop plans (4,488 lines) from 60
      random GT pages are byte-identical before and after.
- [x] Removed the line-pasting era: the rest of diffu-page, `diffu-runner`, `diffu-studio`, the old model `src/diffu`
      with `tests/`, `Makefile`, `slurm/`, diffu-era `scripts/` and `docs/`. All under the tag
      `diffu-mmdit-v1`.
- [x] Root `pyproject.toml` is a virtual workspace root; diffusers pinned to `e04f0c5` (the NeMo environment's FLUX.3
      commit); ruff bans `dataclasses.dataclass` (NeMo's dataloader configs are the one exception).
- [x] `jobs/setup_nemo_env.sh` installs diffu-page, diffu-gen and diffu-train into the NeMo environment.
- [x] README says what the repo is and how it is laid out.
- [x] Grouped by pipeline stage (branch `modular-structure`): `layout/` (+ `page_xml.py`, the one PAGE/ALTO
      XML reader, shared by the layout replay and the crop builder), `crops/`, `captions/`, `modular/`,
      `generator.py`, `labels.py`, `cli/` (the `diffu-generate` / `-evaluate` / `-sheets` commands),
      `data/` (incl. `nemo.py`). `jobs/spark_guard.sh` (was `scripts/`); `.gitignore` trimmed; `.rayignore` dropped.
      Checked: the 331 crop plans byte-identical again; all `_target_`s of the three NeMo configs resolve in NeMo's
      environment. Merge after tonight's queue (its scripts call main's paths and `scripts/spark_guard.sh`).
- [x] Tests for plan (reading order: body block before its margin notes) and restyle (restyled caption == the crop
      captioned in the target style; the v3 edit caption takes only the paper from the reference). 50 tests.
- [ ] Delete `experiments/` once the running v2 job's evals have used it (~16:45 2026-10-21). Kept only for that
      run, which started before the restructure.
- [ ] Run `jobs/setup_nemo_env.sh` once to drop the stale page-synth / diffu-gen / old diffu installs from the NeMo
      environment and install diffu, diffu-page and diffu-train.
- [x] `docs/PIPELINE.md`: the flow from page to labelled crops, training, and the training/generation consistency table.

## Workspace (2026-10-21, branch `modular-structure`)

- [x] Members by role: diffu-page (page data, our blocks, the page and crop pipelines, the CLIs), diffu (the model),
      diffu-train (NeMo glue), diffu-runner (Ray), diffu-studio (Gradio). Pilot LoRA config pruned.
- [x] Every step is a Modular Diffusers block (`diffu_page.blocks`): `LoadLayoutStep`, `RefillStep`, `PlanCropsStep`;
      `CropPromptStep`, `ReferenceGridStep` before a model's own blocks; `LineReadbackStep`, `LabelExportStep` after;
      `CropLoopStep` draws every crop of a page. The model is chosen by its directory (`pipeline.FAMILIES`, keyed by
      the blocks its `modular_model_index.json` names), so FLUX.3 Klein and diffu run through the same blocks. The
      generator class (`CropGenerator`) and diffu-gen are gone: the CLIs, the runner and the studio call pipelines.
- [x] diffu-runner on the crop pipeline: pages planned with `PagePlanBlocks` on CPU workers, one crop per block, GPU
      actors each holding a crop pipeline; Ray 2.59.0. Fixed: a page's crops shared a block, and an actor hands a
      block on only when it has drawn all of it, so Ray Data stopped the CPU smoke run after 30 minutes without
      progress (the same would hit a ~30-crop block on the cluster).
- [x] diffu-studio as a `gr.Workflow` canvas: a plan node (`PagePlanBlocks`) feeding one draw node per drawer (a model
      and its LoRAs; drawers on one model share it and switch its LoRAs), each returning a sheet of the crops and a zip
      of their labels; app view and REST endpoint come with it. The nodes sit where the data is plain values (plans as
      JSON); embeddings and latents stay inside each crop pipeline.
- [ ] Studio: a reference-image node (v3, style by example) into the draw nodes; a glyph-map node for diffu.
- [x] diffu: FLUX.3's architecture with our own weights, laid out like a diffusers model (see "diffu" below), registered
      as a model family: the crop and page pipelines draw with a diffu repo exactly as with FLUX.3 Klein (test).
- [ ] Page assembly: crops overlap (training windows share lines) and windows that would cut a line are dropped, so
      pasting a page's crops back would draw lines twice and leave holes. Needs a tiling plan for generation and crops
      drawn consistently with their neighbours (e.g. the neighbour as a reference image); see "Full pages".

## Code audit against writing-python (2026-10-21, branch `modular-structure`)

- [x] One PAGE reader for training and generation (the replay took a line's last Word's outline, estimated baselines
      ~0.3 pitch off and ignored Transkribus reading order); 60/60 GT pages now planned exactly as the builder cut
      them. One region vocabulary (PAGE's types).
- [x] Pruned the synthetic template engine (templates, sampler, flow, edits), PageSpec, compositor-only fields,
      words / printed / JSON export, and the tests that only covered them.
- [x] Typer + Rich CLIs with exit codes; typed settings; a failed crop or page no longer ends a run.
- [x] DRY: one collection key, one latent-grid rule, slot constants from geometry; pick(…, 1) crash; images closed;
      pathlib links; typed manifest metadata; docstrings; @override on framework hooks; no private cross-imports,
      relative imports, long functions or >3 positional args left.
- [x] pyrefly checks imports again: `missing-import` was off, so an import of a module that no longer exists passed
      the type check. Only NeMo and its torchdata (NeMo's environment only) are typed as Any (red-checked with a stale
      `diffu_gen` import).
- [x] ALTO `softwareVersion` asked for the version of `page-synth`, a package that no longer exists, and always
      wrote "0".
- [x] pytest `strict = true`; pytest-cov and `[tool.coverage.run]` for per-test coverage when pruning the suite.
- [ ] Rebuild `data_flux3/layout_crops_v2` (and steer_v2, style_pairs_v3, val_eval) with the fixed archive names
      before any v2 retrain or new derived set: 521 records carry the raw folder name, and restyle now refuses them
      (`StaleCaptionError`).
- [ ] After tonight's queue: delete the one-off job chains in `jobs/flux3/` that have run (steer, pages smoke, v3).

## Modular Diffusers (checked against huggingface.co/blog/modular-diffusers and the diffusers source, 2026-10-21)

Tested end to end on CPU with diffusers' tiny FLUX.3 test model (`packages/diffu-page/tests/test_pipeline.py`, each
test red-checked); not yet run on the GPU with FLUX.3 Klein.

- [x] Our blocks around the model's own, declared like diffusers declares FLUX.3's: `pipeline.crop_blocks` (crop
      prompt, reference grid, the model's text encoder / VAE encoder / denoise / decode, read-back, labels; workflows
      `text2image` / `image_conditioned`), `pipeline.page_blocks` (the plan steps, then the crop blocks once per crop),
      `pipeline.PagePlanBlocks` (no model). The page pipeline draws every crop exactly as the crop pipeline draws it
      (test).
- [x] One crop pipeline serves v2 and v3: the model's auto blocks choose reference-conditioned denoising when an
      `image` is passed.
- [x] TrOCR loads through `load_components` from the read-back block's `ComponentSpec`;
      `readback.materialize_position_tables` fixes its meta table. `load_pipeline` fails fast when a component
      did not load (`load_components` only logs).
- [x] LoRAs load with the model's own LoRA loader (`lora.load_nemo_lora`, alpha/r folded into lora_B), named
      adapters, `set_adapters`. NeMo is no longer needed at inference. All 21 module kinds of the real v2 checkpoint
      take the adapter.
- [x] Training and inference see the same conditioning (checked 2026-10-21 on the real FLUX.3 text encoder, CPU): NeMo's
      FLUX.3 cache encoder and FLUX.3's text encoder block give bit-identical prompt embeddings on the same device (chat
      template, 80-token padding, position ids, layers 4…32); v2's GPU-made cache vs ours on CPU: mean per-token
      cosine 0.997, bf16 arithmetic differing by device. Latents: same VAE `mode()`, patchify, batch-norm normalization.
      RoPE ids and the reference stream: identical in NeMo's flux3 adapter and diffusers' blocks (image
      (0, h·0.5, w·0.5, 0), reference l = -1, text (0, 0, 0, i), references at timestep 0 inside the transformer).
- [x] Fixed: v3's references (10,299 of 11,152 not 16-px aligned) were squashed onto the grid by NeMo's cache encoder
      in training but would have been reflect-padded by FLUX.3's preprocessing at inference. `ReferenceGridStep` now
      resizes a reference exactly as training did, before FLUX.3's VAE encoder (test red-checked).
- [x] CFG null: NeMo's dropout zeroes the text embeddings, FLUX.3 (BFL, diffusers) uses the empty prompt's. Our LoRA
      configs set `cfg_dropout_prob: 0.0`, so v2/v3 are consistent; diffu must pick one null for training and inference.
- [ ] Publish the blocks to the Hub (`diffusers-cli custom_blocks`) if others should use them; they import diffu-page.
- [ ] Guidance: we use 4.0, BFL's own code's default (the model card says 3.5). Sampling shift: diffusers 6.93, BFL's
      code 16.0 since 2026-10-17. Compare both on the eval set.
- [x] The text-encoder layer choice is `DrawSettings.text_layers` (the model's own when unset: FLUX.3 every 4th of
      Qwen3-VL's; diffu 4, 12, 20, 28 of Qwen3.5's, from its repo).

## diffu: FLUX.3 architecture, our weights

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
- Measured 2026-10-21 (Qwen3.5-4B tokenizer + chat template, all 11,329 v2 captions): median 1,077 tokens (p95 1,483,
  max 3,810); the prose paragraph is only ~162 of them, the JSON layout rows (each repeating the style words around
  the line's text) the rest. A compact prompt for diffu (paragraph + `[id, box, text]` per line) is median 554 tokens:
  at 4 stacked layers 12.4 MB per crop, 1.2 TB per 100k crops; at 2 layers 0.6 TB. The line texts are most of what
  remains; rendering them as glyphs into a reference image (FLUX.3's reference stream) would leave only the style
  paragraph as text and spell exactly by construction.
- Order: measure Qwen3.5 token counts and latent statistics on ~1k crops → init-directory script → caching script →
  YAML (finetune mode, flux3 adapter, logit-normal shift ≈ 3, fp32 weights / bf16 compute, `dp_replicate_size: 8`,
  our interleaved dataloader, consolidated diffusers export) → GB10 smoke run → diffu inference blocks → EMA /
  Self-Flow recipe subclass → Leonardo.

- [x] **Model** (`packages/diffu`, laid out like a diffusers model: `configs.py`, `init.py`, `latents.py`,
      `modular_pipelines/diffu/` with encoders, decoders, `DiffuAutoBlocks`, `DiffuModularPipeline`):
      `Flux3Transformer2DModel` from our presets with random adaLN-Zero init; FLUX.3's denoise blocks reused unchanged.
      diffu-S 618.0M (hidden 1536, 2 per-mode + 12 joint: FLUX.3's per-mode share of depth), diffu-M 1636.7M (2048,
      4 + 16); Klein is 5.54B. `diffu-init` writes a model repo that `from_pretrained` and NeMo's finetune mode load.
      Start with S.
- [x] **Text encoder**: Qwen3.5-4B (Apache-2.0), chat template with thinking off, 80-token windows, hidden layers 4,
      12, 20 and 28 stacked (10,240 wide: the early orthographic, middle compositional and late layers; never 32, the
      post-norm state). Its BPE hides 3 of 4 å/ä/ö inside merged subwords and no layer spells an unseen subword
      linearly, so exact spelling may need the glyph map (below).
  - [ ] Standardize each stacked layer per channel with statistics from ~1k prompts (a GPU pass), stored in the repo
        and applied by both the text encoder step and the cache writer (layer RMS grows 16× from layer 4 to 32).
- [x] **VAE**: Qwen-Image's 16-channel VAE (Apache-2.0) through our encode / decode steps (`diffu.latents`: its
      per-channel mean/std, 2×2 patchify in FLUX.3's channel order, 64 channels). Reconstruction gate on 23 of the 24
      held-out crops (CPU, 2026-10-21): read-back CER 0.107 real vs 0.106 reconstructed (gate |Δ| ≤ 0.01; per crop
      −0.014 to +0.011).
  - [ ] Measure the latent mean/std on our pages; if they differ from Qwen-Image's config, write them into the repo's
        `vae/config.json` (`diffu.latents` reads only that).
- [ ] **References**: FLUX.3's reference stream replaces the old pooled DINO style vector (which gave generic style).
      A glyph map (each line's text rendered into its box, along the PAGE baseline) at l = −1 spells by construction
      and fixes line length; a style reference (another region of the same volume) at l = −2. Both already pass
      through `image=[glyph_map, style_ref]`, but FLUX.3's preprocessing center-crops each of several references to
      the 16-px grid: render the glyph map at exactly the target's grid size. Renderer: Unifont covers all 158
      characters of our transcriptions.
- [ ] **Training**: NeMo finetune mode from `diffu-init`'s repo, no `peft:` (identical weights on every rank). Cache
      writer for NeMo's layout using `diffu.modular_pipelines.diffu.encoders.prompt_embeds` and
      `diffu.latents.encode_latents` (the inference steps, not NeMo's FLUX.3 encoder); YAML (logit-normal, shift ≈ 3
      at 512² and ≈ 6 at 1 MP, fp32 weights / bf16 compute, `cfg_dropout_prob: 0` with the empty prompt swapped in by
      our dataset, zero latents for a dropped reference); then an EMA / Self-Flow recipe subclass (per-token
      timesteps through three forward pre-hooks; bit-identical to the stock forward with uniform timesteps).
- [ ] **Staged experiments** (design study 2026-10-21): E1 GB10 smoke + overfit (~3–4 h) → E2 conditioning ablation
      at 512² (text only / glyph map + style prompt / both) → E3 Self-Flow vs plain flow matching → E4 style reference
      (swap-test copy rate ≤ 5 %) → E5 main run (~12 h on 32 A100s for diffu-S). Each judged on held-out read-back
      CER (≥ 200 crops, bootstrap CIs) against the real crops and the FLUX.3 LoRA on the same crops.
- [ ] **Sampling shift**: FLUX.3's 6.93 for now; sweep 3–7 (`Flux3SetTimestepsStep(shift=…)`), it costs nothing.
- [ ] **Leonardo**: compute nodes have no internet; pre-fetch Qwen3.5-4B and the Qwen-Image VAE at the pinned
      revisions (a local repo copy with the files loads from the copy).
- [ ] **Data**: GT-Lejonet, 103,285 pages (~3M lines). Never FLUX.3 outputs.
- [ ] **Compute**: estimated ~8×10¹⁹ FLOPs for diffu-S (10M samples at 256², 5M at 512², 2M at 1024²; GB10
      measured at ~25 TFLOPS effective on the v2 LoRA). GB10 alone ~40 days: too slow, use it for smoke runs and
      short ablations. Leonardo, 8 nodes (32 A100s): roughly 6–30 h. First a short GB10 run at 256² to see loss
      and read-back CER fall, then book compute.
- [ ] **Eval**: the same held-out crops and metrics as the FLUX.3 LoRAs (per-line CER, layout, style score), so
      diffu and the reference are compared head to head.

## Labels (ALTO / PAGE are only output formats; the label data is what must be right)

- [x] Caption archive names: two GT collection folders end in U+F028 (a trailing space as macOS SMB encodes it), so
      `collection_title` fell back to the raw folder name: 521 of v2's 11,329 training captions (RA arjeplogs
      tingslag, RA carl fredrik pahlmans resejournaler) name "RA … 20260220\uf028". Fixed on `modular-structure`
      (+ test). v3's captions name no archive (unaffected); a v2 retrain would learn the right names.

- [x] Reading order across columns: labels follow the source blocks in page order, top to bottom in each
      (`CropPlan.reading_order`).
- [x] Line polygons: checked against the GT, not a defect (GT bands are 1.6x the line pitch; our slots 1.0x;
      slot IoU 0.72 vs traced ink 0.37). To match the GT convention, widen the slot.
- [ ] Narrow columns get no crops (under the 640 px minimum when a margin note narrows a column). Decide: crop
      narrow columns side by side, or leave such pages out.

## FLUX.3 reference runs

- [ ] Tonight's results: v2 eval, steering test (style by name), first generated crops (`runs/flux3_pages/smoke`).
- [ ] Decide on v3 (style by example) from the steering results; its results land tomorrow morning.
- [ ] Read tonight's profile (`runs/flux3_pages/smoke/profile/trace.json`; pytorch-skills:profile-model): SDPA
      backend on sm_121, GPU idle vs busy, top kernels, text-encoder vs denoise vs VAE vs TrOCR share.
- [ ] Full pages: generate at 2–4 MP, or stitch windows that share a reference.
- [ ] Style score: a classifier on real crops that says whether a crop matches the requested archive or reference.

## Inference at scale

- [x] Ray Data entrypoint (diffu-runner): layout / refill / plan as a CPU `flat_map` (`PagePlanBlocks`), one crop
      per block, the crop pipeline as a stateful GPU actor (`map_batches`, concurrency = actors), TrOCR read-back
      co-located, Parquet sink.
- [ ] After profiling on the cluster: text encoding as its own stage (FLUX.3's text encoder takes one prompt per
      call), a Lance sink if Parquet's per-file writes get in the way.
- [ ] Runs on our Ray cluster's Blackwell GPUs (96 GB each, sm_120). Re-profile there before optimizing; measure
      FP8 / NVFP4, fewer steps, a compiled transformer, several actors per card.
